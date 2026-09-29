"""Tool protocol, shared result/context types, and the Serper budget guard."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..image_store import ImageStore


@dataclass
class ToolImage:
    """An image a tool produced, to be appended as an <image> in the next turn."""
    image_id: str
    rel_path: str
    path: str


# Failure classes, keyed on whose fault it is, because that decides the handling
# (retry? what to tell the model? does it count against the model?):
#   model_error  bad tool name/args/image_id, degenerate box -- no retry, return
#                something correctable
#   infra_error  429/5xx/network/missing key -- nothing to do with the model, and
#                it must read differently from a genuine empty result, or a broken
#                backend teaches the model that the question has no answer
#   empty_result the search really matched nothing -- a legitimate world state
#   budget       budget or context exhausted -- a design constraint, not a fault
ERR_MODEL = "model_error"
ERR_INFRA = "infra_error"
ERR_EMPTY = "empty_result"
ERR_BUDGET = "budget"


@dataclass
class ToolResult:
    text: str                                   # body placed inside <tool_response>...</tool_response>
    images: List[ToolImage] = field(default_factory=list)
    raw: Any = None                             # structured payload for the trajectory log
    # Structured failure signal, so error statistics never depend on matching
    # substrings of the message text.
    ok: bool = True
    error_class: Optional[str] = None           # one of ERR_*; None when ok=True




@dataclass
class ToolContext:
    """Everything a tool call needs beyond its own arguments."""
    sample: Any                                 # eval.dataset.Sample
    image_store: ImageStore
    output_dir: str
    # Per-sample Serper budget (created fresh per sample by the runner). When set,
    # image tools prefer it over their shared/run-wide budget — this makes the cap
    # per-sample (fair across samples, deterministic) instead of a run-wide race.
    serper_budget: Optional["SerperBudget"] = None


class Tool(Protocol):
    name: str

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...


class SerperBudget:
    """Thread-safe counter capping total Serper calls across a run.

    Serper quota is scarce, so image tools share one budget. try_spend()
    atomically reserves one call; returns False when exhausted so the tool can
    degrade gracefully instead of burning quota.
    """

    def __init__(self, limit: int):
        self.limit = max(0, int(limit))
        self._used = 0
        self._lock = threading.Lock()

    def try_spend(self) -> bool:
        with self._lock:
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Return an unused reservation (e.g. the request failed before hitting Serper)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - self._used)


def format_tool_response(result: ToolResult) -> str:
    """Wrap a tool result body in the <tool_response> envelope the model expects."""
    return f"<tool_response>\n{result.text.strip()}\n</tool_response>"

# ---------------------------------------------------------------------------
# One retry policy shared by the three networked tools.
#   429      fixed-interval retry; this sleep blocks a worker thread, so backing
#            off would slow the whole batch
#   5xx/4xx  no retry -- a real server fault only wastes trajectory budget
#   network  one immediate retry
RATE_LIMIT_RETRIES = 10
RATE_LIMIT_INTERVAL = 0.5
NETWORK_RETRIES = 1


def http_post_json(url: str, headers: dict, payload: dict, timeout: int = 45):
    """POST under the policy above and classify failures. Never raises."""
    import time

    import requests

    net_tries = 0
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception:  # noqa: BLE001  connection refused / timeout
            if net_tries < NETWORK_RETRIES:
                net_tries += 1
                continue
            return None, ERR_INFRA, None
        if resp.status_code == 429:
            if attempt < RATE_LIMIT_RETRIES - 1:
                time.sleep(RATE_LIMIT_INTERVAL)
                continue
            return None, "rate_limited", 429          # caller renders the B1 message
        if resp.status_code != 200:
            return None, ERR_INFRA, resp.status_code  # 5xx/4xx: no retry
        try:
            return resp.json(), None, 200
        except Exception:  # noqa: BLE001  200 with a non-JSON body: server fault
            return None, ERR_INFRA, resp.status_code
    return None, "rate_limited", 429
