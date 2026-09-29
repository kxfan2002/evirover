"""Launch a local vLLM OpenAI-compatible server for a checkpoint dir.

Used by run_eval.py --model-path: starts `vllm serve <path>`, waits for /health,
and hands back (base_url, model_name). Registered for cleanup at process exit.

Requires vLLM >= 0.17 in the active env (earlier versions lack qwen3_5 support).
The server is a child process; stdout/stderr stream to a log file under the
results/serve_logs dir so a hang is diagnosable.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import requests


def _derive_max_len(model_path: str) -> Optional[int]:
    """Read max_position_embeddings from a model's config.json (top-level or a
    nested text/llm config). Returns None if not found. Used to cap the
    requested max_model_len so vLLM doesn't refuse to start (e.g. InternVL3.5-8B
    caps at 40960, far below our 262144 default)."""
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    candidates = [cfg]
    for key in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            candidates.append(sub)
    vals = [c["max_position_embeddings"] for c in candidates
            if isinstance(c.get("max_position_embeddings"), int)]
    return min(vals) if vals else None


def _is_qwen3_family(model_path: str) -> bool:
    """True if the model is a qwen3 / qwen3_vl model, i.e. the `--reasoning-parser
    qwen3` + `--tool-call-parser qwen3_coder` flags apply. For other families
    (e.g. InternVL) those parsers mis-route output — the model emits tokens that
    land in neither `content` nor `reasoning_content` — so they must be omitted."""
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    # Match ONLY the top-level architecture / model_type — NOT a full-blob
    # substring. InternVL3.5-8B uses a Qwen3 LLM backbone (so "qwen3" appears in
    # a nested text_config), but its chat/output format is InternVL's, driven by
    # architecture InternVLChatModel — the qwen3 reasoning parser does not apply.
    archs = " ".join(cfg.get("architectures") or []).lower()
    mt = str(cfg.get("model_type", "")).lower()
    return "qwen3" in mt or "qwen3" in archs


def pick_free_gpu(min_free_mib: int = 40000) -> Optional[str]:
    """Return the index of the GPU with the most free memory (>= threshold), or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        return None
    best_idx, best_free = None, -1
    for line in out.strip().splitlines():
        try:
            idx, free = [x.strip() for x in line.split(",")]
            free_i = int(free)
        except Exception:  # noqa: BLE001
            continue
        if free_i > best_free:
            best_idx, best_free = idx, free_i
    if best_idx is not None and best_free >= min_free_mib:
        return best_idx
    return None


class VLLMServer:
    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        gpu: str = "",
        served_model_name: str = "",
        # Capped well below max_position_embeddings: a single sequence at that length
        # needs more KV cache than the pool holds, and vLLM then refuses to start.
        # Still far above the heaviest real episode.
        max_model_len: int = int(os.getenv("EVAL_MAX_MODEL_LEN", "262144")),
        # Images accumulate across the whole episode, so raising --max-tool-calls
        # requires raising this too; otherwise rollouts fail with a bare HTTP 400.
        limit_images: int = int(os.getenv("EVAL_LIMIT_IMAGES", "64")),
        # SAM3 (used by verify_mask) loads into the main process on the same GPU,
        # out of whatever vLLM leaves free. At 0.9 the remainder is too small and
        # SAM3 dies with a segfault rather than a Python OOM -- no traceback.
        gpu_mem_util: float = float(os.getenv("EVAL_GPU_MEM_UTIL", "0.9")),
        max_num_seqs: int = 0,
        log_dir: str = "",
        startup_timeout: int = 900,
        # Whether to attach --reasoning-parser qwen3; see _cmd().
        reasoning_parser: bool = True,
    ):
        if not os.path.isdir(model_path):
            raise ValueError(f"model_path is not a directory: {model_path}")
        self.model_path = os.path.abspath(model_path)
        self.port = port
        self.gpu = gpu
        self.served_model_name = served_model_name or os.path.basename(self.model_path.rstrip("/"))
        # Cap to the model's own max_position_embeddings if it's shorter than
        # requested — otherwise vLLM refuses to start (max_model_len > derived).
        derived = _derive_max_len(self.model_path)
        if derived is not None and derived < max_model_len:
            print(f"[serve] capping max_model_len {max_model_len} -> {derived} "
                  f"(model's max_position_embeddings)")
            max_model_len = derived
        self.max_model_len = max_model_len
        self.is_qwen3 = _is_qwen3_family(self.model_path)
        self.reasoning_parser = reasoning_parser
        self.limit_images = limit_images
        self.gpu_mem_util = gpu_mem_util
        # max_num_seqs: cap on concurrent sequences vLLM will run. 0 = leave to
        # vLLM default (256). Set explicitly to raise concurrency headroom when
        # max_model_len is small (large KV pool freed for more parallel seqs).
        self.max_num_seqs = max_num_seqs
        self.log_dir = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "serve_logs")
        self.startup_timeout = startup_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._log_path = ""

    def _cmd(self) -> List[str]:
        #   --reasoning-parser qwen3 : strip <think> into reasoning_content
        #   --enable-prefix-caching  : reuse shared prefixes across agent turns
        #   --enable-auto-tool-choice / --tool-call-parser: unused by this eval's
        #     text-based tool protocol, kept so the server matches training.
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.served_model_name,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--trust-remote-code",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_mem_util),
            # vllm 0.24 expects a JSON value here (older versions took image=N).
            "--limit-mm-per-prompt", json.dumps({"image": self.limit_images}),
            "--enable-prefix-caching",
            "--mm-encoder-tp-mode", "data",
            # The api_server process preprocesses images CPU-side before EngineCore
            # sees them, which bottlenecks image-heavy rollouts. N frontends share
            # one engine. Env-gated, so the default is unchanged.
            *( ["--api-server-count", os.environ["VLLM_API_SERVER_COUNT"]]
               if int(os.environ.get("VLLM_API_SERVER_COUNT", "0") or 0) > 1 else [] ),
            *( ["--mm-processor-cache-gb", os.environ["VLLM_MM_CACHE_GB"]]
               if os.environ.get("VLLM_MM_CACHE_GB") else [] ),
            # Raise the processor's pixel-area cap, which a checkpoint may carry
            # much lower than the upstream weights. The key must be `max_pixels`:
            # vLLM applies these at construction, where `size` is silently ignored.
            *( ["--mm-processor-kwargs",
                json.dumps({"max_pixels": int(os.environ["VDR_EVAL_MM_MAX_PIXELS"])})]
               if os.environ.get("VDR_EVAL_MM_MAX_PIXELS") else [] ),
            # --mm-processor-cache-type shm is deliberately not used: its shared
            # memory buffers leak if vLLM is killed rather than exited cleanly.
        ]
        # max_num_seqs: only pass when explicitly set (>0). vllm 0.24 rejects 0
        # (must be >=1) and cascades it into max_num_batched_tokens=0; omitting
        # the flag lets vLLM pick its own default.
        if self.max_num_seqs and self.max_num_seqs > 0:
            cmd += ["--max-num-seqs", str(self.max_num_seqs)]
        # qwen3-only parsers. On non-qwen3 models (e.g. InternVL) the qwen3
        # reasoning parser mis-routes output into neither content nor
        # reasoning_content, so the agent loop sees empty replies and stalls.
        if self.is_qwen3:
            # Total loss on non-thinking variants: the parser splits on </think>, so
            # a model that never emits one has its whole output classified as
            # thinking and discarded. The model family cannot tell the two apart, so
            # the caller decides.
            if self.reasoning_parser:
                cmd += ["--reasoning-parser", "qwen3"]
            cmd += [
                "--enable-auto-tool-choice",
                "--tool-call-parser", "qwen3_coder",
            ]
        return cmd

    def _probe_nonempty(self) -> None:
        """Probe once before the run: a normal finish (finish_reason=stop) that
        delivers no text means the parsing layer swallowed the output. This is a
        silent total-loss failure -- HTTP 200, GPU busy, a score still computed, all
        zeros. Refuse to start instead.

        Only finish_reason == "stop" counts as failure; 'length' just means the
        token budget ran out, which is a different problem.
        """
        # The probe must carry the agent system prompt: a fine-tuned checkpoint only
        # emits its <think>/<answer> format with that prompt, so a bare probe is a
        # false positive. A genuinely broken model still fails.
        try:
            from .agent import prompts as _agent_prompts
            _probe_msgs = [{"role": "system", "content": _agent_prompts.SYSTEM_PROMPT},
                           {"role": "user", "content": "What is 2+2? Answer directly."}]
        except Exception:  # noqa: BLE001
            # Fall back to a bare prompt, but say so rather than silently changing
            # the criterion.
            print("[serve] WARNING: could not load the agent SYSTEM_PROMPT; probing "
                  "with a bare prompt, which can be a false positive for fine-tuned "
                  "checkpoints", flush=True)
            _probe_msgs = [{"role": "user", "content": "Reply with the single word OK."}]
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.served_model_name,
                      "messages": _probe_msgs,
                      "max_tokens": 1024, "temperature": 0.7},
                timeout=300,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]
        except Exception as e:  # noqa: BLE001
            self.stop()
            raise RuntimeError(f"[serve] delivery probe request failed: {e}") from e
        msg, finish = choice["message"], choice.get("finish_reason")
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        if not text.strip() and finish == "stop":
            self.stop()
            raise RuntimeError(
                f"[serve] {self.served_model_name} finished normally "
                f"(finish_reason=stop) but delivered empty text -- the parsing layer "
                f"swallowed the output. Current reasoning-parser="
                f"{'qwen3' if (self.is_qwen3 and self.reasoning_parser) else 'none'}。"
                f"If the model never emits </think> (a non-thinking variant), re-run "
                f"with run_eval.py --no-reasoning-parser. Refusing to start: "
                f"continuing yields an all-zero run with no visible symptom."
            )
        print(f"[serve] delivery probe OK (finish={finish}, {len(text.strip())} chars)")

    def start(self) -> Tuple[str, str]:
        os.makedirs(self.log_dir, exist_ok=True)
        env = dict(os.environ)
        gpu = self.gpu or pick_free_gpu()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"[serve] CUDA_VISIBLE_DEVICES={gpu}")
        else:
            print("[serve] WARNING: no free GPU detected; letting vLLM use defaults")

        self._log_path = os.path.join(self.log_dir, f"vllm_{self.port}.log")
        log = open(self._log_path, "w")
        cmd = self._cmd()
        print(f"[serve] launching: {' '.join(cmd)}")
        print(f"[serve] logs -> {self._log_path}")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        atexit.register(self.stop)

        # vLLM 0.24's /health is flaky: returns empty/503 for long stretches
        # during warmup and even intermittently when up, so a stuck server can
        # spin here until startup_timeout. /v1/models is reliable (verified
        # 200 while /health returns empty on this box), so probe that instead.
        ready_url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited early (code {self.proc.returncode}). See {self._log_path}"
                )
            try:
                ready = requests.get(ready_url, timeout=5).status_code == 200
            except Exception:  # noqa: BLE001
                # Swallow only "not up yet" connection errors.
                ready = False
            if ready:
                # _probe_nonempty()'s RuntimeError must escape: the probe has already
                # called self.stop(), so swallowing it surfaces as an unrelated
                # AttributeError next iteration.
                #    "AttributeError: 'NoneType' object has no attribute 'poll'" ——
                self._probe_nonempty()
                return self.base_url, self.served_model_name
            time.sleep(3)
        self.stop()
        raise TimeoutError(
            f"vLLM did not become healthy within {self.startup_timeout}s. See {self._log_path}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
