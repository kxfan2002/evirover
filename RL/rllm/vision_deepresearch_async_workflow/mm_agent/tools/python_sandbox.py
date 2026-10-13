"""Process-isolated executor for model-authored Python (the `python` tool).

WHY THIS EXISTS
---------------
The `python` tool runs code the *model* writes (image math, calculations). It
used to `exec()` that code IN-PROCESS, inside the training driver, on the shared
ThreadPoolExecutor that `asyncio.to_thread` dispatches tool calls to
(adapter.py) — the same driver whose event loop feeds the 8 vLLM engines.

At temperature=1.0 the model occasionally emits code that:
  * segfaults a C extension or corrupts the redirect_stdout StringIO
    (observed: driver SIGSEGV in `stringio_dealloc`, 2026-08-04, killed a whole
    RL run at step 4), or
  * hogs the GIL / hangs (`while True`, huge alloc) and starves the event loop,
    which then trips the vLLM actor keepalive watchdog (ActorUnavailableError).

Either way a single bad snippet took down the entire training driver. Restricting
builtins can't stop a native segfault, so the only robust fix is to run the code
in a SEPARATE PROCESS with a hard wall-clock timeout: a crash or hang there
becomes a tool-error the rollout can see, and the driver survives.

DESIGN
------
* A lazily-created pool of persistent `forkserver` worker processes. forkserver
  forks children from a clean, single-threaded server process, so children never
  inherit the driver's CUDA context / held thread-locks (the classic
  multithreaded-fork deadlock). Workers are reused across calls (no per-call
  spawn cost) and respawned after a crash.
* Per call: send {code, image paths} over a Pipe; wait up to `timeout` for the
  reply. Timeout -> SIGKILL the worker and report a timeout error. Pipe EOF ->
  the worker died (segfault/abort/OOM-kill); report a crash error. Both mark the
  worker unhealthy so it is discarded, not returned to the idle pool.
* Concurrency is bounded by a semaphore (`MM_PYTHON_POOL_SIZE`, default 16).

Env knobs:
  MM_PYTHON_POOL_SIZE     max concurrent worker processes (default 16)
  MM_PYTHON_EXEC_TIMEOUT  per-call wall-clock seconds (default 20)
"""
from __future__ import annotations

import io
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as _mp

_POOL_SIZE = max(1, int(os.getenv("MM_PYTHON_POOL_SIZE", "16")))
_EXEC_TIMEOUT = float(os.getenv("MM_PYTHON_EXEC_TIMEOUT", "20"))
# Cap captured stdout at the source (inside the child) so a runaway print
# (e.g. dumping raw arrays / thousands of coords) never crosses back to the
# driver, where a full trajectory re-encode would otherwise materialize it as
# a ~900K-token tensor and spike host RAM. Default 3000 chars.
_MAX_OUTPUT_CHARS = max(0, int(os.getenv("MM_PYTHON_MAX_OUTPUT_CHARS", "3000")))

# Payload/reply are plain dicts of str/int/bytes -> always picklable.
# payload:  {"code": str, "original_path": Optional[str],
#            "img_items": List[Tuple[image_id, path]]}
# reply:    {"printed": str, "error": str, "image_bytes": Optional[bytes]}


# --------------------------------------------------------------------------- #
# Child side                                                                  #
# --------------------------------------------------------------------------- #
def _child_preinject(original_path: Optional[str],
                     img_items: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Rebuild the tool's pre-injected globals from image paths on disk.

    Mirrors PythonTool._preinject exactly (original_image, IMG_00N,
    observation_N aliases, num_images/num_observations) so valid code behaves
    identically to the old in-process path.
    """
    from PIL import Image  # imported in the child

    env: Dict[str, Any] = {}
    if original_path:
        try:
            env["original_image"] = Image.open(original_path).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    n = 0
    for image_id, path in img_items:
        if not image_id.startswith("IMG_"):
            continue
        try:
            image = Image.open(path).convert("RGB")
            env[image_id] = image
            n += 1
            env[f"observation_{n}"] = image
        except Exception:  # noqa: BLE001
            pass
    env["num_images"] = n
    env["num_observations"] = n
    return env


def _child_run_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one snippet and return a picklable reply dict.

    Ordinary Python exceptions are caught and returned as `error` (the worker
    stays alive and reusable). Only a native crash kills the worker, which the
    parent detects out-of-band.
    """
    import contextlib
    import copy as _copy
    import json as _json
    import math

    import numpy as np
    from PIL import Image, ImageDraw
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001
        cv2 = None
    try:
        import scipy  # noqa: F401
    except Exception:  # noqa: BLE001
        scipy = None

    g: Dict[str, Any] = {
        "np": np, "numpy": np, "Image": Image, "ImageDraw": ImageDraw,
        "math": math, "json": _json, "cv2": cv2, "scipy": scipy, "copy": _copy,
    }
    g.update(_child_preinject(payload.get("original_path"), payload.get("img_items", [])))
    g["_result_image"] = None

    buf = io.StringIO()
    error = ""
    try:
        with contextlib.redirect_stdout(buf):
            exec(payload["code"], g)  # noqa: S102 — model-authored, isolated by design
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    image_bytes: Optional[bytes] = None
    result_img = g.get("_result_image")
    if isinstance(result_img, Image.Image):
        try:
            if result_img.mode != "RGB":
                result_img = result_img.convert("RGB")
            obuf = io.BytesIO()
            result_img.save(obuf, format="JPEG", quality=95)
            image_bytes = obuf.getvalue()
        except Exception:  # noqa: BLE001
            image_bytes = None

    printed = buf.getvalue().strip()
    if _MAX_OUTPUT_CHARS and len(printed) > _MAX_OUTPUT_CHARS:
        printed = (
            printed[:_MAX_OUTPUT_CHARS]
            + f"\n[output truncated to {_MAX_OUTPUT_CHARS} chars — your python output was too long. "
            "A well-scoped analysis rarely needs this much text: print concise summaries "
            "(shapes, counts, key values) instead of dumping raw arrays, or call another tool.]"
        )
    return {"printed": printed, "error": error, "image_bytes": image_bytes}


def _worker_loop(conn) -> None:
    """Persistent child: run one snippet per request until the pipe closes."""
    # Warm the heavy imports once so reuse is fast.
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    while True:
        try:
            payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            reply = _child_run_one(payload)
        except BaseException as exc:  # noqa: BLE001 — never let a handler bug wedge the loop
            reply = {"printed": "", "error": f"{type(exc).__name__}: {exc}", "image_bytes": None}
        try:
            conn.send(reply)
        except Exception:  # noqa: BLE001 — parent gone
            break


# --------------------------------------------------------------------------- #
# Parent side                                                                 #
# --------------------------------------------------------------------------- #
class _Worker:
    def __init__(self, ctx):
        self.parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        self.proc.start()
        child_conn.close()  # only the child keeps its end

    def run(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Send a job and wait. Raises TimeoutError or a crash RuntimeError."""
        self.parent_conn.send(payload)
        if not self.parent_conn.poll(timeout):
            raise TimeoutError
        try:
            return self.parent_conn.recv()
        except (EOFError, OSError) as exc:  # worker died mid-exec
            raise RuntimeError("worker died") from exc

    def alive(self) -> bool:
        return self.proc.is_alive()

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.parent_conn.close()
        except Exception:  # noqa: BLE001
            pass

    def exitcode(self) -> Optional[int]:
        # On a crash the pipe EOFs before the child is reaped, so exitcode is
        # briefly None; a short join lets us report the actual signal number.
        if self.proc.exitcode is None:
            self.proc.join(0.5)
        return self.proc.exitcode


class _Pool:
    def __init__(self, size: int):
        self._size = size
        self._sem = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._idle: List[_Worker] = []
        self._ctx = None

    def _context(self):
        if self._ctx is None:
            try:
                ctx = _mp.get_context("forkserver")
                try:
                    ctx.set_forkserver_preload(["numpy", "PIL.Image"])
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001 — forkserver unavailable -> spawn (clean, slower)
                ctx = _mp.get_context("spawn")
            self._ctx = ctx
        return self._ctx

    def _acquire(self) -> _Worker:
        with self._lock:
            while self._idle:
                w = self._idle.pop()
                if w.alive():
                    return w
                w.kill()
        return _Worker(self._context())

    def _release(self, w: _Worker, healthy: bool) -> None:
        with self._lock:
            if healthy and w.alive():
                self._idle.append(w)
            else:
                w.kill()

    def run(self, payload: Dict[str, Any], timeout: float) -> Tuple[str, str, Optional[bytes]]:
        """Returns (printed, error, image_bytes). Never raises for user-code faults."""
        self._sem.acquire()
        w: Optional[_Worker] = None
        try:
            w = self._acquire()
            try:
                reply = w.run(payload, timeout)
            except TimeoutError:
                self._release(w, healthy=False)
                w = None
                return "", f"[Error] python execution timed out after {int(timeout)}s.", None
            except (RuntimeError, BrokenPipeError, OSError):
                code = w.exitcode() if w else None
                self._release(w, healthy=False)
                w = None
                sig = f" (signal {-code})" if isinstance(code, int) and code < 0 else ""
                return "", f"[Error] python execution crashed{sig}.", None
            self._release(w, healthy=True)
            w = None
            return reply.get("printed", ""), reply.get("error", ""), reply.get("image_bytes")
        finally:
            if w is not None:
                self._release(w, healthy=False)
            self._sem.release()


_pool: Optional[_Pool] = None
_pool_lock = threading.Lock()


def _self_test(pool: "_Pool") -> None:
    """Run one trivial snippet through a real worker at pool creation.

    In a live run the pool is first created inside the Ray driver; if a
    forkserver child can't import this module (wrong sys.path/cwd), the worker
    dies at startup and EVERY python call would silently return a crash-error,
    quietly killing the tool. This makes that failure LOUD in the log instead.
    """
    try:
        printed, error, _ = pool.run(
            {"code": "print('ok')", "original_path": None, "img_items": []},
            min(_EXEC_TIMEOUT, 30.0),
        )
        if error or printed.strip() != "ok":
            print(f"[python_sandbox] SELF-TEST FAILED printed={printed!r} error={error!r} "
                  f"— isolated worker cannot execute model code; python tool is DEAD", flush=True)
        else:
            print("[python_sandbox] SELF-TEST PASSED — isolated worker executes model code", flush=True)
    except Exception as exc:  # noqa: BLE001 — never block startup on the self-test
        print(f"[python_sandbox] SELF-TEST FAILED (exception): {type(exc).__name__}: {exc}", flush=True)


def _get_pool() -> _Pool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = _Pool(_POOL_SIZE)
                _self_test(pool)
                _pool = pool  # publish only after the self-test so callers see a vetted pool
    return _pool


def run_code_isolated(code: str,
                      original_path: Optional[str],
                      img_items: List[Tuple[str, str]],
                      timeout: Optional[float] = None) -> Tuple[str, str, Optional[bytes]]:
    """Run model code in an isolated worker. Returns (printed, error, image_bytes).

    A crash or timeout is reported via `error`; the calling process is never at
    risk. `image_bytes` is JPEG for a `_result_image` the code assigned, else None.
    """
    payload = {"code": code, "original_path": original_path, "img_items": list(img_items)}
    return _get_pool().run(payload, _EXEC_TIMEOUT if timeout is None else timeout)
