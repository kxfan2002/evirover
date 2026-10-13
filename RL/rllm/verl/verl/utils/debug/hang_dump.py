# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-mortem hang instrumentation: faulthandler self-dump + host-RAM probe.

A whole-process HANG (every thread frozen) leaves NO traceback because control
never reaches an ``except``/exit path -- so "print on error" cannot fire. These
helpers make each process photograph ITSELF instead:

* ``faulthandler.enable()``               -> all-thread dump on SIGSEGV/SIGABRT
* ``dump_traceback_later(repeat=True)``    -> auto all-thread dump when a process
                                              is wedged longer than the timeout
                                              (OPT-IN, default OFF -- see below)
* ``SIGUSR1`` handler                      -> ``kill -USR1 <pid>`` dumps on demand

The dumps go to the process's own stderr, which Ray captures into the per-rank
``worker-*.err`` log -- exactly where you look after a freeze. This replaces
external ``py-spy`` (which can't always introspect a wedged Ray actor).

All gated by env ``VDR_DIAG_INSTRUMENT`` (default ON) so production can disable
it without reverting code.

The periodic all-thread dump (``dump_traceback_later``) is OPT-IN via
``VDR_FH_STALL_SECS`` (default 0 = OFF). Its watchdog thread walks EVERY thread's
stack WITHOUT holding the GIL -- by design, so it can dump a GIL-wedged process.
But in a vLLM worker with heavy native-thread churn (inductor compile pool /
triton launchers / NCCL / ZMQ / async engine) that lock-free walk torn-reads a
half-torn-down ``PyThreadState`` -> SIGSEGV / "unknown opcode" / PyThreadState_Get
GIL-NULL interpreter corruption. That was the recurring ~step-3 crash. Set it to
a positive N only in a native-thread-quiet process (e.g. a driver-only debug
session); on-demand SIGUSR1 / external py-spy cover the worker case safely.
"""

import os
import sys

_ARMED = False


def diag_enabled() -> bool:
    """True unless VDR_DIAG_INSTRUMENT is explicitly falsey."""
    return os.environ.get("VDR_DIAG_INSTRUMENT", "1") not in ("0", "false", "False", "no", "")


def arm_hang_dump(tag: str = "") -> None:
    """Arm faulthandler auto-dump + SIGUSR1 on-demand dump for THIS process.

    Idempotent and best-effort: any failure is swallowed (instrumentation must
    never take down training). Call once per process, early (e.g. in a worker's
    ``__init__`` or the driver's ``__init__``).
    """
    global _ARMED
    if _ARMED or not diag_enabled():
        return
    try:
        import faulthandler

        # default file is captured at call time == the (already Ray-redirected)
        # sys.stderr, so dumps land in this rank's worker-*.err
        faulthandler.enable(all_threads=True)
        # default 0 = OFF. dump_traceback_later spawns a watchdog thread that
        # walks ALL thread stacks WITHOUT the GIL; under a vLLM worker's native
        # thread churn it torn-reads a half-torn-down PyThreadState -> SIGSEGV /
        # GIL-NULL interpreter corruption (the recurring ~step-3 crash). Opt in
        # only for a native-thread-quiet process via VDR_FH_STALL_SECS>0.
        timeout = float(os.environ.get("VDR_FH_STALL_SECS", "0"))
        if timeout > 0:
            # repeat=True re-arms after each fire: a genuinely wedged process
            # keeps dumping every `timeout`s; a healthy one dumps periodically
            # (harmless). Fires on wall-clock, not on inactivity. UNSAFE under
            # native-thread churn -- see the module docstring.
            faulthandler.dump_traceback_later(timeout, repeat=True)
        import signal

        if hasattr(signal, "SIGUSR1"):
            # `kill -USR1 <pid>` -> immediate all-thread dump; chain=True keeps
            # any pre-existing handler working.
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        _ARMED = True
        print(
            f"[hang_dump] armed tag={tag} pid={os.getpid()} stall={timeout}s "
            f"(SIGUSR1 = on-demand all-thread dump)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:  # never let instrumentation crash the process
        print(f"[hang_dump] arm failed tag={tag}: {e}", file=sys.stderr, flush=True)


def host_ram_used_gb():
    """Return (used_gb, total_gb) host RAM from /proc/meminfo.

    used = MemTotal - MemAvailable. Returns (nan, nan) on any read failure.
    """
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, val = line.partition(":")
                if val:
                    info[key.strip()] = float(val.split()[0])  # value is in kB
        total = info.get("MemTotal", 0.0) / 1024.0 / 1024.0
        avail = info.get("MemAvailable", 0.0) / 1024.0 / 1024.0
        return (total - avail, total)
    except Exception:
        return (float("nan"), float("nan"))


def proc_rss_by_group():
    """Compact per-process-group private-RAM (RssAnon) summary for THIS node.

    The node-total host-RAM curve tells us memory leaks; it does NOT tell us
    WHICH process. This buckets every /proc pid by cmdline into:
      * ``drv``  -> the calling process (the driver/TaskRunner running the fit loop)
      * ``vllm`` -> vLLM V1 EngineCore subprocesses (cmdline has EngineCore/VLLM)
      * ``wrk``  -> ray::WorkerDict FSDP actors (the 8 training/rollout workers)
      * ``rayo`` -> other ray infra (raylet/gcs/other ray:: actors)
      * ``oth``  -> everything else (api server, tool subprocs, etc.)

    Uses **RssAnon** (private anonymous pages: heap + torch CPU tensors) rather
    than VmRSS, so it EXCLUDES shared /dev/shm (Ray plasma holds the pixel data
    there and would be double-counted across every worker). A per-step climb in
    exactly one bucket localizes the leak; ``wrkMax`` catches a single hot rank
    (the one the OOM-killer targets). Best-effort: returns "" on any failure so
    it can never break a breadcrumb.
    """
    try:
        self_pid = os.getpid()
        g = {"drv": 0.0, "vllm": 0.0, "wrk": 0.0, "sbx": 0.0, "rayo": 0.0, "oth": 0.0}
        n = {"vllm": 0, "wrk": 0, "sbx": 0}
        wrk_max = 0.0
        procs = []  # (anon, pid, tag) for the top-N detail line
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
                anon = 0.0
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("RssAnon:"):
                            anon = float(line.split()[1]) / 1024.0 / 1024.0  # kB -> GB
                            break
            except Exception:
                continue
            # Match on markers that cannot occur inside the interpreter PATH. The
            # conda env is literally named "vdr-vllm", so a bare `"vllm" in cmd`
            # swallowed every un-setproctitle'd process in the cluster into the
            # vllm bucket -- which is exactly the bucket that was climbing.
            if pid == self_pid:
                bucket = "drv"
            elif "WorkerDict" in cmd:
                bucket = "wrk"
                wrk_max = max(wrk_max, anon)
            elif "EngineCore" in cmd or "VLLM::" in cmd or "vLLMHttpServer" in cmd \
                    or "vllm.entrypoints" in cmd or "vllm.v1" in cmd:
                bucket = "vllm"
            elif "forkserver" in cmd or "multiprocessing.spawn" in cmd \
                    or "resource_tracker" in cmd:
                # python-tool sandbox pool: forkserver children are persistent and
                # REUSED (no maxtasksperchild), so anything they retain accumulates.
                bucket = "sbx"
            elif "ray::" in cmd or "raylet" in cmd or "gcs_server" in cmd or "plasma" in cmd:
                bucket = "rayo"
            else:
                bucket = "oth"
            g[bucket] += anon
            if bucket in n:
                n[bucket] += 1
            procs.append((anon, pid, _proc_tag(cmd)))
        procs.sort(reverse=True)
        top = " ".join(f"{t}#{p}:{a:.1f}" for a, p, t in procs[:4])
        return (
            f"rssAnon[drv={g['drv']:.1f} "
            f"vllm={n['vllm']}x{g['vllm']:.1f} "
            f"wrk={n['wrk']}x{g['wrk']:.1f}(max{wrk_max:.1f}) "
            f"sbx={n['sbx']}x{g['sbx']:.1f} "
            f"rayo={g['rayo']:.1f} oth={g['oth']:.1f}]GB top[{top}]"
        )
    except Exception:
        return ""


def _proc_tag(cmd: str) -> str:
    """Short, greppable identity for one process; '' -> 'unknown'.

    Only used inside the top-N detail, so it favours being distinctive over being
    complete: interpreter paths are dropped (they are identical for every process
    here and were what broke the bucketing in the first place).
    """
    for marker in ("EngineCore", "vLLMHttpServer", "WorkerDict", "raylet",
                   "gcs_server", "forkserver", "resource_tracker", "plasma"):
        if marker in cmd:
            return marker
    for tok in cmd.split():
        base = tok.rsplit("/", 1)[-1]
        if base and not base.startswith("-") and "python" not in base:
            return base[:24]
    return "unknown"
