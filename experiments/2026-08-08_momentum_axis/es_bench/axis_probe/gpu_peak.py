#!/usr/bin/env python
"""Wrap any command and record the GPU's peak memory from OUTSIDE the process.

Why this exists: `torch.cuda.max_memory_allocated()` counts only the calling process, and the
two methods here differ architecturally in process layout --

  ES    forces VLLM_ENABLE_V1_MULTIPROCESSING=0, because collective_rpc must reach the weights
        in-process. Its torch counter therefore INCLUDES vLLM's full KV cache.
  GRPO  runs vLLM as a separate process (vLLMHttpServer). verl's actor metric is read in the
        TaskRunner process, so it EXCLUDES the KV cache entirely.

That is a property of the two designs, not a misconfiguration, so no in-framework counter can
be made comparable. Sampling nvidia-smi from outside sidesteps the question: it sees the whole
card regardless of how a framework splits its processes. 2026-07-18's es_train.py recorded
`peak_mem_nvidia_smi_mib` for exactly this reason; the 2026-08-13 instrumentation omitted it and
produced an uncomparable measurement as a result. See MEMORY_REPORT.md.

Usage:
    python gpu_peak.py --out peak.json [--gpu 0] [--interval 0.5] -- <command> [args...]

Records both the card total and the sum over this process tree; they should agree when the GPU
is otherwise idle, and disagreeing means something else was running and the number is suspect.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time


def _smi(query, gpu):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query[0]}={query[1]}", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--label", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command given (put it after --)")

    baseline = _smi(("gpu", "memory.used"), a.gpu)
    baseline_mib = int(baseline) if baseline.isdigit() else 0
    if baseline_mib > 2000:
        print(f"[gpu_peak] WARNING: {baseline_mib} MiB already in use before start; "
              f"the peak below is NOT attributable to this command alone", flush=True)

    state = {"peak_total_mib": baseline_mib, "peak_tree_mib": 0, "samples": 0, "stop": False}

    def sample():
        while not state["stop"]:
            tot = _smi(("gpu", "memory.used"), a.gpu)
            if tot.isdigit():
                state["peak_total_mib"] = max(state["peak_total_mib"], int(tot))
            apps = _smi(("compute-apps", "used_memory"), a.gpu)
            if apps:
                s = sum(int(x) for x in apps.splitlines() if x.strip().isdigit())
                state["peak_tree_mib"] = max(state["peak_tree_mib"], s)
            state["samples"] += 1
            time.sleep(a.interval)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    elapsed = time.perf_counter() - t0
    state["stop"] = True
    t.join(timeout=3)

    res = {
        "label": a.label, "cmd": cmd, "returncode": rc,
        "baseline_mib": baseline_mib,
        "peak_total_mib": state["peak_total_mib"],
        "peak_total_gib": round(state["peak_total_mib"] / 1024, 2),
        "peak_over_baseline_gib": round((state["peak_total_mib"] - baseline_mib) / 1024, 2),
        "peak_compute_apps_mib": state["peak_tree_mib"],
        "peak_compute_apps_gib": round(state["peak_tree_mib"] / 1024, 2),
        "samples": state["samples"], "interval_s": a.interval, "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[gpu_peak] {a.label or cmd[0]}: peak {res['peak_total_gib']} GiB "
          f"(compute-apps {res['peak_compute_apps_gib']} GiB, baseline "
          f"{baseline_mib/1024:.2f} GiB, {state['samples']} samples, rc={rc})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
