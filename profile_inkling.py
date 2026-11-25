#!/usr/bin/env python3
"""Profile local Inkling inference. Results are written as JSON lines."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / ("hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB"
                "/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e")


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def vm_sample():
    output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, env={**os.environ, "LC_ALL": "C"})
    page_size = int(re.search(r"page size of (\d+) bytes", output)[1])
    counters = dict(re.findall(r"^([\w ]+):\s+(\d+)\.", output, re.M))
    names = ("Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins")
    return {name: int(counters[name]) * page_size for name in names}


def worker(args):
    if args.model.resolve() == MODEL:
        sys.path.insert(0, str(ROOT / ".runtime-inkling-3.9gb"))

    import mlx.core as mx
    from runtime_mlx.runtime import Runtime

    def memory():
        return dict(active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory(),
                    peak_active_bytes=mx.get_peak_memory())

    if args.m2_dspark:
        from runtime_mlx.inkling.fast_kernels import DenseStage

        original = DenseStage.matmul_rotated_sg8_f16

        def smaller_threadgroup(self, x_rot_tr_f16, T=8, rb=2, sg=32, ntg=80):
            return original(self, x_rot_tr_f16, T=T, rb=rb, sg=min(sg, 16), ntg=ntg)

        DenseStage.matmul_rotated_sg8_f16 = smaller_threadgroup

    os.environ["HF_HOME"] = str(ROOT / "hf")
    if args.cache_mib is not None:
        mx.set_cache_limit(args.cache_mib * 1024 ** 2)
    mx.reset_peak_memory()
    emit("device", **mx.device_info())
    start = time.perf_counter()
    emit("load_start")
    runtime = Runtime.load(args.model, drafter=args.drafter)
    mx.synchronize()
    emit("loaded", seconds=time.perf_counter() - start, **memory())
    prompt = runtime.encode_prompt(args.prompt)
    for run in range(1, args.runs + 1):
        mx.reset_peak_memory()
        count = 0
        first_at = None
        last_at = None
        decode_count = 0
        start = time.perf_counter()
        emit("generation_start", run=run, prompt_tokens=len(prompt))
        for chunk in runtime.stream_tokens(prompt, max_tokens=args.tokens):
            now = time.perf_counter()
            if chunk.token_ids:
                last_at = now
                count += len(chunk.token_ids)
                if first_at is None:
                    first_at = now
                    decode_count = count
                emit("token_chunk", run=run, tokens=count, elapsed_s=now - start,
                     token_ids=list(chunk.token_ids), text=chunk.text, **memory())
            if chunk.finish_reason:
                seconds = last_at - first_at if first_at is not None else 0
                emit("generation_end", run=run, reason=chunk.finish_reason, tokens=count,
                     seconds=now - start,
                     first_token_s=first_at - start if first_at is not None else None,
                     decode_tok_s=(count - decode_count) / seconds if count > decode_count and seconds else None,
                     runtime_prompt_tok_s=chunk.prompt_tps,
                     runtime_decode_tok_s=chunk.generation_tps,
                     **memory())


def supervise(args):
    baseline = vm_sample()
    start = time.monotonic()
    emit("profile_start", model=str(args.model), drafter=args.drafter, token_limit=args.tokens,
         runs=args.runs, timeout_s=args.timeout, cache_mib=args.cache_mib, m2_dspark=args.m2_dspark, vm_baseline_bytes=baseline)
    child = subprocess.Popen([sys.executable, __file__, "--worker", *sys.argv[1:]],
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})
    timed_out = False
    try:
        while child.poll() is None:
            remaining = args.timeout - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                break
            try:
                child.wait(timeout=min(args.interval, remaining))
            except subprocess.TimeoutExpired:
                current = vm_sample()
                emit("vm_delta", elapsed_s=time.monotonic() - start,
                     **{name: current[name] - baseline[name] for name in baseline})
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        current = vm_sample()
        emit("profile_end", timed_out=timed_out, exit_code=child.returncode,
             elapsed_s=time.monotonic() - start,
             vm_delta_bytes={name: current[name] - baseline[name] for name in baseline})
    return 124 if timed_out else child.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--drafter", help="Path to an already downloaded drafter")
    parser.add_argument("--prompt", default="Write the numbers 1 through 30, separated by spaces.")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1, help="Prompts with a fresh context in one process")
    parser.add_argument("--timeout", type=int, default=180, help="Wall-clock limit in seconds, including model load")
    parser.add_argument("--interval", type=int, default=5, help="VM sampling interval in seconds")
    parser.add_argument("--cache-mib", type=int, help="Override the MLX allocator cache limit in MiB")
    parser.add_argument("--m2-dspark", action="store_true", help="Experimental smaller Metal threadgroups for DSpark")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.tokens, args.runs, args.timeout, args.interval) < 1:
        parser.error("tokens, runs, timeout, and interval must be positive")
    if args.cache_mib is not None and args.cache_mib < 0:
        parser.error("cache-mib must be zero or positive")
    if args.m2_dspark and not args.drafter:
        parser.error("--m2-dspark requires --drafter")
    if args.worker:
        worker(args)
    else:
        sys.exit(supervise(args))
