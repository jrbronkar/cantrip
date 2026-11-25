"""Compare an experimental setting against normal warm Inkling decode."""
import argparse
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from profile_inkling import MODEL, vm_sample


MAX_WINDOW_SWAP_MIB = 256


def main(model, wired_mib, moe_threadgroups, moe_grid_groups, pairs):
    import mlx.core as mx
    from runtime_mlx.inkling.fast_kernels import (
        DenseStage,
        UnifiedMoESide,
        _make_dense_stage_kernel,
        _make_unified_moe_kernel,
        _u32,
    )
    from runtime_mlx.runtime import Runtime

    print("Loading model…", flush=True)
    mx.set_cache_limit(64 * 1024**2)
    runtime = Runtime.load(model)
    print("Model loaded.", flush=True)
    prompt = runtime.encode_prompt(
        "List integers from 1 to 100 separated by commas. Start with 1.",
        thinking=False,
    )
    prompt.extend(runtime.tokenizer.convert_tokens_to_ids(
        ["<|message_model|>", "<|content_text|>"]
    ))
    default = DenseStage.matmul_rotated_f16
    default_moe = UnifiedMoESide.__call__
    set_wired_limit = mx.set_wired_limit

    @contextmanager
    def lower_wired_limit():
        first_call = True

        def limit_generation(size):
            nonlocal first_call
            if first_call:
                first_call = False
                size = min(size, wired_mib * 1024**2)
                previous = set_wired_limit(size)
                print(f"Applied generation wired limit: {size / 1024**2:.0f} MiB.",
                      flush=True)
                return previous
            return set_wired_limit(size)

        with patch.object(mx, "set_wired_limit", limit_generation):
            yield

    def smaller(self, x_rot_f16, T, W_cached=None):
        if T != 1 or self.total_out <= 3072:
            return default(self, x_rot_f16, T, W_cached)
        sg, ntg = 4, 160
        kernel = _make_dense_stage_kernel(self.n, self.fids, 1, rps=2, sg=sg,
                                          stage_rtr=self._stage_rtr)
        return kernel(
            inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels,
                    self.dims, *self._lut_args, _u32(self.total_out, self.n_mats, ntg)],
            grid=(ntg * sg * 32, 1, 1),
            threadgroup=(sg * 32, 1, 1),
            output_shapes=[(1, self.total_out)],
            output_dtypes=[mx.float32],
        )[0]

    def tuned_moe(self, x_rot_f16, eids_flat, x_div, sorted_cache=None, plan=None,
                  dynamic_routing=False, sorted_in=False, sorted_out=False, x_raw_f16=None):
        n_slots = int(eids_flat.shape[0])
        if n_slots > 8 or sorted_in or sorted_out or self.has_nvfp4:
            return default_moe(self, x_rot_f16, eids_flat, x_div, sorted_cache, plan,
                               dynamic_routing, sorted_in, sorted_out, x_raw_f16)
        sg, ntg = (4, 160) if moe_threadgroups else (8, moe_grid_groups)
        kernel = _make_unified_moe_kernel(self.n_cols, self.fids, "t1", rps=4, sg=sg,
                                          n_rg=self.n_rg, stage_rtr=self._stage_rtr)
        return kernel(
            inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid,
                    self.exp_code_off, self.exp_scale_off, *self._lut_args,
                    eids_flat.astype(mx.uint32), _u32(self.m_rows, n_slots, x_div, ntg)],
            grid=(ntg * sg * 32, 1, 1),
            threadgroup=(sg * 32, 1, 1),
            output_shapes=[(n_slots, self.m_rows)],
            output_dtypes=[mx.float32],
        )[0]

    def measure():
        before = vm_sample()
        tokens = []
        stream = runtime.stream_tokens(prompt, max_tokens=64)
        try:
            for chunk in stream:
                if not chunk.token_ids:
                    continue
                tokens.extend(chunk.token_ids)
                if len(tokens) == 10:
                    start = time.perf_counter()
                    start_swap = vm_sample()
                if len(tokens) == 64:
                    end = time.perf_counter()
        finally:
            stream.close()
        if len(tokens) != 64:
            raise RuntimeError(f"Expected 64 generated tokens, got {len(tokens)}")
        after = vm_sample()
        read = (after["Swapins"] - start_swap["Swapins"]) / 2**20
        written = (after["Swapouts"] - start_swap["Swapouts"]) / 2**20
        total_written = (after["Swapouts"] - before["Swapouts"]) / 2**20
        return tokens, 54 / (end - start), read, written, total_written

    def report(label):
        print(f"Starting {label}…", flush=True)
        tokens, speed, read, written, total = measure()
        low_swap = written == 0 and read < MAX_WINDOW_SWAP_MIB
        print(f"{label}: {speed:.2f} tok/s, window swap read/write "
              f"{read:.0f}/{written:.0f} MiB, whole-turn writes {total:.0f} MiB"
              f"{'' if low_swap else ' (paging; excluded)'}", flush=True)
        return tokens, speed, low_swap

    if moe_threadgroups or moe_grid_groups is not None:
        label = "smaller MoE" if moe_threadgroups else f"MoE grid {moe_grid_groups}"
        def experiment():
            return patch.object(UnifiedMoESide, "__call__", tuned_moe)
    elif wired_mib is None:
        label = "smaller"
        def experiment():
            return patch.object(DenseStage, "matmul_rotated_f16", smaller)
    else:
        label = f"wired {wired_mib} MiB"
        experiment = lower_wired_limit

    results = {"default": [], label: []}
    with experiment():
        candidate, _, _ = report(f"Warm-up {label}")
    reference, _, _ = report("Warm-up default")
    if candidate != reference:
        raise RuntimeError(f"{label} changed the greedy token IDs")
    for setting in ("default", *((label, "default") * pairs)):
        if setting == label:
            with experiment():
                tokens, speed, low_swap = report(setting)
        else:
            tokens, speed, low_swap = report(setting)
        if tokens != reference:
            raise RuntimeError(f"{setting} changed the greedy token IDs")
        if low_swap:
            results[setting].append(speed)

    print("All greedy token IDs matched.")
    if len(results["default"]) < pairs + 1 or len(results[label]) < pairs:
        print("Comparison inconclusive: at least one timed window had paging.")
        return
    for label, speeds in results.items():
        print(f"{label} median: {statistics.median(speeds):.2f} tok/s")


def supervise():
    before = vm_sample()["Swapouts"]
    deadline = time.monotonic() + 180
    with subprocess.Popen([sys.executable, "-u", __file__, "--worker", *sys.argv[1:]]) as worker:
        try:
            while True:
                try:
                    return worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if vm_sample()["Swapouts"] - before >= 8 * 2**30:
                        print("Comparison inconclusive: stopped after 8 GiB of system-wide swap writes.",
                              flush=True)
                        return 124
                    if time.monotonic() >= deadline:
                        print("Comparison inconclusive: stopped at the three-minute limit.", flush=True)
                        return 124
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    settings = parser.add_mutually_exclusive_group()
    settings.add_argument("--wired-mib", type=int, help="Test this process-local wired-memory limit")
    settings.add_argument("--moe-threadgroups", action="store_true", help="Test smaller serial MoE threadgroups")
    settings.add_argument("--moe-grid-groups", type=int, help="Test this many threadgroups in the serial MoE grid")
    parser.add_argument("--pairs", type=int, default=2, help="Number of interleaved candidate turns")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.wired_mib is not None and args.wired_mib <= 0:
        parser.error("--wired-mib must be positive")
    if args.moe_grid_groups is not None and args.moe_grid_groups <= 0:
        parser.error("--moe-grid-groups must be positive")
    if args.pairs < 2:
        parser.error("--pairs must be at least two")
    if args.worker:
        main(args.model, args.wired_mib, args.moe_threadgroups, args.moe_grid_groups, args.pairs)
    else:
        sys.exit(supervise())
