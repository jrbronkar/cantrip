"""Benchmark routed expert reads in the pinned Metal decoder on an Apple Silicon Mac."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from export_fixture import mx
from runtime_mlx.inkling import model as inkling


class ExpertCache:
    def __init__(self, limit):
        self.limit = limit
        self.entries = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.read_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key, read):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]

        codes, scales = read()
        size = codes.nbytes + scales.nbytes
        self.misses += 1
        self.read_bytes += size
        if size <= self.limit:
            while self.bytes + size > self.limit:
                _, (old_codes, old_scales) = self.entries.popitem(last=False)
                self.bytes -= old_codes.nbytes + old_scales.nbytes
            self.entries[key] = codes, scales
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return codes, scales


class StreamedSide:
    """Rebind the native single-token projection to a compact slab for the current route."""

    def __init__(self, native, cache, path):
        if native.has_nvfp4 or native.n_rg <= 1:
            raise ValueError("Expected row-group trellis weights in the pinned checkpoint")
        self.native = native
        self.cache = cache
        self.path = path
        self.exp_fid = native.exp_fid
        self.fids = native.fids
        self.has_nvfp4 = native.has_nvfp4

        self.formats = np.array(native._kern_fid).reshape(128, native.n_rg)
        self.code_offsets = np.array(native.exp_code_off).reshape(self.formats.shape)
        self.scale_offsets = np.array(native.exp_scale_off).reshape(self.formats.shape)
        self.rows = native.m_rows // native.n_rg
        bits = np.array([1, 2, 3, 1, 2, 3, 4])
        self.code_counts = self.rows * native.n_cols * bits[self.formats] // 32

        self.code_path = path.with_suffix(".codes.npy")
        self.scale_path = path.with_suffix(".scales.npy")
        np.save(self.code_path, np.array(native.codes))
        np.save(self.scale_path, np.array(native.scales))
        self.codes = np.load(self.code_path, mmap_mode="r")
        self.scales = np.load(self.scale_path, mmap_mode="r")
        del native.codes, native.scales

    def read_expert(self, expert):
        codes = np.concatenate([
            self.codes[offset:int(offset) + int(count)]
            for offset, count in zip(self.code_offsets[expert], self.code_counts[expert])
        ])
        scales = np.concatenate([
            self.scales[offset:int(offset) + self.rows] for offset in self.scale_offsets[expert]
        ])
        return codes, scales

    def __call__(self, inputs, expert_ids, x_div, **kwargs):
        if expert_ids.size != 8:
            raise ValueError("Streamed experts require single-token inference")

        route = np.array(expert_ids.astype(mx.uint32))
        selected = np.unique(route[route != np.iinfo(np.uint32).max])
        if not selected.size or selected[-1] >= len(self.formats):
            raise ValueError("Invalid expert route")

        code_parts = []
        scale_parts = []
        code_offsets = np.zeros_like(self.code_offsets)
        scale_offsets = np.zeros_like(self.scale_offsets)
        code_base = scale_base = 0
        for expert in selected:
            codes, scales = self.cache.get((self.path, int(expert)), lambda: self.read_expert(expert))
            counts = self.code_counts[expert]
            code_offsets[expert] = code_base + np.cumsum(counts) - counts
            scale_offsets[expert] = scale_base + np.arange(len(counts)) * self.rows
            code_parts.append(codes)
            scale_parts.append(scales)
            code_base += codes.size
            scale_base += scales.size

        # The native projection captures these arrays as inputs to its lazy Metal operation.
        native = self.native
        native.codes = mx.array(np.concatenate(code_parts))
        native.scales = mx.array(np.concatenate(scale_parts))
        native.exp_code_off = mx.array(code_offsets.ravel())
        native.exp_scale_off = mx.array(scale_offsets.ravel())
        try:
            return native(inputs, expert_ids, x_div, **kwargs)
        finally:
            del native.codes, native.scales


def run(model, prompt_ids, tokens, cache=None):
    def memory_stats():
        return {} if cache is None else dict(
            expert_read_bytes=cache.read_bytes, expert_hits=cache.hits, expert_misses=cache.misses,
            expert_cache_peak_bytes=cache.peak_bytes,
        )

    state = model.make_cache()
    outputs = []
    seconds = []
    mx.reset_peak_memory()
    for step, input_id in enumerate([*prompt_ids, *([None] * (tokens - 1))]):
        start = perf_counter()
        token, state = model.forward_argmax([outputs[-1] if input_id is None else input_id], cache=state)
        mx.eval(token, *state.state)
        elapsed = perf_counter() - start
        if state.kv_len != [step + 1] * model.cfg.total_visits:
            raise RuntimeError("A recurrent visit did not advance its cache")
        if step >= len(prompt_ids) - 1:
            outputs.append(int(token.item()))
            seconds.append(elapsed)
    return outputs, dict(
        ids=outputs, decode_tokens_per_second=(tokens - 1) / sum(seconds[1:]),
        peak_mlx_bytes=mx.get_peak_memory(), **memory_stats(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=256)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if (args.checkpoint.name != suite["checkpoint"] or not 2 <= args.tokens <= 64
            or args.cache_mib < 0 or not suite["prompts"]):
        parser.error("Use a matching suite, 2–64 output tokens and a nonnegative cache budget")

    mx.set_cache_limit(64 * 1024**2)
    model = inkling.Inkling2LoopPackedMLX(args.checkpoint)
    if not model.cfg.use_fast_metal or model.cfg.global_kv_mode != "fp16":
        parser.error("This experiment requires the fast FP16 decoder")
    references = []
    for prompt in suite["prompts"]:
        ids, stats = run(model, prompt["ids"], args.tokens)
        print(json.dumps(dict(mode="resident", **stats)), flush=True)
        references.append(ids)

    cache = ExpertCache(args.cache_mib * 1024**2)
    with TemporaryDirectory(prefix="inkling-experts-") as directory:
        for index, layer in enumerate(model.layers):
            moe = layer.moe
            moe.gu_unified = StreamedSide(moe.gu_unified, cache, Path(directory) / f"{index:02d}-gu")
            moe.dn_unified = StreamedSide(moe.dn_unified, cache, Path(directory) / f"{index:02d}-dn")
        mx.clear_cache()
        for prompt, expected in zip(suite["prompts"], references):
            ids, stats = run(model, prompt["ids"], args.tokens, cache)
            matches = ids == expected
            print(json.dumps(dict(mode="streamed", matches=matches, **stats)), flush=True)
            if not matches:
                raise RuntimeError("Streamed greedy tokens disagree with resident decode")


if __name__ == "__main__":
    main()
