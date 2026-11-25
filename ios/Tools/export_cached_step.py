"""Export one fixed-position cached decode step for a mobile feasibility test."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from export_fixture import mx
from export_next_token_reference import make_constants_contiguous
from runtime_mlx.inkling.model import Inkling2LoopPackedMLX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if suite["checkpoint"] != args.checkpoint.resolve().name:
        raise ValueError("Checkpoint revision does not match suite")
    ids = suite["prompts"][0]["ids"]
    mx.set_cache_limit(64 * 1024**2)
    model = Inkling2LoopPackedMLX(args.checkpoint)
    if not model.cfg.use_fast_metal or model.cfg.global_kv_mode != "fp16":
        raise ValueError("This probe requires the native fast FP16 cache")
    first, prefix = model.forward_argmax(ids)
    mx.eval(first, *prefix.state)
    first_id = int(first.item())
    token_input = mx.array([first_id], dtype=mx.int32)
    reference = prefix.clone()
    expected, reference = model.forward_argmax(token_input, cache=reference)
    mx.eval(expected, *reference.state)
    expected_id = int(expected.item())

    conv_slots = [(visit, slot) for visit, convs in enumerate(prefix.conv_states)
                  for slot, value in enumerate(convs) if value is not None]
    conv_shapes = [prefix.conv_states[visit][slot].shape for visit, slot in conv_slots]
    conv_offsets = [0]
    for shape in conv_shapes:
        conv_offsets.append(conv_offsets[-1] + shape[0] * shape[1] * shape[2])

    def pack(cache):
        kv = mx.stack(cache.kv_f16)
        conv = mx.concatenate([cache.conv_states[visit][slot].flatten()
                               for visit, slot in conv_slots])
        return kv, conv

    kv_input, conv_input = pack(prefix)
    ref_kv, ref_conv = pack(reference)
    mx.eval(kv_input, conv_input, ref_kv, ref_conv)
    print(f"Native greedy IDs: {first_id}, {expected_id}; prefix length: {len(ids)}", flush=True)

    def forward(token, kv, conv):
        cache = model.make_cache()
        cache.seq_len = prefix.seq_len
        cache.kv_len = prefix.kv_len.copy()
        cache.kv_start = prefix.kv_start.copy()
        cache.kv_cap = prefix.kv_cap.copy()
        cache.kv_is_sliding = prefix.kv_is_sliding.copy()
        cache.kv_f16 = [kv[visit] for visit in range(model.cfg.total_visits)]
        for i, (visit, slot) in enumerate(conv_slots):
            cache.conv_states[visit][slot] = conv[conv_offsets[i]:conv_offsets[i + 1]].reshape(conv_shapes[i])
        with patch.object(mx, "async_eval", lambda *arrays: None):
            result, cache = model.forward_argmax(token, cache=cache)
        return (result, *pack(cache))

    make_constants_contiguous(model, {})
    graph = args.output.with_suffix(".mlxfn")
    print("Exporting one fixed-position decode graph", flush=True)
    mx.export_function(str(graph), forward, token_input, kv_input, conv_input)
    del model, prefix, reference
    mx.clear_cache()

    start = perf_counter()
    imported = mx.import_function(str(graph))
    after_import = perf_counter()
    actual, actual_kv, actual_conv = imported(token_input, kv_input, conv_input)
    del imported
    before_eval = perf_counter()
    mx.eval(actual, actual_kv, actual_conv)
    after_eval = perf_counter()
    kv_error = float(mx.max(mx.abs(actual_kv.astype(mx.float32) - ref_kv.astype(mx.float32))).item())
    conv_error = float(mx.max(mx.abs(actual_conv - ref_conv)).item())
    if int(actual.item()) != expected_id or not (kv_error <= 0.002 and conv_error <= 0.002):
        raise RuntimeError(f"Cached step parity failed: {actual.item()}, KV {kv_error}, conv {conv_error}")
    mx.save_safetensors(str(args.output.with_suffix(".safetensors")), {
        "token": token_input, "kv": kv_input, "conv": conv_input,
    })
    report = dict(
        checkpoint=suite["checkpoint"], prefix_ids=ids, input_token=first_id,
        expected_token=expected_id, visits=len(ref_kv), kv_max_abs_error=kv_error,
        convolution_max_abs_error=conv_error, import_seconds=after_import - start,
        invoke_seconds=before_eval - after_import, evaluate_seconds=after_eval - before_eval,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
