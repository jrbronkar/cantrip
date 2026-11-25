"""Prepare an FP16 cache and layer kernels for the Swift recurrent decoder probe."""
import argparse
import json
from pathlib import Path

from export_fixture import mx
from export_next_token_reference import make_constants_contiguous
from runtime_mlx.inkling.model import Inkling2LoopPackedMLX, block_attn_res_fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=3, choices=range(2, 9))
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    ids = suite["prompts"][0]["ids"]
    if suite["checkpoint"] != args.checkpoint.name or not ids or len(ids) + args.steps > 256:
        raise ValueError("Expected a matching checkpoint and a short prefill")
    args.output.mkdir()
    mx.set_cache_limit(64 * 1024**2)
    model = Inkling2LoopPackedMLX(args.checkpoint)
    cfg = model.cfg
    if not cfg.use_fast_metal or cfg.global_kv_mode != "fp16":
        raise ValueError("Expected the fast FP16 runtime")
    make_constants_contiguous(model, {})
    token, prefix = model.forward_argmax(ids)
    mx.eval(token, *prefix.state)
    token = token.reshape(1).astype(mx.int32)
    seed = token.item()
    conv_shapes = [tuple(x.shape) for x in prefix.conv_states[0]]
    if not all([tuple(x.shape) for x in slot] == conv_shapes for slot in prefix.conv_states):
        raise ValueError("Convolution cache layout differs by visit")
    if any(start != 0 for start in prefix.kv_start) or any(length != len(ids) for length in prefix.kv_len):
        raise ValueError("Expected an uncompacted cache with equal visit lengths")

    def save_cache(name, cache):
        arrays = {}
        for slot in range(cfg.total_visits):
            arrays[f"kv{slot}"] = cache.kv_f16[slot]
            for i, value in enumerate(cache.conv_states[slot]):
                arrays[f"conv{slot}_{i}"] = value
        mx.save_safetensors(str(args.output / name), arrays)

    save_cache("initial.safetensors", prefix)
    expected = []
    reference = prefix.clone()
    for step in range(args.steps):
        token, reference = model.forward_argmax(token, cache=reference)
        mx.eval(token, *reference.state)
        token = token.reshape(1).astype(mx.int32)
        if reference.kv_len != [len(ids) + step + 1] * cfg.total_visits:
            raise ValueError("Reference did not advance every visit")
        expected.append(token.item())
        save_cache(f"expected-{step}.safetensors", reference)

    boundaries = [layer.is_block_boundary for layer in model.layers]
    h = mx.zeros((1, cfg.hidden_size), dtype=mx.float32)
    blocks = lambda count: mx.zeros((count, *h.shape), dtype=h.dtype)
    pre_count = sum(boundaries[:cfg.loop_start])
    loop_count = sum(boundaries[cfg.loop_start:cfg.loop_end])
    head_count = sum(boundaries)
    mx.export_function(str(args.output / "embed.mlxfn"), lambda token: model.embed(token)[0], token)

    def head(hidden, residuals):
        value = block_attn_res_fused(model.final_in_q, residuals, head_count, hidden,
                                     norm_w=model.final_norm_w)[0]
        return model.lm_head.argmax(value[None, :, :])

    mx.export_function(str(args.output / "head.mlxfn"), head, h, blocks(head_count))
    for index, layer in enumerate(model.layers):
        count = sum(boundaries[:index])
        recurrent = cfg.loop_start <= index < cfg.loop_end
        kv = prefix.kv_f16[index]
        conv = prefix.conv_states[index]
        if kv.shape[2] != 256:
            raise ValueError("Probe expects 256-position cache buffers")

        def visit(tag, hidden, residuals, prior_hidden, prior_residuals, kv_input, *conv_inputs):
            position = len(ids) + tag.shape[0] - 1
            n_blocks = residuals.shape[0]
            prior_count = prior_residuals.shape[0]
            cache = model.make_cache()
            cache.seq_len = position
            cache.kv_len[0] = position
            cache.kv_cap[0] = kv.shape[2]
            cache.kv_f16[0] = kv_input
            cache.conv_states[0] = list(conv_inputs)
            messages = None
            if prior_hidden.shape[0]:
                messages = block_attn_res_fused(model.message_in_q[index - cfg.loop_start],
                                                prior_residuals if prior_count else None,
                                                prior_count, prior_hidden)
            result = layer.forward_fast(hidden, residuals if n_blocks else None, n_blocks,
                                        position, 1, cache, 0, messages=messages)
            return result, cache.kv_f16[0], *cache.conv_states[0]

        with mx.exporter(str(args.output / f"layer-{index:02d}.mlxfn"), visit) as export:
            for step in range(args.steps):
                tag = mx.zeros((step + 1,), dtype=mx.int32)
                export(tag, h, blocks(count), h[:0], blocks(0), kv, *conv)
                if recurrent and cfg.loop_passes > 1:
                    export(tag, h, blocks(count), h, blocks(loop_count - 1), kv, *conv)
        print(f"Exported physical layer {index}", flush=True)

    manifest = dict(checkpoint=suite["checkpoint"], prefixLength=len(ids), inputToken=seed,
                    expectedTokens=expected, visits=cfg.total_visits, boundaries=boundaries,
                    loopStart=cfg.loop_start, loopEnd=cfg.loop_end, loopPasses=cfg.loop_passes,
                    hiddenSize=cfg.hidden_size, convolutionWidths=[shape[-1] for shape in conv_shapes])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
