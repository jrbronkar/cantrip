"""Export the pinned model's reusable layer functions and on-device prefill signatures."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from export_fixture import mx
from export_next_token_reference import make_constants_contiguous
from runtime_mlx.inkling import fast_kernels as kernels, model as inkling


def u32_fields(*values):
    return mx.concatenate([v.reshape(1).astype(mx.uint32) if isinstance(v, mx.array)
                           else mx.array([v], dtype=mx.uint32) for v in values])


def dynamic_clip(q, q_scale, rel, rel_scale, position, length, sliding, q_clip, rel_clip, alpha, floor, tree_depth=None):
    if tree_depth is not None:
        raise ValueError("Dynamic query clipping does not use a speculative tree")
    inputs = [x.reshape(-1) for x in (q, q_scale, rel, rel_scale)]
    inputs += [u32_fields(int(sliding), position), kernels._f32_arr(q_clip, rel_clip, alpha, 1 / float(floor))]
    return kernels._make_q_rel_clip_kernel(False)(
        inputs=inputs, grid=(128, 24 * length, 1), threadgroup=(128, 1, 1),
        output_shapes=[(length, 24, 128), (length, 24, 16)],
        output_dtypes=[mx.float32, mx.float32],
    )


def dynamic_attention(q, kv, rel, projection, hidden, sinks, length, kv_length, start, position, capacity,
                      sliding, tree_depth=None, tree_mask=None, kv_fp8=False, kv_q4_scales=None):
    if length != 1 or tree_depth is not None or tree_mask is not None or kv_fp8 or kv_q4_scales is not None:
        raise ValueError("Dynamic attention requires single-token FP16 decode")
    inputs = [x.reshape(-1) for x in (q, kv, rel, projection, hidden, sinks)]
    inputs.append(u32_fields(length, kv_length, start, position, capacity, int(sliding)))
    return kernels._make_gqa_relpos_sdpa_kernel(False, kv_fp8=False)(
        inputs=inputs, grid=(128, 24, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 3072)], output_dtypes=[mx.float32],
    )[0]


@contextmanager
def dynamic_position(cache, position):
    def append(slot, value, sliding):
        updated = mx.slice_update(cache.kv_f16[slot], value, position, [2])
        cache.kv_f16[slot] = updated
        return updated, position + 1, 0, updated.shape[2], False, None

    with patch.object(cache, "update_kv_f16", append), \
            patch.object(inkling, "q_rel_clip_fused", dynamic_clip), \
            patch.object(inkling, "gqa_relpos_sinks_sdpa_fast", dynamic_attention):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--kv-capacity", type=int, choices=(64, 256, 1024), default=256)
    parser.add_argument("--serial-only", action="store_true",
                        help="Export only the single-token signatures used by chat")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    prompts = suite["prompts"]
    widths = sorted({len(prompt["ids"]) for prompt in prompts})
    if suite["checkpoint"] != args.checkpoint.name or not widths or widths[0] <= 1 or not 2 <= args.tokens <= 64:
        raise ValueError("Expected a matching suite with nonempty multi-token prompts")
    if widths[-1] + args.tokens > args.kv_capacity:
        raise ValueError("Prompt and reply exceed the requested KV capacity")
    args.output.mkdir()
    mx.set_cache_limit(64 * 1024**2)
    model = inkling.Inkling2LoopPackedMLX(args.checkpoint)
    cfg = model.cfg
    if not cfg.use_fast_metal or cfg.global_kv_mode != "fp16":
        raise ValueError("Expected the fast FP16 runtime")
    make_constants_contiguous(model, {})
    references = []
    for prompt in prompts:
        ids = prompt["ids"]
        if args.serial_only:
            state = model.make_cache()
            for input_id in ids[:-1]:
                _, state = model.forward_prefill_chunk([input_id], cache=state)
                mx.eval(*state.state)
            token, state = model.forward_argmax(ids[-1:], cache=state)
        else:
            token, state = model.forward_argmax(ids)
        expected = []
        for step in range(args.tokens):
            mx.eval(token, *state.state)
            expected.append(token.item())
            if step + 1 < args.tokens:
                token, state = model.forward_argmax(token.reshape(1).astype(mx.int32), cache=state)
        if state.kv_len != [len(ids) + args.tokens - 1] * cfg.total_visits:
            raise ValueError("Every reference visit must advance its KV cache")
        references.append(dict(text=prompt["text"], ids=ids, expectedTokens=expected))
        print(f"Computed {args.tokens} native reference tokens for {len(ids)}-token prompt", flush=True)
        del state, token
    boundaries = [layer.is_block_boundary for layer in model.layers]
    loop_blocks = sum(boundaries[cfg.loop_start:cfg.loop_end])
    cache_widths = [1024, 1024, 2304, 2304]
    kv = mx.zeros((2, 8, args.kv_capacity, 128), dtype=mx.float16)
    convolution = [mx.zeros((1, 3, width)) for width in cache_widths]
    position = mx.zeros((1,), dtype=mx.int32)
    lengths = [1] if args.serial_only else [1, *widths]

    def blocks(count, length):
        return mx.zeros((count, length, cfg.hidden_size))

    with mx.exporter(str(args.output / "embed.mlxfn"), lambda ids: model.embed(ids)[0]) as export:
        for length in lengths:
            export(mx.zeros((length,), dtype=mx.int32))

    def head(hidden, residuals):
        value = inkling.block_attn_res_fused(model.final_in_q, residuals[:, -1:, :], sum(boundaries),
                                              hidden[-1:], norm_w=model.final_norm_w)[0]
        return model.lm_head.argmax(value[None, :, :])

    with mx.exporter(str(args.output / "head.mlxfn"), head) as export:
        for length in lengths:
            export(mx.zeros((length, cfg.hidden_size)), blocks(sum(boundaries), length))

    for index, layer in enumerate(model.layers):
        count = sum(boundaries[:index])
        recurrent = cfg.loop_start <= index < cfg.loop_end

        def visit(position, hidden, residuals, prior_hidden, prior_residuals, kv_input, *conv_inputs):
            length = hidden.shape[0]
            n_blocks = residuals.shape[0]
            prior_count = prior_residuals.shape[0]
            cache = model.make_cache()
            messages = None
            if prior_hidden.shape[0]:
                messages = inkling.block_attn_res_fused(model.message_in_q[index - cfg.loop_start],
                                                        prior_residuals if prior_count else None,
                                                        prior_count, prior_hidden)
            if length == 1:
                cache.kv_f16[0] = kv_input
                cache.conv_states[0] = list(conv_inputs)
                with dynamic_position(cache, position):
                    result = layer.forward_fast(hidden, residuals if n_blocks else None, n_blocks,
                                                position, 1, cache, 0, messages=messages)
            else:
                cache.kv_f16[0] = kv_input
                cache.kv_cap[0] = args.kv_capacity
                result = layer.forward_fast(hidden, residuals if n_blocks else None, n_blocks,
                                            0, length, cache, 0, messages=messages)
            return result, cache.kv_f16[0], *cache.conv_states[0]

        with mx.exporter(str(args.output / f"layer-{index:02d}.mlxfn"), visit) as export:
            for length in lengths:
                hidden = mx.zeros((length, cfg.hidden_size))
                export(position, hidden, blocks(count, length), hidden[:0], blocks(0, length), kv, *convolution)
                if recurrent:
                    export(position, hidden, blocks(count, length), hidden, blocks(loop_blocks - 1, length),
                           kv, *convolution)
        print(f"Exported physical layer {index}", flush=True)

    manifest = dict(checkpoint=suite["checkpoint"], prompts=references, visits=cfg.total_visits,
                    kvCapacity=args.kv_capacity, serialOnly=args.serial_only,
                    boundaries=boundaries, loopStart=cfg.loop_start, loopEnd=cfg.loop_end, loopPasses=cfg.loop_passes,
                    hiddenSize=cfg.hidden_size, convolutionWidths=cache_widths)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
