"""Export one real packed embedding row per bucket and reference outputs."""
import argparse
import base64
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".runtime-inkling-3.9gb"))
from runtime_mlx.inkling.fast_kernels import (
    build_fast_luts, gqa_relpos_sinks_sdpa_fast, qkv_rel_prep_sliding_fused,
    rht_in_2304_fast, rht_in_3072_gated_fast, short_conv4_res_fused,
)
from runtime_mlx.inkling.model import Inkling2Config, PackedDecoderLayer, PackedEmbed, rms_norm


def export_projections(checkpoint, cfg, luts, weights, embed, token):
    fast_luts = build_fast_luts(luts)
    layer = PackedDecoderLayer(0, cfg, checkpoint, luts, fast_luts, weights)
    assert layer.is_sliding and layer.attn_in_q is None
    stage = layer.qkv_stage
    hidden = embed([token])[0]
    normalized = rms_norm(hidden, layer.in_ln, cfg.rms_norm_eps)
    rotated = rht_in_2304_fast(normalized, stage.signs_in, n_signs=5, T=1,
                               split_qkv=False, out_dtype=mx.float16)
    qkv = stage.matmul_rotated_f16(rotated, T=1)
    expected = np.array(qkv)[0]
    zero_kv = mx.zeros((1, 3, 1024), dtype=mx.float32)
    q, rel, kv, _, _ = qkv_rel_prep_sliding_fused(
        qkv, layer.q_scale, layer.rel_scale, zero_kv, layer.k_conv_w,
        zero_kv, layer.v_conv_w, 1, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
    )
    attention = gqa_relpos_sinks_sdpa_fast(
        q, kv, rel, layer.rel_proj_t_f16, normalized, layer.sink_w_f16, 1, 1, 0, 0, 1, True,
    )
    rotated_output = rht_in_3072_gated_fast(
        attention, qkv[:, 5504:], layer.o_stage.signs_in, out_dtype=mx.float16,
    )
    output = layer.o_stage.matmul_rotated_f16(rotated_output, T=1)
    post, _ = short_conv4_res_fused(
        mx.zeros((1, 3, 2304), dtype=mx.float32), output, layer.post_attn_conv_w, 1, 2304,
    )
    metadata = json.loads((checkpoint / "layer_00_meta.json").read_text())["linears"]
    names = {
        "Query": "self_attn.q_proj.linear",
        "Key": "self_attn.k_proj.linear",
        "Value": "self_attn.v_proj.linear",
        "Relative position": "self_attn.relative_position_bias.linear",
        "Output gate": "self_attn.o_proj.gate_proj",
    }
    def floats(tensor):
        return np.array(tensor).reshape(-1).tolist()

    with np.load(checkpoint / "layer_00_packed.npz") as arrays:
        def matrix(name, key, reference):
            meta = metadata[key]
            assert meta["fmt"] == "ht3" and not meta.get("out_rot")
            prefix = f"lin_{meta['safe']}_"
            words = arrays[prefix + "codes"].astype("<u4", copy=False)
            return dict(
                name=name, words=base64.b64encode(words.tobytes()).decode("ascii"),
                signs=floats(arrays[prefix + "signs_in"]),
                scales=floats(arrays[prefix + "scales"]),
                expected=floats(reference),
            )

        matrices = [
            matrix(name, key, expected[lo:hi])
            for (name, key), (lo, hi) in zip(names.items(), stage.slices, strict=True)
        ]
        output_matrix = matrix("Output projection", "self_attn.o_proj.linear", output)

    attention_fixture = dict(
        qScale=floats(layer.q_scale), relScale=floats(layer.rel_scale),
        qkClip=cfg.qk_l2_clip_threshold, relClip=cfg.rel_l2_clip_threshold,
        keyTap=floats(layer.k_conv_w[-1]), valueTap=floats(layer.v_conv_w[-1]),
        relativeZero=floats(layer.rel_proj_t_f16[0]), sinks=floats(layer.sink_w_f16),
        expectedAttention=floats(attention), output=output_matrix,
        postTap=floats(layer.post_attn_conv_w[-1]), expectedPost=floats(post),
    )
    return dict(token=token, norm=floats(layer.in_ln),
                table=floats(fast_luts["ht3"]), matrices=matrices, attention=attention_fixture)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer-zero", action="store_true", help="Include the first attention projections")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    cfg = Inkling2Config.from_json(checkpoint / "config.json")
    weights = mx.load(str(checkpoint / "steering.safetensors"))
    norm = weights["model.embed_tokens.embed_norm.weight"].astype(mx.float32)
    with np.load(checkpoint / "luts.npz") as arrays:
        luts = {name: mx.array(arrays[name]) for name in arrays.files}
    embed = PackedEmbed(checkpoint, luts, norm, cfg.rms_norm_eps)
    rows = []
    with np.load(checkpoint / "embed_packed.npz") as arrays:
        order = arrays["order"]
        start = 0
        for bucket in embed.meta:
            prefix = f"b{bucket['kb']}_"
            row = bucket["rows"] // 2
            token = int(order[start + row])
            start += bucket["rows"]
            fmt = bucket["opt"]
            if fmt == "lm6":
                codes = arrays[prefix + "codes"].astype(np.uint8).reshape(bucket["rows"], -1)
                words = np.ascontiguousarray(codes[row]).view("<u4")
                table = arrays[prefix + "levels"].astype(np.float16)
            else:
                words = arrays[prefix + "codes"].reshape(bucket["rows"], -1)[row]
                table = np.array(luts[fmt].astype(mx.float16))
            rows.append(dict(
                token=token, bits=bucket["k"], words=words.tolist(),
                scale=float(arrays[prefix + "scales"].reshape(-1)[row]),
                table=table.reshape(-1).tolist(),
                signs=arrays[prefix + "signs_in"].reshape(-1).tolist(),
            ))
    expected = np.array(embed([row["token"] for row in rows]))[0]
    for row, reference in zip(rows, expected, strict=True):
        row["expected"] = reference.tolist()
    fixture = dict(version=1, epsilon=cfg.rms_norm_eps, norm=np.array(norm).tolist(), rows=rows)
    if args.layer_zero:
        fixture["projections"] = export_projections(checkpoint, cfg, luts, weights, embed, rows[0]["token"])
    args.output.write_text(json.dumps(fixture, allow_nan=False))


if __name__ == "__main__":
    main()
