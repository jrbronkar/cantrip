"""Export the packed model's fixed-capacity chat forward pass."""
import argparse
import json
from pathlib import Path

from export_fixture import mx
from export_next_token_reference import make_constants_contiguous
from runtime_mlx.inkling.model import Inkling2LoopPackedMLX
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity", type=int, default=96, choices=(48, 96))
    args = parser.parse_args()
    capacity = args.capacity
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompt = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
              "<|message_user|><|content_text|>What is 2+2?<|end_message|>"
              "<|message_model|><|content_text|>")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    pad = tokenizer.token_to_id("<|endoftext|>")
    packed_ids = mx.array(ids + [pad] * (capacity - len(ids)), dtype=mx.int32)
    position = mx.array([len(ids) - 1], dtype=mx.int32)
    mx.set_cache_limit(64 * 1024**2)
    model = Inkling2LoopPackedMLX(args.checkpoint)
    expected = int(model.forward_argmax(ids)[0].item())
    print(f"Unpadded reference token: {expected}, prompt tokens: {len(ids)}", flush=True)

    def forward(tokens, last):
        try:
            mx.async_eval = lambda *arrays: None
            hidden, _, _ = model.forward_prefill_chunk(tokens, tap_visits=(model.cfg.total_visits,))
            return model.lm_head.argmax(mx.take(hidden, last, axis=0)[None, :, :])
        finally:
            mx.async_eval = async_eval

    async_eval = mx.async_eval
    # Preserve packed projections for longer mobile inputs instead of retaining
    # the Mac prefill optimization's dequantized loop matrices.
    model.cache_loop_weights = False
    actual = int(forward(packed_ids, position).item())
    if actual != expected:
        raise RuntimeError(f"Padded forward returned {actual}, expected {expected}")
    make_constants_contiguous(model, {})
    mx.export_function(str(args.output), forward, packed_ids, position)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        version=1, capacity=capacity, padID=pad, samplePrompt=prompt,
        sampleInputIDs=ids, sampleNextTokenID=expected,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
