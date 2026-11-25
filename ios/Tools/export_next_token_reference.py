"""Generate fixed-input next-token parity data from the packed Mac runtime."""
import argparse
import json
from pathlib import Path

from export_fixture import mx
from runtime_mlx.inkling.model import Inkling2LoopPackedMLX
from tokenizers import Tokenizer


def make_constants_contiguous(value, memo):
    """Preserve the layout of captured weights in the MLX export file."""
    key = id(value)
    if key in memo:
        return memo[key][1]
    result = value
    if isinstance(value, mx.array):
        result = mx.contiguous(value, allow_col_major=False)
        mx.eval(result)
    elif isinstance(value, tuple):
        result = tuple(make_constants_contiguous(item, memo) for item in value)
    elif isinstance(value, list):
        memo[key] = (value, value)
        value[:] = [make_constants_contiguous(item, memo) for item in value]
    elif isinstance(value, dict):
        memo[key] = (value, value)
        for name, item in value.items():
            value[name] = make_constants_contiguous(item, memo)
    elif type(value).__module__.startswith("runtime_mlx.inkling") and hasattr(value, "__dict__"):
        memo[key] = (value, value)
        for name, item in vars(value).items():
            setattr(value, name, make_constants_contiguous(item, memo))
    # Keep the original alive until traversal ends so Python cannot reuse its id.
    memo[key] = (value, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, help="Export a fixed-length MLX function")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    mx.set_cache_limit(64 * 1024**2)
    model = Inkling2LoopPackedMLX(args.checkpoint)
    token, _ = model.forward_argmax(ids)
    expected = int(token.item())
    args.output.write_text(json.dumps(dict(
        version=1, checkpointRevision=args.checkpoint.name, prompt=prompt,
        inputIDs=ids, expectedTokenID=expected,
        expectedText=tokenizer.decode([expected], skip_special_tokens=False),
    ), indent=2) + "\n")
    if args.graph:
        make_constants_contiguous(model, {})
        # The runtime's intermediate scheduling hints require concrete arrays.
        # The imported graph is evaluated by its caller.
        def forward(input_ids):
            async_eval = mx.async_eval
            mx.async_eval = lambda *arrays: None
            try:
                result, _ = model.forward_argmax(input_ids)
                return result
            finally:
                mx.async_eval = async_eval

        mx.export_function(str(args.graph), forward, mx.array(ids, dtype=mx.int32))


if __name__ == "__main__":
    main()
