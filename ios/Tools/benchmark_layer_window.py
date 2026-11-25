"""Measure bounded packed-layer residency using the native cached decode as reference."""
import argparse
from collections import OrderedDict
import json
import mmap
from pathlib import Path
import statistics
import struct
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np

from export_fixture import mx
from runtime_mlx.inkling import model as inkling


class LayerWindow:
    def __init__(self, factories, capacity, pinned=()):
        self.factories = factories
        self.capacity = capacity
        self.pinned = frozenset(pinned)
        self.layers = OrderedDict()
        self.pending = []
        self.loads = 0
        self.load_seconds = 0.0

    def __getitem__(self, index):
        if index not in self.layers:
            if len(self.layers) == self.capacity:
                # Finish consumers before releasing the shard they refer to.
                self.flush()
                victim = next(layer for layer in self.layers if layer not in self.pinned)
                del self.layers[victim]
            start = perf_counter()
            self.layers[index] = self.factories[index]()
            self.load_seconds += perf_counter() - start
            self.loads += 1
        self.layers.move_to_end(index)
        return WindowLayer(self.layers[index], self.pending)

    def flush(self):
        if self.pending:
            mx.eval(*self.pending)
            self.pending.clear()


class WindowLayer:
    def __init__(self, layer, pending):
        self.layer = layer
        self.pending = pending

    @property
    def is_block_boundary(self):
        return self.layer.is_block_boundary

    def forward_fast(self, hidden, blocks, count, position, length, cache, slot, **kwargs):
        result = self.layer.forward_fast(hidden, blocks, count, position, length, cache, slot, **kwargs)
        self.pending.extend([result, cache.kv_f16[slot]])
        self.pending.extend(x for x in cache.conv_states[slot] if x is not None)
        return result


def prepare_layer(factory, path, mapped):
    """Save the runtime's packed arrays and keep only the Python object layout."""
    arrays = {}
    names = {}

    def layout(value):
        if isinstance(value, mx.array):
            key = id(value)
            if key not in names:
                name = f"array_{len(arrays)}"
                names[key] = name
                arrays[name] = value
            name = names[key]
            return lambda loaded: loaded[name]
        if isinstance(value, (list, tuple)):
            kind = type(value)
            children = [layout(item) for item in value]
            return lambda loaded: kind(child(loaded) for child in children)
        if isinstance(value, dict):
            children = {key: layout(item) for key, item in value.items()}
            return lambda loaded: {key: child(loaded) for key, child in children.items()}
        if type(value).__module__.startswith("runtime_mlx.inkling") and hasattr(value, "__dict__"):
            kind = type(value)
            children = {key: layout(item) for key, item in vars(value).items()}

            def restore(loaded):
                instance = object.__new__(kind)
                vars(instance).update({key: child(loaded) for key, child in children.items()})
                return instance

            return restore
        return lambda _: value

    restore = layout(factory())
    mx.save_safetensors(str(path), arrays)

    if mapped:
        with path.open("rb") as file:
            header_size, = struct.unpack("<Q", file.read(8))
            metadata = json.loads(file.read(header_size))
        data_base = 8 + header_size
        packed_path = path.with_suffix(".aligned")
        spans = {}
        offset = 0
        with path.open("rb") as source, packed_path.open("wb") as packed:
            for name, spec in metadata.items():
                if name == "__metadata__":
                    continue
                start, end = spec["data_offsets"]
                length = end - start
                rounded = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
                spans[name] = (offset, rounded, length)
                if length:
                    source.seek(data_base + start)
                    packed.seek(offset)
                    remaining = length
                    while remaining:
                        chunk = source.read(min(remaining, 4 * 1024**2))
                        if not chunk:
                            raise ValueError("Incomplete prepared weight file")
                        packed.write(chunk)
                        remaining -= len(chunk)
                offset += rounded
            packed.truncate(offset)
        path.unlink()
        dtypes = {
            "BOOL": np.bool_, "U8": np.uint8, "I8": np.int8, "U16": np.uint16,
            "I16": np.int16, "U32": np.uint32, "I32": np.int32, "U64": np.uint64,
            "I64": np.int64, "F16": np.float16, "BF16": np.uint16, "F32": np.float32,
        }

        def load_mapped():
            with packed_path.open("rb") as file:
                mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
            loaded = {}
            for name, spec in metadata.items():
                if name == "__metadata__":
                    continue
                offset, rounded, length = spans[name]
                dtype = spec["dtype"]
                if length:
                    itemsize = np.dtype(dtypes[dtype]).itemsize
                    view = np.ndarray(rounded // itemsize, dtype=dtypes[dtype], buffer=mapping, offset=offset)
                    array = mx.asarray(view, copy=False)[:length // itemsize].reshape(spec["shape"])
                else:
                    array = mx.array(np.empty(spec["shape"], dtype=dtypes[dtype]))
                loaded[name] = array.view(mx.bfloat16) if dtype == "BF16" else array
            return restore(loaded)

        return load_mapped

    def load():
        loaded = mx.load(str(path))
        mx.eval(*loaded.values())
        return restore(loaded)

    return load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--layers", type=int, default=4, choices=range(1, 25), metavar="1..24")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--pin-recurrent", action="store_true",
                        help="Keep layers shared across the three recurrent passes in memory")
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--prepared", action="store_true",
                         help="Prepare evictable packed layers once in a temporary directory")
    loading.add_argument("--mapped", action="store_true",
                         help="Map GPU-ready files into MLX using its no-copy input API")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if args.checkpoint.resolve().name != suite["checkpoint"] or not 2 <= args.tokens <= 16:
        parser.error("The checkpoint must match the suite and generation must have 2–16 tokens")
    prompt = suite["prompts"][0]["ids"]
    mx.set_cache_limit(64 * 1024**2)

    layer_type = inkling.PackedDecoderLayer
    with patch.object(inkling, "PackedDecoderLayer", lambda *params: lambda: layer_type(*params)):
        model = inkling.Inkling2LoopPackedMLX(args.checkpoint)
    if (not model.cfg.use_fast_metal or model.cfg.global_kv_mode != "fp16"
            or len(prompt) + args.tokens > min(64, model.cfg.sliding_window_size)):
        parser.error("This experiment requires the fast FP16 path and a short prompt")
    pinned = range(model.cfg.loop_start, model.cfg.loop_end) if args.pin_recurrent else ()
    if len(pinned) >= args.layers:
        parser.error("The window must contain the pinned layers and at least one other layer")
    if args.prepared or args.mapped:
        with TemporaryDirectory(prefix="inkling-packed-layers-") as directory:
            run(args, suite, prompt, model, pinned, Path(directory))
    else:
        run(args, suite, prompt, model, pinned, None)


def run(args, suite, prompt, model, pinned, prepared_dir):
    factories = model.layers
    if prepared_dir is not None:
        start = perf_counter()
        factories = [
            factory if index in pinned else prepare_layer(factory, prepared_dir / f"{index:02}.safetensors", args.mapped)
            for index, factory in enumerate(factories)
        ]
        print(json.dumps({"event": "prepared", "seconds": perf_counter() - start,
                          "bytes": sum(path.stat().st_size for path in prepared_dir.iterdir())}), flush=True)
        mx.reset_peak_memory()
    window = LayerWindow(factories, args.layers, pinned)
    model.layers = window
    cache = model.make_cache()
    inputs = prompt
    records = []
    for step in range(args.tokens):
        loads = window.loads
        load_seconds = window.load_seconds
        start = perf_counter()
        token, cache = model.forward_argmax(inputs, cache=cache)
        window.flush()
        mx.eval(token, *cache.state)
        end = perf_counter()
        token_id = int(token.item())
        inputs = [token_id]
        record = dict(
            step=step, id=token_id, seconds=end - start, shard_loads=window.loads - loads,
            load_seconds=window.load_seconds - load_seconds, peak_mlx_bytes=mx.get_peak_memory(),
            visits=cache.kv_len.count(len(prompt) + step),
        )
        if record["visits"] != model.cfg.total_visits:
            raise RuntimeError("A recurrent visit did not advance its cache")
        print(json.dumps(record), flush=True)
        records.append(record)

    decode = [record["seconds"] for record in records[1:]]
    print(json.dumps(dict(event="complete", layers=args.layers,
                          prepared=args.prepared or args.mapped, mapped=args.mapped,
                          pinned_layers=list(pinned),
                          checkpoint=suite["checkpoint"], prompt_ids=prompt,
                          ids=[record["id"] for record in records],
                          decode_tokens_per_second=len(decode) / sum(decode),
                          median_decode_seconds=statistics.median(decode))), flush=True)


if __name__ == "__main__":
    main()
