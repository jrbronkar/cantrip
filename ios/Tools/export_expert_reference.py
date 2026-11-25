"""Export a small reference for a direct read of the 3.9 GB checkpoint's layer 0."""
import argparse
import json
import struct
import zipfile
from pathlib import Path

from export_fixture import mx, np
from runtime_mlx.inkling.kernels import rht_apply_in, trellis_matvec


def offset(archive, file, name):
    entry = archive.getinfo(name + ".npy")
    if entry.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{name} must be stored without ZIP compression")
    file.seek(entry.header_offset)
    header = file.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP header")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    file.seek(name_size + extra_size, 1)
    version = np.lib.format.read_magic(file)
    if version != (1, 0):
        raise ValueError(f"Unsupported NPY version: {version}")
    np.lib.format.read_array_header_1_0(file)
    return file.tell()


def export_route_screen(checkpoint, output):
    keys = ("gu_rtr1_codes", "gu_rtr1_scales", "gu_signs_in", "gu_rg_map")
    with np.load(checkpoint / "luts.npz") as luts:
        table = luts["rtr1"].astype(np.float32).reshape(-1)
    inputs = ((np.arange(2304, dtype=np.float32) % 31) - 15) / 16
    columns = np.arange(2304)
    layers = []
    for layer in range(6):
        packed = checkpoint / f"layer_{layer:02d}_packed.npz"
        with zipfile.ZipFile(packed) as archive, packed.open("rb") as file:
            offsets = [offset(archive, file, key) for key in keys]
        with np.load(packed) as arrays:
            codes, scales, signs, groups = (arrays[key] for key in keys)
            if codes.shape != (2304, 128, 72) or not np.all(groups[:, :, 0] == 3):
                raise ValueError(f"Layer {layer} is not an all-RTR1 gate/up projection")
            rotated = np.array(rht_apply_in(mx.array(inputs[None]), mx.array(signs.astype(np.float32)))).reshape(-1)
            rounds = []
            for step in range(16):
                samples = []
                for slot in range(8):
                    expert = (step * 11 + slot * 13 + layer * 3) % 128
                    row = (step * 123 + slot * 257 + layer * 43) % 2304
                    location = groups[expert, row // 128, 1]
                    words = codes[location, row % 128].astype(np.uint64)
                    following = np.concatenate([words[1:], np.zeros(1, dtype=np.uint64)])
                    windows = words | (following << 32)
                    states = ((windows[columns // 32] >> (columns % 32).astype(np.uint64)) & 4095)
                    expected = float(np.dot(table[states], rotated[::-1]) * scales[location, row % 128])
                    samples.append(dict(expert=expert, row=row, expected=expected))
                rounds.append(samples)
        layers.append(dict(archiveBytes=packed.stat().st_size, offsets=offsets, rounds=rounds))
    output.write_text(json.dumps(dict(version=1, table=table.tolist(), layers=layers), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route-screen", action="store_true")
    args = parser.parse_args()
    if args.route_screen:
        export_route_screen(args.checkpoint, args.output)
        return
    packed = args.checkpoint / "layer_00_packed.npz"
    keys = ("gu_rtr1_codes", "gu_rtr1_scales", "gu_signs_in", "gu_rg_map")
    with zipfile.ZipFile(packed) as archive, packed.open("rb") as file:
        offsets = [offset(archive, file, key) for key in keys]
    with np.load(packed) as arrays, np.load(args.checkpoint / "luts.npz") as luts:
        codes, scales, signs, groups = (arrays[key] for key in keys)
        if [(a.shape, a.dtype.str) for a in (codes, scales, signs, groups)] != [
            ((2304, 128, 72), "<u4"), ((2304, 128), "<f2"),
            ((2304,), "<f2"), ((128, 18, 2), "<i2"),
        ]:
            raise ValueError("Unexpected layer 0 gate/up layout")
        expert = 0
        if not np.all(groups[expert, :, 0] == 3):
            raise ValueError("Expected RTR1 in every selected row group")
        inputs = ((np.arange(2304, dtype=np.float32) % 31) - 15) / 16
        rotated = rht_apply_in(mx.array(inputs[None]), mx.array(signs.astype(np.float32)))
        table = luts["rtr1"].astype(np.float16)
        expected = []
        for group, location in enumerate(groups[expert, :, 1]):
            rows = slice(group * 128, (group + 1) * 128)
            result = trellis_matvec(
                rotated, mx.array(np.ascontiguousarray(codes[rows, location])),
                mx.array(scales[rows, location].copy()), mx.array(table),
                mx.array([0], dtype=mx.uint32), 128, 2304, 1, 1, 4096, 0,
            )
            expected.extend(np.array(result).reshape(-1).tolist())
    args.output.write_text(json.dumps(dict(
        version=1, archiveBytes=packed.stat().st_size, offsets=offsets,
        table=table.reshape(-1).tolist(), expected=expected,
    ), allow_nan=False))


if __name__ == "__main__":
    main()
