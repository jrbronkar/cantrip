"""Prepare, replay, and compare the iPhone inference benchmark."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import resource
import statistics
import sys
import time

PREFIX = "INKLING_BENCH "
CACHE_LIMIT = 64 * 1024**2
PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Count from one through twenty. Use numerals, separated by commas.",
    "long": ("A box has three red marbles, five blue marbles, and two green marbles. "
             "I remove two blue marbles and add four green marbles. "
             "How many marbles are now in the box? Explain the arithmetic."),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def prepare(args):
    from tokenizers import Tokenizer

    config_path = args.graph.with_suffix(".json")
    config = read(config_path)
    tokenizer_path = args.checkpoint / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sample = config["samplePrompt"]
    prompts = []
    for name in args.prompts:
        text = sample.replace(PROMPTS["short"], PROMPTS[name], 1)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) + args.tokens > config["capacity"]:
            raise ValueError(f"{name}: {len(ids)} input + {args.tokens} output exceeds graph capacity")
        prompts.append(dict(name=name, text=text, ids=ids))
    if tokenizer.encode(sample, add_special_tokens=False).ids != config["sampleInputIDs"]:
        raise ValueError("Checkpoint tokenizer differs from graph configuration")
    if args.tokens < 2 or args.repeats < 1:
        raise ValueError("Require at least two tokens and one repetition")
    suite = dict(
        graphSHA256=digest(args.graph), configSHA256=digest(config_path),
        tokenizerSHA256=digest(tokenizer_path), checkpoint=args.checkpoint.resolve().name,
        source=args.source, optimization=args.optimization, capacity=config["capacity"],
        tokens=args.tokens, repeats=args.repeats, prompts=prompts,
    )
    args.output.write_text(json.dumps(suite, indent=2) + "\n")
    print({prompt["name"]: len(prompt["ids"]) for prompt in prompts})


def emit(event, **fields):
    print(PREFIX + json.dumps(dict(event=event, **fields), sort_keys=True), flush=True)


def replay(args):
    from tokenizers import Tokenizer

    suite = read(args.suite)
    config_path = args.graph.with_suffix(".json")
    config = read(config_path)
    tokenizer_path = args.checkpoint / "tokenizer.json"
    if args.checkpoint.resolve().name != suite["checkpoint"]:
        raise ValueError("Checkpoint directory revision differs from suite")
    if (digest(config_path) != suite["configSHA256"] or
            digest(tokenizer_path) != suite["tokenizerSHA256"] or
            (args.backend == "graph" and digest(args.graph) != suite["graphSHA256"])):
        raise ValueError("Graph, graph configuration or tokenizer differs from suite")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for prompt in suite["prompts"]:
        if tokenizer.encode(prompt["text"], add_special_tokens=False).ids != prompt["ids"]:
            raise ValueError(f"Tokenizer mismatch for {prompt['name']}")

    import mlx.core as mx

    def counters():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return dict(
            mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
            mlx_peak_bytes=mx.get_peak_memory(), peak_resident_bytes=usage.ru_maxrss,
            minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt, block_inputs=usage.ru_inblock,
            cpu_seconds=usage.ru_utime + usage.ru_stime,
        )

    mx.set_cache_limit(CACHE_LIMIT)
    start = time.perf_counter()
    if args.backend == "native":
        from runtime_mlx.inkling.model import Inkling2LoopPackedMLX
        model = Inkling2LoopPackedMLX(args.checkpoint)
        if (model.cfg.loop_passes, model.cfg.num_experts_per_tok, model.cfg.global_kv_mode) != (3, 8, "fp16"):
            raise ValueError("Unexpected recurrent, routing, or KV configuration")
    setup = time.perf_counter() - start
    emit("start", backend=args.backend, capacity=suite["capacity"], source=suite["source"],
         optimization=suite["optimization"], graph_sha256=suite["graphSHA256"],
         config_sha256=suite["configSHA256"], tokenizer_sha256=suite["tokenizerSHA256"],
         checkpoint=suite["checkpoint"], os=platform.platform(), device=mx.device_info(),
         mlx_version=mx.__version__, build="installed MLX wheel", recurrent_passes=3,
         experts_per_token=8, sampling="greedy fixed count", drafting=False,
         kv="fp16" if args.backend == "native" else "no reuse",
         cache_limit_bytes=CACHE_LIMIT, setup_seconds=setup, counters=counters())
    for repetition in range(suite["repeats"]):
        for prompt in suite["prompts"]:
            ids = prompt["ids"].copy()
            if args.backend == "native":
                cache = model.make_cache()
            emit("prompt", name=prompt["name"], repeat=repetition, input_ids=ids,
                 tokens=suite["tokens"], counters=counters())
            prompt_start = time.perf_counter()
            for step in range(suite["tokens"]):
                if args.backend == "graph":
                    inputs = mx.array(ids + [config["padID"]] * (config["capacity"] - len(ids)), dtype=mx.int32)
                    last = mx.array([len(ids) - 1], dtype=mx.int32)
                start = time.perf_counter()
                if args.backend == "graph":
                    forward = mx.import_function(str(args.graph))
                    imported = time.perf_counter()
                    token = forward(inputs, last)[0]
                    del forward
                    invoked = time.perf_counter()
                    output = int(token.item())
                    end = time.perf_counter()
                    phases = {"import": imported - start, "invoke": invoked - imported, "evaluate": end - invoked}
                else:
                    token, _ = model.forward_argmax(ids if not step else ids[-1:], cache=cache)
                    output = int(token.item())
                    end = time.perf_counter()
                    phases = {"prefill_and_first_token" if not step else "cached_decode": end - start}
                if not step and ids == config["sampleInputIDs"] and output != config["sampleNextTokenID"]:
                    raise ValueError(f"Sample next token mismatch: {output}")
                ids.append(output)
                emit("token", step=step, id=output, seconds=end - start, phases=phases,
                     since_prompt_seconds=end - prompt_start, counters=counters())
    emit("complete")


def parse(path):
    with path.open() as stream:
        events = [json.loads(line.split(PREFIX, 1)[1]) for line in stream if PREFIX in line]
    if not events or events[0]["event"] != "start" or events[-1]["event"] != "complete":
        raise ValueError(f"Incomplete benchmark: {path}")
    runs = {}
    for event in events[1:-1]:
        if event["event"] == "prompt":
            prompt = event
            key = (prompt["name"], prompt["repeat"])
            if key in runs:
                raise ValueError(f"Duplicate prompt: {key}")
            runs[key] = (prompt, [])
        elif event["event"] == "token":
            runs[key][1].append(event)
    for prompt, tokens in runs.values():
        if [token["step"] for token in tokens] != list(range(prompt["tokens"])):
            raise ValueError(f"Incomplete prompt: {prompt['name']}")
    return events[0], runs


def summary(args):
    reference = None
    mismatch = False
    for path in args.logs:
        meta, runs = parse(path)
        print(f"{path} ({meta['backend']}, {meta['device']})")
        for key, (prompt, tokens) in runs.items():
            steady = tokens[1:]
            latencies = sorted(token["seconds"] for token in steady)
            elapsed = steady[-1]["since_prompt_seconds"] - tokens[0]["since_prompt_seconds"]
            phases = {name: round(statistics.median(t["phases"][name] for t in steady) * 1000, 1)
                      for name in steady[0]["phases"]}
            before = prompt["counters"]
            after = tokens[-1]["counters"]
            phase_text = ", ".join(f"{name} {value:.1f} ms" for name, value in phases.items())
            first = tokens[0]["since_prompt_seconds"]
            print(f"  {key[0]} run {key[1] + 1}, {len(prompt['input_ids'])} input, "
                  f"TTFT {first:.3f} s, {len(steady) / elapsed:.2f} tok/s, "
                  f"p50/p95 {statistics.median(latencies) * 1000:.1f}/"
                  f"{latencies[math.ceil(.95 * len(latencies)) - 1] * 1000:.1f} ms, {phase_text}")
            print(f"    peak RSS {after['peak_resident_bytes'] / 1024**2:.0f} MiB, "
                  f"major faults {after['major_faults'] - before['major_faults']}, "
                  f"CPU {(after['cpu_seconds'] - before['cpu_seconds']) / (first + elapsed):.2f} cores")
        if reference is None:
            reference = meta, runs
            continue
        ref_meta, ref_runs = reference
        fields = ("checkpoint", "tokenizer_sha256", "config_sha256", "graph_sha256", "capacity",
                  "recurrent_passes", "experts_per_token", "sampling", "drafting")
        if any(meta[field] != ref_meta[field] for field in fields) or runs.keys() != ref_runs.keys():
            raise ValueError("Benchmark metadata or prompt sets differ")
        for key, (prompt, tokens) in runs.items():
            ref_prompt, ref_tokens = ref_runs[key]
            if prompt["input_ids"] != ref_prompt["input_ids"] or len(tokens) != len(ref_tokens):
                raise ValueError(f"Workload differs for {key}")
            for token, ref_token in zip(tokens, ref_tokens):
                if token["id"] != ref_token["id"]:
                    print(f"  {key}: first mismatch at step {token['step']}: "
                          f"{token['id']} vs {ref_token['id']}")
                    mismatch = True
                    break
            else:
                print(f"  {key}: all {len(tokens)} greedy IDs matched {args.logs[0].name}")
    return int(mismatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_args = commands.add_parser("prepare")
    prepare_args.add_argument("checkpoint", type=Path)
    prepare_args.add_argument("graph", type=Path)
    prepare_args.add_argument("output", type=Path)
    prepare_args.add_argument("--source", required=True, help="Source commit or snapshot identifier")
    prepare_args.add_argument("--optimization", default="baseline instrumentation")
    prepare_args.add_argument("--tokens", type=int, default=16)
    prepare_args.add_argument("--repeats", type=int, default=2)
    prepare_args.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    prepare_args.set_defaults(run=prepare)
    replay_args = commands.add_parser("replay")
    replay_args.add_argument("checkpoint", type=Path)
    replay_args.add_argument("graph", type=Path)
    replay_args.add_argument("suite", type=Path)
    replay_args.add_argument("--backend", choices=("graph", "native"), default="graph")
    replay_args.set_defaults(run=replay)
    summary_args = commands.add_parser("summary")
    summary_args.add_argument("logs", nargs="+", type=Path)
    summary_args.set_defaults(run=summary)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
