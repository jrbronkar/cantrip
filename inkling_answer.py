"""Generate an Inkling answer from an explicitly opened text channel."""
import argparse
import os
import sys
import time

import mlx.core as mx
from runtime_mlx.runtime import Runtime


def color(text, style, stream=sys.stderr):
    if stream.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ:
        return f"\033[{style}m{text}\033[0m"
    return text


def configure_terminal_input():
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":
        return

    try:
        import readline
    except ImportError:
        return

    if "libedit" in (readline.__doc__ or "").lower():
        readline.parse_and_bind(r"bind ^L ed-clear-screen")
    else:
        readline.parse_and_bind(r'"\C-l": clear-screen')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Optional output cap. Zero lets Inkling stop at its end marker.")
    parser.add_argument("--cache-mib", type=int, default=64, help="MLX allocator cache limit in MiB (default: 64)")
    parser.add_argument("--warmup-tokens", type=int, default=16,
                        help="Interactive startup warm-up tokens (default: 16; 0 skips warm-up)")
    parser.add_argument("--timings", action="store_true", help="Report early token gaps and paging per answer")
    args = parser.parse_args()
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be zero or positive")
    if args.cache_mib < 0:
        parser.error("--cache-mib must be zero or positive")
    if args.max_tokens < 0:
        parser.error("--max-tokens must be zero or positive")

    if args.prompt is None:
        print(color("Inkling 2 Mini", "1;36") + "  ·  Loading model…", file=sys.stderr, flush=True)
    mx.set_cache_limit(args.cache_mib * 1024 ** 2)
    runtime = Runtime.load(args.model)
    if args.prompt is not None:
        if not answer(runtime, args.prompt, args.max_tokens, timings=args.timings):
            sys.exit(1)
        return

    if args.warmup_tokens:
        warm_up(runtime, args.warmup_tokens)

    configure_terminal_input()
    print(color("● Ready", "1;32") + color("  ·  /quit to exit", "2"), file=sys.stderr)
    while True:
        try:
            question = input("You › ").strip()
            if question == "/quit":
                return
            if question:
                answer(runtime, question, args.max_tokens, timings=args.timings)
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()


def warm_up(runtime, total):
    print("Warming up Inkling before the first question…", file=sys.stderr, flush=True)
    prompt = runtime.encode_prompt(
        "List the integers from 1 to 200 in order, separated by commas. "
        "Start with 1 and do not summarize."
    )
    stream = runtime.stream_tokens(prompt, max_tokens=total)
    try:
        for chunk in stream:
            if chunk.token_ids and (chunk.generation_tokens == 1
                                    or chunk.generation_tokens % 8 == 0):
                print(f"  Warm-up tokens: {chunk.generation_tokens}/{total}",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\nWarm-up interrupted.", file=sys.stderr)
    finally:
        stream.close()


def answer(runtime, question, max_tokens, timings=False):
    started = time.perf_counter()
    visible_at = first_at = last_at = None
    first_count = 0
    if timings:
        from profile_inkling import vm_sample

        before = vm_sample()
        started = time.perf_counter()
        early = []
        swap_checkpoints = [("turn start", before)]
    tokenizer = runtime.tokenizer
    prompt = runtime.encode_prompt(question, thinking=False)
    prompt.extend(tokenizer.convert_tokens_to_ids(
        ["<|message_model|>", "<|content_text|>"]
    ))
    markers = ("<|end_message|>", "<|content_model_end_sampling|>",
               "<|message_model|>", "<|content_thinking|>")
    endings = set(tokenizer.convert_tokens_to_ids(list(markers)))
    # The serial runtime requires a positive integer and iterates lazily.
    stream = runtime.stream_tokens(prompt, max_tokens=max_tokens or sys.maxsize)
    answer = []
    count = 0
    try:
        for chunk in stream:
            if timings and chunk.token_ids and len(early) < 10:
                early.append(time.perf_counter() - started)
            count = chunk.generation_tokens
            if chunk.token_ids:
                last_at = time.perf_counter()
                if first_at is None:
                    first_at = last_at
                    first_count = count
            text = chunk.text
            for marker in markers:
                text = text.split(marker, 1)[0]
            if text:
                if visible_at is None and text.strip():
                    visible_at = time.perf_counter() - started
                answer.append(text)
                print(text, end="", flush=True)
            if timings and chunk.token_ids and count in (1, 10):
                swap_checkpoints.append((f"token {count}", vm_sample()))
            if endings.intersection(chunk.token_ids):
                break
    finally:
        stream.close()

    print()
    if visible_at is not None:
        elapsed = time.perf_counter() - started
        metrics = [f"{visible_at:.2f}s to first text", f"{count} generated tokens"]
        if first_at is not None and last_at > first_at and count > first_count:
            metrics.append(f"{(count - first_count) / (last_at - first_at):.1f} tok/s decode")
        metrics.append(f"{elapsed:.2f}s total")
        print(color("Inkling", "1;36") + "  " + color("  ·  ".join(metrics), "1;32"),
              file=sys.stderr, flush=True)
    if timings:
        elapsed = time.perf_counter() - started
        after = vm_sample()
        if visible_at is not None:
            print(f"[First visible text: {visible_at:.2f}s · turn: {elapsed:.2f}s]", file=sys.stderr)
        if early:
            gaps = [early[0]] + [b - a for a, b in zip(early, early[1:])]
            print("[First token and early gaps: " + ", ".join(f"{gap:.2f}s" for gap in gaps)
                  + "]", file=sys.stderr)
        print(f"[Prompt length: {len(prompt)} tokens]", file=sys.stderr)
        swap_checkpoints.append(("turn end", after))
        for (left, start_vm), (right, end_vm) in zip(swap_checkpoints, swap_checkpoints[1:]):
            read_mib = (end_vm["Swapins"] - start_vm["Swapins"]) / 1024 ** 2
            write_mib = (end_vm["Swapouts"] - start_vm["Swapouts"]) / 1024 ** 2
            print(f"[Swap {left} → {right}, read/written: {read_mib:.0f}/{write_mib:.0f} MiB]",
                  file=sys.stderr)
        reads = (after["Swapins"] - before["Swapins"]) / 1024 ** 2
        writes = (after["Swapouts"] - before["Swapouts"]) / 1024 ** 2
        print(f"[System-wide swap read/written: {reads:.0f}/{writes:.0f} MiB]", file=sys.stderr)
    if not "".join(answer).strip():
        print("Inkling ended the text channel without an answer.", file=sys.stderr)
        return False
    if max_tokens and count >= max_tokens:
        print(f"[Reached the {max_tokens}-token limit.]", file=sys.stderr)
    return True


if __name__ == "__main__":
    main()
