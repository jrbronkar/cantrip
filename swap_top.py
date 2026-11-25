"""Watch macOS swap traffic while a model runs in another terminal."""
import argparse
from collections import deque
import os
import re
import subprocess
import sys
import time

MIB = 1024 ** 2
COUNTERS = ("Swapins", "Swapouts", "Compressions", "Decompressions")


def sample():
    output = subprocess.check_output(
        ["/usr/bin/vm_stat"], text=True, env={**os.environ, "LC_ALL": "C"}
    )
    page_size = int(re.search(r"page size of (\d+) bytes", output)[1])
    counters = dict(re.findall(r"^([\w ]+):\s+(\d+)\.", output, re.M))
    return time.monotonic(), [int(counters[name]) * page_size for name in COUNTERS]


def graph(history, ceiling):
    levels = " ▁▂▃▄▅▆▇█"
    return "|" + "".join(levels[min(8, int(rate / ceiling * 8 + 0.5))]
                          for rate in history) + "|"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, help="Stop after this many readings")
    args = parser.parse_args()
    if args.interval <= 0 or (args.samples is not None and args.samples < 1):
        parser.error("interval and samples must be positive")
    live = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    histories = [deque(maxlen=36) for _ in COUNTERS]
    previous_time, previous = sample()
    readings = 0
    try:
        while args.samples is None or readings < args.samples:
            time.sleep(args.interval)
            now, values = sample()
            rates = [(current - old) / (now - previous_time) / MIB
                     for current, old in zip(values, previous)]
            previous_time, previous = now, values
            for history, rate in zip(histories, rates):
                history.append(rate)
            ceiling = max(1.0, *(max(history) for history in histories[:2]))
            lines = [f"{time.strftime('%H:%M:%S')}   macOS paging · MiB/s · system-wide"]
            lines += [f"{label:12} {rate:8.1f}  {graph(history, ceiling)}"
                      for label, rate, history in zip(("Swap in", "Swap out"), rates, histories)]
            lines.append(f"Compress     {rates[2]:8.1f}  Decompress {rates[3]:8.1f}")
            lines.append(f"Swap graphs: last {len(histories[0])} samples · full height = {ceiling:.1f} MiB/s")
            if live:
                lines.append("Ctrl+C to exit")
            print(("\033[H\033[J" if live else "") + "\n".join(lines), flush=True)
            readings += 1
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
