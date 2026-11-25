# Inkling M2 performance handoff

The goal is a long-lived local chat session on an 8 GB M2 Air with roughly one second to first visible text, about 7 tokens/sec, and enough headroom for recording. Warm performance met that target on short prompts. Startup still occasionally stalls during heavy paging. The current optimization question is what limits steady GPU decode when paging is quiet.

## Swift and iPhone track

The all-resident Swift port reached 18.0 tok/s on the warm M2 prompt. On iPhone 14 Pro, six-layer async decode ran at 0.424–0.433 tok/s versus 0.425–0.427 tok/s for the matched synchronous control. Both matched all 64 native greedy IDs, and their on-device KV and convolution cache fingerprints matched at every token. Async sampled process footprint peaked at 1.87 GiB versus 1.26 GiB for sync. [iPhone benchmark and commands](ios/INFERENCE-PROFILING.md#bounded-async-scheduling-on-iphone-14-pro)

The iPhone 14 Pro completed the 22-token prompt three times with all 24 layers retained after enabling the Increased Memory Limit entitlement. Final-16 throughput was 10.69–10.97 tok/s versus 18.08 tok/s on the M2 for the same prompt and configuration. All three phone runs matched all 32 native greedy IDs and ended with SIGKILL before producing a token on the second prompt. [All-layer benchmark and signing commands](ios/INFERENCE-PROFILING.md#increased-memory-signing-experiment)

## Run and orient

```sh
cd ~/sources/inkling-2-mini
./run.sh
```

- `run.sh` uses the local Python 3.12 virtual environment, the isolated LookAtMyQuant runtime in `.runtime-inkling-3.9gb`, and the offline 3.85 GB SQYFR v3 text checkpoint. `inkling_answer.py` handles formatting, streaming, and timings.
- `./run.sh --3.9gb` is still accepted. The 5.4 GB checkpoint was removed after the earlier benchmarks.
- Interactive defaults are serial decode, a 64 MiB MLX allocator cache, and a one-time 16-token warm-up before Ready. Each question has fresh context. Responses stop at the model's end marker by default.
- `./run.sh --timings --max-tokens 64` adds detailed timing and phase-level swap diagnostics for benchmarking. Plain `./run.sh` prints a compact colored summary.
- `profile_inkling.py` provides JSONL benchmark events. `swap_top.py` shows live system-wide swap traffic. `requirements.txt` pins the local runtime.

To recreate the 3.9 GB installation from the project directory:

```sh
UV_CACHE_DIR="$PWD/.uv-cache" bin/uv pip install --python .venv/bin/python \
  --target "$PWD/.runtime-inkling-3.9gb" --no-deps \
  'lookatmyquant @ git+https://github.com/ShinMegamiBoson/lookatmyquant.git@9dc5cbf7adb08cc41b5ba45c6313556e16f30bdc'
HF_HOME="$PWD/hf" HF_HUB_OFFLINE=0 .venv/bin/hf download \
  shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB \
  --revision 54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
```

## Evidence

- `chat-warmup16-distinct-20261121-221603.log` recorded first visible text at 0.92–1.02 s across three distinct prompts, steady decode near 7 tok/s, and only 14 MiB of system-wide swap writes across the answers. Startup duration was unmeasured.
- `profile-warm-64.jsonl` sustained 7.18 tok/s in a 64-token warm pass. Its cold pass had a 45.5 s inter-token gap. The 64 MiB cache improved first-response time across three paired comparisons with little effect on warm speed. The default cache always ran first in those pairs.
- `stack-profile-20261121-221026/events.jsonl` shows nearly 6 GiB of swap writes before token 10 on the first turn and zero writes on the warm turn. Native samples caught Metal compilation and MLX evaluation. System-wide swap counters cannot attribute faults to a specific allocation.
- DSpark initially hit a 1,024-thread Metal launch against an 832-thread pipeline limit. A profiler-only smaller-threadgroup workaround emitted tokens but peaked at 6.68 GB of active MLX allocations, versus about 5.9 GB serial. It timed out before a warm pass, so serial remains the default.
- `inkling-counters-20261121-233047.trace` is the successful CLI counter capture. It used `~/Library/Application Support/Instruments/Templates/MyTemplate.tracetemplate` with Performance Limiters enabled. Sampled interval averages showed roughly 18–45% ALU utilization and 1–9 GB/s external reads through much of the run, with early buffer-read limiter readings around 73–79%. The run wrote about 3 GiB to swap and lacked token alignment, so these numbers do not establish the steady-decode bottleneck.

## Next capture

### 3.9 GB optimisation pass

The 2026-11-22 run used AC power, the pinned 3.9 GB checkpoint, and a 64 MiB MLX cache. Raw logs are in `inference-bench-3.9gb/`.

| Warm-up tokens | Launch to Ready | Ready to first visible text | Launch to first visible text |
| ---: | ---: | ---: | ---: |
| 16 (first control) | 23.27 s | 0.52 s | 23.79 s |
| 0 | 6.62 s | 11.67 s | 18.29 s |
| 1 | 16.85 s | 1.40 s | 18.25 s |
| 16 (second control) | 25.28 s | 0.53 s | 25.81 s |

Each chat was a fresh process and answered the same short prose prompt. OS caches were not flushed. The zero- and one-token cases each have one sample. The one-token option trades roughly 0.9 s of first-prompt latency for an earlier Ready and earlier first answer than the 16-token controls.

| Decode mode | Warm median tok/s | Warm median first token | Peak active MLX memory, all turns | System-wide swap written, entire run |
| --- | ---: | ---: | ---: | ---: |
| Serial before | 18.29 | 0.62 s | 4.19 GiB | 4.47 GiB |
| DSpark | 15.22 | 0.63 s | 4.79 GiB | 8.38 GiB |
| Serial after | 18.22 | 0.64 s | 4.19 GiB | 6.27 GiB |

The decode comparison used six warm samples per process after a discarded first turn, the same 41-token counting prompt, and 64 generated tokens per turn. The runtime's standard DSpark path ran on M2 without the earlier threadgroup workaround. All seven drafted token sequences matched serial exactly for this prompt. DSpark was about 17% slower and peaked 0.60 GiB higher in MLX active allocations, so serial remains the default. System-wide swap includes other processes and startup. Neither broader workloads nor long contexts were qualified.

The packed tensors imply roughly 1.03 GB of weight payload per generated token before KV reads and steering. The estimate counts 48 layer visits, the full output head, and an average 8/128 share of expert bytes per visit. Dividing Apple's 100 GB/s peak memory bandwidth by that payload gives a rough streaming reference of 97 tok/s if weights are read from DRAM once per visit. Actual traffic depends on routing and cache reuse, and this calculation does not identify the bottleneck.

The 42-second Metal capture in `inference-bench-3.9gb/metal-profile-20261122-144511/` covered tokens 10–64 in the second turn at 18.6 tok/s, with 12 MiB of system-wide swap reads and zero writes in that interval. Across about 78,500 GPU-wide counter samples in that window, median ALU utilisation was 43%, compute occupancy 23%, and external GPU reads 21 GB/s (46 GB/s at p90). ALU and buffer-read limiters both reached about 73–75% at p90. The counters suggest mixed shader and buffer-load pressure with room below peak external bandwidth. A per-kernel GPU time breakdown would determine which kernel to tune first. The 25-second capture ended before steady decode, and the 55-second capture timed out during finalisation.

### Serial kernel and command-buffer follow-up

In the second steady trace window, 2,775 Inkling compute intervals covered 2.722 of 2.810 seconds (97%). Instruments exported command-buffer intervals, so the trace did not attribute that time to individual kernel names.

| Candidate | Control median | Candidate median | Outcome |
| --- | ---: | ---: | --- |
| Dense decode threadgroup 256 → 128 threads for stages with over 3,072 outputs | 18.09 tok/s | 18.21 tok/s | +0.7%, within noise in a two-candidate-turn screen |
| Serial MoE threadgroup 256 → 128 threads | 18.16 tok/s | 17.08 tok/s | −5.9% across six candidate turns |
| Serial MoE dispatch grid 160 → 320 groups | 18.16 tok/s | 17.63 tok/s | Five low-swap candidate turns trailed the controls in aggregate. One candidate turn was excluded for swap writes. |
| `MLX_MAX_MB_PER_BUFFER=256` | 17.83 / 17.74 tok/s | 17.77 tok/s | Warm decode was flat. Power source changed during the block. |

All tested variants matched the greedy control sequence on their respective short prompts. The 256 MB command-buffer run had an unusually fast first token after load (1.13 s versus 14.16 and 15.47 s) and less system-wide swap written across the run (787 MiB versus 3,687 and 3,942 MiB). It peaked at 4.28 GiB active MLX versus 4.19 GiB in both controls. Treat the startup result as provisional because macOS reported a change from AC to battery during the block. A reverse-order repeat stopped before loading the model because the power check reported battery. Launch defaults were left unchanged. The comparison logs are under `inference-bench-3.9gb/`.

The reverse-order repeat completed on AC power. First token after model load took 18.21 s at 256 MB, 1.63 s at the default, and 30.63 s at 256 MB. Model load took 6.38–7.08 s. System-wide swap writes were 3,868, 448, and 4,621 MiB respectively. Warm decode was 17.76, 17.76, and 17.86 tok/s, and all six output token sequences matched. The initial startup result did not replicate. Leave the buffer limit at its default and investigate first-turn paging separately.

### Earlier 5.4 GB capture plan

1. Start with one model process and close memory-heavy apps. The last disk check showed 5.3 GiB free. The duplicate workspace installation and 573 MiB generated XML were removed. Xcode and source traces remain. The retained launcher's help ran after cleanup. Inference was not rerun for that check.
2. For a manual capture, use Xcode’s record tool with the Metal profiling template. Perform a warm-up and one untraced turn, then timestamp three 64-token turns. The earlier automated capture stopped during warm-up before starting xctrace.
3. Inspect the generated `inkling.log`, `xctrace.log`, and `decode.trace`. Use the wall-clock token markers and trace start time to isolate tokens 10–64 on turns with little swap. Export only the needed counter tables. The previous export needed execution outside the sandbox after xctrace reported “Missing features,” and the combined XML consumed hundreds of MiB.
4. Quantify buffer-read/cache limiters, external bandwidth, ALU utilization, and occupancy in that steady window. Pick a kernel change only after tying it to the steady workload. Compare against approximately 7.1 tok/s serial with the same prompts, check greedy output and memory, and retain a baseline run without tracing.
