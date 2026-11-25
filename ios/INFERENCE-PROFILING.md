# iPhone-first inference profiling

Raw device and Mac captures are local artifacts. Filenames in these notes identify the captures behind the recorded measurements.

## Mac launcher measurements

On an 8 GiB M2 MacBook Air, the pinned 3.9 GB checkpoint with a 64 MiB allocator cache took 23–25 seconds to reach Ready with the default 16-token warm-up. The first reply began about 0.53 seconds after Ready. One run with a one-token warm-up reached Ready in 16.85 seconds and produced its first token 1.40 seconds later. OS caches were not flushed between trials.

For a 41-token counting prompt and 64-token output, serial decode reached 18.22–18.29 tok/s warm. DSpark matched the greedy sequence for that prompt and reached 15.22 tok/s, with 4.79 GiB peak active MLX memory versus 4.19 GiB for serial. A repeat of the 256 MB Metal command-buffer limit test found no warm speed improvement and did not reproduce the initial startup gain. A steady Metal trace measured 18.6 tok/s with median external GPU reads of 21 GB/s and median ALU utilization of 43%. It did not identify individual kernel costs.

The experimental `ios/Tools/benchmark_expert_streaming.py` compares a resident Mac decoder with a byte-budgeted cache of routed expert slices. It has yet to produce an end-to-end streamed result and is separate from iPhone chat.

## Baseline and execution paths

The control is the text-only LookAtMyQuant Inkling 2 Mini 3.9 GB SQYFR v3 checkpoint at Hugging Face revision `54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e`. The pack is about 3.85 decimal GB, or 1.18 effective bits per text parameter. It uses packed NPZ shards, trellis tables and safetensors steering weights. The model has 24 physical layers and 48 logical layer visits. Layers 6–17 execute three times. Routing selects eight of 128 experts per token. The workspace has no project-scoped Git history, so the first suite identifies the local source snapshot by date and records the checkpoint and artifact hashes explicitly.

| | macOS control | iOS chat |
| --- | --- | --- |
| Host | Python and LookAtMyQuant at `9dc5cbf7`, with a local export hook | Swift and MLX Swift 0.32.3 |
| Weights | Native pack loaded once | Packed constants in an imported MLX function |
| Prefill | Actual prompt length, with a per-visit cache | Fixed 48 or 96 positions, including padding |
| Subsequent token | Single position with FP16 KV reuse | Fresh graph import, new fixed-width prefill, no KV reuse |
| Routing and projections | MLX GPU routing and fused Metal trellis projections, including dedicated T=1 paths | Exported prefill operations, including packed Metal kernels |
| Sampling in this experiment | Greedy, fixed count, no drafter | Greedy, fixed count, no drafter |

The graph exporter disables the native prefill option that materializes FP16 attention matrices for reuse at widths of 64 or more. The app drops the imported function before evaluating the lazy output. Prior attempts that retained it were terminated after import. MLX import describes constants as CPU-stream lazy loads from file offsets. Import wall time therefore mainly covers graph reconstruction. File reads, allocation, device work and host waits can all occur during evaluation. Treat the evaluation interval as wall time inclusive of these operations.

## Prior evidence

- The M2 3.9 GB control previously sustained 18.22–18.29 tok/s across six warm samples, peaking at 4.19 GiB active MLX memory. DSpark delivered 15.22 tok/s and peaked at 4.79 GiB. All seven drafted outputs matched the serial sequence for the counting prompt. The retained JSONL does not record proposal counts, accepted counts or verification-only time. These measurements do not establish an acceptance rate.
- The earlier 5.4 GB DSpark test hit a 1,024-thread launch against an 832-thread pipeline limit. A profiler-only workaround emitted tokens and peaked at 6.68 GB active MLX memory versus about 5.9 GB serial.
- On M2, the smaller serial MoE threadgroup lost 5.9%, a larger MoE grid lost throughput, and a smaller dense threadgroup was within noise. A 256 MB buffer cap left warm decode flat and its initial startup improvement failed to replicate. The low-swap M2 trace had GPU compute intervals covering 97% of the steady window, with evidence of both ALU and buffer-read pressure.
- On the iPhone 14 Pro, the 28-file scan mapped about 3.82 GB and reported 2.33 GB peak RSS. The individual packed expert projection matched the Mac reference within 4.2e-7 in the Metal output and took about 3 ms of GPU command time, with 215 ms elapsed including setup and copies. Neither probe measures full-generation residency.
- Prior 48-position app smoke testing matched four Mac greedy IDs. The original UI rate included the first output and did not isolate import, evaluation, streaming and tokenization. The new benchmark brackets each inference call directly.

See [Mac launcher measurements](#mac-launcher-measurements) and the [iOS probe guide](README.md).

## Upstream audit

- LookAtMyQuant head `826e5dde33ed615da0cc8c622aa241564d4132e6` retains the packed GPU routing, unified MoE projection, trellis decode with threadgroup lookup tables, recurrent model path and cached decode already used in the pinned Mac runtime. Optional FP8 and Q4 global KV modes can change generated tokens. FP16 is the control. DSpark is already supported on the pinned Mac checkpoint. The iOS export uses the prefill function and exports neither the KV state nor the drafter.
- The current source changes the generic half-warp trellis default group from 512 to 256 threads and increases split-K for attention spans of at least 32K. The current generation code adds opt-in thinking limits. The checkout has no Swift/iOS inference target. The local Mac model has a small export override to disable prefill loop-weight caching. Preserve it when comparing source versions. [LookAtMyQuant source](https://github.com/ShinMegamiBoson/lookatmyquant/tree/826e5dde33ed615da0cc8c622aa241564d4132e6)
- Gall documents an in-place mapped Rust/Metal backend with saved conversation state. Its published model is a 12.7 GB UD-IQ3_XXS GGUF and its published development system has 24 GB of RAM. These results do not establish compatibility with the 3.9 GB NPZ pack. [Gall](https://github.com/ltrgoddard/gall)
- Current MLX release v0.32.4 has a pre-M5 Metal SDPA memory change for head dimension 512. Inkling's custom attention and 128-wide heads need their own measurement. Python uses 0.32.4 and Swift resolves its own MLX dependency. Framework version is a remaining variable in the matched graph replay. [MLX v0.32.4](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- The v5 3.9 GB checkpoint `07939a4e` changes all layer shards, the output head and steering. The tokenizer and model config hashes match v3. Keep v3 for the current comparison. [SQYFR v5 card](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB/blob/07939a4e37e0955cacb4daf1002c7c3fdbadc025/README.md)

## Ranked hypotheses

1. **Repeated padded prefill and lazy weight loads.** The phone repeats a full fixed-width forward and reloads graph constants for every output. The same graph path also runs near 0.3 tok/s on M2. The first experiment measures the combined impact and separates import from evaluation.
2. **Memory pressure inside evaluation.** File-backed packed weights are loaded through lazy operations. Compression and page faults occur during the first phone token. Establish their contribution using File Activity and VM Tracker aligned to evaluation signposts. Process fault counters do not isolate storage reads or Metal allocation copies.
3. **Metal scheduling and timeout.** The phone reported a GPU command-buffer timeout on its second token. Collect Metal System Trace, using the evaluation signpost as the scope, to see command-buffer duration, gaps and any buffer that reaches the watchdog limit. A command-buffer cap is a testable reliability intervention if the default timeout repeats.
4. **Trellis kernels and GPU-family effects.** Use a fixed packed expert shape with setup outside the timing loop, then compare GPU time and occupancy on A16 and M2. Inspect per-pipeline dispatches and register pressure before changing the threadgroup size.
5. **Build, UI and thermal effects.** The Release headless experiment avoids the streaming UI and reports thermal state. Repeat a longer run with controlled power conditions before drawing sustained-performance conclusions.

## Reproduce

Use the same graph file on both devices and confirm the target already holds that file or stage the exact local copy. The manifest records hashes calculated on the Mac. The iPhone validates the token sequence through its tokenizer. It trusts the staged file identity, so check the transfer separately when the existing graph's provenance is uncertain. Do not run the large models concurrently on the same machine.

From the workspace root, prepare a local graph with the existing exporter if needed, then create the suite. Use a 96-position graph for the default three-prompt set. Use the existing 48-position chat graph for the short and medium cases.

```sh
checkpoint=hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
graph=/tmp/inkling-chat.mlxfn
.venv/bin/python -B ios/Tools/benchmark.py prepare "$checkpoint" "$graph" /tmp/inkling-benchmark.json \
  --source local-snapshot-20261124 --tokens 8 --repeats 2 --prompts short medium
export PYTHONPATH="$PWD/.runtime-inkling-3.9gb" HF_HUB_OFFLINE=1
.venv/bin/python -B ios/Tools/benchmark.py replay "$checkpoint" "$graph" /tmp/inkling-benchmark.json > mac-graph.log
.venv/bin/python -B ios/Tools/benchmark.py replay "$checkpoint" "$graph" /tmp/inkling-benchmark.json --backend native > mac-native.log
```

Build the InklingProbe scheme in Release and install it on the physical iPhone. For 96 positions, stage the matching files as `inkling-chat-96.json` and `inkling-chat-96.mlxfn`. For 48 positions, use `inkling-chat.json` and `inkling-chat.mlxfn`.

```sh
xcrun devicectl device copy to --device "$device" \
  --domain-type appDataContainer --domain-identifier local.inkling.probe \
  --source /tmp/inkling-benchmark.json --destination Documents/inkling-3.9gb-checkpoint/inkling-benchmark.json
xcrun devicectl device process launch --device "$device" --terminate-existing \
  --console local.inkling.probe -- --benchmark > iphone-graph.log 2>&1
```

Stop the console attachment after the `complete` or `failure` event. The CLI summary rejects incomplete logs and flags the first greedy mismatch per prompt and repetition. Repetitions use fresh KV on Mac native; the process remains warm. A new process does not clear the OS file cache. Raw token latencies exclude console logging; elapsed throughput includes small per-step harness overhead. The graph yields one argmax, so it cannot separately time prefill and the first output head. Native TTFT also includes the first output head. GPU utilization, actual storage reads and allocation peaks between token boundaries require Instruments. Use the `local.inkling.probe` / `Inference` signposts to align Metal System Trace, File Activity and VM Tracker with import, invocation and evaluation.

```sh
.venv/bin/python -B ios/Tools/benchmark.py summary mac-native.log mac-graph.log iphone-graph.log
```

Keep the suite, unedited JSONL events, run settings, device and OS, and optimization description with each comparison. For sustained tests, increase repetitions and compare first versus late repetitions at the same prompt length. Record cooldown and power conditions and sample thermal state. Use the same token IDs for correctness comparisons and investigate any mismatch before accepting a performance change.

## First measurements

The 48-position suite used 22- and 30-token prompts, eight outputs per prompt and two repetitions. On M2, the native backend delivered **17.48–17.82 tok/s**, with p50 **55.6–56.4 ms**. The exported graph delivered **0.37–0.38 tok/s**, with p50 **2.61–2.68 seconds**. All 32 greedy IDs matched. Median graph import was **131–141 ms**, invocation **2.6–2.9 ms**, and evaluation **2.46–2.54 seconds**. The M2 graph had zero major faults in three of the four prompt runs. The same-Mac comparison reproduces a roughly 47× slowdown with the iPhone execution path.

The iPhone 14 Pro Release control reached the matching first token in **3.319 seconds**, including **3.147 seconds** of evaluation, then reported a Metal command-buffer timeout. An autorelease-scope retry matched its first two IDs at **3.726** and **6.835 seconds**, then stopped making progress. These incomplete runs establish per-token observations. A steady p95 or throughput estimate requires a completed run. Low Power Mode was off and thermal state was nominal at the sampled points. The first run reported approximately 2.13 GB of peak active MLX allocations. No direct GPU-utilization or storage-read attribution is available from this capture.

The 96-position M2 graph processed 22-, 30- and 67-token input cases at 0.35, 0.35 and 0.30 tok/s. Its fixed width applies across the three cases. This graph is a separate control from the 48-position phone run.

Raw results and suite files are in [inference-bench-3.9gb](../inference-bench-3.9gb).

## Cached decode feasibility probe

`export_cached_step.py` starts from a native prefilled v3 state and exports a single T=1 decode graph at the prefix position. It passes FP16 KV and the convolution history explicitly for all 48 visits. The first fixture uses a 22-token prompt, a next-token input of 17 and a reference output of 10. The Mac import reproduced the ID and had zero maximum absolute error for both returned state tensors. Releasing the imported function before evaluating took 2.58 seconds inside evaluation on that export run.

Holding the same imported function alive on the M2 produced a 5.55-second first call and 1.16-second second call. Calls 3–8 took **61–81 ms**, with about **3.95 GB** of peak active MLX allocations. These are repeated evaluations of the same input and state. They show that the imported single-token graph can reuse weights once materialized. They do not measure a full autoregressive loop or establish memory headroom on the iPhone.

```sh
.venv/bin/python -B ios/Tools/export_cached_step.py "$checkpoint" \
  inference-bench-3.9gb/iphone-first-48-suite.json /tmp/inkling-cached-step
```

Stage the generated `inkling-cached-step.mlxfn`, `inkling-cached-step.safetensors` and `inkling-cached-step.json` in the same phone directory as the checkpoint. Launch with `--cached-step` for a one-shot parity and timing sample. Launch with `--cached-step-resident` for eight replays of the fixed step with one imported graph. A production path needs a dynamic position and state updates between calls. The next engineering question is what weight residency the A16 can sustain within the process footprint limit.

The first phone transfer failed with insufficient device storage. Clearing two older graphs and the incomplete transfer reclaimed about 8.2 GiB. The signed Release build installed and the complete 3.79 GB graph transferred. The active 48-position control graph and native v3 checkpoint remained staged.

The iPhone one-shot cached decode returned the expected ID 10. Import took **142 ms**, invocation **1.3 ms**, and evaluation **3.566 seconds**. It reported **1.87 GB** peak active MLX allocations. Holding the imported function alive caused the app to terminate with signal 9 before the first result on both attempts. That is consistent with memory pressure given the Mac's 3.95 GB active allocation peak in this mode. Signal 9 alone does not establish the cause.

## Direct mapped expert weights

The Release probe compares the existing copy-and-gather path with a shared Metal buffer backed by a page-aligned private mapping of the staged v3 layer-0 NPZ. The GPU reads the original interleaved RTR1 codes and FP16 scales. Both kernels use the same SIMD trellis dot product, input and lookup table. The mapping lifetime follows the Metal buffer. A private mapping protects the checkpoint file from changes. [Metal no-copy buffers](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))

Run the app with `--copied-expert` or `--mapped-expert` in a fresh process using the staged layer archive and expert reference. Each launch uses one pipeline, one set of buffers and 16 sequential dispatches. The first dispatch is sampled separately from the warm median of the subsequent 15. OS file caches are shared across processes. The A16 test ran copied, mapped, mapped, copied with nominal thermal state. Both backends and both hardware platforms matched the reference with maximum absolute error **4.17e-7**.

| Device and path | First dispatch wall time | Warm median GPU time | Peak process RSS |
| --- | ---: | ---: | ---: |
| A16 copied | 3.41–4.55 ms | 1.52–1.69 ms | 90.8–92.5 MiB |
| A16 mapped | 24.45–108.08 ms | 1.14–1.36 ms | 53.1 MiB |
| M2 copied | 3.63 ms | 0.78 ms | 51.1 MiB |
| M2 mapped | 155.71 ms | 1.02 ms | 12.7 MiB |

Buffer setup was under 0.5 ms in all runs. The mapped first-use cost showed up in the dispatch wall interval. This experiment did not attribute it to physical storage reads versus Metal resource preparation. The measurements cover one expert gate/up projection and a single shard. Peak RSS alone does not establish how many mapped buffers the phone can hold. Use the result to test a bounded window with multiple distinct shards and measure first-use cost across a real layer traversal before using it in decode.

Raw captures are in the A16 log (`inference-bench-3.9gb/expert-a16-buffer-modes.log`) and the M2 log (`inference-bench-3.9gb/expert-m2-buffer-modes.jsonl`).

## Bounded-weight cached decoding on Mac

`benchmark_layer_window.py` exercises the full v3 recurrent decoder with native prefill followed by single-token FP16 cached steps. It loads physical layers into a bounded window and finishes queued consumers before evicting a layer. It records layer rebuild time separately and checks that every one of the 48 visits advances its cache each step. Use the original Mac native log for the greedy reference.

```sh
.venv/bin/python -B ios/Tools/benchmark_layer_window.py "$checkpoint" \
  inference-bench-3.9gb/iphone-first-48-suite.json --layers 12
```

All tested windows reproduced the same eight greedy tokens as the native Mac control for the 22-token prompt. With four layers, steady decode was 0.15 tok/s at 1.11 GiB peak active MLX and rebuilt 48 layer instances per token. With 12 layers, decode was 0.21 tok/s at 2.44 GiB peak active MLX and rebuilt 24 layers per token. Construction accounted for 92% and 88% of their decode time respectively. A 13-layer window with layers 6–17 pinned reached 0.46 tok/s at 2.57 GiB peak active MLX, with 12 rebuilds per token. Under the 24-layer control, warm tokens took 55–59 ms once weights had been loaded, at a 3.94 GiB peak. A 1.98-second initial decode step reduced the seven-step aggregate to 3.02 tok/s.

These are Mac measurements. The limited windows use the current Python loader, which reads NPZ arrays and constructs the runtime's packed Metal buffers on every miss. The A16 mapping experiment established a way to use checkpoint bytes directly for one projection. A mobile decoder needs to reuse compatible packed buffers or read compatible mapped layouts across all kernels. Recreating the current Python layer loader on the phone would carry over the dominant steady-state cost.

### Prebuilt and mapped layer buffers on Mac

Repeat the 13-layer window with `--pin-recurrent --prepared` to build evictable layers once and reload their GPU-ready arrays. Use `--pin-recurrent --mapped` to wrap page-aligned arrays from private file mappings with `mx.asarray(copy=False)`. The ordinary safetensors offsets could not be adopted without copying. The mapped test adds page padding and excludes preparation from decode timing. Both variants use temporary files and leave the reference installation intact.

| 13-layer window | Decode tok/s | Median measured layer load per token | Peak active MLX |
| --- | ---: | ---: | ---: |
| Rebuild from source NPZ | 0.46 | 1.75 s | 2.57 GiB |
| Read prebuilt arrays | 0.58 | 1.14 s | 2.86 GiB |
| Map prebuilt arrays without copy | 0.27 | 0.08 s | 3.01 GiB |

Both variants matched all eight greedy IDs and advanced all 48 cache slots on every step. Mapping shortened the measured loader calls but increased the remaining wall time. The benchmark does not separate GPU-side preparation, paging and synchronization in that interval. This run rejects mapping and evicting whole layer tensors as a throughput fix. A subsequent experiment can split expert payloads into separately addressable slices and test loading only routed experts with reused GPU buffers.

## Six-shard routed-weight throughput screen

The gate/up screen stages eight changing, predetermined expert routes from each of six shards (physical layers 0–5). These six shards use RTR1 for every gate/up group. The screen follows the native loader's lookup order, where the route map indexes a contiguous 128-row group in the first storage dimension. The earlier single-expert fixture and probe agreed on a different addressing order. They established that Metal could read the mapping. Their parity alone did not establish parity with native expert routing.

Generate the small fixture with `ios/Tools/export_expert_reference.py "$checkpoint" /tmp/inkling-route-screen.json --route-screen`, copy it beside the staged v3 weights, and launch the Release app with `--route-screen`. The screen runs 16 rounds with six projections per round, stages 5,382,144 bytes of codes and expanded scales per projection into shared buffers reused across all rounds, and compares one sampled output row per expert with an independent Mac-generated reference. The GPU uses the existing one-SIMD-group-per-row probe kernel. Timing includes synchronous staging and command completion. Routing, attention, down projections, recurrent messaging and the head are outside this measurement.

| Device | Warm six-shard wall p50 | Staging p50 | GPU p50 | Sampled max abs error |
| --- | ---: | ---: | ---: | ---: |
| A16 | 137.6 ms | 63.6 ms | 68.0 ms | 4.77e-7 |
| M2 | 118.7 ms | 64.7 ms | 47.7 ms | 4.77e-7 |

Warm is the 15 rounds after the initial round. The medians are independently calculated. A16 warm wall ranged from 94.4 to 524.6 ms and thermal state was nominal at both endpoints. A16 peak process RSS was 525 MiB. The run excludes app launch, compilation, and the first round. At 5 tok/s the full token would have 200 ms. This one component exceeded the proposed 20 ms screening budget by about 7×. The run rejects this particular synchronous staging and generic kernel combination as a path to 5 tok/s. It does not determine the ceiling for the native fused top-8 kernel or a different weight access strategy.

Raw captures are in the A16 log (`inference-bench-3.9gb/route-screen-a16.log`) and the M2 log (`inference-bench-3.9gb/route-screen-m2.json`). The linked [inference optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill/blob/main/skills/inference-optimisation/SKILL.md) was used to interpret the screen against the time budget. A full candidate still needs measured A16 bandwidth and computation ceilings, interleaved controls, and an end-to-end correctness and stability run.

### Direct-mapped routes and half-SIMD scheduling

Use the same six-shard fixture with `--route-screen-mapped` to pass 584 bytes of route metadata per projection and let Metal read codes and FP16 scales directly from the mapped NPZ. This mode retains the control's one-SIMD-group-per-row trellis dot product. `--route-screen-half-simd` uses the same mapping and assigns four output rows to each 16-lane group, reusing input loads across rows. Its scheduling follows the native T=1 kernel. It uses the probe's FP32 inputs and scalar RTR1 lookup.

Run the three modes in separate app launches in the order staged, mapped, half-SIMD, half-SIMD, mapped, staged. Each run compares 768 sampled outputs with the existing fixture. Run order can affect the system file cache. Mapping, pipeline creation and fixed input preparation are outside the timed rounds. The first round includes first dispatch and mapped weight access. Warm p50 uses rounds 2–16. Peak process RSS is the process high-water mark and excludes some GPU and system memory costs.

| Device and mode | First round wall | Warm wall p50 | Warm staging p50 | Warm GPU p50 | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A16 staged | 62–152 ms | 52.8–125.0 ms | 10.5–56.7 ms | 40.0–56.5 ms | 500–523 MiB |
| A16 mapped | 178–1,332 ms | 37.6–39.2 ms | 0.018–0.031 ms | 34.1–34.3 ms | 54–75 MiB |
| A16 mapped half-SIMD | 182–222 ms | 34.5–34.6 ms | 0.025–0.026 ms | 30.3–30.4 ms | 54 MiB |
| M2 staged | 30–123 ms | 28.6–118.9 ms | 4.2–67.6 ms | 21.5–48.6 ms | 460 MiB |
| M2 mapped | 115–436 ms | 22.6–22.8 ms | 0.008–0.009 ms | 21.5 ms | 13 MiB |
| M2 mapped half-SIMD | 119–145 ms | 20.1–20.2 ms | 0.007–0.008 ms | 19.0 ms | 13 MiB |

Each range spans two independent launches per mode. Both devices reported nominal thermal state at the start and end, and all runs passed sampled parity with a worst absolute error of 5.37e-7. On A16, direct mapping removes the CPU weight copies and accounts for most of the warm wall-time and RSS improvement. The half-SIMD scheduling independently cuts warm mapped GPU time by about 11%. The large first-use penalty and the last staged run's lower staging time show that cache state matters.

The 34.6 ms A16 result still exceeds the 20 ms screen budget for six gate/up projections. The screen does not include a full decoder traversal or demonstrate higher chat throughput. The next kernel comparison can reuse the mapped routes and test the native grouped RTR1 lookup with its vectorized decode and FP16 input, keeping the measured first-use cost visible.

Raw runs are in the A16 log (`inference-bench-3.9gb/route-iteration-a16.log`) and the M2 log (`inference-bench-3.9gb/route-iteration-m2.jsonl`).

### A16 threadgroup tuning and bandwidth reference

The six-shard RTR1 screen targets 20 ms warm wall p50 on the iPhone 14 Pro using v3 weights and the existing sampled parity tolerance of 1e-4. Each round addresses 32,071,680 bytes of packed codes and FP16 scales across 110,592 output rows, or 1.60 GB/s of effective weight reads at the target. This byte count counts routed rows and omits lookup tables, activations, outputs and page traffic. It also does not account for reuse from hardware caches.

Run `--bandwidth-screen` in a fresh Release process. It reports GPU elapsed times for 16 copies between private 64 MiB Metal buffers. The effective bandwidth counts both the read and the write, using rounds 2–16. Dividing the routed weight bytes by that measured copy rate gives a bandwidth-only comparison point. Blit commands and indirect shader loads have different access patterns, and the estimate does not include trellis decode or dispatch overhead.

Compare `--route-screen-half-simd` (256 threads per group) with `--route-screen-half-simd-128` in fresh launches ordered 256, 128, 256, 128, 256. Both modes run the same kernel, with 16 lanes and four rows per subgroup. Compare warm wall p50 with both neighboring controls and check GPU time, sampled parity, peak RSS and thermal state. The app records Low Power Mode. No clock pinning or OS file-cache reset is applied. Context-length and generation stability require a full decoder.

The signed iPhone Release build completed all six launches. The bandwidth run measured **31.10 GB/s** median effective copy throughput (23.41–34.87 GB/s across 15 warm copies). At that copy rate, the routed payload represents **1.03 ms** of streaming transfer. The projection's roughly 30 ms of GPU work leaves a large gap to this reference, but the copy test cannot attribute the gap to arithmetic versus indirect memory access.

| Threads per group, in run order | First round wall | Warm wall p50 | Warm GPU p50 | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1,043.49 ms | 34.32 ms | 30.23 ms | 69.98 MiB |
| 128 | 1,023.05 ms | 33.59 ms | 30.25 ms | 78.48 MiB |
| 256 | 411.96 ms | 34.18 ms | 30.26 ms | 54.11 MiB |
| 128 | 336.32 ms | 34.57 ms | 30.30 ms | 53.98 MiB |
| 256 | 299.12 ms | 34.45 ms | 30.26 ms | 53.98 MiB |

Every launch passed all 768 sampled comparisons with the same maximum absolute error of 5.36e-7. Thermal state was nominal at each endpoint and Low Power Mode was off. The first 128-thread sample improved warm wall time by less than 3%, and the second failed to beat either adjacent control. GPU medians differed by at most 0.07 ms. Keep 256 as the benchmark default. Retain the 128-thread flag to reproduce this result. The next experiment should change the trellis lookup implementation while preserving the 256-thread launch and the same mapped-route fixture. Full-token bandwidth and stability still require a decoder benchmark.

Raw A16 runs (`inference-bench-3.9gb/route-tuning-a16.log`)

### Grouped RTR1 lookup at 256 threads

Compare `--route-screen-grouped-rtr1` against `--route-screen-half-simd` in fresh launches ordered control, grouped, control, grouped, control. The grouped variant reads 16 consecutive input elements per lane and expands a 14-bit trellis window into three weights per lookup. Its FP32 vector table occupies 256 KiB, versus 16 KiB for the scalar table. This probe uses the pinned runtime's grouped lookup structure and keeps input, weights and accumulation at the control's precision. Both variants use 256 threads per group, process four rows per half-SIMD group and read matching routes from the same NPZ mappings. Table preparation is outside the timed rounds.

| Mode, in run order | First round wall | Warm wall p50 | Warm GPU p50 | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 1,057.27 ms | 32.76 ms | 30.15 ms | 64.47 MiB |
| Grouped | 1,002.45 ms | 22.20 ms | 18.99 ms | 74.20 MiB |
| Scalar | 479.62 ms | 34.90 ms | 30.24 ms | 53.88 MiB |
| Grouped | 661.23 ms | 22.97 ms | 19.02 ms | 54.23 MiB |
| Scalar | 284.16 ms | 34.42 ms | 30.25 ms | 53.94 MiB |

The signed Release build completed all five iPhone launches with nominal thermal endpoints and Low Power Mode off. All runs passed the 768 sampled comparisons. Peak absolute error was 3.58e-7 for both grouped runs and 5.36e-7 for the scalar controls. Grouped lookup consistently reduced GPU time by about 37%. Keep it as the faster experimental path. Warm wall time remains above 20 ms and the first-round and RSS measurements vary by launch. These results establish a faster projection, with no new end-to-end iPhone tokens-per-second measurement. The app's full-generation path still uses the exported graph.

Raw A16 runs (`inference-bench-3.9gb/route-grouped-a16.log`)

## Swift recurrent decode prototype

The `--layer-decode` probe runs the 48-visit model schedule in Swift. It reuses embedding, per-layer and vocabulary-head math from the pinned Mac runtime through separate MLX functions. Swift owns the residual-block history, the three recurrent passes, cache slot assignment, and the feedback of each greedy token and updated state into the next step. Each layer function is imported only for the current visit and released before evaluation to bound weight residency. The fixture contains fixed-position signatures for the first few decode steps. This prototype starts from a Mac-prefilled FP16 cache and is separate from the chat UI.

From the workspace root, prepare a fresh output directory with the v3 checkpoint used by the matching suite.

```sh
checkpoint=hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
.venv/bin/python -B ios/Tools/export_layer_decode.py "$checkpoint" \
  inference-bench-3.9gb/iphone-first-48-suite.json /tmp/inkling-layer-decode --steps 3
xcrun devicectl device copy to --device "$device" --domain-type appDataContainer \
  --domain-identifier local.inkling.probe --source /tmp/inkling-layer-decode \
  --destination Documents/inkling-3.9gb-checkpoint/inkling-layer-decode
xcrun devicectl device process launch --device "$device" --terminate-existing \
  --console local.inkling.probe -- --layer-decode
```

The 22-token fixture expects seed ID 17 followed by IDs 10, 17 and 196977. The probe checks the token and all 48 KV and convolution cache slots after each step. Reported decode latency includes layer imports and evaluation. It excludes loading the initial cache and comparing the reference tensors.

The M2 replay of the exported layer functions reproduced all three native greedy IDs with zero maximum absolute error across the saved KV and convolution state. It took 6.12, 6.43 and 5.55 seconds per step, with about 269 MiB peak active MLX allocations. Reimporting and evaluating one layer at a time trades weight residency for repeated loading. The native Mac decoder's resident weights remain the speed reference. M2 layer replay (`inference-bench-3.9gb/layer-decode-m2.jsonl`)

The signed Swift build completed the same three-token run on the iPhone 14 Pro at **0.199 tok/s** overall and **0.202 tok/s** after the first token. All three greedy IDs matched and all 48 cache slots were visited each step. Peak active MLX memory was **216 MiB**, peak process RSS was **275 MiB**, and the largest physical-footprint sample at a token boundary was **206 MiB**. Thermal state was nominal at the sampled boundaries.

Cache tensor parity failed the 0.002 maximum absolute error tolerance. At all three steps, cache slot 1 was the first visit over tolerance for both KV and convolution state. Worst errors were **0.02734** in KV and **0.03782** in convolution state. The diagnostic probe records per-slot errors and completes all three steps even if numerical parity fails. The next correctness check should compare the hidden state after layer 0 and the inputs and outputs of layer 1 on both devices. This run establishes multi-step decode feasibility with bounded memory. Importing packed weights at each visit leaves it slower than the earlier iPhone chat observation of about 0.31 tok/s. On-device prefill, mapped production weights, and long-context stability are outside this probe.

Initial iPhone run (`inference-bench-3.9gb/iphone-a16-layer-decode-initial.log`) · Three-step iPhone diagnostic (`inference-bench-3.9gb/iphone-a16-layer-decode.log`)

## End-to-end Swift decode benchmark

`--full-decode` and `--full-decode-resident` start from empty caches. The phone validates the two fixture prompts through its tokenizer and runs prefill, the vocabulary head, and up to 31 subsequent cached steps for 32 generated tokens per prompt. Decode functions receive the absolute token position as a tensor, so all decode steps reuse the same exported signatures. Prefill signatures match the prompt lengths in the suite. FP16 KV overlays a 256-position buffer and each visit keeps its own convolution state. The resident mode holds the embedding, vocabulary head and physical layers 6–11 across visits and prompts. It loads the other physical layers on demand. These functions include packed weights from the pinned checkpoint. They do not yet use the direct NPZ mapping from the expert benchmark.

```sh
checkpoint=hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
.venv/bin/python -B ios/Tools/export_full_decode.py "$checkpoint" \
  inference-bench-3.9gb/iphone-first-48-suite.json /tmp/inkling-full-decode --tokens 32
xcrun devicectl device copy to --device "$device" --domain-type appDataContainer \
  --domain-identifier local.inkling.probe --source /tmp/inkling-full-decode \
  --destination Documents/inkling-3.9gb-checkpoint/inkling-full-decode --timeout 650
xcrun devicectl device process launch --device "$device" --terminate-existing \
  --console local.inkling.probe -- --full-decode-resident
```

Each emitted token includes its native Mac reference ID, elapsed time, cache-visit count and memory counters. The prompt summary reports TTFT, decode throughput excluding the first token, and the first greedy mismatch. The final `complete` event marks a completed run. Fixed prompt signatures and the explicit 256-position bound limit the benchmark to the exported suite.

The M2 replay of the resident layer configuration matched all 64 native greedy IDs and executed all 48 visits per step. It measured 6.27 and 6.80 seconds to the first token, and 0.20 and 0.29 tok/s after prefill for the 22- and 30-token prompts respectively. This replay loaded the exported functions in Python while the iPhone uses the Swift implementation. Peak active MLX allocations were about 1.30 GiB. M2 full-model replay (`inference-bench-3.9gb/full-decode-m2.jsonl`)

The first iPhone fixture transfer reset with the device out of storage. Removing the retired fixed-step and three-step benchmark fixtures reclaimed about 7.5 GiB. The active chat graph and checkpoint stayed in the app container. The missing end-to-end fixture files were then transferred individually.

The first on-device attempt produced 14 matching tokens before the app exited and MLX logged a Metal timeout. The rerun used the same installed build, fixture, and runtime settings, including the default Metal command-buffer limits. Interrupted A16 run (`inference-bench-3.9gb/iphone-a16-full-decode-resident.log`)

The unchanged rerun completed on the iPhone 14 Pro. Both prompts matched all 32 native Mac token IDs and completed 48 visits per output. Decode throughput excludes the first token.

| Prompt tokens | Time to first token | Decode throughput | Median decode latency |
| ---: | ---: | ---: | ---: |
| 22 | 4.72 s | 0.439 tok/s | 2.27 s |
| 30 | 3.06 s | 0.440 tok/s | 2.27 s |

The run peaked at 1.30 GiB of active MLX allocations and 1.30 GiB process RSS, with a largest sampled physical footprint of 1.26 GiB. Sampled thermal state remained nominal. The run verifies greedy token parity for these two prompts. Cache tensor parity was not measured in this benchmark. Completed A16 run (`inference-bench-3.9gb/iphone-a16-full-decode-resident-rerun.log`)

### Run the same Swift benchmark on an Apple silicon Mac

Build the probe for macOS and invoke its command-line entry point. Pass the same checkpoint tokenizer and exported fixture directory that were used for the phone. The benchmark calls the same Swift decode function on both platforms. Use `--resident` to match the completed iPhone configuration.

```sh
xcodebuild -project ios/InklingProbe.xcodeproj -scheme InklingProbe \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/inkling-swift-build SDKROOT=macosx CODE_SIGNING_ALLOWED=NO
/tmp/inkling-swift-build/Build/Products/Release/InklingProbe.app/Contents/MacOS/InklingProbe \
  "$checkpoint" /tmp/inkling-full-decode --resident
```

The M2 MacBook Air completed both prompts through the Swift CLI and matched all 64 greedy IDs from the native Mac reference and the iPhone Swift run. Every token executed all 48 visits. Each device ran the resident configuration with the same 3.9 GB checkpoint and fixture.

| Prompt tokens | M2 TTFT | A16 TTFT | M2 decode | A16 decode |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 9.08 s | 4.72 s | 0.268 tok/s | 0.439 tok/s |
| 30 | 4.12 s | 3.06 s | 0.276 tok/s | 0.440 tok/s |

The M2 run peaked at 1.30 GiB of active MLX allocations. Sampled thermal state was nominal on both devices. The comparison covers one completed run per device, with independent file-cache and memory-pressure conditions. The Swift configuration continues to load nonresident layer functions at each visit. Mac Swift raw log (`inference-bench-3.9gb/full-decode-swift-m2.log`) · iPhone Swift raw log (`inference-bench-3.9gb/iphone-a16-full-decode-resident-rerun.log`)

### Mac with every layer retained

Pass `--all-resident` to the Mac CLI with the same checkpoint and fixture. The first prompt imports 26 functions, covering embedding, all 24 physical layers, and the output head. The remaining 63 outputs, including prefill for the second prompt, logged zero imports. The same prompts were then rerun through native LAMQ on the M2, without running the two benchmarks concurrently. All 64 greedy IDs matched.

| Mac configuration | 22-token TTFT | 22-token decode | 30-token TTFT | 30-token decode | Peak active MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swift, six retained layers | 9.08 s | 0.268 tok/s | 4.12 s | 0.276 tok/s | 1.30 GiB |
| Swift, all 24 layers retained | 5.59 s | 7.26 tok/s | 0.59 s | 11.30 tok/s | 3.66 GiB |
| Native LAMQ, resident weights | 1.11 s | 17.92 tok/s | 0.50 s | 18.06 tok/s | 4.17 GiB |

Decode throughput excludes the first token and includes all 31 subsequent steps. All-resident Swift took 1.13 s for its first cached decode step. Its median cached latency was 109 ms on the first prompt and 72 ms on the second, versus about 55 ms in both native runs. The six-layer row is the earlier Mac run with independent file-cache and system memory conditions. The all-resident Swift path synchronizes after each of the 48 visits. Native LAMQ schedules work asynchronously across visits. This difference is a candidate for the remaining gap and has not been isolated by this test. All-resident Swift peaked at 3.78 GiB sampled process footprint on the Mac, and was not run on the iPhone.

All-resident Swift raw log (`inference-bench-3.9gb/full-decode-swift-m2-all-resident.log`) · Matched native LAMQ raw log (`inference-bench-3.9gb/full-decode-lamq-m2-native.log`)

### Async Swift scheduling reached native Mac decode speed

The Mac CLI now accepts `--all-resident-async`. Swift calls MLX `asyncEval` on the hidden state every four visits through visit 44. It synchronizes once at the token boundary with the argmax and every updated KV and convolution tensor. The exported functions, greedy selection, 48-visit schedule, cache layout and 24-layer residency are identical to `--all-resident`. All work is dispatched from the existing `InklingMLXWorker` thread.

Run the Mac executable built above with the same checkpoint and fixture, in this order, with one process at a time.

```sh
probe=/tmp/inkling-swift-build/Build/Products/Release/InklingProbe.app/Contents/MacOS/InklingProbe
"$probe" "$checkpoint" /tmp/inkling-full-decode --all-resident
"$probe" "$checkpoint" /tmp/inkling-full-decode --all-resident-async
"$probe" "$checkpoint" /tmp/inkling-full-decode --all-resident
```

| M2 configuration | 22-token TTFT | 22-token decode | 30-token TTFT | 30-token decode | Peak active MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swift sync before | 5.62 s | 6.69 tok/s | 0.62 s | 11.49 tok/s | 3.66 GiB |
| Swift async | 4.21 s | 11.71 tok/s | 0.50 s | 18.00 tok/s | 4.21 GiB |
| Swift sync after | 4.82 s | 6.95 tok/s | 0.60 s | 10.57 tok/s | 3.66 GiB |
| Native LAMQ, previous matched control | 1.11 s | 17.92 tok/s | 0.50 s | 18.06 tok/s | 4.17 GiB |

Each decode average covers 31 steps after the first token. The async run's final 16 steps sustained 18.08 and 17.98 tok/s on the two prompts. Median decode latency was 55.38 and 55.52 ms. All three Swift runs completed, matched all 64 native greedy IDs, and executed 48 visits per token. Each imported 26 functions during the first prefill and zero from then on. Thermal state was nominal at sampled boundaries. The async run peaked at 3.91 GiB sampled process footprint. Cache tensor parity was not checked by this benchmark.

Sync before (`inference-bench-3.9gb/full-decode-swift-m2-sync-before.log`) · Async (`inference-bench-3.9gb/full-decode-swift-m2-async.log`) · Sync after (`inference-bench-3.9gb/full-decode-swift-m2-sync-after.log`)

### Bounded async scheduling on iPhone 14 Pro

The signed Release probe accepts `--full-decode-resident-async`. Both phone modes retain the embedding, head, and physical layers 6–11. The async mode submits two visits at a time, including their cache outputs, and prepares the next pair while the first is in flight. It waits on the previous pair before submitting again. At most four transient layer imports are held across the two pairs. The token boundary waits for the argmax and every cache tensor.

Run each mode with the fixture staged under `Documents/inkling-3.9gb-checkpoint/inkling-full-decode`. Pass `--fingerprint-cache` on both runs to log a SHA-256 digest over the logical bytes of all KV and convolution tensors in slot order. The digest is computed after the elapsed time and counters for each token are recorded. Compare `cache_sha256` by prompt and token index for byte-level parity with the on-device synchronous control.

```sh
for mode in full-decode-resident-async full-decode-resident; do
  xcrun devicectl device process launch --device "$device" --terminate-existing \
    --console local.inkling.probe -- "--$mode" --fingerprint-cache
done
```

End each console session after the `complete` event to advance to the next run.

Both Release targets built successfully. The first synchronous attempt stopped producing tokens after five outputs when the app may have left the foreground. The foreground rerun completed async first and sync second in fresh processes on the same signed build.

| iPhone 14 Pro configuration | 22-token TTFT | 22-token decode | 30-token TTFT | 30-token decode | Peak active MLX | Peak RSS | Peak sampled footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Six layers, async | 4.25 s | 0.424 tok/s | 2.73 s | 0.433 tok/s | 1.76 GiB | 1.67 GiB | 1.87 GiB |
| Six layers, sync | 4.36 s | 0.427 tok/s | 3.05 s | 0.425 tok/s | 1.30 GiB | 1.39 GiB | 1.26 GiB |

Both runs matched all 64 native greedy IDs, executed 48 visits per token, and imported 30 transient functions per cached token. All 64 paired cache digests matched, establishing byte-level parity between the two iPhone schedules. Native cache tensor error was outside this comparison. Async throughput was within 2.1% of the synchronous control per prompt and its peak sampled footprint was 48% higher. The phone went from nominal to fair thermal state during async and remained fair during the synchronous control.

The M2 result of 18.0 tok/s kept all 24 layers resident and logged zero imports after the first prefill. The earlier Swift M2 six-layer synchronous run measured 0.268–0.276 tok/s on these prompts.

Async rerun (`inference-bench-3.9gb/iphone-a16-bounded-async-rerun.log`) · Sync rerun (`inference-bench-3.9gb/iphone-a16-bounded-sync-after-rerun.log`) · Interrupted sync attempt (`inference-bench-3.9gb/iphone-a16-bounded-sync-before.log`)

### All-layer async comparison on iPhone

The signed Release probe exposes the same shared `allAsync` mode that produced the 18.0 tok/s warm M2 result. It retains all 24 physical layer functions plus embedding and head, submits work every four visits, and uses the same staged two-prompt fixture. The launch command leaves cache fingerprinting disabled, as in the M2 run.

```sh
xcrun devicectl device process launch --device "$device" --terminate-existing \
  --console local.inkling.probe -- --full-decode-all-resident-async
```

Two launches on the iPhone 14 Pro logged the 24-layer configuration and then ended with SIGKILL before the first token. The first launch started at serious thermal state and the retry at fair. No corresponding diagnostic report identified the termination reason. These attempts yielded no throughput sample. The same fixture and signed build can be tried on a second phone. First attempt (`inference-bench-3.9gb/iphone-a16-full-decode-all-resident-async.log`) · Retry (`inference-bench-3.9gb/iphone-a16-full-decode-all-resident-async-retry.log`)

The M2 also ran the iPhone's six-layer async mode with the same prompts and cache fingerprinting enabled. It measured 0.215 and 0.282 tok/s, versus 0.424 and 0.433 tok/s on the iPhone 14 Pro. The M2 and iPhone matched all 64 greedy IDs and all 64 paired KV and convolution cache fingerprints. The runs had independent OS file-cache and memory-pressure conditions. Matched M2 run (`inference-bench-3.9gb/full-decode-swift-m2-bounded-async.log`)

#### Increased-memory signing experiment

The original experiment passed a temporary entitlement file per build, as shown below. The iPhone target now requests Increased Memory Limit by default through its checked-in entitlements file. Verify that the development provisioning profile and signed app both include the entitlement before installing.

```sh
entitlements=/tmp/inkling-increased-memory.entitlements
plutil -create xml1 "$entitlements"
plutil -insert 'com\.apple\.developer\.kernel\.increased-memory-limit' -bool true "$entitlements"
xcodebuild -project ios/InklingProbe.xcodeproj -scheme InklingProbe \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/inkling-full-runtime-release -disableAutomaticPackageResolution \
  -allowProvisioningUpdates "CODE_SIGN_ENTITLEMENTS=$entitlements" \
  CONFIGURATION_BUILD_DIR=/tmp/inkling-entitled-release
codesign -d --entitlements - /tmp/inkling-entitled-release/InklingProbe.app
xcrun devicectl device install app --device "$device" /tmp/inkling-entitled-release/InklingProbe.app
```

The signed app and embedded provisioning profile both contained `com.apple.developer.kernel.increased-memory-limit = true`. The iPhone 14 Pro accepted the app. [Apple's entitlement reference](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)

All three entitled runs completed the 22-token prompt with all 24 layers retained and matched all 32 native greedy IDs. Each imported 26 functions during first prefill and zero during cached decode. All three ended with SIGKILL before the first output of the 30-token prompt. Cache fingerprinting was disabled for these runs, matching the original Mac timing runs.

| All-layer async configuration | 22-token TTFT | All 31 cached steps | Final 16 cached steps | Peak sampled footprint |
| --- | ---: | ---: | ---: | ---: |
| M2 | 4.21 s | 11.71 tok/s | 18.08 tok/s | 3.91 GiB across both prompts |
| iPhone 14 Pro, entitled | 4.27 s | 4.04 tok/s | 10.93 tok/s | 3.81 GiB |
| iPhone 14 Pro, entitled retry | 4.90 s | 2.76 tok/s | 10.69 tok/s | 3.80 GiB |
| iPhone 14 Pro, final entitled run | 4.06 s | 6.27 tok/s | 10.97 tok/s | 3.81 GiB |

The iPhone's early cached steps accounted for the difference between whole-prompt and final-16 rates. Its sampled thermal state was nominal throughout the entitled attempts. First entitled attempt (`inference-bench-3.9gb/iphone-a16-full-decode-all-resident-async-entitled.log`) · Entitled retry (`inference-bench-3.9gb/iphone-a16-full-decode-all-resident-async-entitled-retry.log`) · Final entitled run (`inference-bench-3.9gb/iphone-a16-full-decode-all-resident-async-entitled-final.log`)

#### Interactive 24-layer chat

**Chat with Inkling 2 Mini** uses the staged full-decode layer graphs. The interactive runtime retains all 24 layers, embedding, and head, and uses the benchmark's four-visit async scheduling. It processes typed prompts one token at a time to use the exported single-position signature and limit prefill memory. Chat supports a 256-token context with a 64-token reply reserve. It reuses the live KV and convolution state while its input is an exact continuation of the cached prefix.

The signed increased-memory build ran the chat backend smoke test on the iPhone 14 Pro. All four generated IDs matched the native reference for “What is 2+2?” and decoded to “2+2 =”. The app was then relaunched directly into Inkling Chat for interactive testing. The smoke test covered the first four generated tokens. Chat smoke log (`inference-bench-3.9gb/iphone-a16-async-chat-smoke.log`)

Opening Chat now warms the 24-layer decoder before enabling Send. The warm-up imports the layer graphs and caches the standard prompt through the user-content marker. The iPhone 14 Pro smoke test took 8.07 s to warm up, then 0.90 s to the first token for “What is 2+2?”. The first four token IDs still matched. This measurement covers one launch and one short prompt. Warm-up smoke log (`inference-bench-3.9gb/iphone-a16-async-chat-warm-smoke.log`)

The 256-token context smoke test with a 32-token reply reserve completed on the iPhone 14 Pro with 213 formatted prompt tokens and 256 total. The later 64-token reserve build limits the formatted prompt to 192 tokens. That build installed and reached the first short-prompt token in 0.90 s after warm-up. The diagnostic connection dropped before reporting the long-context result for that build. 256-token run (`inference-bench-3.9gb/iphone-a16-async-chat-context-retry.log`) · 64-token reserve run (`inference-bench-3.9gb/iphone-a16-async-chat-reserve-retry.log`)

#### Mac async submission interval sweep

The M2 ran two fresh processes per interval in order 4, 2, 8, 8, 2, 4 with the same all-resident fixture and unchanged 64 MiB MLX allocator cache limit. Use `--all-resident-async --async-layer-interval=N` on the Mac CLI. All six runs completed both prompts, matched all 64 native greedy IDs per run, and reported nominal thermal state. Across individual runs, system-wide swap writes ranged from 0 to 7.5 MiB. An earlier heavily paging attempt was stopped before completing the first prompt.

| Visits per submission | Final-16 tok/s across prompts and runs | Sampled process footprint peaks (GiB) | macOS peak footprint (GiB) |
| --- | ---: | ---: | ---: |
| 2 | 17.67–17.80 | 3.886–3.930 | 4.436–4.437 |
| 4 | 17.79–17.89 | 3.936–3.936 | 4.384–4.426 |
| 8 | 17.79–17.85 | 3.897–3.910 | 4.387–4.429 |

Eight visits is the iPhone trial default. Its sampled peaks were 27–40 MiB below the four-visit runs, while the operating-system peaks overlapped. The sweep does not establish a repeatable reduction in peak footprint. Raw Mac sweep logs (`inference-bench-3.9gb/async-interval-mac/rerun.6zuPIb/`)

The candidate also has **Chat in Recording Mode**, which retains six recurrent layers and uses synchronous scheduling. It limits chat to 64 context tokens and eight generated tokens per reply, while using the same 256-position KV buffers. Open it from the launcher or pass `--open-recording-chat`. The historical six-layer benchmark measured about 0.43 tok/s and 1.26 GiB sampled footprint. Those figures came from the benchmark path. Interactive recording remains to be tested on the phone.

#### Serial prompt and short-context memory sweep

On the M2, nine fresh all-resident processes ran both prompts using the eight-visit submission interval and serial prompt ingestion. Each run matched all 64 native greedy token IDs. The system wrote 206–716 MiB of swap per run, so small throughput differences are difficult to attribute to the tested settings. Mac logs (`inference-bench-3.9gb/chat-memory-mac/`)

- With the 256-position fixture and a 64 MiB allocator limit, computing the vocabulary head only on the final prompt token reduced first-prompt time to first token from 7.04–7.26 s to 6.57–6.65 s. Warm second-prompt ranges overlapped. Chat still commits all 48 visited cache states for every prompt token.
- At 256 positions, the 16, 32, and 64 MiB limits produced about 3,882 MiB macOS peak footprint in their initial final-head runs. Sampled allocator cache stayed around 10–11 MiB in those runs. The app uses the 64 MiB default. The Mac CLI and iPhone full-decode diagnostics accept `--cache-mib=N`.
- The 64-position fixture uses 12 MiB of raw FP16 KV instead of 48 MiB. Peak active MLX allocation fell by about 36 MiB at each matched allocator limit. At 64 MiB, macOS peak footprint was 3,844 MiB versus 3,882–3,898 MiB for the 256-position fixture and final-head prefill. The short-context run at that limit sustained 14.20–17.04 tok/s over the final 16 tokens.

Export the additional graphs with `export_full_decode.py ... --kv-capacity 64` and stage them in `Documents/inkling-3.9gb-checkpoint/inkling-full-decode-64`. **Chat with Short Context** retains all 24 layers and budgets 64 total tokens with a 16-token reply reserve. Regular chat and the six-layer Recording Mode still use the original fixture. The short fixture can be profiled with `--full-decode-short` or smoke tested with `--chat-short-smoke-test` on iPhone.

#### 1K-context chat

The optional **Chat with 1K Context** mode uses 1,024 positions with a 128-token reply reserve. It retains the layer graphs and ingests prompts one token at a time. Its fixed FP16 KV buffers hold 192 MiB of raw data across the 48 visits, compared with 48 MiB at 256 positions. Allocator and working memory add to those figures.

Export a fixture with the pinned checkpoint and a matching suite. Include a prompt longer than 256 tokens to exercise the expanded window. Serial-only exports omit the batched signatures, and the benchmark automatically uses serial prefill for them.

```sh
.venv/bin/python -B ios/Tools/export_full_decode.py "$checkpoint" "$suite" \
  /tmp/inkling-full-decode-1024 --kv-capacity 1024 --serial-only --tokens 4
```

Stage the resulting directory at `Documents/inkling-3.9gb-checkpoint/inkling-full-decode-1024`. Open **Chat with 1K Context** in the app, or pass `--full-decode-1024` to benchmark the exported prompts on iPhone. To run the same benchmark on Mac, pass the fixture directory to the Swift CLI with `--all-resident-async`.
