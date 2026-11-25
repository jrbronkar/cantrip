# Inkling Probe for iPhone

Inkling Probe compares a Swift CPU implementation of Inkling 2 Mini with reference outputs from the Mac runtime. Import a fixture from Files to run the checks offline. The probe covers packed embeddings and layer 0 attention for one token, including the five HT3 projections, grouped-query attention with a sink and relative-position bias, the gated output projection, and the post-attention convolution.

## Run the probe

From the workspace root, export a fixture from the existing checkpoint on a Mac with Metal access.

```sh
.venv/bin/python -B ios/Tools/export_fixture.py --layer-zero \
  hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e \
  fixtures/inkling-layer-zero.json
```

Open `ios/InklingProbe.xcodeproj` in Xcode, select a signing team and an iPhone with iOS 17 or later, then run the InklingProbe target. Transfer the fixture to Files, open the app, and tap **Import parity fixture**. The roughly 15 MB fixture contains reference outputs for eight sampled embeddings and eight points in the first token's attention pass. The app reports the maximum absolute error for each check. The pass thresholds are 0.0001 for embeddings and 0.002 for attention. Omit `--layer-zero` to generate the smaller embedding-only fixture.

The macOS run passed all checks. The largest errors were 0.00000954 for embeddings and 0.00183 across the attention checks. The post-attention convolution error was 0.000544. The unsigned iPhone build succeeded. The original probe also passed all 16 checks on a physical iPhone.

## Automated preflight

Run this from the workspace root after exporting the layer 0 fixture. Use `DEVELOPER_DIR` to point to a full Xcode installation if your default is Command Line Tools.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer sh ios/Tools/preflight.sh \
  hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e \
  fixtures/inkling-layer-zero.json
```

The preflight uses the cached checkpoint tokenizer to check conversation formatting and token budgets. It also checks transcript save and reload, KV-cache prefix matching, all 16 model outputs, embedding-only fixture compatibility, and rejection of invalid or oversized fixtures. It finishes by building the unsigned iPhone Release target and checking its version and bundled MIT notice. The preflight runs on the Mac.

## Test the 3.9 GB checkpoint on iPhone

Run Inkling Probe on the phone from Xcode. Copy the `fixtures/inkling-3.9gb-expert` folder into **On My iPhone** in Files, open the app, tap **Test 3.9 GB checkpoint expert**, and select that folder. For a clean process memory reading, run this first after force-quitting and reopening the app.

The folder contains the checkpoint's original layer 0 NPZ and a small Mac reference. The app maps the archive and reads the selected expert's packed gate/up weights directly. It reports parity against the Mac MLX projection, elapsed time, and the process's lifetime peak resident memory. The test exercises one CPU projection over 2,304 outputs. Full-checkpoint mapping, generation, and device memory requirements still need separate measurements.

The iPhone 14 Pro passed with a maximum absolute error of `2.384186e-6`. It reported a 132.3 MB mapped archive, 0.203 seconds elapsed, and 120.8 MB process peak resident memory. The elapsed time includes archive mapping and the CPU projection.

Tap **Compare CPU and Metal expert** when the checkpoint and expert reference are in the app's private storage. The Files import can also run the comparison using the two-file expert folder. Both paths use the same CPU input rotation. Metal reads packed RTR1 words for the selected expert and reduces each output row on the GPU. The app shows end-to-end elapsed time for each backend, including Metal setup and buffer copies, and GPU command time when available. The displayed peak resident memory is for the process across both runs.

The iPhone 14 Pro passed both paths. Maximum absolute error was `2.384186e-6` on CPU and `4.172325e-7` on Metal. CPU elapsed time was 0.234 seconds, Metal elapsed time was 0.215 seconds, and the GPU command took 0.003 seconds. The combined run reported 115 MB peak resident memory.

To recreate the folder from the same cached checkpoint, run from the workspace root on a Mac with Metal access.

```sh
checkpoint=hf/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
folder=fixtures/inkling-3.9gb-expert
mkdir -p "$folder"
cp -cL "$checkpoint/layer_00_packed.npz" "$folder/"
.venv/bin/python -B ios/Tools/export_expert_reference.py "$checkpoint" "$folder/inkling-expert-reference.json"
```

Pass `"$folder/inkling-expert-reference.json"` as the optional third argument to the Mac preflight to check expert parity there too.

## Scan the full checkpoint

Use **Scan checkpoint copied from Mac** if the checkpoint is already in the app's private storage. To select a local copy in Files, tap **Scan full 3.9 GB checkpoint** and choose the folder that contains all 24 layer archives, the embedding and head archives, the lookup tables, and the steering weights.

The probe maps all 28 weight files read-only at once and samples one byte from every OS page. It reports the mapped bytes, page count, map time, read time, and process lifetime peak resident memory. Launch the app fresh for a clean peak measurement. The checksum reflects the sampled bytes and can vary with page size. This test measures file access. Decoding adds kernel work, activations, and cache memory.

On the iPhone 14 Pro, the scan mapped all 28 weight files (3,822,202,944 bytes, displayed as 3.56 GB) in 0.009 seconds. It sampled all 233,305 pages in 5.760 seconds and reported 2.33 GB process peak resident memory. The page-sample checksum, `6e53e13c4f577cc8`, matched the Mac scan.

## Predict one offline token

The full-model probe imports a fixed-shape MLX graph exported from the pinned 3.9 GB checkpoint. It runs the existing packed forward pass through the model's output head and compares the greedy token with the Mac reference. The five input IDs encode “The capital of France is” and the expected output is token `152717` (“ Paris”).

Build the iPhone app with the Xcode Metal Toolchain installed. On a Mac with MLX and Metal access, run the exporter with the checkpoint directory and an optional graph output path. Keep the generated MLX graph local because it contains packed model weights.

```sh
.venv/bin/python -B ios/Tools/export_next_token_reference.py "$checkpoint" \
  fixtures/inkling-next-token-reference.json --graph /tmp/inkling-next-token.mlxfn
```

Place the graph and reference JSON in the app's staged checkpoint folder. Tap **Predict one offline token**, or launch the app with `--predict-one-token` to start the probe and log its result to the console. The graph has a fixed five-token input shape and contains about 3.79 GB of packed constants, separate from the NPZ files used by the earlier probes. The corrected export replayed token `152717` on the Mac with 3.80 GB peak active MLX allocations.

The iPhone 14 Pro matched token `152717` in 2.919 seconds, including graph import, with 1,799,638,756 bytes of peak active MLX allocations. Two earlier attempts were terminated after graph import when the graph was retained during evaluation. The successful run released the imported function after creating the lazy result so MLX could free weights as their last consumers finished.

## Chat with Inkling 2 Mini

Tap **Chat with Inkling 2 Mini** on the app's main screen, or use the development launch flag `--open-chat`. Chat uses the staged full-decode graphs and local tokenizer. Follow the [iPhone profiling setup](INFERENCE-PROFILING.md#interactive-24-layer-chat) to stage the fixture and sign the test build.

Chat warms the 24-layer async decoder when it opens. The 256-token context reserves at least 64 tokens for the reply. Chat drops older complete turns when needed, streams generated text, and saves completed turns and their timing locally.

## Runtime and distribution

- The probe uses Swift and SwiftUI. The full-model runner also uses MLX Swift. The packed embedding and attention algorithms are adapted from LookAtMyQuant at revision `9dc5cbf7adb08cc41b5ba45c6313556e16f30bdc`. The app bundles and displays the full upstream MIT notice. [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant/tree/9dc5cbf7adb08cc41b5ba45c6313556e16f30bdc)
- Thinking Machines Lab publishes Inkling 2 Mini under Apache 2.0. The locally cached model card identifies this checkpoint as ShinMegamiBoson’s SQYFR v3 quantization of those weights, also labeled Apache 2.0. The app ships without checkpoint data or fixtures. Keep generated fixtures local during development. If distributing the checkpoint or fixtures, preserve applicable upstream notices, include the Apache 2.0 license, and identify modifications including quantization or extraction. [Original model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Quantized checkpoint](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB)
- For full generation, evaluate the MIT-licensed MLX Swift as the tensor backend and reuse the MIT LookAtMyQuant implementation as the reference for its custom kernels. The existing 3.85 GB checkpoint uses NPZ and safetensors artifacts with custom packing. Preserve that packing in the mobile loader. Stock GGUF support is a separate route using different weights. [MLX Swift](https://github.com/ml-explore/mlx-swift) · [llama.cpp Inkling GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)

Use the existing packed Mac runtime as the reference when bringing up an iPhone backend for this checkpoint. Keep the CPU probe as a parity test for the loader and kernels. Before adding chat, test a complete greedy decode on physical hardware with short context, serial generation, and no drafter. Record first-token latency, peak process memory, and sustained decode before deciding which iPhones to support. The Mac handoff recorded 4.19 GiB peak active MLX memory for short serial runs. Some iPhones support the iOS increased-memory-limit entitlement. Measure how much memory the app can use on each test phone. [Apple’s entitlement documentation](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)

## Conversation context

`ConversationContext.swift` saves each thread in Application Support using atomic writes and iOS file protection. To prepare a reply, it formats the prompt with Inkling’s text-only template and counts tokens through a tokenizer supplied by the caller. If the prompt exceeds the available budget after reserving space for the reply, it drops the oldest complete turns until the prompt fits. It reports how many turns it dropped and keeps the full transcript on disk. The generation loop can use the prepared prompt to check whether it starts with exactly the tokens in the live KV cache. The app still needs to connect an on-device tokenizer, the generation loop, and a chat screen.

## Physical iPhone test plan

### Run the parity probe now

1. Connect the iPhone with a USB data cable and unlock it. Confirm **Trust This Computer** if prompted. In Xcode, verify that the phone appears under **Window → Devices and Simulators** or **Device Hub**, depending on the Xcode version. The project targets iOS 17 and later. Use an Xcode version that supports the phone's installed iOS.
2. Add your Apple Account in **Xcode → Settings → Accounts**. Open `InklingProbe.xcodeproj` and select the **InklingProbe** target. In **Signing & Capabilities**, enable **Automatically manage signing**, select your team, and change `local.inkling.probe` to a unique bundle identifier if Xcode requests it. You can use a personal team to install the app directly. [Apple's physical-device build instructions](https://developer.apple.com/documentation/Xcode/running-your-app-on-simulated-or-physical-devices)
3. On iPhone, enable **Settings → Privacy & Security → Developer Mode** after pairing. Restart and confirm when prompted. Select the phone as the Xcode run destination and press **⌘R**. [Apple's Developer Mode instructions](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device)
4. Generate the layer 0 fixture using the command above. AirDrop it to the phone and save it under **On My iPhone** in Files. Turn on Airplane Mode and verify Wi-Fi is off, then import the fixture in Inkling Probe. Expect eight embedding checks and eight attention checks to pass, including after force-quitting and reopening the app. Record the iPhone model and iOS version. For each failed check, record the stage and its error. This test uses only the fixture.

### Test chat once the app can generate replies

- Start with a short single-turn offline run on the highest-memory supported phone available, then repeat on the intended 8 GB target if available. Use a release build with serial decode, 512 total context tokens, and a 64-token reply cap for the first run. Record the checkpoint revision, model load time, time to first visible text, sustained tokens per second, peak process memory, and thermal state. Profile a separate run in Instruments. Then run with the debugger detached to see whether iOS terminates the app under memory pressure.
- Stay offline. In one thread, say “My orchard key is called Cobalt Finch.” Ask an unrelated question, then ask “What was my orchard key called?” Force-quit, reopen the thread, and ask again. Expect “Cobalt Finch” both times. Open a new thread and verify that its prepared prompt contains no messages from the first.
- Run with a deliberately small context budget. Verify that the UI shows when older turns are omitted, the saved transcript still contains them, and the assistant can recall a detail from the newest retained turn. Compare cached and rebuilt runs with greedy decoding and the same token IDs. Check cancellation and retry, and verify that a failed or cancelled reply does not leave half a completed turn in the transcript.
