# Inkling 2 Mini

A local Metal inference runner for macOS and a SwiftUI inference app for iPhone. Both use the pinned 3.9 GB packed Inkling 2 Mini checkpoint.

## Run on an Apple silicon Mac

With Python 3.12 and `uv` available, run these commands from the repository root.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --target .runtime-inkling-3.9gb --no-deps \
  'lookatmyquant @ git+https://github.com/ShinMegamiBoson/lookatmyquant.git@9dc5cbf7adb08cc41b5ba45c6313556e16f30bdc'
HF_HOME="$PWD/hf" .venv/bin/hf download \
  shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB \
  --revision 54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e
./run.sh
```

## Run on iPhone

Open `ios/InklingProbe.xcodeproj` in Xcode, select your development team under **Signing & Capabilities**, and build the **InklingProbe** scheme for the device. The iPhone target requests the Increased Memory Limit entitlement. Use a development provisioning profile that includes it. Stage the checkpoint manifest and tokenizer files in the app’s Documents folder, then export and copy the full-decode graphs using the [iPhone setup and profiling instructions](ios/INFERENCE-PROFILING.md#end-to-end-swift-decode-benchmark). [Probe and fixture instructions](ios/README.md)

The repository tracks source, build definitions, and small reference inputs. Downloaded weights, exported graphs, signing files, and raw profiling captures are local artifacts. The [performance handoff](INKLING-HANDOFF.md) and [iPhone profiling notes](ios/INFERENCE-PROFILING.md) summarize those captures.

## License and credits

Cantrip is published under the MIT license. The adapted packed embedding, attention, and expert projection code in [EmbeddingProbe.swift](ios/Sources/EmbeddingProbe.swift) comes from [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) by Shin Megami Boson and retains its [MIT copyright and license notice](ios/Notices/LookAtMyQuant-MIT.txt). The Python runner and exporter tools also use LookAtMyQuant as an external dependency.

The setup downloads Shin Megami Boson’s [3.9 GB quantization](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB) of Thinking Machines Lab’s [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). Obtain the model separately and follow its upstream Apache 2.0 terms. This repository contains code and small test references. Model weights, generated graphs, and signing material are excluded.
