# Cantrip

An experimental Inkling 2 Mini Metal inference runner for macOS and a SwiftUI inference app for iPhone. Both use the pinned 3.9 GB packed Inkling 2 Mini checkpoint.

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

Run `./run.sh "What is 2+2?"` for a one-shot answer or `./run.sh` for interactive prompts. Each Mac prompt starts fresh. Use `./run.sh --timings --max-tokens 64` to inspect startup and decode timings.

## Run on iPhone

Open `ios/InklingProbe.xcodeproj` in Xcode, select your development team under **Signing & Capabilities**, and build the **InklingProbe** scheme for the device. The iPhone target requests the Increased Memory Limit entitlement. Use a development provisioning profile that includes it. Follow the [iPhone setup and probe instructions](ios/README.md#chat-with-inkling-2-mini) to stage the tokenizer and exported layer graphs.

## Benchmarks

`profile_inkling.py` emits Mac timing and paging measurements as JSONL. `bench_decode.py` compares experimental Mac decode settings, and `swap_top.py` shows system-wide swap traffic. The [profiling notes](ios/INFERENCE-PROFILING.md) record Mac and iPhone results and commands. The iPhone chat path uses exported MLX layer graphs. Expert weight streaming is a separate experiment.

## License and credits

Cantrip is published under the [MIT license](LICENSE). The adapted packed embedding, attention, and expert projection code in [EmbeddingProbe.swift](ios/Sources/EmbeddingProbe.swift) comes from [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) by Shin Megami Boson and retains its [MIT copyright and license notice](ios/Notices/LookAtMyQuant-MIT.txt). The Python runner and exporter tools also use LookAtMyQuant as an external dependency.

The setup downloads Shin Megami Boson’s [3.9 GB quantization](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB) of Thinking Machines Lab’s [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). Obtain the model separately and follow its upstream Apache 2.0 terms. This repository contains code and small test references. Model weights, generated graphs, and signing material are excluded.
