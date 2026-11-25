#!/bin/sh
set -eu
cd "$(dirname "$0")"
export HF_HOME="$PWD/hf"
export HF_HUB_OFFLINE=1
if [ "${1:-}" = "--3.9gb" ]; then
    shift
fi
runtime="$PWD/.runtime-inkling-3.9gb"
if [ ! -d "$runtime/runtime_mlx" ]; then
    printf '%s\n' 'The 3.9 GB Inkling runtime is missing. See INKLING-HANDOFF.md.' >&2
    exit 1
fi
model="$HF_HOME/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-3.9GB/snapshots/54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e"
export PYTHONPATH="$runtime${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python inkling_answer.py "$model" "$@"
