#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MODEL_SNAPSHOT TRAIN_JSONL OUTPUT_ROOT" >&2
    exit 2
fi

model_snapshot=$(realpath "$1")
train_jsonl=$(realpath "$2")
output_root=$(realpath -m "$3")
mkdir -p "$output_root"

for run in first second; do
    env -u PYTHONPATH -u TT_FINETUNE_WORKSPACE \
        python -m blackhole_training.run_real_workload \
        small train \
        --steps 2 \
        --model-path "$model_snapshot" \
        --dataset "$train_jsonl" \
        --runtime-mode isolated \
        --output-root "$output_root/$run"
done

plugin=$(python - <<'PY'
from pathlib import Path
import pjrt_plugin_tt
print(Path(pjrt_plugin_tt.__file__).resolve().parent / "pjrt_plugin_tt.so")
PY
)

strings "$plugin" | grep -E \
    'onExecutableSerialize|PJRT_Executable_DeserializeAndLoad_Args' \
    > "$output_root/plugin-symbols.txt"

jq '{compile, persistent_runtime}' \
    "$output_root/first/small/training_metrics.json" \
    > "$output_root/first-compile.json"
jq '{compile, persistent_runtime}' \
    "$output_root/second/small/training_metrics.json" \
    > "$output_root/second-compile.json"

jq -n \
    --slurpfile first "$output_root/first-compile.json" \
    --slurpfile second "$output_root/second-compile.json" \
    '{
      format: "tt-finetune-pjrt-persistence-reproducer-v1",
      expected: "lower-level reuse but no fresh-process executable load",
      first: $first[0],
      second: $second[0]
    }' > "$output_root/result.json"

cat "$output_root/result.json"
