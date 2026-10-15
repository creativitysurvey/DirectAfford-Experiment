#!/usr/bin/env bash
# Launch once from the activated project environment; print persistent log paths.
set -euo pipefail
sampling_gpu="${1:-1}"
sampling_epochs="${2:-50}"
if ! [[ "$sampling_gpu" =~ ^[0-9]+$ && "$sampling_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/start_sampling_screen.sh [physical_gpu=1] [epochs=50]" >&2
  exit 2
fi
sampling_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sampling_repo"
export PYTHONNOUSERSITE=1
sampling_python="$(python -c 'import sys; print(sys.executable)')"
CUDA_VISIBLE_DEVICES="$sampling_gpu" "$sampling_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Selected GPU unavailable; activate directafford_env and check GPU index")
print("Python CUDA preflight:", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
mkdir -p "$sampling_repo/outputs"
sampling_output="$(mktemp -d "$sampling_repo/outputs/sampling_lr_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
nohup "$sampling_python" -u "$sampling_repo/scripts/run_sampling_screen.py" \
  --data-dir "$sampling_repo/data/laso_dataset" --gpu "$sampling_gpu" \
  --epochs "$sampling_epochs" --seed 42 --output-root "$sampling_output" \
  > "$sampling_output/screen.log" 2>&1 &
sampling_pid=$!
printf '%s\n' "$sampling_pid" > "$sampling_output/run.pid"
printf 'PID=%s\nOutput=%s\n' "$sampling_pid" "$sampling_output"
printf 'View log: tail -n 20 "%s/screen.log"\n' "$sampling_output"
