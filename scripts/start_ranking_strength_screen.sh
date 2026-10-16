#!/usr/bin/env bash
set -euo pipefail
strength_gpu="${1:-1}"
strength_epochs="${2:-50}"
if ! [[ "$strength_gpu" =~ ^[0-9]+$ && "$strength_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/start_ranking_strength_screen.sh [gpu=1] [epochs=50]" >&2
  exit 2
fi
strength_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$strength_repo"
export PYTHONNOUSERSITE=1
strength_python="$(python -c 'import sys; print(sys.executable)')"
CUDA_VISIBLE_DEVICES="$strength_gpu" "$strength_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Selected GPU unavailable; activate directafford_env and check GPU index")
print("Python CUDA preflight:",torch.__version__,torch.cuda.get_device_name(0),flush=True)
PY
mkdir -p "$strength_repo/outputs"
strength_output="$(mktemp -d "$strength_repo/outputs/ranking_strength_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
nohup "$strength_python" -u "$strength_repo/scripts/run_ranking_strength_screen.py" \
  --data-dir "$strength_repo/data/laso_dataset" --gpu "$strength_gpu" \
  --epochs "$strength_epochs" --seed 42 --output-root "$strength_output" \
  > "$strength_output/screen.log" 2>&1 &
strength_pid=$!
printf '%s\n' "$strength_pid" > "$strength_output/run.pid"
printf 'PID=%s\nOutput=%s\n' "$strength_pid" "$strength_output"
printf 'View log: tail -n 20 "%s/screen.log"\n' "$strength_output"
