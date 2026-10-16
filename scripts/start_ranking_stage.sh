#!/usr/bin/env bash
set -euo pipefail
ranking_source="${1:?Usage: bash scripts/start_ranking_stage.sh <completed_sampling_screen> [gpu=1] [epochs=50]}"
ranking_gpu="${2:-1}"
ranking_epochs="${3:-50}"
if ! [[ "$ranking_gpu" =~ ^[0-9]+$ && "$ranking_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU must be a nonnegative index and epochs a positive integer" >&2
  exit 2
fi
ranking_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ranking_repo"
ranking_source="$(cd -- "$ranking_source" && pwd)"
if ! [[ -f "$ranking_source/screen_summary.json" && -f "$ranking_source/epoch85_lr1e-4/best_auc.pth" ]]; then
  echo "Completed source summary or epoch85_lr1e-4/best_auc.pth missing" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
ranking_python="$(python -c 'import sys; print(sys.executable)')"
CUDA_VISIBLE_DEVICES="$ranking_gpu" "$ranking_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Selected GPU unavailable; activate directafford_env and check GPU index")
print("Python CUDA preflight:",torch.__version__,torch.cuda.get_device_name(0),flush=True)
PY
mkdir -p "$ranking_repo/outputs"
ranking_output="$(mktemp -d "$ranking_repo/outputs/ranking_stage_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
nohup "$ranking_python" -u "$ranking_repo/scripts/run_ranking_stage.py" \
  --source-screen "$ranking_source" --data-dir "$ranking_repo/data/laso_dataset" \
  --gpu "$ranking_gpu" --epochs "$ranking_epochs" --seed 42 --output-root "$ranking_output" \
  > "$ranking_output/stage.log" 2>&1 &
ranking_pid=$!
printf '%s\n' "$ranking_pid" > "$ranking_output/run.pid"
printf 'PID=%s\nOutput=%s\n' "$ranking_pid" "$ranking_output"
printf 'View log: tail -n 20 "%s/stage.log"\n' "$ranking_output"
