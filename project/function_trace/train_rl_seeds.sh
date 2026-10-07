#!/usr/bin/env bash
#
# Train scx_rl_exec or scx_rl_res from scratch for multiple random seeds and
# append all per-episode rows to the scheduler's train_metrics.csv.
#
# Defaults:
#   scheduler: rl_exec
#   seeds:     0 1 2 3 4
#   episodes:  50 per seed
#   trace:     trace_local.txt
#
# Examples:
#   sudo ./train_rl_seeds.sh exec
#   sudo ./train_rl_seeds.sh res
#   sudo EPISODES=10 ./train_rl_seeds.sh rl_exec 5 6
#   sudo TRACE=trace_test_day02_local.txt ./train_rl_seeds.sh res 0 1

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "train_rl_seeds.sh must run as root (it loads a sched_ext scheduler)." >&2
  exit 1
fi

case "${1:-${RL_SCHED:-rl_exec}}" in
  exec|rl_exec|scx_rl_exec)
    RL_SCHED="rl_exec"
    [[ "${1:-}" =~ ^(exec|rl_exec|scx_rl_exec)$ ]] && shift
    ;;
  res|rl_res|scx_rl_res)
    RL_SCHED="rl_res"
    [[ "${1:-}" =~ ^(res|rl_res|scx_rl_res)$ ]] && shift
    ;;
  *)
    RL_SCHED="${RL_SCHED:-rl_exec}"
    ;;
esac

case "$RL_SCHED" in
  rl_exec|rl_res) ;;
  *)
    echo "unknown RL_SCHED='$RL_SCHED' (expected rl_exec or rl_res)" >&2
    exit 1
    ;;
esac

DRIVER="$HERE/$RL_SCHED.sh"
if [[ ! -x "$DRIVER" ]]; then
  echo "missing driver: $DRIVER" >&2
  exit 1
fi

EPISODES="${EPISODES:-50}"
TRACE="${TRACE:-trace_local.txt}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/scx_$RL_SCHED}"
MODEL_ROOT="${MODEL_ROOT:-$RESULT_DIR}"
METRICS_CSV="${METRICS_CSV:-$RESULT_DIR/train_metrics.csv}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

mkdir -p "$MODEL_ROOT"

echo "=== $RL_SCHED multi-seed training ==="
echo "    seeds: ${SEEDS[*]}"
echo "    episodes per seed: $EPISODES"
echo "    trace: $TRACE"
echo "    metrics: $METRICS_CSV"
echo "    model root: $MODEL_ROOT"

for seed in "${SEEDS[@]}"; do
  echo
  echo "===== $RL_SCHED seed $seed started at $(date -Iseconds) ====="
  model_dir="$MODEL_ROOT/seed_$seed"
  mkdir -p "$model_dir"

  FRESH=1 \
    METRICS_APPEND=1 \
    SEED="$seed" \
    MODEL_DIR="$model_dir" \
    RESULT_DIR="$RESULT_DIR" \
    METRICS_CSV="$METRICS_CSV" \
    "$DRIVER" train "$EPISODES" "$TRACE"

  echo "===== $RL_SCHED seed $seed finished at $(date -Iseconds) ====="
done

echo
echo "=== done. all metrics appended to $METRICS_CSV ==="
