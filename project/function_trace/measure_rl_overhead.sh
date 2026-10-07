#!/usr/bin/env bash
#
# measure_rl_overhead.sh -- measure the RL inference COMPUTE overhead of the two
# reinforcement-learning schedulers (scx_rl_res and scx_rl_exec) on a test trace.
#
# For each scheduler it runs ONE greedy eval episode (loads <x>_best.pth, no
# training) on the trace, captures the full run log, then feeds that log to
# summarize_rl_overhead.py, which parses the agent's `overhead_metrics` line and
# writes results/scx_rl_<x>/overhead/{overhead.txt,overhead.csv}. The CSV reports
# the Python-side thread CPU time of the RL calls (train_step = "action",
# infer_time_slice = "slice") -- the actual inference cost (wall-clock is omitted
# on purpose; see summarize_rl_overhead.py).
#
# Usage (must be root -- it loads a sched_ext scheduler):
#   sudo ./measure_rl_overhead.sh
#
# Tunables (environment variables):
#   TRACE    test trace to replay          (default trace_test_day10_local.txt)
#   SCHEDS   which schedulers to measure    (default "rl_res rl_exec")
#   SEED     agent RNG seed                                       (default 0)
#
# Each run replays the FULL trace, so this takes a while. Detach from the
# terminal so a closed session does not kill the run and evict the scheduler:
#   sudo setsid nohup ./measure_rl_overhead.sh >overhead.log 2>&1 &
#   tail -f overhead.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

TRACE="${TRACE:-trace_test_day10_local.txt}"
SCHEDS="${SCHEDS:-rl_res rl_exec}"
SEED="${SEED:-0}"

if [[ $EUID -ne 0 ]]; then
  echo "measure_rl_overhead.sh must run as root (it loads a sched_ext scheduler)." >&2
  exit 1
fi

# Short workload name, matching run_workload.sh: trace_test_day10_local.txt -> test_day10.
WORKLOAD="$(basename "$TRACE")"; WORKLOAD="${WORKLOAD%.*}"
WORKLOAD="${WORKLOAD#trace_}"; WORKLOAD="${WORKLOAD%_local}"

for sched in $SCHEDS; do
  case "$sched" in
    rl_res)  prefix="res" ;;
    rl_exec) prefix="exec" ;;
    *) echo "!! unknown sched '$sched' (expected rl_res or rl_exec) -- skipping" >&2; continue ;;
  esac

  model_dir="$REPO_ROOT/scheds/experimental/scx_$sched/py"
  if [[ ! -f "$model_dir/${prefix}_best.pth" ]]; then
    echo "!! no trained model at $model_dir/${prefix}_best.pth -- train first; skipping $sched" >&2
    continue
  fi

  out_dir="$REPO_ROOT/results/scx_$sched/overhead"
  mkdir -p "$out_dir"
  run_log="$out_dir/run_${WORKLOAD}.log"

  echo "=== scx_$sched: eval on $TRACE (seed $SEED) -> $out_dir ==="
  # One greedy eval episode. Suppress the per-workload result files (queue /
  # per-task / core-backlog) -- we only want the aggregate metrics + the
  # overhead_metrics line -- but keep METRICS inside the overhead dir so it does
  # not clobber the main results/ files.
  env SCHED="$sched" EVAL=1 SEED="$SEED" MODEL_DIR="$model_dir" \
      METRICS="$out_dir/${WORKLOAD}_metrics" \
      QUEUELOG= PERTASK= CORE_BACKLOG= \
      "$HERE/run_workload.sh" "$TRACE" 2>&1 | tee "$run_log"

  echo
  echo "--- summarizing scx_$sched overhead ---"
  python3 "$HERE/summarize_rl_overhead.py" "$run_log" "scx_$sched" "$out_dir"
  echo
done

echo "=== done. overhead written to results/scx_rl_*/overhead/{overhead.txt,overhead.csv} ==="
