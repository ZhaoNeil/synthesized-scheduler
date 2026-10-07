#!/usr/bin/env bash
#
# run_slice_experiments.sh -- two fixed-slice ceilings for scx_rl_exec. The slice DQN is bypassed (FIXED_SLICE_MS);
# only the constant slice differs between the runs. Each: fresh-train 30 episodes
# on trace_local, then test 3 episodes on day02. Separate model dirs + CSVs.
#
#   A: slice = 1633 ms              -> results/scx_rl_exec/train_slice1633.csv
#   B: slice = 1000000 ms (RTC)     -> results/scx_rl_exec/train_slice_rtc.csv
#
# Must run as root (loads a sched_ext scheduler). Detach it:
#   sudo setsid nohup ./run_slice_experiments.sh >slice_experiments.log 2>&1 &

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/scx_rl_exec"
log() { echo "[$(date -Iseconds)] $*"; }

run_one() {
  local tag="$1" slice_ms="$2"
  log "############################################################"
  log "## EXP $tag : fixed slice = ${slice_ms} ms (30 train + 3 test day02)"
  log "############################################################"
  FIXED_SLICE_MS="$slice_ms" FRESH=1 SEED=0 \
    MODEL_DIR="$RES/slice_$tag" \
    METRICS_CSV="$RES/train_slice_$tag.csv" \
    TEST_TRACE=trace_test_day02_local.txt TEST_REPS=3 \
    "$HERE/rl_exec.sh" train 30 trace_local.txt \
    && log "<<< EXP $tag OK" || log "!! EXP $tag FAILED (rc=$?)"
}

log "=== slice-ceiling experiments START ==="
run_one 1633 1633
run_one rtc 1000000
log "=== slice-ceiling experiments DONE ==="
