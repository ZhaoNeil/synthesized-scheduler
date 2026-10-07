#!/usr/bin/env bash
#
# Re-run all 11 non-RL schedulers on the test_day02 workload, sequentially,
# after a workload recalibration. Each scheduler writes the standard three-file
# layout to results/<scheduler>/ (e.g. results/scx_eevdf/test_day02{,_metrics,_queue}),
# overwriting the previous run. The RL schedulers (scx_rl_exec / scx_rl_res) are
# intentionally NOT included here -- they get re-trained separately.
#
# Runs one scheduler at a time (only one sched_ext scheduler can be loaded at
# once). A per-scheduler log goes to results/rerun_logs/<sched>.log.
#
# Per-core occupancy is collected by default (OCCUPANCY=1 -> each run also writes
# results/<sched>/<workload>_occupancy: "core task start_s end_s dur_s" intervals,
# including mid-run core changes). Disable with OCCUPANCY= ; tune the sampler with
# OCCUPANCY_PERIOD_US / OCCUPANCY_MAX_SEG (see run_workload.sh).
#
# Usage (detach from the session so closing it doesn't kill the run):
#   sudo setsid ./rerun_baselines.sh > results/rerun_logs/rerun.log 2>&1 &
# or just foreground:
#   sudo ./rerun_baselines.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "rerun_baselines.sh must run as root (it loads sched_ext schedulers)." >&2
  exit 1
fi

TRACE="${TRACE:-trace_test_day02_local.txt}"

# SCHED values in the same order they appear under results/ (maps run_workload.sh
# SCHED -> results dir): the rr family writes to scx_rr[_<mech>].
SCHEDS=(cfifo cfs alps sfs eevdf hybrid rr preempt p2c shuffle preempt_shuffle)

LOG_DIR="$REPO_ROOT/results/rerun_logs"
mkdir -p "$LOG_DIR"

echo "=== rerun baselines on $TRACE ==="
echo "    schedulers: ${SCHEDS[*]}"
echo "    logs:       $LOG_DIR/<sched>.log"
echo "    occupancy:  ${OCCUPANCY:-1} (per-core intervals -> results/<sched>/<workload>_occupancy)"
echo

declare -a OK=() FAIL=()
for sched in "${SCHEDS[@]}"; do
  log="$LOG_DIR/$sched.log"
  echo "----- [$(date '+%H:%M:%S')] SCHED=$sched -----"
  if env SCHED="$sched" OCCUPANCY="${OCCUPANCY:-1}" "$HERE/run_workload.sh" "$TRACE" > "$log" 2>&1; then
    echo "  OK  (log: $log)"
    OK+=("$sched")
  else
    echo "  FAIL (exit $?, see $log)" >&2
    FAIL+=("$sched")
  fi
done

echo
echo "=== done: ${#OK[@]} ok, ${#FAIL[@]} failed ==="
[[ ${#OK[@]}   -gt 0 ]] && echo "  ok:     ${OK[*]}"
[[ ${#FAIL[@]} -gt 0 ]] && echo "  failed: ${FAIL[*]}"
