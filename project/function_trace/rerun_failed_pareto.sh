#!/usr/bin/env bash
#
# rerun_failed_pareto.sh -- re-run only the slice-Pareto points that died with an
# "RCU CPU stall" (or other EXIT runtime error). It rescans the existing run logs
# for the failure signature, so it always targets whatever is currently broken;
# a point that succeeds on retry drops out of the list on the next invocation.
#
# Mirrors the per-point invocation in run_slice_pareto.sh exactly:
# same model reuse, EVAL/SEED/FIXED_SLICE_MS, and output paths.
#
# Usage:  ./rerun_failed_pareto.sh            # rerun all currently-failed points
#         DRYRUN=1 ./rerun_failed_pareto.sh   # just list what would be rerun
#         OCCUPANCY=1 ./rerun_failed_pareto.sh # also re-emit per-slice occupancy
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/scx_rl_exec"
MODEL_SRC="${MODEL_SRC:-$RES/seed_0}"
SWEEP="$RES/slice_pareto"
OCCUPANCY="${OCCUPANCY:-}"
FAILRE="RCU CPU stall|runtime error|EXIT: .*error|panic|Segmentation|Traceback"
log() { echo "[$(date -Iseconds)] $*"; }

if [[ ! -f "$MODEL_SRC/exec_best.pth" ]]; then
  log "!! no model at $MODEL_SRC/exec_best.pth"; exit 1
fi

# Collect (slice, workload) pairs from failed logs: s<ms>/run_<workload>.log
mapfile -t FAILED < <(
  grep -lE "$FAILRE" "$SWEEP"/s*/run_test_*.log 2>/dev/null \
    | sed -E 's#.*/s([0-9]+)/run_(test_[^/]+)\.log#\1 \2#' | sort -k2,2 -k1,1n)

if [[ ${#FAILED[@]} -eq 0 ]]; then
  log "nothing to rerun -- no failed logs found."; exit 0
fi

log "=== ${#FAILED[@]} failed point(s) to rerun ==="
printf '  slice=%-8s workload=%s\n' $(for p in "${FAILED[@]}"; do echo "$p"; done | awk '{print $1" "$2}')

if [[ -n "${DRYRUN:-}" ]]; then
  log "DRYRUN set -- not executing."; exit 0
fi

for pair in "${FAILED[@]}"; do
  s="${pair%% *}"; w="${pair##* }"
  trace="trace_${w}_local.txt"
  dir="$SWEEP/s$s"
  if [[ ! -f "$HERE/$trace" ]]; then
    log "!! skip s$s/$w: no trace $HERE/$trace"; continue
  fi
  mkdir -p "$dir"
  cp -f "$MODEL_SRC/exec_best.pth" "$dir/exec_best.pth"
  occ="${OCCUPANCY:+$dir/${w}_occupancy}"
  log "## RERUN $w slice=${s} ms -> $dir/${w}_metrics"
  SCHED=rl_exec EVAL=1 SEED=0 \
    FIXED_SLICE_MS="$s" \
    MODEL_DIR="$dir" \
    METRICS="$dir/${w}_metrics" \
    PERTASK="$dir/${w}" \
    OCCUPANCY="$occ" \
    QUEUELOG= CORE_BACKLOG= \
    "$HERE/run_workload.sh" "$trace" >"$dir/run_${w}.log" 2>&1 \
    && log "<<< $w slice=${s} OK  $(grep -hoE 'accumulated task (latency|runtime):[^,]*' "$dir/${w}_metrics" 2>/dev/null | tr '\n' ' ')" \
    || log "!! $w slice=${s} STILL FAILED (rc=$?) -- see $dir/run_${w}.log"
done
log "=== rerun DONE ==="
