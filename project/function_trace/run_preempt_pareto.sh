#!/usr/bin/env bash
#
# run_preempt_pareto.sh -- fixed-slice, global preemption-BUDGET eval sweep for
# scx_rl_exec.
#
# Aims at the runtime/cost void that the fixed-slice sweep (run_slice_pareto.sh)
# leaves in the middle of the cost frontier. The slice knob is a THRESHOLD (all long
# tasks cross into extra preemption at one slice value -> a cost cliff), and a
# per-task preemption cap is bimodal (it still preempts EVERY long task). A GLOBAL
# preemption budget instead caps the TOTAL number of slice-preemptions this episode:
# preemptions are spent chronologically, so the early trace runs time-shared and the
# rest runs FIFO once the budget is exhausted. Sweeping the budget dials the total
# preemption count -- and hence accumulated runtime/cost -- CONTINUOUSLY.
#
# Base macro slice is held FIXED (FIXED_SLICE_MS = BASE_SLICE_MS); each point varies
# only the global budget N (PREEMPT_BUDGET):
#   N=0        -> no preemption at all (pure FIFO: cheap, high latency)
#   0 < N < F  -> only the first N preemptions happen -> intermediate cost/latency
#   N >= F     -> full time-slicing at BASE_SLICE_MS (F = that slice's own total
#                 preemption count, ~2134 for slice=4000 on test_day02)
#
# Eval-only: reuse ONE already-trained placement model, greedy, no training (~6 min
# per point). Each (workload w, budget N) point writes
# results/scx_rl_exec/preempt_pareto/s<BASE>n<N>/<w>_metrics plus the per-task
# lifecycle log s<BASE>n<N>/<w>. Set OCCUPANCY=1 (outer env) for a per-point
# s<BASE>n<N>/<w>_occupancy. The realized total preemption count is printed by the
# scheduler as `rl_metrics preempt=...` in run_<w>.log.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/scx_rl_exec"
# Reuse the same trained placement model the slice sweep uses (seed_0 by default).
SEED="${SEED:-0}"
MODEL_SRC="${MODEL_SRC:-$RES/seed_$SEED}"
# Fixed macro slice that sets the EXPENSIVE (unbudgeted) end of the sweep. 4000 ms is
# the slice=4000 operating point (rel_cost ~0.33x) just past the day02 cliff; the
# budget then trades cost DOWN from there through the void toward the FIFO corner.
BASE_SLICE_MS="${BASE_SLICE_MS:-4000}"
SWEEP="${SWEEP:-$RES/preempt_pareto}"
WORKLOADS="${WORKLOADS:-test_day02}"
# Global total-preemption budgets to sweep (0 = pure-FIFO end; ~2134 = full slice-4000
# on test_day02). Override: BUDGETS="250 500 1000".
BUDGETS="${BUDGETS:-100 250 500 750 1000 1500 2000}"
# Slice to run at AFTER the budget is spent (POST_BUDGET_SLICE_MS). Empty => run-to-
# completion (FIFO tail, the original sweep). Set to a LARGER slice than BASE_SLICE_MS
# to interpolate between two frontier slices, e.g. BASE_SLICE_MS=4000 POST_SLICE_MS=6286
# fills the void between the slice-4000 and slice-6286 operating points.
POST_SLICE_MS="${POST_SLICE_MS:-}"
OCCUPANCY="${OCCUPANCY:-}"
log() { echo "[$(date -Iseconds)] $*"; }

if [[ ! -f "$MODEL_SRC/exec_best.pth" ]]; then
  log "!! no model at $MODEL_SRC/exec_best.pth"; exit 1
fi

mkdir -p "$SWEEP"
log "=== preempt-budget sweep START (model=$MODEL_SRC base_slice=${BASE_SLICE_MS}ms workloads=[$WORKLOADS] BUDGETS=[$BUDGETS]) ==="
for w in $WORKLOADS; do
  trace="trace_${w}_local.txt"
  if [[ ! -f "$HERE/$trace" ]]; then
    log "!! skip workload=$w: no trace $HERE/$trace"; continue
  fi
  log "=== workload=$w (trace=$trace) ==="
  for n in $BUDGETS; do
    dir="$SWEEP/s${BASE_SLICE_MS}${POST_SLICE_MS:+p${POST_SLICE_MS}}n${n}"
    mkdir -p "$dir"
    # --eval loads MODEL_DIR/exec_best.pth, so the reused model must live there.
    cp -f "$MODEL_SRC/exec_best.pth" "$dir/exec_best.pth"
    log "## $w base_slice=${BASE_SLICE_MS} post_slice=${POST_SLICE_MS:-RTC} budget=${n} -> $dir/${w}_metrics"
    occ="${OCCUPANCY:+$dir/${w}_occupancy}"
    SCHED=rl_exec EVAL=1 SEED="$SEED" \
      FIXED_SLICE_MS="$BASE_SLICE_MS" \
      PREEMPT_BUDGET="$n" \
      POST_BUDGET_SLICE_MS="$POST_SLICE_MS" \
      MODEL_DIR="$dir" \
      METRICS="$dir/${w}_metrics" \
      PERTASK="$dir/${w}" \
      OCCUPANCY="$occ" \
      QUEUELOG= CORE_BACKLOG= \
      "$HERE/run_workload.sh" "$trace" >"$dir/run_${w}.log" 2>&1 \
      && log "<<< $w budget=${n} OK  $(grep -hoE 'accumulated task (latency|runtime):[^,]*' "$dir/${w}_metrics" 2>/dev/null | tr '\n' ' ')  $(grep -hoE 'preempt=[0-9]+' "$dir/run_${w}.log" | tail -1)" \
      || log "!! $w budget=${n} FAILED (rc=$?) -- see $dir/run_${w}.log"
  done
  log "=== workload=$w DONE ==="
done
log "=== preempt-budget sweep DONE ==="
