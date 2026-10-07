#!/usr/bin/env bash
#
# run_slice_pareto.sh -- fixed-slice EVAL sweep for scx_rl_exec across test workloads.
#
# Traces the achievable runtime/latency Pareto frontier from the FIFO end to the
# CFS end by reusing ONE already-trained placement model and only varying the
# macro time slice (FIXED_SLICE_MS, which bypasses the slice DQN). Eval-only:
# greedy, no training, so each point is just one trace replay (~6 min).
#
#   large slice / RTC  -> few preemptions -> low runtime, high latency
#   small slice (32ms) -> heavy time-share -> high runtime, low latency
#
# Each (workload w, slice) point writes results/scx_rl_exec/slice_pareto/s<ms>/<w>_metrics
# plus a per-task TaskNew/FirstRun/TaskDead lifecycle log s<ms>/<w> (same format as
# results/scx_cfs/<w>), the source for the per-task CDF/violin plots. Every workload
# sweeps the same slice set, so each s<ms> dir holds one <w>_metrics per workload;
# synthesized_pareto.py --workload <w> reads whichever s-dirs contain that workload.
# Set OCCUPANCY=1 (outer env) to also emit a per-slice s<ms>/<w>_occupancy.
# Slice DQN is bypassed, so this is the *achievable* frontier of the slice knob
# under a fixed placement policy -- NOT a learned-per-point frontier.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/scx_rl_exec"
# SEED selects which trained placement model to reuse (results/scx_rl_exec/seed_<N>)
# and is passed to the eval run. It also picks the output dir so a non-zero seed's
# sweep NEVER clobbers the committed seed_0 frontier under slice_pareto/:
#   SEED=0 (default) -> model seed_0, out slice_pareto/       (the committed frontier)
#   SEED=<N>         -> model seed_N, out slice_pareto_seed<N>/
# Overlay other seeds' operating points to fill runtime gaps in the day02 frontier:
#   SEED=1 sudo -E ./run_slice_pareto.sh          # sweeps test_day02, seed_1 model
# MODEL_SRC / SWEEP still override the seed-derived defaults if set explicitly.
SEED="${SEED:-0}"
MODEL_SRC="${MODEL_SRC:-$RES/seed_$SEED}"      # placement model reused for every slice
if [[ "$SEED" == "0" ]]; then
  SWEEP="${SWEEP:-$RES/slice_pareto}"
else
  SWEEP="${SWEEP:-$RES/slice_pareto_seed$SEED}"
fi
# Workloads to sweep. For seed 0 the default is the 5 held-out test workloads other
# than test_day02 (whose seed_0 frontier is already committed). For a non-zero seed
# the default is test_day02 -- the frontier we overlay seeds onto to fill its gaps.
# Each derives its trace as trace_<w>_local.txt. Override: WORKLOADS="test_day02 ...".
if [[ "$SEED" == "0" ]]; then
  WORKLOADS="${WORKLOADS:-test_day02_hetero test_day10 test_day10_hetero test_noon test_noon_hetero}"
else
  WORKLOADS="${WORKLOADS:-test_day02}"
fi
OCCUPANCY="${OCCUPANCY:-}"
# Default sweep = the slice points already present under $SWEEP (so a plain re-run
# backfills every existing s<ms> dir, e.g. with the per-task test_day02 log),
# ordered large->small so the fast/safe points finish first (32 ms may starve).
# For a FRESH seed dir (no s<ms> yet) it mirrors the committed seed_0 frontier's
# slice set, so a new seed sweeps the SAME slices (a comparable overlay); if even
# that is absent it falls back to a small sketch. Override: SLICES="...".
if [[ -z "${SLICES:-}" ]]; then
  SLICES="$(ls -d "$SWEEP"/s*/ 2>/dev/null | sed -E 's#.*/s([0-9]+)/?$#\1#' | sort -rn | tr '\n' ' ')"
  if [[ -z "${SLICES// }" ]]; then
    SLICES="$(ls -d "$RES/slice_pareto"/s*/ 2>/dev/null | sed -E 's#.*/s([0-9]+)/?$#\1#' | sort -rn | tr '\n' ' ')"
  fi
  SLICES="${SLICES:-1000000 1633 390 174 32}"
fi
log() { echo "[$(date -Iseconds)] $*"; }

if [[ ! -f "$MODEL_SRC/exec_best.pth" ]]; then
  log "!! no model at $MODEL_SRC/exec_best.pth"; exit 1
fi

mkdir -p "$SWEEP"
log "=== slice-Pareto sweep START (model=$MODEL_SRC workloads=[$WORKLOADS]) ==="
for w in $WORKLOADS; do
  trace="trace_${w}_local.txt"
  if [[ ! -f "$HERE/$trace" ]]; then
    log "!! skip workload=$w: no trace $HERE/$trace"; continue
  fi
  log "=== workload=$w (trace=$trace) ==="
  for s in $SLICES; do
    dir="$SWEEP/s$s"
    mkdir -p "$dir"
    # --eval loads MODEL_DIR/exec_best.pth, so the reused model must live there.
    cp -f "$MODEL_SRC/exec_best.pth" "$dir/exec_best.pth"
    log "## $w slice=${s} ms -> $dir/${w}_metrics (+ per-task $dir/${w})"
    occ="${OCCUPANCY:+$dir/${w}_occupancy}"
    SCHED=rl_exec EVAL=1 SEED="$SEED" \
      FIXED_SLICE_MS="$s" \
      MODEL_DIR="$dir" \
      METRICS="$dir/${w}_metrics" \
      PERTASK="$dir/${w}" \
      OCCUPANCY="$occ" \
      QUEUELOG= CORE_BACKLOG= \
      "$HERE/run_workload.sh" "$trace" >"$dir/run_${w}.log" 2>&1 \
      && log "<<< $w slice=${s} OK  $(grep -hoE 'accumulated task (latency|runtime):[^,]*' "$dir/${w}_metrics" 2>/dev/null | tr '\n' ' ')" \
      || log "!! $w slice=${s} FAILED (rc=$?) -- see $dir/run_${w}.log"
  done
  log "=== workload=$w DONE ==="
done
log "=== slice-Pareto sweep DONE ==="
