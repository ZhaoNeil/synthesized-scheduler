#!/usr/bin/env bash
#
# rerun_table_std.sh -- re-run the heterogeneous-workload comparison REPS times so
# its numbers can be reported as mean +/- std.
#
# It runs the 9 policies for each of the three heterogeneous workloads without
# recording occupancy / queue / per-task logs -- only a single aggregate
# *_metrics file per run is written, into a separate tree ($OUT, default
# results/std/) that the plot scripts do not read.
#
# Workloads: test_day02_hetero, test_noon_hetero, test_day10_hetero.
#
# The 9 policies and how each is produced:
#   CFS              SCHED=cfs
#   EEVDF            SCHED=eevdf
#   ALPS             SCHED=alps
#   Synthesized Latency        scx_rl_res  main eval  (best seed, greedy)
#   Hybrid           SCHED=hybrid
#   Synthesized Intermediate   scx_rl_exec eval, FIXED_SLICE_MS=1633, seed_0 model
#   FIFO             SCHED=cfifo
#   SFS              SCHED=sfs
#   Synthesized Runtime        scx_rl_exec main eval  (best seed, greedy)
#
# The RL "main eval" seeds are auto-selected the same way run_all_test_workloads.sh
# does (lowest recorded training objective): rl_res -> min seed_*/.res_best_latency,
# rl_exec -> min seed_*/.exec_best_runtime. Override with RL_RES_SEED= / RL_EXEC_SEED=.
# The Intermediate fixed-slice point reuses the seed_0 placement model (INTER_SEED),
# exactly like run_slice_pareto.sh's committed slice_pareto/ frontier.
#
# Cost is NOT computed here (it is derived in aggregate_table_std.py as
# runtime / CFS_runtime within each repetition & workload).
#
# Layout written:
#   $OUT/rep<r>/<Policy>/<workload>_metrics     (the only artifact per run)
#   $OUT/logs/rep<r>__<Policy>__<workload>.log  (full run log)
# Resumable: a run whose _metrics already exists & is non-empty is SKIPPED
# (pass FORCE=1 to redo). Runs are SEQUENTIAL (only one sched_ext at a time).
#
# Overrides (env):
#   REPS=5                    number of repetitions (default 5)
#   OUT=<dir>                 output tree (default results/std)
#   POLICIES="CFS FIFO ..."   subset of rows to run (default: all 9, table order)
#   WORKLOADS="test_day02_hetero ..."  workloads (default: the 3 hetero ones)
#   RL_RES_SEED= / RL_EXEC_SEED= / INTER_SEED=   force RL seeds (default: auto/0)
#   INTER_SLICE=1633          fixed slice (ms) for Synthesized Intermediate
#   COOLDOWN=5                seconds to settle between runs
#   FORCE=1                   redo runs whose _metrics already exists
#   DRYRUN=1                  print the plan without running
#
# Examples:
#   sudo ./rerun_table_std.sh                       # all 9 rows x 3 wl x 5 reps
#   sudo POLICIES="S_Latency S_Intermediate S_Runtime" ./rerun_table_std.sh
#   sudo REPS=3 DRYRUN=1 ./rerun_table_std.sh
#
# NOTE: a full sweep is 9 x 3 x 5 = 135 runs (~4.5-8.5 min each) => ~12-15 h.
# Launch it detached, e.g.:
#   screen -S std -dm bash -c 'sudo ./rerun_table_std.sh 2>&1 | tee rerun_std.out'

set -uo pipefail   # NOT -e: one failed run must not abort the whole sweep

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
RES="$REPO_ROOT/results"

REPS="${REPS:-5}"
OUT="${OUT:-$REPO_ROOT/results/std}"
COOLDOWN="${COOLDOWN:-5}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"
INTER_SLICE="${INTER_SLICE:-1633}"
INTER_SEED="${INTER_SEED:-0}"

# Policies to run, in output order. Override with POLICIES=.
read -ra POLICIES <<< "${POLICIES:-CFS EEVDF ALPS S_Latency Hybrid S_Intermediate FIFO SFS S_Runtime}"
# The three heterogeneous traces. Override with WORKLOADS=.
read -ra WORKLOADS <<< "${WORKLOADS:-test_day02_hetero test_noon_hetero test_day10_hetero}"

if [[ "$DRYRUN" != "1" && $EUID -ne 0 ]]; then
  echo "rerun_table_std.sh must run as root (it loads sched_ext schedulers)." >&2
  echo "  (use DRYRUN=1 to preview the plan without root)" >&2
  exit 1
fi

# --- auto-select the best RL seed exactly like run_all_test_workloads.sh --------
# Echoes the seed dir number with the lowest value in $2 (lower = better).
select_best_seed() {
  local rl_dir="$1" metric_file="$2" best_pth="$3"
  local best_seed="" best_val="" val seed_dir
  for seed_dir in "$rl_dir"/seed_*; do
    [[ -d "$seed_dir" && -f "$seed_dir/$best_pth" && -f "$seed_dir/$metric_file" ]] || continue
    val="$(tr -d '[:space:]' < "$seed_dir/$metric_file")"
    [[ -n "$val" ]] || continue
    if [[ -z "$best_val" ]] || awk -v a="$val" -v b="$best_val" 'BEGIN{exit !(a<b)}'; then
      best_val="$val"; best_seed="${seed_dir##*/seed_}"
    fi
  done
  [[ -n "$best_seed" ]] && printf '%s' "$best_seed"
}

RES_SEED="${RL_RES_SEED:-$(select_best_seed "$RES/scx_rl_res"  .res_best_latency  res_best.pth)}"
EXEC_SEED="${RL_EXEC_SEED:-$(select_best_seed "$RES/scx_rl_exec" .exec_best_runtime exec_best.pth)}"
RES_SEED="${RES_SEED:-0}"; EXEC_SEED="${EXEC_SEED:-0}"

# --- fill RUN_ENV (global) with the env for one policy label --------------------
RUN_ENV=()
policy_env() {
  RUN_ENV=()
  case "$1" in
    CFS)            RUN_ENV=(SCHED=cfs) ;;
    EEVDF)          RUN_ENV=(SCHED=eevdf) ;;
    ALPS)           RUN_ENV=(SCHED=alps) ;;
    SFS)            RUN_ENV=(SCHED=sfs) ;;
    FIFO)           RUN_ENV=(SCHED=cfifo) ;;
    Hybrid)         RUN_ENV=(SCHED=hybrid) ;;
    S_Latency)      RUN_ENV=(SCHED=rl_res  EVAL=1 MODEL_DIR="$RES/scx_rl_res/seed_$RES_SEED"   SEED="$RES_SEED") ;;
    S_Runtime)      RUN_ENV=(SCHED=rl_exec EVAL=1 MODEL_DIR="$RES/scx_rl_exec/seed_$EXEC_SEED" SEED="$EXEC_SEED") ;;
    S_Intermediate) RUN_ENV=(SCHED=rl_exec EVAL=1 MODEL_DIR="$RES/scx_rl_exec/seed_$INTER_SEED" SEED="$INTER_SEED" FIXED_SLICE_MS="$INTER_SLICE") ;;
    *) echo "  !! unknown policy '$1' -- skipping" >&2; return 1 ;;
  esac
}

# Give the user (not root) ownership of anything we create under $OUT.
restore_user_access() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    local owner="$SUDO_USER"; [[ -n "${SUDO_GID:-}" ]] && owner="$owner:$SUDO_GID"
    chown -R "$owner" "$path" 2>/dev/null || true
  fi
}

# Wait until no sched_ext scheduler is attached (previous run fully detached).
wait_scx_idle() {
  for _ in $(seq 1 100); do
    [[ "$(cat /sys/kernel/sched_ext/state 2>/dev/null)" != "enabled" ]] && return 0
    sleep 0.1
  done
  echo "  !! a sched_ext scheduler is still attached; the next run may fail" >&2
}

mkdir -p "$OUT/logs"
declare -a OK=() FAIL=() SKIP=()
t_start=$SECONDS

total=$(( REPS * ${#POLICIES[@]} * ${#WORKLOADS[@]} ))
echo "=== table std re-run (Workload 4-6, heterogeneous) ==="
echo "    reps:       $REPS"
echo "    policies:   ${POLICIES[*]}"
echo "    workloads:  ${WORKLOADS[*]}"
echo "    RL seeds:   rl_res=seed_$RES_SEED  rl_exec=seed_$EXEC_SEED  intermediate=seed_$INTER_SEED (slice ${INTER_SLICE}ms)"
echo "    out:        ${OUT}   (no results/scx_<name> dir is touched)"
echo "    total runs: $total   force=$FORCE  dryrun=$DRYRUN  cooldown=${COOLDOWN}s"
echo

for r in $(seq 1 "$REPS"); do
  echo "########## repetition $r / $REPS ##########"
  for w in "${WORKLOADS[@]}"; do
    trace="trace_${w}_local.txt"
    if [[ ! -f "$HERE/$trace" ]]; then
      echo "  !! no trace $HERE/$trace -- skipping workload $w" >&2; continue
    fi
    for p in "${POLICIES[@]}"; do
      policy_env "$p" || { FAIL+=("rep$r/$p/$w"); continue; }
      out_dir="$OUT/rep$r/$p"
      metrics="$out_dir/${w}_metrics"
      log="$OUT/logs/rep${r}__${p}__${w}.log"
      label="rep$r $p $w"

      if [[ "$FORCE" != "1" && -s "$metrics" ]]; then
        echo "  SKIP  $label  (exists; FORCE=1 to redo)"; SKIP+=("$label"); continue
      fi
      # A dry run must stay side-effect free, so print the plan BEFORE any mkdir.
      if [[ "$DRYRUN" == "1" ]]; then
        echo "  DRY   $label -> env ${RUN_ENV[*]} METRICS=$metrics run_workload.sh $trace"
        continue
      fi
      mkdir -p "$out_dir"

      wait_scx_idle
      t0=$SECONDS
      # Only a metrics file is written: QUEUELOG/PERTASK/CORE_BACKLOG disabled
      # (empty) and OCCUPANCY left unset (off). Nothing lands under results/.
      if env "${RUN_ENV[@]}" \
             METRICS="$metrics" QUEUELOG= PERTASK= CORE_BACKLOG= \
             "$HERE/run_workload.sh" "$trace" > "$log" 2>&1; then
        lat=$(grep -oE 'accumulated task latency:[ ]*[0-9.]+' "$metrics" | grep -oE '[0-9.]+$')
        run=$(grep -oE 'accumulated task runtime:[ ]*[0-9.]+' "$metrics" | grep -oE '[0-9.]+$')
        echo "  OK    $label  ($((SECONDS - t0))s)  lat=${lat:-?} run=${run:-?}"
        OK+=("$label")
      else
        echo "  FAIL  $label  (rc=$?, $((SECONDS - t0))s)  see ${log#$REPO_ROOT/}" >&2
        FAIL+=("$label")
      fi
      restore_user_access "$out_dir"; restore_user_access "$log"
      sleep "$COOLDOWN"
    done
  done
  restore_user_access "$OUT"
done

restore_user_access "$OUT"
echo
echo "=== done in $(( (SECONDS - t_start) / 60 )) min: ${#OK[@]} ok, ${#FAIL[@]} failed, ${#SKIP[@]} skipped ==="
[[ ${#FAIL[@]} -gt 0 ]] && printf '  FAILED: %s\n' "${FAIL[*]}"
echo "Next: python3 $HERE/aggregate_table_std.py --in $OUT"
[[ ${#FAIL[@]} -eq 0 ]]
