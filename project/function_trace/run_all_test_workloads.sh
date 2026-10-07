#!/usr/bin/env bash
#
# run_all_test_workloads.sh -- run every scheduler on EVERY test workload.
#
# Sweeps all six TEST traces in this directory (trace_test_*_local.txt):
#   test_day02  test_day02_hetero  test_day10  test_day10_hetero
#   test_noon   test_noon_hetero
# for the baselines  cfifo cfs alps sfs eevdf hybrid  and the two RL schedulers
# rl_exec / rl_res. Each writes the standard three-file per-workload layout:
#   results/<scheduler>/<workload>            per-task lifecycle records
#   results/<scheduler>/<workload>_metrics    aggregate metrics
#   results/<scheduler>/<workload>_queue      global backlog series
#
# RL schedulers are EVAL-only (greedy, no training) and use a SINGLE best seed,
# auto-selected as the seed with the lowest recorded training objective:
#   rl_exec -> min seed_*/.exec_best_runtime   (loads seed_<best>/exec_best.pth)
#   rl_res  -> min seed_*/.res_best_latency    (loads seed_<best>/res_best.pth)
# Their results land at the TOP level (results/scx_rl_exec/<workload>{,_metrics,
# _queue}), parallel to the baselines, so every scheduler shares one layout.
# Override the picked seed with RL_EXEC_SEED= / RL_RES_SEED=.
#
# Only one sched_ext scheduler can be loaded at a time, so runs are SEQUENTIAL.
# A full sweep is ~48 runs (~6 min each => ~5h). A per-run log goes to
# results/all_test_logs/<scheduler>__<workload>.log, and a compact comparison
# table is written to results/all_test_logs/summary.csv at the end.
#
# By default a (scheduler,workload) whose *_metrics already exists is SKIPPED so
# the sweep is resumable; pass FORCE=1 to re-run everything (e.g. after a trace
# recalibration). DRYRUN=1 prints the plan without running anything.
#
# Overrides (env):
#   SCHEDS="cfifo eevdf"          baseline schedulers to run   (default: all 6)
#   RL_SCHEDS="rl_exec"           RL schedulers to run         (default: both)
#   WORKLOADS="trace_test_noon_local.txt ..."  traces         (default: all test_)
#   RL_EXEC_SEED=3  RL_RES_SEED=0   force the RL seed (skip auto-select)
#   FORCE=1   redo runs whose _metrics already exists
#   DRYRUN=1  print what would run, then exit
#   OCCUPANCY=1  also record per-core occupancy (<workload>_occupancy) per run;
#                tune with OCCUPANCY_PERIOD_US= / OCCUPANCY_MAX_SEG= (see
#                run_workload.sh). OFF by default. NOTE: a run whose _metrics
#                already exists is SKIPPED, so add FORCE=1 to backfill occupancy
#                onto already-completed runs.
#
# Examples:
#   sudo ./run_all_test_workloads.sh
#   sudo FORCE=1 ./run_all_test_workloads.sh
#   sudo SCHEDS="eevdf" RL_SCHEDS="" ./run_all_test_workloads.sh
#   sudo WORKLOADS="trace_test_noon_local.txt" ./run_all_test_workloads.sh
#   sudo OCCUPANCY=1 FORCE=1 ./run_all_test_workloads.sh   # record occupancy too

# Note: NOT 'set -e' -- a single failed run must not abort the whole sweep; each
# run's status is captured and reported in the final summary instead.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS="$REPO_ROOT/results"
LOG_DIR="$RESULTS/all_test_logs"

FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

# Optional per-core occupancy for every run. OFF unless OCCUPANCY is non-empty
# (e.g. OCCUPANCY=1). run_workload.sh reads OCCUPANCY per-run, but this sweep
# invokes it via `env VAR=... run_workload.sh`, which passes ONLY the vars named
# on that line -- so an outer OCCUPANCY=1 never reaches the run unless forwarded
# here. Each loop below adds an explicit OCCUPANCY=<per-run path>; the tunables
# (fixed across runs) are collected once and forwarded alongside it.
OCCUPANCY="${OCCUPANCY:-}"
declare -a OCC_TUNABLES=()
if [[ -n "$OCCUPANCY" ]]; then
  [[ -n "${OCCUPANCY_PERIOD_US:-}" ]] && OCC_TUNABLES+=("OCCUPANCY_PERIOD_US=$OCCUPANCY_PERIOD_US")
  [[ -n "${OCCUPANCY_MAX_SEG:-}" ]]   && OCC_TUNABLES+=("OCCUPANCY_MAX_SEG=$OCCUPANCY_MAX_SEG")
fi

# Loading a sched_ext scheduler needs root; a dry run only prints the plan.
if [[ "$DRYRUN" != "1" && $EUID -ne 0 ]]; then
  echo "run_all_test_workloads.sh must run as root (it loads sched_ext schedulers)." >&2
  echo "  (use DRYRUN=1 to preview the plan without root)" >&2
  exit 1
fi

# Baseline (non-RL) schedulers. Each maps 1:1 to results/scx_<sched>/.
read -ra BASELINE_SCHEDS <<< "${SCHEDS-cfifo cfs alps sfs eevdf hybrid}"
# RL schedulers (EVAL-only, best single seed). Set RL_SCHEDS="" to skip them.
read -ra RL_SCHEDS_ARR <<< "${RL_SCHEDS-rl_exec rl_res}"

# Test workloads: explicit args > WORKLOADS env > every trace_test_*_local.txt.
if [[ "$#" -gt 0 ]]; then
  WORKLOADS=("$@")
elif [[ -n "${WORKLOADS:-}" ]]; then
  read -ra WORKLOADS <<< "$WORKLOADS"
else
  mapfile -t WORKLOADS < <(cd "$HERE" && ls -1 trace_test_*_local.txt 2>/dev/null)
fi
if [[ "${#WORKLOADS[@]}" -eq 0 ]]; then
  echo "no test workloads found (trace_test_*_local.txt in $HERE)." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

restore_user_access() {
  local path="$1" owner
  [[ -e "$path" ]] || return 0

  if [[ -d "$path" ]]; then
    chmod 0755 "$path" 2>/dev/null || true
  else
    chmod 0644 "$path" 2>/dev/null || true
  fi

  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    owner="$SUDO_USER"
    [[ -n "${SUDO_GID:-}" ]] && owner="$owner:$SUDO_GID"
    chown "$owner" "$path" 2>/dev/null || true
  fi
}

restore_user_access "$LOG_DIR"

# trace filename -> workload name, exactly like run_workload.sh:
#   trace_test_day02_hetero_local.txt -> test_day02_hetero
workload_name() {
  local n; n="$(basename "$1")"; n="${n%.*}"; n="${n#trace_}"; n="${n%_local}"
  printf '%s' "$n"
}

# Pick the best seed for an RL policy: the seed dir with the lowest value in its
# recorded best-objective file (runtime for exec, latency for res; lower better).
# Echoes "<seed> <value>" on success; returns 1 if no usable seed/checkpoint.
select_best_seed() {
  local rl_dir="$1" metric_file="$2" best_pth="$3"
  local best_seed="" best_val="" val seed_dir seed
  for seed_dir in "$rl_dir"/seed_*; do
    [[ -d "$seed_dir" ]] || continue
    [[ -f "$seed_dir/$best_pth" ]] || continue
    [[ -f "$seed_dir/$metric_file" ]] || continue
    val="$(tr -d '[:space:]' < "$seed_dir/$metric_file")"
    [[ -n "$val" ]] || continue
    if [[ -z "$best_val" ]] || awk -v a="$val" -v b="$best_val" 'BEGIN{exit !(a<b)}'; then
      best_val="$val"; best_seed="${seed_dir##*/seed_}"
    fi
  done
  [[ -n "$best_seed" ]] || return 1
  printf '%s %s\n' "$best_seed" "$best_val"
}

# Print every seed's recorded objective (for transparency in the log).
print_seed_ranking() {
  local rl_dir="$1" metric_file="$2" best_pth="$3" seed_dir val
  for seed_dir in "$rl_dir"/seed_*; do
    [[ -d "$seed_dir" ]] || continue
    val="$(tr -d '[:space:]' < "$seed_dir/$metric_file" 2>/dev/null)"
    [[ -f "$seed_dir/$best_pth" ]] || val="${val:-?} (no $best_pth)"
    printf '      seed %-2s %s = %s\n' "${seed_dir##*/seed_}" "$metric_file" "${val:-<empty>}"
  done
}

declare -a OK=() FAIL=() SKIP=()

# run_one <label> <skip-target metrics file> <logfile> -- <command...>
# Skips when the target already exists (unless FORCE=1); records OK/FAIL/SKIP.
run_one() {
  local label="$1" target="$2" log="$3"; shift 3
  if [[ "$FORCE" != "1" && -s "$target" ]]; then
    echo "  SKIP  $label (exists: ${target#$REPO_ROOT/}; FORCE=1 to redo)"
    SKIP+=("$label"); return 0
  fi
  if [[ "$DRYRUN" == "1" ]]; then
    echo "  DRY   $label -> $*"
    return 0
  fi
  local t0=$SECONDS rc
  if "$@" > "$log" 2>&1; then
    restore_user_access "$log"
    echo "  OK    $label  ($((SECONDS - t0))s)  log: ${log#$REPO_ROOT/}"
    OK+=("$label")
  else
    rc=$?
    restore_user_access "$log"
    echo "  FAIL  $label  (rc=$rc, $((SECONDS - t0))s)  see ${log#$REPO_ROOT/}" >&2
    FAIL+=("$label")
  fi
}

echo "=== full test-workload sweep ==="
echo "    workloads:  ${WORKLOADS[*]/#/}"
echo "    baselines:  ${BASELINE_SCHEDS[*]:-(none)}"
echo "    RL:         ${RL_SCHEDS_ARR[*]:-(none)}"
echo "    force:      $FORCE    dryrun: $DRYRUN    occupancy: ${OCCUPANCY:-off}"
echo "    logs:       ${LOG_DIR#$REPO_ROOT/}/<scheduler>__<workload>.log"
echo

# ---- baselines: results/scx_<sched>/<workload>{,_metrics,_queue} ----
for sched in "${BASELINE_SCHEDS[@]}"; do
  [[ -z "$sched" ]] && continue
  echo "----- baseline: $sched -----"
  for trace in "${WORKLOADS[@]}"; do
    name="$(workload_name "$trace")"
    metrics="$RESULTS/scx_$sched/${name}_metrics"
    log="$LOG_DIR/scx_${sched}__${name}.log"
    declare -a occ=()
    [[ -n "$OCCUPANCY" ]] && occ=(OCCUPANCY="$RESULTS/scx_$sched/${name}_occupancy" "${OCC_TUNABLES[@]}")
    run_one "scx_$sched / $name" "$metrics" "$log" \
      env SCHED="$sched" "${occ[@]}" "$HERE/run_workload.sh" "$trace"
  done
done

# ---- RL: best single seed, EVAL-only, top-level results/scx_rl_<p>/ ----
for rl in "${RL_SCHEDS_ARR[@]}"; do
  [[ -z "$rl" ]] && continue
  case "$rl" in
    rl_exec) policy=exec; best_pth=exec_best.pth; metric_file=.exec_best_runtime; seed_override="${RL_EXEC_SEED:-}" ;;
    rl_res)  policy=res;  best_pth=res_best.pth;  metric_file=.res_best_latency;  seed_override="${RL_RES_SEED:-}"  ;;
    *) echo "  !! unknown RL scheduler '$rl' (expected rl_exec or rl_res) -- skipping" >&2; continue ;;
  esac

  rl_dir="$RESULTS/scx_$rl"
  echo "----- RL: $rl (objective file: $metric_file) -----"
  print_seed_ranking "$rl_dir" "$metric_file" "$best_pth"

  if [[ -n "$seed_override" ]]; then
    best_seed="$seed_override"; best_val="(forced)"
    echo "    using FORCED seed $best_seed"
  else
    sel="$(select_best_seed "$rl_dir" "$metric_file" "$best_pth")" || sel=""
    if [[ -z "$sel" ]]; then
      echo "  !! no usable seed/checkpoint under $rl_dir -- skipping $rl" >&2
      continue
    fi
    read -r best_seed best_val <<< "$sel"
    echo "    -> best seed = $best_seed  ($metric_file = $best_val)"
  fi

  model_dir="$rl_dir/seed_$best_seed"
  if [[ ! -f "$model_dir/$best_pth" ]]; then
    echo "  !! no checkpoint at $model_dir/$best_pth -- skipping $rl" >&2
    continue
  fi

  for trace in "${WORKLOADS[@]}"; do
    name="$(workload_name "$trace")"
    metrics="$rl_dir/${name}_metrics"
    queue="$rl_dir/${name}_queue"
    pertask="$rl_dir/${name}"
    log="$LOG_DIR/scx_${rl}__${name}.log"
    declare -a occ=()
    [[ -n "$OCCUPANCY" ]] && occ=(OCCUPANCY="$rl_dir/${name}_occupancy" "${OCC_TUNABLES[@]}")
    run_one "scx_$rl / $name (seed $best_seed)" "$metrics" "$log" \
      env SCHED="$rl" EVAL=1 MODEL_DIR="$model_dir" SEED="$best_seed" \
          METRICS="$metrics" QUEUELOG="$queue" PERTASK="$pertask" "${occ[@]}" \
          "$HERE/run_workload.sh" "$trace"
  done
done

# ---- compact comparison table over every metrics file touched by this run ----
build_summary() {
  local csv="$LOG_DIR/summary.csv"
  local current tmp
  current="$(mktemp)"
  tmp="$(mktemp)"

  {
    echo "scheduler,workload,tasks,acc_task_latency_s,acc_task_runtime_s,makespan_s,peak_queue_backlog"
    local sched name metrics rl
    for sched in "${BASELINE_SCHEDS[@]}"; do
      [[ -z "$sched" ]] && continue
      for trace in "${WORKLOADS[@]}"; do
        name="$(workload_name "$trace")"
        metrics="$RESULTS/scx_$sched/${name}_metrics"
        emit_metric_row "scx_$sched" "$name" "$metrics"
      done
    done
    for rl in "${RL_SCHEDS_ARR[@]}"; do
      [[ -z "$rl" ]] && continue
      for trace in "${WORKLOADS[@]}"; do
        name="$(workload_name "$trace")"
        metrics="$RESULTS/scx_$rl/${name}_metrics"
        emit_metric_row "scx_$rl" "$name" "$metrics"
      done
    done
  } > "$current"

  if [[ -s "$csv" ]]; then
    awk -F, '
      FNR == NR {
        if (FNR > 1) {
          key = $1 "," $2
          cur[key] = $0
          order[++n] = key
        }
        next
      }
      FNR == 1 {
        print
        next
      }
      {
        key = $1 "," $2
        if (!(key in cur)) {
          print
        }
      }
      END {
        for (i = 1; i <= n; i++) {
          print cur[order[i]]
        }
      }
    ' "$current" "$csv" > "$tmp"
    mv "$tmp" "$csv"
  else
    mv "$current" "$csv"
  fi
  rm -f "$current" "$tmp"
  restore_user_access "$csv"

  echo
  echo "=== summary table -> ${csv#$REPO_ROOT/} ==="
  column -t -s, "$csv" 2>/dev/null || cat "$csv"
}

# Parse one *_metrics file into a CSV row (blank fields if absent).
emit_metric_row() {
  local sched="$1" name="$2" path="$3"
  if [[ ! -s "$path" ]]; then
    echo "$sched,$name,,,,,"
    return
  fi
  awk -v s="$sched" -v w="$name" '
    /^tasks:/                     { tasks=$2 }
    /^accumulated task latency:/  { lat=$(NF-1) }
    /^accumulated task runtime:/  { run=$(NF-1) }
    /^makespan:/                  { mk=$(NF-1) }
    /^peak queue backlog:/        { pk=$4 }
    END { printf "%s,%s,%s,%s,%s,%s,%s\n", s, w, tasks, lat, run, mk, pk }
  ' "$path"
}

build_summary

echo
echo "=== sweep done: ${#OK[@]} ok, ${#FAIL[@]} failed, ${#SKIP[@]} skipped ==="
[[ ${#OK[@]}   -gt 0 ]] && printf '  ok:      %s\n' "${OK[*]}"
[[ ${#SKIP[@]} -gt 0 ]] && printf '  skipped: %s\n' "${SKIP[*]}"
[[ ${#FAIL[@]} -gt 0 ]] && printf '  FAILED:  %s\n' "${FAIL[*]}"
[[ ${#FAIL[@]} -eq 0 ]]
