#!/usr/bin/env bash
#
# rl_exec.sh -- driver for the scx_rl_exec reinforcement-learning scheduler.
#
# Wraps run_workload.sh (which loads the scheduler, replays a trace, and writes
# metrics) to make the two RL workflows convenient:
#
#   * train -- run N episodes on a training trace, continuing (resuming) the same
#     model across episodes, recording each episode's return + accumulated task
#     latency / runtime / makespan to results/scx_rl_exec/train_metrics.csv. (No
#     per-workload result files are written during training.)
#       sudo ./rl_exec.sh train [EPISODES] [TRACE] [-- replay args...]
#
#   * eval  -- load the trained best model (exec_best.pth) and run exactly one
#     episode on each of several workloads (greedy, no training), collecting the
#     per-workload metrics.
#       sudo ./rl_exec.sh eval [-- replay args...]
#
# One agent run (one trace replay) is one episode. The scheduler is loaded and
# unloaded per episode (that load/unload is the episode boundary -- the agent
# sends its terminal done=True transition on unload and checkpoints exec.pth /
# exec_best.pth).
#
# Tunables (environment variables):
#   EPISODES         train: number of episodes                       (default 50)
#   TRACE            train: training trace to replay        (trace_local.txt)
#   WORKLOADS        eval:  space-separated TEST trace files  (the *_local traces)
#   SEED             RNG seed for the whole run (same for every episode)  (default 0)
#   MODEL_DIR        checkpoint dir         (scheds/experimental/scx_rl_exec/py by default)
#   FRESH            train: 1 = delete exec.pth/exec_best.pth before episode 1 (0)
#   CPUS / WORKERS   passed through to run_workload.sh                  (1-49)
#   plus any scx_rl_exec knob run_workload.sh forwards: SLICE_MS, AGENT_CPU, ...
#
# Each episode replays the FULL trace (training and evaluation both need a
# complete episode). Examples:
#   sudo ./rl_exec.sh train 50                        # 50 episodes on trace_local.txt
#   sudo ./rl_exec.sh train 3                         # quick 3-episode test
#   sudo FRESH=1 ./rl_exec.sh train 50                # start a brand-new model
#   sudo ./rl_exec.sh eval                            # eval best model on TEST workloads
#
# For long (e.g. 50-episode) runs, detach from the terminal so closing it
# does not kill the in-flight run and evict the scheduler:
#   sudo setsid nohup ./rl_exec.sh train 50 >train.log 2>&1 &
#   tail -f train.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/scheds/experimental/scx_rl_exec/py}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/scx_rl_exec}"

CMD="${1:-}"
shift || true

usage() {
  echo "usage: sudo ./rl_exec.sh {train|eval} ..." >&2
  echo "  sudo ./rl_exec.sh train [EPISODES] [TRACE] [-- replay args...]" >&2
  echo "  sudo ./rl_exec.sh eval  [-- replay args...]   (env WORKLOADS=\"t1 t2 ...\")" >&2
  echo "  env: EPISODES TRACE WORKLOADS SEED MODEL_DIR FRESH CPUS WORKERS ..." >&2
}

case "$CMD" in
  train|eval|test) ;;
  *) usage; exit 1 ;;
esac

if [[ $EUID -ne 0 ]]; then
  echo "rl_exec.sh must run as root (it loads a sched_ext scheduler)." >&2
  exit 1
fi

# Collect any explicit replay-args after a literal `--` (advanced escape hatch;
# each episode replays the full trace by default).
REPLAY_ARGS=()
seen_ddash=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_ddash=1; continue; fi
  if [[ "$seen_ddash" == "1" ]]; then REPLAY_ARGS+=("$a"); fi
done

run_episode() {
  # run_episode <resume|eval> <trace> <log>
  # Runs one episode via run_workload.sh, tee-ing its output (live) to <log>, from
  # which we then parse the aggregate metrics and the episode return. The agent
  # prints "scx_rl_exec: episode return = X" (Rust, flushed even on SIGTERM
  # teardown); replay_trace prints the aggregate metrics to stdout.
  local mode="$1" trace="$2" log="$3"
  local -a envv=(SCHED=rl_exec MODEL_DIR="$MODEL_DIR")
  case "$mode" in
    # Training: suppress the per-workload result files (results/scx_rl_exec/local,
    # *_metrics, *_queue) -- the aggregate metrics print to stdout and go into the
    # consolidated CSV instead. (METRICS= etc. are SET-but-empty, which the harness
    # honors via ${VAR-default}.)
    resume) envv+=(RESUME=1 SEED="$SEED_BASE" METRICS= QUEUELOG= PERTASK= CORE_BACKLOG=) ;;
    eval)   envv+=(EVAL=1) ;;
    # test: greedy eval of the best checkpoint (EVAL=1 loads exec_best.pth) but,
    # like resume, suppress the per-workload result files so the only output is
    # the row we append to train_metrics.csv (phase=test).
    test)   envv+=(EVAL=1 SEED="$SEED_BASE" METRICS= QUEUELOG= PERTASK= CORE_BACKLOG=) ;;
  esac
  env "${envv[@]}" "$HERE/run_workload.sh" "$trace" "${REPLAY_ARGS[@]}" 2>&1 | tee "$log"
}

extract_return() {
  grep -oE 'episode return = -?[0-9.]+' "$1" | tail -1 | grep -oE '\-?[0-9.]+$' || true
}

# extract_metric <log> <label> -> the first number on the "label: <num> s" line.
extract_metric() {
  grep -m1 -E "^$2:" "$1" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || true
}

extract_rl_metric() {
  grep -oE "rl_metrics .*" "$1" | tail -1 | grep -oE "$2=[0-9]+" | tail -1 | cut -d= -f2 || true
}

# emit_row <phase> <episode> <log>: parse the aggregate metrics + rl_metrics
# from <log> and append one CSV row (phase = first column) to $METRICS_CSV,
# also echoing a human-readable one-liner. Shared by the train and test phases
# so both record the exact same columns. Uses SEED_BASE / METRICS_CSV globals.
emit_row() {
  local phase="$1" ep="$2" log="$3"
  local ret lat run mk preempt place_batches place_tasks place_dqn_actions \
    place_direct dispatched_tasks ring_full slow_ticks qlen_peak qlen_acc qlen_mean \
    slice_dqn_updates slice_rtc_ticks slice_other_updates \
    slice_ms_8 slice_ms_32 slice_ms_174 slice_ms_390 slice_ms_1633 rebalance_moves
  ret="$(extract_return "$log")"
  lat="$(extract_metric "$log" 'accumulated task latency')"
  run="$(extract_metric "$log" 'accumulated task runtime')"
  mk="$(extract_metric "$log" 'makespan')"
  preempt="$(extract_rl_metric "$log" preempt)"
  place_batches="$(extract_rl_metric "$log" place_batches)"
  place_tasks="$(extract_rl_metric "$log" place_tasks)"
  place_dqn_actions="$(extract_rl_metric "$log" place_dqn_actions)"
  place_direct="$(extract_rl_metric "$log" place_direct)"
  dispatched_tasks="$(extract_rl_metric "$log" dispatched_tasks)"
  ring_full="$(extract_rl_metric "$log" ring_full)"
  slow_ticks="$(extract_rl_metric "$log" slow_ticks)"
  qlen_peak="$(extract_rl_metric "$log" qlen_peak)"
  qlen_acc="$(extract_rl_metric "$log" qlen_acc)"
  # Mean DSQ backlog per slow tick (float); blank if we lack a clean sample.
  qlen_mean="$(awk -v a="${qlen_acc:-}" -v t="${slow_ticks:-}" \
    'BEGIN { if (a != "" && t ~ /^[0-9]+$/ && t > 0) printf "%.2f", a / t }')"
  slice_dqn_updates="$(extract_rl_metric "$log" slice_dqn_updates)"
  slice_rtc_ticks="$(extract_rl_metric "$log" slice_rtc_ticks)"
  slice_other_updates="$(extract_rl_metric "$log" slice_other_updates)"
  slice_ms_8="$(extract_rl_metric "$log" slice_ms_8)"
  slice_ms_32="$(extract_rl_metric "$log" slice_ms_32)"
  slice_ms_174="$(extract_rl_metric "$log" slice_ms_174)"
  slice_ms_390="$(extract_rl_metric "$log" slice_ms_390)"
  slice_ms_1633="$(extract_rl_metric "$log" slice_ms_1633)"
  rebalance_moves="$(extract_rl_metric "$log" rebalance_moves)"
  echo "[$phase] episode $ep: return=${ret:-NA} latency=${lat:-NA}s runtime=${run:-NA}s makespan=${mk:-NA}s preempt=${preempt:-NA} slices=[8:${slice_ms_8:-NA},32:${slice_ms_32:-NA},174:${slice_ms_174:-NA},390:${slice_ms_390:-NA},1633:${slice_ms_1633:-NA}]"
  echo "$phase,$ep,$SEED_BASE,${ret:-NA},${lat:-NA},${run:-NA},${mk:-NA},${preempt:-NA},${place_batches:-NA},${place_tasks:-NA},${place_dqn_actions:-NA},${place_direct:-NA},${dispatched_tasks:-NA},${ring_full:-NA},${slow_ticks:-NA},${qlen_peak:-NA},${qlen_mean:-NA},${slice_dqn_updates:-NA},${slice_rtc_ticks:-NA},${slice_other_updates:-NA},${slice_ms_8:-NA},${slice_ms_32:-NA},${slice_ms_174:-NA},${slice_ms_390:-NA},${slice_ms_1633:-NA},${rebalance_moves:-NA},$(date -Iseconds)" >> "$METRICS_CSV"
}

case "$CMD" in
  train)
    EPISODES="${1:-${EPISODES:-50}}"
    TRACE="${2:-${TRACE:-trace_local.txt}}"
    # One seed for the whole training run -- every episode in this run uses it.
    SEED_BASE="${SEED:-0}"

    mkdir -p "$RESULT_DIR"
    METRICS_CSV="${METRICS_CSV:-$RESULT_DIR/train_metrics.csv}"
    FRESH_RUN=0

    if [[ "${FRESH:-0}" == "1" ]]; then
      echo "FRESH=1: removing existing checkpoints in $MODEL_DIR"
      rm -f "$MODEL_DIR/exec.pth" "$MODEL_DIR/exec_best.pth"
      FRESH_RUN=1
    fi
    if [[ ! -f "$MODEL_DIR/exec.pth" ]]; then
      FRESH_RUN=1
    fi
    if [[ ! -f "$METRICS_CSV" || ("$FRESH_RUN" == "1" && "${METRICS_APPEND:-0}" != "1") ]]; then
      echo "phase,episode,seed,return,acc_task_latency_s,acc_task_runtime_s,makespan_s,preempt,place_batches,place_tasks,place_dqn_actions,place_direct,dispatched_tasks,ring_full,slow_ticks,qlen_peak,qlen_mean,slice_dqn_updates,slice_rtc_ticks,slice_other_updates,slice_ms_8,slice_ms_32,slice_ms_174,slice_ms_390,slice_ms_1633,rebalance_moves,timestamp" > "$METRICS_CSV"
    fi

    echo "=== scx_rl_exec train: $EPISODES episode(s) on $TRACE (seed $SEED_BASE) ==="
    echo "    model dir: $MODEL_DIR   per-episode metrics -> $METRICS_CSV"

    # Best-checkpoint selection by the objective metric (lowest first-run ->
    # completion runtime), not by episodic return: the reward is only a proxy,
    # so a return-based exec_best.pth can freeze a checkpoint that isn't best on
    # the metric. The agent only sees `return`, so promotion lives here where the
    # post-hoc runtime is available. Persist the running best across resumes.
    BEST_RUN_FILE="$MODEL_DIR/.exec_best_runtime"
    best_run=""
    if [[ "$FRESH_RUN" == "1" ]]; then
      rm -f "$BEST_RUN_FILE"
    elif [[ -f "$BEST_RUN_FILE" ]]; then
      best_run="$(cat "$BEST_RUN_FILE" 2>/dev/null || true)"
    fi

    tmp="$(mktemp)"
    for (( ep=1; ep<=EPISODES; ep++ )); do
      echo
      echo "----- episode $ep / $EPISODES -----"
      run_episode resume "$TRACE" "$tmp"
      run="$(extract_metric "$tmp" 'accumulated task runtime')"
      # Promote exec.pth -> exec_best.pth on a new lowest first-run -> completion runtime.
      if [[ "$run" =~ ^[0-9.]+$ ]]; then
        if [[ -z "$best_run" ]] || awk -v a="$run" -v b="$best_run" 'BEGIN{exit !(a<b)}'; then
          best_run="$run"
          if cp -f "$MODEL_DIR/exec.pth" "$MODEL_DIR/exec_best.pth" 2>/dev/null; then
            echo "    new best runtime=${run}s -> exec_best.pth"
            echo "$best_run" > "$BEST_RUN_FILE"
          fi
        fi
      fi
      emit_row train "$ep" "$tmp"
    done

    # ---- test phase: greedy eval of the best checkpoint on TEST_TRACE ----
    # Appends TEST_REPS rows (phase=test, episodes EPISODES+1 .. EPISODES+REPS)
    # to the SAME train_metrics.csv, loading the objective-best exec_best.pth.
    # Disable by setting TEST_REPS=0 (e.g. multi-seed batch training).
    TEST_TRACE="${TEST_TRACE:-trace_test_day02_local.txt}"
    TEST_REPS="${TEST_REPS:-5}"
    if [[ -n "$TEST_TRACE" && "$TEST_REPS" -gt 0 ]]; then
      if [[ -f "$MODEL_DIR/exec_best.pth" ]]; then
        echo
        echo "=== test phase: $TEST_REPS rep(s) on $TEST_TRACE (best: exec_best.pth) ==="
        for (( t=1; t<=TEST_REPS; t++ )); do
          ep=$((EPISODES + t))
          echo
          echo "----- test rep $t / $TEST_REPS (episode $ep) -----"
          run_episode test "$TEST_TRACE" "$tmp"
          emit_row test "$ep" "$tmp"
        done
      else
        echo "!! no exec_best.pth -- skipping test phase"
      fi
    fi

    rm -f "$tmp"
    echo
    echo "=== done. per-episode metrics -> $METRICS_CSV ==="
    echo "    checkpoints: $MODEL_DIR/exec.pth (latest), exec_best.pth (best runtime)"
    ;;

  eval|test)
    if [[ ! -f "$MODEL_DIR/exec_best.pth" ]]; then
      echo "No trained model at $MODEL_DIR/exec_best.pth -- train first (./rl_exec.sh train)." >&2
      exit 1
    fi
    WORKLOADS="${WORKLOADS:-trace_test_day02_local.txt trace_test_noon_local.txt trace_test_day10_local.txt trace_test_day02_hetero_local.txt}"

    mkdir -p "$RESULT_DIR"
    LOGDIR="$RESULT_DIR/eval_logs"
    mkdir -p "$LOGDIR"

    echo "=== scx_rl_exec eval: model $MODEL_DIR/exec_best.pth ==="
    echo "    workloads: $WORKLOADS"
    echo "    replay args: ${REPLAY_ARGS[*]:-<none>}"
    declare -a done_metrics=()
    for trace in $WORKLOADS; do
      name="$(basename "$trace")"; name="${name%.*}"; name="${name#trace_}"; name="${name%_local}"
      echo
      echo "----- eval on $name ($trace) -----"
      run_episode eval "$trace" "$LOGDIR/${name}.log" >/dev/null
      done_metrics+=("$RESULT_DIR/${name}_metrics")
    done

    echo
    echo "=== eval metrics summary ==="
    for m in "${done_metrics[@]}"; do
      echo "----- $m -----"
      [[ -f "$m" ]] && cat "$m" || echo "(missing)"
    done
    ;;

  *)
    usage; exit 1
    ;;
esac
