#!/usr/bin/env bash
#
# Run a function-arrival trace under a user-space scheduler, selected with the
# SCHED environment variable. Supported values:
#   * cfifo   - scx_cfifo: one centralized global queue, first idle core pulls;
#   * alps    - scx_alps: ALPS (ATC'24) learned per-class SRPT priority on per-CPU CFS;
#   * sfs     - scx_sfs: SFS (SC'22) two-level FILTER(FIFO,adaptive slice)+CFS, SRTF approx;
#   * rl_res  - scx_rl_res: RL (Dueling-DQN) agent optimizing total task response
#               time; learns per-task core placement + the global round-robin time
#               slice (embeds PyTorch q_learning.py);
#   * rl_exec - scx_rl_exec: same agent, reward optimizes total task EXECUTION time
#               (accumulated runtime) instead of response time;
#   * cfs     - scx_cfs: per-CPU vruntime run-queues, idle-first placement +
#               load balancing (a CFS-style scheduler modelled on Linux CFS);
#   * eevdf   - scx_eevdf: per-CPU EEVDF (eligibility + virtual deadlines), the
#               policy that replaced CFS in Linux 6.6;
#   * hybrid  - scx_hybrid: FIFO tier (CPUs 1-24) for short tasks + CFS tier
#               (CPUs 25-49); tasks past the on-CPU-time threshold migrate to CFS;
#   * rr   - scx_rr: one FIFO queue per worker core, round-robin placement;
#   * preempt - RR + preemption (fixed slice, requeue at local tail);
#   * p2c     - RR + power-of-two random choices (join the shorter queue);
#   * shuffle - RR + work shuffling (periodic most->least-loaded migration);
#   * preempt_shuffle - RR + preemption AND work shuffling combined.
# The last four are the scx_rr binary with --mechanism; add more by extending
# the `case "$SCHED"` block below.
#
# It (1) loads the selected scheduler, (2) replays the trace as one process per
# task pinned to the worker CPUs, (3) writes aggregate metrics from the per-task
# data, and (4) unloads the scheduler when done. Results go to
# results/<scheduler>/ (scx_cfifo, scx_rr, scx_rr_preempt, ...).
#
# Usage:
#   sudo ./run_workload.sh [TRACE] [options passed to replay_trace]
#   sudo SCHED=p2c ./run_workload.sh [TRACE] [...]
#
# Examples:
#   sudo ./run_workload.sh trace_test_day02_hetero_local.txt
#   sudo SCHED=shuffle ./run_workload.sh trace_test_day02_hetero_local.txt
#   sudo SCHED=preempt CPUS=0-49 ./run_workload.sh trace_local.txt --max 1000
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

TRACE="${1:-trace_test_day02_hetero_local.txt}"
shift || true

# Which scheduler to run: cfifo (centralized FIFO), alps (learned SRPT priority),
# sfs (two-level SRTF approx),
# cfs (per-CPU CFS), eevdf (per-CPU EEVDF), hybrid (FIFO + CFS two-tier), or rr
# (distributed per-core FIFO, with optional load-balancing mechanisms).
SCHED="${SCHED:-cfifo}"

# Set by per-scheduler cases that maintain per-core run-queues (RR family,
# per-CPU CFS); enables the per-core backlog summary below.
HAS_PERCORE=0

# Substring the active scheduler's ops name must contain, exported to the task
# launcher (launch_function) so it refuses to run on the wrong scheduler. Set
# per-case below.
EXPECT_OPS=""

DRIVER_CPU="${DRIVER_CPU:-63}"     # keep the replay driver off the worker cores
SCX_TIMEOUT_MS="${SCX_TIMEOUT_MS:-600000}"
export SCX_TIMEOUT_MS              # runnable-task stall timeout for the scheduler

# Per-scheduler configuration: crate/binary name, default worker affinity, and
# the scheduler-specific command-line arguments.
case "$SCHED" in
  cfifo)
    CRATE="scx_cfifo"
    EXPECT_OPS="cfifo"
    CPUS="${CPUS:-0-49}"           # task affinity = all 50 worker cores 0-49.
                                   # The agent is a light, unpinned CFS process
                                   # (no dedicated/reserved core needed).
    SCHED_ARGS=()
    ;;
  cfs)
    # scx_cfs: per-CPU vruntime run-queues, idle-first placement and periodic
    # load balancing -- a CFS-style scheduler modelled on the real Linux CFS.
    # Laid out like the RR family (per-core queues, agent time-shares a worker).
    CRATE="scx_cfs"
    EXPECT_OPS="cfs"
    CPUS="${CPUS:-0-49}"           # task affinity = the 50 worker cores 0-49;
                                   # no agent core is reserved (the agent
                                   # time-shares a worker)
    WORKERS="${WORKERS:-$CPUS}"    # per-CPU run-queues placed over this set; keep
                                   # it equal to the task affinity so placement
                                   # never lands a task on a CPU it isn't allowed on
    SCHED_ARGS=(--workers "$WORKERS")
    # (per-core backlog left off by default; force with CORE_BACKLOG=path)
    # Optional CFS tuning passthrough (us / ms): SLICE_US == sched_latency,
    # MIN_SLICE_US == minimum granularity, LB_INTERVAL_MS == load-balance period.
    [[ -n "${SLICE_US:-}" ]] && SCHED_ARGS+=(-s "$SLICE_US")
    [[ -n "${MIN_SLICE_US:-}" ]] && SCHED_ARGS+=(-S "$MIN_SLICE_US")
    [[ -n "${LB_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--lb-interval-ms "$LB_INTERVAL_MS")
    ;;
  alps)
    # scx_alps: ALPS (USENIX ATC'24) ported to the agent model -- per-CPU CFS
    # dispatch where each task's weight comes from its function class's learned
    # SRPT rank (shorter-predicted classes get more CPU). Same per-CPU layout as
    # scx_cfs. (Reads /proc to learn function classes; /proc is in-memory, so it
    # cannot block on an ext4 inode lock.)
    CRATE="scx_alps"
    EXPECT_OPS="alps"
    CPUS="${CPUS:-0-49}"           # task affinity = the 50 worker cores 0-49;
                                   # no agent core is reserved (the agent
                                   # time-shares a worker)
    WORKERS="${WORKERS:-$CPUS}"    # per-CPU run-queues placed over this set; keep
                                   # it equal to the task affinity so placement
                                   # never lands a task on a CPU it isn't allowed on
    SCHED_ARGS=(--workers "$WORKERS")
    # (per-core backlog left off by default; force with CORE_BACKLOG=path)
    # Optional ALPS tuning passthrough (us / ms): SLICE_US == preemption quantum,
    # SCAN_INTERVAL_MS == learning cadence, LB_INTERVAL_MS == load-balance period.
    [[ -n "${SLICE_US:-}" ]] && SCHED_ARGS+=(-s "$SLICE_US")
    [[ -n "${SCAN_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--scan-interval-ms "$SCAN_INTERVAL_MS")
    [[ -n "${LB_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--lb-interval-ms "$LB_INTERVAL_MS")
    ;;
  sfs)
    # scx_sfs: SFS (SC'22) ported to the agent model -- two-level MLFQ. Every
    # function starts in a high-priority FILTER level (FIFO, adaptive slice
    # S=mean_IAT*cpus); overrunning S or blocking on I/O demotes it to a low-
    # priority CFS level. Short functions thus preempt long ones (SRTF approx).
    # Same per-CPU layout as scx_cfs.
    CRATE="scx_sfs"
    EXPECT_OPS="sfs"
    CPUS="${CPUS:-0-49}"           # task affinity = the 50 worker cores 0-49;
                                   # no agent core is reserved (the agent
                                   # time-shares a worker)
    WORKERS="${WORKERS:-$CPUS}"    # per-CPU run-queues placed over this set; keep
                                   # it equal to the task affinity so placement
                                   # never lands a task on a CPU it isn't allowed on
    SCHED_ARGS=(--workers "$WORKERS")
    # (per-core backlog left off by default; force with CORE_BACKLOG=path)
    # Optional SFS tuning passthrough (ms): DEFAULT_SLICE_MS == initial FILTER
    # slice S, CFS_SLICE_MS == Level-2 slice, FILTER_WINDOW == IAT window,
    # LB_INTERVAL_MS == load-balance period.
    [[ -n "${DEFAULT_SLICE_MS:-}" ]] && SCHED_ARGS+=(-s "$DEFAULT_SLICE_MS")
    [[ -n "${CFS_SLICE_MS:-}" ]] && SCHED_ARGS+=(-c "$CFS_SLICE_MS")
    [[ -n "${FILTER_WINDOW:-}" ]] && SCHED_ARGS+=(--filter-window "$FILTER_WINDOW")
    [[ -n "${LB_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--lb-interval-ms "$LB_INTERVAL_MS")
    ;;
  rl_res|rl_exec)
    # scx_rl_res / scx_rl_exec: RL (Dueling-DQN) schedulers (same agent, different
    # reward/policy in py/q_learning.py): rl_res optimizes total task RESPONSE time
    # (accumulated task latency), rl_exec optimizes total task EXECUTION time
    # (accumulated runtime). Ported from the ghOSt RL port. The agent embeds
    # PyTorch and learns (a) which core to place each newly-runnable task on and
    # (b) the global round-robin time slice. Tasks run FIFO per core, preempted at
    # the learned slice. The DQN's action space is all 50 cores 0-49; the agent
    # itself runs pinned on a core OUTSIDE the worker set (AGENT_CPU, default 50)
    # so its torch inference/training never steals a scheduled task's CPU.
    CRATE="scx_$SCHED"             # scx_rl_res or scx_rl_exec
    EXPECT_OPS="$SCHED"            # rl_res or rl_exec
    CPUS="${CPUS:-0-49}"           # task affinity = all 50 worker cores 0-49
    WORKERS="${WORKERS:-$CPUS}"    # per-core FIFO run-queues over this set; keep
                                   # it equal to the task affinity so placement
                                   # never lands a task on a CPU it isn't allowed on
    AGENT_CPU="${AGENT_CPU:-50}"   # the agent's own (pinned) core, outside 0-49
    SCHED_ARGS=(--workers "$WORKERS" --agent-cpu "$AGENT_CPU")
    # Optional RL tuning passthrough: SLICE_MS == initial RR slice (ms),
    # MODEL_DIR == checkpoint dir, SEED == agent RNG seed, EVAL=1 == load
    # res_best.pth and act greedily (no online training).
    [[ -n "${SLICE_MS:-}" ]] && SCHED_ARGS+=(-s "$SLICE_MS")
    [[ -n "${MODEL_DIR:-}" ]] && SCHED_ARGS+=(--model-dir "$MODEL_DIR")
    [[ -n "${SEED:-}" ]] && SCHED_ARGS+=(--seed "$SEED")
    [[ "${EVAL:-0}" == "1" ]] && SCHED_ARGS+=(--eval)
    [[ "${RESUME:-0}" == "1" ]] && SCHED_ARGS+=(--resume)
    [[ "${NO_REBALANCE:-0}" == "1" ]] && SCHED_ARGS+=(--no-rebalance)
    ;;
  eevdf)
    # scx_eevdf: per-CPU EEVDF (Earliest Eligible Virtual Deadline First) -- the
    # policy that replaced CFS in Linux 6.6. Same per-CPU layout as the RR
    # family / scx_cfs (per-core queues, agent time-shares a worker).
    CRATE="scx_eevdf"
    EXPECT_OPS="eevdf"
    CPUS="${CPUS:-0-49}"           # task affinity = the 50 worker cores 0-49;
                                   # no agent core is reserved (the agent
                                   # time-shares a worker)
    WORKERS="${WORKERS:-$CPUS}"    # per-CPU run-queues placed over this set; keep
                                   # it equal to the task affinity so placement
                                   # never lands a task on a CPU it isn't allowed on
    SCHED_ARGS=(--workers "$WORKERS")
    # (per-core backlog left off by default; force with CORE_BACKLOG=path)
    # Optional EEVDF tuning passthrough (us / ms): BASE_SLICE_US == request slice,
    # LAG_BAND_US == eligibility band, LB_INTERVAL_MS == load-balance period.
    [[ -n "${BASE_SLICE_US:-}" ]] && SCHED_ARGS+=(-s "$BASE_SLICE_US")
    [[ -n "${LAG_BAND_US:-}" ]] && SCHED_ARGS+=(-b "$LAG_BAND_US")
    [[ -n "${LB_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--lb-interval-ms "$LB_INTERVAL_MS")
    ;;
  hybrid)
    # scx_hybrid: two tiers. FIFO partition (CPUs 1-24, CPU 0 = global agent) runs
    # new/short tasks; a task whose on-CPU time crosses the preemption threshold
    # is migrated to the CFS partition (CPUs 25-49) for weighted-fair scheduling.
    CRATE="scx_hybrid"
    EXPECT_OPS="hybrid"
    CPUS="${CPUS:-1-49}"           # task affinity = both tiers (1-24 FIFO +
                                   # 25-49 CFS); CPU 0 is left free as the global
                                   # agent (tasks may migrate FIFO->CFS, so they
                                   # must be allowed on the union of both tiers)
    FIFO_CPUS="${FIFO_CPUS:-1-24}"
    CFS_CPUS="${CFS_CPUS:-25-49}"
    PREEMPT_SLICE_MS="${PREEMPT_SLICE_MS:-1200}"
    CFS_SLICE_MS="${CFS_SLICE_MS:-6}"
    SCHED_ARGS=(--fifo-cpus "$FIFO_CPUS" --cfs-cpus "$CFS_CPUS")
    # Optional tuning passthrough (ms): PREEMPT_SLICE_MS == on-CPU-time threshold
    # / FIFO slice, CFS_SLICE_MS == CFS-tier slice, LB_INTERVAL_MS.
    SCHED_ARGS+=(-p "$PREEMPT_SLICE_MS")
    SCHED_ARGS+=(-c "$CFS_SLICE_MS")
    [[ -n "${LB_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--lb-interval-ms "$LB_INTERVAL_MS")
    ;;
  rr|preempt|p2c|shuffle|preempt_shuffle)
    # All are the same scx_rr binary; the mechanism is a flag.
    CRATE="scx_rr"
    EXPECT_OPS="rr"
    CPUS="${CPUS:-0-49}"           # task affinity = the 50 worker cores 0-49;
                                   # no agent core is reserved (the agent
                                   # time-shares a worker)
    WORKERS="${WORKERS:-$CPUS}"    # per-core queues placed over this set; keep it
                                   # equal to the task affinity so placement never
                                   # lands a task on a CPU it isn't allowed on
    case "$SCHED" in
      rr)             MECHANISM="none"            ;;  # plain RR, run-to-completion
      preempt)        MECHANISM="preempt"         ;;  # RR + fixed slice, requeue local
      p2c)            MECHANISM="p2c"             ;;  # power-of-two random choices
      shuffle)        MECHANISM="shuffle"         ;;  # RR + periodic work shuffling
      preempt_shuffle) MECHANISM="preempt-shuffle" ;; # fixed slice + work shuffling
    esac
    SCHED_ARGS=(--workers "$WORKERS" --mechanism "$MECHANISM")
    # Optional tuning passthrough. PREEMPT_SLICE_MS = per-task slice (preempt /
    # preempt_shuffle); SHUFFLE_* tune the periodic rebalance (shuffle /
    # preempt_shuffle). Unset => the binary's built-in defaults.
    [[ -n "${PREEMPT_SLICE_MS:-}" ]]    && SCHED_ARGS+=(--preempt-slice-ms "$PREEMPT_SLICE_MS")
    [[ -n "${SHUFFLE_INTERVAL_MS:-}" ]] && SCHED_ARGS+=(--shuffle-interval-ms "$SHUFFLE_INTERVAL_MS")
    [[ -n "${SHUFFLE_MIN_QLEN:-}" ]]    && SCHED_ARGS+=(--shuffle-min-qlen "$SHUFFLE_MIN_QLEN")
    [[ -n "${SHUFFLE_FRAC:-}" ]]        && SCHED_ARGS+=(--shuffle-frac "$SHUFFLE_FRAC")
    HAS_PERCORE=1                  # per-core queues => per-core backlog is meaningful
    # Plain RR keeps the base results dir; each mechanism gets a suffixed one
    # (results/scx_rr, results/scx_rr_preempt, ...).
    if [[ "$SCHED" == "rr" ]]; then
      SCHED_NAME_DEFAULT="scx_rr"
    else
      SCHED_NAME_DEFAULT="scx_rr_${SCHED}"
    fi
    ;;
  *)
    echo "Unknown SCHED='$SCHED' (expected: cfifo, alps, sfs, rl_res, rl_exec, cfs, eevdf, hybrid, rr, preempt, p2c, shuffle, preempt_shuffle)." >&2
    exit 1
    ;;
esac

# Pin the task launcher's scheduler check to the scheduler we load (see
# launch_function.cc:VerifyScxActive). Exported so it reaches replay_trace's
# fork+exec'd launcher children.
export SCX_EXPECT_OPS="$EXPECT_OPS"

SCX_BIN="${SCX_BIN:-$REPO_ROOT/target/release/$CRATE}"

# Scheduler name and workload name, used to place results under
# results/<scheduler>/<workload>. The workload name is the trace file name with
# its directory, extension, leading "trace_" and trailing "_local" removed
# (e.g. trace_test_day02_hetero_local.txt -> test_day02_hetero).
SCHED_NAME="${SCHED_NAME:-${SCHED_NAME_DEFAULT:-$CRATE}}"
WORKLOAD="$(basename "$TRACE")"; WORKLOAD="${WORKLOAD%.*}"
WORKLOAD="${WORKLOAD#trace_}"; WORKLOAD="${WORKLOAD%_local}"
RESULT_DIR="$REPO_ROOT/results/$SCHED_NAME"

# Aggregate metrics, printed by replay_trace from its in-memory per-task metric
# array (no log file is written during the run -- each task records its own
# timestamps into a private shared-memory slot, so there is no shared-file lock
# to contend on or stall the scheduler).
METRICS="${METRICS-$RESULT_DIR/${WORKLOAD}_metrics}"
# True queue backlog over time (tasks waiting for a CPU), written by replay_trace
# from the same per-task data. Set QUEUELOG= to disable.
QUEUELOG="${QUEUELOG-$RESULT_DIR/${WORKLOAD}_queue}"
# Per-core backlog summary (one row per worker core: tasks, peak and mean queue
# depth), written by replay_trace from each task's first-run CPU. Only meaningful
# where tasks have per-core run-queues, so default it on for the RR family
# (see HAS_PERCORE). scx_cfs/scx_eevdf also have per-core queues but leave it off
# by default; pass CORE_BACKLOG=path to force it on for any scheduler.
if [[ "$HAS_PERCORE" == "1" ]]; then
  CORE_BACKLOG="${CORE_BACKLOG-$RESULT_DIR/${WORKLOAD}_core_backlog}"
else
  CORE_BACKLOG="${CORE_BACKLOG-}"
fi
# Per-task TaskNew/FirstRun/TaskDead lifecycle log, dumped from the in-memory
# array after the run (same format as results/scx_cfifo/<workload>) -- the source
# for the per-task CDF/violin and throughput plots. Set PERTASK= to disable.
PERTASK="${PERTASK-$RESULT_DIR/${WORKLOAD}}"
# Per-core occupancy: each task's (core, start, end) intervals, including mid-run
# core changes (migrations), which each task samples itself via a per-task
# CPU-time timer (no kernel trace; the fib workload is untouched). OFF by default
# -- it adds a periodic signal to every task -- enable with OCCUPANCY=1 (writes
# results/<sched>/<workload>_occupancy) or OCCUPANCY=<path>. Tunables (read from
# the environment by replay_trace / launch_function): OCCUPANCY_PERIOD_US =
# sampling period in on-CPU microseconds (default 1000 = 1ms); OCCUPANCY_MAX_SEG =
# ring size, i.e. max core-changes recorded per task (default 256).
# Note: a task only observes its CPU while running, so this captures core CHANGES,
# not preemption gaps where it is descheduled and resumes on the SAME core.
if [[ -n "${OCCUPANCY:-}" ]]; then
  if [[ "$OCCUPANCY" == "1" ]]; then
    OCCUPANCY="$RESULT_DIR/${WORKLOAD}_occupancy"
  fi
  # Propagate the tunables to replay_trace / launch_function across exec (naming
  # an unset var with export is fine under set -u; the C side treats empty as
  # "use the default").
  export OCCUPANCY_PERIOD_US OCCUPANCY_MAX_SEG
else
  OCCUPANCY=""
fi

if [[ $EUID -ne 0 ]]; then
  echo "This script must run as root (loads a sched_ext scheduler)." >&2
  exit 1
fi

if [[ "$(cat /sys/kernel/sched_ext/state 2>/dev/null)" == "enabled" ]]; then
  echo "A sched_ext scheduler is already running: $(cat /sys/kernel/sched_ext/root/ops 2>/dev/null)" >&2
  echo "Stop the existing scheduler before starting a new workload." >&2
  exit 1
fi

# Build the workload binaries if needed.
if [[ ! -x "$HERE/launch_function" ]]; then
  echo "Building launch_function..."; g++ -O2 -o "$HERE/launch_function" "$HERE/launch_function.cc"
fi
if [[ ! -x "$HERE/replay_trace" ]]; then
  echo "Building replay_trace..."; g++ -O2 -o "$HERE/replay_trace" "$HERE/replay_trace.cc"
fi
if [[ ! -x "$SCX_BIN" ]]; then
  echo "Building $CRATE (release)..."
  ( cd "$REPO_ROOT" && cargo build --release -p "$CRATE" )
fi

if [[ "$SCHED" == "rl_res" || "$SCHED" == "rl_exec" ]]; then
  RL_PY_DIR="$REPO_ROOT/scheds/experimental/$CRATE/py"
  rm -rf "$RL_PY_DIR/__pycache__"
fi

# Per-task metrics are collected in shared memory (no file) and written out by
# replay_trace at the end: aggregates -> $METRICS, backlog series -> $QUEUELOG.
REPLAY_METRIC_ARGS=(--metrics)
if [[ -n "$METRICS" ]]; then
  mkdir -p "$(dirname "$METRICS")"
fi
if [[ -n "$QUEUELOG" ]]; then
  mkdir -p "$(dirname "$QUEUELOG")"
  REPLAY_METRIC_ARGS+=(--queue-series "$QUEUELOG")
fi
if [[ -n "$CORE_BACKLOG" ]]; then
  mkdir -p "$(dirname "$CORE_BACKLOG")"
  REPLAY_METRIC_ARGS+=(--core-backlog "$CORE_BACKLOG")
fi
if [[ -n "$PERTASK" ]]; then
  mkdir -p "$(dirname "$PERTASK")"
  REPLAY_METRIC_ARGS+=(--per-task "$PERTASK")
fi
if [[ -n "$OCCUPANCY" ]]; then
  mkdir -p "$(dirname "$OCCUPANCY")"
  REPLAY_METRIC_ARGS+=(--occupancy "$OCCUPANCY")
fi

# Load the scheduler in the background.
echo "Loading $CRATE ${SCHED_ARGS[*]} (timeout ${SCX_TIMEOUT_MS}ms)..."
"$SCX_BIN" "${SCHED_ARGS[@]}" &
SCX_PID=$!

cleanup() {
  echo "Stopping $CRATE..."
  kill "$SCX_PID" 2>/dev/null || true
  # Wait up to ~60s for a clean exit. On heavy/oversubscribed traces the kernel's
  # scheduler-unregister (struct_ops detach: kick every CPU + RCU quiesce) can
  # take ~4-5s for *any* sched_ext scheduler while the box is still saturated.
  # The RL agent additionally runs finish_episode (terminal train_step + torch
  # checkpoint) before detaching, which is slow on a saturated box; killing it
  # early would lose the episode return / rl_metrics line.
  for _ in $(seq 1 600); do
    kill -0 "$SCX_PID" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$SCX_PID" 2>/dev/null; then
    echo "$CRATE did not stop promptly after 60s; killing it..." >&2
    kill -KILL "$SCX_PID" 2>/dev/null || true
  fi
  wait "$SCX_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until the scheduler is actually attached.
for _ in $(seq 1 50); do
  if ! kill -0 "$SCX_PID" 2>/dev/null; then
    echo "$CRATE exited before attaching." >&2
    exit 1
  fi
  [[ "$(cat /sys/kernel/sched_ext/state 2>/dev/null)" == "enabled" ]] && break
  sleep 0.1
done
if [[ "$(cat /sys/kernel/sched_ext/state 2>/dev/null)" != "enabled" ]]; then
  echo "$CRATE failed to load." >&2
  exit 1
fi
echo "$CRATE loaded: $(cat /sys/kernel/sched_ext/root/ops 2>/dev/null)"

# Replay the trace (driver pinned to DRIVER_CPU; tasks pinned to $CPUS).
# replay_trace prints the aggregate metrics to stdout (diagnostics go to stderr),
# so we capture stdout into $METRICS.
echo "Replaying $TRACE (tasks on CPUs $CPUS)..."
if [[ -n "$METRICS" ]]; then
  taskset -c "$DRIVER_CPU" \
    "$HERE/replay_trace" "$HERE/$TRACE" \
      --launcher "$HERE/launch_function" \
      --cpus "$CPUS" "${REPLAY_METRIC_ARGS[@]}" "$@" > "$METRICS"
  cat "$METRICS"
  echo "Metrics: $METRICS"
  [[ -n "$QUEUELOG" ]] && echo "Queue backlog: $QUEUELOG"
  [[ -n "$CORE_BACKLOG" ]] && echo "Per-core backlog: $CORE_BACKLOG"
  [[ -n "$OCCUPANCY" ]] && echo "Per-core occupancy: $OCCUPANCY"
else
  taskset -c "$DRIVER_CPU" \
    "$HERE/replay_trace" "$HERE/$TRACE" \
      --launcher "$HERE/launch_function" \
      --cpus "$CPUS" "${REPLAY_METRIC_ARGS[@]}" "$@"
fi

echo "Done."
