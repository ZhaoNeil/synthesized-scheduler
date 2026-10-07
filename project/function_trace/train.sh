#!/usr/bin/env bash
#
# train.sh -- fresh-train scx_rl_exec then scx_rl_res back-to-back (one seed
# each), each writing results/scx_rl_<p>/train_metrics.csv: EPISODES train rows
# then TEST_REPS day02 test rows of the objective-best checkpoint.
#
# Usage:
#   sudo ./train.sh
#
# Env overrides go before ./train.sh, e.g.:
#   sudo SCHEDS=rl_exec ./train.sh                  # train one scheduler only
#   sudo EPISODES=100 ./train.sh                    # more training episodes
#   sudo EPISODES=5 TEST_REPS=2 ./train.sh          # quick smoke test
#   sudo TEST_TRACE=trace_test_noon_local.txt ./train.sh   # different test set
#   sudo SCHEDS=rl_exec EPISODES=80 SEED=2 ./train.sh      # combine
#
# Env:
#   SCHEDS       which schedulers, in order               (rl_exec rl_res)
#   EPISODES     training episodes per scheduler          (default 50)
#   TEST_REPS    test reps on the test trace              (default 5)
#   SEED         single training/eval seed                (default 0)
#   TRAIN_TRACE  training trace                           (trace_local.txt)
#   TEST_TRACE   test trace                               (trace_test_day02_local.txt)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "train.sh must run as root (it loads a sched_ext scheduler)." >&2
  exit 1
fi

EPISODES="${EPISODES:-50}"
TEST_REPS="${TEST_REPS:-5}"
SEED="${SEED:-0}"
TRAIN_TRACE="${TRAIN_TRACE:-trace_local.txt}"
TEST_TRACE="${TEST_TRACE:-trace_test_day02_local.txt}"
SCHEDS="${SCHEDS:-rl_exec rl_res}"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*"; }

log "=== train START ==="
log "    scheds: $SCHEDS | episodes: $EPISODES | test reps: $TEST_REPS | seed: $SEED"
log "    train trace: $TRAIN_TRACE | test trace: $TEST_TRACE"

for SCHED in $SCHEDS; do
  DRIVER="$HERE/$SCHED.sh"
  if [[ ! -x "$DRIVER" ]]; then
    log "!! missing driver $DRIVER -- skipping"
    continue
  fi
  log ""
  log "############################################################"
  log "## $SCHED : fresh train ($EPISODES ep) + self-test ($TEST_REPS reps)"
  log "############################################################"

  # FRESH=1 rewrites the checkpoints and the train_metrics.csv header from
  # scratch (a clean single-seed run). The driver runs the test phase itself.
  # Keep the checkpoint under results/scx_rl_<p>/seed_<SEED>/ (existing layout).
  RESULT_DIR="$(cd "$HERE/../.." && pwd)/results/scx_$SCHED"
  if FRESH=1 \
       SEED="$SEED" \
       MODEL_DIR="$RESULT_DIR/seed_$SEED" \
       TEST_TRACE="$TEST_TRACE" \
       TEST_REPS="$TEST_REPS" \
       "$DRIVER" train "$EPISODES" "$TRAIN_TRACE"; then
    log "<<< $SCHED OK"
  else
    log "!! $SCHED FAILED (rc=$?) -- continuing to next scheduler"
  fi
done

log ""
log "=== train DONE ==="
