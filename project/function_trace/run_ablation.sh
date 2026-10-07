#!/usr/bin/env bash
#
# run_ablation.sh -- rebalance-OFF ablation for scx_rl_res.
#
# NO_REBALANCE=1 skips the periodic RL rebalance migration pass entirely; the DQN
# still does all initial placement. Trains 60 episodes from scratch on
# trace_local.txt (seed 0) and writes its own CSV, so the rebalance-ON baseline
# (results/scx_rl_res/train_metrics.csv) is preserved.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/scx_rl_res"
EPISODES="${EPISODES:-60}"
OUT="$RES/train_metrics_norebalance.csv"

echo "===== rebalance OFF (NO_REBALANCE=1), $EPISODES episodes, current mechanism ====="
echo "      out: $OUT   (rebalance-ON baseline kept at $RES/train_metrics.csv)"
FRESH=1 NO_REBALANCE=1 METRICS_CSV="$OUT" "$HERE/rl_res.sh" train "$EPISODES"

echo "===== done -> $OUT ====="
echo "  sanity: the rebalance_moves column MUST be 0 (else NO_REBALANCE didn't take)."
echo "  compare explosions(>100k) and last-10 spread vs train_metrics.csv."
