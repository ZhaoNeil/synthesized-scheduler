import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- per-scheduler config: which secondary metric to plot and how to label it ---
VARIANTS = {
    "exec": {
        "results_dir": "scx_rl_exec",
        "metric_col": "acc_task_runtime_s",
        "ylabel": "Accumulated Task Runtime (s)",
        "legend": "Acc. Task Runtime",
        "out_stem": "reward_runtime_ci_band",
    },
    "res": {
        "results_dir": "scx_rl_res",
        "metric_col": "acc_task_latency_s",
        "ylabel": "Accumulated Task Latency (s)",
        "legend": "Acc. Task Latency",
        "out_stem": "reward_latency_ci_band",
    },
}

parser = argparse.ArgumentParser(description="Reward + secondary-metric CI-band plot.")
parser.add_argument("variant", choices=sorted(VARIANTS), help="which scheduler to plot")
args = parser.parse_args()
cfg = VARIANTS[args.variant]

seeds = [0, 1, 2, 3, 4]

# Column order written by the training loop. Used as a fallback when a CSV
# was produced without its header row.
COLUMNS = [
    "episode", "seed", "return", "acc_task_latency_s", "acc_task_runtime_s",
    "makespan_s", "preempt", "place_batches", "place_tasks", "place_dqn_actions",
    "place_direct", "dispatched_tasks", "ring_full", "slow_ticks", "qlen_peak",
    "qlen_mean", "slice_dqn_updates", "slice_rtc_ticks", "slice_other_updates",
    "slice_ms_8", "slice_ms_32", "slice_ms_174", "slice_ms_390", "slice_ms_1633",
    "rebalance_moves", "timestamp",
]

plot_dir = Path(__file__).resolve().parent
csv_path = plot_dir.parents[1] / "results" / cfg["results_dir"] / "train_metrics.csv"

reward = defaultdict(list)
metric = defaultdict(list)

# --- read logs (tolerate a missing header row and optional phase column) ---
with csv_path.open("r", newline="") as f:
    first_line = f.readline()
    f.seek(0)

    first_cols = next(csv.reader([first_line]))
    has_header = "episode" in first_cols and "seed" in first_cols
    fieldnames = None if has_header else COLUMNS
    if not has_header and len(first_cols) == len(COLUMNS) + 1:
        fieldnames = ["phase", *COLUMNS]

    for row in csv.DictReader(f, fieldnames=fieldnames):
        if row.get("phase") not in (None, "", "train"):
            continue

        s = int(row["seed"])
        if s not in seeds:
            continue
        reward[s].append(float(row["return"]))
        metric[s].append(float(row[cfg["metric_col"]]))  # already in s

# --- align episode lengths across seeds ---
min_len = min(len(reward[s]) for s in seeds)
R = np.vstack([np.array(reward[s][:min_len]) for s in seeds])  # (n_seeds, episodes)
T = np.vstack([np.array(metric[s][:min_len]) for s in seeds])

episodes = np.arange(1, min_len + 1)

# --- stats (mean and 95% CI across seeds) ---
n = R.shape[0]
mean_R = R.mean(axis=0)
std_R = R.std(axis=0, ddof=1)
ci_R = 1.96 * std_R / np.sqrt(n)

mean_T = T.mean(axis=0)
std_T = T.std(axis=0, ddof=1)
ci_T = 1.96 * std_T / np.sqrt(n)


# --- smoothing: reflection-padded moving average (no extra deps) ---
def smooth_ma(y, window):
    """Reflection-padded moving average with an odd window size."""
    if window < 3:
        return np.array(y)
    if window % 2 == 0:
        window = window - 1
    window = min(window, len(y))
    pad = window // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_pad, kernel, mode="valid")


mean_R_s = smooth_ma(mean_R, window=5)
mean_T_s = smooth_ma(mean_T, window=5)
ci_R_s = smooth_ma(ci_R, window=5)
ci_T_s = smooth_ma(ci_T, window=5)

# --- plot on twin axes ---
fig, ax1 = plt.subplots(figsize=(9, 4.5))

(line_r,) = ax1.plot(
    episodes, mean_R_s, linewidth=2, label="Reward", color="lightcoral"
)
band_r = ax1.fill_between(
    episodes,
    mean_R_s - ci_R_s,
    mean_R_s + ci_R_s,
    alpha=0.2,
    label="Reward 95% CI",
    color="lightcoral",
)
ax1.set_xlabel("Episode", fontsize=20)
ax1.set_ylabel("Reward", fontsize=20)
ax1.tick_params(axis="both", labelsize=16)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax1.yaxis.get_offset_text().set_fontsize(16)

ax2 = ax1.twinx()
(line_t,) = ax2.plot(
    episodes,
    mean_T_s,
    linewidth=2,
    linestyle="--",
    label=cfg["legend"],
    color="cornflowerblue",
)
band_t = ax2.fill_between(
    episodes,
    mean_T_s - ci_T_s,
    mean_T_s + ci_T_s,
    alpha=0.2,
    label=cfg["legend"] + " 95% CI",
    color="cornflowerblue",
)
ax2.set_ylabel(cfg["ylabel"], fontsize=19)
ax2.tick_params(axis="y", labelsize=16)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax2.yaxis.get_offset_text().set_fontsize(16)

handles = [line_r, band_r, line_t, band_t]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="center right", fontsize=15)

plt.tight_layout()
out_png = plot_dir / f"{cfg['out_stem']}.png"
out_pdf = plot_dir / f"{cfg['out_stem']}.pdf"
plt.savefig(out_png, dpi=600)
plt.savefig(out_pdf, dpi=600)
print(f"saved {out_png}")
print(f"saved {out_pdf}")
