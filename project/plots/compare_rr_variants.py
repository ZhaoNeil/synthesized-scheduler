"""Two-figure comparison of the four RR scheduler variants.

Produces two figures:
  1. CDF of per-task response time (latency)   -> response.png / .pdf
  2. Violin of per-task runtime (execution)    -> execution.png / .pdf

for: RR, Preemption, Workshuffling, Preempt+Shuffle.

Per-task data source
--------------------
``time_measure`` parses the per-task lifecycle log written by replay_trace's
``--per-task`` (``results/<sched>/<workload>``, same TaskNew/FirstRun/TaskDead
format as results/scx_cfifo/<workload>). For each task:
    response (latency) = FirstRun - TaskNew
    runtime  (exec)    = TaskDead - FirstRun
    turnaround         = TaskDead - TaskNew
Only the response and runtime series are plotted.
"""

import os
import re
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results")
OUTDIR = HERE

# label (as passed to plot_cdf_compare) -> scheduler results dir
SCHED_DIR = {
    "Round-Robin": "scx_rr",
    "Preemption": "scx_rr_preempt",
    "Workshuffling": "scx_rr_shuffle",
    "Preempt+Shuffle": "scx_rr_preempt_shuffle",
}

# Per-scheduler color, keyed by label so the mapping is fixed regardless of the
# order labels are passed in.
LABEL_COLORS = {
    "Round-Robin": "cornflowerblue",
    "Preemption": "lightcoral",
    "Workshuffling": "mediumseagreen",
    "Preempt+Shuffle": "orange",
}


# --- style helpers -----------------------------------------------------------
def style_axes(ax, grid_axis="both"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", length=6, width=1.2)


def style_current_axes(grid_axis="both"):
    style_axes(plt.gca(), grid_axis=grid_axis)


def style_legend(**kwargs):
    return plt.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.8",
        borderpad=0.6,
        **kwargs,
    )


def save_plot(outfile):
    plt.savefig(outfile, dpi=600, bbox_inches="tight", facecolor="white")
    root, ext = os.path.splitext(outfile)
    if ext.lower() == ".png":
        plt.savefig(f"{root}.pdf", bbox_inches="tight", facecolor="white")


def plot_cdf(data, label, color, linestyle="solid", linewidth=4.0):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(
        sorted_data,
        yvals,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def plot_violin_broken(
    exec_data,
    labels,
    colors,
    outfile,
    ylabel,
    percentile=95,
    x_label_rotation=15,
    x_label_size=18,
    y_label_y=0.5,
    figsize=(9, 4),
    y_label_size=18,
):
    """Violin plot with broken y-axis -- bulk on bottom, tail on top."""
    exec_data = [np.asarray(d) for d in exec_data]
    all_exec = np.concatenate(exec_data)
    bulk_top = np.percentile(all_exec, percentile)
    tail_max = all_exec.max() * 1.02
    bulk_bot = max(0, all_exec.min() - 0.02 * bulk_top)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [5, 8], "hspace": 0.08},
        constrained_layout=True,
    )

    for ax in (ax_top, ax_bot):
        parts = ax.violinplot(
            exec_data, showmeans=False, showmedians=True, showextrema=True
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i % len(colors)])
            body.set_edgecolor("black")
            body.set_alpha(0.7)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_edgecolor("black")

    ax_top.set_ylim(bulk_top, tail_max)
    ax_bot.set_ylim(bulk_bot, bulk_top)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    d = 0.5
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

    ax_bot.set_xticks(range(1, len(labels) + 1))
    ax_bot.set_xticklabels(labels, size=x_label_size, rotation=x_label_rotation)
    fig.supylabel(ylabel, size=27, x=-0.03, y=y_label_y, va="center")
    ax_top.tick_params(axis="y", labelsize=y_label_size)
    ax_bot.tick_params(axis="y", labelsize=y_label_size)
    style_axes(ax_top, grid_axis="y")
    style_axes(ax_bot, grid_axis="y")
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    save_plot(outfile)
    plt.close(fig)


# --- per-task data: parse the TaskNew/FirstRun/TaskDead lifecycle log ---------
_LINE_RE = re.compile(
    r"^(TaskNew|FirstRun|TaskDead): (\S+) "
    r"(?:enqueue at|starts at|at) "
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$"
)
_FIELD = {"TaskNew": 0, "FirstRun": 1, "TaskDead": 2}


def _ts_ns(base, frac, offset):
    if offset == "Z":
        offset = "+00:00"
    ts = datetime.fromisoformat(base + offset)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = ts.astimezone(timezone.utc) - epoch
    secs = delta.days * 86400 + delta.seconds
    return secs * 1_000_000_000 + int((frac or "").ljust(9, "0"))


def time_measure(path):
    """Return (response, turnaround, runtime) per-task lists in seconds."""
    events = {}  # task_id -> [tasknew, firstrun, taskdead]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = _LINE_RE.match(line.strip())
            if m is None:
                continue
            event, tid, base, frac, off = m.groups()
            events.setdefault(tid, [None, None, None])[_FIELD[event]] = _ts_ns(
                base, frac, off
            )

    res, tur, exe = [], [], []
    for tnew, frun, tdead in events.values():
        if tnew is None or frun is None or tdead is None:
            continue  # incomplete lifecycle
        res.append((frun - tnew) / 1e9)
        tur.append((tdead - tnew) / 1e9)
        exe.append((tdead - frun) / 1e9)
    return res, tur, exe


def log_path(label, workload):
    return os.path.join(RESULTS, SCHED_DIR[label], workload)


# --- the two figures ---------------------------------------------------------
def plot_cdf_compare(*labels, workload="test_day02"):
    data, valid, vcolors = [], [], []
    for label in labels:
        path = log_path(label, workload)
        if os.path.exists(path):
            data.append(time_measure(path))
            valid.append(label)
            vcolors.append(LABEL_COLORS[label])
        else:
            print(f"Warning: no per-task log for {label} at {path}")

    if not data:
        print("No per-task lifecycle logs found -- nothing to plot.")
        return

    # 1. response-time CDF
    plt.figure(figsize=(6, 5))
    for i, label in enumerate(valid):
        plot_cdf(data[i][0], label, vcolors[i])
    plt.xlabel("Latency (s)", size=27)
    plt.ylabel("Cumulative Prob.", size=27)
    plt.xticks(size=20)
    plt.yticks(size=20)
    style_legend(fontsize=20)
    style_current_axes()
    plt.tight_layout()
    save_plot(os.path.join(OUTDIR, "rr_response.png"))
    plt.close()

    # 2. runtime violin (broken y-axis)
    exec_data = [d[2] for d in data]
    plot_violin_broken(
        exec_data,
        valid,
        vcolors,
        os.path.join(OUTDIR, "rr_execution.png"),
        "Runtime (s)",
        x_label_rotation=15,
        x_label_size=24,
        y_label_size=25,
        figsize=(6, 5),
    )
    print(f"Wrote rr_response and rr_execution into {OUTDIR}")


if __name__ == "__main__":
    plot_cdf_compare("Round-Robin", "Preemption", "Workshuffling", "Preempt+Shuffle")
