"""Two-figure comparison of the seven main schedulers.

Produces two figures:
  1. CDF of per-task response time (latency)  -> response_<workload>.png / .pdf
  2. Violin of per-task runtime (execution)   -> execution_<workload>.png / .pdf

for: CFS, EEVDF, c-FIFO, Hybrid, ALPS, Synth. Latency (rl_res),
Synth. Runtime (rl_exec).

Per-task data source
--------------------
``time_measure`` parses the per-task lifecycle log written by replay_trace's
``--per-task`` (``results/<sched>/<workload>``, same TaskNew/FirstRun/TaskDead
format as results/scx_cfifo/<workload>). For each task:
    response (latency) = FirstRun - TaskNew
    runtime  (exec)    = TaskDead - FirstRun
    turnaround         = TaskDead - TaskNew

Usage
-----
    python compare_main_schedulers.py                # all 6 (homo + hetero)
    python compare_main_schedulers.py test_day02 ... # only the named ones
"""

import os
import re
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results")
OUTDIR = HERE

# scheduler results dir -> (legend label, color), in plot order.
SCHEDULERS = [
    ("scx_cfs", "CFS", "mediumpurple"),
    ("scx_eevdf", "EEVDF", "orange"),
    ("scx_cfifo", "FIFO", "lightcoral"),
    ("scx_hybrid", "Hybrid", "pink"),
    ("scx_alps", "ALPS", "tab:brown"),
    ("scx_sfs", "SFS", "tab:gray"),
    ("scx_rl_res", "Synth. Latency", "mediumseagreen"),
    ("scx_rl_exec/slice_pareto/s2000", "Synth. Inter.", "goldenrod"),
    ("scx_rl_exec", "Synth. Runtime", "cornflowerblue"),
]

# workloads plotted when no CLI argument is given: homogeneous + heterogeneous.
WORKLOADS = [
    "test_noon",
    "test_day02",
    "test_day10",
    "test_noon_hetero",
    "test_day02_hetero",
    "test_day10_hetero",
]


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


# --- the two figures ---------------------------------------------------------
def plot_cdf_response_execution(workload="test_day02"):
    data, labels, colors = [], [], []
    for sched, label, color in SCHEDULERS:
        path = os.path.join(RESULTS, sched, workload)
        if os.path.exists(path):
            data.append(time_measure(path))
            labels.append(label)
            colors.append(color)
        else:
            print(f"Warning: {path} not found.")

    if not data:
        print("No per-task lifecycle logs found -- nothing to plot.")
        return

    # 1. response-time CDF
    plt.figure(figsize=(9, 4.5))
    for i, label in enumerate(labels):
        plot_cdf(data[i][0], label, colors[i])
    plt.xlabel("Latency (s)", size=27)
    plt.ylabel("Cumulative Prob.", size=27)
    plt.xticks(size=20)
    plt.yticks(size=20)
    style_legend(fontsize=15)
    style_current_axes()
    plt.tight_layout()
    save_plot(os.path.join(OUTDIR, f"response_{workload}.png"))
    plt.close()

    # 2. runtime violin (broken y-axis)
    exec_data = [d[2] for d in data]
    plot_violin_broken(
        exec_data,
        labels,
        colors,
        os.path.join(OUTDIR, f"execution_{workload}.png"),
        "Runtime (s)",
        y_label_y=0.58,
        x_label_rotation=20,
    )
    print(f"Wrote response_{workload} and execution_{workload} into {OUTDIR}")


if __name__ == "__main__":
    # No args -> all 6 workloads; otherwise plot only the ones named.
    for wl in sys.argv[1:] or WORKLOADS:
        plot_cdf_response_execution(wl)
