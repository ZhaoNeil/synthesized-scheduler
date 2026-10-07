#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = REPO_ROOT / "results"
OUTDIR = HERE
WORKLOAD = "test_day02"

# Optional CSV that curates the synthesized points; if absent, all slice dirs are used.
INPUT = OUTDIR / f"provider_headroom_client_cost_all_points_{WORKLOAD}.csv"
OUTPUT = OUTDIR / f"wait_gini_heuristics_vs_synthesized_{WORKLOAD}.png"

HEURISTICS = {
    "CFS": RESULTS / "scx_cfs" / WORKLOAD,
    "EEVDF": RESULTS / "scx_eevdf" / WORKLOAD,
    "ALPS": RESULTS / "scx_alps" / WORKLOAD,
    "Hybrid": RESULTS / "scx_hybrid" / WORKLOAD,
    "SFS": RESULTS / "scx_sfs" / WORKLOAD,
    "FIFO": RESULTS / "scx_cfifo" / WORKLOAD,
}
SYNTH_BASE = RESULTS / "scx_rl_exec" / "slice_pareto"

TEXT = "#222222"
BAR_EDGE = "#222222"
AXIS = "#33475b"
GRID = "#c9d2da"


def parse_ts(value: str) -> datetime:
    if "." in value:
        head, frac = value.split(".", 1)
        return datetime.fromisoformat(head + "." + frac[:6])
    return datetime.fromisoformat(value)


def gini(values: list[float]) -> float:
    sorted_values = sorted(value for value in values if value >= 0)
    n = len(sorted_values)
    total = sum(sorted_values)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return (2 * weighted / (n * total)) - ((n + 1) / n)


def wait_stats(log_path: Path) -> tuple[float, float]:
    task_new: dict[int, datetime] = {}
    first_run: dict[int, datetime] = {}
    new_re = re.compile(r"TaskNew: C(\d+) enqueue at ([0-9T:\-.]+)")
    first_re = re.compile(r"FirstRun: C(\d+) starts at ([0-9T:\-.]+)")

    with log_path.open(errors="replace") as handle:
        for line in handle:
            match = new_re.search(line)
            if match:
                task_new[int(match.group(1))] = parse_ts(match.group(2))
                continue
            match = first_re.search(line)
            if match and int(match.group(1)) not in first_run:
                first_run[int(match.group(1))] = parse_ts(match.group(2))

    waits = [
        (first_run[task] - task_new[task]).total_seconds()
        for task in task_new.keys() & first_run.keys()
    ]
    avg_wait = sum(waits) / len(waits) if waits else 0.0
    return gini(waits), avg_wait


def load_provider_rows() -> dict[str, dict[str, str]]:
    if not INPUT.exists():
        return {}
    with INPUT.open(newline="") as handle:
        return {row["policy"]: row for row in csv.DictReader(handle)}


def build_rows() -> list[dict[str, float | str]]:
    provider_rows = load_provider_rows()

    rows: list[dict[str, float | str]] = []
    for policy, path in HEURISTICS.items():
        wait_gini_value, avg_wait = wait_stats(path)
        rows.append(
            {
                "group": "heuristic",
                "policy": policy,
                "wait_gini": wait_gini_value,
                "avg_wait_s_per_task": avg_wait,
            }
        )

    if provider_rows:
        synth_policies = [
            policy
            for policy, row in provider_rows.items()
            if row.get("group") == "synthesized"
        ]
    else:
        synth_policies = [
            slice_dir.name
            for slice_dir in sorted(SYNTH_BASE.glob("s*"))
            if re.match(r"s\d+$", slice_dir.name)
        ]

    synthesized: list[dict[str, float | str]] = []
    for policy in synth_policies:
        path = SYNTH_BASE / policy / WORKLOAD
        if not path.exists():
            continue
        wait_gini_value, avg_wait = wait_stats(path)
        synthesized.append(
            {
                "group": "synthesized",
                "policy": policy,
                "wait_gini": wait_gini_value,
                "avg_wait_s_per_task": avg_wait,
            }
        )

    synthesized.sort(key=lambda row: float(row["avg_wait_s_per_task"]))
    return rows + synthesized


def make_plot(rows: list[dict[str, float | str]]) -> plt.Figure:
    heuristics = [row for row in rows if row["group"] == "heuristic"]
    synthesized = [row for row in rows if row["group"] == "synthesized"]

    cmap = mpl.colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=95, clip=True)

    heuristic_x = list(range(len(heuristics)))
    synth_step = 0.105
    synth_width = 0.105
    separator_gap = 0.55
    synth_gap = 0.98
    synth_start = heuristic_x[-1] + synth_gap
    separator_x = heuristic_x[-1] + separator_gap
    synth_x = [synth_start + index * synth_step for index in range(len(synthesized))]

    xs = heuristic_x + synth_x
    values = [float(row["wait_gini"]) for row in rows]
    latencies = [float(row["avg_wait_s_per_task"]) for row in rows]
    colors = [cmap(norm(value)) for value in latencies]
    widths = [0.52] * len(heuristics) + [synth_width] * len(synthesized)
    linewidths = [1.2] * len(heuristics) + [0.0] * len(synthesized)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(
        xs,
        values,
        width=widths,
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=linewidths,
        align="center",
    )
    ax.axvline(separator_x, color=AXIS, lw=1.2, ls=(0, (3, 2)))
    ax.set_ylabel("Wait-Time Gini", size=27, labelpad=10)
    ax.set_ylim(0, max(values) * 1.10)
    ax.set_xlim(-0.55, synth_x[-1] + synth_width)
    ax.set_xticks(heuristic_x)
    ax.set_xticklabels(
        [str(row["policy"]) for row in heuristics],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="both", labelsize=24, width=1.2, length=6)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scalar = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Mean Latency (s)", size=27, labelpad=10)
    cbar.set_ticks([0, 45, 95])
    cbar.set_ticklabels(["0", "45", "95+"])
    cbar.ax.tick_params(labelsize=20, width=1.0, length=4)
    cbar.outline.set_edgecolor(AXIS)
    cbar.outline.set_linewidth(1.0)

    ax.text(
        (synth_x[0] + synth_x[-1]) / 2,
        -0.085,
        "Synthesized Schedulers",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        size=24,
    )
    fig.subplots_adjust(left=0.09, right=0.92, top=0.96, bottom=0.20)
    return fig


def main() -> None:
    rows = build_rows()
    fig = make_plot(rows)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {OUTPUT.with_suffix('.png')} and {OUTPUT.with_suffix('.pdf')}")
    print(f"synth_points {sum(1 for row in rows if row['group'] == 'synthesized')}")


if __name__ == "__main__":
    main()
