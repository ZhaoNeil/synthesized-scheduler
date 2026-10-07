#!/usr/bin/env python3
"""Throughput comparison for the main schedulers.

Parses per-task TaskNew/TaskDead lifecycle logs from ``results/<sched>/<workload>``
and writes a completion CDF, the per-bucket CSV, and a percentile summary.

Usage:
    python project/plots/throughput_schedulers.py
    python project/plots/throughput_schedulers.py --workload test_noon --bucket-size 40
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[1] / "results"
OUTDIR = HERE

DEFAULT_BUCKET_SIZE = 40.0
DEFAULT_MAX_X_TICKS = 7
PERCENTILES = [90, 95, 99]

_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?"
)


def parse_datetime(token):
    """Parse an ISO-8601 timestamp, tolerating up to 9 fractional digits."""
    match = _TS_RE.match(token)
    if match is None:
        raise ValueError(f"unparseable timestamp: {token!r}")
    base, frac, offset = match.groups()
    offset = "+00:00" if offset in (None, "Z") else offset
    frac = (frac or "")[:6].ljust(6, "0")  # datetime tops out at microseconds
    return datetime.fromisoformat(f"{base}.{frac}{offset}")


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #
@dataclass
class RunThroughput:
    start_time: object = None
    completion_times: list = field(default_factory=list)
    duration_s: float = None

    def has_data(self):
        return self.start_time is not None or self.completion_times


def parse_elapsed(line):
    match = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)", line)
    return float(match.group(1)) if match else None


def timestamp_from_log_line(line):
    return parse_datetime(line.split()[-1])


def parse_runs(log_file):
    runs = []
    current = RunThroughput()

    def finish_current():
        nonlocal current
        if current.has_data():
            if current.start_time is None and current.completion_times:
                current.start_time = min(current.completion_times)
            runs.append(current)
        current = RunThroughput()

    with open(log_file, "r") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("TaskNew:"):
                timestamp = timestamp_from_log_line(line)
                if current.start_time is None:
                    current.start_time = timestamp
            elif line.startswith("TaskDead:"):
                timestamp = timestamp_from_log_line(line)
                if current.start_time is None:
                    current.start_time = timestamp
                current.completion_times.append(timestamp)
            elif line.startswith("time elapsed:") or line.startswith("Total time:"):
                current.duration_s = parse_elapsed(line)
                finish_current()

    finish_current()
    return [run for run in runs if run.completion_times]


def elapsed_for_run(run):
    return np.array(
        [
            (completion_time - run.start_time).total_seconds()
            for completion_time in run.completion_times
        ]
    )


def counts_for_run(run, elapsed, bucket_size):
    max_elapsed = float(elapsed.max()) if elapsed.size else 0.0
    duration = max(run.duration_s or 0.0, max_elapsed)
    num_buckets = max(1, int(np.ceil(duration / bucket_size)))
    counts = np.zeros(num_buckets, dtype=float)

    for seconds in elapsed:
        bucket = min(int(seconds // bucket_size), num_buckets - 1)
        counts[bucket] += 1

    return counts


def tick_indices(num_buckets, max_ticks):
    if max_ticks <= 0:
        return []
    if num_buckets <= max_ticks:
        return list(range(num_buckets))
    step = int(np.ceil(num_buckets / max_ticks))
    return list(range(0, num_buckets, step))


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

OUTPUT_STEM = "throughput_schedulers"


def load_series(workload, bucket_size):
    series = []
    for sched, label, color in SCHEDULERS:
        path = RESULTS_DIR / sched / workload
        if not path.exists():
            print(f"Warning: {path} not found; skipping {label}.")
            continue

        runs = parse_runs(path)
        if not runs:
            print(f"Warning: {path} has no TaskDead records; skipping {label}.")
            continue

        elapsed = elapsed_for_run(runs[0])
        series.append(
            {
                "label": label,
                "color": color,
                "elapsed": elapsed,
                "counts": counts_for_run(runs[0], elapsed, bucket_size),
            }
        )

    return series


def write_csv(series, bucket_size, outfile):
    max_buckets = max(len(item["counts"]) for item in series)
    with open(outfile, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["bucket_start_s", "bucket_end_s"] + [item["label"] for item in series]
        )
        for bucket in range(max_buckets):
            row = [bucket * bucket_size, (bucket + 1) * bucket_size]
            for item in series:
                counts = item["counts"]
                row.append(counts[bucket] if bucket < len(counts) else 0)
            writer.writerow(row)


def plot_completion_cdf(series, bucket_size, outfile, max_x_ticks):
    max_elapsed = max(float(item["elapsed"].max()) for item in series)
    x_max = np.ceil(max_elapsed / bucket_size) * bucket_size
    num_buckets = max(1, int(np.ceil(max_elapsed / bucket_size)))
    fig_width = max(9, num_buckets * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    for item in series:
        elapsed = np.sort(item["elapsed"])
        completed_fraction = np.arange(1, len(elapsed) + 1, dtype=float) / len(elapsed)
        ax.step(
            np.concatenate(([0.0], elapsed)),
            np.concatenate(([0.0], completed_fraction)),
            where="post",
            label=item["label"],
            color=item["color"],
            linewidth=3.3,
            alpha=0.95,
        )

    ax.set_xlim(left=0.0, right=x_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Completion Time (s)", fontsize=27)
    ax.set_ylabel("Invocations Completed (%)", fontsize=27)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    x_tick_indices = tick_indices(num_buckets + 1, max_x_ticks)
    ax.set_xticks([i * bucket_size for i in x_tick_indices])
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.8",
        fontsize=16,
        loc="lower right",
        ncol=2,
        columnspacing=1.1,
        handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=600, bbox_inches="tight", facecolor="white")
    root = Path(outfile).with_suffix("")
    fig.savefig(f"{root}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(series, outfile):
    headers = ["Scheduler"] + [
        f"Time to Complete {p}% of Tasks (s)" for p in PERCENTILES
    ]
    rows = []
    for item in series:
        values = np.percentile(item["elapsed"], PERCENTILES)
        rows.append([item["label"]] + [f"{value:.1f}" for value in values])

    table = [headers] + rows
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = [
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in table
    ]
    Path(outfile).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Throughput comparison for the main schedulers."
    )
    parser.add_argument("--workload", default="test_noon")
    parser.add_argument("--bucket-size", type=float, default=DEFAULT_BUCKET_SIZE)
    parser.add_argument("--max-x-ticks", type=int, default=DEFAULT_MAX_X_TICKS)
    args = parser.parse_args()

    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")

    bucket_label = f"{args.bucket_size:g}".replace(".", "p")
    stem = f"{OUTPUT_STEM}_{args.workload}_{bucket_label}s"
    png = OUTDIR / f"{stem}.png"
    csv_out = OUTDIR / f"{stem}.csv"
    summary = OUTDIR / f"{stem}_summary.txt"

    series = load_series(args.workload, args.bucket_size)
    if not series:
        raise RuntimeError("No scheduler throughput data found.")

    plot_completion_cdf(series, args.bucket_size, str(png), args.max_x_ticks)
    write_csv(series, args.bucket_size, csv_out)
    write_summary(series, summary)

    print(f"Wrote {png}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
