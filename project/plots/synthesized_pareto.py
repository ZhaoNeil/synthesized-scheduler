#!/usr/bin/env python3
"""Pareto scatter for synthesized workload scheduler results.

Reads aggregate metrics from ``results/<scheduler>/<workload>_metrics`` and
plots accumulated task runtime vs accumulated task latency for:
CFS, EEVDF, cFIFO, Hybrid, ALPS, SFS, and scx_rl_exec test runs.
"""

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[1] / "results"
OUTDIR = HERE

SCHEDULERS = {
    "CFS": "scx_cfs",
    "EEVDF": "scx_eevdf",
    "cFIFO": "scx_cfifo",
    "Hybrid": "scx_hybrid",
    "ALPS": "scx_alps",
    "SFS": "scx_sfs",
}

RL_EXEC_LABEL = "scx_rl_exec"
RL_EXEC_DIR = "scx_rl_exec"

# When labels are on, only annotate a frontier point if the runtime gap (s) to a
# neighbour is at least this large -- i.e. the points flanking a gap. Points
# buried in a dense cluster stay unlabelled so the text does not pile up.
LABEL_GAP_THRESHOLD = 8000

COLORS = {
    "CFS": "orange",
    "EEVDF": "tab:green",
    "cFIFO": "lightcoral",
    "Hybrid": "pink",
    "ALPS": "pink",
    "SFS": "pink",
    RL_EXEC_LABEL: "tab:blue",
}

MARKERS = {
    "CFS": "s",
    "EEVDF": "v",
    "cFIFO": "^",
    "Hybrid": "D",
    "ALPS": "D",
    "SFS": "D",
    RL_EXEC_LABEL: "o",
}

DISPLAY_LABELS = {
    "CFS": "CFS",
    "EEVDF": "EEVDF",
    "cFIFO": "FIFO",
    "Hybrid": "Related Work",
    "ALPS": "Related Work",
    "SFS": "Related Work",
    RL_EXEC_LABEL: "Synthesized algorithms",
}

TASKS_RE = re.compile(r"^tasks:\s*(\d+)", re.MULTILINE)
LATENCY_RE = re.compile(
    r"^accumulated task latency:\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.MULTILINE
)
RUNTIME_RE = re.compile(
    r"^accumulated task runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.MULTILINE
)


def parse_metrics(path):
    """Return aggregate metrics from a replay/analyze ``*_metrics`` file."""
    text = path.read_text(encoding="utf-8")
    tasks = TASKS_RE.search(text)
    latency = LATENCY_RE.search(text)
    runtime = RUNTIME_RE.search(text)
    if tasks is None or latency is None or runtime is None:
        raise ValueError(f"{path} does not contain the expected aggregate metrics")
    return {
        "tasks": int(tasks.group(1)),
        "latency": float(latency.group(1)),
        "runtime": float(runtime.group(1)),
    }


def discover_workloads(results_dir):
    """All workload names with baseline CFS metrics, used as the default set."""
    cfs_dir = results_dir / SCHEDULERS["CFS"]
    return sorted(p.name[: -len("_metrics")] for p in cfs_dir.glob("*_metrics"))


def load_points(workload, results_dir):
    points = []
    for label, scheduler_dir in SCHEDULERS.items():
        path = results_dir / scheduler_dir / f"{workload}_metrics"
        if not path.exists():
            print(f"Warning: {path} not found; skipping {label}.")
            continue

        metrics = parse_metrics(path)
        points.append(
            {
                "label": label,
                "runtime": metrics["runtime"],
                "latency": metrics["latency"],
                "tasks": metrics["tasks"],
            }
        )
    return points


def load_rl_exec_points(workload, results_dir):
    points = []
    rl_dir = results_dir / RL_EXEC_DIR

    for seed_dir in sorted(rl_dir.glob("seed_*")):
        path = seed_dir / f"{workload}_metrics"
        if not path.exists():
            continue

        metrics = parse_metrics(path)
        points.append(
            {
                "label": RL_EXEC_LABEL,
                "runtime": metrics["runtime"],
                "latency": metrics["latency"],
                "tasks": metrics["tasks"],
            }
        )

    if points:
        return points

    csv_path = rl_dir / "train_metrics.csv"
    if workload != "test_day02" or not csv_path.exists():
        return []

    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "test":
                continue

            points.append(
                {
                    "label": RL_EXEC_LABEL,
                    "runtime": float(row["acc_task_runtime_s"]),
                    "latency": float(row["acc_task_latency_s"]),
                    "tasks": int(row["dispatched_tasks"]),
                    "seed": int(row["seed"]),
                    "episode": int(row["episode"]),
                }
            )

    return points


def load_slice_pareto_points(workload, results_dir):
    """RL slice-sweep frontier, POOLED across every placement seed. Each sweep dir
    fixes the macro time slice to a series of values (FIXED_SLICE_MS, slice DQN
    bypassed): large slice / RTC sits at the FIFO end (low runtime, high latency),
    shrinking the slice trades runtime up for latency down toward the CFS end.
    ``slice_pareto/`` is seed_0's committed backbone; ``slice_pareto_seed<N>/`` are
    the per-seed gap-fill runs. All points are pooled so a void filled by ANY seed
    shows up in the one frontier."""
    rl_dir = results_dir / RL_EXEC_DIR
    points = []
    for sweep_dir in sorted(rl_dir.glob("slice_pareto*")):
        sm = re.search(r"slice_pareto_seed(\d+)$", sweep_dir.name)
        seed = int(sm.group(1)) if sm else 0
        for d in sorted(sweep_dir.glob("s*")):
            m = re.match(r"s(\d+)$", d.name)
            if m is None:
                continue
            path = d / f"{workload}_metrics"
            if not path.exists():
                continue
            metrics = parse_metrics(path)
            points.append(
                {
                    "label": RL_EXEC_LABEL,
                    "seed": seed,
                    "slice_ms": int(m.group(1)),
                    "runtime": metrics["runtime"],
                    "latency": metrics["latency"],
                    "tasks": metrics["tasks"],
                }
            )
    points.sort(key=lambda p: p["runtime"])
    return points


def load_preempt_pareto_points(workload, results_dir):
    """Global preemption-budget sweep points (results/scx_rl_exec/preempt_pareto/
    s<BASE>n<N> and s<BASE>p<POST>n<N>): a fixed macro slice whose TOTAL preemption
    count is capped at N (the post-budget tail runs at slice POST, or run-to-
    completion), populating the runtime bands the pure-slice sweep skips (the day02
    voids). Pooled into the same RL frontier so mark_dominated / pareto_filter judge
    them against the slice points. Control runs (n>=99999, re-measured pure-slice
    anchors) are skipped."""
    rl_dir = results_dir / RL_EXEC_DIR / "preempt_pareto"
    points = []
    for d in sorted(rl_dir.glob("s*n*")):
        m = re.match(r"s(\d+)(?:p(\d+))?n(\d+)$", d.name)
        if m is None:
            continue
        base, budget = int(m.group(1)), int(m.group(3))
        if budget >= 99999:
            continue
        path = d / f"{workload}_metrics"
        if not path.exists():
            continue
        try:
            metrics = parse_metrics(path)
        except ValueError:
            # An in-progress or failed sweep point leaves an empty/partial metrics
            # file; skip it rather than crash the whole plot.
            continue
        points.append(
            {
                "label": RL_EXEC_LABEL,
                "seed": 0,
                "slice_ms": base,
                "budget": budget,
                "runtime": metrics["runtime"],
                "latency": metrics["latency"],
                "tasks": metrics["tasks"],
            }
        )
    return points


def pareto_filter(points):
    """Keep only the Pareto-optimal RL points (minimise BOTH runtime and latency).

    A point is dropped if another RL point matches or beats it on both axes (and
    strictly beats on at least one). Compared only against other RL points, not the
    baselines, so the frontier's full FIFO->CFS extent is preserved.
    """
    kept = []
    for p in points:
        dominated = any(
            q is not p
            and q["runtime"] <= p["runtime"]
            and q["latency"] <= p["latency"]
            and (q["runtime"] < p["runtime"] or q["latency"] < p["latency"])
            for q in points
        )
        if not dominated:
            kept.append(p)
    return kept


def thin_frontier(points, min_frac):
    """Declutter dense clusters without widening gaps.

    Sweeping by runtime, drop any point closer than ``min_frac`` (a fraction of the
    normalized runtime/latency diagonal) to the previously kept point. Dense runs
    get thinned to ~min_frac spacing; sparse regions (already farther apart than
    min_frac) keep every point, so no existing gap grows. The two endpoints (FIFO
    and CFS extremes) are always kept. min_frac <= 0 disables thinning.
    """
    if min_frac <= 0 or len(points) <= 2:
        return points
    rs = [p["runtime"] for p in points]
    ls = [p["latency"] for p in points]
    rspan = (max(rs) - min(rs)) or 1.0
    lspan = (max(ls) - min(ls)) or 1.0
    ordered = sorted(points, key=lambda p: p["runtime"])
    kept = [ordered[0]]
    for p in ordered[1:-1]:
        dr = (p["runtime"] - kept[-1]["runtime"]) / rspan
        dl = (p["latency"] - kept[-1]["latency"]) / lspan
        if (dr * dr + dl * dl) ** 0.5 >= min_frac:
            kept.append(p)
    kept.append(ordered[-1])
    return kept


def kfmt(value, _pos):
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def style_axes(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.tick_params(axis="both", labelsize=24, length=6, width=1.2)
    ax.margins(x=0.08, y=0.08)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def plot(points, frontier, outfile, show_labels=True, label_gap=LABEL_GAP_THRESHOLD):
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    legend_labels = set()

    for point in points:
        label = point["label"]
        display_label = DISPLAY_LABELS[label]
        legend_label = display_label if display_label not in legend_labels else None
        legend_labels.add(display_label)
        ax.scatter(
            point["runtime"],
            point["latency"],
            c=COLORS[label],
            s=420,
            marker=MARKERS[label],
            edgecolor="black",
            linewidth=1.2,
            label=legend_label,
            zorder=4,
        )

    # RL slice-sweep frontier. If points carry a "dominated" flag (set when
    # --mark-dominated is on), the non-dominated ones are drawn as solid circles
    # (the Pareto frontier) and the dominated ones as hollow circles.
    if frontier:
        dom = [p for p in frontier if p.get("dominated")]
        nd = [p for p in frontier if not p.get("dominated")]
        if dom:
            ax.scatter(
                [p["runtime"] for p in dom],
                [p["latency"] for p in dom],
                facecolors="none",
                edgecolors=COLORS[RL_EXEC_LABEL],
                s=210,
                marker=MARKERS[RL_EXEC_LABEL],
                linewidth=1.3,
                zorder=5,
                label="Synthesized (dominated)",
            )
        ax.scatter(
            [p["runtime"] for p in nd],
            [p["latency"] for p in nd],
            c=COLORS[RL_EXEC_LABEL],
            s=260,
            marker=MARKERS[RL_EXEC_LABEL],
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label=DISPLAY_LABELS[RL_EXEC_LABEL],
        )
        # Slice (ms) labels are a working aid to spot which regions need more
        # operating points; disable with --no-labels for a clean figure. Only the
        # points flanking a runtime gap >= label_gap (plus the two endpoints) are
        # labelled -- points inside a dense cluster are skipped so text doesn't pile
        # up. frontier is sorted by runtime (see load_slice_pareto_points).
        if show_labels:
            rts = [q["runtime"] for q in frontier]
            n = len(frontier)
            for i, p in enumerate(frontier):
                gap_left = rts[i] - rts[i - 1] if i > 0 else float("inf")
                gap_right = rts[i + 1] - rts[i] if i < n - 1 else float("inf")
                if max(gap_left, gap_right) < label_gap:
                    continue
                tag = "RTC" if p["slice_ms"] >= 100000 else f"{p['slice_ms']}"
                ax.annotate(
                    tag,
                    (p["runtime"], p["latency"]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=11,
                    color=COLORS[RL_EXEC_LABEL],
                    # Draw above the markers (zorder 6) with a semi-opaque white
                    # backing so the text stays legible where it overlaps a point.
                    zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )

    ax.set_xlabel("Accumulated Task Runtime (s)", size=27, labelpad=10)
    ax.set_ylabel("Accumulated Task Latency (s)", size=27, labelpad=10)
    style_axes(ax)
    ax.legend(
        fontsize=20,
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.8",
        borderpad=0.6,
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    root = outfile.with_suffix("")
    fig.savefig(root.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(root.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {root.with_suffix('.png')}")
    print(f"Wrote {root.with_suffix('.pdf')}")


def process_workload(workload, args):
    outfile = args.out or OUTDIR / f"synthesized_pareto_{workload}.png"
    points = load_points(workload, args.results_dir)
    frontier = load_slice_pareto_points(workload, args.results_dir)
    if args.preempt_fill:
        pf = load_preempt_pareto_points(workload, args.results_dir)
        print(f"preempt-fill: pooled {len(pf)} budget points into the frontier")
        frontier = sorted(frontier + pf, key=lambda p: p["runtime"])
    if args.mark_dominated:
        nd_ids = {id(p) for p in pareto_filter(frontier)}
        for p in frontier:
            p["dominated"] = id(p) not in nd_ids
        print(f"mark-dominated: {sum(p['dominated'] for p in frontier)}/{len(frontier)} dominated")
    if args.pareto_only:
        kept = pareto_filter(frontier)
        print(f"pareto-only: kept {len(kept)}/{len(frontier)} non-dominated RL points")
        frontier = kept
    if args.thin_frac > 0:
        thinned = thin_frontier(frontier, args.thin_frac)
        print(f"thin: kept {len(thinned)}/{len(frontier)} RL points (min_frac={args.thin_frac})")
        frontier = thinned
    if not points and not frontier:
        raise RuntimeError(f"No scheduler data found for workload {workload!r}.")

    for point in points:
        print(
            f"{point['label']}: tasks={point['tasks']}, "
            f"runtime={point['runtime']:.1f}s, latency={point['latency']:.1f}s"
        )
    for point in frontier:
        print(
            f"{point['label']} slice={point['slice_ms']}ms: "
            f"runtime={point['runtime']:.1f}s, latency={point['latency']:.1f}s"
        )
    plot(points, frontier, outfile, show_labels=args.labels, label_gap=args.label_gap_threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Plot synthesized workload Pareto scatter for baseline schedulers."
    )
    parser.add_argument(
        "--workload",
        default=None,
        help="workload name to plot. Default: all workloads found under "
        "results/scx_cfs/*_metrics.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (without extension logic applied). Only valid when "
        "plotting a single --workload.",
    )
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the per-point slice-ms annotations on the RL frontier as a "
        "densification aid. Default: off (clean figure); pass --labels to enable.",
    )
    parser.add_argument(
        "--label-gap-threshold",
        type=float,
        default=LABEL_GAP_THRESHOLD,
        help="only annotate frontier points flanking a runtime gap (s) at least "
        f"this large; clustered points stay unlabelled (default {LABEL_GAP_THRESHOLD}).",
    )
    parser.add_argument(
        "--pareto-only",
        action="store_true",
        help="drop RL points dominated by another RL point (keep only the "
        "non-dominated Pareto frontier). Default: show every operating point.",
    )
    parser.add_argument(
        "--thin-frac",
        type=float,
        default=0.0,
        help="declutter: drop RL points closer than this fraction of the data "
        "diagonal to the previous kept point (thins dense clusters, keeps sparse "
        "regions and endpoints, doesn't widen gaps). 0 = off; try ~0.02-0.03.",
    )
    parser.add_argument(
        "--mark-dominated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw RL points dominated by another RL point as hollow circles and "
        "the non-dominated Pareto frontier as solid circles (computed before any "
        "thinning, so the status is the true one). Default: on; pass "
        "--no-mark-dominated to draw every RL point as a solid circle.",
    )
    parser.add_argument(
        "--preempt-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pool the global preemption-budget sweep (results/scx_rl_exec/"
        "preempt_pareto/s<BASE>[p<POST>]n<N>) into the RL frontier -- the operating "
        "points that populate the runtime bands the pure-slice sweep skips (the "
        "day02 voids). Default: on (only test_day02 has this data; other workloads "
        "are unaffected). Pass --no-preempt-fill for the slice-only frontier.",
    )
    args = parser.parse_args()

    workloads = [args.workload] if args.workload else discover_workloads(args.results_dir)
    if not workloads:
        raise RuntimeError(f"No workloads found under {args.results_dir / SCHEDULERS['CFS']}.")
    if args.out and len(workloads) > 1:
        raise SystemExit("--out only supported with a single --workload.")

    for workload in workloads:
        print(f"=== {workload} ===")
        process_workload(workload, args)


if __name__ == "__main__":
    main()
