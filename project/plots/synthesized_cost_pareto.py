#!/usr/bin/env python3
"""Cost-vs-latency Pareto scatter for synthesized workload results.

Reads aggregate metrics from ``results/<scheduler>/<workload>_metrics`` and
plots estimated Lambda cost vs accumulated task latency for the baseline
schedulers (CFS, EEVDF, cFIFO, and the related-work group Hybrid/ALPS/SFS)
together with the scx_rl_exec slice-sweep frontier.

This is the cost-axis twin of ``synthesized_pareto.py``: the x-axis is the
estimated Lambda cost (a positive-linear rescale of accumulated task runtime),
so the Pareto structure -- which RL operating points dominate which -- is
identical to the runtime plot; only the x scale/label differ.
"""

import argparse
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

LAMBDA_MEMORY_PORTION = [
    0.1172747,
    0.83123968,
    0.03785329,
    0.01363232,
]
LAMBDA_PRICE_PER_MS = [
    0.0000000021,
    0.0000000083,
    0.0000000167,
    0.0000000250,
]
PRICE_EXPECTATION_USD_PER_SECOND = (
    sum(x * y for x, y in zip(LAMBDA_MEMORY_PORTION, LAMBDA_PRICE_PER_MS)) * 1000
)

SCHEDULERS = {
    "CFS": "scx_cfs",
    "EEVDF": "scx_eevdf",
    "FIFO": "scx_cfifo",
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
    "FIFO": "lightcoral",
    "Hybrid": "pink",
    "ALPS": "tab:brown",
    "SFS": "tab:purple",
    RL_EXEC_LABEL: "tab:blue",
}

MARKERS = {
    "CFS": "s",
    "EEVDF": "v",
    "FIFO": "^",
    "Hybrid": "D",
    "ALPS": "X",
    "SFS": "P",
    RL_EXEC_LABEL: "o",
}

# Each baseline scheduler keeps its own legend entry; only the RL frontier gets a
# synthesized-algorithms label.
RL_EXEC_DISPLAY = "Synthesized algorithms"

TASKS_RE = re.compile(r"^tasks:\s*(\d+)", re.MULTILINE)
LATENCY_RE = re.compile(
    r"^accumulated task latency:\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.MULTILINE
)
RUNTIME_RE = re.compile(
    r"^accumulated task runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.MULTILINE
)


def cost(execution_seconds):
    return execution_seconds * PRICE_EXPECTATION_USD_PER_SECOND


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
                "cost": cost(metrics["runtime"]),
                "runtime": metrics["runtime"],
                "latency": metrics["latency"],
                "tasks": metrics["tasks"],
            }
        )
    return points


def load_slice_pareto_points(workload, results_dir):
    """RL slice-sweep frontier, POOLED across every placement seed. Each sweep dir
    fixes the macro time slice to a series of values (FIXED_SLICE_MS, slice DQN
    bypassed): large slice / RTC sits at the FIFO end (low cost, high latency),
    shrinking the slice trades cost up for latency down toward the CFS end.
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
                    "cost": cost(metrics["runtime"]),
                    "runtime": metrics["runtime"],
                    "latency": metrics["latency"],
                    "tasks": metrics["tasks"],
                }
            )
    points.sort(key=lambda p: p["runtime"])
    return points


def load_preempt_pareto_points(workload, results_dir):
    """Global preemption-BUDGET sweep points (results/scx_rl_exec/preempt_pareto/
    s<BASE>n<N>): a fixed macro slice whose TOTAL preemption count is capped at N, so
    the trace runs time-shared until the budget is spent and FIFO after. Sweeping N
    dials cost continuously, populating the cost band the pure-slice sweep skips (the
    day02 void). Pooled into the same RL frontier so mark_dominated / pareto_filter
    judge them against the slice points. Control runs (huge budget n>=99999, i.e.
    re-measured pure-slice anchors) and per-task-cap (k) dirs are skipped."""
    rl_dir = results_dir / RL_EXEC_DIR / "preempt_pareto"
    points = []
    # s<BASE>n<N>: budget sweep with a run-to-completion (FIFO) tail.
    # s<BASE>p<POST>n<N>: budget sweep whose post-budget tail runs at slice POST,
    # interpolating between the slice-BASE and slice-POST frontier points.
    for d in sorted(rl_dir.glob("s*n*")):
        m = re.match(r"s(\d+)(?:p(\d+))?n(\d+)$", d.name)
        if m is None:
            continue
        base, budget = int(m.group(1)), int(m.group(3))
        if budget >= 99999:  # control anchor (pure fixed slice); not a budget point
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
                "cost": cost(metrics["runtime"]),
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
    baselines, so the frontier's full FIFO->CFS extent is preserved. Cost is a
    positive-linear rescale of runtime, so dominance is the same on either axis.
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


def relfmt(value, _pos):
    return f"{value:.2f}×"


def kfmt(value, _pos):
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def style_axes(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(relfmt))
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

    for point in points:
        label = point["label"]
        ax.scatter(
            point["cost"],
            point["latency"],
            c=COLORS[label],
            s=420,
            marker=MARKERS[label],
            edgecolor="black",
            linewidth=1.2,
            label=label,
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
                [p["cost"] for p in dom],
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
            [p["cost"] for p in nd],
            [p["latency"] for p in nd],
            c=COLORS[RL_EXEC_LABEL],
            s=260,
            marker=MARKERS[RL_EXEC_LABEL],
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label=RL_EXEC_DISPLAY,
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
                    (p["cost"], p["latency"]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=11,
                    color=COLORS[RL_EXEC_LABEL],
                    # Draw above the markers (zorder 6) with a semi-opaque white
                    # backing so the text stays legible where it overlaps a point.
                    zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )

    ax.set_xlabel("Relative Cost (CFS = 1)", size=27, labelpad=10)
    ax.set_ylabel("Accumulated Task Latency (s)", size=27, labelpad=10)
    style_axes(ax)
    ax.legend(
        fontsize=18.5,
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.8",
        borderpad=0.4,
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    root = outfile.with_suffix("")
    fig.savefig(root.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(root.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {root.with_suffix('.png')}")
    print(f"Wrote {root.with_suffix('.pdf')}")


def process_workload(workload, args):
    outfile = args.out or OUTDIR / f"synthesized_cost_pareto_{workload}.png"
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

    # Normalise x-axis to CFS cost so the plot shows relative cost (CFS = 1).
    cfs_cost = next((p["cost"] for p in points if p["label"] == "CFS"), None)
    if cfs_cost is None or cfs_cost == 0:
        raise RuntimeError(f"CFS baseline not found for {workload!r}; cannot normalise cost axis.")
    for p in points:
        p["cost"] /= cfs_cost
    for p in frontier:
        p["cost"] /= cfs_cost

    for point in points:
        print(
            f"{point['label']}: tasks={point['tasks']}, "
            f"runtime={point['runtime']:.1f}s, rel_cost={point['cost']:.4f}×, "
            f"latency={point['latency']:.1f}s"
        )
    for point in frontier:
        print(
            f"{point['label']} slice={point['slice_ms']}ms: "
            f"rel_cost={point['cost']:.4f}×, latency={point['latency']:.1f}s"
        )
    plot(points, frontier, outfile, show_labels=args.labels, label_gap=args.label_gap_threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Plot synthesized workload estimated-cost Pareto scatter."
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
        "regions and endpoints, doesn't widen gaps). 0 = off (default, matches "
        "synthesized_pareto.py); try ~0.02-0.03 to declutter dense clusters.",
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
        "points that populate the cost bands the pure-slice sweep skips (the day02 "
        "voids). Default: on (only test_day02 has this data; other workloads are "
        "unaffected). Pass --no-preempt-fill for the slice-only frontier.",
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
