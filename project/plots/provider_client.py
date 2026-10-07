#!/usr/bin/env python3
"""Provider-headroom vs client-cost scatter, driven by the real result files.

For every operating point this reads two files:

  results/<dir>/<workload>_metrics     (text: tasks / latency / runtime / makespan)
  results/<dir>/<workload>_occupancy   (per-core busy intervals: core task start end dur)

and derives the three encodings in the figure:

  x  Client cost improvement vs CFS (%)
        = (runtime_CFS - runtime) / runtime_CFS * 100
        client-billed cost is linear in accumulated task runtime, so the runtime
        reduction IS the cost reduction (see project/plots/synthesized_cost_pareto.py).

  y  Provider headroom vs CFS (%)
        = spare(point) - spare(CFS),  where
          spare = (1 - occupied_core_seconds / (NCORES * makespan)) * 100
        occupied_core_seconds is the WALL-CLOCK union of each core's busy
        intervals (overlapping segments counted once) -- the core-seconds the
        provider actually ties up, as opposed to runtime which double-counts
        oversubscribed/preempted segments. Headroom is the idle core capacity the
        scheduler frees up for the provider relative to CFS.

  color  Latency increase vs CFS (0s -> 95+s)
        = max(0, mean_per_task_latency - CFS mean_per_task_latency)
        the per-task wait a point adds on top of CFS (clipped at 0), so anything
        no worse than CFS sits at the green end of the scale.

Baselines (CFS, ALPS, Hybrid, cFIFO) are drawn as diamonds ("heuristic") and the
scx_rl_exec run + slice-sweep as circles ("synthesized"), coloured by the latency
increase over CFS. Hybrid's headroom is pinned to 0 and any point that costs the
client more than CFS (x < 0) is dropped.

    python3 project/plots/provider_client.py            # -> provider_client.{png,pdf}
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[1] / "results"

NCORES = 50                       # scx_rl_exec workers pinned to cores 0-49
LATENCY_MAX_S = 95.0              # colorbar top ("95+s"); latencies are clipped here
THIN_FRAC = 0.03                  # declutter dense clusters (0 = keep every point)

# Baselines shown as labelled diamonds. cFIFO is displayed as "FIFO/cFIFO".
BASELINES = {
    "CFS": "scx_cfs",
    "ALPS": "scx_alps",
    "Hybrid": "scx_hybrid",
    "FIFO": "scx_cfifo",
    "SFS": "scx_sfs",
}
RL_DIR = "scx_rl_exec"

LAT_CMAP = matplotlib.colormaps["viridis_r"]

TASKS_RE = re.compile(r"^tasks:\s*(\d+)", re.M)
LAT_RE = re.compile(r"^accumulated task latency:\s*([0-9.]+)\s*s", re.M)
RUN_RE = re.compile(r"^accumulated task runtime:\s*([0-9.]+)\s*s", re.M)
MK_RE = re.compile(r"^makespan:\s*([0-9.]+)\s*s", re.M)


def parse_metrics(path):
    t = path.read_text(encoding="utf-8")
    return {
        "tasks": int(TASKS_RE.search(t).group(1)),
        "latency": float(LAT_RE.search(t).group(1)),
        "runtime": float(RUN_RE.search(t).group(1)),
        "makespan": float(MK_RE.search(t).group(1)),
    }


def occupied_core_seconds(path):
    """Wall-clock union of busy intervals summed over cores (overlaps once)."""
    per = defaultdict(list)
    with open(path) as f:
        for ln in f:
            if not ln or ln[0] == "#":
                continue
            p = ln.split()
            if len(p) < 5:
                continue
            per[int(p[0])].append((float(p[2]), float(p[3])))
    total = 0.0
    for intervals in per.values():
        intervals.sort()
        cur_end = float("-inf")
        for s, e in intervals:
            if s > cur_end:
                total += e - s
                cur_end = e
            elif e > cur_end:
                total += e - cur_end
                cur_end = e
    return total


def read_point(dir_path, workload):
    mpath = dir_path / f"{workload}_metrics"
    opath = dir_path / f"{workload}_occupancy"
    if not mpath.exists() or not opath.exists():
        return None
    m = parse_metrics(mpath)
    m["occ"] = occupied_core_seconds(opath)
    m["spare"] = (1.0 - m["occ"] / (NCORES * m["makespan"])) * 100.0
    m["mean_latency"] = m["latency"] / m["tasks"]
    return m


def load_all(workload, results_dir, preempt_fill=True):
    cfs = read_point(results_dir / BASELINES["CFS"], workload)
    if cfs is None:
        raise RuntimeError("CFS baseline metrics/occupancy not found.")
    cfs_wait = cfs["mean_latency"]  # CFS mean per-task wait, the color baseline

    def finish(m, label):
        m["x"] = (cfs["runtime"] - m["runtime"]) / cfs["runtime"] * 100.0  # cost impr
        raw_headroom = m["spare"] - cfs["spare"]                            # headroom vs CFS
        m["y"] = 0.0 if label == "Hybrid" else raw_headroom
        # Color encodes the per-task wait INCREASE over CFS (clipped at 0), not the
        # absolute mean latency -- schedulers no worse than CFS sit at the green end.
        m["lat_color"] = max(0.0, m["mean_latency"] - cfs_wait)
        return m

    baselines = []
    for label, d in BASELINES.items():
        m = read_point(results_dir / d, workload)
        if m is None:
            print(f"warning: missing {d}; skipping {label}")
            continue
        m["label"] = label
        baselines.append(finish(m, label))

    rl = []
    # The main scx_rl_exec run is drawn as a synthesized circle (unlabelled).
    main_rl = read_point(results_dir / RL_DIR, workload)
    if main_rl is not None:
        main_rl["slice_ms"] = None
        rl.append(finish(main_rl, "RL-exec"))
    for d in sorted((results_dir / RL_DIR / "slice_pareto").glob("s*")):
        mm = re.match(r"s(\d+)$", d.name)
        if mm is None:
            continue
        slice_ms = int(mm.group(1))
        m = read_point(d, workload)
        if m is None:
            continue
        m["slice_ms"] = slice_ms
        rl.append(finish(m, "RL-exec"))

    # Preemption-budget fillers (results/scx_rl_exec/preempt_pareto/
    # s<BASE>[p<POST>]n<N>): the operating points that populate the voids the pure
    # slice sweep skips. Same "RL-exec" series, so they merge into the synthesized
    # curve. Control anchors (n>=99999) are skipped. Default on (only test_day02 has
    # this data; other workloads read nothing here).
    if preempt_fill:
        for d in sorted((results_dir / RL_DIR / "preempt_pareto").glob("s*n*")):
            mm = re.match(r"s(\d+)(?:p(\d+))?n(\d+)$", d.name)
            if mm is None or int(mm.group(3)) >= 99999:
                continue
            m = read_point(d, workload)
            if m is None:
                continue
            m["slice_ms"] = int(mm.group(1))
            rl.append(finish(m, "RL-exec"))

    # Keep only operating points that don't cost the client more than CFS.
    baselines = [p for p in baselines if p["x"] >= 0]
    rl = [p for p in rl if p["x"] >= 0]
    rl.sort(key=lambda p: p["x"])
    return baselines, rl


def thin(points, min_frac):
    """Drop points closer than min_frac of the x/y diagonal to the last kept one."""
    if min_frac <= 0 or len(points) <= 2:
        return points
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    xspan = (max(xs) - min(xs)) or 1.0
    yspan = (max(ys) - min(ys)) or 1.0
    ordered = sorted(points, key=lambda p: p["x"])
    kept = [ordered[0]]
    for p in ordered[1:-1]:
        dx = (p["x"] - kept[-1]["x"]) / xspan
        dy = (p["y"] - kept[-1]["y"]) / yspan
        if (dx * dx + dy * dy) ** 0.5 >= min_frac:
            kept.append(p)
    kept.append(ordered[-1])
    return kept

def plot(baselines, rl, outfile):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20})
    
    fig, ax = plt.subplots(figsize=(12, 7.5))
    EDGE, SPINE = "#222222", "#33475b"
    norm = Normalize(vmin=0.0, vmax=LATENCY_MAX_S)

    def scatter(points, marker, size):
        ax.scatter([p["x"] for p in points], [p["y"] for p in points],
                   c=[min(p["lat_color"], LATENCY_MAX_S) for p in points],
                   cmap=LAT_CMAP, norm=norm, marker=marker, s=size,
                   edgecolors=EDGE, linewidths=1.3, zorder=3)

    scatter(rl, "o", 360)
    scatter(baselines, "D", 300)

    offs = {"CFS": (-6, 26, "right", "bottom"), "ALPS": (10, 16, "left", "bottom"),
            "Hybrid": (10, 12, "left", "bottom"), "FIFO": (0, -18, "center", "top"),
            "SFS": (10, 12, "left", "bottom")}
    for p in baselines:
        dx, dy, ha, va = offs.get(p["label"], (8, 8, "left", "bottom"))
        ax.annotate(p["label"], (p["x"], p["y"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=24,
                    color="#1a1a1a", zorder=4)

    ax.set_xlim(-2, 100)
    ax.set_ylim(-1, 21) 
    ax.set_yticks(range(0, 21, 5)) 
    
    ax.set_xticks(range(0, 101, 20))
    pct = FuncFormatter(lambda v, _: f"{int(v)}%")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Client Cost Improvement vs CFS (%)", fontsize=27)
    ax.set_ylabel("Provider Headroom vs CFS (%)", fontsize=27)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, color="#c9d2da", zorder=0)
    ax.set_axisbelow(True)
    
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(1.6)
        
    ax.tick_params(colors="#222222", labelsize=24)

    heur = mlines.Line2D([], [], marker="D", linestyle="None", markersize=16,
                         markerfacecolor="#1a9d4f", markeredgecolor=EDGE,
                         markeredgewidth=1.3, label="Heuristic")
    synth = mlines.Line2D([], [], marker="o", linestyle="None", markersize=17,
                          markerfacecolor="#8dbf2e", markeredgecolor=EDGE,
                          markeredgewidth=1.3, label="Synthesized")
    
    leg = ax.legend(handles=[heur, synth], loc="center left", frameon=False,
                    fontsize=20, labelspacing=0.5, handletextpad=0.2,
                    columnspacing=1.2, handlelength=1.0, ncol=2,
                    bbox_to_anchor=(0.02, 0.94))
    leg.set_zorder(6)

    cax = ax.inset_axes([0.65, 0.925, 0.32, 0.03]) 
    cax.set_zorder(7)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=LAT_CMAP), cax=cax,
                      orientation="horizontal")
    cb.outline.set_edgecolor("#c9d2da")
    
    cb.set_ticks([0, LATENCY_MAX_S])
    cb.set_ticklabels(["0s", "95+s"])
    cax.tick_params(labelsize=18, colors="#6b7c8c", length=0, pad=3)

    cax.text(-0.06, 0.5, "Δ Latency\nvs CFS", transform=cax.transAxes, fontsize=16,
             color="#6b7c8c", fontweight="bold", ha="right", va="center", zorder=7)

    fig.tight_layout()
    root = outfile.with_suffix("")
    fig.savefig(root.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(root.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {root.with_suffix('.png')} and {root.with_suffix('.pdf')}")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", default="test_day02")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=HERE / "provider_client.png")
    ap.add_argument("--thin-frac", type=float, default=THIN_FRAC,
                    help="declutter dense synthesized clusters; 0 keeps every point")
    ap.add_argument("--preempt-fill", action=argparse.BooleanOptionalAction, default=True,
                    help="include the preemption-budget fillers from "
                         "results/scx_rl_exec/preempt_pareto (default on; only "
                         "test_day02 has this data). --no-preempt-fill for slice-only.")
    args = ap.parse_args()

    baselines, rl = load_all(args.workload, args.results_dir, preempt_fill=args.preempt_fill)
    rl = thin(rl, args.thin_frac)

    for p in baselines:
        print(f"{p['label']:11s} x={p['x']:6.1f}%  y={p['y']:6.1f}%  "
              f"lat={p['mean_latency']:5.1f}s")
    for p in rl:
        tag = "RL-exec" if p["slice_ms"] is None else f"s{p['slice_ms']}"
        print(f"{tag:<9s} x={p['x']:6.1f}%  y={p['y']:6.1f}%  "
              f"lat={p['mean_latency']:5.1f}s (+{p['lat_color']:4.1f}s vs CFS)")
    plot(baselines, rl, args.out)


if __name__ == "__main__":
    main()