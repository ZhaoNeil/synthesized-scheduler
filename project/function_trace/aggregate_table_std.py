#!/usr/bin/env python3
"""Aggregate the rerun_table_std.sh repetitions into mean +/- std.

Reads every ``<OUT>/rep<r>/<Policy>/<workload>_metrics`` produced by
rerun_table_std.sh and reports, per (workload, policy):
  * accumulated task latency  (s)  -- mean +/- sample std over reps
  * accumulated task runtime  (s)  -- mean +/- sample std over reps
  * Cost (CFS = 1)                 -- runtime / CFS_runtime, normalised WITHIN
                                      each repetition, then mean +/- std

Cost is a positive-linear rescale of runtime
normalised to CFS: for each rep the policy's runtime is divided by that same
rep's CFS runtime. If CFS was not re-run in a rep (e.g. a synthesized-only
sweep), the committed results/scx_cfs/<workload>_metrics runtime is used as a
constant denominator instead (reported in the log).

Outputs (next to the reps, under OUT):
  raw_all.csv       every rep's raw latency/runtime/cost
  summary_std.csv   per (workload, policy): mean/std/n for each metric
It also prints a human-readable table (mean +/- std).
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

LAT_RE = re.compile(r"^accumulated task latency:\s*([0-9.]+)\s*s", re.MULTILINE)
RUN_RE = re.compile(r"^accumulated task runtime:\s*([0-9.]+)\s*s", re.MULTILINE)

# Policy order + display names + grouping.
POLICY_ORDER = [
    "CFS", "EEVDF", "ALPS", "S_Latency",
    "Hybrid", "S_Intermediate",
    "FIFO", "SFS", "S_Runtime",
]
DISPLAY = {
    "CFS": "CFS", "EEVDF": "EEVDF", "ALPS": "ALPS",
    "S_Latency": "Synthesized Latency",
    "Hybrid": "Hybrid",
    "S_Intermediate": "Synthesized Intermediate",
    "FIFO": "FIFO", "SFS": "SFS",
    "S_Runtime": "Synthesized Runtime",
}
REGION = [
    ("Low-latency region", ["CFS", "EEVDF", "ALPS", "S_Latency"]),
    ("Intermediate region", ["Hybrid", "S_Intermediate"]),
    ("Low-runtime region", ["FIFO", "SFS", "S_Runtime"]),
]
# workload name -> display number
WORKLOAD_NUM = {
    "test_day02_hetero": 4,
    "test_noon_hetero": 5,
    "test_day10_hetero": 6,
}
CFS_POLICY = "CFS"


def parse_metrics(path):
    """Return (latency_s, runtime_s) or None if the file is missing/partial."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    lat, run = LAT_RE.search(text), RUN_RE.search(text)
    if not (lat and run):
        return None
    return float(lat.group(1)), float(run.group(1))


def mean_std(values):
    """(mean, sample-std, n). std is None when fewer than 2 samples."""
    n = len(values)
    if n == 0:
        return None, None, 0
    m = statistics.fmean(values)
    s = statistics.stdev(values) if n >= 2 else None
    return m, s, n


def collect(out_dir):
    """data[workload][policy] = {rep: (lat, run)}; also the set of reps seen."""
    data, reps = {}, set()
    for rep_dir in sorted(out_dir.glob("rep*")):
        m = re.match(r"rep(\d+)$", rep_dir.name)
        if not m:
            continue
        rep = int(m.group(1))
        for pol_dir in rep_dir.iterdir():
            if not pol_dir.is_dir():
                continue
            policy = pol_dir.name
            for mf in pol_dir.glob("*_metrics"):
                workload = mf.name[: -len("_metrics")]
                parsed = parse_metrics(mf)
                if parsed is None:
                    print(f"  warn: unparseable/empty {mf} -- skipped")
                    continue
                data.setdefault(workload, {}).setdefault(policy, {})[rep] = parsed
                reps.add(rep)
    return data, sorted(reps)


def cfs_reference_runtime(workload, results_dir):
    """Committed CFS runtime for a workload, used as a constant cost denominator
    when CFS was not re-run in a given rep. None if unavailable."""
    parsed = parse_metrics(results_dir / "scx_cfs" / f"{workload}_metrics")
    return parsed[1] if parsed else None


def fnum(x, decimals=0):
    if x is None:
        return "NA"
    if decimals == 0:
        return f"{x:,.0f}"
    return f"{x:.{decimals}f}"


def pm(mean, std, decimals=0):
    """'mean ± std' formatting; drops the ± term when std is undefined (n<2)."""
    if mean is None:
        return "NA"
    if std is None:
        return fnum(mean, decimals)
    return f"{fnum(mean, decimals)} ± {fnum(std, decimals)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="out_dir", type=Path,
                    default=REPO_ROOT / "results" / "std",
                    help="the OUT tree written by rerun_table_std.sh (default results/std/)")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                    help="committed results/ (for the CFS cost denominator fallback)")
    args = ap.parse_args()

    if not args.out_dir.exists():
        raise SystemExit(f"no such directory: {args.out_dir}")

    data, reps = collect(args.out_dir)
    if not data:
        raise SystemExit(f"no rep*/<policy>/<workload>_metrics found under {args.out_dir}")
    print(f"reps found: {reps}   workloads: {sorted(data)}")

    # ---- per-rep cost = runtime / CFS_runtime(same rep, else committed CFS) ----
    raw_rows = []          # (workload, policy, rep, lat, run, cost)
    agg = {}               # (workload, policy) -> {'lat':[], 'run':[], 'cost':[]}
    for workload, per_policy in data.items():
        ref_cfs = cfs_reference_runtime(workload, args.results_dir)
        cfs_by_rep = {rep: v[1] for rep, v in per_policy.get(CFS_POLICY, {}).items()}
        for policy, per_rep in per_policy.items():
            bucket = agg.setdefault((workload, policy), {"lat": [], "run": [], "cost": []})
            for rep, (lat, run) in sorted(per_rep.items()):
                denom = cfs_by_rep.get(rep, ref_cfs)
                cost = run / denom if denom else None
                raw_rows.append((workload, policy, rep, lat, run, cost))
                bucket["lat"].append(lat)
                bucket["run"].append(run)
                if cost is not None:
                    bucket["cost"].append(cost)

    # ---- raw_all.csv -----------------------------------------------------------
    raw_path = args.out_dir / "raw_all.csv"
    with raw_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["workload", "workload_num", "policy", "display", "rep",
                     "latency_s", "runtime_s", "cost"])
        for workload, policy, rep, lat, run, cost in sorted(
                raw_rows, key=lambda r: (WORKLOAD_NUM.get(r[0], 99), r[0],
                                         POLICY_ORDER.index(r[1]) if r[1] in POLICY_ORDER else 99,
                                         r[2])):
            wr.writerow([workload, WORKLOAD_NUM.get(workload, ""), policy,
                         DISPLAY.get(policy, policy), rep,
                         f"{lat:.6f}", f"{run:.6f}",
                         "" if cost is None else f"{cost:.6f}"])
    print(f"wrote {raw_path}")

    # ---- summary_std.csv + console table ---------------------------------------
    summary_path = args.out_dir / "summary_std.csv"
    sf = summary_path.open("w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["workload_num", "workload", "policy", "display", "n",
                 "latency_mean_s", "latency_std_s",
                 "runtime_mean_s", "runtime_std_s",
                 "cost_mean", "cost_std"])

    workloads_sorted = sorted(data, key=lambda w: (WORKLOAD_NUM.get(w, 99), w))
    for workload in workloads_sorted:
        num = WORKLOAD_NUM.get(workload, "?")
        print(f"\n=== Workload {num}  ({workload})  n_reps={reps} ===")
        print(f"  {'Policy':26s} {'Latency (s) mean±std':>26s} "
              f"{'Runtime (s) mean±std':>26s} {'Cost mean±std':>20s}")
        # REGION only fixes the row order/grouping; the region name is not printed.
        for _region_name, policies in REGION:
            for policy in policies:
                bucket = agg.get((workload, policy))
                if not bucket:
                    continue
                lm, ls, ln = mean_std(bucket["lat"])
                rm, rs, _ = mean_std(bucket["run"])
                cm, cs, _ = mean_std(bucket["cost"])
                disp = DISPLAY.get(policy, policy)
                print(f"  {disp:26s} {pm(lm, ls):>26s} {pm(rm, rs):>26s} "
                      f"{pm(cm, cs, 4):>20s}   (n={ln})")
                sw.writerow([num, workload, policy, disp, ln,
                             f"{lm:.6f}" if lm is not None else "",
                             f"{ls:.6f}" if ls is not None else "",
                             f"{rm:.6f}" if rm is not None else "",
                             f"{rs:.6f}" if rs is not None else "",
                             f"{cm:.6f}" if cm is not None else "",
                             f"{cs:.6f}" if cs is not None else ""])
    sf.close()
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
