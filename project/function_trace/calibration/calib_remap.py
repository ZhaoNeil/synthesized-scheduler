#!/usr/bin/env python3
"""Remap remote trace.txt fib levels to LOCAL levels of equal duration.

Inputs:
  results/scx_cfs/calib_trace  -- per-task log from the isolated scx_cfs run
  calib_levels.txt             -- level per task index (C0,C1,...)
  trace.txt                    -- remote trace: "<inter_arrival> <remote_fib>"

For each remote level f, the intended (remote) duration is all_dur_list[f-29] ms.
We measure the LOCAL intrinsic duration of every calibrated level g as the MIN
FirstRun->TaskDead execution time across its reps (min = least-interfered =
pure CPU service time). Then for each remote line we keep column 0 (inter-arrival)
unchanged and replace the level with the local g whose local duration is closest
to the remote target.

Writes:
  trace_local.txt  (column 0 identical to trace.txt; column 1 = local level)
  calibration_table.txt  (human-readable: remote level/dur -> local level/dur)
"""
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# Remote calibration (function duration in ms <-> fib N).
ALL_DUR_MS = [
    9, 11, 13, 16, 17, 23, 34, 52, 81, 126,
    209, 218, 353, 531, 931, 1391, 2454, 3654,
    6422, 9605, 16687, 25280, 43169, 66502,
]
ALL_FIB = list(range(29, 53))          # 29..52
REMOTE_DUR = dict(zip(ALL_FIB, ALL_DUR_MS))

LINE_RE = re.compile(
    r"^(TaskNew|FirstRun|TaskDead): (\S+) (?:enqueue at|starts at|at) "
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$"
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ts_ns(base, frac, off):
    off = "+00:00" if off == "Z" else off
    d = datetime.fromisoformat(base + off).astimezone(timezone.utc) - EPOCH
    return (d.days * 86400 + d.seconds) * 1_000_000_000 + int((frac or "").ljust(9, "0"))


def parse_log(path):
    """label -> {first_run_ns, complete_ns}"""
    tasks = {}
    for line in open(path, encoding="utf-8"):
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        event, label, base, frac, off = m.groups()
        t = tasks.setdefault(label, {})
        if event == "FirstRun":
            t["first"] = ts_ns(base, frac, off)
        elif event == "TaskDead":
            t["dead"] = ts_ns(base, frac, off)
    return tasks


def main():
    log_path = os.path.join(HERE, "..", "..", "results", "scx_cfs", "calib_trace")
    log_path = os.path.normpath(log_path)
    levels = [int(x) for x in open(os.path.join(HERE, "calib_levels.txt"))]
    tasks = parse_log(log_path)

    # Per local level: collect FirstRun->TaskDead execution times (ms).
    durs_by_level = {}
    for i, lvl in enumerate(levels):
        t = tasks.get(f"C{i}")
        if not t or "first" not in t or "dead" not in t:
            continue
        durs_by_level.setdefault(lvl, []).append((t["dead"] - t["first"]) / 1e6)

    if not durs_by_level:
        sys.exit(f"no task durations parsed from {log_path}")

    # Intrinsic local duration = MIN across reps (least-interfered).
    local_dur = {lvl: min(v) for lvl, v in sorted(durs_by_level.items())}
    cand_levels = sorted(local_dur)

    print("local calibration (FirstRun->TaskDead, ms):")
    for lvl in cand_levels:
        v = durs_by_level[lvl]
        print(f"  fib {lvl}: min={local_dur[lvl]:9.2f}  "
              f"(n={len(v)}, max={max(v):9.2f})")

    # Build remote-level -> local-level map by nearest duration.
    remote_to_local = {}
    table = []
    for f in ALL_FIB:
        target = REMOTE_DUR[f]
        g = min(cand_levels, key=lambda c: abs(local_dur[c] - target))
        remote_to_local[f] = g
        table.append((f, target, g, local_dur[g]))

    print("\nremote fib (target ms) -> local fib (local ms):")
    for f, target, g, ld in table:
        print(f"  {f} ({target:6d}) -> {g}  ({ld:8.2f})   "
              f"err {(ld - target) / target * 100:+6.1f}%")

    # Rewrite trace.txt: keep column 0, remap the level.
    out_lines = []
    n_unmapped = 0
    with open(os.path.join(HERE, "trace.txt")) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            ia = parts[0]
            rfib = None
            for tok in parts[1:]:
                try:
                    rfib = int(tok)
                except ValueError:
                    pass
            if rfib is None:
                continue
            if rfib not in remote_to_local:
                n_unmapped += 1
                # clamp to nearest known remote level
                rfib = min(ALL_FIB, key=lambda x: abs(x - rfib))
            out_lines.append(f"{ia} {remote_to_local[rfib]}\n")

    with open(os.path.join(HERE, "trace_local.txt"), "w") as f:
        f.writelines(out_lines)
    with open(os.path.join(HERE, "calibration_table.txt"), "w") as f:
        f.write("remote_fib remote_dur_ms local_fib local_dur_ms err_pct\n")
        for fa, target, g, ld in table:
            f.write(f"{fa} {target} {g} {ld:.2f} {(ld-target)/target*100:.1f}\n")

    print(f"\nwrote trace_local.txt ({len(out_lines)} lines), "
          f"calibration_table.txt"
          + (f"  [{n_unmapped} lines had out-of-range remote levels, clamped]"
             if n_unmapped else ""))


if __name__ == "__main__":
    main()
