#!/usr/bin/env python3
"""Remap the remote test traces to LOCAL fib levels of equal duration.

Reuses the same local fib calibration as calib_remap.py (the isolated scx_cfs
run in results/scx_cfs/calib_trace). Handles both trace formats:

  plain : "<inter_arrival> <fib_level>"            -> remap the fib level
  hetero: "<inter_arrival> <kind> <level>"         -> remap ONLY cpu_fib lines;
          io_sleep_read (a fixed level-ms sleep, machine-independent) and
          mem_stream (no fib calibration available) are passed through unchanged.

Column 0 (inter-arrival) and any kind token are preserved verbatim. Writes a
"<name>_local.txt" next to each input.
"""
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_DUR_MS = [
    9, 11, 13, 16, 17, 23, 34, 52, 81, 126,
    209, 218, 353, 531, 931, 1391, 2454, 3654,
    6422, 9605, 16687, 25280, 43169, 66502,
]
ALL_FIB = list(range(29, 53))
REMOTE_DUR = dict(zip(ALL_FIB, ALL_DUR_MS))

PLAIN = ["trace_test_day02.txt", "trace_test_day10.txt", "trace_test_noon.txt"]
HETERO = ["trace_test_day02_hetero.txt", "trace_test_day10_hetero.txt",
          "trace_test_noon_hetero.txt"]

LINE_RE = re.compile(
    r"^(TaskNew|FirstRun|TaskDead): (\S+) (?:enqueue at|starts at|at) "
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$"
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ts_ns(base, frac, off):
    off = "+00:00" if off == "Z" else off
    d = datetime.fromisoformat(base + off).astimezone(timezone.utc) - EPOCH
    return (d.days * 86400 + d.seconds) * 1_000_000_000 + int((frac or "").ljust(9, "0"))


def load_remote_to_local():
    """Build the remote-fib -> local-fib map from the calibration run."""
    log_path = os.path.normpath(
        os.path.join(HERE, "..", "..", "results", "scx_cfs", "calib_trace"))
    levels = [int(x) for x in open(os.path.join(HERE, "calib_levels.txt"))]
    tasks = {}
    for line in open(log_path, encoding="utf-8"):
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        event, label, base, frac, off = m.groups()
        t = tasks.setdefault(label, {})
        if event == "FirstRun":
            t["first"] = ts_ns(base, frac, off)
        elif event == "TaskDead":
            t["dead"] = ts_ns(base, frac, off)

    durs = {}
    for i, lvl in enumerate(levels):
        t = tasks.get(f"C{i}")
        if t and "first" in t and "dead" in t:
            durs.setdefault(lvl, []).append((t["dead"] - t["first"]) / 1e6)
    local_dur = {lvl: min(v) for lvl, v in durs.items()}   # min = least-interfered
    cand = sorted(local_dur)

    remap = {}
    for f in ALL_FIB:
        remap[f] = min(cand, key=lambda c: abs(local_dur[c] - REMOTE_DUR[f]))
    return remap


def remap_fib(remap, level):
    if level not in remap:                       # clamp out-of-range to nearest known
        level = min(ALL_FIB, key=lambda x: abs(x - level))
    return remap[level]


def process(remap, infile, hetero):
    out = []
    n_fib = n_other = 0
    for line in open(os.path.join(HERE, infile)):
        parts = line.split()
        if not parts:
            continue
        if hetero:
            ia, kind, lvl = parts[0], parts[1], int(parts[2])
            if kind == "cpu_fib":
                out.append(f"{ia} {kind} {remap_fib(remap, lvl)}\n")
                n_fib += 1
            else:                                # io_sleep_read / mem_stream: unchanged
                out.append(f"{ia} {kind} {lvl}\n")
                n_other += 1
        else:
            ia, lvl = parts[0], int(parts[-1])
            out.append(f"{ia} {remap_fib(remap, lvl)}\n")
            n_fib += 1

    outfile = infile[:-4] + "_local.txt"
    with open(os.path.join(HERE, outfile), "w") as f:
        f.writelines(out)
    tag = f"  (cpu_fib remapped: {n_fib}, passthrough: {n_other})" if hetero else ""
    print(f"  {infile} -> {outfile}  [{len(out)} lines]{tag}")


def main():
    remap = load_remote_to_local()
    print("remote_fib -> local_fib:", {f: remap[f] for f in ALL_FIB if f <= 50})
    print("plain traces:")
    for f in PLAIN:
        process(remap, f, hetero=False)
    print("hetero traces (only cpu_fib remapped; io_sleep_read/mem_stream unchanged):")
    for f in HETERO:
        process(remap, f, hetero=True)


if __name__ == "__main__":
    main()
