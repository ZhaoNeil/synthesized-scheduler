#!/usr/bin/env python3
"""Calibrate the hetero test traces: ALL three kinds at original level L should
have local duration ~= all_dur_list[L-29] ms (the fib duration at that level).

  cpu_fib       -> local fib level nearest that duration   (calib_trace log)
  io_sleep_read -> sleep = target ms; launch_function sleeps level ms with no
                   tsv, so the output level IS the target ms (machine-independent)
  mem_stream    -> stream U MB s.t. local duration ~= target; written as level
                   U+28 (launch_function: units = level-28). U from the SERIAL
                   mem_stream calibration (calib_mem_trace log).

Inter-arrival (col 0) and the kind token are preserved verbatim. Plain traces
(cpu_fib only) are handled by calib_remap_all.py and not touched here.
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


def parse_durations(log_path, keys):
    """label -> FirstRun->TaskDead ms; returns {key: [ms,...]} via keys[index]."""
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
    by_key = {}
    for i, k in enumerate(keys):
        t = tasks.get(f"C{i}")
        if t and "first" in t and "dead" in t:
            by_key.setdefault(k, []).append((t["dead"] - t["first"]) / 1e6)
    return by_key


def fib_remap():
    log = os.path.normpath(os.path.join(HERE, "..", "..", "results", "scx_cfs", "calib_trace"))
    levels = [int(x) for x in open(os.path.join(HERE, "calib_levels.txt"))]
    durs = parse_durations(log, levels)
    local = {lvl: min(v) for lvl, v in durs.items()}
    cand = sorted(local)
    return {f: min(cand, key=lambda c: abs(local[c] - REMOTE_DUR[f])) for f in ALL_FIB}


def mem_curve():
    """Sorted (units, isolated_dur_ms) from the serial mem_stream calibration."""
    log = os.path.normpath(os.path.join(HERE, "..", "..", "results", "scx_cfs", "calib_mem_trace"))
    units = [int(x) for x in open(os.path.join(HERE, "calib_mem_units.txt"))]
    durs = parse_durations(log, units)
    pts = sorted((u, min(v)) for u, v in durs.items())
    # enforce monotonic non-decreasing duration (guard against measurement noise)
    mono = []
    for u, d in pts:
        if mono and d <= mono[-1][1]:
            d = mono[-1][1] + 1e-6
        mono.append((u, d))
    return mono


def units_for(curve, target_ms):
    """Invert the (units,dur) curve: smallest-error units for target duration."""
    if target_ms <= curve[0][1]:
        return curve[0][0]
    if target_ms >= curve[-1][1]:
        return curve[-1][0]
    for (u0, d0), (u1, d1) in zip(curve, curve[1:]):
        if d0 <= target_ms <= d1:
            u = u0 + (u1 - u0) * (target_ms - d0) / (d1 - d0)
            return max(1, int(round(u)))
    return curve[-1][0]


def clamp_fib(fmap, level):
    if level not in fmap:
        level = min(ALL_FIB, key=lambda x: abs(x - level))
    return fmap[level]


def main():
    fmap = fib_remap()
    curve = mem_curve()
    print(f"mem_stream curve: {len(curve)} points, "
          f"units {curve[0][0]}..{curve[-1][0]}, "
          f"dur {curve[0][1]:.1f}..{curve[-1][1]:.0f} ms")

    # Precompute per-original-level outputs for io/mem and report achieved error.
    print("\nlevel -> target_ms | io_sleep(level=ms) | mem_stream(units,->lvl)")
    io_out, mem_out = {}, {}
    for L in ALL_FIB:
        T = REMOTE_DUR[L]
        io_out[L] = max(1, int(round(T)))            # sleep = T ms
        u = units_for(curve, T)
        mem_out[L] = u + 28
        # achieved mem duration at u (interp) for the error report
        ach = next((d for uu, d in curve if uu == u), None)
        print(f"  {L} -> {T:6d} | io {io_out[L]:6d} | mem U={u:7d} lvl={mem_out[L]:7d}")

    for infile in HETERO:
        out, c_fib, c_io, c_mem = [], 0, 0, 0
        for line in open(os.path.join(HERE, infile)):
            p = line.split()
            if not p:
                continue
            ia, kind, lvl = p[0], p[1], int(p[2])
            L = lvl if lvl in ALL_FIB else min(ALL_FIB, key=lambda x: abs(x - lvl))
            if kind == "cpu_fib":
                out.append(f"{ia} {kind} {clamp_fib(fmap, lvl)}\n"); c_fib += 1
            elif kind == "io_sleep_read":
                out.append(f"{ia} {kind} {io_out[L]}\n"); c_io += 1
            elif kind == "mem_stream":
                out.append(f"{ia} {kind} {mem_out[L]}\n"); c_mem += 1
            else:
                out.append(line if line.endswith("\n") else line + "\n")
        outfile = infile[:-4] + "_local.txt"
        with open(os.path.join(HERE, outfile), "w") as f:
            f.writelines(out)
        print(f"{outfile}: {len(out)} lines  (fib {c_fib}, io {c_io}, mem {c_mem})")


if __name__ == "__main__":
    main()
