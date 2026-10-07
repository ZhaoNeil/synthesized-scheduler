#!/usr/bin/env python3
"""Generate an isolated fib-level calibration trace for the LOCAL machine.

Each fib level 29..54 is launched several times under scx_cfs on the 50 worker
cores. Inter-arrival gaps are sized to ~est_dur(level)/CONCURRENCY so only a
handful of same-level tasks overlap; with 50 cores and idle-first placement each
lands on its own core, so its FirstRun->TaskDead execution time is the level's
intrinsic CPU service time (no contention).

Writes:
  calib_trace.txt   -- "<inter_arrival_s> <level>" lines (fed to replay_trace)
  calib_levels.txt  -- one level per line, index-aligned to the per-task log's
                       C0,C1,... task labels (used by calib_remap.py)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Standalone naive-fib probe durations (ms) on this box, levels 29..45 measured,
# 46..54 extrapolated at the asymptotic ~1.47x/level growth. Used ONLY to size
# arrival gaps; the real durations come from the scx_cfs run.
EST_MS = {
    29: 2.083, 30: 3.052, 31: 4.828, 32: 7.149, 33: 10.370, 34: 16.151,
    35: 24.265, 36: 43.334, 37: 62.843, 38: 116.558, 39: 164.823, 40: 312.376,
    41: 432.910, 42: 819.135, 43: 1139.979, 44: 2135.224, 45: 3015.018,
}
for lvl in range(46, 55):
    EST_MS[lvl] = EST_MS[lvl - 1] * 1.47

LEVELS = list(range(29, 55))           # 29..54 (covers remote dur range 9..66502 ms)
REPS = {lvl: (6 if lvl <= 49 else 2) for lvl in LEVELS}
CONCURRENCY = 6                        # target same-level overlap (<< 50 cores)


def main():
    lines = []      # (inter_arrival_s, level)
    levels = []     # level per task index, aligned to C0,C1,...
    prev_gap = 0.0  # inter-arrival before the very first task is 0
    for lvl in LEVELS:
        gap_s = max(EST_MS[lvl] / 1000.0 / CONCURRENCY, 0.0005)
        for _ in range(REPS[lvl]):
            lines.append((prev_gap, lvl))
            levels.append(lvl)
            prev_gap = gap_s

    with open(os.path.join(HERE, "calib_trace.txt"), "w") as f:
        for ia, lvl in lines:
            f.write(f"{ia:.6f} {lvl}\n")
    with open(os.path.join(HERE, "calib_levels.txt"), "w") as f:
        for lvl in levels:
            f.write(f"{lvl}\n")

    total_gap = sum(ia for ia, _ in lines)
    print(f"tasks: {len(lines)}  levels: {LEVELS[0]}..{LEVELS[-1]}")
    print(f"sum of inter-arrival gaps: {total_gap:.1f} s "
          f"(+ ~{EST_MS[LEVELS[-1]]/1000:.0f}s for the last task to finish)")


if __name__ == "__main__":
    main()
