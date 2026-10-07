#!/usr/bin/env python3
"""Generate a SERIAL mem_stream calibration trace for the LOCAL machine.

mem_stream is memory-bandwidth bound: concurrent tasks contend for bandwidth and
inflate each other's duration. To measure the *intrinsic* (isolated) duration of
streaming U MB we run the tasks one at a time (inter-arrival >= the previous
task's duration), so each owns the full memory subsystem -- the analogue of the
isolated fib calibration.

launch_function with no hetero_calibration.tsv computes units = level - 28, so a
task that should stream U MB is written as "<ia> mem_stream <U+28>".

Writes:
  calib_mem_trace.txt   -- "<ia> mem_stream <U+28>" lines
  calib_mem_units.txt   -- one U per line, index-aligned to C0,C1,... task labels
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def est_ms(u):
    """Rough isolated duration (ms) of streaming u MB, from a standalone probe.
    u<=64: single fresh buffer ~1.7 ms/MB (cold). u>64: 64MB buffer looped,
    warm, ~0.193 ms/MB marginal. Used ONLY to space arrivals."""
    return u * 1.7 if u <= 64 else 108.0 + (u - 64) * 0.193


def unit_ladder():
    us, u = set(), 1.0
    while u <= 170000:
        us.add(int(round(u)))
        u *= 1.33
    us.update([1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64])  # dense small end
    return sorted(us)


def reps(u):
    return 3 if u <= 64 else (2 if u <= 2048 else 1)


def main():
    units_ladder = unit_ladder()
    lines, units_seq, prev_gap = [], [], 0.0
    for u in units_ladder:
        gap_s = est_ms(u) * 1.10 / 1000.0          # serial: next arrives after prev done
        for _ in range(reps(u)):
            lines.append((prev_gap, u))
            units_seq.append(u)
            prev_gap = gap_s

    with open(os.path.join(HERE, "calib_mem_trace.txt"), "w") as f:
        for ia, u in lines:
            f.write(f"{ia:.6f} mem_stream {u + 28}\n")
    with open(os.path.join(HERE, "calib_mem_units.txt"), "w") as f:
        for u in units_seq:
            f.write(f"{u}\n")

    total = sum(ia for ia, _ in lines)
    print(f"tasks: {len(lines)}  units: {units_ladder[0]}..{units_ladder[-1]} "
          f"({len(units_ladder)} distinct)")
    print(f"serial wall estimate: ~{total + est_ms(units_ladder[-1])/1000:.0f} s")


if __name__ == "__main__":
    main()
