#!/usr/bin/env python3
"""Generate a *_preempt.csv next to every *_occupancy file under results/.

    python3 project/plots/occupancy_analyze.py            # all files under results/
    python3 project/plots/occupancy_analyze.py <file>...  # only the given ones

Each row is one inferred same-core preemption (a task starting on a core while
another is still resident there); the CSV's own header explains the columns.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results"
SUFFIX = "_occupancy"


def load(path):
    """Load an occupancy file into parallel arrays (cores, tasks, starts, ends).
    A task label "C<idx>" is stored as its integer index."""
    cores, tasks, starts, ends = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            cores.append(int(parts[0]))
            tasks.append(int(parts[1][1:]) if parts[1][:1] == "C" else int(parts[1]))
            starts.append(float(parts[2]))
            ends.append(float(parts[3]))
    return (np.array(cores), np.array(tasks),
            np.array(starts, float), np.array(ends, float))


def preemption_events(cores, tasks, starts, ends):
    """Return preemption event rows (core, time, incoming, preempted,
    resident_before), in timeline order. On each core, an interval start landing
    while others are still open is a preemption; the bumped task is the
    most-recently-started still-open one (top of the stack); resident_before is
    the set of tasks already resident on the core when incoming starts (its size
    == the concurrency-just-before)."""
    rows = []
    for c in np.unique(cores):
        idx = np.nonzero(cores == c)[0]
        n = len(idx)
        times = np.concatenate([starts[idx], ends[idx]])
        is_start = np.concatenate([np.ones(n, np.int8), np.zeros(n, np.int8)])
        which = np.concatenate([idx, idx])
        open_ = {}  # gi -> (task, start) for intervals currently resident on core c
        # ends before starts at equal time, so a clean handoff (end == next start)
        # is not miscounted as a preemption.
        for k in np.lexsort((is_start, times)):
            gi = int(which[k])
            if is_start[k]:
                if open_:
                    victim = open_[max(open_, key=lambda j: open_[j][1])][0]
                    resident = sorted(v[0] for v in open_.values())
                    rows.append((int(c), float(times[k]), int(tasks[gi]),
                                 int(victim), resident))
                open_[gi] = (int(tasks[gi]), float(starts[gi]))
            else:
                open_.pop(gi, None)
    rows.sort(key=lambda r: (r[1], r[0]))  # by time, then core
    return rows


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("# same-core preemption events inferred from overlapping occupancy "
                "intervals.\n")
        f.write("# preempted_task = most-recently-started still-resident task (best "
                "guess of the\n")
        f.write("# runner that was bumped); resident_before = '|'-joined tasks "
                "already resident\n")
        f.write("# on the core when incoming started. times are seconds since first "
                "TaskNew; the\n")
        f.write("# contention envelope, not exact deschedule/resume instants.\n")
        f.write("core,time_s,incoming_task,preempted_task,resident_before\n")
        for c, t, inc, vic, res in rows:
            f.write(f"{c},{t:.6f},C{inc},C{vic},"
                    f"{'|'.join('C' + str(x) for x in res)}\n")


def main():
    args = sys.argv[1:]
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(RESULTS.rglob("*" + SUFFIX))
    if not files:
        sys.exit(f"no *{SUFFIX} files found under {RESULTS}")

    total = 0
    for f in files:
        if not f.name.endswith(SUFFIX):
            print(f"  skip  {f}  (not a *{SUFFIX} file)", file=sys.stderr)
            continue
        out = f.with_name(f.name[:-len(SUFFIX)] + "_preempt.csv")
        cores, tasks, starts, ends = load(f)
        if len(cores) == 0:
            print(f"  skip  {f}  (no intervals)", file=sys.stderr)
            continue
        rows = preemption_events(cores, tasks, starts, ends)
        write_csv(rows, out)
        total += 1
        print(f"  {len(rows):>9} events  ->  {out}")
    print(f"done: wrote {total} *_preempt.csv")


if __name__ == "__main__":
    main()
