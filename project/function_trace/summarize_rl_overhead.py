#!/usr/bin/env python3
"""Summarize the RL inference + Rust<->Python communication overhead.

The agent prints one line per episode:

  scx_rl_<x>: overhead_metrics ... py_action_count=.. py_action_wall_us=.. \
      py_action_cpu_us=.. py_action_rt_us=.. py_slice_count=.. py_slice_wall_us=.. \
      py_slice_cpu_us=.. py_slice_rt_us=..

We report two costs per decision kind ("action" = train_step, "slice" =
infer_time_slice):
  * cpu  -- time.thread_time(): the agent thread's CPU time = pure RL inference
            compute (machine-load independent).
  * comm -- rt - wall = the Rust<->Python COMMUNICATION overhead (pyo3/GIL
            boundary glue), where rt is the full Rust->Python->Rust round-trip
            timed on the Rust side and wall is perf_counter inside the Python
            call. rt and wall are intermediates -- only their difference (comm)
            is reported.

Usage: summarize_rl_overhead.py <run_log> <sched_name> <out_dir>
Writes <out_dir>/overhead.txt and <out_dir>/overhead.csv.
"""
import re
import sys


def parse_log(path):
    fields, tasks = None, None
    with open(path) as f:
        for line in f:
            m = re.search(r"Loaded (\d+) tasks", line)
            if m:
                tasks = int(m.group(1))
            if "overhead_metrics" in line:
                fields = dict(re.findall(r"(\w+)=(\d+)", line))
            m = re.search(r"^tasks:\s*(\d+)", line)
            if m and tasks is None:
                tasks = int(m.group(1))
    if fields is None:
        raise SystemExit(f"no overhead_metrics line in {path}")
    return fields, tasks


def main():
    run_log, sched, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    f, tasks = parse_log(run_log)
    tasks = tasks or 0

    # comm = rt - wall. Tolerate older logs lacking the wall/rt fields (comm -> 0).
    kinds = {}
    for k in ("action", "slice"):
        c = int(f[f"py_{k}_count"])
        cpu = int(f[f"py_{k}_cpu_us"])
        wall = int(f.get(f"py_{k}_wall_us", 0))
        rt = int(f.get(f"py_{k}_rt_us", 0))
        kinds[k] = dict(count=c, cpu_us=cpu, comm_us=max(rt - wall, 0))

    lines = []
    lines.append(f"RL inference (cpu) + Rust<->Python communication (comm) overhead — {sched}")
    lines.append(f"workload tasks: {tasks}   episodes: 1")
    lines.append("cpu  = thread_time (pure inference compute)")
    lines.append("comm = round_trip - python_wall = pyo3/GIL boundary (Rust<->Python)")
    lines.append("=" * 74)
    lines.append(f"  {'Kind':<10}{'Calls':>9}{'CPU/call (ms)':>16}{'Comm/call (ms)':>17}")
    lines.append("  " + "-" * 62)
    tot_cpu = tot_comm = 0
    for k, d in kinds.items():
        c = d["count"]
        cpu, comm = d["cpu_us"], d["comm_us"]
        tot_cpu += cpu
        tot_comm += comm
        cpu_pc = cpu / c / 1e3 if c else 0.0
        comm_pc = comm / c / 1e3 if c else 0.0
        lines.append(f"  {k:<10}{c:>9}{cpu_pc:>16.3f}{comm_pc:>17.3f}")
    lines.append("  " + "-" * 62)
    lines.append(f"  per-episode total (s):  cpu {tot_cpu/1e6:.3f}   comm {tot_comm/1e6:.3f}")
    if tasks:
        lines.append(f"  per task (us):          cpu {tot_cpu/tasks:.2f}   comm {tot_comm/tasks:.2f}")
    out = "\n".join(lines) + "\n"

    with open(f"{out_dir}/overhead.txt", "w") as fh:
        fh.write(out)

    with open(f"{out_dir}/overhead.csv", "w") as fh:
        fh.write("sched,kind,tasks,calls,"
                 "cpu_per_call_us,comm_per_call_us,"
                 "cpu_episode_s,comm_episode_s,"
                 "cpu_per_task_us,comm_per_task_us\n")
        for k, d in kinds.items():
            c = d["count"]
            cpu, comm = d["cpu_us"], d["comm_us"]
            fh.write(f"{sched},{k},{tasks},{c},"
                     f"{cpu/c if c else 0:.3f},{comm/c if c else 0:.3f},"
                     f"{cpu/1e6:.6f},{comm/1e6:.6f},"
                     f"{cpu/tasks if tasks else 0:.3f},{comm/tasks if tasks else 0:.3f}\n")

    print(out, end="")


if __name__ == "__main__":
    main()
