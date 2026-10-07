#!/usr/bin/env python3
"""
rl_sim.py -- a fast, pure-Python scheduling SIMULATOR for iterating on the
scx_rl_res / scx_rl_exec policies WITHOUT loading a real sched_ext scheduler.

It runs the *exact same* `py/q_learning.py` the production agent embeds (pick
which with --policy), but replaces the real workload + BPF backend with a
light discrete-time model of: trace arrivals -> a global queue -> the DQN's
per-task core placement -> per-core FIFO run-queues whose heads run until
completion or slice expiry -> completion, plus the same two-level cadence
(placement gate + a slow loop that
sets the learned slice, the load-adaptive run-to-completion mode, and RL
rebalance).

Because everything is in one Python process and decoupled from wall-clock, it
runs many episodes in seconds and the agent persists across episodes naturally
(no per-episode process restart / checkpoint reload). Tune reward/state/network
in q_learning.py here, then validate the winner on real hardware via rl_res.sh /
rl_exec.sh.

It reports the SAME metrics replay_trace does:
  * accumulated task latency  = sum(first_run - arrival)      [queueing]
  * accumulated task runtime  = sum(taskdead - first_run)     [run span incl.
                                                                preemption gaps]
  * makespan                  = max(taskdead) - min(arrival)

NOTE: this is an APPROXIMATION (no kernel overheads, cache, NUMA, real timing).
Use it for *relative* policy comparison and reward shaping; trust absolute
numbers only after calibrating --calls-per-sec against one real run, and always
confirm the final policy on the real scheduler.

Examples:
  ./rl_sim.py --policy res  --episodes 50
  ./rl_sim.py --policy exec --episodes 50 --trace trace_local.txt
  ./rl_sim.py --policy res  --episodes 1 --eval   # greedy, load best, no train
"""

import argparse
import csv
import math
import os
import shutil
import sys
import time
import types

PHI = (1.0 + 5.0 ** 0.5) / 2.0
PLACE_MS = 20.0
SLOW_LOOP_MS = 500.0
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def fib_calls(n):
    """Number of recursive calls naive Fibonacci(n) makes ~ its CPU cost.
    calls(n) = 2*Fib(n+1) - 1 (each non-base call spawns two children)."""
    if n <= 1:
        return 1
    a, b = 1, 1  # Fib(1), Fib(2)
    for _ in range(n - 1):
        a, b = b, a + b  # after loop, b = Fib(n+1)
    return 2 * b - 1


def parse_trace(path):
    """Return [(arrival_seconds, level), ...] from a `<inter_arrival> [kind] <level>`
    trace; arrival is the cumulative inter-arrival time."""
    tasks = []
    t = 0.0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                ia = float(parts[0])
            except ValueError:
                continue
            t += ia
            level = None
            for tok in parts[1:]:
                try:
                    level = int(tok)
                except ValueError:
                    pass  # skip 'cpu_fib' etc.; keep the last integer
            if level is None:
                continue
            tasks.append((t, level))
    return tasks


def sim_objective(policy, result):
    if policy == "res":
        return "latency", result["latency_s"]
    return "runtime", result["runtime_s"]


def best_metric_from_csv(policy, path):
    if not os.path.exists(path):
        return None
    col = "acc_task_latency_s" if policy == "res" else "acc_task_runtime_s"
    best_value = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if col not in row:
                continue
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                continue
            if best_value is None or value < best_value:
                best_value = value
    return best_value


class Task:
    __slots__ = ("tid", "arrival", "service", "rem", "run_since", "first_run", "core")

    def __init__(self, tid, arrival, service):
        self.tid = tid
        self.arrival = arrival
        self.service = service
        self.rem = service
        self.run_since = 0.0   # CPU time since last (re)dispatch; vs slice
        self.first_run = None
        self.core = None


class SchedSim:
    """One episode of the discrete-time scheduling model."""

    def __init__(self, opts, trace):
        self.opts = opts
        self.ncores = opts.cores
        self.dt = PLACE_MS / 1000.0
        self.slow = SLOW_LOOP_MS / 1000.0
        self.init_slice = opts.slice_ms / 1000.0
        self.rtc_slice = 1e6  # run-to-completion: effectively no preemption
        # If set, the slice is held constant (slice DQN + load-adaptive mode
        # disabled) so the placement policy can be tuned in isolation.
        self.fixed_slice = opts.fixed_slice_ms / 1000.0 if opts.fixed_slice_ms > 0 else None
        # Per-level service time (seconds): naive-fib call count / calls_per_sec.
        self.trace = [
            (arr, lvl, fib_calls(lvl) / opts.calls_per_sec) for (arr, lvl) in trace
        ]
        self.slice_candidates = [int(ms) for ms in getattr(opts, "slice_candidates", [])]

    def run(self, ql, scxbus, training):
        opts = self.opts
        ncores = self.ncores
        dt = self.dt
        # Tasks sorted by arrival; released into global_q as time advances.
        arrivals = sorted(
            (Task(i, arr, svc) for i, (arr, _lvl, svc) in enumerate(self.trace)),
            key=lambda t: t.arrival,
        )
        ai = 0
        ntasks = len(arrivals)
        cores = [[] for _ in range(ncores)]   # per-core FIFO queues of Task
        global_q = []                          # arrived, not yet placed
        slice_s = self.fixed_slice if self.fixed_slice is not None else self.init_slice
        preempt = 0
        sim_t = 0.0
        last_slow = 0.0
        done = 0
        lat_sum = 0.0
        rt_sum = 0.0
        min_arr = arrivals[0].arrival if arrivals else 0.0
        max_dead = 0.0
        slice_hist = {ms: 0 for ms in self.slice_candidates}
        slice_slow_ticks = 0
        slice_dqn_updates = 0
        slice_rtc_ticks = 0
        slice_other_updates = 0
        last_place_reward_t = 0.0
        last_slice_reward_t = 0.0

        def set_snapshot(extra_pending=0):
            rq = [len(c) for c in cores]
            running = sum(1 for c in cores if c)  # cores actively running a task
            not_started_on_cores = sum(
                1 for q in cores for t in q if t.first_run is None
            )
            started_unfinished = sum(
                1 for q in cores for t in q if t.first_run is not None
            )
            started_waiting = sum(
                1
                for q in cores
                for i, t in enumerate(q)
                if i > 0 and t.first_run is not None
            )
            scxbus._latest = {
                "started_tasks": running,
                "unstarted_tasks": len(global_q) + extra_pending + not_started_on_cores,
                "finished_tasks": done,
                "started_unfinished": started_unfinished,
                "started_waiting": started_waiting,
                "sum_res": 0,
                "sum_exec": 0,
                "sum_preempt": preempt,
                "rq_sizes": rq + [0] * (ncores - len(rq)) if len(rq) < ncores else rq,
            }

        def waiting_first_run():
            return len(global_q) + sum(
                1 for q in cores for t in q if t.first_run is None
            )

        def drain_place_reward(now):
            nonlocal last_place_reward_t
            reward = -float(waiting_first_run()) * max(0.0, now - last_place_reward_t)
            last_place_reward_t = now
            return reward

        def drain_slice_reward(now):
            nonlocal last_slice_reward_t
            reward = -float(waiting_first_run()) * max(0.0, now - last_slice_reward_t)
            last_slice_reward_t = now
            return reward

        def request_actions(k, extra_pending=0):
            if k <= 0:
                return []
            set_snapshot(extra_pending=extra_pending)
            actions = ql.train_step(
                batch_size=k,
                reward=drain_place_reward(sim_t),
                done=False,
            )
            return actions if isinstance(actions, list) else [actions]

        max_steps = opts.max_steps
        steps = 0
        while steps < max_steps:
            steps += 1
            # 1) Release arrivals due by now.
            while ai < ntasks and arrivals[ai].arrival <= sim_t:
                global_q.append(arrivals[ai])
                ai += 1

            # 2) Placement gate: ask the DQN where to place the queued tasks.
            if global_q:
                k = min(len(global_q), opts.batch_cap)
                if opts.placement == "jsq":
                    # Oracle: join-shortest-queue. Under FIFO + 8ms round-robin a
                    # new task's first-run wait ~ (#tasks ahead on its core) x slice,
                    # so least-loaded placement is the mechanism's latency-optimal
                    # core choice. Bypasses the DQN -> the floor the learned
                    # placement should aim for.
                    for _ in range(k):
                        t = global_q.pop(0)
                        c = min(range(ncores), key=lambda x: len(cores[x]))
                        t.core = c
                        t.run_since = 0.0
                        cores[c].append(t)
                else:
                    actions = request_actions(k)
                    for j in range(min(k, len(actions))):
                        t = global_q.pop(0)
                        c = actions[j]
                        if not (0 <= c < ncores):
                            # fall back to the least-loaded core
                            c = min(range(ncores), key=lambda x: len(cores[x]))
                        t.core = c
                        t.run_since = 0.0
                        cores[c].append(t)

            # 3) Execute each core over this simulation window. The FIFO head
            #    accumulates run time across windows and is preempted only when
            #    its slice is exhausted.
            for c in range(ncores):
                q = cores[c]
                step_budget = dt
                now = sim_t
                guard = 0
                while step_budget > 1e-12 and q and guard < 100000:
                    guard += 1
                    head = q[0]
                    if head.first_run is None:
                        head.first_run = now
                        lat_sum += now - head.arrival
                    run = min(step_budget, slice_s - head.run_since, head.rem)
                    if run <= 0:
                        run = min(step_budget, head.rem)  # slice already exhausted edge
                    head.rem -= run
                    head.run_since += run
                    step_budget -= run
                    now += run
                    if head.rem <= 1e-12:
                        q.pop(0)
                        rt_sum += now - head.first_run
                        if now > max_dead:
                            max_dead = now
                        done += 1
                    elif head.run_since >= slice_s - 1e-12:
                        # slice expired -> preempt: requeue at this core's tail
                        q.pop(0)
                        head.run_since = 0.0
                        q.append(head)
                        preempt += 1

            sim_t += dt

            # 4) Fixed 500 ms slow-loop tick: learned slice + load-adaptive mode
            #    + RL rebalance.
            #    With --fixed-slice-ms the slice DQN and load-adaptive mode are
            #    disabled (slice held constant) to isolate the placement policy;
            #    RL rebalance still runs.
            if sim_t - last_slow >= self.slow:
                last_slow = sim_t
                slice_slow_ticks += 1
                if self.fixed_slice is None:
                    active = sum(len(c) for c in cores) + len(global_q)
                    first_wait = waiting_first_run()
                    if 0 < active and (active < ncores or first_wait == 0):
                        # Low load, or no task is still waiting for first CPU
                        # time. In the `res` objective, more preemption after
                        # that point only hurts runtime/cache behavior.
                        drain_slice_reward(sim_t)
                        slice_s = self.rtc_slice
                        slice_rtc_ticks += 1
                    elif active >= ncores:
                        set_snapshot()
                        # The slice head computes its reward from the snapshot
                        # via compute_slice_reward (same path as the real
                        # agent). Keep draining the timer so its bookkeeping
                        # stays sane.
                        drain_slice_reward(sim_t)
                        ms = ql.infer_time_slice()
                        if ms and ms > 0:
                            slice_dqn_updates += 1
                            ms_key = int(round(ms))
                            if ms_key in slice_hist:
                                slice_hist[ms_key] += 1
                            else:
                                slice_other_updates += 1
                            slice_s = ms / 1000.0
                if ncores >= 2 and opts.placement != "jsq":
                    self._rl_rebalance(cores, request_actions)

            # 5) Termination: nothing left to arrive, queue, or run.
            if ai >= ntasks and not global_q and not any(cores):
                break

        # Terminal transition so the agent records the episode end + checkpoints.
        set_snapshot()
        ql.train_step(batch_size=1, reward=drain_place_reward(sim_t), done=True)
        ret = ql.get_and_reset_episode_return()
        makespan = max(0.0, max_dead - min_arr)
        return {
            "return": ret,
            "latency_s": lat_sum,
            "runtime_s": rt_sum,
            "makespan_s": makespan,
            "done": done,
            "ntasks": ntasks,
            "preempt": preempt,
            "steps": steps,
            "slice_slow_ticks": slice_slow_ticks,
            "slice_dqn_updates": slice_dqn_updates,
            "slice_rtc_ticks": slice_rtc_ticks,
            "slice_other_updates": slice_other_updates,
            "slice_fixed_ms": opts.fixed_slice_ms if opts.fixed_slice_ms > 0 else 0.0,
            "slice_hist": slice_hist,
        }

    def _rl_rebalance(self, cores, request_actions):
        """ghOSt-style RLRebalanceTasks: find overloaded cores, shed queued tasks
        off their TAIL, and let the DQN (train_step) re-pick a fresh target core
        for each."""
        ncores = len(cores)
        loads = [(c, len(cores[c])) for c in range(ncores)]
        min_load = min(l for _, l in loads)
        min_cpu = min(range(ncores), key=lambda c: len(cores[c]))
        loads.sort(key=lambda x: -x[1])  # most-loaded first
        cap = self.opts.rebalance_max
        plan, total = [], 0
        for c, load in loads:
            trigger = (min_load == 0 and load >= 2) or (load > 10 and load - min_load >= 5)
            if not trigger:
                break  # sorted desc: nothing after this qualifies either
            to_move = max(1, (load - min_load) // 2)
            if total + to_move > cap:
                to_move = cap - total
            if to_move > 0:
                plan.append((c, to_move))
                total += to_move
            if total >= cap:
                break
        if total <= 0:
            return
        # Shed from each overloaded core's TAIL (newest queued tasks), never the
        # running head (index 0).
        shed = []
        for c, to_move in plan:
            for _ in range(to_move):
                if len(cores[c]) > 1:
                    shed.append(cores[c].pop())
        if not shed:
            return
        # Re-run the DQN for the shed tasks' fresh targets (same call the agent
        # uses for placement -> this is also a training step).
        actions = request_actions(len(shed), extra_pending=len(shed))
        for i, t in enumerate(shed):
            c = actions[i] if i < len(actions) and 0 <= actions[i] < ncores else min_cpu
            t.run_since = 0.0
            cores[c].append(t)

def main():
    ap = argparse.ArgumentParser(description="Fast simulator for the scx_rl_* policies")
    ap.add_argument("--policy", choices=["res", "exec"], default="res",
                    help="which crate's py/q_learning.py to train (res=latency, exec=runtime)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--trace", default="trace_local.txt", help="trace file (under this dir or absolute)")
    ap.add_argument("--cores", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice-ms", type=float, default=3000.0, help="initial RR slice (until the slice DQN sets one)")
    ap.add_argument("--fixed-slice-ms", type=float, default=0.0,
                    help="hold the RR slice constant at this value and DISABLE the slice DQN + "
                         "load-adaptive mode (isolates the placement policy). 0 = use the slice DQN. "
                         "A large value (e.g. 100000) ~ run-to-completion / no preemption.")
    ap.add_argument("--calls-per-sec", type=float, default=8.7e8,
                    help="fib recursive calls per second (service time = fib_calls(level)/this). "
                         "Calibrated to real hardware: real scx_cfifo is run-to-completion, so its "
                         "'accumulated task runtime' is the pure total service demand. "
                         "sum(fib_calls(level))/cFIFO_runtime gives 8.92e8 (day02), 8.70e8 (noon), "
                         "8.61e8 (day10) -- consistent across traces, mean ~8.7e8. Re-calibrate if the trace's fib "
                         "levels are remapped for new hardware.")
    ap.add_argument("--batch-cap", type=int, default=256)
    ap.add_argument("--rebalance-max", type=int, default=50, help="max tasks migrated per RL rebalance pass")
    ap.add_argument("--placement", choices=["dqn", "jsq"], default="dqn",
                    help="dqn = learned placement (default); jsq = join-shortest-queue oracle "
                         "(least-loaded core, DQN + RL-rebalance bypassed) to measure the "
                         "mechanism's latency floor the learned placement should aim for.")
    ap.add_argument("--eval", action="store_true", help="greedy eval (load *_best.pth, no training/checkpoint writes)")
    ap.add_argument("--resume", action="store_true", help="load *.pth before training (continue)")
    ap.add_argument("--model-dir", default=None,
                    help="checkpoint dir (default results/sim/<crate>, user-owned). To run the real "
                         "scheduler on a sim-trained model, point its MODEL_DIR here.")
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--out", default=None, help="metrics CSV (default <model-dir>/train_metrics.csv or eval_metrics.csv)")
    opts = ap.parse_args()

    crate = "scx_rl_res" if opts.policy == "res" else "scx_rl_exec"
    tag = "res" if opts.policy == "res" else "exec"
    py_dir = os.path.join(REPO_ROOT, "scheds", "experimental", crate, "py")
    # Default outputs to a user-owned dir (NOT the crate py/, which a prior `sudo`
    # real run may have left root-owned), so the sim always runs as a normal user.
    model_dir = opts.model_dir or os.path.join(REPO_ROOT, "results", "sim", crate)
    os.makedirs(model_dir, exist_ok=True)
    best = os.path.join(model_dir, f"{tag}_best.pth")
    latest = os.path.join(model_dir, f"{tag}.pth")
    trace_path = opts.trace if os.path.isabs(opts.trace) else os.path.join(HERE, opts.trace)
    metrics_name = "eval_metrics.csv" if opts.eval else "train_metrics.csv"
    out_csv = opts.out or os.path.join(model_dir, metrics_name)

    # Inject a fake `scxbus` module BEFORE importing q_learning (which imports it).
    scxbus = types.ModuleType("scxbus")
    scxbus._latest = None
    scxbus.get_latest = lambda: scxbus._latest
    sys.modules["scxbus"] = scxbus

    sys.path.insert(0, py_dir)
    import q_learning as ql  # the policy under test

    ql.set_model_dir(model_dir)
    ql.set_seed(opts.seed)
    ql.reset_agent()
    if opts.eval:
        ql.load_trained_model(best)
    elif opts.resume:
        ql.load_trained_model(latest)
    ql.set_training(not opts.eval)
    if not opts.eval and hasattr(getattr(ql, "_global_agent", None), "best_return"):
        # The q_learning modules save *_best.pth by return. In the simulator,
        # *_best.pth is reserved for the metric objective below: latency for
        # res, runtime for exec.
        ql._global_agent.best_return = float("inf")
    opts.slice_candidates = [
        int(ms)
        for ms in getattr(getattr(ql, "_global_agent", None), "slice_candidates", [])
    ]

    trace = parse_trace(trace_path)
    sim = SchedSim(opts, trace)

    total_work = sum(svc for (_a, _l, svc) in sim.trace)
    print(f"=== rl_sim: policy={opts.policy} ({crate}), {len(trace)} tasks from {os.path.basename(trace_path)} ===")
    print(f"    cores={opts.cores}  total work={total_work:.1f}s  ideal makespan~{total_work/opts.cores:.1f}s  "
          f"(calls_per_sec={opts.calls_per_sec:.2e}; tune --calls-per-sec to match a real run)")
    slice_desc = f"fixed {opts.fixed_slice_ms:.0f}ms (slice DQN OFF)" if opts.fixed_slice_ms > 0 else "slice DQN (learned)"
    print(f"    model_dir={model_dir}  mode={'eval' if opts.eval else ('train(resume)' if opts.resume else 'train(fresh)')}  slice={slice_desc}")

    f = w = None
    best_metric_value = None
    if opts.resume and not opts.eval:
        best_metric_value = best_metric_from_csv(opts.policy, out_csv)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Fresh train/eval runs truncate the CSV; --resume appends training runs.
    mode = "a" if (opts.resume and not opts.eval) else "w"
    write_header = (mode == "w") or (not os.path.exists(out_csv))
    f = open(out_csv, mode, newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(
            [
                "episode",
                "seed",
                "return",
                "acc_task_latency_s",
                "acc_task_runtime_s",
                "makespan_s",
                "preempt",
                "done",
                "wall_s",
                "slice_fixed_ms",
                "slice_slow_ticks",
                "slice_dqn_updates",
                "slice_rtc_ticks",
                "slice_other_updates",
            ]
            + [f"slice_ms_{ms}" for ms in opts.slice_candidates]
        )

    for ep in range(1, opts.episodes + 1):
        ql.reset_episode()
        t0 = time.time()
        r = sim.run(ql, scxbus, training=not opts.eval)
        wall = time.time() - t0
        metric_note = ""
        if not opts.eval:
            metric_name, metric_value = sim_objective(opts.policy, r)
            if best_metric_value is None or metric_value < best_metric_value:
                best_metric_value = metric_value
                if os.path.exists(latest):
                    shutil.copy2(latest, best)
                    metric_note = f" new_best_{metric_name}"
        slice_counts = ",".join(
            f"{ms}:{r['slice_hist'].get(ms, 0)}" for ms in opts.slice_candidates
        )
        print(f"ep {ep}/{opts.episodes}: return={r['return']:.3f} "
              f"latency={r['latency_s']:.1f}s runtime={r['runtime_s']:.1f}s makespan={r['makespan_s']:.1f}s "
              f"preempt={r['preempt']} done={r['done']}/{r['ntasks']} "
              f"slices=[{slice_counts}] rtc={r['slice_rtc_ticks']}{metric_note} ({wall:.1f}s wall)")
        w.writerow(
            [
                ep,
                opts.seed,
                f"{r['return']:.4f}",
                f"{r['latency_s']:.6f}",
                f"{r['runtime_s']:.6f}",
                f"{r['makespan_s']:.6f}",
                r["preempt"],
                r["done"],
                f"{wall:.2f}",
                f"{r['slice_fixed_ms']:.0f}",
                r["slice_slow_ticks"],
                r["slice_dqn_updates"],
                r["slice_rtc_ticks"],
                r["slice_other_updates"],
            ]
            + [r["slice_hist"].get(ms, 0) for ms in opts.slice_candidates]
        )
        f.flush()
    if f is not None:
        f.close()
        if opts.eval:
            print(f"=== done. eval metrics -> {out_csv} ===")
        else:
            print(f"=== done. metrics -> {out_csv}; checkpoints -> {latest} (best: {best}) ===")


if __name__ == "__main__":
    main()
