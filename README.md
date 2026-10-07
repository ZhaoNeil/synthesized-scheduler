# (Near-)Pareto-Optimal Synthesized OS-Level Scheduling for Serverless Systems

This repository contains the code and experiment harness for the SoCC '26 paper
[**"(Near-)Pareto-Optimal Synthesized OS-Level Scheduling for Serverless
Systems"**](https://doi.org/10.1145/3828161.3857877).

It is a fork of [sched-ext/scx](https://github.com/sched-ext/scx). All schedulers
in this work are user-space `sched_ext` schedulers built on
[`scx_rustland_core`](rust/scx_rustland_core): a small BPF backend relays runnable
tasks to a Rust agent, which makes every scheduling decision.

## Upstream base

This tree is based on sched-ext/scx commit
[`e29bb76`](https://github.com/sched-ext/scx/commit/e29bb76422a2cf5dfafb15f8cb6344af4a7b13c8)
(2026-05-28, workspace version 1.1.1). Everything outside the paths listed
below is unmodified upstream code. The original upstream README is kept as
[README.scx.md](README.scx.md).

## What this work adds

### New schedulers (`scheds/experimental/`)

| Scheduler | Role in the paper | Description |
|---|---|---|
| [`scx_rl_res`](scheds/experimental/scx_rl_res) | Synth. Latency | RL (Dueling-DQN) agent optimizing accumulated task latency: learns per-task core placement and the preemption time slice |
| [`scx_rl_exec`](scheds/experimental/scx_rl_exec) | Synth. Runtime; with a fixed slice, Synth. Intermediate and the Pareto sweeps | Same agent, reward targets accumulated task runtime |
| [`scx_cfs`](scheds/experimental/scx_cfs) | CFS | Per-CPU CFS model (weighted vruntime, load balancing) |
| [`scx_eevdf`](scheds/experimental/scx_eevdf) | EEVDF | Per-CPU EEVDF model (virtual deadlines, eligibility) |
| [`scx_cfifo`](scheds/experimental/scx_cfifo) | FIFO | Centralized global-queue FIFO, run-to-completion |
| [`scx_hybrid`](scheds/experimental/scx_hybrid) | Hybrid | FIFO tier for short tasks + CFS tier for long ones |
| [`scx_alps`](scheds/experimental/scx_alps) | ALPS | Port of [ALPS](https://github.com/ds2-lab/ALPS) (USENIX ATC '24) |
| [`scx_sfs`](scheds/experimental/scx_sfs) | SFS | Port of [SFS](https://github.com/ds2-lab/SFS) (SC '22) |
| [`scx_rr`](scheds/experimental/scx_rr) | Round-Robin | Round-robin over per-core queues, with optional preemption, power-of-two choices, and work shuffling |

The RL agents embed CPython via `pyo3`; the learning code is in each crate's
`py/q_learning.py`. The trained models used in the paper (5 seeds each) are in
`results/scx_rl_res/seed_*/` and `results/scx_rl_exec/seed_*/`.

### Changes to upstream code

- [`rust/scx_rustland_core`](rust/scx_rustland_core) (BPF backend + Rust API):
  configurable watchdog timeout (`BpfScheduler::init_with_timeout`); per-CPU queue-length snapshot
  (`refresh_qlen` / `qlen`); a BPF-side task migration plan between per-CPU
  queues (`request_migration`); first-run latency counters
  (`nr_waiting_first_run`, `first_run_lat_ns`, `first_run_cnt`) used by the RL
  reward. The existing API is unchanged, so upstream schedulers built on
  `scx_rustland_core` need no modification.
- [`Cargo.toml`](Cargo.toml): adds the new schedulers to the workspace.

### Experiment harness (`project/`)

| Path | Contents |
|---|---|
| [`project/function_trace/`](project/function_trace) | Workload traces, the trace replayer, and scripts to run, train, and sweep the schedulers |
| [`project/plots/`](project/plots) | Plotting scripts for the paper figures |
| [`results/`](results) | Measured results used by the plotting scripts |

## Requirements

- A Linux kernel with `sched_ext` and a **raised watchdog limit**, plus root
  access to load schedulers. Under the oversubscribed workloads a task can stay
  runnable far longer than the stock `sched_ext` watchdog allows, so the harness
  loads every scheduler with a 600 s timeout (`SCX_TIMEOUT_MS=600000` in
  `run_workload.sh`). A stock kernel caps this at `SCX_WATCHDOG_MAX_TIMEOUT`
  (30 s) and rejects the scheduler at load time, so `SCX_WATCHDOG_MAX_TIMEOUT`
  in `kernel/sched/ext.c` must be raised and the kernel rebuilt.

## Build

```bash
cargo build --release -p scx_rl_res -p scx_rl_exec \
    -p scx_cfs -p scx_eevdf -p scx_cfifo -p scx_hybrid -p scx_alps -p scx_sfs -p scx_rr
```

## Running experiments

All harness scripts live in `project/function_trace/` and must run as root.
Each full trace replay takes about 5–6 minutes.

```bash
cd project/function_trace

# One scheduler on one workload -> results/scx_cfs/test_day02{,_metrics,...}
sudo SCHED=cfs ./run_workload.sh trace_test_day02_local.txt

# Train an RL scheduler (5 seeds x 50 episodes on trace_local.txt)
sudo ./train_rl_seeds.sh res
sudo ./train_rl_seeds.sh exec

# Evaluate every scheduler on every test workload (picks the best RL seed)
sudo ./run_all_test_workloads.sh

# Synthesized Pareto frontier of scx_rl_exec
sudo ./run_slice_pareto.sh
sudo ./run_preempt_pareto.sh
```

## Citation

```bibtex
@inproceedings{zhao2026pareto,
  author    = {Zhao, Yuxuan and Yang, Zhao and Weng, Weikang and Fran{\c{c}}ois-Lavet, Vincent and van Nieuwpoort, Rob and Uta, Alexandru},
  title     = {{(Near-)Pareto-Optimal Synthesized OS-Level Scheduling for Serverless Systems}},
  booktitle = {ACM Symposium on Cloud Computing V.2 (SoCC '26)},
  year      = {2026},
  month     = nov,
  location  = {Singapore, Singapore},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {14},
  isbn      = {979-8-4007-2816-7},
  doi       = {10.1145/3828161.3857877},
  url       = {https://doi.org/10.1145/3828161.3857877}
}
```

## License

GPL-2.0-only, the same as upstream scx; see [LICENSE](LICENSE). The ALPS and SFS
ports follow the designs of their original authors.
