// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # scx_rl_res: a reinforcement-learning user-space scheduler
//!
//! ## Overview
//!
//! `scx_rl_res` ports the **ghOSt RL scheduler** (Dueling-DQN core placement + a
//! second DQN that learns the global time slice) onto `scx_rustland_core`, the
//! same user-space-agent framework as this project's other schedulers
//! (cFIFO/RR/CFS/EEVDF/ALPS/SFS). All policy lives in user space; the BPF
//! backend just relays runnable tasks up and dispatches what the agent decides.
//!
//! The learning itself is the **unchanged** PyTorch agent from the reference
//! implementation ([`py/q_learning.py`]): a `DuelingDQN` over a 56-dim state
//! (6 global features + 50 per-CPU run-queue lengths) whose action is *which
//! core* to place a newly-runnable, non-preempted task on, plus a small `DQN`
//! that picks the round-robin time slice from a fixed candidate set. We embed
//! CPython via `pyo3` and call that module directly, exactly as the ghOSt port
//! embedded it via pybind11 (`ghost_python_bridge.cc`). The Rust agent only
//! supplies the state (a snapshot the Python `scxbus` module reads) and enacts
//! the actions.
//!
//! ## Mapping the ghOSt design onto the agent model
//!
//!   * **State.** Each round the agent publishes a snapshot — per-CPU run-queue
//!     lengths (`rq_sizes[50]`) plus task counts — into a global the embedded
//!     `scxbus` module exposes to Python (mirroring ghOSt's `g_latest_snapshot`
//!     + `scxbus.get_latest()`).
//!
//!   * **Action = core placement.** The DQN's action space is the 50 worker cores
//!     0..=49 (all of them — no reserved core). The agent itself runs on a core
//!     *outside* the worker set (`--agent-cpu`, default 50) so its embedded torch
//!     never competes with scheduled tasks. The agent calls
//!     `train_step(batch_size=k)` to get one core per placeable task, then
//!     dispatches each to that core.
//!
//!   * **Per-core order + learned slice.** Within a core, tasks run FIFO (the DSQ
//!     is keyed by a monotonic arrival sequence) and are preempted after the
//!     learned time slice. A slice-expiry preemption is requeued at the tail of
//!     the task's current per-core queue instead of being placed by the DQN
//!     again. The slice is refreshed from `infer_time_slice()`.
//!
//!   * **Reward + online training.** Reward (penalising queued/unstarted tasks
//!     and load imbalance) and the DQN updates happen inside `q_learning.py`,
//!     driven purely by the snapshots and the per-round `train_step` calls. With
//!     `--eval` the agent loads `res_best.pth` and runs greedily; otherwise it
//!     trains online and checkpoints `res.pth`/`res_best.pth` on exit.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[rustfmt::skip]
mod bpf;

use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;

use anyhow::Context;
use anyhow::Result;
use bpf::*;
use clap::Parser;
use libbpf_rs::OpenObject;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::scx_enums;
use scx_utils::UserExitInfo;

/// Number of cores the DQN schedules over (input_dim = 6 + N_CORES, action space
/// = the N_CORES cores). Must match `N_CORES` in py/q_learning.py. All N_CORES
/// cores (0..N_CORES) are workers; the agent itself runs on a core *outside* this
/// range (see `--agent-cpu`).
const N_CORES: usize = 50;
/// Default core to pin the agent (and its embedded torch) onto: the first core
/// past the worker set, so the agent never competes with scheduled tasks.
const DEFAULT_AGENT_CPU: i32 = N_CORES as i32;

const NSEC_PER_MSEC: u64 = 1_000_000;
const IDLE_BACKOFF: Duration = Duration::from_micros(100);
const PLACE_INTERVAL_MS: u64 = 20;
const PLACE_TRIGGER_QLEN: usize = 50;
const SLOW_LOOP_INTERVAL_MS: u64 = 500;
/// Cap on tasks placed per `train_step` call (the Python side caps batch arrays
/// at 1000; stay well under).
const BATCH_CAP: usize = 256;
/// Cap on tasks migrated in one RL rebalance pass. Must match the BPF migration
/// plan capacity.
const GLOBAL_MAX_MIGRATE: usize = 50;

#[derive(Default)]
struct RlMetrics {
    place_batches: u64,
    place_tasks: u64,
    place_dqn_actions: u64,
    place_direct: u64,
    dispatched_tasks: u64,
    ring_full: u64,
    slow_ticks: u64,
    // Per-core DSQ backlog (sum of qlen over workers) sampled once per slow
    // tick: `qlen_peak` is the max seen, `qlen_acc` the running sum so the mean
    // (qlen_acc / slow_ticks) can be derived downstream. Lets us confirm how
    // much real backlog exists vs. the simulator's load and reason about the
    // saturation criterion.
    qlen_peak: u64,
    qlen_acc: u64,
    slice_dqn_updates: u64,
    slice_other_updates: u64,
    slice_ms_8: u64,
    slice_ms_32: u64,
    slice_ms_174: u64,
    slice_ms_390: u64,
    slice_ms_1633: u64,
    rebalance_moves: u64,
    // Full Rust->Python->Rust round-trip wall time (microseconds), measured on
    // the Rust side around the pyo3 call incl. GIL acquire + argument/return
    // marshaling. (round_trip - Python-side wall) = the pyo3/GIL boundary glue,
    // i.e. the Rust<->Python communication overhead reported as `comm`.
    action_rt_us: u64,
    slice_rt_us: u64,
}

impl RlMetrics {
    fn record_slice(&mut self, ms: i64) {
        match ms {
            8 => self.slice_ms_8 = self.slice_ms_8.wrapping_add(1),
            32 => self.slice_ms_32 = self.slice_ms_32.wrapping_add(1),
            174 => self.slice_ms_174 = self.slice_ms_174.wrapping_add(1),
            390 => self.slice_ms_390 = self.slice_ms_390.wrapping_add(1),
            1633 => self.slice_ms_1633 = self.slice_ms_1633.wrapping_add(1),
            _ => self.slice_other_updates = self.slice_other_updates.wrapping_add(1),
        }
    }
}

mod scx_bridge {
    //! A CPython extension module the embedded `q_learning.py` imports to read the
    //! latest scheduler snapshot — the `pyo3` analogue of the reference's
    //! `ghost_python_bridge.cc` (`PYBIND11_EMBEDDED_MODULE(scxbus, ...)`). The
    //! Rust agent publishes into a process-global slot (ghOSt's
    //! `g_latest_snapshot`) and `get_latest()` hands Python a dict of it.
    use super::N_CORES;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    #[derive(Clone)]
    pub struct Snapshot {
        pub started_tasks: i64,
        pub unstarted_tasks: i64,
        pub finished_tasks: i64,
        pub sum_res: i64,
        pub sum_exec: i64,
        pub sum_preempt: i64,
        pub rq_sizes: Vec<i32>,
    }

    static LATEST: OnceLock<Mutex<Option<Snapshot>>> = OnceLock::new();

    fn slot() -> &'static Mutex<Option<Snapshot>> {
        LATEST.get_or_init(|| Mutex::new(None))
    }

    /// Publish the snapshot Python's next `scxbus.get_latest()` will return.
    pub fn publish(s: Snapshot) {
        *slot().lock().unwrap() = Some(s);
    }

    #[pyfunction]
    fn get_latest(py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = slot().lock().unwrap();
        let Some(s) = guard.as_ref() else {
            return Ok(py.None());
        };
        let d = PyDict::new(py);
        d.set_item("started_tasks", s.started_tasks)?;
        d.set_item("unstarted_tasks", s.unstarted_tasks)?;
        d.set_item("finished_tasks", s.finished_tasks)?;
        d.set_item("sum_res", s.sum_res)?;
        d.set_item("sum_exec", s.sum_exec)?;
        d.set_item("sum_preempt", s.sum_preempt)?;
        // The DQN was trained on exactly N_CORES entries; pad/truncate to match.
        let mut rq = s.rq_sizes.clone();
        rq.resize(N_CORES, 0);
        d.set_item("rq_sizes", rq)?;
        Ok(d.into_any().unbind())
    }

    #[pymodule]
    pub fn scxbus(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(get_latest, m)?)?;
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(name = "scx_rl_res", about = "RL (Dueling-DQN) user-space scheduler: learned core placement + time slice")]
struct Opts {
    /// Worker CPUs that each own a per-CPU FIFO run-queue, as a CPU list (e.g.
    /// "0-49"). Set the workload's affinity to the same set. Defaults to cores
    /// 0..N_CORES (the DQN's action space). Keep these distinct from --agent-cpu.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// CPU to pin the agent (and its embedded torch inference/training) onto.
    /// Should be OUTSIDE the worker set so the agent never competes with
    /// scheduled tasks. Default = N_CORES (the first core past the workers). Set
    /// to -1 to leave the agent unpinned (kernel CFS places it). Also via AGENT_CPU.
    #[arg(long, env = "AGENT_CPU", default_value_t = DEFAULT_AGENT_CPU)]
    agent_cpu: i32,

    /// Initial round-robin time slice (milliseconds), used until the slice DQN
    /// produces its first value.
    #[arg(short = 's', long, default_value_t = 3_000)]
    slice_ms: u64,

    /// Directory holding/receiving the DQN checkpoints (res.pth / res_best.pth).
    /// Defaults to the crate's bundled py/ directory.
    #[arg(long)]
    model_dir: Option<String>,

    /// Directory containing q_learning.py. Defaults to the crate's py/ directory.
    #[arg(long)]
    py_dir: Option<String>,

    /// Evaluate only: load res_best.pth and act greedily (no online training,
    /// no checkpoint writes). Default is online training.
    #[arg(long, default_value_t = false)]
    eval: bool,

    /// Resume training from the latest checkpoint (res.pth) instead of starting
    /// from fresh random weights. Each agent run is one episode; with --resume,
    /// successive runs continue training (weights + optimizer + step count carry
    /// over). Harmless if res.pth doesn't exist yet (starts fresh). Ignored under
    /// --eval (which always loads res_best.pth).
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// RNG seed for the Python agent (torch/numpy/random).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Ablation: skip the periodic RL rebalance migration pass entirely. The DQN
    /// still does ALL initial placement; only the slow-loop migration is disabled.
    /// Set via NO_REBALANCE=1 (run_workload.sh maps it to this flag).
    #[arg(long = "no-rebalance", default_value_t = false)]
    no_rebalance: bool,

    /// Schedule the whole system. By default scx_rl_res runs in *partial* mode: it
    /// only schedules tasks that opt in (SCHED_EXT); everything else (including
    /// the agent + embedded Python/torch) stays on the kernel's scheduler.
    #[arg(long, default_value_t = false)]
    system_wide: bool,

    /// Print scheduling statistics once per second.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// sched_ext runnable-task watchdog timeout, in milliseconds. If any task
    /// stays runnable longer than this the kernel ejects the scheduler; an
    /// oversubscribed system can exceed the ~30 s default, so raise it. Also
    /// settable via SCX_TIMEOUT_MS.
    #[arg(long, env = "SCX_TIMEOUT_MS", default_value_t = 30_000)]
    timeout_ms: u32,
}

fn parse_cpu_list(s: &str) -> Result<Vec<i32>> {
    let mut cpus = std::collections::BTreeSet::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let a: i32 = a.trim().parse().with_context(|| format!("bad CPU range {part:?}"))?;
                let b: i32 = b.trim().parse().with_context(|| format!("bad CPU range {part:?}"))?;
                if a < 0 || a > b {
                    anyhow::bail!("invalid CPU range {part:?}");
                }
                cpus.extend(a..=b);
            }
            None => {
                let c: i32 = part.parse().with_context(|| format!("bad CPU id {part:?}"))?;
                if c < 0 {
                    anyhow::bail!("invalid CPU id {part:?}");
                }
                cpus.insert(c);
            }
        }
    }
    if cpus.is_empty() {
        anyhow::bail!("empty worker CPU list");
    }
    Ok(cpus.into_iter().collect())
}

/// Number of online CPUs, independent of the caller's affinity mask (unlike
/// `available_parallelism`, which would report 1 once the agent is pinned).
fn online_cpus() -> i32 {
    (unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } as i32).max(1)
}

fn resolve_workers(opts: &Opts) -> Result<Vec<i32>> {
    match &opts.workers {
        Some(s) => parse_cpu_list(s),
        None => {
            // Cores [0, N_CORES): the DQN's action space and state are exactly
            // that wide, so cores >= N_CORES are invisible to it and never
            // targeted (the agent core lives there, see --agent-cpu).
            let hi = online_cpus().min(N_CORES as i32);
            Ok((0..hi).collect())
        }
    }
}

fn default_py_dir() -> String {
    format!("{}/py", env!("CARGO_MANIFEST_DIR"))
}

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Worker CPUs, each owning one per-CPU FIFO run-queue.
    workers: Vec<i32>,
    is_worker: Vec<bool>,
    /// Monotonic arrival sequence used as the per-CPU DSQ key (=> FIFO order).
    seq: u64,
    /// Current round-robin time slice (ns), set by the slice DQN.
    slice_ns: u64,

    preempt_count: u64,
    metrics: RlMetrics,

    /// The embedded q_learning module (GIL-independent handle).
    ql: Py<PyAny>,
    training: bool,

    /// Slow loop (slice DQN + load-adaptive mode + RL rebalance) period and
    /// last-fire time.
    slow_loop_interval: Duration,
    last_slow_loop: Instant,

    /// Placement gate: the reference's fixed 20 ms fast loop and last-fire time.
    place_interval: Duration,
    last_place: Instant,

    /// Ablation flag: when true, skip the periodic RL rebalance migration pass.
    no_rebalance: bool,

    verbose: bool,
}

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
        let workers = resolve_workers(opts)?;
        if let Some(&hi) = workers.last() {
            if hi as usize >= N_CORES {
                eprintln!(
                    "scx_rl_res: warning: worker core {hi} >= N_CORES ({N_CORES}); the DQN's \
                     state/action space only covers cores [1, {N_CORES}), so cores >= {N_CORES} \
                     are invisible to it. Restrict --workers to 1-{}.",
                    N_CORES - 1
                );
            }
        }
        let max_cpu = *workers.last().unwrap_or(&0) as usize;
        let mut is_worker = vec![false; max_cpu.max(N_CORES - 1) + 1];
        for &c in &workers {
            is_worker[c as usize] = true;
        }

        let py_dir = opts.py_dir.clone().unwrap_or_else(default_py_dir);
        let model_dir = opts.model_dir.clone().unwrap_or_else(default_py_dir);
        let eval = opts.eval;
        let resume = opts.resume;
        let seed = opts.seed;

        // Bring up the embedded interpreter, load q_learning.py, and prime the
        // agent (seed, mode, optional pretrained weights). Mirrors the reference
        // LaunchPython setup, minus the workload-driving loop (the harness drives
        // the workload here, not the agent).
        let ql: Py<PyAny> = Python::attach(|py| -> Result<Py<PyAny>> {
            let sys = py.import("sys")?;
            let scxbus_mod = pyo3::types::PyModule::new(py, "scxbus")?;
            scx_bridge::scxbus(&scxbus_mod)?;
            sys.getattr("modules")?.set_item("scxbus", scxbus_mod)?;

            let path = sys.getattr("path")?;
            path.call_method1("insert", (0, py_dir.as_str()))?;

            let ql = py.import("q_learning")?;
            ql.call_method1("set_model_dir", (model_dir.as_str(),))?;
            ql.call_method1("set_seed", (seed,))?;
            ql.call_method0("reset_agent")?;
            if eval {
                // Eval: always load the best checkpoint and act greedily.
                let p = format!("{model_dir}/res_best.pth");
                ql.call_method1("load_trained_model", (p,))?;
            } else if resume {
                // Continue training from the latest checkpoint (weights +
                // optimizer + step count). load_trained_model no-ops with a
                // message if res.pth is absent (first episode).
                let p = format!("{model_dir}/res.pth");
                ql.call_method1("load_trained_model", (p,))?;
            }
            ql.call_method1("set_training", (!eval,))?;
            ql.call_method0("reset_episode")?;
            Ok(ql.into_any().unbind())
        })
        .context("failed to initialise embedded Python RL agent")?;

        // SCX_BPF_DEBUG=1 turns on libbpf's verbose loader/verifier log (useful to
        // diagnose a BPF load failure, e.g. a verifier complexity E2BIG).
        let bpf_debug = std::env::var("SCX_BPF_DEBUG").is_ok();
        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,                  // exit_dump_len
            !opts.system_wide,  // partial
            bpf_debug,          // debug (verbose verifier log when SCX_BPF_DEBUG=1)
            false,              // builtin_idle: agent makes all placement decisions
            false,              // numa_local
            5_000_000,          // BPF-layer default slice (agent heartbeat)
            opts.timeout_ms,
            "rl_res",
        )?;

        Ok(Self {
            bpf,
            workers,
            is_worker,
            seq: 0,
            slice_ns: opts.slice_ms.saturating_mul(NSEC_PER_MSEC).max(1),
            preempt_count: 0,
            metrics: RlMetrics::default(),
            ql,
            training: !eval,
            slow_loop_interval: Duration::from_millis(SLOW_LOOP_INTERVAL_MS),
            last_slow_loop: Instant::now(),
            place_interval: Duration::from_millis(PLACE_INTERVAL_MS),
            last_place: Instant::now(),
            no_rebalance: opts.no_rebalance,
            verbose: opts.verbose,
        })
    }

    fn is_worker(&self, cpu: i32) -> bool {
        cpu >= 0 && self.is_worker.get(cpu as usize).copied().unwrap_or(false)
    }

    /// Build a snapshot of the current scheduler state and hand it to the
    /// `scxbus` module so the next Python call sees it as the "current" state.
    fn publish_snapshot(&mut self, _pending: usize) {
        self.bpf.refresh_qlen();
        let mut rq_sizes = vec![0i32; N_CORES];
        for c in 0..N_CORES {
            rq_sizes[c] = self.bpf.qlen(c as i32);
        }
        let running = *self.bpf.nr_running_mut() as i64;
        scx_bridge::publish(scx_bridge::Snapshot {
            // True never-run backlog: tasks that woke but have NOT had their
            // first run yet -- BPF nr_waiting_first_run, +1 at wakeup, -1 at
            // first run. Unlike the total queue depth, it excludes preempted
            // tasks requeued to the tail (they already ran, so they don't add to
            // first-run latency). The agent `pending` buffer needs no separate
            // add: those tasks have not run, so they are already counted.
            unstarted_tasks: self.bpf.nr_waiting_first_run().max(0),
            started_tasks: running,
            finished_tasks: 0,
            sum_res: 0,
            sum_exec: 0,
            sum_preempt: self.preempt_count as i64,
            rq_sizes,
        });
    }

    /// Ask the placement DQN for `n` core assignments (one per placeable task).
    /// `train_step` also advances online training one step using the reward it
    /// derives from the previous vs. current snapshot.
    fn request_actions(&mut self, n: usize) -> Vec<i32> {
        if n == 0 {
            return Vec::new();
        }
        // Time the FULL round-trip on the Rust side (GIL acquire + marshaling +
        // the Python call) so (round_trip - Python wall) exposes the boundary glue.
        let t0 = Instant::now();
        let out = Python::attach(|py| -> PyResult<Vec<i32>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", n)?;
            kwargs.set_item("reward", py.None())?;
            kwargs.set_item("done", false)?;
            let ret = self.ql.bind(py).call_method("train_step", (), Some(&kwargs))?;
            ret.extract::<Vec<i32>>()
        })
        .unwrap_or_else(|e| {
            eprintln!("scx_rl_res: train_step failed: {e}");
            Vec::new()
        });
        self.metrics.action_rt_us = self
            .metrics
            .action_rt_us
            .wrapping_add(t0.elapsed().as_micros() as u64);
        out
    }

    /// The reference's fixed 500 ms slow-loop tick: update the load-adaptive
    /// slice policy and run RL rebalance once per tick. `pending` is the agent's
    /// retry-buffer depth (tasks not yet placed), added to the active-task count.
    fn tick_slow_loop(&mut self, pending: usize) {
        if self.last_slow_loop.elapsed() < self.slow_loop_interval {
            return;
        }
        self.last_slow_loop = Instant::now();
        self.metrics.slow_ticks = self.metrics.slow_ticks.wrapping_add(1);

        let running = *self.bpf.nr_running_mut() as usize;
        let queued = *self.bpf.nr_queued_mut() as usize;
        // Per-core DSQ backlog: tasks already placed on a core but not yet
        // running. The simulator's `active` load counts these (sum of per-core
        // FIFO lengths); we count them too so `active` mirrors that load signal.
        self.bpf.refresh_qlen();
        let dsq_backlog: usize = self
            .workers
            .iter()
            .map(|&c| self.bpf.qlen(c).max(0) as usize)
            .sum();
        let active = running + dsq_backlog + queued + pending;

        // Sample the DSQ backlog this tick so we can compare real load against
        // the simulator's.
        if dsq_backlog as u64 > self.metrics.qlen_peak {
            self.metrics.qlen_peak = dsq_backlog as u64;
        }
        self.metrics.qlen_acc = self.metrics.qlen_acc.wrapping_add(dsq_backlog as u64);

        if active > 0 {
            // Always let the slice DQN pick the preemption slice, even at low
            // load: a single long task running to completion can hold a CPU for
            // tens of seconds and trip the kernel RCU-stall watchdog on
            // heavy-tailed (hetero) traces.
            let t0 = Instant::now();
            let ms = Python::attach(|py| -> PyResult<i64> {
                self.ql.bind(py).call_method0("infer_time_slice")?.extract::<i64>()
            });
            self.metrics.slice_rt_us = self
                .metrics
                .slice_rt_us
                .wrapping_add(t0.elapsed().as_micros() as u64);
            match ms {
                Ok(ms) if ms > 0 => {
                    self.metrics.slice_dqn_updates =
                        self.metrics.slice_dqn_updates.wrapping_add(1);
                    self.metrics.record_slice(ms);
                    self.slice_ns = (ms as u64).saturating_mul(NSEC_PER_MSEC);
                }
                Ok(_) => {}
                Err(e) => eprintln!("scx_rl_res: infer_time_slice failed: {e}"),
            }
        }
        // active == 0: nothing runnable; leave the slice as-is.

        // RL rebalance runs once on every fixed 500 ms slow-loop tick. It no-ops
        // internally when no worker queue is overloaded. --no-rebalance (env
        // NO_REBALANCE=1, mapped by run_workload.sh) disables it entirely
        // (ablation; see run_ablation.sh) -- the DQN still does all initial
        // placement, only the periodic migration pass is skipped.
        if !self.no_rebalance {
            self.rl_rebalance();
        }
    }

    /// ghOSt-style RL rebalance trigger: scan all worker queues, identify every
    /// overloaded source with the reference thresholds, ask the DQN for one target
    /// per migrated task, and submit the resulting migration plan to BPF.
    fn rl_rebalance(&mut self) {
        if self.workers.len() < 2 {
            return;
        }

        self.bpf.refresh_qlen();

        let mut cpu_loads = Vec::with_capacity(self.workers.len());
        let mut min_load = usize::MAX;
        let mut min_cpu = self.workers[0];

        for &cpu in &self.workers {
            let load = self.bpf.qlen(cpu).max(0) as usize;
            cpu_loads.push((cpu, load));
            if load < min_load {
                min_load = load;
                min_cpu = cpu;
            }
        }

        cpu_loads.sort_by(|a, b| b.1.cmp(&a.1));

        let mut plans = Vec::new();
        let mut total = 0usize;
        for &(src_cpu, load) in &cpu_loads {
            let should_steal = (min_load == 0 && load >= 2)
                || (load > 10 && load.saturating_sub(min_load) >= 5);
            if !should_steal {
                break;
            }

            let mut to_move = (load - min_load) / 2;
            if to_move == 0 {
                to_move = 1;
            }
            if total + to_move > GLOBAL_MAX_MIGRATE {
                to_move = GLOBAL_MAX_MIGRATE - total;
            }
            if to_move > 0 {
                plans.push((src_cpu, to_move));
                total += to_move;
            }
            if total >= GLOBAL_MAX_MIGRATE {
                break;
            }
        }

        if total == 0 {
            return;
        }

        self.publish_snapshot(0);
        let actions = self.request_actions(total);
        let mut migrations = Vec::with_capacity(total);
        let mut action_idx = 0usize;

        for (src_cpu, to_move) in plans {
            for _ in 0..to_move {
                let action = actions.get(action_idx).copied().unwrap_or(-1);
                let dst_cpu = if self.is_worker(action) { action } else { min_cpu };
                migrations.push((src_cpu, dst_cpu));
                action_idx += 1;
            }
        }

        self.bpf.request_migrations(&migrations);
        self.metrics.rebalance_moves = self
            .metrics
            .rebalance_moves
            .wrapping_add(migrations.len() as u64);
    }

    /// Pick the final target core for a task given the DQN's chosen core: honour
    /// pinned tasks, and fall back to the least-loaded worker if the action is
    /// not a usable worker for this task.
    fn target_cpu(&mut self, task: &QueuedTask, action: i32) -> i32 {
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }
        if self.is_worker(action) {
            return action;
        }
        // Fallback: least-loaded worker (the action was out of range, or not a
        // worker the task is allowed on).
        let mut best = self.workers[0];
        let mut best_q = self.bpf.qlen(best);
        for &c in &self.workers {
            let q = self.bpf.qlen(c);
            if q < best_q {
                best_q = q;
                best = c;
            }
        }
        best
    }

    fn is_slice_preempt(task: &QueuedTask) -> bool {
        // A re-enqueue (slice-expiry preemption on this CPU-bound trace) has
        // enq_cnt > 1: the BPF backend bumps enq_cnt on every enqueue to user
        // space and never resets it, so a fresh arrival is 1 and any requeue is
        // >= 2. (SCX_ENQ_PREEMPT is not set on slice-expiry enqueues, so it
        // cannot be used here.)
        task.enq_cnt > 1
    }

    fn requeue_cpu(&self, task: &QueuedTask) -> i32 {
        if task.cpu >= 0 {
            task.cpu
        } else {
            self.workers[0]
        }
    }

    fn dispatch_on_cpu(
        &mut self,
        task: &QueuedTask,
        cpu: i32,
        force_tail: bool,
    ) -> std::result::Result<(), libbpf_rs::Error> {
        let mut d = DispatchedTask::new(task);
        d.cpu = cpu;
        d.vtime = self.seq; // per-CPU DSQ keyed by arrival => FIFO
        d.slice_ns = self.slice_ns;
        if force_tail {
            d.flags &= !scx_enums.SCX_ENQ_HEAD;
        }

        self.bpf.dispatch_task(&d)?;
        self.seq = self.seq.wrapping_add(1);
        self.metrics.dispatched_tasks = self.metrics.dispatched_tasks.wrapping_add(1);
        Ok(())
    }

    fn dispatch_preempted_local(
        &mut self,
        task: &QueuedTask,
    ) -> std::result::Result<(), libbpf_rs::Error> {
        let cpu = self.requeue_cpu(task);
        self.dispatch_on_cpu(task, cpu, true)
    }

    /// One scheduling round: drain runnable tasks, ask the DQN where to place
    /// new tasks, while slice-expiry preemptions go straight back to the tail of
    /// their current per-core FIFO run-queue. Returns whether any task was
    /// dispatched.
    fn dispatch_tasks(&mut self, pending: &mut VecDeque<QueuedTask>) -> bool {
        let mut dispatched = false;
        let mut ring_blocked = false;

        // Drain newly-runnable tasks onto the retry buffer (which may already
        // carry ring-full leftovers from last round). Slice-expiry preemptions
        // already have a current queue, so put them directly back at that
        // queue's tail instead of sending them through the RL placement loop.
        while let Ok(Some(t)) = self.bpf.dequeue_task() {
            if Self::is_slice_preempt(&t) {
                self.preempt_count = self.preempt_count.wrapping_add(1);
                if self.dispatch_preempted_local(&t).is_err() {
                    pending.push_front(t);
                    self.metrics.ring_full = self.metrics.ring_full.wrapping_add(1);
                    ring_blocked = true;
                    break;
                }
                dispatched = true;
            } else {
                pending.push_back(t);
            }
        }

        // Retry any preempted tasks that were left pending by an earlier ring
        // full condition. They still bypass the placement gate and DQN actions.
        if !ring_blocked {
            let mut idx = 0;
            while idx < pending.len() {
                let is_preempt = pending
                    .get(idx)
                    .map(Self::is_slice_preempt)
                    .unwrap_or(false);
                if !is_preempt {
                    idx += 1;
                    continue;
                }

                let t = pending.remove(idx).unwrap();
                if self.dispatch_preempted_local(&t).is_err() {
                    pending.insert(idx, t);
                    self.metrics.ring_full = self.metrics.ring_full.wrapping_add(1);
                    ring_blocked = true;
                    break;
                }
                dispatched = true;
            }
        }

        // Placement gate: fire the DQN on the fixed 20 ms fast loop, or earlier
        // once a large enough batch has accumulated.
        let gate_open = !ring_blocked
            && !pending.is_empty()
            && (self.last_place.elapsed() >= self.place_interval
                || pending.len() >= PLACE_TRIGGER_QLEN);

        if gate_open {
            self.last_place = Instant::now();
            // Take a batch from the head; pinned tasks are placed directly and do
            // not consume a DQN action. Preempted tasks should have been flushed
            // above; exclude them here too so ring-full retries preserve the
            // local-tail rule.
            let take = pending.len().min(BATCH_CAP);
            let mut batch: Vec<QueuedTask> = Vec::with_capacity(take);
            for _ in 0..take {
                batch.push(pending.pop_front().unwrap());
            }
            let n_placeable = batch
                .iter()
                .filter(|t| !Self::is_slice_preempt(t) && t.nr_cpus_allowed > 1)
                .count();
            self.metrics.place_batches = self.metrics.place_batches.wrapping_add(1);
            self.metrics.place_tasks = self.metrics.place_tasks.wrapping_add(batch.len() as u64);
            self.metrics.place_dqn_actions = self
                .metrics
                .place_dqn_actions
                .wrapping_add(n_placeable as u64);
            self.metrics.place_direct = self
                .metrics
                .place_direct
                .wrapping_add(batch.len().saturating_sub(n_placeable) as u64);

            // Publish the state the DQN will see, then request one core per
            // placeable task (this also drives one online-training step).
            self.publish_snapshot(pending.len() + batch.len());
            let actions = self.request_actions(n_placeable);

            let mut ai = 0usize;
            let mut stop_at = None;
            for bi in 0..batch.len() {
                let task = &batch[bi];
                let preempted = Self::is_slice_preempt(task);
                let action = if !preempted && task.nr_cpus_allowed > 1 {
                    let a = actions.get(ai).copied().unwrap_or(-1);
                    ai += 1;
                    a
                } else {
                    -1
                };
                let cpu = if preempted {
                    self.requeue_cpu(task)
                } else {
                    self.target_cpu(task, action)
                };

                if self.dispatch_on_cpu(task, cpu, preempted).is_err() {
                    // Ring full: requeue this task and the untouched remainder,
                    // retry next round. DQN actions are dropped; non-preempted
                    // tasks get fresh ones, while preempted tasks still bypass
                    // the DQN on retry.
                    stop_at = Some(bi);
                    self.metrics.ring_full = self.metrics.ring_full.wrapping_add(1);
                    break;
                }
                dispatched = true;
            }
            if let Some(bi) = stop_at {
                // Push back batch[bi..] to the head of pending, preserving order.
                for t in batch.drain(bi..).rev() {
                    pending.push_front(t);
                }
            }
        }

        self.tick_slow_loop(pending.len());
        // Report still-pending tasks (gate-held batch + ring-full leftovers) so the
        // backend re-invokes us to drain them.
        self.bpf.notify_complete(pending.len() as u64);
        dispatched
    }

    /// Final episode tick: let the agent record the terminal transition and, when
    /// training, checkpoint res.pth / res_best.pth.
    fn finish_episode(&mut self) {
        self.publish_snapshot(0);
        let _ = Python::attach(|py| -> PyResult<()> {
            let ql = self.ql.bind(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", 1)?;
            kwargs.set_item("reward", py.None())?;
            kwargs.set_item("done", true)?;
            ql.call_method("train_step", (), Some(&kwargs))?;
            let ret = ql.call_method0("get_and_reset_episode_return")?;
            if let Ok(r) = ret.extract::<f64>() {
                println!("scx_rl_res: episode return = {r:.2}");
            }
            Ok(())
        })
        .map_err(|e| eprintln!("scx_rl_res: finish_episode failed: {e}"));

        // Python-side per-call overhead, excluding the terminal done=True call
        // above. cpu (thread_time) = pure inference compute; wall (perf_counter) =
        // Python-side elapsed; rt (Rust Instant) = full Rust->Python->Rust
        // round-trip. wall and rt are not reported on their own -- they only feed
        // comm = rt - wall = the Rust<->Python communication (pyo3/GIL) overhead.
        let (pa_cnt, pa_wall_us, pa_cpu_us, ps_cnt, ps_wall_us, ps_cpu_us) =
            Python::attach(|py| -> PyResult<(i64, f64, f64, i64, f64, f64)> {
                self.ql.bind(py).call_method0("get_overhead")?.extract()
            })
            .unwrap_or((0, 0.0, 0.0, 0, 0.0, 0.0));
        println!(
            "scx_rl_res: overhead_metrics \
             py_action_count={} py_action_wall_us={:.0} py_action_cpu_us={:.0} \
             py_action_rt_us={} \
             py_slice_count={} py_slice_wall_us={:.0} py_slice_cpu_us={:.0} \
             py_slice_rt_us={}",
            pa_cnt, pa_wall_us, pa_cpu_us, self.metrics.action_rt_us,
            ps_cnt, ps_wall_us, ps_cpu_us, self.metrics.slice_rt_us,
        );
        println!(
            "scx_rl_res: rl_metrics preempt={} place_batches={} place_tasks={} \
             place_dqn_actions={} place_direct={} dispatched_tasks={} ring_full={} \
             slow_ticks={} qlen_peak={} qlen_acc={} slice_dqn_updates={} \
             slice_other_updates={} slice_ms_8={} slice_ms_32={} slice_ms_174={} \
             slice_ms_390={} slice_ms_1633={} rebalance_moves={}",
            self.preempt_count,
            self.metrics.place_batches,
            self.metrics.place_tasks,
            self.metrics.place_dqn_actions,
            self.metrics.place_direct,
            self.metrics.dispatched_tasks,
            self.metrics.ring_full,
            self.metrics.slow_ticks,
            self.metrics.qlen_peak,
            self.metrics.qlen_acc,
            self.metrics.slice_dqn_updates,
            self.metrics.slice_other_updates,
            self.metrics.slice_ms_8,
            self.metrics.slice_ms_32,
            self.metrics.slice_ms_174,
            self.metrics.slice_ms_390,
            self.metrics.slice_ms_1633,
            self.metrics.rebalance_moves,
        );
    }

    fn print_stats(&mut self) {
        let nr_running = *self.bpf.nr_running_mut();
        let nr_queued = *self.bpf.nr_queued_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();
        println!(
            "cpus={} slice_ms={} training={} running={} queued={} user_disp={} migrations={}",
            self.workers.len(),
            self.slice_ns / NSEC_PER_MSEC,
            self.training,
            nr_running,
            nr_queued,
            nr_user,
            nr_migr,
        );
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn run(&mut self) -> Result<UserExitInfo> {
        let mut pending: VecDeque<QueuedTask> = VecDeque::new();
        let mut prev_ts = Self::now();
        while !self.bpf.exited() {
            let dispatched = self.dispatch_tasks(&mut pending);
            if !dispatched {
                std::thread::sleep(IDLE_BACKOFF);
            }
            if self.verbose {
                let curr_ts = Self::now();
                if curr_ts > prev_ts {
                    self.print_stats();
                    prev_ts = curr_ts;
                }
            }
        }
        self.finish_episode();
        self.bpf.shutdown_and_report()
    }
}

fn print_warning() {
    let warning = r#"
**************************************************************************

WARNING: scx_rl_res is a research scheduler that makes all of its scheduling
decisions in user-space via an embedded reinforcement-learning agent
(PyTorch), based on scx_rustland_core. It is not intended for use in
production environments.

**************************************************************************"#;
    println!("{}", warning);
}

/// Pin the agent process (the calling thread, which all the work and the GIL run
/// on; torch threads spawned later inherit this) onto CPU `cpu`. A negative `cpu`
/// leaves the agent unpinned; an out-of-range `cpu` warns and skips.
fn pin_agent(cpu: i32) {
    if cpu < 0 {
        println!("scx_rl_res: agent unpinned (--agent-cpu -1)");
        return;
    }
    let online = online_cpus();
    if cpu >= online {
        eprintln!(
            "scx_rl_res: warning: --agent-cpu {cpu} >= online CPUs ({online}); \
             leaving the agent unpinned."
        );
        return;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu as usize, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            eprintln!(
                "scx_rl_res: warning: failed to pin agent to CPU {cpu}: {}",
                std::io::Error::last_os_error()
            );
        } else {
            println!("scx_rl_res: agent pinned to CPU {cpu}");
        }
    }
}

fn main() -> Result<()> {
    // Register the `scxbus` extension module BEFORE the interpreter starts
    // (auto-initialize brings it up lazily on the first Python::attach). The
    // pymodule fn must be in scope here for the macro to resolve it.
    use scx_bridge::scxbus;
    pyo3::append_to_inittab!(scxbus);

    let opts = Opts::parse();
    print_warning();

    // Pin the agent (this process, incl. the torch threads spawned at Python
    // import) onto its own core, outside the worker set, BEFORE bringing up the
    // interpreter — so the heavy inference/training never steals a scheduled
    // task's CPU.
    pin_agent(opts.agent_cpu);

    let workers = resolve_workers(&opts)?;
    let wlo = workers.first().copied().unwrap_or(0);
    let whi = workers.last().copied().unwrap_or(0);
    println!(
        "scx_rl_res: RL (Dueling-DQN placement + learned slice), {} workers (cores {}-{}), \
         agent core = {}, initial slice = {} ms, place gate = {} ms, \
         slow loop = {} ms, rebalance = rl, mode = {}, partial = {}, watchdog = {} ms",
        workers.len(),
        wlo,
        whi,
        if opts.agent_cpu < 0 { "unpinned".to_string() } else { opts.agent_cpu.to_string() },
        opts.slice_ms,
        PLACE_INTERVAL_MS,
        SLOW_LOOP_INTERVAL_MS,
        if opts.eval {
            "eval".to_string()
        } else if opts.resume {
            "train (resume res.pth)".to_string()
        } else {
            "train (fresh)".to_string()
        },
        !opts.system_wide,
        opts.timeout_ms,
    );

    let mut open_object = MaybeUninit::uninit();
    loop {
        let mut sched = Scheduler::init(&mut open_object, &opts)?;
        if !sched.run()?.should_restart() {
            break;
        }
    }

    Ok(())
}
