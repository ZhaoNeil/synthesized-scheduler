// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # SFS: a two-level (FILTER + CFS) SRTF-approximating scheduler (user-space)
//!
//! ## Overview
//!
//! `scx_sfs` is a port of **SFS** ("Smart OS Scheduling for Serverless
//! Functions", SC'22, ds2-lab/SFS) onto `scx_rustland_core`, the same user-space
//! agent framework as this project's other schedulers. SFS speeds up serverless
//! (FaaS) workloads by approximating **Shortest-Remaining-Time-First (SRTF)**
//! with a **two-level** schedule that orchestrates FIFO and CFS:
//!
//!   * **Level 1 — FILTER** (high priority): every function starts here and runs
//!     in **FIFO** (arrival) order, but is **preempted and demoted** if it
//!     overruns an **adaptive time slice** `S`. Functions that finish within `S`
//!     (the short ones) thus run at high priority with little interference.
//!   * **Level 2 — CFS** (low priority): functions that overran `S` (the long
//!     ones) or blocked on I/O are demoted here and time-share fairly by virtual
//!     runtime. Because Level 1 always outranks Level 2, short functions preempt
//!     long ones — the SRTF approximation — while CFS being work-conserving keeps
//!     the long ones from starving.
//!
//! ## Adaptive time slice
//!
//! Per the paper, the slice adapts to load via the inter-arrival time:
//!
//! ```text
//!   S = mean_IAT * c          (c = number of worker CPUs)
//! ```
//!
//! where `mean_IAT` is the mean inter-arrival time over the last `N`
//! (`--filter-window`, default 100) function arrivals. With no data yet `S`
//! defaults to `--default-slice-ms`; it is clamped to `[S_MIN, S_MAX]`.
//!
//! ## How the two levels map onto the per-CPU run-queues
//!
//! Same per-CPU dispatch as [`scx_cfs`](../scx_cfs): one vtime-ordered run-queue
//! per CPU. The two priority levels are expressed by **vtime banding**:
//!
//!   * Level 1 (FILTER): `vtime = arrival_seq` (a monotonic counter — FIFO order),
//!     which is always `< 2^63`.
//!   * Level 2 (CFS): `vtime = 2^63 + vruntime`, always `>= 2^63`.
//!
//! Since every Level-1 key sorts below every Level-2 key, each CPU drains all its
//! FILTER tasks before any CFS task — so FILTER preempts CFS. (The framework
//! can't do `SCHED_FIFO`-style *instant* preemption of a running task, so CFS
//! tasks are given a small slice and FILTER preempts them at the next slice
//! boundary — see Simplifications.)
//!
//! ## Demotion (Level 1 -> Level 2), sticky
//!
//!   * **Slice exhausted:** a FILTER task is dispatched with slice `S`; if it
//!     comes back having used `>= S` of on-CPU time it is demoted.
//!   * **I/O block:** if a FILTER task's on-CPU time (`exec_runtime`, which the
//!     backend resets on sleep) drops, it slept on I/O and is demoted.
//!
//! Demotion is one-way: once in CFS a task stays there.
//!
//! ## Simplifications vs. the paper
//!
//! Educational model. FILTER preempts CFS only at the CFS task's (small) slice
//! boundary rather than instantly (no `SCHED_FIFO`); the third demotion trigger
//! (overload: queuing delay >= 3*S) is omitted — Level-1 occupancy is already
//! bounded by short-function completion and CFS is work-conserving; CFS-level
//! vruntime is a single monotonic clock with per-CPU `min_vruntime` floors; and
//! the function "class" is not needed (SFS, unlike ALPS, does not classify). The
//! substantive mechanics — two priority levels, the adaptive `S = mean_IAT * c`
//! FILTER slice, FIFO Level 1, demotion on slice-exhaustion / I/O, and
//! weighted-fair CFS Level 2 — are faithful.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[rustfmt::skip]
mod bpf;

use std::collections::HashMap;
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
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::UserExitInfo;

const IDLE_BACKOFF: Duration = Duration::from_micros(100);
const NSEC_PER_USEC: u64 = 1_000;
const NSEC_PER_MSEC: u64 = 1_000_000;

/// Nice-0 load for the CFS (Level-2) vruntime scaling.
const NICE_0_WEIGHT: u64 = 100;

/// vtime band that separates the two levels: Level-1 (FILTER) keys are below it,
/// Level-2 (CFS) keys are at/above it, so every CPU drains FILTER before CFS.
const L2_BAND: u64 = 1u64 << 63;

/// Clamp range for the adaptive FILTER slice S.
const S_MIN_NS: u64 = 1 * NSEC_PER_MSEC;
const S_MAX_NS: u64 = 50 * NSEC_PER_MSEC;

/// A task not dispatched for this long is pruned from the tracking map (bounds
/// memory). Pure in-memory, no /proc — see scx_alps for why /proc liveness is
/// avoided in the agent loop.
const REAP_GRACE: Duration = Duration::from_secs(2);

/// CFS-level periodic load balancer thresholds (as in scx_cfs).
const LB_MIN_QLEN: i32 = 4;
const LB_IMBALANCE_FRAC: f64 = 0.25;

#[derive(Parser, Debug)]
#[command(name = "scx_sfs", about = "SFS: a two-level FILTER + CFS SRTF-approximating scheduler")]
struct Opts {
    /// Worker CPUs that each own a per-CPU run-queue, as a CPU list (e.g.
    /// "0-49"). Set the workload's affinity to the same set. Defaults to every
    /// online CPU.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// Default FILTER time slice S, in milliseconds, used until enough arrivals
    /// have been seen to compute the adaptive `S = mean_IAT * cpus`.
    #[arg(short = 's', long, default_value_t = 6)]
    default_slice_ms: u64,

    /// Number of recent arrivals over which the mean inter-arrival time (and
    /// thus S) is computed.
    #[arg(long, default_value_t = 100)]
    filter_window: usize,

    /// CFS-level (Level 2) time slice, in milliseconds. Kept small so FILTER
    /// tasks preempt demoted tasks promptly at the slice boundary.
    #[arg(short = 'c', long, default_value_t = 3)]
    cfs_slice_ms: u64,

    /// How often (milliseconds) to run the CFS-level periodic load balancer
    /// (busiest => idlest). 0 disables it.
    #[arg(long, default_value_t = 250)]
    lb_interval_ms: u64,

    /// Schedule the whole system. By default scx_sfs runs in *partial* mode: it
    /// only schedules tasks that opt in (policy SCHED_EXT); everything else stays
    /// on the kernel's native scheduler.
    #[arg(long, default_value_t = false)]
    system_wide: bool,

    /// Print scheduling statistics once per second.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// sched_ext runnable-task watchdog timeout, in milliseconds. Also settable
    /// via SCX_TIMEOUT_MS. Raise it on oversubscribed systems.
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

fn resolve_workers(opts: &Opts) -> Result<Vec<i32>> {
    match &opts.workers {
        Some(s) => parse_cpu_list(s),
        None => {
            let n = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            Ok((0..n as i32).collect())
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    Filter,
    Cfs,
}

struct TaskState {
    level: Level,
    /// Last observed on-CPU time since sleep (`exec_runtime`); a drop means the
    /// task slept (I/O) and triggers demotion.
    last_exec: u64,
    /// CFS virtual runtime (used once the task is in the CFS level).
    vruntime: u64,
    last_seen: Instant,
}

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    workers: Vec<i32>,
    is_worker: Vec<bool>,
    /// Per-CPU CFS-level virtual-time floor.
    min_vruntime_pc: Vec<u64>,
    /// CFS-level (Level-2) slice (ns).
    cfs_slice_ns: u64,
    /// FILTER slice used before enough arrivals are seen (ns).
    default_slice_ns: u64,
    filter_window: usize,

    tasks: HashMap<i32, TaskState>,
    /// Drained-but-not-yet-dispatched tasks (carries ring-full leftovers).
    pending: VecDeque<QueuedTask>,
    /// Monotonic FILTER arrival sequence (Level-1 FIFO order; always < L2_BAND).
    fifo_seq: u64,

    /// Recent inter-arrival times and the time of the last arrival, for S.
    iat_window: VecDeque<Duration>,
    last_arrival: Option<Instant>,
    /// Current adaptive FILTER slice S (ns).
    s_ns: u64,

    last_reap: Instant,
    lb_interval: Duration,
    last_lb: Instant,
    verbose: bool,
}

// The agent does no persistent-filesystem I/O (no ext4 inode-lock priority
// inversion). SFS needs no per-task introspection at all (unlike ALPS), so it
// reads nothing from /proc.

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
        let workers = resolve_workers(opts)?;
        let max_cpu = *workers.last().unwrap_or(&0) as usize;
        let mut is_worker = vec![false; max_cpu + 1];
        for &c in &workers {
            is_worker[c as usize] = true;
        }
        let default_slice_ns = opts
            .default_slice_ms
            .saturating_mul(NSEC_PER_MSEC)
            .clamp(S_MIN_NS, S_MAX_NS);

        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,        // exit_dump_len
            !opts.system_wide, // partial
            false,    // debug
            false,    // builtin_idle: agent makes all placement decisions
            false,    // numa_local
            5_000_000, // BPF-layer default slice (agent heartbeat)
            opts.timeout_ms,
            "sfs",
        )?;

        Ok(Self {
            bpf,
            workers,
            is_worker,
            min_vruntime_pc: vec![0; max_cpu + 1],
            cfs_slice_ns: opts.cfs_slice_ms.saturating_mul(NSEC_PER_MSEC).max(NSEC_PER_USEC),
            default_slice_ns,
            filter_window: opts.filter_window.max(1),
            tasks: HashMap::new(),
            pending: VecDeque::new(),
            fifo_seq: 0,
            iat_window: VecDeque::new(),
            last_arrival: None,
            s_ns: default_slice_ns,
            last_reap: Instant::now(),
            lb_interval: Duration::from_millis(opts.lb_interval_ms),
            last_lb: Instant::now(),
            verbose: opts.verbose,
        })
    }

    /// Record a new function's arrival and recompute the adaptive FILTER slice
    /// `S = mean_IAT * cpus` over the last `filter_window` arrivals.
    fn note_arrival(&mut self, now: Instant) {
        if let Some(prev) = self.last_arrival {
            self.iat_window.push_back(now.duration_since(prev));
            while self.iat_window.len() > self.filter_window {
                self.iat_window.pop_front();
            }
        }
        self.last_arrival = Some(now);

        self.s_ns = if self.iat_window.is_empty() {
            self.default_slice_ns
        } else {
            let sum: u128 = self.iat_window.iter().map(|d| d.as_nanos()).sum();
            let mean = (sum / self.iat_window.len() as u128) as u64;
            mean.saturating_mul(self.workers.len() as u64)
        }
        .clamp(S_MIN_NS, S_MAX_NS);
    }

    /// Choose a target CPU: an idle CPU if the backend's idle picker finds one
    /// (it prefers the previous CPU); else, for a `sticky` (returning CFS) task,
    /// its previous CPU; else the least-loaded worker.
    fn place(
        bpf: &mut BpfScheduler,
        is_worker: &[bool],
        workers: &[i32],
        task: &QueuedTask,
        qlen: &[i32],
        sticky: bool,
    ) -> i32 {
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }
        let idle = bpf.select_cpu(task.pid, task.cpu, task.flags);
        if idle >= 0 && is_worker.get(idle as usize).copied().unwrap_or(false) {
            return idle;
        }
        if sticky && is_worker.get(task.cpu as usize).copied().unwrap_or(false) {
            return task.cpu;
        }
        let mut best = workers[0];
        let mut best_q = qlen.get(best as usize).copied().unwrap_or(0);
        for &c in workers {
            let q = qlen.get(c as usize).copied().unwrap_or(0);
            if q < best_q {
                best_q = q;
                best = c;
            }
        }
        best
    }

    /// Prune tasks not dispatched for REAP_GRACE (completed). Pure in-memory.
    fn reap(&mut self, now: Instant) {
        if now.duration_since(self.last_reap) < REAP_GRACE {
            return;
        }
        self.last_reap = now;
        self.tasks
            .retain(|_, st| now.duration_since(st.last_seen) < REAP_GRACE);
    }

    fn maybe_load_balance(&mut self) {
        if self.lb_interval.is_zero() || self.last_lb.elapsed() < self.lb_interval {
            return;
        }
        self.last_lb = Instant::now();
        if self.workers.len() < 2 {
            return;
        }
        self.bpf.refresh_qlen();
        let mut busiest = self.workers[0];
        let mut idlest = self.workers[0];
        let mut max_q = self.bpf.qlen(busiest);
        let mut min_q = max_q;
        for &c in &self.workers {
            let q = self.bpf.qlen(c);
            if q > max_q {
                max_q = q;
                busiest = c;
            }
            if q < min_q {
                min_q = q;
                idlest = c;
            }
        }
        let diff = max_q - min_q;
        if busiest != idlest
            && max_q > LB_MIN_QLEN
            && (diff as f64) > LB_IMBALANCE_FRAC * (max_q as f64)
        {
            let nr = (diff / 2) as u32;
            if nr > 0 {
                self.bpf.request_migration(busiest, idlest, nr);
            }
        }
    }

    /// One scheduling round: drain newly-runnable tasks, route each to its level
    /// (FILTER until it overruns the adaptive slice or blocks on I/O, then CFS),
    /// and dispatch it onto its CPU's vtime-banded run-queue.
    fn dispatch_tasks(&mut self) -> bool {
        let now = Instant::now();

        while let Ok(Some(t)) = self.bpf.dequeue_task() {
            self.pending.push_back(t);
        }

        let mut dispatched = false;
        if !self.pending.is_empty() {
            let mut qlen: Vec<i32> = vec![0; self.min_vruntime_pc.len()];
            self.bpf.refresh_qlen();
            for &c in &self.workers {
                qlen[c as usize] = self.bpf.qlen(c);
            }

            while let Some(task) = self.pending.pop_front() {
                let pid = task.pid;
                let exec = task.exec_runtime;
                let weight = task.weight.max(1);

                if !self.tasks.contains_key(&pid) {
                    self.note_arrival(now);
                }

                // Decide the level (FILTER until demoted; demotion is sticky).
                let (level, just_demoted) = {
                    let st = self.tasks.entry(pid).or_insert(TaskState {
                        level: Level::Filter,
                        last_exec: 0,
                        vruntime: 0,
                        last_seen: now,
                    });
                    let mut jd = false;
                    if st.level == Level::Filter {
                        let io_block = st.last_exec > 0 && exec < st.last_exec;
                        let slice_done = exec >= self.s_ns;
                        if io_block || slice_done {
                            st.level = Level::Cfs;
                            jd = true;
                        }
                    }
                    st.last_exec = exec;
                    st.last_seen = now;
                    (st.level, jd)
                };

                let mut d = DispatchedTask::new(&task);
                let cpu;
                let mut l2_vruntime = 0u64;
                match level {
                    Level::Filter => {
                        cpu = Self::place(
                            &mut self.bpf,
                            &self.is_worker,
                            &self.workers,
                            &task,
                            &qlen,
                            false,
                        );
                        d.cpu = cpu;
                        d.vtime = self.fifo_seq; // Level-1 band (< L2_BAND), FIFO order
                        d.slice_ns = self.s_ns; // run up to the adaptive slice
                    }
                    Level::Cfs => {
                        cpu = Self::place(
                            &mut self.bpf,
                            &self.is_worker,
                            &self.workers,
                            &task,
                            &qlen,
                            true,
                        );
                        let floor = self.min_vruntime_pc.get(cpu as usize).copied().unwrap_or(0);
                        let delta = task.stop_ts.saturating_sub(task.start_ts);
                        let v = {
                            let st = self.tasks.get_mut(&pid).unwrap();
                            let base = if just_demoted {
                                floor
                            } else {
                                st.vruntime.max(floor.saturating_sub(self.cfs_slice_ns))
                            };
                            let v = base.saturating_add(delta.saturating_mul(NICE_0_WEIGHT) / weight);
                            st.vruntime = v;
                            v
                        };
                        l2_vruntime = v;
                        d.cpu = cpu;
                        d.vtime = L2_BAND.saturating_add(v); // Level-2 band (> all L1)
                        d.slice_ns = self.cfs_slice_ns;
                    }
                }

                if self.bpf.dispatch_task(&d).is_err() {
                    self.pending.push_front(task);
                    break;
                }
                if let Some(q) = qlen.get_mut(cpu as usize) {
                    *q += 1;
                }
                match level {
                    Level::Filter => self.fifo_seq = self.fifo_seq.wrapping_add(1),
                    Level::Cfs => {
                        if let Some(f) = self.min_vruntime_pc.get_mut(cpu as usize) {
                            *f = (*f).max(l2_vruntime);
                        }
                    }
                }
                dispatched = true;
            }
        }

        self.reap(now);
        self.maybe_load_balance();
        self.bpf.notify_complete(self.pending.len() as u64);
        dispatched
    }

    fn print_stats(&mut self) {
        let nr_cfs = self.tasks.values().filter(|t| t.level == Level::Cfs).count();
        let nr_filter = self.tasks.len() - nr_cfs;
        let nr_running = *self.bpf.nr_running_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();
        println!(
            "cpus={} S={}us filter={} cfs={} pending={} running={} user_disp={} lb_migr={}",
            self.workers.len(),
            self.s_ns / NSEC_PER_USEC,
            nr_filter,
            nr_cfs,
            self.pending.len(),
            nr_running,
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
        let mut prev_ts = Self::now();
        while !self.bpf.exited() {
            let dispatched = self.dispatch_tasks();
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
        self.bpf.shutdown_and_report()
    }
}

fn print_warning() {
    let warning = r#"
**************************************************************************

WARNING: scx_sfs is a simple educational scheduler that makes all of its
scheduling decisions in user-space, based on scx_rustland_core. It is not
intended for use in production environments.

**************************************************************************"#;
    println!("{}", warning);
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    print_warning();

    let workers = resolve_workers(&opts)?;
    println!(
        "scx_sfs: SFS two-level (FILTER + CFS), {} CPUs, adaptive S = mean_IAT*cpus \
         (default {} ms, window {}), CFS slice = {} ms, load balance every {} ms, \
         partial = {}, watchdog = {} ms",
        workers.len(),
        opts.default_slice_ms,
        opts.filter_window,
        opts.cfs_slice_ms,
        opts.lb_interval_ms,
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
