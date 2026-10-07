// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # Hybrid: a two-tier FIFO + CFS scheduler (user-space)
//!
//! ## Overview
//!
//! `scx_hybrid` is a hybrid scheduler that makes all of its decisions in
//! user-space, in 100% Rust, on top of `scx_rustland_core` (the kernel's
//! `sched_ext` feature). It partitions the machine into two tiers and routes
//! each task to the tier that fits it:
//!
//!   * **FIFO tier** (default CPUs 1-24, with **CPU 0 as the global agent CPU**)
//!     — every task starts here. The agent (running on the global CPU) keeps the
//!     incoming tasks in arrival order and dispatches each to an **idle FIFO
//!     core**, where it runs FIFO. A task is given a finite slice equal to the
//!     preemption threshold, so a task that finishes quickly effectively runs to
//!     completion, cheaply, on a dedicated core.
//!
//!   * **CFS tier** (default CPUs 25-49) — the *long-running* tasks. The agent
//!     records each task's on-CPU time; once a task's on-CPU time crosses the
//!     **preemption time slice** (default 1633 ms) it is preempted, **migrated to
//!     the CFS tier, and never returns**. There it is scheduled by a weighted-fair
//!     virtual-runtime policy (per-CPU run-queues ordered by vruntime, preemptive
//!     time-slicing), so the long tasks time-share the CFS cores fairly instead
//!     of monopolizing FIFO cores and blocking short tasks behind them.
//!
//! The point of the split: short tasks get low-overhead run-to-completion on the
//! FIFO partition with no head-of-line blocking from a long task, while long,
//! CPU-bound tasks are quarantined onto the CFS partition where fairness keeps
//! any one of them from starving the others.
//!
//! ## On-CPU time and the migration rule
//!
//! "On-CPU time" is the time a task has actually spent executing on a CPU since
//! it last slept — the backend's `exec_runtime`. Because a FIFO task is
//! dispatched with a slice equal to the threshold, a task that keeps running is
//! preempted right at the threshold and re-enqueued; the agent then sees
//! `exec_runtime >= threshold` and routes it to the CFS tier. A task that sleeps
//! (e.g. on I/O) resets its on-CPU time and stays on the low-latency FIFO tier.
//! Migration is one-way and sticky: once a task is in the CFS tier it is always
//! scheduled there.
//!
//! ## How the tiers map onto DSQs
//!
//! There is **no shared DSQ**: if there were, an idle CFS core would steal FIFO
//! tasks from it and break the partition. Instead the agent places every task on
//! a specific core's per-CPU DSQ, choosing within the task's tier — an idle core
//! if one is free (the backend's idle picker); otherwise a task already resident
//! in the tier stays on its previous core (cache affinity, like scx_cfs), while a
//! task entering the tier (new, or just promoted FIFO->CFS) goes to the tier's
//! least-loaded core. FIFO cores order their DSQ by arrival (a monotonic
//! sequence); CFS cores order theirs by virtual runtime. Sustained CFS-tier
//! imbalance is corrected by the periodic load balancer, not by re-placing a task
//! every slice.
//!
//! ## Simplifications
//!
//! Educational model: the CFS tier tracks vruntime as a single monotonic clock
//! with per-CPU `min_vruntime` floors (not per-run-queue-relative, so migrations
//! within the CFS tier don't carry exact renormalization); "on-CPU time" is
//! on-CPU time since the last sleep (`exec_runtime`) rather than a lifetime sum;
//! and there are no cgroups/NUMA. The substantive mechanics — the FIFO/CFS split,
//! idle-first placement within a tier, the on-CPU-time preemption threshold and
//! one-way migration, and weighted-fair vruntime on the CFS tier — are faithful.

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

/// When the dispatch ring is momentarily full, back off this long before
/// retrying instead of busy-spinning the agent CPU.
const IDLE_BACKOFF: Duration = Duration::from_micros(100);

const NSEC_PER_USEC: u64 = 1_000;
const NSEC_PER_MSEC: u64 = 1_000_000;

/// Load weight of a nice-0 task (the framework's default task weight); plays the
/// role of `NICE_0_LOAD` in the CFS-tier vruntime scaling.
const NICE_0_WEIGHT: u64 = 100;

/// CFS-tier load balancer: only rebalance a run-queue at least this deep, and
/// only when the imbalance exceeds this fraction of the busiest queue.
const LB_MIN_QLEN: i32 = 4;
const LB_IMBALANCE_FRAC: f64 = 0.25;

#[derive(Parser, Debug)]
#[command(name = "scx_hybrid", about = "Hybrid: a two-tier FIFO + CFS scheduler")]
struct Opts {
    /// CPUs forming the **FIFO tier** (new/short tasks), as a CPU list (e.g.
    /// "1-24"). CPU 0 is left out as the global agent CPU. Must be disjoint from
    /// --cfs-cpus.
    #[arg(long, default_value = "1-24")]
    fifo_cpus: String,

    /// CPUs forming the **CFS tier** (migrated long tasks), as a CPU list (e.g.
    /// "25-49"). Must be disjoint from --fifo-cpus.
    #[arg(long, default_value = "25-49")]
    cfs_cpus: String,

    /// Preemption time slice / migration threshold, in milliseconds. A task is
    /// dispatched on the FIFO tier with a slice of this length; once its on-CPU
    /// time reaches this, it is preempted and migrated to the CFS tier.
    #[arg(short = 'p', long, default_value_t = 1633)]
    preempt_slice_ms: u64,

    /// Time slice, in milliseconds, used on the CFS tier (how long a migrated
    /// long task runs before being re-evaluated against the others on its CFS
    /// core). Smaller ⇒ finer fairness, more context switches.
    #[arg(short = 'c', long, default_value_t = 6)]
    cfs_slice_ms: u64,

    /// How often (milliseconds) to run the CFS tier's periodic load balancer
    /// (busiest ⇒ idlest CFS core). 0 disables it (placement-time balancing only).
    #[arg(long, default_value_t = 250)]
    lb_interval_ms: u64,

    /// Schedule the whole system. By default scx_hybrid runs in *partial* mode:
    /// it only schedules tasks that opt in (policy SCHED_EXT); everything else
    /// stays on the kernel's native scheduler.
    #[arg(long, default_value_t = false)]
    system_wide: bool,

    /// Print scheduling statistics once per second.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// sched_ext runnable-task watchdog timeout, in milliseconds. If any task
    /// stays runnable (waiting for a CPU) longer than this, the kernel ejects the
    /// scheduler and mass-migrates every task back to CFS — which, with thousands
    /// of tasks, can freeze the machine. The FIFO tier runs tasks for up to a
    /// full preemption slice, so on an oversubscribed system this can exceed the
    /// ~30 s default; raise it. Above the kernel's SCX_WATCHDOG_MAX_TIMEOUT,
    /// attach fails.
    #[arg(long, env = "SCX_TIMEOUT_MS", default_value_t = 30_000)]
    timeout_ms: u32,
}

/// Parse a CPU list ("0-49", "0,2,4", "0-3,8,10-12") into a sorted,
/// de-duplicated list of CPU ids.
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
        anyhow::bail!("empty CPU list");
    }
    Ok(cpus.into_iter().collect())
}

/// Build a CPU-id-indexed membership bitmap covering both tiers.
fn membership(cpus: &[i32], size: usize) -> Vec<bool> {
    let mut m = vec![false; size];
    for &c in cpus {
        if (c as usize) < size {
            m[c as usize] = true;
        }
    }
    m
}

/// Which tier a task is currently scheduled in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    Fifo,
    Cfs,
}

/// Per-task state the agent tracks across enqueues.
struct TaskState {
    tier: Tier,
    /// CFS virtual runtime (meaningful once the task is in the CFS tier).
    vruntime: u64,
}

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Per-task tier + vruntime, keyed by pid.
    tasks: HashMap<i32, TaskState>,
    /// Transient buffer: tasks drained from the kernel, dispatched in order;
    /// only retains a task if its dispatch failed (ring full) for next-round retry.
    pending: VecDeque<QueuedTask>,

    /// FIFO tier.
    fifo_cpus: Vec<i32>,
    is_fifo: Vec<bool>,
    /// Monotonic arrival sequence stamped as the FIFO DSQ key (strict FIFO order).
    fifo_seq: u64,
    /// FIFO slice == migration threshold (ns).
    preempt_slice_ns: u64,

    /// CFS tier.
    cfs_cpus: Vec<i32>,
    is_cfs: Vec<bool>,
    /// CFS-tier slice (ns).
    cfs_slice_ns: u64,
    /// Per-CPU virtual-time floor (CFS `min_vruntime`), indexed by CPU id.
    min_vruntime_pc: Vec<u64>,
    /// CFS-tier periodic load balancer cadence.
    lb_interval: Duration,
    last_lb: Instant,

    verbose: bool,
}

// The agent deliberately performs no file I/O: it could block on an ext4 inode
// lock held by a task it must schedule -- a priority inversion that deadlocks
// the scheduler. Per-task lifecycle timestamps (TaskNew/FirstRun/TaskDead) are
// recorded by the task processes themselves into shared memory (see
// project/function_trace/metrics_shm.h).

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
        let fifo_cpus = parse_cpu_list(&opts.fifo_cpus).context("parsing --fifo-cpus")?;
        let cfs_cpus = parse_cpu_list(&opts.cfs_cpus).context("parsing --cfs-cpus")?;

        // The two tiers must be disjoint (a CPU can't be both FIFO and CFS).
        for &c in &fifo_cpus {
            if cfs_cpus.contains(&c) {
                anyhow::bail!("CPU {c} is in both --fifo-cpus and --cfs-cpus");
            }
        }

        let size = (*fifo_cpus.last().unwrap_or(&0))
            .max(*cfs_cpus.last().unwrap_or(&0)) as usize
            + 1;
        let is_fifo = membership(&fifo_cpus, size);
        let is_cfs = membership(&cfs_cpus, size);

        let preempt_slice_ns = opts.preempt_slice_ms.saturating_mul(NSEC_PER_MSEC).max(1);
        let cfs_slice_ns = opts.cfs_slice_ms.saturating_mul(NSEC_PER_MSEC).max(NSEC_PER_USEC);

        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,        // exit_dump_len (0 = default)
            // partial: true (default) => only SCHED_EXT opt-in tasks are scheduled
            // by scx_hybrid; the rest of the system stays on the native scheduler.
            !opts.system_wide,
            false,    // debug
            false,    // builtin_idle: false => every task is funneled through the
                      // agent so it makes all tier/placement decisions.
            false,    // numa_local
            5_000_000, // BPF-layer default slice (5 ms) — the agent's own
                      // heartbeat. Worker tasks get a tier-specific slice.
            opts.timeout_ms,
            "hybrid",
        )?;

        Ok(Self {
            bpf,
            tasks: HashMap::new(),
            pending: VecDeque::new(),
            fifo_cpus,
            is_fifo,
            fifo_seq: 0,
            preempt_slice_ns,
            cfs_cpus,
            is_cfs,
            cfs_slice_ns,
            min_vruntime_pc: vec![0; size],
            lb_interval: Duration::from_millis(opts.lb_interval_ms),
            last_lb: Instant::now(),
            verbose: opts.verbose,
        })
    }

    /// Drain every newly-runnable task into the tail of the pending buffer.
    fn drain(&mut self) {
        while let Ok(Some(task)) = self.bpf.dequeue_task() {
            self.pending.push_back(task);
        }
    }

    /// Choose a target CPU for `task` within one tier: an idle CPU if the
    /// backend's idle picker finds one in this tier (it prefers the task's
    /// previous CPU). Failing that, a `returning` task (already resident in this
    /// tier) stays on its previous CPU for cache affinity -- mirroring scx_cfs's
    /// `place()` -- while a task entering the tier (`returning == false`: new, or
    /// just promoted FIFO->CFS with its prev CPU in the other tier) spreads to the
    /// least-loaded CPU in the tier. This keeps a preempted task on the same core
    /// across slices instead of migrating it every reschedule; sustained imbalance
    /// is handled by the periodic load balancer. Static so it borrows only the
    /// disjoint fields it needs.
    fn place_in(
        bpf: &mut BpfScheduler,
        is_member: &[bool],
        members: &[i32],
        task: &QueuedTask,
        qlen: &[i32],
        returning: bool,
    ) -> i32 {
        // A single-CPU task can only run on its one CPU.
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }
        let idle = bpf.select_cpu(task.pid, task.cpu, task.flags);
        if idle >= 0 && is_member.get(idle as usize).copied().unwrap_or(false) {
            return idle;
        }
        // No idle CPU in the tier: a returning task sticks to its previous CPU (if
        // that CPU belongs to this tier); only a task entering the tier spreads to
        // the least-loaded core.
        if returning && is_member.get(task.cpu as usize).copied().unwrap_or(false) {
            return task.cpu;
        }
        let mut best = members[0];
        let mut best_q = qlen.get(best as usize).copied().unwrap_or(0);
        for &c in members {
            let q = qlen.get(c as usize).copied().unwrap_or(0);
            if q < best_q {
                best_q = q;
                best = c;
            }
        }
        best
    }

    /// Periodically migrate half of the excess from the busiest CFS-tier run-queue
    /// to the idlest, subject to a depth floor and imbalance threshold. Tasks keep
    /// their vtime (vruntime), so destination ordering stays correct.
    fn maybe_load_balance(&mut self) {
        if self.lb_interval.is_zero() || self.last_lb.elapsed() < self.lb_interval {
            return;
        }
        self.last_lb = Instant::now();
        if self.cfs_cpus.len() < 2 {
            return;
        }

        self.bpf.refresh_qlen();
        let mut busiest = self.cfs_cpus[0];
        let mut idlest = self.cfs_cpus[0];
        let mut max_q = self.bpf.qlen(busiest);
        let mut min_q = max_q;
        for &c in &self.cfs_cpus {
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

    /// One scheduling round: drain newly-runnable tasks and route each to its
    /// tier — FIFO (arrival-ordered, threshold-sliced) until its on-CPU time
    /// crosses the preemption threshold, then CFS (vruntime-ordered) for good.
    /// Returns true if at least one task was dispatched.
    fn dispatch_tasks(&mut self) -> bool {
        self.drain();
        if self.pending.is_empty() {
            self.bpf.notify_complete(0);
            return false;
        }

        // Per-CPU queue-depth snapshot for least-loaded placement; updated locally
        // as we place so repeated decisions within the round stay balanced.
        let mut qlen: Vec<i32> = vec![0; self.min_vruntime_pc.len()];
        self.bpf.refresh_qlen();
        for c in self.fifo_cpus.iter().chain(self.cfs_cpus.iter()) {
            qlen[*c as usize] = self.bpf.qlen(*c);
        }

        let mut dispatched = false;
        while let Some(task) = self.pending.pop_front() {
            let pid = task.pid;
            let weight = task.weight.max(1);
            let delta_exec = task.stop_ts.saturating_sub(task.start_ts);

            // Decide the tier. A task is in the CFS tier if it was already
            // migrated, or if its on-CPU time (exec_runtime, since last sleep) has
            // reached the preemption threshold. Migration is sticky.
            let (tier, just_migrated) = {
                let st = self
                    .tasks
                    .entry(pid)
                    .or_insert(TaskState { tier: Tier::Fifo, vruntime: 0 });
                let mut jm = false;
                if st.tier == Tier::Fifo && task.exec_runtime >= self.preempt_slice_ns {
                    st.tier = Tier::Cfs;
                    jm = true;
                }
                (st.tier, jm)
            };

            let mut d = DispatchedTask::new(&task);
            let cpu = match tier {
                Tier::Fifo => {
                    let cpu = Self::place_in(
                        &mut self.bpf,
                        &self.is_fifo,
                        &self.fifo_cpus,
                        &task,
                        &qlen,
                        false, // FIFO tier: latency-first, no prev-CPU stickiness
                    );
                    d.cpu = cpu;
                    d.vtime = self.fifo_seq; // strict arrival order on the FIFO core
                    d.slice_ns = self.preempt_slice_ns; // run up to the threshold
                    cpu
                }
                Tier::Cfs => {
                    let cpu = Self::place_in(
                        &mut self.bpf,
                        &self.is_cfs,
                        &self.cfs_cpus,
                        &task,
                        &qlen,
                        // returning CFS task sticks to its core; a just-promoted
                        // task (prev CPU still in the FIFO tier) spreads instead.
                        !just_migrated,
                    );
                    let floor = self.min_vruntime_pc.get(cpu as usize).copied().unwrap_or(0);
                    // Advance (or, on first migration, initialize) the vruntime.
                    let vruntime = {
                        let st = self.tasks.get_mut(&pid).unwrap();
                        let base = if just_migrated {
                            floor // start fair on the CFS run-queue it lands on
                        } else {
                            st.vruntime.max(floor.saturating_sub(self.cfs_slice_ns))
                        };
                        let v = base
                            .saturating_add(delta_exec.saturating_mul(NICE_0_WEIGHT) / weight);
                        st.vruntime = v;
                        v
                    };
                    d.cpu = cpu;
                    d.vtime = vruntime; // vruntime order on the CFS core
                    d.slice_ns = self.cfs_slice_ns;
                    cpu
                }
            };

            if let Err(e) = self.bpf.dispatch_task(&d) {
                // Ring full: put it back and retry next round (state already
                // updated; exec_runtime-based tiering is idempotent on retry).
                self.pending.push_front(task);
                if self.verbose {
                    eprintln!("dispatch failed: {e}");
                }
                break;
            }

            if let Some(q) = qlen.get_mut(cpu as usize) {
                *q += 1;
            }
            if tier == Tier::Fifo {
                self.fifo_seq = self.fifo_seq.wrapping_add(1);
            } else if let Some(f) = self.min_vruntime_pc.get_mut(cpu as usize) {
                *f = (*f).max(d.vtime);
            }
            dispatched = true;
        }

        self.maybe_load_balance();
        self.bpf.notify_complete(self.pending.len() as u64);
        dispatched
    }

    fn print_stats(&mut self) {
        let nr_cfs = self.tasks.values().filter(|t| t.tier == Tier::Cfs).count();
        let nr_running = *self.bpf.nr_running_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_bounce = *self.bpf.nr_bounce_dispatches_mut();
        let nr_failed = *self.bpf.nr_failed_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();

        println!(
            "fifo_cpus={} cfs_cpus={} tasks_seen={} migrated_to_cfs={} pending={} running={} user_disp={} bounce={} fail={} cfs_lb_migr={}",
            self.fifo_cpus.len(),
            self.cfs_cpus.len(),
            self.tasks.len(),
            nr_cfs,
            self.pending.len(),
            nr_running,
            nr_user,
            nr_bounce,
            nr_failed,
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

WARNING: scx_hybrid is a simple educational scheduler that makes all of its
scheduling decisions in user-space, based on scx_rustland_core. It is not
intended for use in production environments.

**************************************************************************"#;
    println!("{}", warning);
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    print_warning();

    let fifo = parse_cpu_list(&opts.fifo_cpus)?;
    let cfs = parse_cpu_list(&opts.cfs_cpus)?;
    println!(
        "scx_hybrid: FIFO tier = {} CPUs ({}), CFS tier = {} CPUs ({}), \
         preemption slice / migration threshold = {} ms, CFS slice = {} ms, \
         CFS load balance every {} ms, partial = {}, watchdog = {} ms",
        fifo.len(),
        opts.fifo_cpus,
        cfs.len(),
        opts.cfs_cpus,
        opts.preempt_slice_ms,
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
