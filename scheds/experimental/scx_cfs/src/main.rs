// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # CFS: a per-CPU Completely-Fair-Scheduler (user-space)
//!
//! ## Overview
//!
//! `scx_cfs` is a CFS-style scheduler for the Linux kernel that makes all of its
//! decisions in user-space, in 100% Rust, on top of `scx_rustland_core` (which
//! leverages the kernel's `sched_ext` feature). It models how the real Linux CFS
//! schedules on an SMP system.
//!
//! The heart of CFS is **virtual runtime**: every task accumulates vruntime
//! equal to the time it has spent on a CPU scaled *inversely by its weight*, and
//! the scheduler always runs the task with the smallest vruntime. Equal-weight
//! tasks therefore converge to equal CPU time; a higher-weight task accrues
//! vruntime more slowly and so runs more often. We rely on the backend's
//! vtime-ordered dispatch queues (`scx_bpf_dsq_insert_vtime`): stamping a task's
//! vruntime as its vtime makes the kernel keep the queue in vruntime order, so
//! "pick the smallest vruntime" is automatic.
//!
//! ## Per-CPU run-queues
//!
//! Like the real Linux CFS, scx_cfs keeps **one vruntime-ordered run-queue per
//! CPU** (its per-CPU DSQ), each with its own `min_vruntime`:
//!
//!   * **Placement** (`select_task_rq_fair`): a newly-runnable task is placed
//!     onto an **idle CPU** if one exists, preferring its previous CPU for cache
//!     affinity (via the backend's idle picker). If no CPU is idle, a *new* task
//!     spreads to the **least-loaded** run-queue, while a *running* task **stays
//!     on its CPU** (no needless migration).
//!   * **Load balancing** (`run_rebalance_domains`): a periodic balancer pulls
//!     tasks from the busiest run-queue to the idlest when they drift out of
//!     balance (depth and imbalance thresholds guard against churn).
//!   * **Time slice** (`sched_slice`): a task's weighted share of a scheduling
//!     period that stretches with the run-queue depth. Slices are finite and
//!     preemptive, so a long task is preempted and short tasks interleave.
//!
//! ## Virtual runtime accounting
//!
//! ```text
//!   vruntime += delta_exec * NICE_0_WEIGHT / weight        (NICE_0_WEIGHT = 100)
//! ```
//!
//! `delta_exec` is the time the task just spent on the CPU (`stop_ts - start_ts`)
//! and `weight` is its load weight (`[1..10000]`, default `100` == nice 0). We do
//! not keep a vruntime table in the agent: the backend stores the vtime we stamp
//! in `p->scx.dsq_vtime` and reports it back as `QueuedTask.vtime` on the task's
//! next enqueue, so each task's vruntime round-trips through the kernel (a
//! reported vtime of 0 marks a freshly-created task). New and just-woken tasks
//! are realigned to their CPU's `min_vruntime` floor so neither a new task nor a
//! long sleeper can monopolize a CPU.
//!
//! ## Simplifications vs. the kernel CFS
//!
//! This is an educational model. vruntime is tracked as a single monotonic
//! virtual clock rather than per-run-queue-relative, so migrations do not carry
//! the exact `vruntime -= src_min; vruntime += dst_min` renormalization (the
//! per-CPU `min_vruntime` floors approximate it on placement); ordering is pure
//! vruntime (no EEVDF eligibility/deadline); the load balancer is a single coarse
//! busiest->idlest pull rather than hierarchical sched-domains; and there are no
//! cgroups, autogroups, NUMA domains, or wakeup-preemption granularity. The
//! substantive mechanics — weighted vruntime, pick-smallest, per-CPU run-queues,
//! idle-first placement with affinity, `sched_slice`, `min_vruntime` realignment,
//! and periodic load balancing — are faithful.

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
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::UserExitInfo;

/// When tasks are waiting but the dispatch ring is momentarily full, back off
/// this long before the next round to avoid busy-spinning the agent CPU.
const IDLE_BACKOFF: Duration = Duration::from_micros(100);

const NSEC_PER_USEC: u64 = 1_000;

/// Load weight of a nice-0 task. In `scx_rustland_core` task weights are in the
/// range `[1..10000]` with `100` as the default, so 100 plays the role of the
/// kernel's `NICE_0_LOAD` here: vruntime advances by `delta_exec` for a task of
/// this weight, faster for lighter tasks and slower for heavier ones.
const NICE_0_WEIGHT: u64 = 100;

/// Only consider rebalancing a CPU whose run-queue is at least this deep (don't
/// churn shallow queues).
const LB_MIN_QLEN: i32 = 4;
/// Only rebalance when (max_qlen - min_qlen) exceeds this fraction of max_qlen
/// (anti-thrashing guard, like the imbalance threshold in load_balance).
const LB_IMBALANCE_FRAC: f64 = 0.25;

#[derive(Parser, Debug)]
#[command(name = "scx_cfs", about = "CFS: a per-CPU Completely-Fair-Scheduler")]
struct Opts {
    /// Worker CPUs that each own a per-CPU run-queue, as a CPU list (e.g.
    /// "0-49", "0-3,8,12-15"). Tasks are placed across exactly this set; set the
    /// workload's affinity to the same set. Defaults to every online CPU.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// Target scheduling latency (CFS `sched_latency`), in microseconds: the
    /// period over which every runnable task on a CPU should get a turn. The
    /// per-task slice is a weighted share of this period. Smaller values
    /// interleave tasks more finely (more context switches).
    #[arg(short = 's', long, default_value_t = 6_000)]
    sched_latency_us: u64,

    /// Minimum time slice, in microseconds (CFS's "minimum granularity"). A
    /// task's slice is never reduced below this, bounding context-switch overhead.
    #[arg(short = 'S', long, default_value_t = 750)]
    min_slice_us: u64,

    /// How often (milliseconds) to run the periodic load balancer that migrates
    /// tasks from the busiest CPU's run-queue to the idlest. 0 disables it
    /// (placement-time balancing only).
    #[arg(long, default_value_t = 250)]
    lb_interval_ms: u64,

    /// Schedule the whole system. By default scx_cfs runs in *partial* mode: it
    /// only schedules tasks that explicitly opt in (policy SCHED_EXT), and
    /// everything else stays on the kernel's native scheduler (CFS/EEVDF). Set
    /// this to take over every task instead.
    #[arg(long, default_value_t = false)]
    system_wide: bool,

    /// Print scheduling statistics once per second.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// sched_ext runnable-task watchdog timeout, in milliseconds. If any task
    /// stays runnable (waiting for a CPU) longer than this, the kernel ejects
    /// the scheduler and mass-migrates every task back to CFS — which, with
    /// thousands of tasks, can freeze the whole machine. An oversubscribed
    /// system can keep a task waiting well past the ~30 s default, so raise this.
    /// Values above the kernel's compiled SCX_WATCHDOG_MAX_TIMEOUT make attach
    /// fail, so this only takes effect on a kernel built to allow it.
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
        anyhow::bail!("empty worker CPU list");
    }
    Ok(cpus.into_iter().collect())
}

/// Resolve the worker CPU set (explicit list, or every online CPU by default).
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

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Transient user-space buffer holding drained tasks not yet pushed to a
    /// run-queue (only retains a task if its dispatch failed because the ring was
    /// full). The real backlog lives in the kernel per-CPU DSQs.
    pending: VecDeque<QueuedTask>,
    /// Target scheduling latency / scheduling period (ns).
    sched_latency_ns: u64,
    /// Floor on the per-task time slice in ns (minimum granularity).
    min_slice_ns: u64,
    /// Worker CPUs, each owning one per-CPU vruntime run-queue.
    workers: Vec<i32>,
    /// Per-CPU virtual-time floor, indexed by CPU id (CFS's per-`cfs_rq`
    /// `min_vruntime`). Sized to cover the largest worker CPU id.
    min_vruntime_pc: Vec<u64>,
    /// `is_worker[cpu]` — fast membership test for the worker set.
    is_worker: Vec<bool>,
    /// Periodic load-balancer cadence and last-run timestamp.
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
        let sched_latency_ns = opts.sched_latency_us.saturating_mul(NSEC_PER_USEC);
        let min_slice_ns = opts.min_slice_us.saturating_mul(NSEC_PER_USEC);

        let workers = resolve_workers(opts)?;
        let max_cpu = *workers.last().unwrap_or(&0) as usize;
        let mut is_worker = vec![false; max_cpu + 1];
        for &c in &workers {
            is_worker[c as usize] = true;
        }

        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,        // exit_dump_len (0 = default)
            // partial: true (default) => only tasks that opt into SCHED_EXT are
            // scheduled by scx_cfs; the rest of the system stays on the native
            // kernel scheduler. This keeps background tasks out of the fair
            // run-queues so they can't perturb the measured workload (or trip
            // the sched_ext watchdog).
            !opts.system_wide,
            false,    // debug
            false,    // builtin_idle: false => every task is funneled through the
                      // user-space scheduler (we make all placement decisions);
                      // true would let the kernel fast-path wakeups to prev_cpu.
            false,    // numa_local
            5_000_000, // BPF-layer default slice (5 ms) — the agent's own
                      // scheduling heartbeat. Worker tasks get a CFS slice at
                      // dispatch (see sched_slice).
            opts.timeout_ms,
            "cfs",    // scx ops name
        )?;

        Ok(Self {
            bpf,
            pending: VecDeque::new(),
            sched_latency_ns,
            min_slice_ns,
            workers,
            min_vruntime_pc: vec![0; max_cpu + 1],
            is_worker,
            lb_interval: Duration::from_millis(opts.lb_interval_ms),
            last_lb: Instant::now(),
            verbose: opts.verbose,
        })
    }

    /// Drain every task the kernel has made runnable into the tail of the
    /// pending buffer.
    fn drain(&mut self) {
        while let Ok(Some(task)) = self.bpf.dequeue_task() {
            self.pending.push_back(task);
        }
    }

    /// Charge the weighted time a task just ran (`delta_exec * NICE_0 / weight`)
    /// on top of a base vruntime, realigning new/woken tasks to its CPU's
    /// `floor`.
    fn advance_vruntime(&self, task: &QueuedTask, floor: u64) -> u64 {
        // `task.vtime` round-trips the vruntime we stamped last time (kept by the
        // backend in p->scx.dsq_vtime). 0 means brand-new (reset in ops.enable).
        let base = if task.vtime == 0 {
            // New task: start exactly at the floor — neither leading nor buried.
            floor
        } else {
            // Returning/woken task: give back at most one scheduling period of
            // sleep credit (so a long sleeper can't preempt everything), but
            // don't penalize a merely-stale vruntime (CFS place_entity).
            task.vtime.max(floor.saturating_sub(self.sched_latency_ns))
        };
        // delta_exec = duration of the most recent on-CPU stint (0 if never run).
        let delta_exec = task.stop_ts.saturating_sub(task.start_ts);
        let weight = task.weight.max(1);
        base.saturating_add(delta_exec.saturating_mul(NICE_0_WEIGHT) / weight)
    }

    /// CFS `sched_slice`: a task's weighted share of a scheduling period that
    /// stretches with the run-queue depth.
    ///   period = max(sched_latency, nr_running * min_slice)
    ///   slice  = period * weight / (nr_running * NICE_0_WEIGHT)
    /// For equal weights this is `period / nr_running`, clamped to
    /// `[min_slice, sched_latency]`.
    fn sched_slice(&self, task: &QueuedTask, nr_running: u64) -> u64 {
        let nr = nr_running.max(1);
        let period = self.sched_latency_ns.max(nr.saturating_mul(self.min_slice_ns));
        let weight = task.weight.max(1);
        let slice = period
            .saturating_mul(weight)
            .saturating_div(nr.saturating_mul(NICE_0_WEIGHT));
        slice.clamp(self.min_slice_ns, self.sched_latency_ns)
    }

    /// Pick the least-loaded worker CPU from a per-CPU queue-depth snapshot
    /// (CFS `find_idlest_cpu` fallback when no CPU is fully idle).
    fn least_loaded(&self, qlen: &[i32]) -> i32 {
        let mut best = self.workers[0];
        let mut best_q = qlen.get(best as usize).copied().unwrap_or(0);
        for &c in &self.workers {
            let q = qlen.get(c as usize).copied().unwrap_or(0);
            if q < best_q {
                best_q = q;
                best = c;
            }
        }
        best
    }

    /// Choose the target CPU for a task, `select_task_rq_fair`-style: prefer an
    /// idle CPU (the backend's idle picker prefers the task's previous CPU); if
    /// none is idle, a *new* task spreads to the least-loaded run-queue while a
    /// *returning* task stays on its previous CPU (cache affinity — periodic load
    /// balancing corrects sustained imbalance instead).
    fn place(&mut self, task: &QueuedTask, qlen: &[i32]) -> i32 {
        // A task pinned to a single CPU can only run there.
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }

        // Ask the backend's idle picker (prefers prev_cpu, then any idle CPU in
        // the task's allowed set). It claims the CPU it returns, so repeated
        // calls within a round naturally spread across distinct idle CPUs.
        let idle = self.bpf.select_cpu(task.pid, task.cpu, task.flags);
        if idle >= 0 && self.is_worker.get(idle as usize).copied().unwrap_or(false) {
            return idle;
        }

        // No idle CPU: keep a running task on its previous CPU (stickiness), but
        // spread a freshly-created task to the least-loaded run-queue.
        let prev_is_worker = self.is_worker.get(task.cpu as usize).copied().unwrap_or(false);
        if task.vtime != 0 && prev_is_worker {
            task.cpu
        } else {
            self.least_loaded(qlen)
        }
    }

    /// Periodically migrate half of the excess from the busiest run-queue to the
    /// idlest, subject to a depth floor and an imbalance threshold — a coarse
    /// stand-in for CFS's `run_rebalance_domains`. Tasks keep their (global)
    /// vruntime across the move, so ordering on the destination stays correct.
    fn maybe_load_balance(&mut self) {
        if self.lb_interval.is_zero() || self.last_lb.elapsed() < self.lb_interval {
            return;
        }
        self.last_lb = Instant::now();

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

    /// One scheduling round: drain newly-runnable tasks, place each on a CPU
    /// (idle-first, with affinity), stamp it with its vruntime relative to that
    /// CPU's floor and a CFS `sched_slice`, push it onto that CPU's run-queue,
    /// then run the periodic load balancer. Returns true if at least one task was
    /// dispatched.
    fn dispatch_tasks(&mut self) -> bool {
        self.drain();

        let mut dispatched = false;

        // Per-CPU queue-depth snapshot for this round: drives least-loaded
        // placement and the sched_slice run-queue depth. We track placements
        // locally so repeated decisions within the round stay balanced.
        let mut qlen: Vec<i32> = vec![0; self.min_vruntime_pc.len()];
        if !self.pending.is_empty() {
            self.bpf.refresh_qlen();
            for &c in &self.workers {
                qlen[c as usize] = self.bpf.qlen(c);
            }
        }

        // Smallest vruntime dispatched to each CPU this round (to advance floors).
        let mut round_min: Vec<Option<u64>> = vec![None; self.min_vruntime_pc.len()];

        while let Some(task) = self.pending.pop_front() {
            let cpu = self.place(&task, &qlen);
            let ci = cpu as usize;

            let floor = self.min_vruntime_pc.get(ci).copied().unwrap_or(0);
            let vruntime = self.advance_vruntime(&task, floor);
            // nr_running on the target run-queue ~= queued depth + the task itself.
            let nr_running = qlen.get(ci).copied().unwrap_or(0).max(0) as u64 + 1;
            let slice_ns = self.sched_slice(&task, nr_running);

            let mut d = DispatchedTask::new(&task);
            d.cpu = cpu; // bind to this CPU's per-CPU run-queue (vtime-ordered)
            d.vtime = vruntime;
            d.slice_ns = slice_ns;

            if let Err(e) = self.bpf.dispatch_task(&d) {
                self.pending.push_front(task);
                if self.verbose {
                    eprintln!("dispatch failed: {e}");
                }
                break;
            }

            if let Some(q) = qlen.get_mut(ci) {
                *q += 1;
            }
            if let Some(slot) = round_min.get_mut(ci) {
                *slot = Some(slot.map_or(vruntime, |m| m.min(vruntime)));
            }
            dispatched = true;
        }

        // Advance each CPU's floor monotonically toward the smallest vruntime
        // dispatched to it (CFS keeps min_vruntime non-decreasing).
        for (ci, slot) in round_min.iter().enumerate() {
            if let Some(m) = slot {
                self.min_vruntime_pc[ci] = self.min_vruntime_pc[ci].max(*m);
            }
        }

        self.maybe_load_balance();

        // Report leftover (non-zero only if a push failed) so the agent keeps
        // being re-invoked to retry; workers drain their run-queues meanwhile.
        self.bpf.notify_complete(self.pending.len() as u64);
        dispatched
    }

    fn print_stats(&mut self) {
        let nr_running = *self.bpf.nr_running_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_bounce = *self.bpf.nr_bounce_dispatches_mut();
        let nr_cancel = *self.bpf.nr_cancel_dispatches_mut();
        let nr_failed = *self.bpf.nr_failed_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();

        println!(
            "cpus={} pending={} running={} user_disp={} bounce={} cancel={} fail={} migrations={}",
            self.workers.len(),
            self.pending.len(),
            nr_running,
            nr_user,
            nr_bounce,
            nr_cancel,
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

            // If this round pushed nothing (no new arrivals, or a transient ring
            // full), back off briefly instead of busy-spinning the agent CPU.
            // Workers drain their run-queues on their own. (The periodic load
            // balancer still runs from dispatch_tasks even when idle.)
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

WARNING: scx_cfs is a simple educational scheduler that makes all of its
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
        "scx_cfs: per-CPU run-queues ({} CPUs), weighted-fair (vruntime), preemptive, \
         idle-first placement, load balance every {} ms, \
         sched_latency = {} us (min slice {} us), partial = {}, watchdog = {} ms",
        workers.len(),
        opts.lb_interval_ms,
        opts.sched_latency_us,
        opts.min_slice_us,
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
