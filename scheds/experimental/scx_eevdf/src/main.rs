// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # EEVDF: a per-CPU Earliest Eligible Virtual Deadline First scheduler
//!
//! ## Overview
//!
//! `scx_eevdf` is an EEVDF-style scheduler for the Linux kernel that makes all
//! of its decisions in user-space, in 100% Rust, on top of `scx_rustland_core`
//! (which leverages the kernel's `sched_ext` feature). EEVDF is the policy that
//! replaced CFS in Linux 6.6; it keeps CFS's weighted **virtual runtime** but
//! adds two ideas — **eligibility** and **virtual deadlines** — that together
//! give it explicit latency control while preserving fairness.
//!
//! It is built on the same per-CPU structure as [`scx_cfs`](../scx_cfs): one
//! run-queue per CPU (its per-CPU DSQ), idle-first placement, cache-affine
//! stickiness, and a periodic load balancer.
//!
//! ## Virtual runtime, deadline, and eligibility
//!
//! Each task accrues virtual runtime as it runs, scaled inversely by its weight
//! (so equal-weight tasks converge to equal CPU time):
//!
//! ```text
//!   vruntime += delta_exec * NICE_0_WEIGHT / weight        (NICE_0_WEIGHT = 100)
//! ```
//!
//! Each task also has a **request** `r` (its desired time slice). From it we
//! compute a **virtual deadline**:
//!
//! ```text
//!   vslice   = r * NICE_0_WEIGHT / weight
//!   deadline = vruntime + vslice
//! ```
//!
//! EEVDF runs, among all **eligible** tasks, the one with the **earliest virtual
//! deadline**. A task is *eligible* when it has not run ahead of its fair share
//! (its vruntime is at or behind the run-queue's virtual time). The smaller a
//! task's request, the earlier its deadline relative to its vruntime, so a task
//! can ask for a smaller slice to get **lower latency** (scheduled sooner, for
//! shorter bursts) without changing its long-run CPU share (set by weight).
//!
//! ## How this maps onto the per-CPU DSQ
//!
//! Each CPU's DSQ is consumed in vtime order (`scx_bpf_dsq_insert_vtime`), so we
//! stamp each task's **virtual deadline** as its vtime: the kernel then pops the
//! earliest-deadline task automatically (the "Virtual Deadline First" half).
//!
//! The single-key DSQ can't also express the eligibility filter, so the agent
//! enforces it: a task whose vruntime has run ahead of its CPU's virtual-time
//! floor (by more than a lag band) is **parked** in user-space and only released
//! into the DSQ once the floor catches up. The most-behind task on a CPU is
//! always released, so a CPU with runnable work never idles. With this, the DSQ
//! only ever holds eligible tasks, ordered by deadline — i.e. EEVDF.
//!
//! We keep no separate vruntime table: the backend stores the vtime we stamp
//! (the deadline) in `p->scx.dsq_vtime` and reports it back as `QueuedTask.vtime`
//! on the next enqueue. Since `deadline = vruntime + vslice` is a closed form, we
//! recover vruntime by subtracting vslice (a reported vtime of 0 marks a
//! freshly-created task).
//!
//! ## Simplifications vs. the kernel EEVDF
//!
//! This is an educational model. Eligibility is approximated: the exact kernel
//! computes the load-weighted average virtual time V over the run-queue and
//! deems a task eligible when `vruntime <= V`; tracking V exactly needs the live
//! per-CPU runnable set, which a batch user-space agent can't observe, so we use
//! a per-CPU `min_vruntime` floor plus a lag band instead (the most-behind task
//! is always eligible, and a task can get at most ~one band ahead before being
//! parked — the same bounded-lag fairness). The request `r` is a single
//! configurable base slice rather than a per-task latency-nice (the framework's
//! `QueuedTask` carries no latency attribute); vruntime is a single monotonic
//! clock rather than per-run-queue-relative, so migrations don't carry the exact
//! renormalization; and there are no cgroups, NUMA domains, or hierarchical
//! sched-domains. The substantive mechanics — weighted vruntime, virtual
//! deadlines, eligibility/bounded lag, per-CPU run-queues, idle-first placement,
//! and periodic load balancing — are faithful.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[rustfmt::skip]
mod bpf;

use std::collections::HashMap;
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

/// When there is nothing newly runnable, back off this long before the next
/// round instead of busy-spinning the agent CPU. Parked (ineligible) tasks
/// become eligible only as running tasks consume their slices and re-enqueue —
/// which itself wakes the agent — so this also paces the eligibility re-check.
const IDLE_BACKOFF: Duration = Duration::from_micros(100);

const NSEC_PER_USEC: u64 = 1_000;

/// Load weight of a nice-0 task. In `scx_rustland_core` task weights are in the
/// range `[1..10000]` with `100` as the default, so 100 plays the role of the
/// kernel's `NICE_0_LOAD`: vruntime advances by `delta_exec` for a task of this
/// weight, faster for lighter tasks and slower for heavier ones.
const NICE_0_WEIGHT: u64 = 100;

/// Only consider rebalancing a CPU whose run-queue is at least this deep.
const LB_MIN_QLEN: i32 = 4;
/// Only rebalance when (max_qlen - min_qlen) exceeds this fraction of max_qlen.
const LB_IMBALANCE_FRAC: f64 = 0.25;

#[derive(Parser, Debug)]
#[command(name = "scx_eevdf", about = "EEVDF: a per-CPU Earliest Eligible Virtual Deadline First scheduler")]
struct Opts {
    /// Worker CPUs that each own a per-CPU run-queue, as a CPU list (e.g.
    /// "0-49", "0-3,8,12-15"). Tasks are placed across exactly this set; set the
    /// workload's affinity to the same set. Defaults to every online CPU.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// Base request slice, in microseconds (Linux's `sysctl_sched_base_slice`):
    /// each task's requested time slice. It is both how long a task runs before
    /// being re-evaluated and the basis for its virtual deadline (a smaller
    /// request ⇒ earlier deadline ⇒ lower latency, same long-run CPU share).
    #[arg(short = 's', long, default_value_t = 750)]
    base_slice_us: u64,

    /// Eligibility lag band, in microseconds. A task is held (ineligible) once
    /// its virtual runtime runs more than this far ahead of its CPU's
    /// virtual-time floor, bounding how far ahead of its fair share it can get
    /// (an approximation of EEVDF's zero-lag eligibility). Larger ⇒ more batching
    /// / higher throughput; smaller ⇒ tighter fairness and lower latency.
    #[arg(short = 'b', long, default_value_t = 3_000)]
    lag_band_us: u64,

    /// How often (milliseconds) to run the periodic load balancer that migrates
    /// tasks from the busiest CPU's run-queue to the idlest. 0 disables it
    /// (placement-time balancing only).
    #[arg(long, default_value_t = 250)]
    lb_interval_ms: u64,

    /// Schedule the whole system. By default scx_eevdf runs in *partial* mode: it
    /// only schedules tasks that explicitly opt in (policy SCHED_EXT), and
    /// everything else stays on the kernel's native scheduler. Set this to take
    /// over every task instead.
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

/// A task parked in user-space until it becomes eligible. Its vruntime does not
/// change while parked (it isn't running), so we cache the computed vruntime and
/// virtual deadline alongside the queued task and its assigned CPU.
struct Parked {
    qtask: QueuedTask,
    cpu: i32,
    vruntime: u64,
    deadline: u64,
}

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Tasks awaiting eligibility (and ring-full retries), keyed by pid so a
    /// duplicate enqueue simply refreshes the entry. A task leaves here the
    /// moment it is dispatched into its CPU's DSQ.
    parked: HashMap<i32, Parked>,
    /// Base request slice (ns): the dispatched time slice and the deadline basis.
    base_slice_ns: u64,
    /// Eligibility lag band (ns) above each CPU's virtual-time floor.
    lag_band_ns: u64,
    /// Worker CPUs, each owning one per-CPU virtual-deadline run-queue.
    workers: Vec<i32>,
    /// Per-CPU virtual-time floor, indexed by CPU id (≈ CFS/EEVDF `min_vruntime`,
    /// the zero-lag reference). Monotonically non-decreasing. Sized to cover the
    /// largest worker CPU id.
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
        let base_slice_ns = opts.base_slice_us.saturating_mul(NSEC_PER_USEC).max(1);
        let lag_band_ns = opts.lag_band_us.saturating_mul(NSEC_PER_USEC);

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
            // scheduled by scx_eevdf; the rest of the system stays on the native
            // kernel scheduler. This keeps background tasks out of the run-queues
            // so they can't perturb the measured workload (or trip the watchdog).
            !opts.system_wide,
            false,    // debug
            false,    // builtin_idle: false => every task is funneled through the
                      // user-space scheduler (we make all placement decisions);
                      // true would let the kernel fast-path wakeups to prev_cpu.
            false,    // numa_local
            5_000_000, // BPF-layer default slice (5 ms) — the agent's own
                      // scheduling heartbeat. Worker tasks get base_slice at
                      // dispatch.
            opts.timeout_ms,
            "eevdf",  // scx ops name
        )?;

        Ok(Self {
            bpf,
            parked: HashMap::new(),
            base_slice_ns,
            lag_band_ns,
            workers,
            min_vruntime_pc: vec![0; max_cpu + 1],
            is_worker,
            lb_interval: Duration::from_millis(opts.lb_interval_ms),
            last_lb: Instant::now(),
            verbose: opts.verbose,
        })
    }

    /// Recover the task's virtual runtime from the round-tripped deadline, charge
    /// the time it just ran, realign new/woken tasks to `floor`, and return
    /// (vruntime, virtual_deadline).
    ///
    /// The backend stores the vtime we stamp (the deadline) in p->scx.dsq_vtime
    /// and reports it back as task.vtime; since `deadline = vruntime + vslice`
    /// with a closed-form vslice, we recover vruntime by subtracting vslice.
    fn compute(&self, task: &QueuedTask, floor: u64) -> (u64, u64) {
        let weight = task.weight.max(1);
        let vslice = self.base_slice_ns.saturating_mul(NICE_0_WEIGHT) / weight;

        let prev_vruntime = if task.vtime == 0 {
            // New task: start at the floor — neither leading nor buried.
            floor
        } else {
            // Recover last vruntime from the stored deadline, but give a woken
            // task at most one lag band of sleep credit (EEVDF place_entity).
            task.vtime
                .saturating_sub(vslice)
                .max(floor.saturating_sub(self.lag_band_ns))
        };

        // delta_exec = duration of the most recent on-CPU stint (0 if never run).
        let delta_exec = task.stop_ts.saturating_sub(task.start_ts);
        let vruntime =
            prev_vruntime.saturating_add(delta_exec.saturating_mul(NICE_0_WEIGHT) / weight);
        let deadline = vruntime.saturating_add(vslice);
        (vruntime, deadline)
    }

    /// Pick the least-loaded worker CPU from a per-CPU queue-depth snapshot
    /// (the `find_idlest_cpu` fallback when no CPU is fully idle).
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
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }
        let idle = self.bpf.select_cpu(task.pid, task.cpu, task.flags);
        if idle >= 0 && self.is_worker.get(idle as usize).copied().unwrap_or(false) {
            return idle;
        }
        let prev_is_worker = self.is_worker.get(task.cpu as usize).copied().unwrap_or(false);
        if task.vtime != 0 && prev_is_worker {
            task.cpu
        } else {
            self.least_loaded(qlen)
        }
    }

    /// Periodically migrate half of the excess from the busiest run-queue to the
    /// idlest, subject to a depth floor and an imbalance threshold — a coarse
    /// stand-in for CFS/EEVDF's `run_rebalance_domains`. Tasks keep their vtime
    /// (deadline) across the move, so ordering on the destination stays correct.
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

    /// Release the eligible tasks parked on each CPU into that CPU's DSQ, stamped
    /// with their virtual deadline (so the kernel pops earliest-deadline-first).
    /// Per CPU: sort the parked tasks by vruntime, then dispatch a prefix — the
    /// most-behind task is always eligible (guaranteeing progress / no idle CPU),
    /// and each subsequent task while its vruntime is within one lag band of the
    /// CPU's floor. The rest stay parked until the floor advances. Advances each
    /// CPU's floor toward the smallest vruntime it dispatched.
    fn release_eligible(&mut self) -> bool {
        // Bucket parked tasks by CPU: (vruntime, pid).
        let mut by_cpu: Vec<Vec<(u64, i32)>> = vec![Vec::new(); self.min_vruntime_pc.len()];
        for (pid, p) in &self.parked {
            if let Some(b) = by_cpu.get_mut(p.cpu as usize) {
                b.push((p.vruntime, *pid));
            }
        }

        let mut dispatched = false;
        for ci in 0..by_cpu.len() {
            let cands = &mut by_cpu[ci];
            if cands.is_empty() {
                continue;
            }
            cands.sort_unstable(); // by vruntime, then pid

            let floor = self.min_vruntime_pc[ci];
            let limit = floor.saturating_add(self.lag_band_ns);
            let mut dispatched_min: Option<u64> = None;

            for (idx, &(vrt, pid)) in cands.iter().enumerate() {
                // Always release the most-behind task (idx 0) so a CPU with
                // runnable work never idles; release the rest while eligible.
                if idx != 0 && vrt > limit {
                    break; // sorted => everything after is ineligible too
                }
                let (cpu, deadline, mut d) = {
                    let p = &self.parked[&pid];
                    (p.cpu, p.deadline, DispatchedTask::new(&p.qtask))
                };
                d.cpu = cpu; // bind to this CPU's per-CPU DSQ (vtime-ordered)
                d.vtime = deadline; // earliest-deadline-first among the eligible
                d.slice_ns = self.base_slice_ns;

                if let Err(e) = self.bpf.dispatch_task(&d) {
                    // Ring full: leave it parked and retry next round.
                    if self.verbose {
                        eprintln!("dispatch failed: {e}");
                    }
                    break;
                }
                self.parked.remove(&pid);
                dispatched_min.get_or_insert(vrt);
                dispatched = true;
            }

            // Advance the floor monotonically toward the smallest vruntime
            // dispatched to this CPU.
            if let Some(m) = dispatched_min {
                self.min_vruntime_pc[ci] = self.min_vruntime_pc[ci].max(m);
            }
        }
        dispatched
    }

    /// One scheduling round: drain newly-runnable tasks (placing each on a CPU
    /// and parking it with its computed vruntime/deadline), release the eligible
    /// parked tasks into their CPUs' DSQs, then run the periodic load balancer.
    /// Returns true if there was new work this round.
    fn dispatch_tasks(&mut self) -> bool {
        // Drain all newly-runnable tasks first (cheap, no syscalls per task).
        let mut arrivals: Vec<QueuedTask> = Vec::new();
        while let Ok(Some(task)) = self.bpf.dequeue_task() {
            arrivals.push(task);
        }

        // Fully idle: nothing new and nothing parked. Report 0 and let the caller
        // back off.
        if arrivals.is_empty() && self.parked.is_empty() {
            self.bpf.notify_complete(0);
            return false;
        }

        // Snapshot per-CPU queue depths for placement (least-loaded fallback) and
        // track placements locally so repeated decisions within the round stay
        // balanced.
        let mut qlen: Vec<i32> = vec![0; self.min_vruntime_pc.len()];
        self.bpf.refresh_qlen();
        for &c in &self.workers {
            qlen[c as usize] = self.bpf.qlen(c);
        }

        // Place each arrival on a CPU and park it with its computed vruntime and
        // virtual deadline. (Parking, then releasing the eligible ones below, is
        // uniform: new tasks start at the floor and so are eligible immediately.)
        let had_arrivals = !arrivals.is_empty();
        for task in arrivals {
            let cpu = self.place(&task, &qlen);
            let ci = cpu as usize;
            if let Some(q) = qlen.get_mut(ci) {
                *q += 1;
            }
            let floor = self.min_vruntime_pc.get(ci).copied().unwrap_or(0);
            let (vruntime, deadline) = self.compute(&task, floor);
            self.parked.insert(
                task.pid,
                Parked {
                    qtask: task,
                    cpu,
                    vruntime,
                    deadline,
                },
            );
        }

        self.release_eligible();
        self.maybe_load_balance();

        // Report still-parked tasks so the backend keeps re-invoking the agent to
        // re-check their eligibility (they become eligible as running tasks
        // consume their slices and advance the floor).
        self.bpf.notify_complete(self.parked.len() as u64);
        had_arrivals
    }

    fn print_stats(&mut self) {
        let nr_running = *self.bpf.nr_running_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_bounce = *self.bpf.nr_bounce_dispatches_mut();
        let nr_cancel = *self.bpf.nr_cancel_dispatches_mut();
        let nr_failed = *self.bpf.nr_failed_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();

        println!(
            "cpus={} parked={} running={} user_disp={} bounce={} cancel={} fail={} migrations={}",
            self.workers.len(),
            self.parked.len(),
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
            let had_arrivals = self.dispatch_tasks();

            // Back off when nothing newly runnable arrived. Parked tasks become
            // eligible only as running tasks consume their slices and re-enqueue
            // (which counts as an arrival and wakes us), so there is no value in
            // spinning to re-check eligibility in between.
            if !had_arrivals {
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

WARNING: scx_eevdf is a simple educational scheduler that makes all of its
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
        "scx_eevdf: per-CPU run-queues ({} CPUs), earliest-eligible-virtual-deadline-first, \
         preemptive, base slice = {} us, lag band = {} us, load balance every {} ms, \
         partial = {}, watchdog = {} ms",
        workers.len(),
        opts.base_slice_us,
        opts.lag_band_us,
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
