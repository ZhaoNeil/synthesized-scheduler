// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # ALPS: a learned-priority, SRPT-approximating scheduler (user-space)
//!
//! ## Overview
//!
//! `scx_alps` is a port of **ALPS** (USENIX ATC'24, ds2-lab/ALPS) onto
//! `scx_rustland_core`, the same user-space-agent framework as this project's
//! other schedulers (cFIFO/RR/CFS/EEVDF). ALPS speeds up serverless (FaaS)
//! workloads by approximating **Shortest-Remaining-Processing-Time**: it learns a
//! per-*function-class* execution-time prediction from recent history and gives
//! shorter-predicted classes higher priority, so short functions finish quickly
//! instead of queueing behind long ones.
//!
//! This is the **core SRPT-approximation** variant (no alpha/beta/theta/gamma
//! finetuning, no learned-regression predictor) expressed on the agent model:
//!
//!   * **Mechanism = weighted-fair (CFS) by learned priority.** The agent runs the
//!     same per-CPU virtual-runtime dispatch as [`scx_cfs`](../scx_cfs) — one
//!     vruntime-ordered run-queue per CPU, idle-first placement, periodic load
//!     balancing — but a task's weight comes from its **class's SRPT rank**, not
//!     its nice value. The shortest-predicted class gets the largest CFS weight,
//!     so its vruntime advances slowest and it receives proportionally more CPU
//!     (ALPS' "priority on CFS"). Rank maps to a nice level spread across
//!     [-20, 19], and nice maps to the kernel's `sched_prio_to_weight` table.
//!
//!   * **Learning frontend.** Periodically the agent (re)learns each class's
//!     execution time as a moving average of recently *completed* tasks, ranks the
//!     classes shortest-first, and recomputes their weights. This is off the
//!     dispatch hot path — between dispatch rounds, every `--scan-interval-ms`.
//!
//! ## What is a "class"?
//!
//! A function's class is the worker's launch arguments — e.g. `cpu_fib 32` or
//! `mem_stream 33`. The BPF backend can't see this (every worker's `comm` is
//! `launch_function`), so the agent reads `/proc/<pid>/cmdline` **once per task**
//! to classify it (skipping the launcher's own `--quiet`/`--metrics-*` flags).
//! Completed tasks are detected by a not-dispatched-for-1s grace (see
//! `REAP_GRACE`), and their total on-CPU time becomes a class sample. Unlike the
//! other agents — which need no task introspection and so do no I/O — ALPS must
//! read `/proc`; this is safe because `/proc` is in-memory (procfs), so it cannot
//! block on an ext4 inode lock held by a task the agent must schedule.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[rustfmt::skip]
mod bpf;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
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

/// CFS weight of a nice-0 task in the kernel's `sched_prio_to_weight` table; the
/// reference load against which vruntime advances (`vruntime += delta * NICE_0 /
/// weight`).
const NICE_0_LOAD: u64 = 1024;

/// Kernel CFS `sched_prio_to_weight[40]`: weight for nice -20..+19 (index =
/// nice + 20). Higher weight => slower vruntime => more CPU share. We map a
/// class's SRPT rank onto a nice level and then to this weight, so the
/// shortest-predicted class is scheduled like a high-priority (very negative
/// nice) task.
const SCHED_PRIO_TO_WEIGHT: [u64; 40] = [
    88761, 71755, 56483, 46273, 36291, // nice -20 ..  -16
    29154, 23254, 18705, 14949, 11916, // nice -15 ..  -11
    9548, 7620, 6100, 4904, 3906, //       nice -10 ..   -6
    3121, 2501, 1991, 1586, 1277, //       nice  -5 ..   -1
    1024, //                               nice   0
    820, 655, 526, 423, //                 nice   1 ..    4
    335, 272, 215, 172, 137, //            nice   5 ..    9
    110, 87, 70, 56, 45, //                nice  10 ..   14
    36, 29, 23, 18, 15, //                 nice  15 ..   19
];
const NICE_MIN: i32 = -20;
const NICE_MAX: i32 = 19;

/// Recent completed-execution samples kept per class for the moving average.
const CLASS_WINDOW: usize = 64;
/// Predicted exec time for a class with no samples yet (neutral priority).
const DEFAULT_PRED_NS: u64 = 4_000_000;
/// A task not dispatched for this long is treated as completed (its total on-CPU
/// time becomes a class sample). Under preemptive CFS slicing a still-runnable
/// task is re-enqueued every slice, so only completed (or long-blocked) tasks go
/// quiet this long. This replaces /proc liveness checks, which a batch agent
/// can't afford per-task and which `replay_trace` defeats by leaving completed
/// workers as zombies (their /proc entry lingers).
const REAP_GRACE: Duration = Duration::from_secs(1);

/// CFS-tier load balancer thresholds (as in scx_cfs).
const LB_MIN_QLEN: i32 = 4;
const LB_IMBALANCE_FRAC: f64 = 0.25;

#[derive(Parser, Debug)]
#[command(name = "scx_alps", about = "ALPS: a learned-priority, SRPT-approximating scheduler")]
struct Opts {
    /// Worker CPUs that each own a per-CPU run-queue, as a CPU list (e.g.
    /// "0-49"). Set the workload's affinity to the same set. Defaults to every
    /// online CPU.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// Time slice in microseconds (the preemption quantum). Priority is expressed
    /// through vruntime/weight, not the slice, so this is a fixed quantum.
    #[arg(short = 's', long, default_value_t = 3_000)]
    slice_us: u64,

    /// How often (milliseconds) to re-learn class predictions, re-rank classes
    /// shortest-first, and recompute their weights. Off the dispatch hot path.
    #[arg(long, default_value_t = 50)]
    scan_interval_ms: u64,

    /// How often (milliseconds) to run the per-CPU load balancer (busiest =>
    /// idlest). 0 disables it (placement-time balancing only).
    #[arg(long, default_value_t = 250)]
    lb_interval_ms: u64,

    /// Schedule the whole system. By default scx_alps runs in *partial* mode: it
    /// only schedules tasks that opt in (policy SCHED_EXT); everything else stays
    /// on the kernel's native scheduler.
    #[arg(long, default_value_t = false)]
    system_wide: bool,

    /// Print scheduling statistics once per second.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// sched_ext runnable-task watchdog timeout, in milliseconds. If any task
    /// stays runnable longer than this the kernel ejects the scheduler and
    /// mass-migrates back to CFS; an oversubscribed system can exceed the ~30 s
    /// default, so raise it. Also settable via SCX_TIMEOUT_MS.
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

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Worker CPUs, each owning one per-CPU vruntime run-queue.
    workers: Vec<i32>,
    is_worker: Vec<bool>,
    /// Per-CPU virtual-time floor (CFS `min_vruntime`), indexed by CPU id.
    min_vruntime_pc: Vec<u64>,
    /// Dispatch slice / preemption quantum (ns).
    slice_ns: u64,

    // ---- ALPS learning state ----
    /// pid -> class id (cached; the class is read from /proc/<pid>/cmdline once).
    pid_class: HashMap<i32, u32>,
    /// pid -> cumulative on-CPU time (ns); the task's total service time so far.
    pid_exec: HashMap<i32, u64>,
    /// pid -> last time it was dispatched; a pid not seen for REAP_GRACE is reaped.
    pid_seen: HashMap<i32, Instant>,
    /// Drained-but-not-yet-dispatched tasks (carries ring-full leftovers to the
    /// next round so a full dispatch ring never drops a task).
    pending: VecDeque<QueuedTask>,
    /// class name -> class id.
    name_to_id: HashMap<String, u32>,
    /// class id -> recent completed-execution samples (ns).
    class_samples: HashMap<u32, VecDeque<u64>>,
    /// class id -> predicted execution time (ns).
    class_pred: HashMap<u32, u64>,
    /// class id -> CFS weight derived from its SRPT rank.
    class_weight: HashMap<u32, u64>,

    scan_interval: Duration,
    last_scan: Instant,

    lb_interval: Duration,
    last_lb: Instant,

    verbose: bool,
}

// The agent does no *persistent-filesystem* I/O (which could block on an ext4
// inode lock held by a task it must schedule -- a priority inversion). It does
// read /proc (procfs, in-memory) to learn function classes, which is free of that
// hazard.

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
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
            0,        // exit_dump_len
            !opts.system_wide, // partial
            false,    // debug
            false,    // builtin_idle: agent makes all placement decisions
            false,    // numa_local
            5_000_000, // BPF-layer default slice (agent heartbeat)
            opts.timeout_ms,
            "alps",
        )?;

        let mut sched = Self {
            bpf,
            workers,
            is_worker,
            min_vruntime_pc: vec![0; max_cpu + 1],
            slice_ns: opts.slice_us.saturating_mul(NSEC_PER_USEC).max(1),
            pid_class: HashMap::new(),
            pid_exec: HashMap::new(),
            pid_seen: HashMap::new(),
            pending: VecDeque::new(),
            name_to_id: HashMap::new(),
            class_samples: HashMap::new(),
            class_pred: HashMap::new(),
            class_weight: HashMap::new(),
            scan_interval: Duration::from_millis(opts.scan_interval_ms),
            last_scan: Instant::now(),
            lb_interval: Duration::from_millis(opts.lb_interval_ms),
            last_lb: Instant::now(),
            verbose: opts.verbose,
        };
        // Reserve a neutral "default" class for workers we can't classify.
        sched.intern_class("default");
        Ok(sched)
    }

    // ---- per-class learning --------------------------------------------------

    fn intern_class(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.name_to_id.len() as u32;
        self.name_to_id.insert(name.to_string(), id);
        self.class_pred.insert(id, DEFAULT_PRED_NS);
        // Until the next scan recomputes ranks, a new class runs at nice 0.
        self.class_weight.insert(id, NICE_0_LOAD);
        id
    }

    /// The CFS weight to schedule a class at (defaults to nice-0 if unranked).
    fn weight_of(&self, class_id: u32) -> u64 {
        self.class_weight.get(&class_id).copied().unwrap_or(NICE_0_LOAD)
    }

    fn record_sample(&mut self, class_id: u32, exec_ns: u64) {
        if exec_ns == 0 {
            return;
        }
        let w = self.class_samples.entry(class_id).or_default();
        w.push_back(exec_ns);
        while w.len() > CLASS_WINDOW {
            w.pop_front();
        }
    }

    /// Re-learn predictions (moving average), rank classes shortest-first (SRPT),
    /// and map each rank to a CFS weight via nice.
    fn relearn(&mut self) {
        for (&id, w) in &self.class_samples {
            if !w.is_empty() {
                let sum: u128 = w.iter().map(|&x| x as u128).sum();
                self.class_pred.insert(id, (sum / w.len() as u128) as u64);
            }
        }
        let mut classes: Vec<u32> = self.class_pred.keys().copied().collect();
        // Ascending predicted exec => shortest gets rank 0 (SRPT).
        classes.sort_by_key(|c| (self.class_pred[c], *c));
        let n = classes.len().max(1) as i64;
        for (rank, c) in classes.into_iter().enumerate() {
            let nice = if n <= 1 {
                0
            } else {
                let span = (NICE_MAX - NICE_MIN) as i64;
                (NICE_MIN as i64 + (rank as i64 * span) / (n - 1))
                    .clamp(NICE_MIN as i64, NICE_MAX as i64)
            };
            let weight = SCHED_PRIO_TO_WEIGHT[(nice + 20) as usize];
            self.class_weight.insert(c, weight);
        }
    }

    /// Reap completed tasks: a pid not dispatched for REAP_GRACE is treated as
    /// done, and its total on-CPU time becomes a sample for its class. This is a
    /// pure in-memory scan (no syscalls), so it stays cheap even with thousands of
    /// tracked pids and never stalls the agent (which a per-pid /proc liveness
    /// check would, especially since completed workers linger as zombies).
    fn reap(&mut self, now: Instant) {
        let gone: Vec<i32> = self
            .pid_seen
            .iter()
            .filter(|(_, &seen)| now.duration_since(seen) >= REAP_GRACE)
            .map(|(&pid, _)| pid)
            .collect();
        for pid in gone {
            let class_id = self.pid_class.remove(&pid).unwrap_or(0);
            if let Some(exec) = self.pid_exec.remove(&pid) {
                self.record_sample(class_id, exec);
            }
            self.pid_seen.remove(&pid);
        }
    }

    /// A worker's function class: its launch args (kind + level), e.g.
    /// "cpu_fib 32", skipping the launcher's own option flags.
    fn read_class_name(pid: i32) -> Option<String> {
        let data = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        let toks: Vec<String> = data
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        // toks[0] is the launcher path; the function args come after the
        // launcher's own flags (--quiet, and --calibration/--label/--metrics-*
        // which each take one value).
        let mut out: Vec<String> = Vec::new();
        let mut i = 1;
        while i < toks.len() {
            let t = &toks[i];
            match t.as_str() {
                "--quiet" => {}
                "--calibration" | "--label" | "--metrics-fd" | "--metrics-index"
                | "--metrics-count" => {
                    i += 1; // skip this flag's value too
                }
                _ => out.push(t.clone()),
            }
            i += 1;
        }
        if out.is_empty() {
            None
        } else {
            Some(out.join(" "))
        }
    }

    // ---- placement (same as scx_cfs) ----------------------------------------

    fn place(
        bpf: &mut BpfScheduler,
        is_worker: &[bool],
        workers: &[i32],
        task: &QueuedTask,
        qlen: &[i32],
    ) -> i32 {
        if task.nr_cpus_allowed <= 1 {
            return task.cpu;
        }
        let idle = bpf.select_cpu(task.pid, task.cpu, task.flags);
        if idle >= 0 && is_worker.get(idle as usize).copied().unwrap_or(false) {
            return idle;
        }
        let prev_is_worker = is_worker.get(task.cpu as usize).copied().unwrap_or(false);
        if task.vtime != 0 && prev_is_worker {
            return task.cpu; // stickiness for a returning task
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

    /// One scheduling round: drain newly-runnable tasks, classify each, advance
    /// its virtual runtime scaled by its *class weight* (SRPT priority), and
    /// dispatch it onto its CPU's vruntime run-queue. Periodically re-learn the
    /// class predictions/weights and reap completed tasks.
    fn dispatch_tasks(&mut self) -> bool {
        let now = Instant::now();

        // Drain newly-runnable tasks onto the tail of the retry buffer (which may
        // already hold tasks a full dispatch ring made us defer last round).
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

                // Classify once; track total service time and last-seen time.
                let class_id = match self.pid_class.get(&pid) {
                    Some(&id) => id,
                    None => {
                        let name = Self::read_class_name(pid)
                            .unwrap_or_else(|| "default".to_string());
                        let id = self.intern_class(&name);
                        self.pid_class.insert(pid, id);
                        id
                    }
                };
                let delta_exec = task.stop_ts.saturating_sub(task.start_ts);
                *self.pid_exec.entry(pid).or_insert(0) += delta_exec;
                self.pid_seen.insert(pid, now);

                // Place on a CPU and advance vruntime scaled by the class weight.
                let cpu = Self::place(
                    &mut self.bpf,
                    &self.is_worker,
                    &self.workers,
                    &task,
                    &qlen,
                );
                let ci = cpu as usize;
                let floor = self.min_vruntime_pc.get(ci).copied().unwrap_or(0);
                let weight = self.weight_of(class_id);
                let base = if task.vtime == 0 {
                    floor
                } else {
                    task.vtime.max(floor.saturating_sub(self.slice_ns))
                };
                let vruntime =
                    base.saturating_add(delta_exec.saturating_mul(NICE_0_LOAD) / weight);

                let mut d = DispatchedTask::new(&task);
                d.cpu = cpu;
                d.vtime = vruntime; // per-CPU run-queue ordered by vruntime
                d.slice_ns = self.slice_ns;

                if self.bpf.dispatch_task(&d).is_err() {
                    // Ring full: put it back at the head and retry next round.
                    self.pending.push_front(task);
                    break;
                }
                if let Some(q) = qlen.get_mut(ci) {
                    *q += 1;
                }
                if let Some(f) = self.min_vruntime_pc.get_mut(ci) {
                    *f = (*f).max(vruntime);
                }
                dispatched = true;
            }
        }

        // Periodic ALPS frontend: reap completed tasks, re-learn, re-rank.
        if self.last_scan.elapsed() >= self.scan_interval {
            self.reap(now);
            self.relearn();
            self.last_scan = Instant::now();
        }

        self.maybe_load_balance();
        // Report still-pending (ring-full leftovers) so the backend re-invokes us.
        self.bpf.notify_complete(self.pending.len() as u64);
        dispatched
    }

    fn print_stats(&mut self) {
        let nr_running = *self.bpf.nr_running_mut();
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();
        // Shortest and longest currently-predicted classes (us), for insight.
        let shortest = self.class_pred.values().min().copied().unwrap_or(0) / 1000;
        let longest = self.class_pred.values().max().copied().unwrap_or(0) / 1000;
        println!(
            "cpus={} classes={} tracked_pids={} pred_us=[{}..{}] running={} user_disp={} lb_migr={}",
            self.workers.len(),
            self.name_to_id.len(),
            self.pid_class.len(),
            shortest,
            longest,
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

WARNING: scx_alps is a simple educational scheduler that makes all of its
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
        "scx_alps: ALPS (learned per-class SRPT priority on per-CPU CFS), {} CPUs, \
         slice = {} us, learn every {} ms, load balance every {} ms, partial = {}, \
         watchdog = {} ms",
        workers.len(),
        opts.slice_us,
        opts.scan_interval_ms,
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
