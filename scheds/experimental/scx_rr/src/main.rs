// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # RR: round-robin scheduler with per-core run-queues (user-space), with
//! pluggable load-balancing mechanisms
//!
//! ## Overview
//!
//! `scx_rr` assigns arriving tasks to worker cores in round-robin order. Unlike
//! the centralized [`scx_cfifo`](../scx_cfifo), each core has its own queue. It
//! makes all of its decisions in user-space, in 100% Rust, on top of
//! `scx_rustland_core` (which leverages the kernel's `sched_ext` feature).
//!
//! There is **one FIFO queue per worker core**, implemented as that core's
//! per-CPU `sched_ext` dispatch queue (DSQ) in the kernel. **There is no global
//! queue** — a core only ever runs tasks from its own queue, consumed FIFO by
//! the `scx_rustland_core` backend in `ops.dispatch`.
//!
//! ## Mechanisms (`--mechanism`)
//!
//! The base scheduler is plain RR; one optional load-balancing mechanism can
//! be layered on top, selected at launch:
//!
//!   * **none** (default) — round-robin placement, run-to-completion (infinite
//!     slice, never preempted).
//!   * **preempt** — round-robin placement, but tasks run with a fixed time
//!     slice (default 1000 ms); a task that exhausts its slice is requeued at the
//!     tail of *its own* core's queue (FIFO, no migration).
//!   * **p2c** (power-of-two random choices) — each arriving task samples two
//!     random cores and joins the one with the shorter queue; run-to-completion.
//!   * **shuffle** (work shuffling) — round-robin placement, run-to-completion,
//!     plus a periodic rebalance: every `--shuffle-interval-ms` the most-loaded
//!     core's queue, if deep enough, sheds half its excess to the least-loaded
//!     core.
//!   * **preempt-shuffle** — combination of the two above: tasks run with a fixed
//!     time slice (preempt) *and* the agent periodically rebalances per-core queue
//!     depth (shuffle). A preempted task is requeued on the core it currently sits
//!     on (its last-run CPU), so a shuffle migration is preserved instead of being
//!     bounced back to the core round-robin originally assigned it.
//!
//! ## Why an agent at all?
//!
//! In `scx_rustland_core` the scheduling *policy* lives entirely in user-space;
//! the BPF backend is policy-agnostic and only reports runnable tasks to the
//! agent and executes the placement it sends back. The placement and
//! load-balancing decisions *are* the policy, so they run here in the agent. For
//! the queue-length-aware mechanisms (p2c, shuffle) the agent reads each core's
//! current backlog via a per-CPU DSQ-depth snapshot the backend exposes
//! (`refresh_qlen`/`qlen`), and shuffling asks the backend to move tasks between
//! per-CPU DSQs (`request_migration`).

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
use clap::ValueEnum;
use libbpf_rs::OpenObject;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::UserExitInfo;

/// When tasks are waiting but the dispatch ring is momentarily full, back off
/// this long before the next round to avoid busy-spinning the agent CPU.
const IDLE_BACKOFF: Duration = Duration::from_micros(100);

/// Load-balancing mechanism layered on top of the base RR scheduler.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lower")]
enum Mechanism {
    /// Plain RR: round-robin placement, run-to-completion.
    None,
    /// Round-robin + fixed time slice; over-quota tasks requeued at local tail.
    Preempt,
    /// Power-of-two random choices: sample two cores, join the shorter queue.
    P2c,
    /// Round-robin + periodic migration from the most- to the least-loaded core.
    Shuffle,
    /// preempt + shuffle: fixed time slice *and* periodic queue rebalancing. A
    /// preempted task is requeued on the core it currently sits on (its last-run
    /// CPU), so a shuffle migration is preserved rather than bounced back.
    #[value(name = "preempt-shuffle")]
    PreemptShuffle,
}

#[derive(Parser, Debug)]
#[command(name = "scx_rr", about = "RR: round-robin scheduler with per-core run-queues")]
struct Opts {
    /// Worker CPUs that each own a per-core FIFO queue, given as a CPU list
    /// (e.g. "0-49", "0-3,8,12-15"). Tasks are placed across exactly this set.
    /// For a clean run, set the workload's CPU affinity to the same set: a task
    /// placed on a CPU it isn't allowed on is bounced back to its previous CPU
    /// by the backend, which skews the distribution. Defaults to every online
    /// CPU.
    #[arg(short = 'w', long)]
    workers: Option<String>,

    /// Load-balancing mechanism layered on the base RR scheduler.
    #[arg(short = 'm', long, value_enum, default_value_t = Mechanism::None)]
    mechanism: Mechanism,

    /// [preempt] Per-task time slice in milliseconds. A task that exhausts it is
    /// requeued at the tail of its own core's queue.
    #[arg(long, default_value_t = 1000)]
    preempt_slice_ms: u64,

    /// [shuffle] How often (milliseconds) to check whether to rebalance.
    #[arg(long, default_value_t = 250)]
    shuffle_interval_ms: u64,

    /// [shuffle] Only rebalance when the most-loaded core has strictly more than
    /// this many queued tasks.
    #[arg(long, default_value_t = 2)]
    shuffle_min_qlen: i32,

    /// [shuffle] Only rebalance when (max_qlen - min_qlen) exceeds this fraction
    /// of max_qlen (anti-thrashing guard).
    #[arg(long, default_value_t = 0.05)]
    shuffle_frac: f64,

    /// Schedule the whole system. By default scx_rr runs in *partial* mode:
    /// it only schedules tasks that explicitly opt in (policy SCHED_EXT), and
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
    /// thousands of tasks, can freeze the whole machine. Run-to-completion on an
    /// oversubscribed system routinely exceeds the ~30 s default, so raise this.
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

/// Resolve the worker CPU set from the options (explicit list, or every online
/// CPU by default).
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

/// Tiny non-cryptographic PRNG (xorshift64*) for power-of-two random sampling.
/// Avoids pulling in an external rng crate.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform-ish index in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    mechanism: Mechanism,
    /// The worker cores, each owning one per-CPU FIFO queue.
    workers: Vec<i32>,
    /// Round-robin cursor into `workers`; advances once per round-robined task.
    rr: usize,
    /// Monotonically increasing vtime stamped on each dispatched task, so tasks
    /// inserted into any single per-CPU DSQ stay in strict arrival (FIFO) order.
    seq: u64,
    /// Per-task time slice (ns): u64::MAX (run-to-completion) for every mechanism
    /// except `preempt`, which uses a finite quota.
    slice_ns: u64,
    /// Transient user-space buffer holding drained tasks not yet pushed to a
    /// per-CPU DSQ (only retains tasks when a dispatch fails). The real per-core
    /// backlog lives in the kernel per-CPU DSQs.
    pending: VecDeque<QueuedTask>,
    /// [preempt] pid -> assigned core, so a task requeued after exhausting its
    /// slice returns to the same (local) core instead of being round-robined
    /// afresh. Grows with distinct pids; fine for bounded trace runs.
    assigned: HashMap<i32, i32>,
    /// [p2c] PRNG for sampling cores.
    rng: Rng,
    /// [shuffle] rebalance cadence and trigger thresholds.
    shuffle_interval: Duration,
    shuffle_min_qlen: i32,
    shuffle_frac: f64,
    last_shuffle: Instant,
    verbose: bool,
}

// The agent deliberately performs no file I/O: it could block on an ext4 inode
// lock held by a task it must schedule -- a priority inversion that deadlocks
// the scheduler. Per-task lifecycle timestamps (TaskNew/FirstRun/TaskDead) are
// recorded by the task processes themselves into shared memory (see
// project/function_trace/metrics_shm.h).

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
        let workers = resolve_workers(opts)?;

        // Per-task slice: finite quota for the preempting mechanisms,
        // run-to-completion otherwise.
        let slice_ns = match opts.mechanism {
            Mechanism::Preempt | Mechanism::PreemptShuffle => {
                opts.preempt_slice_ms.saturating_mul(1_000_000)
            }
            _ => u64::MAX,
        };

        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
            ^ (std::process::id() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);

        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,        // exit_dump_len (0 = default)
            // partial: true (default) => only SCHED_EXT opt-in tasks are
            // scheduled by scx_rr; the rest of the system stays on the native
            // kernel scheduler. Keeps background tasks off the per-core queues.
            !opts.system_wide,
            false,    // debug
            false,    // builtin_idle: false => every task is funneled through the
                      // agent so the placement policy is honored; true would let
                      // the kernel fast-path wakeups to prev_cpu, bypassing it.
            false,    // numa_local
            5_000_000, // BPF-layer default slice (5 ms) — the agent's own
                      // heartbeat. Worker tasks get `slice_ns` at dispatch.
            opts.timeout_ms,
            "rr",  // scx ops name
        )?;

        Ok(Self {
            bpf,
            mechanism: opts.mechanism,
            workers,
            rr: 0,
            seq: 0,
            slice_ns,
            pending: VecDeque::new(),
            assigned: HashMap::new(),
            rng: Rng::new(seed),
            shuffle_interval: Duration::from_millis(opts.shuffle_interval_ms),
            shuffle_min_qlen: opts.shuffle_min_qlen,
            shuffle_frac: opts.shuffle_frac,
            last_shuffle: Instant::now(),
            verbose: opts.verbose,
        })
    }

    /// Drain every task the kernel has made runnable into the tail of the
    /// pending buffer, preserving arrival (FIFO) order.
    fn drain(&mut self) {
        while let Ok(Some(task)) = self.bpf.dequeue_task() {
            self.pending.push_back(task);
        }
    }

    /// Pick a worker index for a power-of-two task: sample two distinct workers
    /// and return the one with the shorter (locally-tracked) queue.
    fn p2c_pick(&mut self, qlen: &[i32]) -> usize {
        let n = qlen.len();
        if n <= 1 {
            return 0;
        }
        let a = self.rng.below(n);
        let mut b = self.rng.below(n);
        if b == a {
            b = (b + 1) % n;
        }
        if qlen[a] <= qlen[b] {
            a
        } else {
            b
        }
    }

    /// [shuffle] Periodically migrate half of the excess from the most-loaded
    /// core to the least-loaded one, subject to the trigger thresholds.
    fn maybe_shuffle(&mut self) {
        if self.last_shuffle.elapsed() < self.shuffle_interval {
            return;
        }
        self.last_shuffle = Instant::now();

        // Snapshot every worker's current per-CPU DSQ depth.
        self.bpf.refresh_qlen();
        let mut max_c = self.workers[0];
        let mut min_c = self.workers[0];
        let mut max_q = self.bpf.qlen(max_c);
        let mut min_q = max_q;
        for &c in &self.workers {
            let q = self.bpf.qlen(c);
            if q > max_q {
                max_q = q;
                max_c = c;
            }
            if q < min_q {
                min_q = q;
                min_c = c;
            }
        }

        // Trigger: most-loaded core deep enough, and the imbalance exceeds the
        // anti-thrashing fraction of the maximum queue length.
        let diff = max_q - min_q;
        if max_c != min_c
            && max_q > self.shuffle_min_qlen
            && (diff as f64) > self.shuffle_frac * (max_q as f64)
        {
            let nr = (diff / 2) as u32;
            if nr > 0 {
                self.bpf.request_migration(max_c, min_c, nr);
            }
        }
    }

    /// One scheduling round: drain newly-runnable tasks, place each on a worker's
    /// per-CPU queue according to the active mechanism, and (for shuffle) run the
    /// periodic rebalance. Returns true if at least one task was dispatched.
    fn dispatch_tasks(&mut self) -> bool {
        self.drain();

        let n = self.workers.len();
        let mut dispatched = false;

        // [p2c] Take a fresh per-core depth snapshot for this round, then track
        // placements locally so repeated picks within the round stay balanced.
        let mut local_qlen: Vec<i32> = Vec::new();
        if self.mechanism == Mechanism::P2c && !self.pending.is_empty() {
            self.bpf.refresh_qlen();
            local_qlen = self.workers.iter().map(|&c| self.bpf.qlen(c)).collect();
        }

        while let Some(task) = self.pending.pop_front() {
            // A task pinned to a single CPU can only run there; send it straight
            // to that core and don't apply the placement policy. (The agent only
            // knows the *count* of allowed CPUs, not the mask; any other affinity
            // mismatch is caught by the backend, which bounces the task to its
            // previous CPU -- still a per-CPU queue, never the shared DSQ.)
            let single_cpu = task.nr_cpus_allowed <= 1;

            // round_robin: whether this placement consumed a round-robin slot.
            let (cpu, round_robin) = if single_cpu {
                (task.cpu, false)
            } else {
                match self.mechanism {
                    Mechanism::None | Mechanism::Shuffle => (self.workers[self.rr % n], true),
                    Mechanism::Preempt => {
                        if let Some(&c) = self.assigned.get(&task.pid) {
                            (c, false) // requeue to its local core
                        } else {
                            let c = self.workers[self.rr % n];
                            self.assigned.insert(task.pid, c);
                            (c, true)
                        }
                    }
                    Mechanism::PreemptShuffle => {
                        // First sighting of a pid is a fresh arrival: round-robin
                        // it (and remember we've seen it). A later sighting is a
                        // preemption requeue: keep it on the core it currently
                        // sits on (task.cpu = its last-run CPU) so that a shuffle
                        // migration is preserved instead of being bounced back to
                        // the original round-robin core.
                        if self.assigned.contains_key(&task.pid) {
                            let c = if task.cpu >= 0 {
                                task.cpu
                            } else {
                                self.workers[self.rr % n]
                            };
                            (c, false)
                        } else {
                            let c = self.workers[self.rr % n];
                            self.assigned.insert(task.pid, c);
                            (c, true)
                        }
                    }
                    Mechanism::P2c => {
                        let i = self.p2c_pick(&local_qlen);
                        local_qlen[i] += 1;
                        (self.workers[i], false)
                    }
                }
            };

            let mut d = DispatchedTask::new(&task);
            d.cpu = cpu; // bind to this worker's per-CPU queue (never SHARED_DSQ)
            d.vtime = self.seq; // monotonic => per-core FIFO
            d.slice_ns = self.slice_ns;

            if let Err(e) = self.bpf.dispatch_task(&d) {
                // Ring full: put the task back at the head and retry next round.
                self.pending.push_front(task);
                if self.verbose {
                    eprintln!("dispatch failed: {e}");
                }
                break;
            }

            if round_robin {
                self.rr = self.rr.wrapping_add(1);
            }
            self.seq += 1;
            dispatched = true;
        }

        if matches!(self.mechanism, Mechanism::Shuffle | Mechanism::PreemptShuffle) {
            self.maybe_shuffle();
        }

        // Report leftover (non-zero only if a push failed) so the agent keeps
        // being re-invoked to retry; workers drain their own queues meanwhile.
        self.bpf.notify_complete(self.pending.len() as u64);
        dispatched
    }

    fn print_stats(&mut self) {
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_bounce = *self.bpf.nr_bounce_dispatches_mut();
        let nr_cancel = *self.bpf.nr_cancel_dispatches_mut();
        let nr_failed = *self.bpf.nr_failed_dispatches_mut();
        let nr_migr = *self.bpf.nr_lb_migrations_mut();

        println!(
            "mech={:?} workers={} pending={} rr={} user_disp={} bounce={} cancel={} fail={} migrations={}",
            self.mechanism,
            self.workers.len(),
            self.pending.len(),
            self.rr,
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
            // Workers drain their own per-CPU queues independently. (For shuffle,
            // dispatch_tasks still runs the periodic rebalance even when idle.)
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

WARNING: scx_rr is a simple educational scheduler that makes all of its
scheduling decisions in user-space, based on scx_rustland_core. It is not
intended for use in production environments.

**************************************************************************"#;
    println!("{}", warning);
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    print_warning();

    let workers = resolve_workers(&opts)?;
    let mech = match opts.mechanism {
        Mechanism::None => "none (plain RR, run-to-completion)".to_string(),
        Mechanism::Preempt => format!("preempt ({} ms slice, requeue local)", opts.preempt_slice_ms),
        Mechanism::P2c => "p2c (power-of-two random choices)".to_string(),
        Mechanism::Shuffle => format!(
            "shuffle (every {} ms, min_qlen>{}, frac>{})",
            opts.shuffle_interval_ms, opts.shuffle_min_qlen, opts.shuffle_frac
        ),
        Mechanism::PreemptShuffle => format!(
            "preempt-shuffle ({} ms slice; shuffle every {} ms, min_qlen>{}, frac>{})",
            opts.preempt_slice_ms,
            opts.shuffle_interval_ms,
            opts.shuffle_min_qlen,
            opts.shuffle_frac
        ),
    };
    println!(
        "scx_rr: {} per-core FIFO queues, mechanism = {}, mode = {}, watchdog = {} ms",
        workers.len(),
        mech,
        if opts.system_wide {
            "system-wide"
        } else {
            "partial (only SCHED_EXT tasks)"
        },
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
