// Copyright (c) Yuxuan Zhao
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

//! # cFIFO: centralized global-queue FIFO scheduler (user-space)
//!
//! ## Overview
//!
//! `scx_cfifo` is a FIFO scheduler for the Linux kernel that makes all of its
//! decisions in user-space, in 100% Rust, on top of `scx_rustland_core` (which
//! leverages the kernel's `sched_ext` feature).
//!
//! It models the classic "centralized" design (e.g. the ghOSt centralized FIFO
//! scheduler):
//!
//!   * There is a single **global queue**, implemented as one shared
//!     `sched_ext` dispatch queue (DSQ) in the kernel. Every runnable task is
//!     placed there, in arrival order.
//!   * The scheduling **agent** (this user-space process) owns that queue: it
//!     drains newly-runnable tasks from the kernel and pushes them into the
//!     shared DSQ in FIFO order. It does not pick a CPU for any task.
//!   * Every **worker** CPU pulls the head of the shared DSQ for itself the
//!     moment it goes idle (done by the `scx_rustland_core` backend in
//!     `ops.dispatch`). So a task is run by the *first* core to become free.
//!
//! ## Scheduling policy: one FIFO queue, workers self-serve
//!
//! On every round the agent:
//!   1. drains all newly-runnable tasks from the kernel into a temporary
//!      user-space buffer, preserving arrival (FIFO) order, then
//!   2. pushes each one into the shared DSQ with a monotonically increasing
//!      vtime (so the DSQ stays in strict FIFO order) and an infinite time
//!      slice (run-to-completion: a task is never preempted once it starts).
//!
//! Because tasks are not bound to any particular CPU, the backlog lives in this
//! single kernel queue (observable via `scx_bpf_dsq_nr_queued`), no task can
//! wait behind a long run-to-completion task on one core while another core is
//! free, and there is no risk of the agent starving the workers: workers pull
//! from the shared DSQ on their own, the agent only feeds it.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[rustfmt::skip]
mod bpf;

use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::time::Duration;
use std::time::SystemTime;

use anyhow::Result;
use bpf::*;
use clap::Parser;
use libbpf_rs::OpenObject;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::UserExitInfo;

/// When tasks are waiting but no core is free, back off this long before the
/// next round to avoid busy-spinning the agent CPU (bounds dispatch latency).
const IDLE_BACKOFF: Duration = Duration::from_micros(100);

#[derive(Parser, Debug)]
#[command(name = "scx_cfifo", about = "cFIFO: centralized global-queue FIFO scheduler")]
struct Opts {
    /// Schedule the whole system. By default scx_cfifo runs in *partial* mode:
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

struct Scheduler<'a> {
    bpf: BpfScheduler<'a>,
    /// Temporary user-space buffer: newly-drained tasks are held here only for
    /// the moment between draining them from the kernel (in FIFO order) and
    /// pushing them into the shared DSQ. It is emptied every round.
    global_queue: VecDeque<QueuedTask>,
    /// Monotonically increasing vtime stamped on each task pushed to the shared
    /// DSQ, so the DSQ preserves strict global FIFO order.
    seq: u64,
    verbose: bool,
}

// The agent deliberately performs no file I/O: it could block on an ext4 inode
// lock held by a task it must schedule -- a priority inversion that deadlocks
// the scheduler. Per-task lifecycle timestamps (TaskNew/FirstRun/TaskDead) are
// recorded by the task processes themselves into shared memory (see
// project/function_trace/metrics_shm.h).

impl<'a> Scheduler<'a> {
    fn init(open_object: &'a mut MaybeUninit<OpenObject>, opts: &Opts) -> Result<Self> {
        let open_opts = LibbpfOpts::default();
        let bpf = BpfScheduler::init_with_timeout(
            open_object,
            open_opts.clone().into_bpf_open_opts(),
            0,        // exit_dump_len (0 = default)
            // partial: true (default) => only tasks that opt into SCHED_EXT are
            // scheduled by scx_cfifo; the rest of the system stays on the native
            // kernel scheduler. This keeps system background tasks off the
            // global queue so they can't be delayed by it (or trip the
            // sched_ext watchdog).
            !opts.system_wide,
            false,    // debug
            false,    // builtin_idle: false => every task is funneled through
                      // the user-space global queue (true centralized FIFO);
                      // true would let the kernel fast-path wakeups to prev_cpu.
            false,    // numa_local
            5_000_000, // BPF-layer default slice (5 ms) — controls the agent's own
                      // scheduling heartbeat. Worker tasks get u64::MAX at dispatch.
            opts.timeout_ms,
            "cfifo",  // scx ops name
        )?;

        Ok(Self {
            bpf,
            global_queue: VecDeque::new(),
            seq: 0,
            verbose: opts.verbose,
        })
    }

    /// Drain every task the kernel has made runnable into the tail of the
    /// global queue, preserving FIFO order.
    fn drain_into_global_queue(&mut self) {
        while let Ok(Some(task)) = self.bpf.dequeue_task() {
            self.global_queue.push_back(task);
        }
    }

    /// One scheduling round. Drains newly-runnable tasks and pushes them, in
    /// strict FIFO order, into the single shared DSQ that is the global queue.
    /// Idle workers pull the head of that DSQ themselves (in the backend's
    /// ops.dispatch), so the agent never picks a CPU and the backlog lives in
    /// one kernel queue rather than scattered across per-CPU DSQs.
    /// Returns true if at least one task was pushed this round.
    fn dispatch_tasks(&mut self) -> bool {
        // 1. New tasks enter the FIFO buffer.
        self.drain_into_global_queue();

        let mut dispatched = false;

        // 2. Push every queued task into the shared DSQ in arrival order. Each
        //    gets a monotonically increasing vtime (the DSQ stays strict FIFO),
        //    an infinite slice (run-to-completion), and target RL_CPU_ANY so the
        //    first idle worker to look will run it. Nothing is bound to a
        //    specific CPU, so no task can wait behind a long task on one core
        //    while another core is free.
        while let Some(task) = self.global_queue.pop_front() {
            let mut d = DispatchedTask::new(&task);
            d.cpu = RL_CPU_ANY;
            d.vtime = self.seq;
            d.slice_ns = u64::MAX; // run-to-completion: no preemption

            if let Err(e) = self.bpf.dispatch_task(&d) {
                // Couldn't push (e.g. ring full): keep FIFO order, retry next round.
                self.global_queue.push_front(task);
                if self.verbose {
                    eprintln!("dispatch failed: {e}");
                }
                break;
            }
            self.seq += 1;
            dispatched = true;
        }

        // 3. Nothing is held in user space now (unless a push failed above); the
        //    real backlog sits in the shared DSQ. Report any leftover so the
        //    agent is re-invoked to retry; workers drain the DSQ on their own.
        self.bpf.notify_complete(self.global_queue.len() as u64);

        dispatched
    }

    fn print_stats(&mut self) {
        let nr_user = *self.bpf.nr_user_dispatches_mut();
        let nr_kernel = *self.bpf.nr_kernel_dispatches_mut();
        let nr_cancel = *self.bpf.nr_cancel_dispatches_mut();
        let nr_failed = *self.bpf.nr_failed_dispatches_mut();
        let nr_congested = *self.bpf.nr_sched_congested_mut();

        println!(
            "global_q={} user_disp={} kernel_disp={} cancel={} fail={} cong={}",
            self.global_queue.len(),
            nr_user,
            nr_kernel,
            nr_cancel,
            nr_failed,
            nr_congested,
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
            // Workers drain the shared DSQ on their own, so the agent only needs
            // to wake often enough to feed newly-arrived tasks into it.
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

WARNING: scx_cfifo is a simple educational scheduler that makes all of its
scheduling decisions in user-space, based on scx_rustland_core. It is not
intended for use in production environments.

**************************************************************************"#;
    println!("{}", warning);
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    print_warning();
    println!(
        "scx_cfifo: centralized shared-DSQ FIFO, run-to-completion; agent unpinned \
         (CFS-scheduled, no reserved core), workers self-serve from the shared DSQ; \
         mode = {}, watchdog = {} ms",
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
