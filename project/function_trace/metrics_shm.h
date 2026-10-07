// Shared-memory per-task metric record, used by replay_trace (creator/reader)
// and launch_function (one writer per task).
//
// replay_trace creates one TaskMetric slot per task in an anonymous, in-memory
// file (memfd) and passes the fd + each task's index to launch_function. Every
// task writes ONLY its own slot (three plain stores, no lock, no syscall), so
// there is zero cross-task contention -- unlike a shared log file, whose ext4
// inode lock, held across a deschedule under run-to-completion, could deadlock
// the scheduler. After all tasks finish, replay_trace reads the whole array and
// computes the aggregate metrics (and the queue-backlog time series).
//
// Timestamps are CLOCK_MONOTONIC nanoseconds: monotonic and comparable across
// processes on the same machine, with no wall-clock skew.
#ifndef SCX_RL_METRICS_SHM_H
#define SCX_RL_METRICS_SHM_H

#include <cstdint>

struct TaskMetric {
  uint64_t tasknew_ns;   // about to enter the run queue (before the nanosleep)
  uint64_t firstrun_ns;  // first running on a worker CPU (after the nanosleep)
  uint64_t taskdead_ns;  // work done, about to exit
  uint64_t cpu;          // worker CPU it first ran on (its per-core queue under RR)
  // reserved[0]: number of CpuSeg occupancy entries this task recorded (0 when
  //              occupancy sampling is disabled -- see CpuSeg below).
  // reserved[1..3]: unused padding to one 64-byte cache line (avoid false sharing).
  uint64_t reserved[4];
};

// One per-task CPU-occupancy sample: the worker CPU this task was observed on at
// `ts_ns`, recorded by launch_function every time it notices its core changed
// while running (migration). replay_trace allocates a fixed-size ring of these
// per task in a second shared memfd; each task writes ONLY its own region (same
// lock-free, no-I/O pattern as the TaskMetric slot above). The sequence of segs
// for a task is its occupancy timeline: seg k means "ran on cpu from segs[k].ts_ns
// until segs[k+1].ts_ns" (the last seg runs until taskdead_ns).
//
// NOTE: a userspace task can only observe its CPU while it is running, so these
// capture core CHANGES (migrations), not preemption gaps where the task is
// descheduled and later resumes on the SAME core (that still reads as one seg).
struct CpuSeg {
  uint64_t ts_ns;  // CLOCK_MONOTONIC ns when the task was first seen on `cpu`
  uint32_t cpu;    // worker CPU id at that observation
  uint32_t flags;  // bit0: ring was full -- later core changes were dropped
};

#endif  // SCX_RL_METRICS_SHM_H
