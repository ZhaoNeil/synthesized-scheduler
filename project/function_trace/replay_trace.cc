// Replay a function-arrival trace as one process per task, for use with a
// sched_ext scheduler (e.g. scx_cfifo).
//
// Trace format (whitespace separated), one task per line:
//     <inter_arrival_seconds> <launcher args...>
// e.g.
//     0.00012   cpu_fib 34          (hetero trace)
//     0.00012   34                  (plain trace, kind defaults to cpu_fib)
//
// For every line we wait the inter-arrival time (relative to the previous
// task, using an absolute monotonic deadline so timing does not drift), then
// fork+exec the launcher (launch_function) with the line's arguments. Each
// spawned task is pinned to the given CPU set (default 0-49) so it is served
// by the workers of the centralized FIFO scheduler.
//
// Build:  g++ -O2 -o replay_trace replay_trace.cc
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "metrics_shm.h"

namespace {

volatile sig_atomic_t interrupted_signal = 0;

void HandleSignal(int signal) {
  interrupted_signal = signal;
}

struct Task {
  double arrival;            // inter-arrival time (seconds) from previous task
  std::vector<std::string> args;  // launcher arguments (kind/level, etc.)
  std::string descr;         // human readable description for error messages
};

struct Record {
  int index;
  std::string descr;
};

uint64_t NowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// True only while a sched_ext scheduler is attached. Used to detect mid-run
// eviction so we never keep launching tasks that would silently land on CFS.
bool ScxEnabled() {
  std::ifstream in("/sys/kernel/sched_ext/state");
  std::string s;
  if (in) std::getline(in, s);
  return s == "enabled";
}

bool SleepUntilNs(uint64_t deadline_ns) {
  timespec ts;
  ts.tv_sec = static_cast<time_t>(deadline_ns / 1000000000ULL);
  ts.tv_nsec = static_cast<long>(deadline_ns % 1000000000ULL);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
    if (interrupted_signal != 0) return false;
  }
  return interrupted_signal == 0;
}

// Parse a CPU list like "0-49" or "0-3,8,12-15" into a cpu_set_t.
bool ParseCpuList(const std::string& spec, cpu_set_t* set) {
  CPU_ZERO(set);
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    size_t dash = item.find('-');
    if (dash == std::string::npos) {
      int cpu = std::atoi(item.c_str());
      if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
      CPU_SET(cpu, set);
    } else {
      int lo = std::atoi(item.substr(0, dash).c_str());
      int hi = std::atoi(item.substr(dash + 1).c_str());
      if (lo < 0 || hi >= CPU_SETSIZE || lo > hi) return false;
      for (int c = lo; c <= hi; ++c) CPU_SET(c, set);
    }
  }
  return true;
}

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " <trace_file> [options]\n"
      << "  --launcher PATH   path to launch_function (default ./launch_function)\n"
      << "  --cpus LIST       CPU affinity for tasks (default 0-49)\n"
      << "  --time-scale F    multiply inter-arrival times by F (default 1.0)\n"
      << "  --max N           replay at most N tasks (default: all)\n"
      << "  --metrics         collect per-task timings in shared memory (no file)\n"
      << "                    and print aggregate metrics when done\n"
      << "  --queue-series F  with --metrics, write the queue backlog over time\n"
      << "  --core-backlog F  with --metrics, write per-core peak/mean backlog\n"
      << "  --per-task F      with --metrics, dump a TaskNew/FirstRun/TaskDead\n"
      << "                    lifecycle log (post-run, from shared memory)\n"
      << "  --occupancy F     with --metrics, dump per-core occupancy intervals\n"
      << "                    (core task start_s end_s dur_s), incl. mid-run core\n"
      << "                    changes; tasks sample their own CPU while running\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);

  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string trace_path = argv[1];
  std::string launcher = "./launch_function";
  std::string cpus = "0-49";
  std::string queue_series;
  std::string core_backlog;
  std::string per_task;
  std::string occupancy;
  bool metrics = false;
  double time_scale = 1.0;
  long max_tasks = -1;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--launcher" && i + 1 < argc) {
      launcher = argv[++i];
    } else if (a == "--cpus" && i + 1 < argc) {
      cpus = argv[++i];
    } else if (a == "--time-scale" && i + 1 < argc) {
      time_scale = std::atof(argv[++i]);
    } else if (a == "--max" && i + 1 < argc) {
      max_tasks = std::atol(argv[++i]);
    } else if (a == "--metrics") {
      metrics = true;
    } else if (a == "--queue-series" && i + 1 < argc) {
      queue_series = argv[++i];
    } else if (a == "--core-backlog" && i + 1 < argc) {
      core_backlog = argv[++i];
    } else if (a == "--per-task" && i + 1 < argc) {
      per_task = argv[++i];
    } else if (a == "--occupancy" && i + 1 < argc) {
      occupancy = argv[++i];
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }

  cpu_set_t cpuset;
  if (!ParseCpuList(cpus, &cpuset)) {
    std::cerr << "Invalid --cpus list: " << cpus << std::endl;
    return 1;
  }

  // Read the whole trace into memory.
  std::ifstream in(trace_path);
  if (!in) {
    std::cerr << "Cannot open trace: " << trace_path << std::endl;
    return 1;
  }
  std::vector<Task> tasks;
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ls(line);
    Task t;
    if (!(ls >> t.arrival)) continue;  // skip blank/comment lines
    std::string tok;
    std::ostringstream descr;
    while (ls >> tok) {
      t.args.push_back(tok);
      if (!descr.str().empty()) descr << " ";
      descr << tok;
    }
    if (t.args.empty()) continue;
    t.descr = descr.str();
    tasks.push_back(std::move(t));
    if (max_tasks > 0 && static_cast<long>(tasks.size()) >= max_tasks) break;
  }
  std::cerr << "Loaded " << tasks.size() << " tasks from " << trace_path
            << ", pinning to CPUs " << cpus << std::endl;

  // Optional per-task metrics in shared memory: one TaskMetric slot per task in
  // an anonymous in-memory file (memfd). Each task writes only its own slot, so
  // there is no shared lock and no contention -- and no log file. We (the
  // parent) read the whole array after every task has exited. The memfd is not
  // CLOEXEC, so it survives fork+exec and the children inherit the same fd.
  int metrics_fd = -1;
  TaskMetric* metrics_arr = nullptr;
  std::string metrics_fd_str, metrics_count_str;
  if (metrics && !tasks.empty()) {
    metrics_fd = memfd_create("scx_metrics", 0);
    if (metrics_fd < 0) {
      std::perror("replay_trace: memfd_create");
      return 1;
    }
    size_t bytes = tasks.size() * sizeof(TaskMetric);
    if (ftruncate(metrics_fd, static_cast<off_t>(bytes)) != 0) {
      std::perror("replay_trace: ftruncate");
      return 1;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                   metrics_fd, 0);
    if (p == MAP_FAILED) {
      std::perror("replay_trace: mmap");
      return 1;
    }
    metrics_arr = static_cast<TaskMetric*>(p);  // memfd is zero-filled
    metrics_fd_str = std::to_string(metrics_fd);
    metrics_count_str = std::to_string(tasks.size());
  }

  // Optional per-task CPU-occupancy ring: a second memfd holding seg_max CpuSeg
  // entries per task. Each launch_function writes only its own region (lock-free,
  // no I/O); we read it after every task exits. Sized seg_max core-changes/task
  // (OCCUPANCY_MAX_SEG); pages are faulted lazily so short tasks cost ~one page.
  int seg_fd = -1;
  CpuSeg* seg_arr = nullptr;
  long seg_max = 256;
  std::string seg_fd_str, seg_max_str;
  if (!occupancy.empty() && metrics && !tasks.empty()) {
    const char* e = std::getenv("OCCUPANCY_MAX_SEG");
    if (e != nullptr && *e != '\0') {
      long v = std::atol(e);
      if (v > 0) seg_max = v;
    }
    seg_fd = memfd_create("scx_occ", 0);
    if (seg_fd < 0) {
      std::perror("replay_trace: memfd_create(occ)");
      return 1;
    }
    size_t bytes = tasks.size() * static_cast<size_t>(seg_max) * sizeof(CpuSeg);
    if (ftruncate(seg_fd, static_cast<off_t>(bytes)) != 0) {
      std::perror("replay_trace: ftruncate(occ)");
      return 1;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, seg_fd, 0);
    if (p == MAP_FAILED) {
      std::perror("replay_trace: mmap(occ)");
      return 1;
    }
    seg_arr = static_cast<CpuSeg*>(p);  // memfd is zero-filled
    seg_fd_str = std::to_string(seg_fd);
    seg_max_str = std::to_string(seg_max);
  }

  // Pre-build the argv for execv: launcher --skip-enclave --quiet <args...>
  // We rebuild per task because args differ.
  std::unordered_map<pid_t, Record> live;
  size_t completed = 0;

  uint64_t start_ns = NowNs();
  uint64_t deadline_ns = start_ns;  // absolute deadline accumulator

  auto kill_all_live = [&]() {
    for (const auto& kv : live) kill(kv.first, SIGKILL);
  };

  auto abort_if_interrupted = [&]() {
    if (interrupted_signal == 0) return;
    std::cerr << "replay_trace: interrupted; killing " << live.size()
              << " unfinished task(s)..." << std::endl;
    kill_all_live();
    std::exit(128 + interrupted_signal);
  };

  auto reap = [&](bool block) {
    int status;
    int flags = block ? 0 : WNOHANG;
    pid_t pid;
    while ((pid = waitpid(-1, &status, flags)) > 0) {
      auto it = live.find(pid);
      if (it == live.end()) continue;

      // A task must complete successfully under sched_ext. Any non-zero exit
      // (e.g. launch_function aborting because it could not enter SCHED_EXT, or
      // a failed CPU pinning/exec) aborts the whole run loudly, so a
      // misconfiguration can never silently pollute the metrics.
      bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
      if (!ok) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
        std::cerr << "replay_trace: task " << it->second.index << " ("
                  << it->second.descr << ") exited abnormally (code " << code
                  << "); aborting. This usually means the task could not enter "
                     "SCHED_EXT -- is a sched_ext scheduler loaded in partial mode and are "
                     "we root?" << std::endl;
        live.erase(it);
        kill_all_live();
        std::exit(1);
      }

      ++completed;
      live.erase(it);
      if (!block) flags = WNOHANG;  // keep draining non-blocking
    }
  };

  for (size_t i = 0; i < tasks.size(); ++i) {
    const Task& t = tasks[i];
    deadline_ns += static_cast<uint64_t>(t.arrival * time_scale * 1e9);
    if (!SleepUntilNs(deadline_ns)) abort_if_interrupted();

    // Stop immediately if the scheduler was evicted/unloaded mid-run, rather
    // than launching tasks that would silently fall back to CFS.
    if (!ScxEnabled()) {
      std::cerr << "replay_trace: sched_ext is no longer enabled (scheduler "
                   "evicted?); aborting at task " << i << std::endl;
      kill_all_live();
      std::exit(1);
    }

    // Build argv: launcher --quiet
    //             [--metrics-fd FD --metrics-index I --metrics-count N] <args>
    std::string metrics_index_str;  // must outlive the fork() below
    std::vector<char*> cargv;
    cargv.push_back(const_cast<char*>(launcher.c_str()));
    cargv.push_back(const_cast<char*>("--quiet"));
    if (metrics_arr) {
      metrics_index_str = std::to_string(i);
      cargv.push_back(const_cast<char*>("--metrics-fd"));
      cargv.push_back(const_cast<char*>(metrics_fd_str.c_str()));
      cargv.push_back(const_cast<char*>("--metrics-index"));
      cargv.push_back(const_cast<char*>(metrics_index_str.c_str()));
      cargv.push_back(const_cast<char*>("--metrics-count"));
      cargv.push_back(const_cast<char*>(metrics_count_str.c_str()));
    }
    if (seg_arr) {
      // The launcher reuses --metrics-index to locate its ring region.
      cargv.push_back(const_cast<char*>("--seg-fd"));
      cargv.push_back(const_cast<char*>(seg_fd_str.c_str()));
      cargv.push_back(const_cast<char*>("--seg-max"));
      cargv.push_back(const_cast<char*>(seg_max_str.c_str()));
    }
    for (const auto& a : t.args) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
      // Child: pin to the worker CPU set, then exec the launcher. Both are
      // mandatory -- abort loudly rather than run on the wrong CPUs or skip the
      // exec, so a misconfiguration can never silently pollute the run.
      if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        std::perror("replay_trace: sched_setaffinity");
        _exit(126);
      }
      execv(launcher.c_str(), cargv.data());
      std::perror("replay_trace: execv");
      _exit(127);  // exec failed
    } else if (pid > 0) {
      Record r{static_cast<int>(i), t.descr};
      live[pid] = r;
    } else {
      std::cerr << "fork failed at task " << i << std::endl;
    }

    // Opportunistically reap completed children without blocking.
    reap(false);
    abort_if_interrupted();
  }

  std::cerr << "All " << tasks.size()
            << " tasks submitted; waiting for completion..." << std::endl;
  // Wait for the rest.
  while (!live.empty()) {
    reap(true);
    abort_if_interrupted();
  }

  std::cerr << "Tasks completed: " << completed << std::endl;

  // Every task has exited, so its shared-memory slot is fully written and
  // visible to us. Compute the aggregate metrics (and the queue backlog over
  // time) from the array -- no log file was ever written during the run.
  if (metrics_arr) {
    // Optional per-task lifecycle log, dumped from the shared-memory array now
    // that the run is over -- safe, because no task process writes a shared log
    // file during the run (a shared log's ext4 inode lock, held across a
    // deschedule under run-to-completion, could stall the scheduler). Same
    // format as results/scx_cfifo/<workload>:
    //   TaskNew:  <label> enqueue at <ISO8601>
    //   FirstRun: <label> starts at  <ISO8601>
    //   TaskDead: <label> at         <ISO8601>
    // The TaskMetric timestamps are CLOCK_MONOTONIC ns; we map them to wall-clock
    // with a single CLOCK_REALTIME/CLOCK_MONOTONIC offset captured here, so the
    // absolute times look right and all per-task differences are exact.
    if (!per_task.empty()) {
      struct timespec rt, mo;
      clock_gettime(CLOCK_REALTIME, &rt);
      clock_gettime(CLOCK_MONOTONIC, &mo);
      int64_t offset_ns =
          (int64_t)rt.tv_sec * 1000000000LL + rt.tv_nsec -
          ((int64_t)mo.tv_sec * 1000000000LL + mo.tv_nsec);

      auto iso = [offset_ns](uint64_t mono_ns) {
        int64_t real_ns = (int64_t)mono_ns + offset_ns;
        time_t secs = (time_t)(real_ns / 1000000000LL);
        unsigned long long frac = (unsigned long long)(real_ns % 1000000000LL);
        struct tm tmv;
        gmtime_r(&secs, &tmv);
        char date[32];
        strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S", &tmv);
        char out[64];
        std::snprintf(out, sizeof out, "%s.%09llu+00:00", date, frac);
        return std::string(out);
      };

      std::ofstream pt(per_task);
      if (pt) {
        for (size_t i = 0; i < tasks.size(); ++i) {
          const TaskMetric& m = metrics_arr[i];
          if (m.tasknew_ns == 0 || m.firstrun_ns == 0 || m.taskdead_ns == 0)
            continue;
          if (!(m.tasknew_ns <= m.firstrun_ns && m.firstrun_ns <= m.taskdead_ns))
            continue;
          std::string label = "C" + std::to_string(i);
          pt << "TaskNew: " << label << " enqueue at " << iso(m.tasknew_ns)
             << '\n'
             << "FirstRun: " << label << " starts at " << iso(m.firstrun_ns)
             << '\n'
             << "TaskDead: " << label << " at " << iso(m.taskdead_ns) << '\n';
        }
      }
    }

    unsigned long long sum_lat_ns = 0, sum_rt_ns = 0;
    uint64_t min_tn = UINT64_MAX, max_td = 0;
    size_t valid = 0;
    // Change-point events for the backlog: a task is waiting for a CPU during
    // [tasknew, firstrun).
    std::vector<std::pair<uint64_t, int>> events;
    events.reserve(tasks.size() * 2);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const TaskMetric& m = metrics_arr[i];
      if (m.tasknew_ns == 0 || m.firstrun_ns == 0 || m.taskdead_ns == 0)
        continue;  // task aborted/killed before finishing its lifecycle
      if (!(m.tasknew_ns <= m.firstrun_ns && m.firstrun_ns <= m.taskdead_ns))
        continue;  // clock anomaly; skip
      sum_lat_ns += m.firstrun_ns - m.tasknew_ns;
      sum_rt_ns += m.taskdead_ns - m.firstrun_ns;
      if (m.tasknew_ns < min_tn) min_tn = m.tasknew_ns;
      if (m.taskdead_ns > max_td) max_td = m.taskdead_ns;
      events.emplace_back(m.tasknew_ns, +1);
      events.emplace_back(m.firstrun_ns, -1);
      ++valid;
    }

    long peak = 0;
    if (!events.empty()) {
      std::sort(events.begin(), events.end());
      std::ofstream qs;
      if (!queue_series.empty()) {
        qs.open(queue_series);
        if (qs) qs << "# seconds_since_start  waiting_for_cpu\n";
      }
      long waiting = 0;
      size_t k = 0;
      while (k < events.size()) {
        uint64_t ts = events[k].first;
        while (k < events.size() && events[k].first == ts) {
          waiting += events[k].second;
          ++k;
        }
        if (waiting > peak) peak = waiting;
        if (qs.is_open())
          qs << static_cast<double>(ts - min_tn) / 1e9 << ' ' << waiting << '\n';
      }
    }

    // Per-core backlog (e.g. RR): a task assigned to core c -- the worker CPU
    // it first ran on -- waits in that core's queue during [tasknew, firstrun).
    // Bucket the same change-point events by core to get each core's peak and
    // time-averaged queue depth, written as a one-row-per-core summary.
    if (!core_backlog.empty() && valid) {
      std::map<int, std::vector<std::pair<uint64_t, int>>> cev;
      std::map<int, long> ccount;
      std::map<int, unsigned long long> cwait_ns;
      for (size_t i = 0; i < tasks.size(); ++i) {
        const TaskMetric& m = metrics_arr[i];
        if (m.tasknew_ns == 0 || m.firstrun_ns == 0 || m.taskdead_ns == 0)
          continue;
        if (!(m.tasknew_ns <= m.firstrun_ns && m.firstrun_ns <= m.taskdead_ns))
          continue;
        if (m.cpu == UINT64_MAX)
          continue;  // task never recorded a CPU
        int c = static_cast<int>(m.cpu);
        cev[c].emplace_back(m.tasknew_ns, +1);
        cev[c].emplace_back(m.firstrun_ns, -1);
        ccount[c] += 1;
        cwait_ns[c] += m.firstrun_ns - m.tasknew_ns;
      }
      double makespan_s = (max_td > min_tn) ? (max_td - min_tn) / 1e9 : 0.0;
      std::ofstream cs(core_backlog);
      if (cs) cs << "# core  tasks  peak_backlog  mean_backlog\n";
      long worst_peak = 0;
      int worst_cpu = -1;
      for (auto& kv : cev) {
        int c = kv.first;
        std::vector<std::pair<uint64_t, int>>& ev = kv.second;
        std::sort(ev.begin(), ev.end());
        long w = 0, pk = 0;
        size_t k = 0;
        while (k < ev.size()) {
          uint64_t ts = ev[k].first;
          while (k < ev.size() && ev[k].first == ts) {
            w += ev[k].second;
            ++k;
          }
          if (w > pk) pk = w;
        }
        // Time-averaged queue depth = integral of backlog / makespan = total
        // wait on this core / makespan.
        double mean = makespan_s > 0.0 ? (cwait_ns[c] / 1e9) / makespan_s : 0.0;
        if (cs) cs << c << ' ' << ccount[c] << ' ' << pk << ' ' << mean << '\n';
        if (pk > worst_peak) {
          worst_peak = pk;
          worst_cpu = c;
        }
      }
      std::printf("peak per-core backlog: %ld tasks (on core %d)\n", worst_peak,
                  worst_cpu);
    }

    // Per-core occupancy: each task's sequence of CpuSeg's (core it ran on +
    // when), recorded by launch_function on every mid-run core change. Seg k
    // means task i ran on segs[k].cpu from segs[k].ts_ns until the next seg's
    // ts_ns (the last seg runs until taskdead_ns). Tasks that never changed core
    // (run-to-completion) have one seg = one exact interval; if a task recorded
    // none (e.g. occupancy off), fall back to its first-run cpu for [firstrun,
    // taskdead]. Written sorted by (core, start) as: core task start_s end_s dur_s
    // with times relative to the first TaskNew, matching the queue-series origin.
    if (!occupancy.empty() && seg_arr && valid) {
      struct OccRow {
        int cpu;
        size_t task;
        double start_s;
        double end_s;
      };
      std::vector<OccRow> rows;
      long max_seg = 0;
      size_t truncated = 0;
      for (size_t i = 0; i < tasks.size(); ++i) {
        const TaskMetric& m = metrics_arr[i];
        if (m.firstrun_ns == 0 || m.taskdead_ns == 0) continue;
        if (!(m.firstrun_ns <= m.taskdead_ns)) continue;
        long n = static_cast<long>(m.reserved[0]);  // segs this task recorded
        const CpuSeg* segs = &seg_arr[i * seg_max];
        if (n <= 0) {
          if (m.cpu != UINT64_MAX)
            rows.push_back({static_cast<int>(m.cpu), i,
                            static_cast<double>(m.firstrun_ns - min_tn) / 1e9,
                            static_cast<double>(m.taskdead_ns - min_tn) / 1e9});
          continue;
        }
        if (n > seg_max) n = seg_max;  // defensive
        if (n > max_seg) max_seg = n;
        for (long k = 0; k < n; ++k) {
          uint64_t start = segs[k].ts_ns;
          uint64_t end = (k + 1 < n) ? segs[k + 1].ts_ns : m.taskdead_ns;
          if (end < start) end = start;  // clamp clock anomaly
          if (segs[k].flags & 1u) ++truncated;
          rows.push_back({static_cast<int>(segs[k].cpu), i,
                          static_cast<double>(start - min_tn) / 1e9,
                          static_cast<double>(end - min_tn) / 1e9});
        }
      }
      std::sort(rows.begin(), rows.end(),
                [](const OccRow& a, const OccRow& b) {
                  if (a.cpu != b.cpu) return a.cpu < b.cpu;
                  return a.start_s < b.start_s;
                });
      std::ofstream os(occupancy);
      if (os) {
        os << "# core  task  start_s  end_s  dur_s   "
              "(start/end relative to first TaskNew)\n";
        for (const OccRow& r : rows)
          os << r.cpu << " C" << r.task << ' ' << r.start_s << ' ' << r.end_s
             << ' ' << (r.end_s - r.start_s) << '\n';
      }
      std::printf("occupancy: %zu intervals, max %ld seg/task%s -> %s\n",
                  rows.size(), max_seg,
                  truncated ? " (some tasks hit the ring cap)" : "",
                  occupancy.c_str());
    }

    uint64_t makespan_ns = valid ? (max_td - min_tn) : 0;
    // Printed to stdout (diagnostics go to stderr), matching the field names
    // analyze_task_log.py uses so the rest of the pipeline is unchanged.
    std::printf("tasks: %zu\n", valid);
    std::printf("accumulated task latency: %.9f s\n", sum_lat_ns / 1e9);
    std::printf("accumulated task runtime: %.9f s\n", sum_rt_ns / 1e9);
    std::printf("makespan: %.9f s\n", makespan_ns / 1e9);
    std::printf("peak queue backlog: %ld tasks waiting for a CPU\n", peak);
  }
  return 0;
}
