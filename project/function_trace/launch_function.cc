#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "metrics_shm.h"

namespace {

constexpr const char* kDefaultCalibrationPath =
    "project/function_trace/hetero_calibration.tsv";
constexpr const char* kDefaultReadPath = "project/function_trace/trace.txt";
constexpr size_t kMemUnitBytes = 1 * 1024 * 1024;
constexpr size_t kMaxMemBytes = 64 * 1024 * 1024;
constexpr size_t kReadBytes = 64 * 1024;

volatile uint64_t sink = 0;

// CLOCK_MONOTONIC nanoseconds: monotonic, comparable across processes, used for
// the shared-memory per-task metric timestamps.
uint64_t MonotonicNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// --- Per-task CPU-occupancy sampling -----------------------------------------
// When replay_trace provides an occupancy ring (--seg-fd/--seg-max), we record a
// CpuSeg every time this task notices its core changed while running. We do NOT
// poll inside the workload (that would perturb the calibrated fib cost); instead
// a per-thread CPU-time timer (CLOCK_THREAD_CPUTIME_ID) fires SIGRTMIN every
// OCCUPANCY_PERIOD_US of on-CPU time, and the handler appends a seg on a core
// change. Because the clock only advances while running, short tasks fire few/no
// signals and the overhead scales with on-CPU time (~one cheap getcpu per tick).
CpuSeg* g_occ_segs = nullptr;        // this task's region of the shared ring
long g_occ_max = 0;                  // ring capacity (segs) for this task
volatile sig_atomic_t g_occ_on = 0;  // 1 while sampling is armed
volatile long g_occ_count = 0;       // segs recorded so far (handler + init write)
volatile int g_occ_last_cpu = -1;    // last core we saw, to detect changes
timer_t g_occ_timer;
bool g_occ_have_timer = false;

// SIGRTMIN handler: append a seg if our core changed. async-signal-safe (getcpu,
// clock_gettime, plain stores only).
void OccSampleHandler(int /*signo*/) {
  if (!g_occ_on || g_occ_segs == nullptr) return;
  int cpu = sched_getcpu();
  if (cpu < 0 || cpu == g_occ_last_cpu) return;
  long n = g_occ_count;
  if (n < g_occ_max) {
    g_occ_segs[n].ts_ns = MonotonicNs();
    g_occ_segs[n].cpu = static_cast<uint32_t>(cpu);
    g_occ_segs[n].flags = 0;
    g_occ_count = n + 1;
  } else if (g_occ_max > 0) {
    g_occ_segs[g_occ_max - 1].flags |= 1u;  // ring full -> mark dropped changes
  }
  g_occ_last_cpu = cpu;
}

// Seed seg[0] with the FirstRun core/time, then start the CPU-time timer.
void ArmOccupancyTimer(uint64_t firstrun_ns, int first_cpu) {
  if (g_occ_segs == nullptr || g_occ_max <= 0) return;
  g_occ_segs[0].ts_ns = firstrun_ns;
  g_occ_segs[0].cpu = (first_cpu >= 0) ? static_cast<uint32_t>(first_cpu) : 0;
  g_occ_segs[0].flags = 0;
  g_occ_count = 1;
  g_occ_last_cpu = first_cpu;

  struct sigaction sa;
  std::memset(&sa, 0, sizeof sa);
  sa.sa_handler = OccSampleHandler;
  sa.sa_flags = SA_RESTART;  // let the task's nanosleeps resume transparently
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGRTMIN, &sa, nullptr) != 0) return;

  struct sigevent sev;
  std::memset(&sev, 0, sizeof sev);
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGRTMIN;
  if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &g_occ_timer) != 0) return;
  g_occ_have_timer = true;

  long period_us = 1000;  // default 1 ms of on-CPU time between samples
  const char* e = std::getenv("OCCUPANCY_PERIOD_US");
  if (e != nullptr && *e != '\0') {
    long v = std::atol(e);
    if (v > 0) period_us = v;
  }
  struct itimerspec its;
  its.it_value.tv_sec = period_us / 1000000;
  its.it_value.tv_nsec = (period_us % 1000000) * 1000;
  its.it_interval = its.it_value;
  g_occ_on = 1;
  timer_settime(g_occ_timer, 0, &its, nullptr);
}

void DisarmOccupancyTimer() {
  g_occ_on = 0;  // stop the handler from writing before we read the count
  if (!g_occ_have_timer) return;
  struct itimerspec z;
  std::memset(&z, 0, sizeof z);
  timer_settime(g_occ_timer, 0, &z, nullptr);
  timer_delete(g_occ_timer);
  g_occ_have_timer = false;
}

struct CalibrationEntry {
  bool found = false;
  uint64_t arg1 = 0;
  uint64_t arg2 = 0;
};

unsigned long long Fibonacci(int n) {
  if (n <= 1) {
    return 1;
  }
  return Fibonacci(n - 1) + Fibonacci(n - 2);
}

bool ParseUint64(const char* text, uint64_t* value) {
  if (text == nullptr || *text == '\0') return false;
  if (text[0] == '-') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool ParseInt(const char* text, int* value) {
  uint64_t parsed = 0;
  if (!ParseUint64(text, &parsed)) return false;
  if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

#ifndef SCHED_EXT
#define SCHED_EXT 7
#endif

constexpr const char* kScxStatePath = "/sys/kernel/sched_ext/state";
constexpr const char* kScxOpsPath = "/sys/kernel/sched_ext/root/ops";
// Optional env var naming a substring the active scheduler's ops name must
// contain (e.g. "cfs", "cfifo", "rr"). The run harness sets it per scheduler.
// If unset/empty, any enabled sched_ext scheduler is accepted.
constexpr const char* kExpectOpsEnv = "SCX_EXPECT_OPS";

std::string ReadFileTrim(const char* path) {
  std::ifstream in(path);
  if (!in) return "";
  std::string s;
  std::getline(in, s);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
    s.pop_back();
  return s;
}

// Verify that the expected sched_ext scheduler is *active*. This is essential:
// sched_setscheduler(SCHED_EXT) succeeds even when no BPF scheduler is loaded
// (the task then just runs on the kernel fallback), so entering SCHED_EXT is
// not by itself proof of being scheduled by our scheduler. The "enabled" state
// already proves a BPF scheduler is attached; if $SCX_EXPECT_OPS is set we
// additionally require the active scheduler's ops name to contain it (so the
// harness can pin the check to a specific scheduler). Returns an empty string
// on success, or a human-readable reason on failure.
std::string VerifyScxActive() {
  std::string state = ReadFileTrim(kScxStatePath);
  if (state != "enabled") {
    return "sched_ext state is '" + state +
           "' (expected 'enabled'); no sched_ext scheduler is loaded";
  }
  const char* expect = std::getenv(kExpectOpsEnv);
  if (expect != nullptr && *expect != '\0') {
    std::string ops = ReadFileTrim(kScxOpsPath);
    if (ops.find(expect) == std::string::npos) {
      return "active sched_ext scheduler is '" + ops + "', expected one matching '" +
             expect + "'";
    }
  }
  return "";
}

// Move this process into the sched_ext scheduling class so that the loaded
// scheduler (in partial mode) takes it over. The sched_ext equivalent of attaching to a
// ghOSt enclave. Returns true on success; on failure errno is set. We never
// fall back to the native scheduler.
bool UseSchedExt() {
  struct sched_param param;
  param.sched_priority = 0;
  return sched_setscheduler(0, SCHED_EXT, &param) == 0;
}

CalibrationEntry LookupCalibration(const std::string& kind, int level,
                                   const std::string& path) {
  CalibrationEntry entry;
  std::ifstream in(path);
  if (!in) return entry;

  std::string line_kind;
  int line_level = 0;
  uint64_t arg1 = 0;
  uint64_t arg2 = 0;
  while (in >> line_kind) {
    if (!line_kind.empty() && line_kind[0] == '#') {
      std::string rest;
      std::getline(in, rest);
      continue;
    }
    if (!(in >> line_level >> arg1 >> arg2)) {
      break;
    }
    std::string rest;
    std::getline(in, rest);
    if (line_kind == kind && line_level == level) {
      entry.found = true;
      entry.arg1 = arg1;
      entry.arg2 = arg2;
      return entry;
    }
  }
  return entry;
}

uint64_t RunMemStream(uint64_t units) {
  units = std::max<uint64_t>(1, units);
  uint64_t total_bytes = 0;
  if (units > std::numeric_limits<uint64_t>::max() / kMemUnitBytes) {
    total_bytes = std::numeric_limits<uint64_t>::max();
  } else {
    total_bytes = units * kMemUnitBytes;
  }

  const size_t buffer_bytes = static_cast<size_t>(
      std::min<uint64_t>(std::max<uint64_t>(kMemUnitBytes, total_bytes),
                         kMaxMemBytes));
  const size_t elements = buffer_bytes / sizeof(uint64_t);
  std::vector<uint64_t> src(elements);
  std::vector<uint64_t> dst(elements);

  for (size_t i = 0; i < elements; ++i) {
    src[i] = i * 1315423911ULL + 0x9e3779b97f4a7c15ULL;
  }

  uint64_t checksum = 0;
  uint64_t full_passes = total_bytes / buffer_bytes;
  size_t remainder_elements =
      (total_bytes % buffer_bytes) / sizeof(uint64_t);

  auto stream_elements = [&](size_t count, uint64_t pass) {
    for (size_t i = 0; i < count; ++i) {
      uint64_t value = src[i] + pass;
      dst[i] = value;
      checksum += value;
    }
    std::swap(src, dst);
  };

  for (uint64_t pass = 0; pass < full_passes; ++pass) {
    stream_elements(elements, pass);
  }
  if (remainder_elements > 0) {
    stream_elements(remainder_elements, full_passes);
  }

  sink ^= checksum;
  return checksum;
}

uint64_t RunIoSleepRead(uint64_t sleep_ns, uint64_t read_iters) {
  if (sleep_ns > 0) {
    timespec req;
    req.tv_sec = static_cast<time_t>(sleep_ns / 1000000000ULL);
    req.tv_nsec = static_cast<long>(sleep_ns % 1000000000ULL);
    while (nanosleep(&req, &req) < 0 && errno == EINTR) {
    }
  }

  read_iters = std::max<uint64_t>(1, read_iters);
  std::vector<char> buffer(kReadBytes);
  uint64_t checksum = 0;

  for (uint64_t iter = 0; iter < read_iters; ++iter) {
    int fd = open(kDefaultReadPath, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    }
    if (fd < 0) {
      break;
    }

    ssize_t n = read(fd, buffer.data(), buffer.size());
    close(fd);
    if (n <= 0) {
      continue;
    }
    for (ssize_t i = 0; i < n; ++i) {
      checksum = checksum * 131 + static_cast<unsigned char>(buffer[i]);
    }
  }

  sink ^= checksum;
  return checksum;
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " [opts] <level>\n"
      << "  " << program
      << " [opts] cpu_fib|mem_stream|io_sleep_read <level>\n"
      << "  " << program << " [opts] mem_stream_raw <units>\n"
      << "  " << program
      << " [opts] io_sleep_read_raw <sleep_ns> <read_iters>\n"
      << "opts: [--quiet] [--calibration PATH] [--label NAME]\n"
      << "      [--metrics-fd FD --metrics-index I --metrics-count N]\n"
      << "      [--seg-fd FD --seg-max N]  (per-task CPU-occupancy ring)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  // This launcher runs a single workload function and exits. Under a
  // sched_ext scheduler (e.g. scx_cfifo) the process is scheduled
  // automatically once the scheduler is loaded; no enclave/attach step is
  // needed.
  bool quiet = false;
  std::string calibration_path = kDefaultCalibrationPath;
  std::string label;      // task name (set as comm so the scheduler sees it)
  int metrics_fd = -1;    // shared-memory metric array fd (from replay_trace)
  long metrics_index = -1;  // this task's slot in that array
  long metrics_count = 0;   // number of slots in the array
  int seg_fd = -1;          // shared-memory occupancy ring fd (from replay_trace)
  long seg_max = 0;         // ring capacity (CpuSeg entries) per task

  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--calibration" && i + 1 < argc) {
      calibration_path = argv[++i];
    } else if (arg == "--label" && i + 1 < argc) {
      label = argv[++i];
    } else if (arg == "--metrics-fd" && i + 1 < argc) {
      metrics_fd = std::atoi(argv[++i]);
    } else if (arg == "--metrics-index" && i + 1 < argc) {
      metrics_index = std::atol(argv[++i]);
    } else if (arg == "--metrics-count" && i + 1 < argc) {
      metrics_count = std::atol(argv[++i]);
    } else if (arg == "--seg-fd" && i + 1 < argc) {
      seg_fd = std::atoi(argv[++i]);
    } else if (arg == "--seg-max" && i + 1 < argc) {
      seg_max = std::atol(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      args.push_back(arg);
    }
  }

  // Map this task's metric slot (written by us alone -- no shared lock, so no
  // contention with other tasks). replay_trace owns the array (an in-memory
  // memfd) and reads it after we exit.
  TaskMetric* metric = nullptr;
  if (metrics_fd >= 0 && metrics_index >= 0 && metrics_index < metrics_count) {
    size_t bytes = static_cast<size_t>(metrics_count) * sizeof(TaskMetric);
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                   metrics_fd, 0);
    if (p != MAP_FAILED) {
      metric = &static_cast<TaskMetric*>(p)[metrics_index];
    }
  }

  // Map this task's region of the shared occupancy ring (second memfd, sized
  // metrics_count*seg_max entries). Like the metric slot, we touch only our own
  // region [metrics_index*seg_max, ...]. Armed after FirstRun, below.
  if (seg_fd >= 0 && seg_max > 0 && metrics_index >= 0 &&
      metrics_index < metrics_count) {
    size_t bytes =
        static_cast<size_t>(metrics_count) * seg_max * sizeof(CpuSeg);
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, seg_fd, 0);
    if (p != MAP_FAILED) {
      g_occ_segs = &static_cast<CpuSeg*>(p)[metrics_index * seg_max];
      g_occ_max = seg_max;
    }
  }

  // Name this task before it joins the scheduler, so the scheduler's TaskNew /
  // FirstRun log lines and our TaskDead line all carry the same name.
  if (!label.empty()) {
    prctl(PR_SET_NAME, label.c_str());
  }

  // Opt this task into the active sched_ext scheduler. This is mandatory: we
  // abort rather than ever run on the native scheduler, which would silently
  // pollute the experiment. (1) Confirm a sched_ext scheduler is actually
  // active (and matches $SCX_EXPECT_OPS if set), then (2) enter the SCHED_EXT
  // class so the partial-mode scheduler enrolls this task.
  std::string scx_err = VerifyScxActive();
  if (!scx_err.empty()) {
    std::cerr << "launch_function: " << scx_err << std::endl;
    return 2;
  }
  if (!UseSchedExt()) {
    std::cerr << "launch_function: failed to enter SCHED_EXT: "
              << std::strerror(errno) << " (are we root?)" << std::endl;
    return 2;
  }

  // TaskNew: record here, on the task itself, right before we deactivate -- the
  // moment we are about to enter the scheduler's queue. The shared-memory slot
  // write touches only our own slot (no lock, no I/O).
  if (metric) metric->tasknew_ns = MonotonicNs();

  // Block briefly right after joining SCHED_EXT. A task that switches into the
  // class while already running would otherwise keep running in place and, if
  // it finishes within one time slice, never pass through the scheduler's
  // global queue at all. A short sleep deactivates the task; its wakeup is then
  // enqueued through the scheduler and dispatched from there -- so every
  // task goes through the scheduler's queue and gets a TaskNew/FirstRun record
  // (mirroring ghOSt, where a task does not run until the agent schedules it).
  {
    timespec req{0, 100000};  // 100 us
    while (nanosleep(&req, &req) < 0 && errno == EINTR) {
    }
  }
  // The nanosleep above dequeued us from the CPU and re-enqueued us through the
  // scheduler. We are now back on a worker CPU, scheduled by the scx scheduler.
  // Record FirstRun here — on the task itself — so the timestamp reflects when
  // the task is actually running on a CPU, not when the agent called dispatch.
  // Also record which worker CPU we landed on: under RR that is the per-core
  // queue this task was round-robined into, which lets replay_trace compute the
  // per-core backlog offline.
  if (metric) {
    metric->firstrun_ns = MonotonicNs();
    int cpu = sched_getcpu();
    metric->cpu = (cpu >= 0) ? static_cast<uint64_t>(cpu) : UINT64_MAX;
    // Start occupancy sampling now that we are running on a worker CPU. The first
    // seg is this FirstRun (core, time); the timer records every later core change
    // until DisarmOccupancyTimer() below. (No-op when no ring was provided.)
    ArmOccupancyTimer(metric->firstrun_ns, cpu);
  }

  std::string kind = "cpu_fib";
  int level = 0;
  uint64_t raw_arg1 = 0;
  if (args.size() == 1) {
    if (!ParseInt(args[0].c_str(), &level)) {
      PrintUsage(argv[0]);
      return 1;
    }
  } else if (args.size() >= 2) {
    kind = args[0];
    if (kind == "mem_stream_raw" || kind == "io_sleep_read_raw") {
      if (!ParseUint64(args[1].c_str(), &raw_arg1)) {
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (!ParseInt(args[1].c_str(), &level)) {
      PrintUsage(argv[0]);
      return 1;
    }
  } else {
    PrintUsage(argv[0]);
    return 1;
  }

  uint64_t result = 0;
  if (kind == "cpu_fib") {
    if (args.size() != 1 && args.size() != 2) {
      PrintUsage(argv[0]);
      return 1;
    }
    result = Fibonacci(level);
  } else if (kind == "mem_stream") {
    if (args.size() != 2) {
      PrintUsage(argv[0]);
      return 1;
    }
    CalibrationEntry entry =
        LookupCalibration("mem_stream", level, calibration_path);
    uint64_t units = entry.found ? entry.arg1 : std::max(1, level - 28);
    result = RunMemStream(units);
  } else if (kind == "io_sleep_read") {
    if (args.size() != 2) {
      PrintUsage(argv[0]);
      return 1;
    }
    CalibrationEntry entry =
        LookupCalibration("io_sleep_read", level, calibration_path);
    uint64_t sleep_ns =
        entry.found ? entry.arg1 : static_cast<uint64_t>(level) * 1000000ULL;
    uint64_t read_iters = entry.found ? entry.arg2 : 1;
    result = RunIoSleepRead(sleep_ns, read_iters);
  } else if (kind == "mem_stream_raw") {
    if (args.size() != 2) {
      PrintUsage(argv[0]);
      return 1;
    }
    result = RunMemStream(raw_arg1);
  } else if (kind == "io_sleep_read_raw") {
    if (args.size() != 3) {
      PrintUsage(argv[0]);
      return 1;
    }
    uint64_t read_iters = 0;
    if (!ParseUint64(args[2].c_str(), &read_iters)) {
      PrintUsage(argv[0]);
      return 1;
    }
    result = RunIoSleepRead(raw_arg1, read_iters);
  } else {
    PrintUsage(argv[0]);
    return 1;
  }

  // TaskDead: the task has finished its work and is about to exit.
  if (metric) metric->taskdead_ns = MonotonicNs();

  // Stop occupancy sampling and publish the seg count for replay_trace to read.
  // Disarm first so the handler can't write past the count we record. The last
  // recorded seg implicitly runs until taskdead_ns.
  if (g_occ_segs != nullptr && g_occ_max > 0) {
    DisarmOccupancyTimer();
    if (metric) metric->reserved[0] = static_cast<uint64_t>(g_occ_count);
  }

  if (!quiet) {
    std::cout << result << std::endl;
  }
  return 0;
}