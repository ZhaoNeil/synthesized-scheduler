#!/usr/bin/env python3
"""Compute aggregate workload metrics from a TaskNew/FirstRun/TaskDead log."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LINE_RE = re.compile(
    r"^(TaskNew|FirstRun|TaskDead): (\S+) "
    r"(?:enqueue at|starts at|at) "
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(\d{1,9}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)
EVENT_FIELDS = {
    "TaskNew": "task_new_ns",
    "FirstRun": "first_run_ns",
    "TaskDead": "complete_ns",
}


@dataclass
class Task:
    task_new_ns: int | None = None
    first_run_ns: int | None = None
    complete_ns: int | None = None


def parse_timestamp_ns(base: str, fraction: str | None, offset: str) -> int:
    if offset == "Z":
        offset = "+00:00"
    timestamp = datetime.fromisoformat(base + offset)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = timestamp.astimezone(timezone.utc) - epoch
    seconds = delta.days * 86400 + delta.seconds
    nanoseconds = int((fraction or "").ljust(9, "0"))
    return seconds * 1_000_000_000 + nanoseconds


def parse_log(path: Path) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    ignored_lines = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        match = LINE_RE.match(line)
        if match is None:
            ignored_lines += 1
            continue

        event, task_id, base, fraction, offset = match.groups()
        field = EVENT_FIELDS[event]
        task = tasks.setdefault(task_id, Task())
        if getattr(task, field) is not None:
            raise ValueError(
                f"{path}:{line_number}: duplicate {event} for task {task_id}"
            )
        setattr(task, field, parse_timestamp_ns(base, fraction, offset))

    if not tasks:
        raise ValueError(f"{path}: no TaskNew/FirstRun/TaskDead events found")
    if ignored_lines:
        print(f"warning: ignored {ignored_lines} unrecognized line(s)", file=sys.stderr)
    return tasks


def require_complete_tasks(path: Path, tasks: dict[str, Task]) -> None:
    invalid: list[str] = []
    for task_id, task in tasks.items():
        missing = [
            name
            for name, value in (
                ("TaskNew", task.task_new_ns),
                ("FirstRun", task.first_run_ns),
                ("TaskDead", task.complete_ns),
            )
            if value is None
        ]
        if missing:
            invalid.append(f"{task_id}: missing {', '.join(missing)}")
            continue

        assert task.task_new_ns is not None
        assert task.first_run_ns is not None
        assert task.complete_ns is not None
        if not task.task_new_ns <= task.first_run_ns <= task.complete_ns:
            invalid.append(f"{task_id}: events are not in lifecycle order")

    if invalid:
        details = "\n  ".join(invalid)
        raise ValueError(f"{path}: invalid task lifecycle(s):\n  {details}")


def format_seconds(nanoseconds: int) -> str:
    return f"{nanoseconds / 1_000_000_000:.9f} s"


def queue_depth_series(tasks: dict[str, Task]) -> tuple[list[tuple[int, int]], int]:
    """Reconstruct the true queue backlog over time from the lifecycle log.

    A task is *waiting for a CPU* during [TaskNew, FirstRun): it has entered the
    global queue but has not yet started running. This is the real backlog the
    scheduler's own queue_log cannot see, because under saturation the waiting
    tasks sit in kernel per-CPU DSQs, not in the agent's user-space queue.

    Tolerant of partial logs: a task with a TaskNew but no FirstRun (e.g. from a
    run that froze or was aborted) is counted as waiting from TaskNew onward and
    never leaves the queue.

    Returns (series, peak) where series is a list of (ns_since_start, waiting)
    sampled at every change point, and peak is the maximum concurrent waiting.
    """
    events: list[tuple[int, int]] = []
    for task in tasks.values():
        if task.task_new_ns is None:
            continue  # never entered the queue; nothing to count
        events.append((task.task_new_ns, +1))
        if task.first_run_ns is not None:
            events.append((task.first_run_ns, -1))
    if not events:
        return [], 0
    events.sort()

    start = events[0][0]
    series: list[tuple[int, int]] = []
    waiting = 0
    peak = 0
    i = 0
    n = len(events)
    while i < n:
        # Collapse all events at the same timestamp into one sample.
        ts = events[i][0]
        while i < n and events[i][0] == ts:
            waiting += events[i][1]
            i += 1
        peak = max(peak, waiting)
        series.append((ts - start, waiting))
    return series, peak


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute aggregate metrics from a TaskNew/FirstRun/TaskDead log."
    )
    parser.add_argument("log", type=Path, help="path to a per-task lifecycle log")
    parser.add_argument(
        "--queue-series",
        type=Path,
        default=None,
        help="write the true queue backlog over time (seconds_since_start "
        "waiting) reconstructed from the lifecycle log to this file",
    )
    args = parser.parse_args()

    try:
        tasks = parse_log(args.log)
    except ValueError as error:
        parser.error(str(error))

    # Reconstruct and write the true backlog first, so it is available even for a
    # partial log (e.g. from a run that froze) where the strict checks below fail.
    series, peak_waiting = queue_depth_series(tasks)
    if args.queue_series is not None:
        with args.queue_series.open("w", encoding="utf-8") as out:
            out.write("# seconds_since_start  waiting_for_cpu\n")
            for ns, waiting in series:
                out.write(f"{ns / 1_000_000_000:.6f} {waiting}\n")

    try:
        require_complete_tasks(args.log, tasks)
    except ValueError as error:
        print(f"peak queue backlog: {peak_waiting} tasks waiting for a CPU")
        parser.error(str(error))

    task_values = list(tasks.values())
    task_new_times = [task.task_new_ns for task in task_values]
    first_run_times = [task.first_run_ns for task in task_values]
    complete_times = [task.complete_ns for task in task_values]
    assert all(value is not None for value in task_new_times)
    assert all(value is not None for value in first_run_times)
    assert all(value is not None for value in complete_times)

    accumulated_latency_ns = sum(
        first_run - task_new
        for task_new, first_run in zip(task_new_times, first_run_times)
    )
    accumulated_runtime_ns = sum(
        complete - first_run
        for first_run, complete in zip(first_run_times, complete_times)
    )
    makespan_ns = max(complete_times) - min(task_new_times)

    print(f"log: {args.log}")
    print(f"tasks: {len(tasks)}")
    print(f"accumulated task latency: {format_seconds(accumulated_latency_ns)}")
    print(f"accumulated task runtime: {format_seconds(accumulated_runtime_ns)}")
    print(f"makespan: {format_seconds(makespan_ns)}")
    print(f"peak queue backlog: {peak_waiting} tasks waiting for a CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
