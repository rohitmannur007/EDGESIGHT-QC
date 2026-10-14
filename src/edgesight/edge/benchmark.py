"""Honest single-process latency benchmarking."""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    runs: int
    warmup_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_per_second: float
    environment: dict[str, str]
    timings_ms: tuple[float, ...] = ()
    status: str = "OK"
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    operation: Callable[[Any], Any],
    sample: Any,
    *,
    runs: int = 20,
    warmup_runs: int = 3,
    synchronize: Callable[[], None] | None = None,
) -> BenchmarkResult:
    """Benchmark a callable; callers must provide GPU synchronization if needed."""

    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs must be positive and warmup_runs non-negative")
    for _ in range(warmup_runs):
        operation(sample)
        if synchronize:
            synchronize()
    timings: list[float] = []
    for _ in range(runs):
        if synchronize:
            synchronize()
        started = time.perf_counter_ns()
        operation(sample)
        if synchronize:
            synchronize()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    mean = statistics.fmean(timings)
    return BenchmarkResult(
        runs=runs,
        warmup_runs=warmup_runs,
        mean_ms=mean,
        median_ms=statistics.median(timings),
        p95_ms=_percentile(timings, 0.95),
        throughput_per_second=1000 / mean if mean else float("inf"),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "synchronized": str(synchronize is not None).lower(),
        },
        timings_ms=tuple(timings),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def unsupported_benchmark(reason: str) -> BenchmarkResult:
    """Create an explicit non-result without fabricating latency numbers."""
    return BenchmarkResult(
        runs=0,
        warmup_runs=0,
        mean_ms=0.0,
        median_ms=0.0,
        p95_ms=0.0,
        throughput_per_second=0.0,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "synchronized": "false",
        },
        status="NOT_SUPPORTED",
        reason=reason,
    )
