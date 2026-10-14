import json

from edgesight.edge import benchmark, optimize_joint_thresholds


def test_benchmark_contains_real_samples_and_serializes():
    result = benchmark(lambda value: value + 1, 1, runs=3, warmup_runs=1)
    assert len(result.timings_ms) == 3
    assert result.mean_ms >= 0
    json.dumps(result.as_dict())


def test_joint_cost_sweep_respects_false_alarm_constraint():
    result = optimize_joint_thresholds(
        [0.1, 0.8, 0.2, 0.9],
        [0.1, 0.2, 0.9, 0.8],
        [0, 1, 1, 1],
        max_false_alarm_rate=0,
    )
    assert result.status == "OK"
    assert result.selected is not None
    assert result.selected.false_alarm_rate == 0
    json.dumps(result.as_dict())
