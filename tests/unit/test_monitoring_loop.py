from edgesight.monitoring import MonitoringLoop
from edgesight.persistence import EdgeSightStore


def test_consecutive_monitoring_alert_is_persisted():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    monitor = MonitoringLoop(
        {"brightness": [10.0] * 20},
        store=store,
        window_size=2,
        consecutive_windows=2,
    )
    assert monitor.observe({"brightness": 100.0}) is None
    first = monitor.observe({"brightness": 100.0})
    monitor.observe({"brightness": 100.0})
    second = monitor.observe({"brightness": 100.0})
    assert first is not None and not first.alerted
    assert second is not None and second.alerted
    assert store.query("SELECT COUNT(*) AS n FROM drift_events")[0]["n"] == 1
