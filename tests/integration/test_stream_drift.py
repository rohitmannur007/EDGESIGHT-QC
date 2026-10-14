import asyncio
import io
import signal
import sys
import types

from PIL import Image

from edgesight.monitoring import MonitoringLoop
from edgesight.persistence import EdgeSightStore
from edgesight.streaming import InMemoryBroker, StreamingWorker


class Pipeline:
    def predict(self, payload: bytes, filename: str | None = None, metadata=None):
        return {"decision": "PASS", "anomaly_score": 0.1, "defect_class": None}


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def test_shifted_frames_persist_a_drift_alert() -> None:
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    loop = MonitoringLoop(
        {"brightness": [10, 12, 11, 9, 13, 10, 12, 11]},
        store=store,
        baseline_version="test-clean",
        window_size=2,
        consecutive_windows=1,
        psi_threshold=0.2,
        ks_threshold=0.2,
    )
    broker = InMemoryBroker()
    worker = StreamingWorker(Pipeline(), store, broker, monitoring_loop=loop)
    for index in range(2):
        outcome = asyncio.run(
            worker.handle(
                {
                    "message_id": f"shifted-{index}",
                    "payload": _png((255, 255, 255)),
                    "filename": "shifted.png",
                }
            )
        )
        assert outcome == "processed"

    rows = store._connection.execute("SELECT feature FROM drift_events").fetchall()
    assert any(row[0] == "brightness" for row in rows)
    event = broker.poll("edgesight.monitoring.events")
    assert event is not None
    assert event["value"]["alerted"] is True


def test_stream_command_passes_configured_monitoring_loop(monkeypatch, tmp_path) -> None:
    from edgesight.cli import _stream

    handlers: dict[int, object] = {}
    monkeypatch.chdir(tmp_path)
    project = tmp_path
    config_dir = project / "configs"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text("mode: demo\ncategory: capsule\n", encoding="utf-8")
    (config_dir / "drift.yaml").write_text(
        "baseline_version: clean-v1\nwindow_size: 4\nconsecutive_windows: 1\n"
        "psi_threshold: 0.2\nks_threshold: 0.2\n",
        encoding="utf-8",
    )
    baseline = config_dir / "drift_baseline.json"
    baseline.write_text('{"brightness": [10, 11, 12, 13]}\n', encoding="utf-8")
    stream_config = project / "stream.yaml"
    stream_config.write_text(
        "mode: demo\n"
        "app_config: configs/app.yaml\n"
        "database:\n"
        f"  path: {project / 'edgesight.duckdb'}\n"
        "stream:\n"
        "  brokers: localhost:19092\n"
        "monitoring:\n"
        "  config: configs/drift.yaml\n"
        "  baseline: configs/drift_baseline.json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))

    class Consumer:
        def __init__(self, config):
            pass

        def subscribe(self, topics):
            pass

        def poll(self, timeout):
            handlers[signal.SIGINT]()
            return None

        def close(self):
            pass

    class Producer:
        def __init__(self, config):
            pass

        def flush(self, timeout=10):
            return 0

    kafka = types.ModuleType("confluent_kafka")
    kafka.Consumer = Consumer
    kafka.Producer = Producer
    monkeypatch.setitem(sys.modules, "confluent_kafka", kafka)

    import edgesight.streaming.worker as worker_module

    captured: dict[str, object] = {}
    original = worker_module.StreamingWorker

    class Spy(original):
        def __init__(self, *args, **kwargs):
            captured["loop"] = kwargs.get("monitoring_loop")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(worker_module, "StreamingWorker", Spy)
    assert _stream(stream_config) == 0
    loop = captured["loop"]
    assert isinstance(loop, MonitoringLoop)
    assert loop.window_size == 4
    assert loop.baseline_version == "clean-v1"
