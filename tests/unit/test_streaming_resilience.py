import asyncio

from edgesight.persistence import EdgeSightStore
from edgesight.streaming import InMemoryBroker, StreamingWorker


class FlakyPipeline:
    def __init__(self):
        self.calls = 0

    def predict(self, payload, filename=None):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary")
        return {"decision": "PASS"}


def test_transient_failure_is_redeliverable_and_metadata_propagates():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    broker = InMemoryBroker()
    worker = StreamingWorker(FlakyPipeline(), store, broker)
    raw = {"message_id": "one", "payload": "bytes", "metadata": {"camera_id": "A"}}
    assert asyncio.run(worker.handle(raw)) == "retry"
    assert store.message_state("one")["state"] == "retry"
    assert asyncio.run(worker.handle(raw)) == "processed"
    result = broker.poll("edgesight.results")["value"]
    assert result["metadata"]["camera_id"] == "A"
    assert store.message_state("one")["attempts"] == 2


def test_dlq_omits_payload():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    broker = InMemoryBroker()
    worker = StreamingWorker(object(), store, broker, max_attempts=1)
    asyncio.run(worker.handle({"message_id": "bad", "payload": b"secret"}))
    assert broker.poll("edgesight.dlq")["value"]["original"]["payload"] == {"omitted": True}
