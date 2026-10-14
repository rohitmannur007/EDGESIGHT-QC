import asyncio

from edgesight.persistence import EdgeSightStore
from edgesight.streaming import InMemoryBroker, StreamingWorker


class Pipeline:
    def predict(self, payload: bytes, filename: str | None = None):
        return {"bytes": len(payload), "filename": filename}


def test_worker_publishes_and_deduplicates():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    broker = InMemoryBroker()
    worker = StreamingWorker(Pipeline(), store, broker)
    message = {"message_id": "event-1", "payload": "part", "filename": "part.jpg"}

    assert asyncio.run(worker.handle(message)) == "processed"
    assert asyncio.run(worker.handle(message)) == "duplicate"
    assert broker.poll("edgesight.results")["value"]["prediction"]["bytes"] == 4
    assert broker.poll("edgesight.results") is None


def test_invalid_message_goes_to_dlq():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    broker = InMemoryBroker()
    worker = StreamingWorker(Pipeline(), store, broker)

    assert asyncio.run(worker.handle({"payload": ""})) == "dead_letter"
    assert broker.poll("edgesight.dlq")["value"]["error_type"] == "InvalidMessage"
