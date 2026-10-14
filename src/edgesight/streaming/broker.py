"""Broker interfaces with a deterministic in-memory implementation."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MessageProducer(Protocol):
    def send(self, topic: str, message: Mapping[str, Any], *, key: str | None = None) -> None: ...


class InMemoryBroker:
    """Small FIFO broker intended for tests and local demos."""

    def __init__(self) -> None:
        self.topics: dict[str, deque[dict[str, Any]]] = defaultdict(deque)

    def send(self, topic: str, message: Mapping[str, Any], *, key: str | None = None) -> None:
        self.topics[topic].append({"key": key, "value": dict(message)})

    def poll(self, topic: str) -> dict[str, Any] | None:
        return self.topics[topic].popleft() if self.topics[topic] else None


class KafkaProducerAdapter:
    """Optional confluent-kafka adapter loaded only when instantiated."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise RuntimeError("install the 'kafka' extra to use KafkaProducerAdapter") from exc
        self._producer = Producer(dict(config))

    def send(self, topic: str, message: Mapping[str, Any], *, key: str | None = None) -> None:
        self._producer.produce(topic, key=key, value=json.dumps(message, default=str).encode())
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> int:
        return int(self._producer.flush(timeout))
