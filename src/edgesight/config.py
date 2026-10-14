"""Typed configuration with environment and YAML support."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path | None = None
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"


@dataclass(frozen=True)
class AnomalyConfig:
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    memory_samples: int = 2048
    distance_chunk_size: int = 1024
    device: str = "cpu"


@dataclass(frozen=True)
class RouterConfig:
    detection_acceptance: float = 0.70
    detection_rejection: float = 0.20
    anomaly_threshold: float = 0.50
    fail_closed: bool = True


@dataclass(frozen=True)
class EdgeConfig:
    input_size: tuple[int, int] = (640, 640)
    quantize: bool = False
    warmup_runs: int = 3
    benchmark_runs: int = 20


@dataclass(frozen=True)
class EdgeSightConfig:
    seed: int = 42
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


T = TypeVar("T")


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _coerce_dataclass(cls: type[T], values: Mapping[str, Any]) -> T:
    valid = {item.name: item for item in fields(cls)}
    unknown = set(values) - set(valid)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    converted = dict(values)
    if cls is DetectorConfig and converted.get("model_path") is not None:
        converted["model_path"] = Path(converted["model_path"])
    if cls is AnomalyConfig and "layers" in converted:
        converted["layers"] = tuple(converted["layers"])
    if cls is EdgeConfig and "input_size" in converted:
        converted["input_size"] = tuple(converted["input_size"])
    return cls(**converted)


def _parse_document(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML configuration; use JSON or install pyyaml"
            ) from exc
        value = yaml.safe_load(text) or {}
    if not isinstance(value, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return value


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> EdgeSightConfig:
    """Load config and apply ``EDGESIGHT_*`` environment overrides.

    Nested environment keys use double underscores, for example
    ``EDGESIGHT_ROUTER__ANOMALY_THRESHOLD=0.7``. Values are JSON-decoded when
    possible, preserving strings when decoding fails.
    """

    raw: dict[str, Any] = {}
    if path is not None:
        raw.update(_parse_document(Path(path)))
    if overrides:
        _deep_merge(raw, dict(overrides))
    prefix = "EDGESIGHT_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        try:
            cursor[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            cursor[parts[-1]] = value

    valid_top = {item.name for item in fields(EdgeSightConfig)}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(f"Unknown EdgeSightConfig keys: {sorted(unknown)}")
    return EdgeSightConfig(
        seed=int(raw.get("seed", 42)),
        detector=_coerce_dataclass(DetectorConfig, raw.get("detector", {})),
        anomaly=_coerce_dataclass(AnomalyConfig, raw.get("anomaly", {})),
        router=_coerce_dataclass(RouterConfig, raw.get("router", {})),
        edge=_coerce_dataclass(EdgeConfig, raw.get("edge", {})),
    )


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
