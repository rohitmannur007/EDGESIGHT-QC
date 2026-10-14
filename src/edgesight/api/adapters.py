"""Compatibility adapters for inference implementations."""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return serializable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return serializable(value.dict())
    if is_dataclass(value):
        return {field.name: serializable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return serializable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [serializable(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "tolist"):
            return serializable(tensor.tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return serializable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return serializable(value.item())
        except (TypeError, ValueError):
            pass
    return value


async def run_pipeline(
    pipeline: Any,
    payload: bytes,
    *,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Invoke the first supported pipeline interface.

    Expected interface is one of ``predict(bytes, filename=...)``,
    ``process(bytes, filename=...)``, ``infer(bytes, filename=...)``, or a
    callable with that signature. Keyword compatibility is detected rather
    than requiring a concrete core implementation.
    """
    target = next(
        (
            getattr(pipeline, name)
            for name in ("predict", "process", "infer")
            if callable(getattr(pipeline, name, None))
        ),
        pipeline if callable(pipeline) else None,
    )
    if target is None:
        raise TypeError("pipeline must implement predict, process, infer, or __call__")
    parameters = inspect.signature(target).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "filename" in parameters:
        kwargs["filename"] = filename
    if accepts_kwargs or "metadata" in parameters:
        kwargs["metadata"] = metadata
    result = target(payload, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return serializable(result)
