"""Wire-message validation without coupling to a schema library version."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class InvalidMessage(ValueError):
    pass


@dataclass(frozen=True)
class InferenceMessage:
    message_id: str
    payload: bytes
    filename: str | None = None
    metadata: dict[str, Any] | None = None


def validate_message(value: Mapping[str, Any], schema: type[Any] | None = None) -> InferenceMessage:
    """Validate a mapping, optionally first through an edgesight.schemas model."""
    candidate: Any = value
    if schema is not None:
        if hasattr(schema, "model_validate"):
            candidate = schema.model_validate(value)
        elif hasattr(schema, "parse_obj"):
            candidate = schema.parse_obj(value)
        else:
            candidate = schema(**value)
        candidate = candidate.model_dump() if hasattr(candidate, "model_dump") else vars(candidate)
    if not isinstance(candidate, Mapping):
        raise InvalidMessage("message must be an object")
    message_id = candidate.get("message_id") or candidate.get("id")
    if not isinstance(message_id, str) or not message_id.strip():
        raise InvalidMessage("message_id must be a non-empty string")
    payload = candidate.get("payload")
    encoding = candidate.get("encoding")
    if isinstance(payload, str):
        try:
            payload = (
                base64.b64decode(payload, validate=True)
                if encoding == "base64"
                else payload.encode()
            )
        except ValueError as exc:
            raise InvalidMessage("payload is not valid base64") from exc
    if not isinstance(payload, bytes) or not payload:
        raise InvalidMessage("payload must be non-empty bytes or text")
    metadata = candidate.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise InvalidMessage("metadata must be an object")
    return InferenceMessage(
        message_id=message_id,
        payload=payload,
        filename=candidate.get("filename"),
        metadata=dict(metadata),
    )
