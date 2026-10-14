"""DuckDB-backed operational repository."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .migrations import MIGRATIONS


class PersistenceError(RuntimeError):
    """Raised when the requested persistence backend cannot be used."""


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class EdgeSightStore:
    """Thread-safe repository with explicit opt-in SQLite fallback.

    DuckDB is always preferred. Set ``allow_sqlite_fallback=True`` or
    ``EDGESIGHT_ALLOW_SQLITE_FALLBACK=1`` only for constrained environments.
    """

    def __init__(
        self, path: str | Path = "edgesight.duckdb", *, allow_sqlite_fallback: bool = False
    ):
        self.path = str(path)
        self._lock = threading.RLock()
        self.backend = "duckdb"
        try:
            import duckdb

            self._connection = duckdb.connect(self.path)
        except ImportError as exc:
            allowed = allow_sqlite_fallback or os.getenv("EDGESIGHT_ALLOW_SQLITE_FALLBACK") == "1"
            if not allowed:
                raise PersistenceError(
                    "duckdb is required; explicitly enable SQLite fallback for tests/development"
                ) from exc
            self.backend = "sqlite-explicit-fallback"
            self._connection = sqlite3.connect(
                self.path, check_same_thread=False, isolation_level=None
            )
        self.migrate()

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def migrate(self) -> None:
        with self._lock:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL
                )"""
            )
            rows = self._connection.execute("SELECT version FROM schema_version").fetchall()
            applied = {int(row[0]) for row in rows}
            for version, statements in MIGRATIONS:
                if version in applied:
                    continue
                with self.transaction() as connection:
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                        [version, _now()],
                    )

    def create_batch(self, item_count: int, metadata: Mapping[str, Any] | None = None) -> str:
        batch_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                [batch_id, _now(), "processing", item_count, _json(metadata or {})],
            )
        return batch_id

    def set_batch_status(self, batch_id: str, status: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE batches SET status = ? WHERE id = ?", [status, batch_id]
            )

    def save_result(
        self,
        *,
        status: str,
        prediction: Any = None,
        error: str | None = None,
        batch_id: str | None = None,
        request_id: str | None = None,
        source_name: str | None = None,
        result_id: str | None = None,
    ) -> str:
        result_id = result_id or str(uuid.uuid4())
        now = _now()
        with self._lock:
            self._connection.execute(
                """INSERT INTO results
                   (id,batch_id,request_id,source_name,status,prediction,error,created_at,updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    result_id,
                    batch_id,
                    request_id,
                    source_name,
                    status,
                    _json(prediction),
                    error,
                    now,
                    now,
                ],
            )
        return result_id

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        rows = self.query(
            """SELECT id,batch_id,request_id,source_name,status,prediction,error,
                      created_at,updated_at
               FROM results WHERE id = ?""",
            [result_id],
        )
        if not rows:
            return None
        row = rows[0]
        row["prediction"] = json.loads(row["prediction"]) if row["prediction"] else None
        return row

    def list_results(
        self, *, limit: int = 100, offset: int = 0, batch_id: str | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id,batch_id,request_id,source_name,status,prediction,error,"
            "created_at,updated_at FROM results"
        )
        params: list[Any] = []
        if batch_id:
            sql += " WHERE batch_id = ?"
            params.append(batch_id)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([max(1, min(limit, 1000)), max(offset, 0)])
        rows = self.query(sql, params)
        for row in rows:
            row["prediction"] = json.loads(row["prediction"]) if row["prediction"] else None
        return rows

    def add_review(
        self, result_id: str, reviewer: str, verdict: str, comment: str | None = None
    ) -> str:
        review_id = str(uuid.uuid4())
        with self._lock:
            if not self._connection.execute(
                "SELECT 1 FROM results WHERE id = ?", [result_id]
            ).fetchone():
                raise KeyError(result_id)
            self._connection.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
                [review_id, result_id, reviewer, verdict, comment, _now()],
            )
        return review_id

    def list_reviews(self, result_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id,result_id,reviewer,verdict,comment,created_at FROM reviews"
        params: list[Any] = []
        if result_id:
            sql += " WHERE result_id = ?"
            params.append(result_id)
        return self.query(sql + " ORDER BY created_at DESC", params)

    def mark_processed(self, message_id: str, result_id: str | None = None) -> bool:
        """Legacy atomic completion marker; returns False for a duplicate."""
        with self._lock:
            if self._connection.execute(
                "SELECT 1 FROM processed_messages WHERE message_id = ?", [message_id]
            ).fetchone():
                return False
            self._connection.execute(
                "INSERT INTO processed_messages VALUES (?, ?, ?)", [message_id, _now(), result_id]
            )
            return True

    def claim_message(
        self,
        message_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Claim a new/failed message and return its durable processing state.

        Complete and currently-processing messages are not claimed. Failed messages
        may be redelivered and atomically transition back to processing.
        """
        now = _now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT state,attempts,result_id,last_error FROM message_states "
                "WHERE message_id = ?",
                [message_id],
            ).fetchone()
            exhausted = bool(
                row
                and row[0] == "failed"
                and max_attempts is not None
                and int(row[1]) >= max_attempts
            )
            if row and (row[0] in {"processing", "complete"} or exhausted):
                return {
                    "claimed": False,
                    "state": str(row[0]),
                    "attempts": int(row[1]),
                    "result_id": row[2],
                    "last_error": row[3],
                }
            attempts = int(row[1]) + 1 if row else 1
            if row:
                connection.execute(
                    """UPDATE message_states SET state='processing', attempts=?,
                       last_error=NULL, metadata=?, updated_at=? WHERE message_id=?""",
                    [attempts, _json(metadata or {}), now, message_id],
                )
            else:
                connection.execute(
                    "INSERT INTO message_states VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        message_id,
                        "processing",
                        attempts,
                        None,
                        None,
                        _json(metadata or {}),
                        now,
                        now,
                    ],
                )
        return {"claimed": True, "state": "processing", "attempts": attempts}

    def finish_message(
        self,
        message_id: str,
        state: str,
        *,
        result_id: str | None = None,
        error: str | None = None,
    ) -> None:
        if state not in {"complete", "failed", "retry"}:
            raise ValueError("message state must be complete, failed, or retry")
        with self._lock:
            self._connection.execute(
                """UPDATE message_states SET state=?, result_id=?, last_error=?,
                   updated_at=? WHERE message_id=?""",
                [state, result_id, error, _now(), message_id],
            )

    def message_state(self, message_id: str) -> dict[str, Any] | None:
        rows = self.query(
            """SELECT message_id,state,attempts,result_id,last_error,metadata,
                      created_at,updated_at FROM message_states WHERE message_id=?""",
            [message_id],
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
        return row

    def record_drift(
        self, feature: str, value: float, metadata: Mapping[str, Any] | None = None
    ) -> str:
        observation_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO drift_observations VALUES (?, ?, ?, ?, ?)",
                [observation_id, feature, float(value), _now(), _json(metadata or {})],
            )
        return observation_id

    def drift_values(self, feature: str, limit: int = 10000) -> list[float]:
        rows = self._connection.execute(
            """SELECT value FROM drift_observations
               WHERE feature = ? ORDER BY observed_at DESC LIMIT ?""",
            [feature, limit],
        ).fetchall()
        return [float(row[0]) for row in rows]

    def record_metric(
        self, name: str, value: float, labels: Mapping[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO service_metrics VALUES (?, ?, ?, ?)",
                [name, float(value), _json(labels or {}), _now()],
            )

    def save_baseline(
        self,
        name: str,
        features: Mapping[str, Sequence[float] | Mapping[str, float]],
        *,
        version: str = "v1",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        clean: dict[str, Any] = {
            key: (
                {str(label): float(value) for label, value in values.items()}
                if isinstance(values, Mapping)
                else [float(value) for value in values]
            )
            for key, values in features.items()
        }
        numeric_lengths = [len(values) for values in clean.values() if isinstance(values, list)]
        sample_count = min(numeric_lengths, default=len(clean.get("class_distribution", {})))
        if not clean or sample_count == 0:
            raise ValueError("baseline requires non-empty feature values")
        with self._lock:
            self._connection.execute("DELETE FROM monitoring_baselines WHERE name=?", [name])
            self._connection.execute(
                "INSERT INTO monitoring_baselines VALUES (?, ?, ?, ?, ?, ?)",
                [name, _json(clean), sample_count, version, _json(metadata or {}), _now()],
            )

    def get_baseline(self, name: str) -> dict[str, Any] | None:
        rows = self.query(
            """SELECT name,features,sample_count,version,metadata,created_at
               FROM monitoring_baselines WHERE name=?""",
            [name],
        )
        if not rows:
            return None
        row = rows[0]
        row["features"] = json.loads(row["features"])
        row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
        return row

    def record_monitoring_window(
        self,
        baseline_name: str,
        reports: Mapping[str, Any],
        *,
        sample_count: int,
        consecutive_drift: int,
        alerted: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        window_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO monitoring_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    window_id,
                    baseline_name,
                    sample_count,
                    _json(reports),
                    consecutive_drift,
                    alerted,
                    _json(metadata or {}),
                    _now(),
                ],
            )
        return window_id

    def record_drift_event(
        self,
        feature: str,
        evidence: Mapping[str, Any],
        *,
        baseline_version: str,
        severity: str = "warning",
    ) -> str:
        event_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO drift_events VALUES (?, ?, ?, ?, ?, ?)",
                [event_id, feature, severity, _json(evidence), baseline_version, _now()],
            )
        return event_id

    def record_artifact(
        self,
        kind: str,
        name: str,
        status: str,
        payload: Mapping[str, Any],
        *,
        artifact_path: str | Path | None = None,
        synthetic: bool = False,
    ) -> str:
        artifact_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO operational_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    artifact_id,
                    kind,
                    name,
                    status,
                    _json(payload),
                    str(artifact_path) if artifact_path else None,
                    synthetic,
                    _now(),
                ],
            )
        return artifact_id

    def record_benchmark(
        self,
        *,
        variant: str,
        runtime: str,
        device: str,
        status: str,
        mean_ms: float | None = None,
        p95_ms: float | None = None,
        throughput_fps: float | None = None,
        size_mb: float | None = None,
        quality_delta: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        record_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO benchmark_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    record_id,
                    variant,
                    runtime,
                    device,
                    mean_ms,
                    p95_ms,
                    throughput_fps,
                    size_mb,
                    quality_delta,
                    status,
                    _json(metadata or {}),
                    _now(),
                ],
            )
        return record_id

    def record_experiment(
        self,
        name: str,
        *,
        configuration: Mapping[str, Any],
        metrics: Mapping[str, Any],
        artifact_paths: Sequence[str | Path] = (),
        source_revision: str | None = None,
        manifest_hash: str | None = None,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO experiment_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    experiment_id,
                    name,
                    source_revision,
                    manifest_hash,
                    _json(configuration),
                    _json(metrics),
                    _json([str(path) for path in artifact_paths]),
                    _now(),
                ],
            )
        return experiment_id

    def list_artifacts(self, kind: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        sql = (
            "SELECT id,kind,name,status,payload,artifact_path,synthetic,created_at "
            "FROM operational_artifacts"
        )
        params: list[Any] = []
        if kind:
            sql += " WHERE kind=?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        rows = self.query(sql, params)
        for row in rows:
            row["payload"] = json.loads(row["payload"]) if row["payload"] else {}
        return rows

    def operational_metrics(self) -> dict[str, Any]:
        with self._lock:
            total = int(self._connection.execute("SELECT COUNT(*) FROM results").fetchone()[0])
            failed = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM results WHERE status = 'failed'"
                ).fetchone()[0]
            )
            pending = int(
                self._connection.execute(
                    """SELECT COUNT(*) FROM results r WHERE r.status = 'complete'
                       AND NOT EXISTS (SELECT 1 FROM reviews v WHERE v.result_id = r.id)"""
                ).fetchone()[0]
            )
            prediction_rows = self._connection.execute(
                "SELECT prediction FROM results WHERE prediction IS NOT NULL"
            ).fetchall()
        latencies: list[float] = []
        decisions: dict[str, int] = {}
        for row in prediction_rows:
            try:
                prediction = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(prediction, dict):
                continue
            decision = prediction.get("decision")
            if decision:
                decisions[str(decision)] = decisions.get(str(decision), 0) + 1
            latency = prediction.get("inference_latency_ms")
            if isinstance(latency, int | float):
                latencies.append(float(latency))
        ordered = sorted(latencies)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else 0.0
        return {
            "results_total": total,
            "results_failed": failed,
            "reviews_pending": pending,
            "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "latency_p95_ms": p95,
            **{f"decision_{name.lower()}": count for name, count in decisions.items()},
        }

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(sql, list(params))
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
