"""Ordered DuckDB schema migrations.

Migrations are append-only.  Each statement must be safe inside a transaction.
"""

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """CREATE TABLE IF NOT EXISTS batches (
                id VARCHAR PRIMARY KEY, created_at TIMESTAMP NOT NULL,
                status VARCHAR NOT NULL, item_count INTEGER NOT NULL DEFAULT 0,
                metadata JSON
            )""",
            """CREATE TABLE IF NOT EXISTS results (
                id VARCHAR PRIMARY KEY, batch_id VARCHAR, request_id VARCHAR,
                source_name VARCHAR, status VARCHAR NOT NULL,
                prediction JSON, error VARCHAR, created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS reviews (
                id VARCHAR PRIMARY KEY, result_id VARCHAR NOT NULL,
                reviewer VARCHAR NOT NULL, verdict VARCHAR NOT NULL,
                comment VARCHAR, created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS drift_observations (
                id VARCHAR PRIMARY KEY, feature VARCHAR NOT NULL,
                value DOUBLE NOT NULL, observed_at TIMESTAMP NOT NULL,
                metadata JSON
            )""",
            """CREATE TABLE IF NOT EXISTS processed_messages (
                message_id VARCHAR PRIMARY KEY, processed_at TIMESTAMP NOT NULL,
                result_id VARCHAR
            )""",
            """CREATE TABLE IF NOT EXISTS service_metrics (
                name VARCHAR NOT NULL, value DOUBLE NOT NULL,
                labels JSON, observed_at TIMESTAMP NOT NULL
            )""",
        ),
    ),
    (
        2,
        (
            """CREATE TABLE IF NOT EXISTS model_information (
                version VARCHAR PRIMARY KEY, runtime VARCHAR NOT NULL,
                device VARCHAR NOT NULL, mode VARCHAR NOT NULL,
                metadata JSON, registered_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS drift_events (
                id VARCHAR PRIMARY KEY, feature VARCHAR NOT NULL,
                severity VARCHAR NOT NULL, evidence JSON,
                baseline_version VARCHAR NOT NULL, created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS benchmark_records (
                id VARCHAR PRIMARY KEY, variant VARCHAR NOT NULL,
                runtime VARCHAR NOT NULL, device VARCHAR NOT NULL,
                latency_mean_ms DOUBLE, latency_p95_ms DOUBLE,
                throughput_fps DOUBLE, size_mb DOUBLE, quality_delta DOUBLE,
                status VARCHAR NOT NULL, metadata JSON, created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS experiment_summaries (
                id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL,
                source_revision VARCHAR, manifest_hash VARCHAR,
                configuration JSON, metrics JSON, artifact_paths JSON,
                created_at TIMESTAMP NOT NULL
            )""",
        ),
    ),
    (
        3,
        (
            """CREATE TABLE IF NOT EXISTS message_states (
                message_id VARCHAR PRIMARY KEY, state VARCHAR NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, result_id VARCHAR,
                last_error VARCHAR, metadata JSON,
                created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS monitoring_baselines (
                name VARCHAR PRIMARY KEY, features JSON NOT NULL,
                sample_count INTEGER NOT NULL, version VARCHAR NOT NULL,
                metadata JSON, created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS monitoring_windows (
                id VARCHAR PRIMARY KEY, baseline_name VARCHAR NOT NULL,
                sample_count INTEGER NOT NULL, reports JSON NOT NULL,
                consecutive_drift INTEGER NOT NULL, alerted BOOLEAN NOT NULL,
                metadata JSON, created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS operational_artifacts (
                id VARCHAR PRIMARY KEY, kind VARCHAR NOT NULL,
                name VARCHAR NOT NULL, status VARCHAR NOT NULL,
                payload JSON, artifact_path VARCHAR,
                synthetic BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL
            )""",
        ),
    ),
)
