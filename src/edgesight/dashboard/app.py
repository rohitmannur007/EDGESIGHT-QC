"""EdgeSight-QC operations dashboard.

Run with: ``streamlit run src/edgesight/dashboard/app.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from edgesight.persistence import EdgeSightStore, PersistenceError

st.set_page_config(page_title="EdgeSight-QC Operations", page_icon="◈", layout="wide")


@st.cache_resource
def open_store(path: str) -> EdgeSightStore:
    return EdgeSightStore(path)


def render() -> None:
    st.title("EdgeSight-QC")
    st.caption("Quality inspection operations center")
    configured_path = os.getenv("EDGESIGHT_DB_PATH", "edgesight.duckdb")
    with st.sidebar:
        st.header("Data source")
        db_path = st.text_input("DuckDB path", configured_path)
        st.caption(f"Using `{Path(db_path).name}`")
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        store = open_store(db_path)
        metrics = store.operational_metrics()
    except (PersistenceError, OSError, RuntimeError) as exc:
        st.error(f"Database unavailable: {exc}")
        st.info("Configure EDGESIGHT_DB_PATH or enter a writable DuckDB path in the sidebar.")
        return

    results = store.list_results(limit=250)
    predictions = [row["prediction"] for row in results if isinstance(row.get("prediction"), dict)]
    decisions = {
        name: sum(item.get("decision") == name for item in predictions)
        for name in ("PASS", "KNOWN_DEFECT", "UNKNOWN_DEFECT", "HUMAN_REVIEW")
    }
    latencies = [
        float(item["inference_latency_ms"])
        for item in predictions
        if item.get("inference_latency_ms") is not None
    ]
    drift_events = store.query("SELECT * FROM drift_events ORDER BY created_at DESC LIMIT 100")
    artifacts = store.list_artifacts(limit=250)
    by_kind = {
        kind: [record for record in artifacts if record["kind"] == kind]
        for kind in ("evaluation", "benchmark", "cost", "drift", "export")
    }
    tabs = st.tabs(
        [
            "Overview",
            "Live feed",
            "Inspection details",
            "Human review",
            "Performance",
            "Drift",
            "Edge benchmarks",
            "Business thresholds",
            "Operational status",
        ]
    )

    with tabs[0]:
        columns = st.columns(6)
        columns[0].metric("Inspections", metrics["results_total"])
        for column, name in zip(
            columns[1:5],
            ("PASS", "KNOWN_DEFECT", "UNKNOWN_DEFECT", "HUMAN_REVIEW"),
            strict=True,
        ):
            column.metric(name.replace("_", " ").title(), decisions[name])
        columns[5].metric("Drift", "ALERT" if drift_events else "No alert")
        latency_columns = st.columns(3)
        latency_columns[0].metric(
            "Mean latency",
            f"{sum(latencies) / len(latencies):.2f} ms" if latencies else "NOT_RUN",
        )
        latency_columns[1].metric(
            "p95 latency",
            (
                f"{sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]:.2f} ms"
                if latencies
                else "NOT_RUN"
            ),
        )
        latency_columns[2].metric("Failures", metrics["results_failed"])
        st.warning("Demo records are synthetic software checks, not manufacturing performance.")

    with tabs[1]:
        if not results:
            st.info("No inspections yet. Submit an image through the API to get started.")
        else:
            decision_filter = st.multiselect(
                "Decision",
                sorted({str(item.get("decision")) for item in predictions}),
                default=[],
            )
            visible = [
                row
                for row in results
                if not decision_filter
                or (
                    isinstance(row.get("prediction"), dict)
                    and row["prediction"].get("decision") in decision_filter
                )
            ]
            st.dataframe(visible, use_container_width=True, hide_index=True)

    with tabs[2]:
        if not results:
            st.info("No inspection is available.")
        else:
            selected_id = st.selectbox(
                "Inspection", [row["id"] for row in results], key="inspection_detail"
            )
            selected = next(row for row in results if row["id"] == selected_id)
            prediction = selected.get("prediction") or {}
            references = prediction.get("artifact_references", {})
            image_columns = st.columns(3)
            displayed = False
            for column, (label, key) in zip(
                image_columns,
                (
                    ("Original", "original"),
                    ("Detection overlay", "annotated"),
                    ("Heatmap", "heatmap"),
                ),
                strict=True,
            ):
                path = Path(str(references.get(key, "")))
                if path.is_file():
                    column.image(str(path), caption=label, use_container_width=True)
                    displayed = True
            st.json(prediction)
            if not displayed:
                st.caption(
                    "Original/annotated image and heatmap are unavailable because no artifact "
                    "reference was persisted; this is distinct from an empty detection."
                )

    with tabs[3]:
        reviews = store.list_reviews()
        reviewed_ids = {item["result_id"] for item in reviews}
        pending = [
            row
            for row in results
            if row["status"] == "complete"
            and row["id"] not in reviewed_ids
            and isinstance(row.get("prediction"), dict)
            and row["prediction"].get("review_required")
        ]
        if not pending:
            st.success("Review queue is clear.")
        else:
            st.dataframe(pending, use_container_width=True, hide_index=True)
            item_id = st.selectbox("Item", [row["id"] for row in pending])
            verdict = st.selectbox(
                "Label",
                [
                    "true_defect",
                    "false_alarm",
                    "known_defect",
                    "new_defect_type",
                    "needs_further_investigation",
                ],
            )
            reviewer = st.text_input("Reviewer", "dashboard-reviewer")
            if st.button("Record review"):
                store.add_review(item_id, reviewer, verdict)
                st.success("Review persisted with reviewer provenance.")
        if reviews:
            st.subheader("Recent decisions")
            st.dataframe(reviews[:100], use_container_width=True, hide_index=True)

    with tabs[4]:
        experiments = store.query(
            "SELECT * FROM experiment_summaries ORDER BY created_at DESC LIMIT 100"
        )
        if experiments:
            st.dataframe(experiments, use_container_width=True, hide_index=True)
        if by_kind["evaluation"]:
            st.subheader("Persisted evaluation artifacts")
            st.dataframe(by_kind["evaluation"], use_container_width=True, hide_index=True)
        if not experiments and not by_kind["evaluation"]:
            st.info("No measured evaluation artifacts are registered. Metrics are not invented.")

    with tabs[5]:
        features = store.query(
            "SELECT feature, COUNT(*) AS observations, AVG(value) AS mean, "
            "STDDEV_POP(value) AS stddev "
            "FROM drift_observations GROUP BY feature ORDER BY feature"
        )
        if not features:
            st.info("No drift observations have been recorded.")
        else:
            st.dataframe(features, use_container_width=True, hide_index=True)
            selected = st.selectbox("Feature", [row["feature"] for row in features])
            history: list[dict[str, Any]] = store.query(
                """SELECT observed_at, value FROM drift_observations
                   WHERE feature = ? ORDER BY observed_at""",
                [selected],
            )
            st.line_chart(history, x="observed_at", y="value")
        if drift_events:
            st.subheader("Persistent alerts")
            st.dataframe(drift_events, use_container_width=True, hide_index=True)
        if by_kind["drift"]:
            st.subheader("Drift experiments")
            st.caption("Rows marked synthetic are illustrative experiments.")
            st.dataframe(by_kind["drift"], use_container_width=True, hide_index=True)

    with tabs[6]:
        benchmarks = store.query(
            "SELECT * FROM benchmark_records ORDER BY created_at DESC LIMIT 100"
        )
        if benchmarks:
            st.dataframe(benchmarks, use_container_width=True, hide_index=True)
        if by_kind["benchmark"]:
            st.subheader("Benchmark artifacts")
            st.dataframe(by_kind["benchmark"], use_container_width=True, hide_index=True)
        if by_kind["export"]:
            st.subheader("Export records")
            st.dataframe(by_kind["export"], use_container_width=True, hide_index=True)
        if not benchmarks and not by_kind["benchmark"]:
            st.info("No benchmark has been run; status is NOT_RUN.")

    with tabs[7]:
        if by_kind["cost"]:
            st.caption("Computed from persisted labeled threshold sweeps.")
            st.dataframe(by_kind["cost"], use_container_width=True, hide_index=True)
        else:
            st.info(
                "No labeled cost optimization is persisted. Unlabeled operational "
                "records are not used to invent an estimate."
            )

    with tabs[8]:
        status_columns = st.columns(4)
        status_columns[0].metric("Database", "connected")
        status_columns[1].metric("Mode", os.getenv("EDGESIGHT_MODE", "demo"))
        status_columns[2].metric("Broker", "not probed by dashboard")
        status_columns[3].metric("Pending reviews", len(pending))
        model_rows = store.query(
            "SELECT * FROM model_information ORDER BY registered_at DESC LIMIT 10"
        )
        if model_rows:
            st.dataframe(model_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No deployed model record is registered.")


if __name__ == "__main__":
    render()
