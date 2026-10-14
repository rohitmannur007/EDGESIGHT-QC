"""Evaluation pipeline with validation-only calibration and explicit missing metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from edgesight.decision import QualityRouter
from edgesight.schemas import AnomalyResult, Detection, InferenceMode, InferenceResult

from .bootstrap import bootstrap_metric
from .metrics import binary_metrics, detection_metrics

UNMEASURED = "UNMEASURED"
TARGET_FALSE_ALARM_RATE = 0.05
Predictor = Callable[[Mapping[str, Any]], float | None]


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value.get("records", value) if isinstance(value, dict) else value
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("prediction input must be a JSON list or JSONL objects")
    return records


def _label(record: Mapping[str, Any]) -> int:
    if "label" in record:
        return int(bool(record["label"]))
    if "is_anomaly" in record:
        return int(bool(record["is_anomaly"]))
    return int(str(record.get("defect_type", "good")) != "good")


def _assignment(record: Mapping[str, Any]) -> str:
    return str(record.get("assignment", record.get("split", "")))


def _score(record: Mapping[str, Any], family: str) -> float | None:
    value = record.get(f"{family}_score")
    if value is None and family == "detector":
        detections = record.get("detections")
        if isinstance(detections, list):
            values = [
                float(item.get("confidence", 0.0)) for item in detections if isinstance(item, dict)
            ]
            value = max(values, default=0.0)
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def _good_scores(records: list[dict[str, Any]], family: str) -> list[float]:
    return [
        float(score)
        for record in records
        if _assignment(record) in {"validation", "val"}
        and _label(record) == 0
        and (score := _score(record, family)) is not None
    ]


def _threshold_for_false_alarm(scores: list[float], target: float) -> float | str:
    """Lowest threshold whose validation-good false-alarm rate stays within budget."""

    if not scores:
        return UNMEASURED
    candidates = sorted({float(score) for score in scores})
    candidates.append(float(np.nextafter(max(candidates), np.inf)))
    valid: list[float] = []
    for threshold in candidates:
        if not math.isfinite(threshold):
            continue
        false_alarms = sum(score >= threshold for score in scores) / len(scores)
        if false_alarms <= target + 1e-12:
            valid.append(threshold)
    return float(min(valid)) if valid else UNMEASURED


def _candidate_thresholds(records: list[dict[str, Any]], family: str) -> list[float]:
    scores = [
        float(score)
        for record in records
        if _assignment(record) in {"validation", "val", "known_validation"}
        and (score := _score(record, family)) is not None
    ]
    if not scores:
        return []
    candidates = sorted(set(scores))
    candidates.append(float(np.nextafter(max(candidates), np.inf)))
    return [value for value in candidates if math.isfinite(value)]


def _router_decision(
    record: Mapping[str, Any],
    system: str,
    detector_threshold: float | str,
    anomaly_threshold: float | str,
) -> str | None:
    detector_score = _score(record, "detector")
    anomaly_score = _score(record, "anomaly")
    detector_on = system in {"detector", "combined"} and detector_score is not None
    anomaly_on = system in {"anomaly", "combined"} and anomaly_score is not None
    if system == "detector" and (not detector_on or not isinstance(detector_threshold, float)):
        return None
    if system == "anomaly" and (not anomaly_on or not isinstance(anomaly_threshold, float)):
        return None
    if system == "combined" and not (
        detector_on
        and anomaly_on
        and isinstance(detector_threshold, float)
        and isinstance(anomaly_threshold, float)
    ):
        return None
    detections: tuple[Detection, ...] = ()
    if detector_score is not None and detector_on:
        confidence = min(1.0, max(0.0, float(detector_score)))
        detections = (Detection(0, confidence, (0.0, 0.0, 1.0, 1.0)),)
    anomaly = (
        AnomalyResult(float(anomaly_score)) if anomaly_score is not None and anomaly_on else None
    )
    result = InferenceResult(
        None,
        InferenceMode.REAL,
        detections,
        anomaly,
        metadata={"configured_models": {"detector": detector_on, "anomaly": anomaly_on}},
    )
    acceptance = float(detector_threshold) if detector_on else 1.0
    router = QualityRouter(
        detection_acceptance=acceptance,
        # Equal bounds leave only the selected acceptance threshold active.
        detection_rejection=acceptance,
        anomaly_threshold=float(anomaly_threshold) if anomaly_on else math.inf,
        human_review_enabled=True,
    )
    return router.decide(result).decision.value


def _operating_rate(
    records: list[dict[str, Any]],
    system: str,
    detector_threshold: float | str,
    anomaly_threshold: float | str,
    *,
    assignments: set[str],
    positive: Callable[[str], bool],
) -> tuple[float | None, int]:
    decisions = []
    for record in records:
        if _assignment(record) not in assignments:
            continue
        goods_only = assignments <= {"validation", "val"}
        if goods_only and _label(record) != 0:
            continue
        decision = _router_decision(record, system, detector_threshold, anomaly_threshold)
        if decision is not None:
            decisions.append(decision)
    if not decisions:
        return None, 0
    return sum(1 for decision in decisions if positive(decision)) / len(decisions), len(decisions)


def _select_single(records: list[dict[str, Any]], system: str, target: float) -> dict[str, Any]:
    scores = _good_scores(records, system)
    threshold = _threshold_for_false_alarm(scores, target)
    detector_threshold = threshold if system == "detector" else math.inf
    anomaly_threshold = threshold if system == "anomaly" else math.inf
    false_alarm_rate, good_count = _operating_rate(
        records,
        system,
        detector_threshold,
        anomaly_threshold,
        assignments={"validation", "val"},
        positive=lambda decision: decision != "PASS",
    )
    known_count = sum(
        1
        for record in records
        if _assignment(record) == "known_validation" and _score(record, system) is not None
    )
    return {
        "value": threshold,
        "source": "validation_false_alarm_budget",
        "target_false_alarm_rate": target,
        "validation_false_alarm_rate": (
            false_alarm_rate if false_alarm_rate is not None else UNMEASURED
        ),
        "validation_good_count": good_count,
        "validation_known_count": known_count,
        "constraint_met": false_alarm_rate is not None and false_alarm_rate <= target + 1e-12,
    }


def _select_combined(records: list[dict[str, Any]], target: float) -> dict[str, Any]:
    detector_candidates = _candidate_thresholds(records, "detector")
    anomaly_candidates = _candidate_thresholds(records, "anomaly")
    good_count = len(
        [
            record
            for record in records
            if _assignment(record) in {"validation", "val"}
            and _label(record) == 0
            and _score(record, "detector") is not None
            and _score(record, "anomaly") is not None
        ]
    )
    known_count = sum(
        1
        for record in records
        if _assignment(record) == "known_validation"
        and _score(record, "detector") is not None
        and _score(record, "anomaly") is not None
    )
    empty = {
        "detector": UNMEASURED,
        "anomaly": UNMEASURED,
        "source": "validation_router_false_alarm_budget",
        "target_false_alarm_rate": target,
        "validation_false_alarm_rate": UNMEASURED,
        "validation_known_recall": UNMEASURED,
        "validation_good_count": good_count,
        "validation_known_count": known_count,
        "constraint_met": False,
    }
    if not detector_candidates or not anomaly_candidates:
        return empty
    best: tuple[tuple[float, float, float, float], float, float, float, float] | None = None
    for detector_threshold in detector_candidates:
        for anomaly_threshold in anomaly_candidates:
            false_alarm_rate, _ = _operating_rate(
                records,
                "combined",
                detector_threshold,
                anomaly_threshold,
                assignments={"validation", "val"},
                positive=lambda decision: decision != "PASS",
            )
            if false_alarm_rate is None or false_alarm_rate > target + 1e-12:
                continue
            known_recall, _ = _operating_rate(
                records,
                "combined",
                detector_threshold,
                anomaly_threshold,
                assignments={"known_validation"},
                positive=lambda decision: decision == "KNOWN_DEFECT",
            )
            recall = 0.0 if known_recall is None else known_recall
            rank = (recall, false_alarm_rate, -detector_threshold, -anomaly_threshold)
            if best is None or rank > best[0]:
                best = (rank, detector_threshold, anomaly_threshold, false_alarm_rate, recall)
    if best is None:
        return empty
    return {
        "detector": best[1],
        "anomaly": best[2],
        "source": "validation_router_false_alarm_budget",
        "target_false_alarm_rate": target,
        "validation_false_alarm_rate": best[3],
        "validation_known_recall": best[4],
        "validation_good_count": good_count,
        "validation_known_count": known_count,
        "constraint_met": True,
    }


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return UNMEASURED
    return value


def _family_metrics(
    records: list[dict[str, Any]],
    family: str,
    threshold: float | str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    measured = [
        record
        for record in records
        if _assignment(record) in {"known_test", "good_test", "unseen_test", "test"}
        and _score(record, family) is not None
    ]
    if threshold == UNMEASURED or not measured:
        return {
            "threshold": threshold,
            "classification": UNMEASURED,
            "roc_auc_ci": UNMEASURED,
            "localization": UNMEASURED,
        }
    labels = [_label(record) for record in measured]
    scores = [float(_score(record, family)) for record in measured]
    result = {
        key: _finite(value)
        for key, value in binary_metrics(labels, scores, threshold=float(threshold)).items()
    }
    ci: Any = UNMEASURED
    if len(set(labels)) == 2 and bootstrap_samples > 0:
        pairs = list(zip(labels, scores, strict=True))
        try:
            interval = bootstrap_metric(
                pairs,
                lambda sample: float(
                    binary_metrics(
                        [item[0] for item in sample],
                        [item[1] for item in sample],
                        threshold=float(threshold),
                    )["roc_auc"]
                ),
                resamples=bootstrap_samples,
                seed=seed,
            )
            ci = {
                "estimate": _finite(interval.estimate),
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence": interval.confidence,
                "successful_resamples": interval.successful_resamples,
            }
        except ValueError:
            pass
    localization: Any = UNMEASURED
    if family == "detector":
        with suppress(ValueError):
            localization = {
                key: _finite(value)
                for key, value in detection_metrics(
                    measured, confidence_threshold=float(threshold)
                ).items()
            }
    return {
        "threshold": threshold,
        "classification": result,
        "roc_auc_ci": ci,
        "localization": localization,
    }


def _decision_metrics(
    records: list[dict[str, Any]],
    family: str,
    threshold: Any,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    key = f"{family}_decision"
    measured = [
        record
        for record in records
        if _assignment(record) in {"known_test", "good_test", "unseen_test", "test"}
        and isinstance(record.get(key), str)
    ]
    if not measured:
        return {
            "threshold": threshold,
            "classification": UNMEASURED,
            "roc_auc_ci": UNMEASURED,
            "localization": UNMEASURED,
        }
    labels = [_label(record) for record in measured]
    scores = [1.0 if record[key] != "PASS" else 0.0 for record in measured]
    result = {
        key: _finite(value) for key, value in binary_metrics(labels, scores, threshold=0.5).items()
    }
    return {
        "threshold": threshold,
        "classification": result,
        "roc_auc_ci": UNMEASURED,
        "localization": UNMEASURED,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def _open_set_summary(
    records: list[dict[str, Any]],
    family: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    groups = {
        "known_defect_recall": ("known_test", lambda decision: decision == "KNOWN_DEFECT"),
        "unseen_defect_recall": ("unseen_test", lambda decision: decision == "UNKNOWN_DEFECT"),
        "false_alarm_rate": ("good_test", lambda decision: decision != "PASS"),
    }
    output: dict[str, Any] = {}
    decision_key = f"{family}_decision"
    for index, (name, (assignment, predicate)) in enumerate(groups.items()):
        group = [record for record in records if _assignment(record) == assignment]
        outcomes = [
            float(predicate(decision))
            for record in group
            if isinstance(decision := record.get(decision_key), str)
        ]
        if not outcomes:
            output[name] = {
                "estimate": UNMEASURED,
                "count": 0,
                "group_count": len(group),
                "ci95": UNMEASURED,
            }
            continue
        estimate = sum(outcomes) / len(outcomes)
        interval: Any = UNMEASURED
        if len(outcomes) > 1 and bootstrap_samples > 0:
            ci = bootstrap_metric(
                outcomes,
                lambda sample: sum(sample) / len(sample),
                resamples=bootstrap_samples,
                seed=seed + index,
            )
            interval = {"lower": ci.lower, "upper": ci.upper}
        output[name] = {
            "estimate": estimate,
            "count": len(outcomes),
            "group_count": len(group),
            "ci95": interval,
        }
    return output


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    predictors: Mapping[str, Predictor] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate records; injected predictors may supply detector/anomaly scores."""

    prepared = [dict(record) for record in records]
    if not prepared:
        raise ValueError("evaluation requires at least one record")
    for family, predictor in (predictors or {}).items():
        if family not in {"detector", "anomaly"}:
            raise ValueError(f"unsupported predictor family: {family}")
        for record in prepared:
            value = predictor(record)
            if value is not None:
                record[f"{family}_score"] = float(value)
    thresholds = {
        "detector": _select_single(prepared, "detector", TARGET_FALSE_ALARM_RATE),
        "anomaly": _select_single(prepared, "anomaly", TARGET_FALSE_ALARM_RATE),
        "combined": _select_combined(prepared, TARGET_FALSE_ALARM_RATE),
    }
    detector_threshold = thresholds["detector"]["value"]
    anomaly_threshold = thresholds["anomaly"]["value"]
    combined = thresholds["combined"]
    for record in prepared:
        record["detector_decision"] = _router_decision(
            record, "detector", detector_threshold, math.inf
        )
        record["anomaly_decision"] = _router_decision(
            record, "anomaly", math.inf, anomaly_threshold
        )
        record["combined_decision"] = _router_decision(
            record, "combined", combined["detector"], combined["anomaly"]
        )
    test_records = [
        record
        for record in prepared
        if _assignment(record) in {"known_test", "good_test", "unseen_test", "test"}
    ]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in test_records:
        categories[str(record.get("category", "unknown"))].append(record)
    return {
        "protocol": "router-false-alarm-matched-open-set-v1",
        "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
        "calibration_assignments": ["validation", "known_validation", "val"],
        "test_assignments": ["good_test", "known_test", "unseen_test", "test"],
        "thresholds": thresholds,
        "overall": {
            "detector": _family_metrics(
                prepared,
                "detector",
                detector_threshold,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "anomaly": _family_metrics(
                prepared,
                "anomaly",
                anomaly_threshold,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "combined": _decision_metrics(
                prepared,
                "combined",
                combined,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        },
        "open_set_summary": {
            family: _open_set_summary(
                prepared,
                family,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for family in ("detector", "anomaly", "combined")
        },
        "per_category": {
            category: {
                "detector": _family_metrics(
                    records_for_category,
                    "detector",
                    detector_threshold,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                "anomaly": _family_metrics(
                    records_for_category,
                    "anomaly",
                    anomaly_threshold,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                "combined": _decision_metrics(
                    records_for_category,
                    "combined",
                    combined,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
            }
            for category, records_for_category in sorted(categories.items())
        },
        "record_count": len(prepared),
        "test_record_count": len(test_records),
        "_records": prepared,
    }


def write_evaluation(
    result: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write machine-readable, Markdown, and optional plotted evaluation artifacts."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in result.items() if key != "_records"}
    json_path = output / "evaluation.json"
    json_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Evaluation report",
        "",
        f"Protocol: `{public['protocol']}`",
        f"Test records: {public['test_record_count']}",
        "",
        "## Overall",
        "",
    ]
    for family, values in public["overall"].items():
        classification = values["classification"]
        if isinstance(classification, dict):
            lines.append(
                f"- **{family}**: F1={classification['f1']:.4f}, "
                f"ROC AUC={classification['roc_auc']}"
            )
        else:
            lines.append(f"- **{family}**: {UNMEASURED}")
    lines.extend(("", "## Known, unseen, and good-test operating point", ""))
    for family, summary in public["open_set_summary"].items():
        lines.append(f"### {family}")
        for name, metric in summary.items():
            estimate = metric["estimate"]
            rendered = f"{estimate:.4f}" if isinstance(estimate, float) else UNMEASURED
            interval = metric.get("ci95")
            confidence = ""
            if isinstance(interval, dict):
                confidence = f", 95% CI [{interval['lower']:.4f}, {interval['upper']:.4f}]"
            lines.append(f"- {name}: {rendered}{confidence} (n={metric['count']})")
        lines.append("")
    lines.extend(("", "## Per category", ""))
    for category, values in public["per_category"].items():
        lines.append(f"### {category}")
        for family, metrics in values.items():
            classification = metrics["classification"]
            summary = (
                f"F1={classification['f1']:.4f}" if isinstance(classification, dict) else UNMEASURED
            )
            lines.append(f"- {family}: {summary}")
        lines.append("")
    markdown_path = output / "evaluation.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    category_directory = output / "categories"
    category_directory.mkdir(exist_ok=True)
    for category, values in public["per_category"].items():
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_" for character in category
        )
        (category_directory / f"{safe_name}.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        category_lines = [f"# {category}", ""]
        for family, metrics in values.items():
            classification = metrics["classification"]
            summary = (
                f"F1={classification['f1']:.4f}, ROC AUC={classification['roc_auc']}"
                if isinstance(classification, dict)
                else UNMEASURED
            )
            category_lines.append(f"- **{family}**: {summary}")
        (category_directory / f"{safe_name}.md").write_text(
            "\n".join(category_lines) + "\n", encoding="utf-8"
        )
    plot_status: str = UNMEASURED
    try:
        import matplotlib.pyplot as plt

        records = list(result.get("_records", []))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plotted = False
        for axis, family in zip(axes, ("detector", "anomaly", "combined"), strict=True):
            normal = [_score(record, family) for record in records if _label(record) == 0]
            anomalous = [_score(record, family) for record in records if _label(record) == 1]
            normal = [value for value in normal if value is not None]
            anomalous = [value for value in anomalous if value is not None]
            if normal:
                axis.hist(normal, bins=15, alpha=0.6, label="normal")
                plotted = True
            if anomalous:
                axis.hist(anomalous, bins=15, alpha=0.6, label="anomaly")
                plotted = True
            axis.set_title(family)
            axis.set_xlabel("score")
            if normal or anomalous:
                axis.legend()
        if plotted:
            fig.tight_layout()
            plot_path = output / "score_distributions.png"
            fig.savefig(plot_path, dpi=150)
            plot_status = str(plot_path)
        plt.close(fig)
    except ImportError:
        pass
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "plots": plot_status,
    }
