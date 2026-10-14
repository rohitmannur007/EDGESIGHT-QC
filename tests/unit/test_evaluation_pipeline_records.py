import json

from edgesight.evaluation import (
    UNMEASURED,
    detection_metrics,
    evaluate_records,
    write_evaluation,
)


def test_evaluation_calibrates_only_validation_and_reports_missing(tmp_path):
    records = [
        {"assignment": "validation", "label": 0, "anomaly_score": 0.1, "category": "a"},
        {
            "assignment": "known_validation",
            "label": 1,
            "anomaly_score": 0.8,
            "category": "a",
        },
        {"assignment": "good_test", "label": 0, "anomaly_score": 0.2, "category": "a"},
        {"assignment": "unseen_test", "label": 1, "anomaly_score": 0.9, "category": "a"},
    ]
    result = evaluate_records(records, bootstrap_samples=20, seed=3)
    anomaly_threshold = result["thresholds"]["anomaly"]["value"]
    assert result["thresholds"]["anomaly"]["constraint_met"] is True
    assert anomaly_threshold > 0.1
    assert result["open_set_summary"]["anomaly"]["unseen_defect_recall"]["estimate"] == 1.0
    assert result["open_set_summary"]["anomaly"]["unseen_defect_recall"]["count"] == 1
    assert result["open_set_summary"]["anomaly"]["false_alarm_rate"]["count"] == 1
    assert result["overall"]["detector"]["classification"] == UNMEASURED
    assert "combined_score" not in result["_records"][0]
    artifacts = write_evaluation(result, tmp_path)
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["test_record_count"] == 2
    assert artifacts["markdown"].endswith("evaluation.md")


def test_evaluation_accepts_injected_predictors():
    records = [
        {"assignment": "validation", "label": 0, "value": 0.1},
        {"assignment": "known_validation", "label": 1, "value": 0.9},
        {"assignment": "good_test", "label": 0, "value": 0.2},
        {"assignment": "known_test", "label": 1, "value": 0.8},
    ]
    result = evaluate_records(
        records,
        predictors={"detector": lambda record: float(record["value"])},
        bootstrap_samples=5,
    )
    assert result["thresholds"]["detector"]["validation_false_alarm_rate"] <= 0.05
    summary = result["open_set_summary"]["detector"]
    assert summary["known_defect_recall"]["estimate"] == 1.0
    assert summary["known_defect_recall"]["count"] == 1
    assert summary["false_alarm_rate"]["count"] == 1


def test_detector_localization_metrics_match_by_class_and_iou():
    metrics = detection_metrics(
        [
            {
                "ground_truth_boxes": [{"class_id": 2, "xyxy": [0, 0, 10, 10]}],
                "detections": [
                    {"class_id": 2, "confidence": 0.9, "xyxy": [0, 0, 10, 10]},
                    {"class_id": 1, "confidence": 0.8, "xyxy": [0, 0, 10, 10]},
                ],
            }
        ],
        confidence_threshold=0.5,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["map50"] == 1.0


def test_open_set_report_uses_router_decisions_not_averaged_scores():
    records = [
        {
            "assignment": "validation",
            "label": 0,
            "detector_score": 0.05,
            "anomaly_score": 0.05,
        },
        {
            "assignment": "known_validation",
            "label": 1,
            "detector_score": 0.9,
            "anomaly_score": 0.05,
        },
        {
            "assignment": "known_test",
            "label": 1,
            "detector_score": 0.95,
            "anomaly_score": 0.05,
        },
        {
            "assignment": "known_test",
            "label": 1,
            "detector_score": 0.95,
            "anomaly_score": 0.95,
        },
        {
            "assignment": "unseen_test",
            "label": 1,
            "detector_score": 0.05,
            "anomaly_score": 0.9,
        },
        {
            "assignment": "good_test",
            "label": 0,
            "detector_score": 0.04,
            "anomaly_score": 0.04,
        },
    ]
    result = evaluate_records(records, bootstrap_samples=10, seed=1)
    combined = result["open_set_summary"]["combined"]
    assert combined["known_defect_recall"]["estimate"] == 0.5
    assert combined["known_defect_recall"]["count"] == 2
    assert combined["unseen_defect_recall"]["estimate"] == 1.0
    assert combined["false_alarm_rate"]["estimate"] == 0.0
    assert result["thresholds"]["combined"]["constraint_met"] is True
    decisions = {
        record["assignment"]: record["combined_decision"]
        for record in result["_records"]
        if record["assignment"] == "unseen_test"
        or (record["assignment"] == "known_test" and record["anomaly_score"] == 0.95)
    }
    assert decisions["unseen_test"] == "UNKNOWN_DEFECT"
    assert "HUMAN_REVIEW" in decisions.values()
