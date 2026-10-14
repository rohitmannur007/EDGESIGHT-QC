"""Dependency-light binary quality-control metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negative_count = int((~positives).sum())
    positive_count = int(positives.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    )


def binary_metrics(
    labels: Iterable[int], scores: Iterable[float], *, threshold: float = 0.5
) -> dict[str, float | int]:
    labels_array = np.asarray(list(labels), dtype=int)
    scores_array = np.asarray(list(scores), dtype=float)
    if labels_array.shape != scores_array.shape or labels_array.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("labels must be binary")
    predictions = scores_array >= threshold
    tp = int(np.sum(predictions & (labels_array == 1)))
    fp = int(np.sum(predictions & (labels_array == 0)))
    tn = int(np.sum(~predictions & (labels_array == 0)))
    fn = int(np.sum(~predictions & (labels_array == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "count": len(labels_array),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(labels_array),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": _auc(labels_array, scores_array),
    }


def _box(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    values = item.get("xyxy", item.get("box"))
    if not isinstance(values, list | tuple) or len(values) != 4:
        raise ValueError("detection boxes must contain four xyxy coordinates")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def detection_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute class-aware micro precision/recall and interpolated AP@IoU."""

    ground_truth: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    predictions: list[tuple[float, int, int, tuple[float, float, float, float]]] = []
    annotated_records = 0
    for image_index, record in enumerate(records):
        truth_items = record.get("ground_truth_boxes", record.get("ground_truth"))
        if not isinstance(truth_items, list):
            continue
        annotated_records += 1
        for item in truth_items:
            if isinstance(item, Mapping):
                class_id = int(item.get("class_id", 0))
                ground_truth[(image_index, class_id)].append(_box(item))
        detection_items = record.get("detections", [])
        if isinstance(detection_items, list):
            for item in detection_items:
                if isinstance(item, Mapping):
                    predictions.append(
                        (
                            float(item.get("confidence", item.get("score", 0.0))),
                            image_index,
                            int(item.get("class_id", 0)),
                            _box(item),
                        )
                    )
    if not annotated_records:
        raise ValueError("no records contain ground-truth boxes")
    predictions.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    total_truth = sum(len(items) for items in ground_truth.values())
    matched: dict[tuple[int, int], set[int]] = defaultdict(set)
    true_flags: list[int] = []
    for _, image_index, class_id, predicted_box in predictions:
        key = (image_index, class_id)
        candidates = ground_truth[key]
        available = [
            (index, _iou(predicted_box, target))
            for index, target in enumerate(candidates)
            if index not in matched[key]
        ]
        best = max(available, key=lambda item: item[1], default=(-1, 0.0))
        is_true = best[1] >= iou_threshold
        true_flags.append(int(is_true))
        if is_true:
            matched[key].add(best[0])
    cumulative_true = np.cumsum(true_flags)
    cumulative_false = np.cumsum([1 - value for value in true_flags])
    recalls = cumulative_true / total_truth if total_truth else np.zeros(len(predictions))
    precisions = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1)
    if total_truth and len(predictions):
        recall_points = np.r_[0.0, recalls, 1.0]
        precision_points = np.r_[1.0, precisions, 0.0]
        precision_points = np.maximum.accumulate(precision_points[::-1])[::-1]
        changes = np.where(recall_points[1:] != recall_points[:-1])[0]
        average_precision = float(
            np.sum(
                (recall_points[changes + 1] - recall_points[changes])
                * precision_points[changes + 1]
            )
        )
    else:
        average_precision = 0.0 if total_truth else float("nan")
    selected = [
        index
        for index, prediction in enumerate(predictions)
        if prediction[0] >= confidence_threshold
    ]
    true_positive = sum(true_flags[index] for index in selected)
    false_positive = len(selected) - true_positive
    false_negative = total_truth - true_positive
    return {
        "annotated_images": annotated_records,
        "ground_truth_boxes": total_truth,
        "predicted_boxes": len(selected),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / len(selected) if selected else 0.0,
        "recall": true_positive / total_truth if total_truth else 0.0,
        "map50": average_precision,
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
    }
