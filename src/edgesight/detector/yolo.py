"""Optional Ultralytics YOLO training and inference adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from edgesight.schemas import Detection


class MissingModelError(FileNotFoundError):
    """Raised when real inference is requested without model weights."""


def _ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLO operations; install `ultralytics`"
        ) from exc
    return YOLO


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise MissingModelError(f"YOLO model weights not found: {self.model_path}")
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = _ultralytics_yolo()(str(self.model_path))
        return self._model

    @staticmethod
    def _model_input(image: Any) -> Any:
        """Convert RGB NumPy images to BGR, which Ultralytics expects for arrays."""

        array = np.asarray(image)
        if (
            isinstance(image, np.ndarray)
            and array.ndim == 3
            and array.shape[-1] == 3
            and np.issubdtype(array.dtype, np.number)
        ):
            return np.ascontiguousarray(array[..., ::-1])
        return image

    def predict(self, image: Any) -> tuple[Detection, ...]:
        results = self._load().predict(
            source=self._model_input(image),
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for coordinates, class_value, confidence in zip(
                xyxy, classes, confidences, strict=True
            ):
                class_id = int(class_value)
                detections.append(
                    Detection(
                        class_id=class_id,
                        confidence=float(confidence),
                        xyxy=tuple(float(value) for value in coordinates),
                        label=str(names.get(class_id, class_id)),
                    )
                )
        return tuple(detections)


def train_yolo(
    data: str | Path,
    *,
    base_model: str | Path,
    output_directory: str | Path,
    epochs: int = 50,
    image_size: int = 640,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Train YOLO explicitly; pretrained weights are never downloaded implicitly."""

    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO dataset configuration not found: {data_path}")
    model_path = Path(base_model)
    if not model_path.is_file():
        raise MissingModelError(
            f"Base YOLO model not found: {model_path}. Download it explicitly first."
        )
    model = _ultralytics_yolo()(str(model_path))
    return model.train(
        data=str(data_path),
        project=str(output_directory),
        epochs=epochs,
        imgsz=image_size,
        seed=seed,
        **kwargs,
    )
