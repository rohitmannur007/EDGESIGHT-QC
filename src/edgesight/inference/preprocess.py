"""Decode and prepare one image for every configured inference model."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    """Raised when an input is not a safe, supported image."""


class PreprocessingUnavailable(RuntimeError):
    """Raised when production tensor preprocessing cannot be provided."""


@dataclass(frozen=True)
class PreprocessedImage:
    """Representations derived from one image decode.

    ``detector`` is an RGB uint8 HWC array. The YOLO adapter converts it to BGR
    before Ultralytics sees a NumPy image. ``tensor`` is an RGB float32 NCHW
    tensor normalized with the same ImageNet mean and standard deviation used
    in anomaly training.
    """

    pil: Image.Image
    detector: np.ndarray
    tensor: Any
    sha256: str
    original_size: tuple[int, int]
    tensor_size: tuple[int, int]
    format: str


class ImagePreprocessor:
    VERSION = "rgb-imagenet-normalized-v2"
    SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})
    IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int] = 256,
        max_pixels: int = 25_000_000,
        torch_module: Any | None = None,
    ) -> None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions")
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_pixels = max_pixels
        self._torch_module = torch_module

    def __call__(self, payload: bytes) -> PreprocessedImage:
        return self.process(payload)

    def process(self, payload: bytes, *, require_tensor: bool = True) -> PreprocessedImage:
        if not payload:
            raise InvalidImage("image payload is empty")
        try:
            # Image.open is deliberately called once. load() validates and
            # materializes the image before the in-memory stream is closed.
            with Image.open(io.BytesIO(payload)) as decoded:
                image_format = decoded.format
                if image_format not in self.SUPPORTED_FORMATS:
                    raise InvalidImage(f"unsupported image format: {image_format}")
                width, height = decoded.size
                if width < 1 or height < 1 or width * height > self.max_pixels:
                    raise InvalidImage("image dimensions exceed configured safety limits")
                decoded.load()
                rgb = decoded.convert("RGB")
        except InvalidImage:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImage("payload is not a valid image") from exc

        detector = np.asarray(rgb, dtype=np.uint8)
        tensor_image = rgb.resize(self.image_size, Image.Resampling.BILINEAR)
        tensor_array = np.asarray(tensor_image, dtype=np.float32) / 255.0
        tensor_array = (tensor_array - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = self._to_tensor(
            np.transpose(tensor_array, (2, 0, 1))[None, ...],
            required=require_tensor,
        )
        return PreprocessedImage(
            pil=rgb,
            detector=detector,
            tensor=tensor,
            sha256=hashlib.sha256(payload).hexdigest(),
            original_size=(width, height),
            tensor_size=self.image_size,
            format=str(image_format),
        )

    def _to_tensor(self, value: np.ndarray, *, required: bool = True) -> Any:
        torch = self._torch_module
        if torch is None:
            try:
                import torch
            except ImportError as exc:
                if required:
                    raise PreprocessingUnavailable(
                        "real anomaly inference requires PyTorch"
                    ) from exc
                return np.ascontiguousarray(value)
        return torch.from_numpy(np.ascontiguousarray(value))
