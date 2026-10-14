"""Convert binary masks into YOLO detection labels."""

from __future__ import annotations

from collections import deque

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    foreground = np.asarray(mask).astype(bool)
    if foreground.ndim == 3:
        foreground = foreground.any(axis=2)
    if foreground.ndim != 2:
        raise ValueError("mask must be two-dimensional or HWC")
    visited = np.zeros_like(foreground)
    components: list[np.ndarray] = []
    height, width = foreground.shape
    for row, col in np.argwhere(foreground):
        if visited[row, col]:
            continue
        queue = deque([(int(row), int(col))])
        visited[row, col] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and foreground[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        components.append(np.asarray(pixels, dtype=np.int64))
    return components


def mask_to_yolo(
    mask: np.ndarray,
    *,
    class_id: int = 0,
    min_area: int = 1,
) -> list[str]:
    """Return one normalized YOLO bounding-box row per connected component."""

    if class_id < 0:
        raise ValueError("class_id must be non-negative")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    height, width = np.asarray(mask).shape[:2]
    rows: list[str] = []
    for component in _components(mask):
        if len(component) < min_area:
            continue
        y1, x1 = component.min(axis=0)
        y2, x2 = component.max(axis=0) + 1
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return rows
