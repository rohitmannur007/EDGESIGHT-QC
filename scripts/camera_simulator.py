#!/usr/bin/env python3
"""Publish bounded, self-contained camera frames to the `frames` topic."""

from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from edgesight.streaming.broker import KafkaProducerAdapter


def encode(path: Path, perturbation: str, quality: int) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if perturbation == "brightness":
        image = ImageEnhance.Brightness(image).enhance(1.4)
    elif perturbation == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--brokers", default="localhost:19092")
    parser.add_argument("--topic", default="frames")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--cameras", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--category", default="demo")
    parser.add_argument("--perturbation", choices=["none", "brightness", "blur"], default="none")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    args = parser.parse_args()
    images = sorted(
        path
        for path in args.directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        parser.error(f"no supported images in {args.directory}")
    producer = KafkaProducerAdapter(
        {
            "bootstrap.servers": args.brokers,
            "message.max.bytes": args.max_bytes + 64 * 1024,
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    sent = failed = 0
    try:
        for cycle in range(args.repeat):
            for index, path in enumerate(images):
                payload = encode(path, args.perturbation, args.jpeg_quality)
                if len(payload) > args.max_bytes:
                    failed += 1
                    continue
                message_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{cycle}:{path.resolve()}"))
                try:
                    producer.send(
                        args.topic,
                        {
                            "message_id": message_id,
                            "payload": base64.b64encode(payload).decode("ascii"),
                            "encoding": "base64",
                            "filename": path.name,
                            "metadata": {
                                "frame_id": message_id,
                                "camera_id": f"camera-{index % max(1, args.cameras) + 1}",
                                "captured_at": datetime.now(UTC).isoformat(),
                                "category": args.category,
                                "perturbation": args.perturbation,
                            },
                        },
                        key=message_id,
                    )
                    sent += 1
                except Exception:
                    failed += 1
                if args.rate > 0:
                    time.sleep(1 / args.rate)
    finally:
        undelivered = producer.flush()
    print({"sent": sent, "failed": failed, "undelivered": undelivered})
    return 0 if failed == 0 and undelivered == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
