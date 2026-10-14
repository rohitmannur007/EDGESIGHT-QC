"""Command-line adapters for real training and Kafka-compatible streaming."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def _train(config_path: Path, family: str) -> int:
    config = _load(config_path)
    if config.get("dataset", {}).get("license_acknowledged") is not True:
        raise RuntimeError("dataset license must be acquired and acknowledged before training")
    if family == "detector":
        from edgesight.detector.yolo import train_yolo

        detector = config.get("detector", {})
        train_yolo(
            detector.get("data", "data/processed/yolo/data.yaml"),
            base_model=detector.get("base_model", "models/detector/yolo11n.pt"),
            output_directory=detector.get("output_dir", "models/detector/runs"),
            epochs=int(detector.get("epochs", 50)),
            image_size=int(detector.get("image_size", 640)),
            seed=int(config.get("seed", 42)),
            batch=int(detector.get("batch_size", 4)),
            device=detector.get("device", "mps"),
        )
        return 0
    if family == "anomaly":
        import torch
        from PIL import Image
        from torchvision import transforms

        from edgesight.anomaly import PatchCore

        anomaly = config.get("anomaly", {})
        normal_root = Path(anomaly.get("normal_image_dir", "data/processed/anomaly/normal_train"))
        image_paths = sorted(
            path
            for path in normal_root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not image_paths:
            raise FileNotFoundError(f"no normal fitting images found under {normal_root}")
        maximum_images = int(anomaly.get("maximum_fit_images", 200))
        image_paths = image_paths[:maximum_images]
        size = int(anomaly.get("image_size", 256))
        preprocess = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        tensors = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
        batch_size = int(anomaly.get("batch_size", 4))
        batches = [
            torch.stack(tensors[start : start + batch_size])
            for start in range(0, len(tensors), batch_size)
        ]
        requested_device = anomaly.get("device", "mps")
        device = "mps" if requested_device == "mps" and torch.backends.mps.is_available() else "cpu"
        model = PatchCore(
            backbone_name=anomaly.get("backbone", "resnet18"),
            layers=tuple(anomaly.get("layers", ("layer2", "layer3"))),
            memory_samples=int(anomaly.get("maximum_bank_size", 10_000)),
            distance_chunk_size=int(anomaly.get("distance_chunk_size", 512)),
            device=device,
            seed=int(config.get("seed", 42)),
            pretrained=bool(anomaly.get("pretrained_backbone", True)),
        ).fit(batches)
        output = Path(anomaly.get("output", "models/anomaly/patchcore.pt"))
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        print(f"saved {output} from {len(image_paths)} normal images on {device}")
        return 0
    raise ValueError(f"unsupported family {family!r}; choose detector or anomaly")


def _monitoring_loop(config: dict[str, Any], store: Any) -> Any:
    """Build the configured drift loop so streaming can persist alerts."""

    monitoring = config.get("monitoring") or {}
    baseline_path = Path(str(monitoring.get("baseline", "configs/drift_baseline.json")))
    if not baseline_path.is_file():
        raise FileNotFoundError(
            "streaming drift monitoring requires a baseline JSON at "
            f"{baseline_path}. Record clean validation observations before streaming."
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"drift baseline must be an object: {baseline_path}")
    drift_path = Path(str(monitoring.get("config", "configs/drift.yaml")))
    drift = _load(drift_path) if drift_path.is_file() else {}
    from edgesight.monitoring import MonitoringLoop

    return MonitoringLoop(
        baseline,
        store=store,
        baseline_name=str(drift.get("baseline_version", "configured")),
        baseline_version=str(drift.get("baseline_version", "v1")),
        window_size=int(drift.get("window_size", 100)),
        consecutive_windows=int(drift.get("consecutive_windows", 3)),
        psi_threshold=float(drift.get("psi_threshold", 0.2)),
        ks_threshold=float(drift.get("ks_threshold", 0.2)),
    )


def _stream(config_path: Path) -> int:
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:
        raise RuntimeError("install the stream extra: uv sync --extra stream") from exc
    from edgesight.api.main import build_pipeline, load_settings
    from edgesight.persistence import EdgeSightStore
    from edgesight.streaming.broker import KafkaProducerAdapter
    from edgesight.streaming.worker import StreamingWorker

    config = _load(config_path)
    broker = config.get("broker", config.get("stream", {}))
    servers = os.getenv(
        "EDGESIGHT_KAFKA_BROKERS",
        str(broker.get("bootstrap_servers", broker.get("brokers", "localhost:19092"))),
    )
    if servers.startswith("${"):
        servers = "localhost:19092"
    app_config = Path(config.get("app_config", "configs/app.yaml"))
    settings = load_settings(app_config)
    settings["mode"] = os.getenv(
        "EDGESIGHT_MODE", str(config.get("mode", settings.get("mode", "demo")))
    )
    anomaly_models = (settings.get("models") or {}).get("anomaly_models") or {}
    if settings["mode"] == "real" and isinstance(anomaly_models, dict) and anomaly_models:
        pipeline: Any = {}
        for category in anomaly_models:
            category_settings = dict(settings)
            category_settings["category"] = category
            pipeline[category] = build_pipeline(category_settings)
    else:
        pipeline = build_pipeline(settings)
    store = EdgeSightStore(config.get("database", {}).get("path", "results/edgesight.duckdb"))
    producer = KafkaProducerAdapter({"bootstrap.servers": servers})
    consumer = Consumer(
        {
            "bootstrap.servers": servers,
            "group.id": broker.get("group_id", broker.get("consumer_group", "edgesight-worker")),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    worker = StreamingWorker(
        pipeline,
        store,
        producer,
        monitoring_loop=_monitoring_loop(config, store),
        output_topic=os.getenv(
            "EDGESIGHT_OUTPUT_TOPIC",
            str(broker.get("results_topic", broker.get("output_topic", "results"))),
        ),
        dlq_topic=os.getenv(
            "EDGESIGHT_DLQ_TOPIC",
            str(broker.get("dlq_topic", broker.get("dead_letter_topic", "frames.dlq"))),
        ),
        events_topic=os.getenv("EDGESIGHT_MONITORING_TOPIC", "monitoring.events"),
    )
    running = True

    def stop(*_: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    consumer.subscribe([broker.get("frames_topic", broker.get("input_topic", "frames"))])
    try:
        while running:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                continue
            try:
                raw = json.loads(message.value())
                outcome = worker.handle_sync(raw)
                if outcome != "retry":
                    consumer.commit(message=message, asynchronous=False)
            except Exception:
                # Malformed values are routed by the worker when representable.
                worker.handle_sync({"message_id": str(message.offset()), "payload": b""})
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
        producer.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="edgesight")
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--family", required=True)
    stream = subcommands.add_parser("stream")
    stream.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        return _train(args.config, args.family)
    return _stream(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
