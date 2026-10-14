"""A compact PatchCore-style nearest-neighbour anomaly detector."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PatchCore requires PyTorch") from exc
    return torch


class PatchCore:
    """Frozen CNN features, sampled memory bank, and chunked nearest distances.

    The default backbone is created without pretrained weights, avoiding hidden
    downloads. Production callers should inject a locally initialized backbone
    or load weights under their own artifact policy.
    """

    def __init__(
        self,
        *,
        backbone: Any | None = None,
        backbone_name: str = "resnet18",
        layers: Sequence[str] = ("layer2", "layer3"),
        memory_samples: int = 2048,
        distance_chunk_size: int = 1024,
        device: str = "cpu",
        seed: int = 42,
        pretrained: bool = False,
        coreset_strategy: str = "greedy",
        candidate_reservoir_size: int | None = None,
        projection_dimension: int = 128,
        category: str | None = None,
        preprocess: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        artifact_references: dict[str, str] | None = None,
    ) -> None:
        torch = _torch()
        if memory_samples < 1 or distance_chunk_size < 1:
            raise ValueError("memory_samples and distance_chunk_size must be positive")
        if coreset_strategy not in {"greedy", "random"}:
            raise ValueError("coreset_strategy must be 'greedy' or 'random'")
        if candidate_reservoir_size is not None and candidate_reservoir_size < memory_samples:
            raise ValueError("candidate_reservoir_size cannot be smaller than memory_samples")
        if backbone is None:
            try:
                import torchvision.models as models
            except ImportError as exc:
                raise RuntimeError("Default PatchCore backbones require torchvision") from exc
            factory = getattr(models, backbone_name, None)
            if factory is None:
                raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")
            backbone = factory(weights="DEFAULT" if pretrained else None)
        self.backbone = backbone.to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone_name = backbone_name
        self.layers = tuple(layers)
        self.memory_samples = memory_samples
        self.distance_chunk_size = distance_chunk_size
        self.device = device
        self.seed = seed
        self.pretrained = pretrained
        self.coreset_strategy = coreset_strategy
        self.candidate_reservoir_size = candidate_reservoir_size or memory_samples * 4
        self.projection_dimension = projection_dimension
        self.category = category
        self.preprocess = dict(preprocess or {})
        self.calibration = dict(calibration or {})
        self.artifact_references = dict(artifact_references or {})
        self.memory_bank: Any | None = None
        self._features: dict[str, Any] = {}
        module_map = dict(self.backbone.named_modules())
        missing = set(self.layers) - set(module_map)
        if missing:
            raise ValueError(f"Backbone layers not found: {sorted(missing)}")
        self._hooks = [
            module_map[name].register_forward_hook(self._capture(name)) for name in self.layers
        ]
        self._torch_version = torch.__version__

    def _capture(self, name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._features[name] = output.detach()

        return hook

    def _extract(self, images: Any) -> tuple[Any, tuple[int, int]]:
        torch = _torch()
        functional = torch.nn.functional
        images = images.to(self.device)
        self._features.clear()
        with torch.inference_mode():
            self.backbone(images)
        if not self._features:
            raise RuntimeError("backbone produced no hooked features")
        target_size = self._features[self.layers[0]].shape[-2:]
        resized = [
            feature
            if feature.shape[-2:] == target_size
            else functional.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            for feature in (self._features[name] for name in self.layers)
        ]
        combined = torch.cat(resized, dim=1)
        patches = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
        patches = functional.normalize(patches, dim=1)
        return patches, (int(target_size[0]), int(target_size[1]))

    def _select_coreset(self, candidates: Any, generator: Any) -> Any:
        """Select a deterministic random or approximate farthest-first coreset."""

        torch = _torch()
        count = min(self.memory_samples, len(candidates))
        if len(candidates) <= count:
            return candidates
        if self.coreset_strategy == "random":
            return candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        projected = candidates
        if candidates.shape[1] > self.projection_dimension:
            projection = (
                torch.randn(
                    candidates.shape[1],
                    self.projection_dimension,
                    generator=generator,
                    dtype=candidates.dtype,
                )
                / self.projection_dimension**0.5
            )
            projected = candidates @ projection
        first = int(torch.randint(len(candidates), (1,), generator=generator).item())
        selected = [first]
        minimum = torch.cdist(projected, projected[first : first + 1]).squeeze(1)
        minimum[first] = -1
        for _ in range(1, count):
            index = int(torch.argmax(minimum).item())
            selected.append(index)
            distance = torch.cdist(projected, projected[index : index + 1]).squeeze(1)
            minimum = torch.minimum(minimum, distance)
            minimum[index] = -1
        return candidates[torch.tensor(selected, dtype=torch.long)]

    def fit(
        self,
        batches: Iterable[Any],
        *,
        calibration_batches: Iterable[Any] | None = None,
        calibration_quantile: float = 0.99,
    ) -> PatchCore:
        torch = _torch()
        if not 0 < calibration_quantile <= 1:
            raise ValueError("calibration_quantile must be in (0, 1]")
        generator = torch.Generator().manual_seed(self.seed)
        reservoir: Any | None = None
        priorities: Any | None = None
        observed = 0
        for batch in batches:
            features = self._extract(batch)[0].cpu()
            if not len(features):
                continue
            observed += len(features)
            batch_priorities = torch.rand(len(features), generator=generator)
            reservoir = features if reservoir is None else torch.cat((reservoir, features))
            priorities = (
                batch_priorities
                if priorities is None
                else torch.cat((priorities, batch_priorities))
            )
            if len(reservoir) > self.candidate_reservoir_size:
                keep = torch.topk(priorities, self.candidate_reservoir_size).indices
                reservoir = reservoir[keep]
                priorities = priorities[keep]
        if reservoir is None:
            raise ValueError("fit requires at least one non-empty batch")
        self.memory_bank = self._select_coreset(reservoir, generator).contiguous()
        self.artifact_references.setdefault("fit_patch_count", str(observed))
        self.artifact_references.setdefault("candidate_patch_count", str(len(reservoir)))
        if calibration_batches is not None:
            scores: list[float] = []
            for batch in calibration_batches:
                batch_scores, _ = self.predict(batch)
                scores.extend(float(value) for value in batch_scores.tolist())
            if not scores:
                raise ValueError("calibration_batches contained no images")
            threshold = float(torch.quantile(torch.tensor(scores), calibration_quantile))
            self.calibration = {
                "threshold": threshold,
                "quantile": calibration_quantile,
                "sample_count": len(scores),
                "source": "normal_validation",
            }
        return self

    def _nearest_distances(self, patches: Any) -> Any:
        torch = _torch()
        if self.memory_bank is None:
            raise RuntimeError("PatchCore is not fitted; memory bank is empty")
        memory = self.memory_bank.to(self.device)
        chunks = []
        for start in range(0, len(patches), self.distance_chunk_size):
            query = patches[start : start + self.distance_chunk_size]
            try:
                chunks.append(torch.cdist(query, memory).min(dim=1).values)
            except RuntimeError:
                # Some MPS releases do not support every cdist shape/dtype.
                # Distance search is independent of feature extraction, so a
                # local CPU fallback avoids moving the backbone repeatedly.
                chunks.append(torch.cdist(query.cpu(), self.memory_bank).min(dim=1).values)
        return torch.cat(chunks)

    def predict(self, images: Any) -> tuple[Any, Any]:
        """Return per-image max scores and input-resolution anomaly maps."""

        torch = _torch()
        patches, feature_size = self._extract(images)
        distances = self._nearest_distances(patches)
        batch_size = int(images.shape[0])
        maps = distances.reshape(batch_size, 1, *feature_size)
        maps = torch.nn.functional.interpolate(
            maps, size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        scores = maps.flatten(1).max(dim=1).values
        return scores.cpu(), maps.cpu()

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save an unfitted PatchCore model")
        _torch().save(
            {
                "format_version": 2,
                "backbone_name": self.backbone_name,
                "layers": self.layers,
                "memory_samples": self.memory_samples,
                "distance_chunk_size": self.distance_chunk_size,
                "seed": self.seed,
                "pretrained": self.pretrained,
                "coreset_strategy": self.coreset_strategy,
                "candidate_reservoir_size": self.candidate_reservoir_size,
                "projection_dimension": self.projection_dimension,
                "category": self.category,
                "preprocess": self.preprocess,
                "calibration": self.calibration,
                "artifact_references": self.artifact_references,
                "memory_bank": self.memory_bank.cpu(),
                "backbone_state": self.backbone.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(
        cls, path: str | Path, *, backbone: Any | None = None, device: str = "cpu"
    ) -> PatchCore:
        torch = _torch()
        try:
            state = torch.load(Path(path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(Path(path), map_location="cpu")
        if state.get("format_version") not in {1, 2}:
            raise ValueError("Unsupported PatchCore artifact format")
        model = cls(
            backbone=backbone,
            backbone_name=state["backbone_name"],
            layers=state["layers"],
            memory_samples=state["memory_samples"],
            distance_chunk_size=state["distance_chunk_size"],
            device=device,
            seed=state["seed"],
            pretrained=False,
            coreset_strategy=state.get("coreset_strategy", "random"),
            candidate_reservoir_size=state.get("candidate_reservoir_size"),
            projection_dimension=state.get("projection_dimension", 128),
            category=state.get("category"),
            preprocess=state.get("preprocess"),
            calibration=state.get("calibration"),
            artifact_references=state.get("artifact_references"),
        )
        model.backbone.load_state_dict(state["backbone_state"])
        model.memory_bank = state["memory_bank"]
        return model

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
