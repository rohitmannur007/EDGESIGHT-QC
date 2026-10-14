import pytest

torch = pytest.importorskip("torch")

from edgesight.anomaly import PatchCore  # noqa: E402


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = torch.nn.Conv2d(3, 3, 1)

    def forward(self, value):
        return self.layer2(value)


def test_patchcore_bounded_reservoir_calibration_and_metadata(tmp_path):
    torch.manual_seed(1)
    state = TinyBackbone().state_dict()
    first_backbone = TinyBackbone()
    first_backbone.load_state_dict(state)
    second_backbone = TinyBackbone()
    second_backbone.load_state_dict(state)
    batches = [torch.rand(2, 3, 8, 8) for _ in range(3)]
    calibration = [torch.rand(2, 3, 8, 8)]
    kwargs = {
        "layers": ("layer2",),
        "memory_samples": 7,
        "candidate_reservoir_size": 11,
        "coreset_strategy": "random",
        "category": "capsule",
        "preprocess": {"resize": [8, 8]},
        "seed": 9,
    }
    first = PatchCore(backbone=first_backbone, **kwargs).fit(
        batches, calibration_batches=calibration
    )
    second = PatchCore(backbone=second_backbone, **kwargs).fit(batches)
    assert first.memory_bank.shape[0] == 7
    assert torch.equal(first.memory_bank, second.memory_bank)
    assert first.calibration["source"] == "normal_validation"
    artifact = tmp_path / "capsule.pt"
    first.save(artifact)
    restored = PatchCore.load(artifact, backbone=TinyBackbone())
    assert restored.category == "capsule"
    assert restored.preprocess == {"resize": [8, 8]}
    assert restored.calibration["sample_count"] == 2
