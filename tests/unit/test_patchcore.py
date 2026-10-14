import pytest

torch = pytest.importorskip("torch")

from edgesight.anomaly import PatchCore  # noqa: E402


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer2 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.layer3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), torch.nn.Conv2d(4, 5, 3, padding=1)
        )

    def forward(self, value):
        return self.layer3(self.layer2(value))


def test_patchcore_fit_predict_and_round_trip(tmp_path) -> None:
    torch.manual_seed(4)
    training = torch.rand(2, 3, 16, 16)
    model = PatchCore(backbone=TinyBackbone(), memory_samples=20, distance_chunk_size=7).fit(
        [training]
    )
    scores, maps = model.predict(training[:1])
    assert scores.shape == (1,)
    assert maps.shape == (1, 16, 16)
    assert model.memory_bank.shape[0] == 20

    artifact = tmp_path / "patchcore.pt"
    model.save(artifact)
    restored = PatchCore.load(artifact, backbone=TinyBackbone())
    restored_scores, _ = restored.predict(training[:1])
    assert torch.allclose(scores, restored_scores)
