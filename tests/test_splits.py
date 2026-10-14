from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edgesight.data.splits import mvtec_held_out_split, stratified_group_split


@dataclass(frozen=True)
class Item:
    identity: str
    label: str
    source_split: str = "train"


def test_group_split_is_repeatable_and_leak_free() -> None:
    items = [
        Item(f"{label}-{group}-{view}", label)
        for label in ("good", "bad")
        for group in range(10)
        for view in range(2)
    ]

    def group(item: Item) -> str:
        return item.identity.rsplit("-", 1)[0]

    first = stratified_group_split(items, label=lambda x: x.label, group=group, seed=5)
    second = stratified_group_split(items, label=lambda x: x.label, group=group, seed=5)
    assert first == second
    first.assert_disjoint(group)
    assert {item.label for item in first.test} == {"good", "bad"}


def test_official_test_samples_never_enter_training() -> None:
    samples = [
        *(Item(f"train-{index}", "good") for index in range(10)),
        *(Item(f"held-{index}", "bad", "test") for index in range(3)),
    ]
    result = mvtec_held_out_split(
        samples, split=lambda item: item.source_split, identity=lambda item: item.identity
    )
    assert {item.identity for item in result.test} == {"held-0", "held-1", "held-2"}
    assert all(item.source_split == "train" for item in result.train + result.validation)
