"""Leakage-resistant deterministic dataset splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[T, ...]
    validation: tuple[T, ...]
    test: tuple[T, ...]

    def assert_disjoint(self, key: Callable[[T], Hashable] | None = None) -> None:
        key = key or (lambda item: item)  # type: ignore[assignment]
        groups = [{key(item) for item in part} for part in (self.train, self.validation, self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("split leakage detected")


def _stable_order(value: Hashable, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value!r}".encode()).hexdigest()


def stratified_group_split(
    items: Sequence[T] | Iterable[T],
    *,
    label: Callable[[T], Hashable],
    group: Callable[[T], Hashable],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Split whole groups while approximately preserving label proportions.

    A group is assigned exactly once. A group containing multiple labels is
    rejected because silently selecting one label makes stratification unsafe.
    """

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    grouped: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        grouped[group(item)].append(item)
    by_label: dict[Hashable, list[tuple[Hashable, list[T]]]] = defaultdict(list)
    for group_id, members in grouped.items():
        labels = {label(member) for member in members}
        if len(labels) != 1:
            raise ValueError(f"group {group_id!r} spans labels: {sorted(map(str, labels))}")
        by_label[next(iter(labels))].append((group_id, members))

    output: list[list[T]] = [[], [], []]
    for groups in by_label.values():
        ordered = sorted(groups, key=lambda pair: _stable_order(pair[0], seed))
        count = len(ordered)
        test_count = int(round(count * test_fraction))
        validation_count = int(round(count * validation_fraction))
        if test_count + validation_count > count:
            validation_count = max(0, count - test_count)
        for index, (_, members) in enumerate(ordered):
            bucket = 2 if index < test_count else 1 if index < test_count + validation_count else 0
            output[bucket].extend(members)
    result = DatasetSplit(*(tuple(part) for part in output))
    result.assert_disjoint(group)
    return result


def mvtec_held_out_split(
    samples: Sequence[T],
    *,
    split: Callable[[T], str],
    identity: Callable[[T], Hashable],
) -> DatasetSplit:
    """Preserve official test isolation; derive validation from training only."""

    train_pool = [item for item in samples if split(item) == "train"]
    held_out = [item for item in samples if split(item) == "test"]
    derived = stratified_group_split(
        train_pool,
        label=lambda _: "normal",
        group=identity,
        validation_fraction=0.15,
        test_fraction=0.0,
    )
    result = DatasetSplit(derived.train, derived.validation, tuple(held_out))
    result.assert_disjoint(identity)
    return result
