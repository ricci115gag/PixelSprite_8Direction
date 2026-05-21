from __future__ import annotations

import random

from torch.utils.data import Subset

from pixel_sprite.model.dataset import DirectionPairDataset


def split_dataset_by_character(
    dataset: DirectionPairDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, Subset]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1.")

    character_ids = sorted({sample.character_id for sample in dataset.samples})
    random.Random(seed).shuffle(character_ids)
    n_test = round(len(character_ids) * test_ratio)
    n_val = round(len(character_ids) * val_ratio)
    test_chars = set(character_ids[:n_test])
    val_chars = set(character_ids[n_test : n_test + n_val])
    train_chars = set(character_ids[n_test + n_val :])
    return {
        "train": Subset(dataset, _indices_for_characters(dataset, train_chars)),
        "val": Subset(dataset, _indices_for_characters(dataset, val_chars)),
        "test": Subset(dataset, _indices_for_characters(dataset, test_chars)),
    }


def subset_character_ids(subset: Subset) -> set[str]:
    dataset = subset.dataset
    if not isinstance(dataset, DirectionPairDataset):
        return set()
    return {dataset.samples[index].character_id for index in subset.indices}


def _indices_for_characters(
    dataset: DirectionPairDataset,
    character_ids: set[str],
) -> list[int]:
    return [
        index
        for index, sample in enumerate(dataset.samples)
        if sample.character_id in character_ids
    ]

