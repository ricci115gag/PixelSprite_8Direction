from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DIRECTIONS = ("down", "left", "right", "up")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class DirectionPair:
    character_id: str
    source_path: Path
    target_path: Path
    source_direction: str
    target_direction: str


class DirectionPairDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        directions: tuple[str, ...] = DEFAULT_DIRECTIONS,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.directions = directions
        self.samples = self._load_samples()

    def _load_samples(self) -> list[DirectionPair]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root_dir}")

        samples: list[DirectionPair] = []
        character_dirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
        for character_dir in character_dirs:
            images_by_direction: dict[str, list[Path]] = {}
            for image_path in sorted(character_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                direction = self._parse_direction(image_path.name)
                if direction:
                    images_by_direction.setdefault(direction, []).append(image_path)

            for source_direction, source_paths in images_by_direction.items():
                for target_direction, target_paths in images_by_direction.items():
                    if source_direction == target_direction:
                        continue
                    for source_path in source_paths:
                        for target_path in target_paths:
                            samples.append(
                                DirectionPair(
                                    character_id=character_dir.name,
                                    source_path=source_path,
                                    target_path=target_path,
                                    source_direction=source_direction,
                                    target_direction=target_direction,
                                )
                            )
        return samples

    def _parse_direction(self, filename: str) -> str | None:
        stem = Path(filename).stem.lower()
        for direction in sorted(self.directions, key=len, reverse=True):
            if stem.endswith(f"_{direction}") or f"_{direction}_" in stem:
                return direction
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        source_image = Image.open(sample.source_path).convert("RGBA")
        target_image = Image.open(sample.target_path).convert("RGBA")
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return {
            "source_image": source_image,
            "target_image": target_image,
            "source_direction": sample.source_direction,
            "target_direction": sample.target_direction,
            "character_id": sample.character_id,
            "source_path": str(sample.source_path),
            "target_path": str(sample.target_path),
            "base_image": source_image,
            "base_img_path": str(sample.source_path),
            "target_img_path": str(sample.target_path),
        }


SpriteDataLoader = DirectionPairDataset
