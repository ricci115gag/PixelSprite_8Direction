from __future__ import annotations

from collections import Counter, defaultdict

from pixel_sprite.model.dataset import DirectionPairDataset, IMAGE_SUFFIXES


def build_dataset_report(dataset: DirectionPairDataset) -> dict[str, object]:
    directions_by_character: dict[str, set[str]] = defaultdict(set)
    pair_directions: Counter[str] = Counter()
    image_files = 0

    for character_dir in sorted(p for p in dataset.root_dir.iterdir() if p.is_dir()):
        image_files += sum(
            1
            for path in character_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    for sample in dataset.samples:
        directions_by_character[sample.character_id].update(
            [sample.source_direction, sample.target_direction]
        )
        pair_directions[f"{sample.source_direction}->{sample.target_direction}"] += 1

    direction_counts = Counter()
    for directions in directions_by_character.values():
        direction_counts.update(directions)

    return {
        "root_dir": str(dataset.root_dir),
        "characters": len(directions_by_character),
        "image_files": image_files,
        "pairs": len(dataset.samples),
        "direction_counts": dict(sorted(direction_counts.items())),
        "pair_direction_counts": dict(sorted(pair_directions.items())),
        "characters_missing_pairs": [
            character_id
            for character_id, directions in sorted(directions_by_character.items())
            if len(directions) < 2
        ],
    }


def format_dataset_report(report: dict[str, object]) -> str:
    lines = [
        f"root_dir: {report['root_dir']}",
        f"characters: {report['characters']}",
        f"image_files: {report['image_files']}",
        f"pairs: {report['pairs']}",
        "direction_counts:",
    ]
    for direction, count in report["direction_counts"].items():
        lines.append(f"  {direction}: {count}")
    lines.append("pair_direction_counts:")
    for pair_name, count in report["pair_direction_counts"].items():
        lines.append(f"  {pair_name}: {count}")
    lines.append(f"characters_missing_pairs: {len(report['characters_missing_pairs'])}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    from pixel_sprite.config import load_config, resolve_project_path
    
    parser = argparse.ArgumentParser(description="Report sprite direction dataset stats.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = DirectionPairDataset(resolve_project_path(config["train"]["dataset_path"]))
    report = build_dataset_report(dataset)
    print(format_dataset_report(report))

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


