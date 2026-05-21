from __future__ import annotations

from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEFAULT_CROP_BOXES = {
    "down": (208, 64, 240, 112),
    "left": (208, 128, 240, 176),
}


def extract_sprites(
    root_dir: str | Path,
    output_dir: str | Path,
    crop_boxes: dict[str, tuple[int, int, int, int]] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    crop_boxes = crop_boxes or DEFAULT_CROP_BOXES
    written: list[Path] = []

    for source_file in tqdm(find_octopath_ui_textures(root_dir), desc="Extracting sprites"):
        character_id = source_file.parents[1].name
        for direction, crop_box in crop_boxes.items():
            output_path = output_dir / character_id / f"{source_file.stem}_{direction}{source_file.suffix}"
            if dry_run:
                written.append(output_path)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(source_file).convert("RGBA").crop(crop_box).save(output_path)
            written.append(output_path)
    return written


def find_octopath_ui_textures(root_dir: str | Path) -> list[Path]:
    files: list[Path] = []
    for character_dir in sorted(Path(root_dir).iterdir()):
        if not character_dir.is_dir() or not character_dir.name.startswith(("Evc", "Npc", "Ply")):
            continue
        textures_dir = character_dir / "Textures"
        if textures_dir.is_dir():
            files.extend(sorted(textures_dir.glob("*UI_cl.png")))
    return files


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract Octopath character UI sprites.")
    parser.add_argument("root_dir", help="Octopath Character/Resource directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outputs = extract_sprites(
        args.root_dir,
        args.output_dir,
        crop_boxes=DEFAULT_CROP_BOXES,
        dry_run=args.dry_run,
    )
    print(f"Prepared {len(outputs)} extracted sprite files.")


if __name__ == "__main__":
    main()


