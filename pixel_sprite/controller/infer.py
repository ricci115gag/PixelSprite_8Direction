from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from pixel_sprite.config import load_config, resolve_project_path
from pixel_sprite.model.factory import get_model_class
from pixel_sprite.model.image_ops import apply_palette_and_binarize


DIRECTIONS = ("down", "left", "right", "up")


def load_model(
    config: dict, 
    device: torch.device, 
    timestamp: str | None = None, 
    version: int = 2
) -> nn.Module:
    checkpoints_path = config["misc"].get("checkpoints_path", "./checkpoints")
    checkpoints_dir = resolve_project_path(checkpoints_path) / f"v{version}"
    
    if timestamp and timestamp != "latest":
        model_file = checkpoints_dir / f"generator_{timestamp}.safetensors"
    else:
        model_file = _find_latest_generator_checkpoint(checkpoints_dir)
        
    if model_file is None or not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist or checkpoints folder is empty: {model_file}")
        
    GeneratorClass = get_model_class(version, "generator")
    model = GeneratorClass(**config["model"]).to(device)
    state_dict = load_file(str(model_file), device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _find_latest_generator_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None
    gen_files = list(checkpoints_dir.glob("generator_*.safetensors"))
    if not gen_files:
        return None
    # Sort checkpoints by filename reverse (newest timestamp first)
    gen_files.sort(key=lambda x: x.name, reverse=True)
    return gen_files[0]


def predict_file(
    image_path: str | Path,
    source_direction: str,
    target_direction: str,
    model: nn.Module,
    device: torch.device,
) -> Path:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGBA")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    output = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    output_image = Image.fromarray((output * 255).astype(np.uint8), mode="RGBA")
    output_image = apply_palette_and_binarize(output_image, image)
    output_path = image_path.with_name(f"{image_path.stem}_{target_direction}{image_path.suffix}")
    output_image.save(output_path, format="PNG")
    return output_path


def load_default_model(version: int = 2) -> tuple[nn.Module, torch.device]:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(config, device, version=version), device
