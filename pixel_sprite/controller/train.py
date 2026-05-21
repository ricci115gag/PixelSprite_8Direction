from __future__ import annotations
 
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
 
from pixel_sprite.config import load_config, resolve_project_path
from pixel_sprite.controller.split import split_dataset_by_character, subset_character_ids
from pixel_sprite.model.dataset import DirectionPairDataset
from pixel_sprite.model.discriminator import Discriminator
from pixel_sprite.model.generator import UNetGenerator


def train_model(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_g = UNetGenerator(
        in_channels=config["model"].get("in_channels", 4),
        base_channels=config["model"].get("base_channels", 32),
    ).to(device)

    model_d = Discriminator(
        in_channels=config["model"].get("in_channels", 4) * 2,  # 8 channels
        base_channels=config["model"].get("base_channels", 32),
    ).to(device)

    dataset = DirectionPairDataset(
        resolve_project_path(config["train"]["dataset_path"]),
        transform=transforms.Compose(
            [
                transforms.Resize((config["train"]["h"], config["train"]["w"])),
                transforms.ToTensor(),
            ]
        ),
    )
    splits = split_dataset_by_character(
        dataset,
        val_ratio=config["train"].get("val_ratio", 0.1),
        test_ratio=config["train"].get("test_ratio", 0.0),
        seed=config["train"].get("seed", 42),
    )
    train_loader = DataLoader(
        splits["train"],
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"].get("num_workers", 0),
    )
    print(
        f"Split by character: train={len(subset_character_ids(splits['train']))} chars/"
        f"{len(splits['train'])} pairs, val={len(subset_character_ids(splits['val']))} chars/"
        f"{len(splits['val'])} pairs"
    )

    checkpoints_dir = resolve_project_path(config["misc"].get("checkpoints_path", "./checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    csv_file = checkpoints_dir / "loss.csv"

    start_epoch, global_step, train_g_losses, train_d_losses, val_g_losses = _load_checkpoint(
        model_g, model_d, checkpoints_dir, csv_file, device
    )

    tb_log_dir = resolve_project_path("runs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    criterion_recon = nn.L1Loss().to(device)
    criterion_gan = nn.BCELoss().to(device)

    lr_g = config["train"].get("lr_g", config["train"].get("lr", 0.0002))
    lr_d = config["train"].get("lr_d", config["train"].get("lr", 0.0002))
    lambda_adv = config["train"].get("lambda_adv", 0.01)
    lambda_fm = config["train"].get("lambda_fm", 0.1)
    lambda_alpha = config["train"].get("lambda_alpha", 0.1)

    optimizer_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(0.5, 0.999))

    end_epoch = start_epoch + config["train"]["epochs"]
    sample_every = config["train"].get("sample_every", 10)

    last_img_time = 0.0

    try:
        for epoch in range(start_epoch, end_epoch):
            train_loss_g, train_loss_d, global_step, last_img_time = _train_epoch(
                model_g,
                model_d,
                train_loader,
                criterion_recon,
                criterion_gan,
                optimizer_g,
                optimizer_d,
                lambda_adv,
                lambda_fm,
                lambda_alpha,
                device,
                epoch,
                end_epoch,
                writer,
                global_step,
                val_loader,
                last_img_time,
            )
            val_loss_g = _validate(model_g, val_loader, criterion_recon, device)

            train_g_losses.append(train_loss_g)
            train_d_losses.append(train_loss_d)
            val_g_losses.append(val_loss_g)

            print(
                f"Epoch [{epoch + 1}/{end_epoch}], "
                f"loss_g: {train_loss_g:.6f}, loss_d: {train_loss_d:.6f}, val_loss_g: {val_loss_g:.6f}"
            )

            # Save epoch checkpoint safely
            _save_checkpoint(model_g, model_d, checkpoints_dir, global_step)

            _write_loss_csv(csv_file, train_g_losses, train_d_losses, val_g_losses)

            if sample_every and (epoch + 1) % sample_every == 0:
                sample_loader = val_loader if len(val_loader) > 0 else train_loader
                _save_sample_predictions(
                    model_g, sample_loader, device, checkpoints_dir / "samples" / f"epoch_{epoch + 1:04d}.png"
                )
    except KeyboardInterrupt:
        print("\n[Warning] Training interrupted by user. Saving current checkpoint safely...")
        _save_checkpoint(model_g, model_d, checkpoints_dir, global_step)
        _write_loss_csv(csv_file, train_g_losses, train_d_losses, val_g_losses)
        print("Progress saved successfully. Exiting.")
    finally:
        writer.close()


def _load_checkpoint(
    model_g: UNetGenerator,
    model_d: Discriminator,
    checkpoints_dir: Path,
    csv_file: Path,
    device: torch.device,
) -> tuple[int, int, list[float], list[float], list[float]]:
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0, [], [], []

    gen_files = list(checkpoints_dir.glob("generator_*.safetensors"))
    if not gen_files:
        return 0, 0, [], [], []

    # Sort checkpoints by filename reverse (newest timestamp first)
    gen_files.sort(key=lambda x: x.name, reverse=True)
    latest_gen = gen_files[0]
    
    timestamp = latest_gen.name.replace("generator_", "").replace(".safetensors", "")
    corresponding_d = checkpoints_dir / f"discriminator_{timestamp}.safetensors"

    global_step = 0
    try:
        state_dict_g = load_file(str(latest_gen), device=str(device))
        model_g.load_state_dict(state_dict_g)
        print(f"Loaded Generator from {latest_gen}")
        
        with safe_open(str(latest_gen), framework="pt") as f:
            meta = f.metadata()
            if meta:
                global_step = int(meta.get("global_step", "0"))
                print(f"Restored training step to {global_step}")
    except Exception as e:
        print(f"Error loading Generator checkpoint: {e}")

    if corresponding_d.exists():
        try:
            state_dict_d = load_file(str(corresponding_d), device=str(device))
            model_d.load_state_dict(state_dict_d)
            print(f"Loaded Discriminator from {corresponding_d}")
        except Exception as e:
            print(f"Error loading Discriminator checkpoint: {e}")
    else:
        print(f"No corresponding Discriminator checkpoint found at {corresponding_d}")

    if not csv_file.exists():
        return 0, global_step, [], [], []

    try:
        df = pd.read_csv(csv_file)
        if "train_loss_g" in df and "train_loss_d" in df and "val_loss_g" in df:
            train_g_losses = df["train_loss_g"].tolist()
            train_d_losses = df["train_loss_d"].tolist()
            val_g_losses = df["val_loss_g"].tolist()
        else:
            train_g_losses = df["train_loss"].tolist() if "train_loss" in df else df["loss"].tolist()
            val_g_losses = df["val_loss"].tolist() if "val_loss" in df else [float("nan")] * len(train_g_losses)
            train_d_losses = [float("nan")] * len(train_g_losses)
        return len(train_g_losses), global_step, train_g_losses, train_d_losses, val_g_losses
    except Exception:
        return 0, global_step, [], [], []


def _train_epoch(
    model_g: UNetGenerator,
    model_d: Discriminator,
    dataloader: DataLoader,
    criterion_recon: nn.Module,
    criterion_gan: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    lambda_adv: float,
    lambda_fm: float,
    lambda_alpha: float,
    device: torch.device,
    epoch: int,
    end_epoch: int,
    writer: SummaryWriter,
    global_step: int,
    val_loader: DataLoader,
    last_img_time: float,
) -> tuple[float, float, int, float]:
    model_g.train()
    model_d.train()
    running_loss_g = 0.0
    running_loss_d = 0.0

    progress = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch + 1}/{end_epoch}")
    for batch in progress:
        source_images = batch["source_image"].to(device)
        target_images = batch["target_image"].to(device)
        b_size = source_images.size(0)

        global_step += 1

        # -----------------
        # Train Discriminator
        # -----------------
        optimizer_d.zero_grad()

        # Real pair: D(source, target)
        real_pair = torch.cat([source_images, target_images], dim=1)
        pred_real, feat_real = model_d(real_pair)
        loss_d_real = criterion_gan(pred_real, torch.ones(b_size, 1, device=device))

        # Fake pair: D(source, G(source))
        fake_targets = model_g(source_images)
        fake_pair = torch.cat([source_images, fake_targets.detach()], dim=1)
        pred_fake, _ = model_d(fake_pair)
        loss_d_fake = criterion_gan(pred_fake, torch.zeros(b_size, 1, device=device))

        # Real/Fake output discriminator loss
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_g.zero_grad()

        # Reconstruction loss (L1)
        loss_recon = criterion_recon(fake_targets, target_images)

        # Adversarial loss: G wants D to think fake_pair is Real
        fake_pair_for_g = torch.cat([source_images, fake_targets], dim=1)
        pred_fake_g, feat_fake = model_d(fake_pair_for_g)
        loss_g_gan = criterion_gan(pred_fake_g, torch.ones(b_size, 1, device=device))

        # Feature Matching Loss
        loss_fm = sum(torch.mean(torch.abs(f_r.detach() - f_f)) for f_r, f_f in zip(feat_real, feat_fake))

        # Alpha Binarization Loss
        alpha = fake_targets[:, 3:4, :, :]
        loss_alpha = torch.mean(alpha * (1.0 - alpha))

        loss_g = loss_recon + lambda_adv * loss_g_gan + lambda_fm * loss_fm + lambda_alpha * loss_alpha
        loss_g.backward()
        optimizer_g.step()

        running_loss_g += loss_g.item()
        running_loss_d += loss_d.item()

        # TensorBoard logging
        writer.add_scalar("Loss/Generator", loss_g.item(), global_step)
        writer.add_scalar("Loss/Discriminator", loss_d.item(), global_step)

        # Log prediction image to TensorBoard every 5 seconds
        if time.time() - last_img_time >= 5.0:
            _log_sample_predictions_to_tb(model_g, val_loader, device, writer, global_step)
            last_img_time = time.time()

        progress.set_postfix(
            {
                "loss_g": f"{loss_g.item():.4f}",
                "loss_d": f"{loss_d.item():.4f}",
                "recon": f"{loss_recon.item():.4f}",
                "fm": f"{loss_fm.item():.4f}",
                "alpha": f"{loss_alpha.item():.4f}",
            }
        )

    avg_loss_g = running_loss_g / max(len(dataloader), 1)
    avg_loss_d = running_loss_d / max(len(dataloader), 1)
    return avg_loss_g, avg_loss_d, global_step, last_img_time


def _validate(
    model_g: UNetGenerator,
    dataloader: DataLoader,
    criterion_recon: nn.Module,
    device: torch.device,
) -> float:
    if len(dataloader) == 0:
        return float("nan")
    model_g.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            source_images = batch["source_image"].to(device)
            target_images = batch["target_image"].to(device)
            fake_targets = model_g(source_images)
            running_loss += criterion_recon(fake_targets, target_images).item()
    return running_loss / max(len(dataloader), 1)


def _write_loss_csv(
    csv_file: Path,
    train_g_losses: list[float],
    train_d_losses: list[float],
    val_g_losses: list[float],
) -> None:
    pd.DataFrame(
        {
            "epoch": range(1, len(train_g_losses) + 1),
            "train_loss_g": train_g_losses,
            "train_loss_d": train_d_losses,
            "val_loss_g": val_g_losses,
            "train_loss": train_g_losses,  # compatibility
            "val_loss": val_g_losses,  # compatibility
        }
    ).to_csv(csv_file, index=False)


def _save_sample_predictions(
    model: UNetGenerator,
    dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_samples: int = 4,
) -> None:
    if len(dataloader) == 0:
        return
    model.eval()
    batch = next(iter(dataloader))
    source_images = batch["source_image"][:max_samples].to(device)
    target_images = batch["target_image"][:max_samples].to(device)
    with torch.no_grad():
        predictions = model(source_images)
    rows = [
        _concat_images(
            [_tensor_to_rgba(source), _tensor_to_rgba(pred), _tensor_to_rgba(target)]
        )
        for source, pred, target in zip(source_images, predictions, target_images)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _concat_images(rows, vertical=True).save(output_path)


def _tensor_to_rgba(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGBA")


def _concat_images(images: list[Image.Image], vertical: bool = False) -> Image.Image:
    width = (
        max(image.width for image in images)
        if vertical
        else sum(image.width for image in images)
    )
    height = (
        sum(image.height for image in images)
        if vertical
        else max(image.height for image in images)
    )
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x = y = 0
    for image in images:
        canvas.alpha_composite(image, (x, y))
        if vertical:
            y += image.height
        else:
            x += image.width
    return canvas


def _save_checkpoint(
    model_g: UNetGenerator,
    model_d: Discriminator,
    checkpoints_dir: Path,
    global_step: int,
) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gen_file = checkpoints_dir / f"generator_{timestamp}.safetensors"
    disc_file = checkpoints_dir / f"discriminator_{timestamp}.safetensors"

    metadata = {
        "global_step": str(global_step),
        "timestamp": timestamp,
    }

    # Temporary files for write safety (Write-then-replace)
    tmp_gen = gen_file.with_suffix(".tmp")
    tmp_disc = disc_file.with_suffix(".tmp")

    try:
        # Save Generator (ensure contiguous tensors for safetensors saving)
        state_dict_g = {k: v.contiguous() for k, v in model_g.state_dict().items()}
        save_file(state_dict_g, str(tmp_gen), metadata=metadata)
        if tmp_gen.exists():
            tmp_gen.replace(gen_file)

        # Save Discriminator (ensure contiguous tensors for safetensors saving)
        state_dict_d = {k: v.contiguous() for k, v in model_d.state_dict().items()}
        save_file(state_dict_d, str(tmp_disc), metadata=metadata)
        if tmp_disc.exists():
            tmp_disc.replace(disc_file)

        print(f"\n[Info] Saved checkpoint safely:")
        print(f"  Generator: {gen_file}")
        print(f"  Discriminator: {disc_file}")
    except Exception as e:
        print(f"Error saving checkpoint safely: {e}")
        if tmp_gen.exists():
            tmp_gen.unlink()
        if tmp_disc.exists():
            tmp_disc.unlink()

    return gen_file, disc_file


def _log_sample_predictions_to_tb(
    model: UNetGenerator,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    max_samples: int = 4,
) -> None:
    if len(dataloader) == 0:
        return
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return
    source_images = batch["source_image"][:max_samples].to(device)
    target_images = batch["target_image"][:max_samples].to(device)
    with torch.no_grad():
        predictions = model(source_images)
    rows = [
        _concat_images(
            [_tensor_to_rgba(source), _tensor_to_rgba(pred), _tensor_to_rgba(target)]
        )
        for source, pred, target in zip(source_images, predictions, target_images)
    ]
    full_img = _concat_images(rows, vertical=True)
    # Convert PIL Image to Tensor (C, H, W)
    arr = np.array(full_img)
    arr_tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # shape: (4, H, W)
    
    # Write to Tensorboard under Tag "Generated_Image"
    writer.add_image("Generated_Image", arr_tensor, global_step)
    model.train()
