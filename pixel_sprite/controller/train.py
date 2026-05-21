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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
 
from pixel_sprite.config import load_config, resolve_project_path
from pixel_sprite.controller.split import split_dataset_by_character, subset_character_ids
from pixel_sprite.model.dataset import DirectionPairDataset
from pixel_sprite.model.factory import get_model_class


def train_model(config_path: str | Path = "config.yaml", version: int = 2) -> None:
    config = load_config(config_path)
    tb_port = config["misc"].get("tb_port", 6006)
    _start_tensorboard_if_needed(tb_port)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model version: {version}")

    GeneratorClass = get_model_class(version, "generator")
    DiscriminatorClass = get_model_class(version, "discriminator")

    model_g = GeneratorClass(
        in_channels=config["model"].get("in_channels", 4),
        base_channels=config["model"].get("base_channels", 32),
    ).to(device)

    model_d = DiscriminatorClass(
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

    lr_g = config["train"].get("lr_g", config["train"].get("lr", 0.0002))
    lr_d = config["train"].get("lr_d", config["train"].get("lr", 0.0002))
    lambda_adv = config["train"].get("lambda_adv", 0.01)
    lambda_fm = config["train"].get("lambda_fm", 0.1)
    lambda_alpha = config["train"].get("lambda_alpha", 0.1)

    optimizer_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(0.5, 0.999))

    checkpoints_base = resolve_project_path(config["misc"].get("checkpoints_path", "./checkpoints"))
    checkpoints_dir = checkpoints_base / f"v{version}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    csv_file = checkpoints_dir / "loss.csv"

    start_epoch, global_step, train_g_losses, train_d_losses, val_g_losses = _load_checkpoint(
        model_g, model_d, optimizer_g, optimizer_d, checkpoints_dir, csv_file, device
    )

    # Ensure learning rates are set from configuration (even if loaded state dict overwrote them)
    for param_group in optimizer_g.param_groups:
        param_group['lr'] = lr_g
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = lr_d

    tb_log_dir = resolve_project_path("runs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    criterion_recon = nn.L1Loss().to(device)
    criterion_gan = nn.BCELoss().to(device)

    end_epoch = start_epoch + config["train"]["epochs"]
    sample_every = config["train"].get("sample_every", 10)

    last_img_time = 0.0
    last_backup_time = time.time()

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

            # Update loss CSV
            _write_loss_csv(csv_file, train_g_losses, train_d_losses, val_g_losses)

            # Save temporary backup if 30s has passed since last save
            current_time = time.time()
            if current_time - last_backup_time >= 30.0:
                _save_checkpoint(model_g, model_d, optimizer_g, optimizer_d, checkpoints_dir, global_step, suffix="temp")
                last_backup_time = current_time

            if sample_every and (epoch + 1) % sample_every == 0:
                _save_sample_predictions(
                    model_g, val_loader, device, checkpoints_dir / "samples" / f"epoch_{epoch + 1:04d}.png"
                )

        # Save final checkpoint upon successful normal completion
        _save_checkpoint(model_g, model_d, optimizer_g, optimizer_d, checkpoints_dir, global_step)
        _delete_temp_checkpoints(checkpoints_dir)

    except KeyboardInterrupt:
        print("\n[Warning] Training interrupted by user. Saving current checkpoint safely...")
        _save_checkpoint(model_g, model_d, optimizer_g, optimizer_d, checkpoints_dir, global_step)
        _write_loss_csv(csv_file, train_g_losses, train_d_losses, val_g_losses)
        _delete_temp_checkpoints(checkpoints_dir)
        print("Progress saved successfully. Exiting.")
    finally:
        writer.close()


def _load_checkpoint(
    model_g: nn.Module,
    model_d: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    checkpoints_dir: Path,
    csv_file: Path,
    device: torch.device,
) -> tuple[int, int, list[float], list[float], list[float]]:
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0, [], [], []

    # 1. Try to load from temp files first (e.g. if training was interrupted/crashed)
    temp_gen = checkpoints_dir / "generator_temp.safetensors"
    if temp_gen.exists():
        latest_gen = temp_gen
        corresponding_d = checkpoints_dir / "discriminator_temp.safetensors"
        corresponding_opt_g = checkpoints_dir / "optimizer_g_temp.pt"
        corresponding_opt_d = checkpoints_dir / "optimizer_d_temp.pt"
        print("Found temporary checkpoint to resume training.")
    else:
        # 2. Fall back to latest timestamped checkpoints (excluding generator_temp.safetensors)
        gen_files = [f for f in checkpoints_dir.glob("generator_*.safetensors") if f.name != "generator_temp.safetensors"]
        if not gen_files:
            return 0, 0, [], [], []

        # Sort checkpoints by filename reverse (newest timestamp first)
        gen_files.sort(key=lambda x: x.name, reverse=True)
        latest_gen = gen_files[0]
        
        timestamp = latest_gen.name.replace("generator_", "").replace(".safetensors", "")
        corresponding_d = checkpoints_dir / f"discriminator_{timestamp}.safetensors"
        corresponding_opt_g = checkpoints_dir / f"optimizer_g_{timestamp}.pt"
        corresponding_opt_d = checkpoints_dir / f"optimizer_d_{timestamp}.pt"

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

    if corresponding_opt_g.exists():
        try:
            state_dict_opt_g = torch.load(corresponding_opt_g, map_location=device)
            optimizer_g.load_state_dict(state_dict_opt_g)
            print(f"Loaded Generator Optimizer state from {corresponding_opt_g}")
        except Exception as e:
            print(f"Error loading Generator Optimizer state: {e}")

    if corresponding_opt_d.exists():
        try:
            state_dict_opt_d = torch.load(corresponding_opt_d, map_location=device)
            optimizer_d.load_state_dict(state_dict_opt_d)
            print(f"Loaded Discriminator Optimizer state from {corresponding_opt_d}")
        except Exception as e:
            print(f"Error loading Discriminator Optimizer state: {e}")

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
    model_g: nn.Module,
    model_d: nn.Module,
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
    preview_dataset: Dataset | DataLoader,
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
        loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

        # Fake pair: D(source, G(source))
        fake_targets = model_g(source_images)
        fake_pair = torch.cat([source_images, fake_targets.detach()], dim=1)
        pred_fake, _ = model_d(fake_pair)
        loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

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
        loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))

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
            _log_sample_predictions_to_tb(model_g, preview_dataset, device, writer, global_step)
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
    model_g: nn.Module,
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
    model: nn.Module,
    dataset_or_loader: Dataset | DataLoader,
    device: torch.device,
    output_path: Path,
    max_samples: int = 1,
) -> None:
    if dataset_or_loader is None:
        return
    model.eval()
    
    if hasattr(dataset_or_loader, "dataset"):
        dataset = dataset_or_loader.dataset
    else:
        dataset = dataset_or_loader
        
    if len(dataset) == 0:
        return
    
    import random
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    
    source_images = torch.stack([s["source_image"] for s in samples]).to(device)
    target_images = torch.stack([s["target_image"] for s in samples]).to(device)
    
    with torch.no_grad():
        predictions = model(source_images)
    rows = [
        _concat_images(
            [_tensor_to_rgba(source), _tensor_to_rgba(pred), _tensor_to_rgba(target)]
        )
        for source, pred, target in zip(source_images, predictions, target_images)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_img = _concat_images(rows, vertical=True)
    w, h = full_img.size
    full_img.resize((w * 4, h * 4), Image.NEAREST).save(output_path)


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
    model_g: nn.Module,
    model_d: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    checkpoints_dir: Path,
    global_step: int,
    suffix: str | None = None,
) -> tuple[Path, Path]:
    if suffix:
        tag = suffix
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")

    gen_file = checkpoints_dir / f"generator_{tag}.safetensors"
    disc_file = checkpoints_dir / f"discriminator_{tag}.safetensors"
    opt_g_file = checkpoints_dir / f"optimizer_g_{tag}.pt"
    opt_d_file = checkpoints_dir / f"optimizer_d_{tag}.pt"

    metadata = {
        "global_step": str(global_step),
        "tag": tag,
    }

    # Temporary files for write safety (Write-then-replace)
    tmp_gen = gen_file.with_suffix(".tmp")
    tmp_disc = disc_file.with_suffix(".tmp")
    tmp_opt_g = opt_g_file.with_suffix(".tmp")
    tmp_opt_d = opt_d_file.with_suffix(".tmp")

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

        # Save Optimizers
        torch.save(optimizer_g.state_dict(), tmp_opt_g)
        if tmp_opt_g.exists():
            tmp_opt_g.replace(opt_g_file)

        torch.save(optimizer_d.state_dict(), tmp_opt_d)
        if tmp_opt_d.exists():
            tmp_opt_d.replace(opt_d_file)

        print(f"\n[Info] Saved checkpoint safely ({tag}):")
        print(f"  Generator: {gen_file}")
        print(f"  Discriminator: {disc_file}")
        print(f"  Generator Optimizer: {opt_g_file}")
        print(f"  Discriminator Optimizer: {opt_d_file}")
    except Exception as e:
        print(f"Error saving checkpoint safely: {e}")
        for tmp_f in [tmp_gen, tmp_disc, tmp_opt_g, tmp_opt_d]:
            if tmp_f.exists():
                tmp_f.unlink()

    return gen_file, disc_file


def _delete_temp_checkpoints(checkpoints_dir: Path) -> None:
    temp_files = [
        checkpoints_dir / "generator_temp.safetensors",
        checkpoints_dir / "discriminator_temp.safetensors",
        checkpoints_dir / "optimizer_g_temp.pt",
        checkpoints_dir / "optimizer_d_temp.pt",
        checkpoints_dir / "generator_temp.tmp",
        checkpoints_dir / "discriminator_temp.tmp",
        checkpoints_dir / "optimizer_g_temp.tmp",
        checkpoints_dir / "optimizer_d_temp.tmp",
    ]
    for f in temp_files:
        if f.exists():
            try:
                f.unlink()
                print(f"Deleted temporary checkpoint file: {f.name}")
            except Exception as e:
                print(f"Error deleting temporary checkpoint file {f.name}: {e}")


def _log_sample_predictions_to_tb(
    model: nn.Module,
    dataset_or_loader: Dataset | DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    max_samples: int = 1,
) -> None:
    if dataset_or_loader is None:
        return
    model.eval()
    
    if hasattr(dataset_or_loader, "dataset"):
        dataset = dataset_or_loader.dataset
    else:
        dataset = dataset_or_loader
        
    if len(dataset) == 0:
        return
    
    import random
    
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    batch_samples = [dataset[i] for i in indices]
    
    source_images = torch.stack([s["source_image"] for s in batch_samples]).to(device)
    target_images = torch.stack([s["target_image"] for s in batch_samples]).to(device)
    
    with torch.no_grad():
        predictions = model(source_images)
    rows = [
        _concat_images(
            [_tensor_to_rgba(source), _tensor_to_rgba(pred), _tensor_to_rgba(target)]
        )
        for source, pred, target in zip(source_images, predictions, target_images)
    ]
    full_img = _concat_images(rows, vertical=True)
    w, h = full_img.size
    full_img_large = full_img.resize((w * 4, h * 4), Image.NEAREST)
    # Convert PIL Image to Tensor (C, H, W)
    arr = np.array(full_img_large)
    arr_tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # shape: (4, H, W)
    
    # Write to Tensorboard under Tag "Generated_Image"
    writer.add_image("Generated_Image", arr_tensor, global_step)
    model.train()


def _start_tensorboard_if_needed(tb_port: int) -> None:
    import socket
    import sys
    import subprocess

    # Check if port is in use (robust check covering IPv4, IPv6, and localhost resolution)
    in_use = False

    # 1. Try IPv4 loopback
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", tb_port)) == 0:
                in_use = True
    except Exception:
        pass

    # 2. Try IPv6 loopback
    if not in_use:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("::1", tb_port)) == 0:
                    in_use = True
        except Exception:
            pass

    # 3. Try generic localhost resolved addresses
    if not in_use:
        try:
            for res in socket.getaddrinfo("localhost", tb_port, socket.AF_UNSPEC, socket.SOCK_STREAM):
                af, socktype, proto, canonname, sa = res
                try:
                    with socket.socket(af, socktype, proto) as s:
                        s.settimeout(0.5)
                        if s.connect_ex(sa) == 0:
                            in_use = True
                            break
                except Exception:
                    pass
        except Exception:
            pass

    if in_use:
        print(f"[Info] TensorBoard is already running/accessible on port {tb_port}.")
        return

    print(f"[Info] Launching TensorBoard on port {tb_port}...")
    log_dir = resolve_project_path("runs")
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(tb_port),
    ]

    kwargs = {}
    if sys.platform == "win32":
        # Use subprocess.DETACHED_PROCESS (0x00000008) to run independently
        kwargs["creationflags"] = 0x00000008
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    try:
        log_file_path = resolve_project_path("tensorboard.log")
        log_file = open(log_file_path, "w", encoding="utf-8")
        kwargs["stdout"] = log_file
        kwargs["stderr"] = log_file
        subprocess.Popen(cmd, **kwargs)
        print(f"[Info] TensorBoard started successfully. Open http://localhost:{tb_port}/ to view.")
    except Exception as e:
        print(f"[Warning] Failed to auto-start TensorBoard: {e}")
