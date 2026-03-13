"""VideoSeal training pipeline.

Trains the encoder-decoder pair adversarially:
- The encoder learns to produce imperceptible residuals that carry the payload.
- The decoder learns to recover the payload from watermarked (and distorted) frames.

Robustness augmentations (JPEG compression, scaling, Gaussian noise, cropping)
are applied between encoder and decoder during training so the watermark
survives real-world video processing.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from watermark.video_watermark import VideoSealWatermark, WatermarkDecoder, WatermarkEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robustness augmentations (differentiable or applied in-place)
# ---------------------------------------------------------------------------


def _jpeg_augment(frames: torch.Tensor, quality_range: tuple[int, int] = (50, 95)) -> torch.Tensor:
    """Simulate JPEG compression on a batch of frames.

    Not differentiable — applied with ``torch.no_grad`` and the gradient is
    passed through via straight-through estimation (STE).

    Args:
        frames: (B, 3, H, W) float tensor in [0, 1].
        quality_range: (min_quality, max_quality) for random JPEG quality.

    Returns:
        JPEG-compressed frames as a float tensor in [0, 1].
    """
    quality = random.randint(*quality_range)
    device = frames.device
    result = torch.empty_like(frames)

    for i in range(frames.shape[0]):
        # Convert to PIL, compress, decompress.
        arr = (frames[i].detach().cpu().permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        img = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img_back = Image.open(buf)
        arr_back = np.array(img_back, dtype=np.float32) / 255.0
        result[i] = torch.from_numpy(arr_back).permute(2, 0, 1)

    result = result.to(device)
    # Straight-through estimator: forward uses JPEG'd version, backward passes through.
    return frames + (result - frames).detach()


def _gaussian_noise_augment(frames: torch.Tensor, std_range: tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    """Add Gaussian noise to frames (differentiable)."""
    std = random.uniform(*std_range)
    noise = torch.randn_like(frames) * std
    return (frames + noise).clamp(0.0, 1.0)


def _resize_augment(frames: torch.Tensor, scale_range: tuple[float, float] = (0.5, 0.9)) -> torch.Tensor:
    """Downscale then upscale back to original size (differentiable)."""
    scale = random.uniform(*scale_range)
    _, _, h, w = frames.shape
    new_h, new_w = max(int(h * scale), 8), max(int(w * scale), 8)
    down = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up


def _crop_augment(frames: torch.Tensor, crop_fraction: tuple[float, float] = (0.7, 0.9)) -> torch.Tensor:
    """Random crop then resize back to original dimensions (differentiable)."""
    frac = random.uniform(*crop_fraction)
    _, _, h, w = frames.shape
    crop_h, crop_w = max(int(h * frac), 8), max(int(w * frac), 8)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    cropped = frames[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


_AUGMENTATIONS = [
    ("jpeg", _jpeg_augment),
    ("noise", _gaussian_noise_augment),
    ("resize", _resize_augment),
    ("crop", _crop_augment),
]


def _apply_random_augmentations(
    frames: torch.Tensor,
    *,
    num_augmentations: int = 2,
) -> torch.Tensor:
    """Apply a random subset of robustness augmentations.

    During training, this sits between the encoder (watermark embedding) and
    the decoder (watermark extraction), forcing the network to produce
    watermarks robust to common video processing operations.

    Args:
        frames: (B, 3, H, W) watermarked frames.
        num_augmentations: How many augmentations to apply per call.

    Returns:
        Augmented frames.
    """
    selected = random.sample(_AUGMENTATIONS, k=min(num_augmentations, len(_AUGMENTATIONS)))
    for name, aug_fn in selected:
        frames = aug_fn(frames)
        logger.debug("Applied augmentation: %s", name)
    return frames


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for VideoSeal training."""

    # Data
    image_dir: str = ""  # Directory of training images (any format PIL can read)
    image_size: int = 256  # Resize training images to this square size

    # Training
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    payload_length: int = 64

    # Loss weights
    image_loss_weight: float = 1.0  # Weight for imperceptibility (MSE between original and watermarked)
    payload_loss_weight: float = 10.0  # Weight for payload recovery (BCE between input and decoded payload)

    # Augmentation
    augmentation_probability: float = 0.8  # Probability of applying augmentations per batch
    num_augmentations: int = 2

    # Watermark
    strength: float = 0.03  # Residual scaling factor

    # Output
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs

    # Device
    device: str = "cuda"

    # Logging
    log_every: int = 50  # Log metrics every N steps

    # Curriculum: increase augmentation difficulty over training
    curriculum_warmup_epochs: int = 10  # Epochs before augmentations start

    # Early stopping
    patience: int = 20  # Stop if no improvement for N epochs

    # Derived (populated at runtime)
    _augmentation_names: list[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _load_training_images(image_dir: str, image_size: int) -> torch.Tensor:
    """Load and preprocess all images from a directory.

    Returns:
        (N, 3, H, W) float tensor in [0, 1].
    """
    path = Path(image_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    tensors: list[torch.Tensor] = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensors.append(t)

    dataset = torch.stack(tensors)
    logger.info("Loaded %d training images from %s (%dx%d)", len(tensors), image_dir, image_size, image_size)
    return dataset


def _generate_random_payloads(batch_size: int, payload_length: int, device: str) -> torch.Tensor:
    """Generate random binary payloads for training.

    Returns:
        (B, payload_length) float tensor of 0.0 / 1.0 values.
    """
    return torch.randint(0, 2, (batch_size, payload_length), dtype=torch.float32, device=device)


def train(config: TrainingConfig) -> VideoSealWatermark:
    """Train VideoSeal encoder-decoder from scratch.

    The training objective is adversarial in nature:
    - **Imperceptibility loss** (MSE): The watermarked frame should look
      identical to the original.
    - **Payload recovery loss** (BCE): The decoder should recover the exact
      payload bits from the (augmented) watermarked frame.

    Robustness augmentations (JPEG, noise, resize, crop) are applied between
    encoder and decoder, teaching the network to survive real-world distortions.

    A curriculum schedule delays augmentations for the first few epochs so the
    networks can first learn basic embedding/recovery, then gradually face
    harder distortions.

    Args:
        config: Training configuration.

    Returns:
        A trained ``VideoSealWatermark`` instance with weights on the
        specified device.
    """
    device = config.device
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    images = _load_training_images(config.image_dir, config.image_size)
    num_images = images.shape[0]

    # Initialise networks.
    encoder = WatermarkEncoder(config.payload_length).to(device)
    decoder = WatermarkDecoder(config.payload_length).to(device)

    # Optimisers — one for each network, trained jointly.
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01
    )

    # Loss functions.
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    best_total_loss = float("inf")
    epochs_without_improvement = 0

    logger.info(
        "Starting VideoSeal training: %d epochs, batch_size=%d, lr=%.1e, payload=%d bits",
        config.num_epochs,
        config.batch_size,
        config.learning_rate,
        config.payload_length,
    )

    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        encoder.train()
        decoder.train()

        # Shuffle dataset each epoch.
        perm = torch.randperm(num_images)
        epoch_image_loss = 0.0
        epoch_payload_loss = 0.0
        epoch_bit_accuracy = 0.0
        num_batches = 0

        # Curriculum: only apply augmentations after warmup.
        use_augmentations = epoch > config.curriculum_warmup_epochs

        for batch_start in range(0, num_images, config.batch_size):
            batch_indices = perm[batch_start : batch_start + config.batch_size]
            if len(batch_indices) < 2:
                continue  # Skip very small batches (BatchNorm needs ≥2)

            frames = images[batch_indices].to(device)
            batch_size = frames.shape[0]

            # Random payloads.
            payloads = _generate_random_payloads(batch_size, config.payload_length, device)

            # --- Forward pass ---
            # 1. Encoder produces a residual.
            residual = encoder(frames, payloads)

            # 2. Add scaled residual to get watermarked frames.
            watermarked = (frames + config.strength * residual).clamp(0.0, 1.0)

            # 3. Apply robustness augmentations (if past warmup).
            if use_augmentations and random.random() < config.augmentation_probability:
                augmented = _apply_random_augmentations(
                    watermarked,
                    num_augmentations=config.num_augmentations,
                )
            else:
                augmented = watermarked

            # 4. Decoder tries to recover payload from augmented frames.
            predicted_payload = decoder(augmented)

            # --- Loss computation ---
            # Imperceptibility: watermarked should match original.
            l_image = mse_loss(watermarked, frames)

            # Payload recovery: decoded bits should match input payload.
            l_payload = bce_loss(predicted_payload, payloads)

            loss = config.image_loss_weight * l_image + config.payload_loss_weight * l_payload

            # --- Backward pass ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            # --- Metrics ---
            with torch.no_grad():
                bit_acc = ((predicted_payload > 0.5).float() == payloads).float().mean().item()

            epoch_image_loss += l_image.item()
            epoch_payload_loss += l_payload.item()
            epoch_bit_accuracy += bit_acc
            num_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                logger.info(
                    "step=%d  image_loss=%.5f  payload_loss=%.4f  bit_acc=%.2f%%  lr=%.2e",
                    global_step,
                    l_image.item(),
                    l_payload.item(),
                    bit_acc * 100,
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()

        # Epoch summary.
        if num_batches > 0:
            avg_img = epoch_image_loss / num_batches
            avg_pay = epoch_payload_loss / num_batches
            avg_acc = epoch_bit_accuracy / num_batches
            avg_total = config.image_loss_weight * avg_img + config.payload_loss_weight * avg_pay

            logger.info(
                "Epoch %d/%d — image_loss=%.5f  payload_loss=%.4f  bit_acc=%.2f%%  total=%.4f",
                epoch,
                config.num_epochs,
                avg_img,
                avg_pay,
                avg_acc * 100,
                avg_total,
            )

            # Save periodic checkpoints.
            if epoch % config.save_every == 0:
                ckpt_path = checkpoint_dir / f"videoseal_epoch_{epoch:04d}.pt"
                torch.save(
                    {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch},
                    str(ckpt_path),
                )
                logger.info("Checkpoint saved: %s", ckpt_path)

            # Early stopping check.
            if avg_total < best_total_loss:
                best_total_loss = avg_total
                epochs_without_improvement = 0
                # Save best model.
                best_path = checkpoint_dir / "videoseal_best.pt"
                torch.save(
                    {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch},
                    str(best_path),
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        config.patience,
                    )
                    break

    # Build the final VideoSealWatermark instance with trained weights.
    final_path = checkpoint_dir / "videoseal_final.pt"
    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
        str(final_path),
    )
    logger.info("Final model saved: %s", final_path)

    watermarker = VideoSealWatermark(
        model_path=str(final_path),
        payload_length=config.payload_length,
        device=device,
    )
    return watermarker
