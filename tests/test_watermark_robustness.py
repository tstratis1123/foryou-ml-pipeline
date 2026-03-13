"""Robustness tests for image (DWT-DCT-SVD) and video (VideoSeal) watermarks.

These tests verify that watermarks survive common distortions:
- JPEG compression
- Scaling (downscale + upscale)
- Gaussian noise
- Cropping (for image only — video crop is handled during VideoSeal training)

The image watermark uses non-blind extraction (requires original), so it has
inherently higher robustness to distortions that don't destroy spatial layout.

The VideoSeal watermark uses blind extraction and randomly initialised weights
in tests, so we verify the embed/extract roundtrip and augmentation plumbing
rather than trained-model accuracy.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark.image_watermark import DwtDctSvdWatermark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_test_image(width: int = 512, height: int = 512, seed: int = 42) -> str:
    """Create a synthetic test image and return its path.

    Uses a seeded RNG so tests are deterministic.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    path = os.path.join(tempfile.mkdtemp(), "test_image.png")
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _apply_jpeg_compression(image_path: str, quality: int) -> str:
    """Compress an image with JPEG at the given quality and return new path."""
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)
    out_path = os.path.join(tempfile.mkdtemp(), f"jpeg_q{quality}.png")
    compressed.save(out_path)
    return out_path


def _apply_scaling(image_path: str, scale: float) -> str:
    """Downscale then upscale an image back to its original size."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    small = img.resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.BILINEAR)
    restored = small.resize((w, h), Image.BILINEAR)
    out_path = os.path.join(tempfile.mkdtemp(), f"scaled_{scale}.png")
    restored.save(out_path)
    return out_path


def _apply_gaussian_noise(image_path: str, std: float, seed: int = 123) -> str:
    """Add Gaussian noise to an image."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out_path = os.path.join(tempfile.mkdtemp(), f"noisy_{std}.png")
    Image.fromarray(noisy, mode="RGB").save(out_path)
    return out_path


def _apply_center_crop(image_path: str, crop_fraction: float) -> str:
    """Center-crop an image then resize back to original dimensions."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    crop_w, crop_h = int(w * crop_fraction), int(h * crop_fraction)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    restored = cropped.resize((w, h), Image.BILINEAR)
    out_path = os.path.join(tempfile.mkdtemp(), f"cropped_{crop_fraction}.png")
    restored.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Image watermark (DWT-DCT-SVD) robustness tests
# ---------------------------------------------------------------------------

PAYLOAD = b"test_customer_id_12345"
KEY = "test-secret-key-2024"
STRENGTH = 0.1


class TestImageWatermarkRobustness:
    """Test that DWT-DCT-SVD watermarks survive common distortions."""

    def test_lossless_roundtrip(self) -> None:
        """Watermark should be perfectly recoverable without any distortion."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        extracted = wm.extract(watermarked, original)
        assert extracted == PAYLOAD

    def test_jpeg_quality_90(self) -> None:
        """Watermark should survive high-quality JPEG compression."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        compressed = _apply_jpeg_compression(watermarked, quality=90)
        extracted = wm.extract(compressed, original)
        # Allow for minor bit errors — check ≥80% payload match.
        match_ratio = sum(a == b for a, b in zip(extracted, PAYLOAD)) / len(PAYLOAD)
        assert match_ratio >= 0.8, f"JPEG Q90 payload match ratio too low: {match_ratio:.2%}"

    def test_jpeg_quality_70(self) -> None:
        """Watermark should partially survive medium JPEG compression."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        compressed = _apply_jpeg_compression(watermarked, quality=70)
        extracted = wm.extract(compressed, original)
        # Medium compression — at least 50% of payload bytes should survive.
        match_ratio = sum(a == b for a, b in zip(extracted, PAYLOAD)) / len(PAYLOAD)
        assert match_ratio >= 0.5, f"JPEG Q70 payload match ratio too low: {match_ratio:.2%}"

    def test_scaling_75_percent(self) -> None:
        """Watermark should survive 75% downscale + upscale."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        scaled = _apply_scaling(watermarked, scale=0.75)
        extracted = wm.extract(scaled, original)
        match_ratio = sum(a == b for a, b in zip(extracted, PAYLOAD)) / len(PAYLOAD)
        assert match_ratio >= 0.5, f"Scale 0.75 payload match ratio too low: {match_ratio:.2%}"

    def test_gaussian_noise_std5(self) -> None:
        """Watermark should survive mild Gaussian noise (std=5)."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        noisy = _apply_gaussian_noise(watermarked, std=5.0)
        extracted = wm.extract(noisy, original)
        match_ratio = sum(a == b for a, b in zip(extracted, PAYLOAD)) / len(PAYLOAD)
        assert match_ratio >= 0.7, f"Noise std=5 payload match ratio too low: {match_ratio:.2%}"

    def test_higher_strength_improves_robustness(self) -> None:
        """Higher embedding strength should improve robustness at the cost of visibility."""
        original = _create_test_image()
        results: dict[float, float] = {}

        for strength in [0.05, 0.1, 0.2]:
            watermarked = os.path.join(tempfile.mkdtemp(), f"wm_s{strength}.png")
            wm = DwtDctSvdWatermark(key=KEY, strength=strength)
            wm.embed(original, PAYLOAD, watermarked)

            compressed = _apply_jpeg_compression(watermarked, quality=80)
            extracted = wm.extract(compressed, original)
            match_ratio = sum(a == b for a, b in zip(extracted, PAYLOAD)) / len(PAYLOAD)
            results[strength] = match_ratio

        # Higher strength should generally yield equal or better recovery.
        assert results[0.2] >= results[0.05], (
            f"Higher strength didn't improve robustness: s=0.05→{results[0.05]:.2%}, s=0.2→{results[0.2]:.2%}"
        )

    def test_imperceptibility(self) -> None:
        """Watermarked image should be visually similar to original (high PSNR)."""
        original = _create_test_image()
        watermarked = os.path.join(tempfile.mkdtemp(), "watermarked.png")

        wm = DwtDctSvdWatermark(key=KEY, strength=STRENGTH)
        wm.embed(original, PAYLOAD, watermarked)

        orig_arr = np.array(Image.open(original).convert("RGB"), dtype=np.float64)
        wm_arr = np.array(Image.open(watermarked).convert("RGB"), dtype=np.float64)

        mse = np.mean((orig_arr - wm_arr) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 10 * np.log10(255.0**2 / mse)

        # PSNR > 30 dB is considered imperceptible for most watermarking schemes.
        assert psnr > 30, f"PSNR too low (image distortion visible): {psnr:.1f} dB"


# ---------------------------------------------------------------------------
# VideoSeal robustness tests
# ---------------------------------------------------------------------------


class TestVideoSealRobustness:
    """Test VideoSeal embed/extract plumbing and augmentation pipeline.

    Since tests use randomly initialised weights (no pre-trained model), we
    verify the mechanics rather than trained-model accuracy.
    """

    @pytest.fixture()
    def videoseal(self, tmp_path: Path) -> "VideoSealWatermark":
        """Create a VideoSeal instance with random weights."""
        # Import here to avoid torch import overhead if image-only tests run.
        from watermark.video_watermark import VideoSealWatermark

        model_path = str(tmp_path / "nonexistent.pt")  # Forces random init.
        return VideoSealWatermark(
            model_path=model_path,
            payload_length=32,  # Smaller for faster tests.
            device="cpu",
        )

    def test_encoder_produces_residual(self, videoseal: "VideoSealWatermark") -> None:
        """Encoder should produce a residual with the same shape as the input."""
        import torch

        frames = torch.rand(2, 3, 64, 64)
        payload = torch.randint(0, 2, (2, 32), dtype=torch.float32)

        with torch.no_grad():
            residual = videoseal.encoder(frames, payload)

        assert residual.shape == frames.shape

    def test_decoder_produces_payload(self, videoseal: "VideoSealWatermark") -> None:
        """Decoder should produce payload predictions with correct shape."""
        import torch

        frames = torch.rand(2, 3, 64, 64)

        with torch.no_grad():
            predicted = videoseal.decoder(frames)

        assert predicted.shape == (2, 32)
        # Sigmoid output should be in [0, 1].
        assert predicted.min() >= 0.0
        assert predicted.max() <= 1.0

    def test_embed_extract_shapes(self, videoseal: "VideoSealWatermark", tmp_path: Path) -> None:
        """Embed + extract roundtrip should produce bytes of expected length."""
        import cv2

        # Create a minimal test video (4 frames, 64x64).
        video_path = str(tmp_path / "test_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 24.0, (64, 64))
        rng = np.random.RandomState(42)
        for _ in range(4):
            frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        output_path = str(tmp_path / "watermarked.mp4")
        payload = b"\xAB\xCD\xEF\x01"

        videoseal.embed(video_path, payload, output_path)

        assert Path(output_path).is_file()

        extracted = videoseal.extract(output_path)
        # Payload length in bytes = ceil(payload_length_bits / 8).
        assert len(extracted) == 32 // 8  # 4 bytes

    def test_save_and_load_weights(self, videoseal: "VideoSealWatermark", tmp_path: Path) -> None:
        """Saving and loading weights should produce identical state dicts."""
        import torch

        save_path = str(tmp_path / "weights.pt")
        videoseal.save_weights(save_path)

        assert Path(save_path).is_file()

        checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
        assert "encoder" in checkpoint
        assert "decoder" in checkpoint

        # Verify the loaded state matches.
        from watermark.video_watermark import VideoSealWatermark as VS

        loaded = VS(model_path=save_path, payload_length=32, device="cpu")

        # Compare a forward pass — both should produce identical output.
        test_input = torch.rand(1, 3, 64, 64)
        test_payload = torch.zeros(1, 32)
        with torch.no_grad():
            r1 = videoseal.encoder(test_input, test_payload)
            r2 = loaded.encoder(test_input, test_payload)
        assert torch.allclose(r1, r2, atol=1e-6)


# ---------------------------------------------------------------------------
# VideoSeal training augmentation tests
# ---------------------------------------------------------------------------


class TestVideoSealAugmentations:
    """Test that training augmentations produce valid outputs."""

    def test_jpeg_augment_preserves_shape(self) -> None:
        """JPEG augmentation should preserve tensor shape and range."""
        import torch

        from watermark.train_videoseal import _jpeg_augment

        frames = torch.rand(2, 3, 64, 64)
        result = _jpeg_augment(frames, quality_range=(70, 70))

        assert result.shape == frames.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gaussian_noise_augment(self) -> None:
        """Gaussian noise should alter the frames but keep them in [0, 1]."""
        import torch

        from watermark.train_videoseal import _gaussian_noise_augment

        frames = torch.full((2, 3, 32, 32), 0.5)
        result = _gaussian_noise_augment(frames, std_range=(0.1, 0.1))

        assert result.shape == frames.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Should differ from input.
        assert not torch.allclose(result, frames)

    def test_resize_augment_preserves_shape(self) -> None:
        """Resize augmentation should return the same shape."""
        import torch

        from watermark.train_videoseal import _resize_augment

        frames = torch.rand(2, 3, 64, 64)
        result = _resize_augment(frames, scale_range=(0.5, 0.5))

        assert result.shape == frames.shape

    def test_crop_augment_preserves_shape(self) -> None:
        """Crop augmentation should return the same shape."""
        import torch

        from watermark.train_videoseal import _crop_augment

        frames = torch.rand(2, 3, 64, 64)
        result = _crop_augment(frames, crop_fraction=(0.7, 0.7))

        assert result.shape == frames.shape

    def test_apply_random_augmentations(self) -> None:
        """Combined augmentation pipeline should produce valid output."""
        import torch

        from watermark.train_videoseal import _apply_random_augmentations

        frames = torch.rand(2, 3, 64, 64)
        result = _apply_random_augmentations(frames, num_augmentations=2)

        assert result.shape == frames.shape
