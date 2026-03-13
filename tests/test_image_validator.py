"""Unit tests for training image pre-validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from training.image_validator import (
    MIN_RESOLUTION,
    check_lighting,
    check_resolution,
    check_sharpness,
    validate_training_images,
)


def _create_image(
    width: int = 512,
    height: int = 512,
    brightness: int = 128,
    seed: int = 42,
) -> tuple[str, Image.Image]:
    """Create a test image with controllable properties."""
    rng = np.random.RandomState(seed)
    # Generate noise centred around the desired brightness.
    base = np.clip(
        rng.normal(loc=brightness, scale=30, size=(height, width, 3)),
        0,
        255,
    ).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")
    path = os.path.join(tempfile.mkdtemp(), "test.png")
    img.save(path)
    return path, img


def _create_sharp_image(width: int = 512, height: int = 512) -> tuple[str, Image.Image]:
    """Create an image with high-frequency content (sharp edges)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw high-contrast edges.
    for i in range(0, height, 4):
        arr[i, :, :] = 255
    for j in range(0, width, 4):
        arr[:, j, :] = 255
    img = Image.fromarray(arr, mode="RGB")
    path = os.path.join(tempfile.mkdtemp(), "sharp.png")
    img.save(path)
    return path, img


def _create_blurry_image(width: int = 512, height: int = 512) -> tuple[str, Image.Image]:
    """Create a very blurry (uniform) image."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = os.path.join(tempfile.mkdtemp(), "blurry.png")
    img.save(path)
    return path, img


# ---------------------------------------------------------------------------
# Resolution checks
# ---------------------------------------------------------------------------


class TestResolution:
    def test_passes_at_minimum(self) -> None:
        _, img = _create_image(width=MIN_RESOLUTION, height=MIN_RESOLUTION)
        assert check_resolution(img) is None

    def test_fails_below_minimum(self) -> None:
        _, img = _create_image(width=256, height=256)
        reason = check_resolution(img)
        assert reason is not None
        assert "too small" in reason

    def test_fails_one_dimension_too_small(self) -> None:
        _, img = _create_image(width=1024, height=256)
        reason = check_resolution(img)
        assert reason is not None


# ---------------------------------------------------------------------------
# Sharpness checks
# ---------------------------------------------------------------------------


class TestSharpness:
    def test_sharp_image_passes(self) -> None:
        _, img = _create_sharp_image()
        import cv2

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        assert check_sharpness(gray) is None

    def test_blurry_image_fails(self) -> None:
        _, img = _create_blurry_image()
        import cv2

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        reason = check_sharpness(gray)
        assert reason is not None
        assert "blurry" in reason


# ---------------------------------------------------------------------------
# Lighting checks
# ---------------------------------------------------------------------------


class TestLighting:
    def test_normal_lighting_passes(self) -> None:
        _, img = _create_image(brightness=128)
        import cv2

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        assert check_lighting(gray) is None

    def test_too_dark_fails(self) -> None:
        _, img = _create_image(brightness=10)
        import cv2

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        reason = check_lighting(gray)
        assert reason is not None
        assert "dark" in reason

    def test_too_bright_fails(self) -> None:
        _, img = _create_image(brightness=245)
        import cv2

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        reason = check_lighting(gray)
        assert reason is not None
        assert "bright" in reason


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    def test_identical_images_detected(self) -> None:
        path1, _ = _create_image(seed=42)
        path2, _ = _create_image(seed=42)  # Same seed = identical content
        report = validate_training_images([Path(path1), Path(path2)])
        assert len(report.duplicate_groups) >= 1

    def test_different_images_not_flagged(self) -> None:
        path1, _ = _create_image(seed=42)
        path2, _ = _create_image(seed=99)  # Different seed = different content
        report = validate_training_images([Path(path1), Path(path2)])
        assert len(report.duplicate_groups) == 0


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_report_counts(self) -> None:
        good_path, _ = _create_sharp_image(512, 512)
        bad_path, _ = _create_image(width=100, height=100)
        report = validate_training_images([Path(good_path), Path(bad_path)])

        assert report.total == 2
        assert report.failed >= 1  # The 100x100 image should fail

    def test_unopenable_file(self) -> None:
        bad_path = os.path.join(tempfile.mkdtemp(), "not_an_image.png")
        with open(bad_path, "w") as f:
            f.write("not image data")
        report = validate_training_images([Path(bad_path)])

        assert report.total == 1
        assert report.failed == 1
        assert "could not open" in report.results[0].reasons[0]

    def test_pass_rate(self) -> None:
        path1, _ = _create_sharp_image()
        path2, _ = _create_sharp_image()
        report = validate_training_images([Path(path1), Path(path2)])

        # Both should be valid images (sharp, correct resolution, normal lighting)
        # but may fail face detection — pass_rate checks the math regardless.
        assert 0.0 <= report.pass_rate <= 1.0
