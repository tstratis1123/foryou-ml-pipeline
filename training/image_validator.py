"""Pre-validation checks for training image uploads.

Runs lightweight CPU-based quality checks before the expensive GPU training
pipeline starts. Returns per-image rejection reasons so the performer gets
specific, actionable feedback (e.g. "photo_003.jpg: too blurry").

All checks run in milliseconds per image — no GPU required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_RESOLUTION = 512
MIN_FACE_AREA_FRACTION = 0.15
BLUR_THRESHOLD = 100.0
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 220.0
DUPLICATE_HASH_BITS = 64


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImageCheckResult:
    """Result of validating a single image."""

    path: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation report for all uploaded images."""

    total: int
    passed: int
    failed: int
    results: list[ImageCheckResult]
    duplicate_groups: list[list[str]]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_resolution(img: Image.Image) -> str | None:
    """Reject images smaller than MIN_RESOLUTION on either dimension."""
    w, h = img.size
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        return f"too small ({w}x{h}), minimum {MIN_RESOLUTION}x{MIN_RESOLUTION}"
    return None


def check_sharpness(gray: np.ndarray) -> str | None:
    """Reject blurry images using Laplacian variance.

    The Laplacian operator highlights edges. Blurry images have low edge
    energy, producing a low variance of Laplacian values.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    if variance < BLUR_THRESHOLD:
        return f"too blurry (sharpness={variance:.0f}, minimum={BLUR_THRESHOLD:.0f})"
    return None


def check_lighting(gray: np.ndarray) -> str | None:
    """Reject images with extreme under- or overexposure.

    Uses the mean pixel intensity of the grayscale image as a proxy for
    overall exposure level.
    """
    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_THRESHOLD:
        return f"too dark (brightness={mean_brightness:.0f}, minimum={DARK_THRESHOLD:.0f})"
    if mean_brightness > BRIGHT_THRESHOLD:
        return f"too bright (brightness={mean_brightness:.0f}, maximum={BRIGHT_THRESHOLD:.0f})"
    return None


def check_single_face(gray: np.ndarray) -> str | None:
    """Reject images with zero or multiple faces.

    Uses OpenCV's Haar cascade for fast CPU-based face counting. This is a
    coarse check — the full InsightFace detection in Stage 2 handles
    alignment and quality scoring.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return "no face detected"
    if len(faces) > 1:
        return f"multiple faces detected ({len(faces)}), upload photos with only you"
    return None


def check_face_size(gray: np.ndarray) -> str | None:
    """Reject images where the face is too small relative to the image.

    A tiny face in a large image won't provide enough detail for LoRA
    training after cropping and resizing.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) != 1:
        return None  # Handled by check_single_face

    _, _, fw, fh = faces[0]
    img_h, img_w = gray.shape
    face_area_fraction = (fw * fh) / (img_w * img_h)

    if face_area_fraction < MIN_FACE_AREA_FRACTION:
        return f"face too small ({face_area_fraction:.0%} of image, minimum {MIN_FACE_AREA_FRACTION:.0%})"
    return None


def _perceptual_hash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a difference hash (dHash) for near-duplicate detection.

    dHash compares adjacent pixel intensities in a downscaled grayscale
    image. Two images with Hamming distance ≤ 5 are considered duplicates.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(resized)
    # Each bit is 1 if the pixel is brighter than its right neighbour.
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join("1" if b else "0" for b in diff.flatten())


def _hamming_distance(h1: str, h2: str) -> int:
    """Count differing bits between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

DUPLICATE_HAMMING_THRESHOLD = 5


def validate_training_images(image_paths: list[Path]) -> ValidationReport:
    """Run all pre-validation checks on a set of training images.

    Checks per image:
    - Minimum resolution (≥ 512×512)
    - Sharpness (Laplacian variance > threshold)
    - Lighting (not too dark or too bright)
    - Single face present
    - Face large enough relative to image

    Cross-image checks:
    - Near-duplicate detection via perceptual hashing

    Args:
        image_paths: Paths to the uploaded images.

    Returns:
        A ``ValidationReport`` with per-image results and duplicate groups.
    """
    results: list[ImageCheckResult] = []
    hashes: list[tuple[str, str]] = []  # (path, hash)

    for img_path in image_paths:
        reasons: list[str] = []
        path_str = str(img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            results.append(ImageCheckResult(path=path_str, passed=False, reasons=["could not open image file"]))
            logger.warning("Could not open image", extra={"path": path_str})
            continue

        # Resolution check.
        reason = check_resolution(img)
        if reason:
            reasons.append(reason)

        # Convert to grayscale for OpenCV-based checks.
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Sharpness check.
        reason = check_sharpness(gray)
        if reason:
            reasons.append(reason)

        # Lighting check.
        reason = check_lighting(gray)
        if reason:
            reasons.append(reason)

        # Single face check.
        reason = check_single_face(gray)
        if reason:
            reasons.append(reason)

        # Face size check (only meaningful if exactly 1 face was found).
        reason = check_face_size(gray)
        if reason:
            reasons.append(reason)

        passed = len(reasons) == 0
        results.append(ImageCheckResult(path=path_str, passed=passed, reasons=reasons))

        # Compute perceptual hash for duplicate detection.
        phash = _perceptual_hash(img)
        hashes.append((path_str, phash))

        if not passed:
            logger.info("Image failed validation", extra={"path": path_str, "reasons": reasons})

    # Cross-image duplicate detection.
    duplicate_groups: list[list[str]] = []
    seen_in_group: set[int] = set()

    for i in range(len(hashes)):
        if i in seen_in_group:
            continue
        group = [hashes[i][0]]
        for j in range(i + 1, len(hashes)):
            if j in seen_in_group:
                continue
            if _hamming_distance(hashes[i][1], hashes[j][1]) <= DUPLICATE_HAMMING_THRESHOLD:
                group.append(hashes[j][0])
                seen_in_group.add(j)
        if len(group) > 1:
            seen_in_group.add(i)
            duplicate_groups.append(group)
            # Mark duplicates (keep first, flag rest) in results.
            for dup_path in group[1:]:
                for r in results:
                    if r.path == dup_path:
                        r.reasons.append(f"near-duplicate of {group[0]}")
                        r.passed = False

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    logger.info(
        "Image validation complete",
        extra={
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "duplicate_groups": len(duplicate_groups),
        },
    )

    return ValidationReport(
        total=len(results),
        passed=passed_count,
        failed=failed_count,
        results=results,
        duplicate_groups=duplicate_groups,
    )
