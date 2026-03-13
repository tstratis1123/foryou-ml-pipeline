"""Content moderation scanner for images and videos.

Implements the post-generation content scanning pipeline:
  1. NSFW classification (single-frame for images, keyframe-sampled for video)
  2. Face detection with optional blocklist matching
  3. Content policy checks (OCR contact info, violence/hate classification)

Models are lazily loaded on first use and cached globally to amortise
startup cost across sequential scan calls.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline

from shared.config import Config
from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global model cache (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_nsfw_classifier: Any | None = None
_policy_classifier: Any | None = None
_face_detector: cv2.dnn.Net | None = None
_ocr_available: bool | None = None

_config = Config()

# Contact-info regex patterns (phone, email, URL, social handles)
_CONTACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),  # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"https?://\S+", re.IGNORECASE),  # URL
    re.compile(r"(?:^|\s)@[\w.]{2,30}\b"),  # social handle
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of a content moderation scan."""

    approved: bool
    violations: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------


def _get_nsfw_classifier() -> Any:
    """Return the cached NSFW image-classification pipeline."""
    global _nsfw_classifier
    if _nsfw_classifier is not None:
        return _nsfw_classifier
    with _model_lock:
        if _nsfw_classifier is not None:
            return _nsfw_classifier
        logger.info("Loading NSFW classifier", extra={"model": "Falconsai/nsfw_image_classification"})
        _nsfw_classifier = hf_pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_classification",
            device=0 if _config.device == "cuda" and torch.cuda.is_available() else -1,
        )
        return _nsfw_classifier


def _get_policy_classifier() -> Any:
    """Return the cached multi-label violence/hate classifier."""
    global _policy_classifier
    if _policy_classifier is not None:
        return _policy_classifier
    with _model_lock:
        if _policy_classifier is not None:
            return _policy_classifier
        logger.info("Loading policy classifier", extra={"model": "michellejieli/NSFW_text_classifier"})
        # Zero-shot image classification for violence / hate / self-harm categories.
        # Uses CLIP backbone which aligns with the pipeline-stages doc.
        _policy_classifier = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-large-patch14",
            device=0 if _config.device == "cuda" and torch.cuda.is_available() else -1,
        )
        return _policy_classifier


def _ensure_face_detector_models() -> tuple[str, str]:
    """Download OpenCV DNN face detector model files if not present.

    Returns:
        Tuple of (proto_path, model_path) on disk.
    """
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    proto_path = models_dir / "deploy.prototxt"
    model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    proto_base = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector"
    model_base = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830"

    if not proto_path.exists():
        logger.info("Downloading face detector prototxt")
        import urllib.request

        urllib.request.urlretrieve(f"{proto_base}/deploy.prototxt", str(proto_path))

    if not model_path.exists():
        logger.info("Downloading face detector caffemodel (~10 MB)")
        import urllib.request

        urllib.request.urlretrieve(
            f"{model_base}/res10_300x300_ssd_iter_140000.caffemodel",
            str(model_path),
        )

    return str(proto_path), str(model_path)


def _get_face_detector() -> cv2.dnn.Net:
    """Return the cached OpenCV DNN face detector (Caffe SSD)."""
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    with _model_lock:
        if _face_detector is not None:
            return _face_detector
        logger.info("Loading OpenCV DNN face detector")
        proto_path, model_path = _ensure_face_detector_models()
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        if _config.device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        _face_detector = net
        return _face_detector


def _is_ocr_available() -> bool:
    """Check whether pytesseract is importable."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pytesseract as _pt  # noqa: F401

        _ocr_available = True
    except ImportError:
        logger.warning("pytesseract not installed — OCR-based contact detection disabled")
        _ocr_available = False
    return _ocr_available


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------


def _load_image(media_path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(media_path).convert("RGB")
    return img


def _extract_keyframes(video_path: str, fps_sample: float = 1.0) -> list[Image.Image]:
    """Sample keyframes from a video at *fps_sample* frames per second.

    Args:
        video_path: Path to the video file.
        fps_sample: Number of frames to sample per second of video.

    Returns:
        List of PIL Images (RGB) representing the sampled keyframes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps_sample))

    keyframes: list[Image.Image] = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV returns BGR; convert to RGB for PIL / transformers
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keyframes.append(Image.fromarray(rgb))
        frame_idx += frame_interval

    cap.release()

    if not keyframes:
        raise ValueError(f"No keyframes extracted from video: {video_path}")

    logger.info(
        "Extracted keyframes",
        extra={"video_path": video_path, "count": len(keyframes)},
    )
    return keyframes


# ---------------------------------------------------------------------------
# Scan stages
# ---------------------------------------------------------------------------


def _classify_nsfw(
    frames: list[Image.Image],
    threshold: float,
    violations: list[str],
    scores: dict[str, float],
) -> None:
    """Run NSFW classification on one or more frames.

    Records the maximum NSFW probability across all frames in
    ``scores["nsfw"]``.  Appends ``"nsfw_violation"`` if the score
    exceeds *threshold*.
    """
    classifier = _get_nsfw_classifier()
    max_nsfw: float = 0.0

    for frame in frames:
        results: list[dict[str, Any]] = classifier(frame)
        for entry in results:
            label = entry.get("label", "").lower()
            score = float(entry.get("score", 0.0))
            if label == "nsfw":
                max_nsfw = max(max_nsfw, score)

    scores["nsfw"] = round(max_nsfw, 4)

    if max_nsfw >= threshold:
        violations.append("nsfw_violation")
        logger.warning(
            "NSFW violation detected",
            extra={"score": max_nsfw, "threshold": threshold},
        )


def _detect_faces(
    frames: list[Image.Image],
    blocklist_embeddings: list[np.ndarray] | None,
    confidence_threshold: float,
    violations: list[str],
    scores: dict[str, float],
) -> None:
    """Detect faces and optionally match against a blocklist.

    Records the total face count across all frames in
    ``scores["face_count"]``.  If a blocklist is provided and a face
    matches, appends ``"blocked_identity"`` to *violations*.
    """
    detector = _get_face_detector()
    total_faces = 0

    for frame in frames:
        img_array = np.array(frame)
        h, w = img_array.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        detector.setInput(blob)
        detections = detector.forward()

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence_threshold:
                continue
            total_faces += 1

            # Blocklist matching (if embeddings are provided)
            if blocklist_embeddings:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                # Clamp coordinates to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    face_crop = img_array[y1:y2, x1:x2]
                    if _match_blocklist(face_crop, blocklist_embeddings):
                        violations.append("blocked_identity")
                        logger.warning("Blocked identity detected in content")

    scores["face_count"] = float(total_faces)
    logger.info("Face detection complete", extra={"face_count": total_faces})


def _match_blocklist(
    face_crop: np.ndarray,
    blocklist_embeddings: list[np.ndarray],
    similarity_threshold: float = 0.6,
) -> bool:
    """Compare a face crop against blocklist embeddings using cosine similarity.

    Uses a simple resize-and-flatten approach as a lightweight proxy when
    a full face-recognition model (e.g. ArcFace) is not loaded.  In
    production this should be replaced with a proper embedding model.

    Returns:
        ``True`` if any blocklist entry exceeds the similarity threshold.
    """
    resized = cv2.resize(face_crop, (112, 112)).astype(np.float32).flatten()
    norm = np.linalg.norm(resized)
    if norm == 0:
        return False
    embedding = resized / norm

    for ref in blocklist_embeddings:
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            continue
        cosine_sim = float(np.dot(embedding, ref / ref_norm))
        if cosine_sim >= similarity_threshold:
            return True
    return False


def _check_content_policy(
    frames: list[Image.Image],
    policy: dict[str, Any],
    violations: list[str],
    scores: dict[str, float],
) -> None:
    """Run content-policy checks: OCR contact-info detection and
    violence/hate zero-shot classification.
    """
    _check_contact_info(frames, violations, scores)
    _check_violence_hate(frames, policy, violations, scores)


def _check_contact_info(
    frames: list[Image.Image],
    violations: list[str],
    scores: dict[str, float],
) -> None:
    """Detect prohibited contact information via OCR on frames."""
    if not _is_ocr_available():
        scores["contact_info"] = 0.0
        return

    import pytesseract  # noqa: WPS433 (conditional import)

    detected = False
    for frame in frames:
        try:
            text: str = pytesseract.image_to_string(frame, timeout=5)
        except Exception:
            logger.debug("OCR failed on frame, skipping", exc_info=True)
            continue

        for pattern in _CONTACT_PATTERNS:
            if pattern.search(text):
                detected = True
                break
        if detected:
            break

    scores["contact_info"] = 1.0 if detected else 0.0
    if detected:
        violations.append("contact_info_detected")
        logger.warning("Contact information detected in content via OCR")


def _check_violence_hate(
    frames: list[Image.Image],
    policy: dict[str, Any],
    violations: list[str],
    scores: dict[str, float],
) -> None:
    """Zero-shot classification for violence, hate, and self-harm content."""
    candidate_labels = policy.get(
        "policy_categories",
        ["violence", "hate symbols", "self-harm", "child exploitation"],
    )
    category_threshold: float = policy.get("policy_category_threshold", 0.70)

    classifier = _get_policy_classifier()

    # Aggregate max score per category across all frames
    category_scores: dict[str, float] = {label: 0.0 for label in candidate_labels}

    for frame in frames:
        try:
            results: list[dict[str, Any]] = classifier(frame, candidate_labels=candidate_labels)
        except Exception:
            logger.debug("Policy classifier failed on frame, skipping", exc_info=True)
            continue

        for entry in results:
            label = entry.get("label", "")
            score = float(entry.get("score", 0.0))
            if label in category_scores:
                category_scores[label] = max(category_scores[label], score)

    for label, score in category_scores.items():
        key = f"policy_{label.replace(' ', '_')}"
        scores[key] = round(score, 4)
        if score >= category_threshold:
            violations.append(f"policy_{label.replace(' ', '_')}")
            logger.warning(
                "Policy violation detected",
                extra={"category": label, "score": score, "threshold": category_threshold},
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(
    media_path: str,
    media_type: Literal["image", "video"],
    policy: dict[str, Any] | None = None,
) -> ScanResult:
    """Run content moderation checks on a media asset.

    Executes the full post-generation scanning pipeline:
      1. NSFW classification
      2. Face detection (with optional blocklist)
      3. Content policy checks (OCR contact info, violence/hate)

    Args:
        media_path: Local filesystem path to the image or video file.
        media_type: Either ``"image"`` or ``"video"``.
        policy: Optional policy overrides.  Recognised keys:

            - ``nsfw_threshold`` (float): Override the default 0.85 threshold.
            - ``face_confidence`` (float): Min confidence for face detection
              (default 0.5).
            - ``blocklist_embeddings`` (list[ndarray]): Reference embeddings
              of protected identities.
            - ``policy_categories`` (list[str]): Zero-shot categories to scan.
            - ``policy_category_threshold`` (float): Score threshold for
              category violations (default 0.70).
            - ``keyframe_fps`` (float): Keyframe sampling rate for video
              (default 1.0).

    Returns:
        A :class:`ScanResult` indicating whether the content is approved and
        listing any detected violations with associated scores.

    Raises:
        FileNotFoundError: If *media_path* does not exist.
        ValueError: If *media_type* is not ``"image"`` or ``"video"``, or if
            a video cannot be opened.
    """
    policy = policy or {}
    violations: list[str] = []
    scores: dict[str, float] = {}

    # Validate path
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    logger.info(
        "Starting content moderation scan",
        extra={"media_path": media_path, "media_type": media_type},
    )

    # ---- Extract frames ----
    if media_type == "image":
        frames = [_load_image(media_path)]
    elif media_type == "video":
        keyframe_fps: float = policy.get("keyframe_fps", 1.0)
        frames = _extract_keyframes(media_path, fps_sample=keyframe_fps)
    else:
        raise ValueError(f"Unsupported media_type: {media_type!r}")

    # ---- 1. NSFW classification ----
    nsfw_threshold: float = policy.get("nsfw_threshold", _config.nsfw_threshold)
    try:
        _classify_nsfw(frames, nsfw_threshold, violations, scores)
    except Exception:
        logger.error("NSFW classification failed — treating as violation", exc_info=True)
        violations.append("nsfw_classification_error")
        scores["nsfw"] = -1.0

    # ---- 2. Face detection ----
    face_confidence: float = policy.get("face_confidence", 0.5)
    blocklist: list[np.ndarray] | None = policy.get("blocklist_embeddings")
    try:
        _detect_faces(frames, blocklist, face_confidence, violations, scores)
    except Exception:
        logger.error("Face detection failed", exc_info=True)
        scores["face_count"] = -1.0

    # ---- 3. Content policy checks ----
    try:
        _check_content_policy(frames, policy, violations, scores)
    except Exception:
        logger.error("Content policy check failed — treating as violation", exc_info=True)
        violations.append("policy_check_error")

    # ---- Build result ----
    approved = len(violations) == 0

    logger.info(
        "Scan complete",
        extra={
            "media_path": media_path,
            "media_type": media_type,
            "approved": approved,
            "violations": violations,
            "scores": scores,
        },
    )

    return ScanResult(approved=approved, violations=violations, scores=scores)


def scan_with_consent_check(
    media_path: str,
    media_type: Literal["image", "video"],
    performer_id: str,
    consent_base_url: str,
    policy: dict[str, Any] | None = None,
) -> ScanResult:
    """Run content moderation with a consent pre-check.

    Verifies consent is active before running the scan.  Per Critical
    Invariant #1, no content processing without consent verification.

    Args:
        media_path: Local filesystem path to the image or video file.
        media_type: Either ``"image"`` or ``"video"``.
        performer_id: UUID of the performer whose content is being scanned.
        consent_base_url: Base URL of the Consent Service.
        policy: Optional policy overrides (see :func:`scan`).

    Returns:
        A :class:`ScanResult`.

    Raises:
        ConsentDeniedError: If consent is not active.
        ConsentCheckUnavailableError: If the consent service is unreachable.
    """
    from shared.consent_client import ConsentClient

    consent_client = ConsentClient(base_url=consent_base_url)
    consent_client.check_consent(performer_id, media_type)
    return scan(media_path, media_type, policy)
