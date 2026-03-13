"""Step Functions handler for dual-track content generation."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from diffusers import (
    AnimateDiffPipeline,
    DDIMScheduler,
    MotionAdapter,
    StableDiffusionXLPipeline,
)
from PIL import Image, ImageEnhance, ImageFilter
from transformers import CLIPModel, CLIPProcessor

from shared.config import Config
from shared.s3_client import S3Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
CLIP_SCORE_THRESHOLD = 0.20
BRISQUE_THRESHOLD = 50.0  # lower is better; reject above this

# ---------------------------------------------------------------------------
# Video defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_DURATION_SECONDS = 4


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def _compute_clip_score(image: Image.Image, prompt: str, device: str) -> float:
    """Compute CLIP similarity score between an image and a text prompt.

    Args:
        image: PIL Image to evaluate.
        prompt: The text prompt used to generate the image.
        device: Torch device string (e.g. ``"cuda"``).

    Returns:
        Cosine similarity score between 0 and 1.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = (image_embeds @ text_embeds.T).squeeze().item()

    # Clean up to free GPU memory
    del model, processor
    torch.cuda.empty_cache()

    return float(score)


def _compute_brisque(image: Image.Image) -> float:
    """Compute BRISQUE no-reference image quality score using OpenCV.

    Lower values indicate better quality.

    Args:
        image: PIL Image to evaluate.

    Returns:
        BRISQUE quality score (float). Lower is better.
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    brisque = cv2.quality.QualityBRISQUE_compute(
        img_array,
        cv2.quality.QualityBRISQUE_computeFeatures(img_array),
    )
    # QualityBRISQUE_compute returns a tuple; take the first element
    return float(brisque[0])


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _postprocess_image(image: Image.Image) -> Image.Image:
    """Apply colour correction and sharpening to a generated image.

    Args:
        image: Raw PIL Image from the diffusion pipeline.

    Returns:
        Post-processed PIL Image.
    """
    # Auto-contrast for colour correction
    image = image.convert("RGB")

    # Mild contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)

    # Mild colour saturation boost
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.05)

    # Unsharp mask sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    return image


def _postprocess_video_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply colour correction and sharpening to video frames.

    Args:
        frames: List of BGR numpy arrays from the video pipeline.

    Returns:
        Post-processed frames.
    """
    processed: list[np.ndarray] = []
    for frame in frames:
        # Colour correction via CLAHE on L channel (LAB space)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Mild sharpening kernel
        kernel = np.array(
            [
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0],
            ],
            dtype=np.float32,
        )
        frame = cv2.filter2D(frame, -1, kernel)

        processed.append(frame)
    return processed


def _interpolate_frames(
    frames: list[np.ndarray],
    source_fps: int,
    target_fps: int,
) -> list[np.ndarray]:
    """Linearly interpolate frames to reach target FPS if needed.

    Args:
        frames: Source frames (BGR numpy arrays).
        source_fps: FPS the frames were generated at.
        target_fps: Desired output FPS.

    Returns:
        Interpolated frame list. Returns original list if no interpolation is needed.
    """
    if target_fps <= source_fps or len(frames) < 2:
        return frames

    ratio = target_fps / source_fps
    interpolated: list[np.ndarray] = []
    for i in range(len(frames) - 1):
        interpolated.append(frames[i])
        steps = int(ratio) - 1
        for s in range(1, steps + 1):
            alpha = s / (steps + 1)
            blended = cv2.addWeighted(frames[i], 1.0 - alpha, frames[i + 1], alpha, 0)
            interpolated.append(blended)
    interpolated.append(frames[-1])
    return interpolated


def _encode_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int,
) -> None:
    """Encode a list of BGR frames into an MP4 file.

    Args:
        frames: List of BGR numpy arrays to encode.
        output_path: Filesystem path for the output MP4.
        fps: Output frame rate.
    """
    if not frames:
        raise ValueError("Cannot encode video with zero frames")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_image_pipeline(
    base_model_id: str,
    lora_local_path: str,
    device: str,
) -> StableDiffusionXLPipeline:
    """Load SDXL pipeline with LoRA weights fused in.

    Args:
        base_model_id: HuggingFace model ID for the base SDXL model.
        lora_local_path: Local filesystem path to the LoRA adapter weights.
        device: Torch device string.

    Returns:
        Ready-to-use ``StableDiffusionXLPipeline``.
    """
    logger.info("Loading SDXL base model: %s", base_model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    logger.info("Loading LoRA adapter from: %s", lora_local_path)
    pipe.load_lora_weights(lora_local_path)
    pipe.fuse_lora()

    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    logger.info("Image pipeline ready on %s", device)
    return pipe


def _load_video_pipeline(
    base_model_id: str,
    lora_local_path: str,
    device: str,
) -> AnimateDiffPipeline:
    """Load AnimateDiff pipeline with LoRA weights for video generation.

    Args:
        base_model_id: HuggingFace model ID for the base model.
        lora_local_path: Local filesystem path to the LoRA adapter weights.
        device: Torch device string.

    Returns:
        Ready-to-use ``AnimateDiffPipeline``.
    """
    logger.info("Loading motion adapter for AnimateDiff")
    motion_adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
    )

    logger.info("Loading AnimateDiff pipeline with base model: %s", base_model_id)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model_id,
        motion_adapter=motion_adapter,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        base_model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    logger.info("Loading LoRA adapter from: %s", lora_local_path)
    pipe.load_lora_weights(lora_local_path)
    pipe.fuse_lora()

    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    logger.info("Video pipeline ready on %s", device)
    return pipe


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------


def _generate_thumbnail(image: Image.Image, max_size: int = 256) -> Image.Image:
    """Create a thumbnail from an image.

    Args:
        image: Source PIL Image.
        max_size: Maximum dimension for the thumbnail.

    Returns:
        Thumbnail PIL Image.
    """
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return thumb


def _extract_video_thumbnail(video_path: str, max_size: int = 256) -> Image.Image:
    """Extract a thumbnail from the first frame of a video file.

    Args:
        video_path: Path to the MP4 file.
        max_size: Maximum dimension for the thumbnail.

    Returns:
        Thumbnail PIL Image.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from video")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Step Functions task handler for content generation.

    Supports two media tracks:
      - **image**: single-frame diffusion inference.
      - **video**: multi-frame temporal diffusion inference.

    Args:
        event: Step Functions input containing at minimum:
            - job_id: Unique identifier for this generation job.
            - creator_id: ID of the creator requesting content.
            - model_s3_path: S3 path to the fine-tuned LoRA model.
            - prompt: Text prompt for generation.
            - media_type: Either ``"image"`` or ``"video"``.
            - generation_params: Optional dict of inference settings
              (guidance_scale, num_inference_steps, seed, resolution,
              duration_seconds for video, fps for video, etc.).
        context: Lambda/Step Functions context object.

    Returns:
        Dict with job status and output artifact location.

    Raises:
        ValueError: If ``media_type`` is not ``"image"`` or ``"video"``.
        RuntimeError: If quality validation fails (caller should retry or refund).
    """
    start_time = time.monotonic()

    job_id: str = event["job_id"]
    creator_id: str = event["creator_id"]
    model_s3_path: str = event["model_s3_path"]
    prompt: str = event["prompt"]
    media_type: Literal["image", "video"] = event["media_type"]
    generation_params: dict[str, Any] = event.get("generation_params", {})

    logger.info(
        "Starting generation job=%s creator=%s media_type=%s",
        job_id,
        creator_id,
        media_type,
    )

    config = Config()
    s3 = S3Client(bucket=config.s3_bucket, region=config.aws_region)

    # ------------------------------------------------------------------
    # Stage 1: Load model from S3
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix=f"gen-{job_id}-") as tmp_dir:
        lora_local_dir = str(Path(tmp_dir) / "lora")
        Path(lora_local_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Downloading LoRA adapter from s3://%s/%s", config.s3_bucket, model_s3_path)
        lora_files = s3.list_objects(model_s3_path)
        if not lora_files:
            raise RuntimeError(f"No model files found at s3://{config.s3_bucket}/{model_s3_path}")
        for s3_key in lora_files:
            filename = Path(s3_key).name
            local_path = str(Path(lora_local_dir) / filename)
            s3.download_file(s3_key, local_path)
        logger.info("Downloaded %d model files to %s", len(lora_files), lora_local_dir)

        # ------------------------------------------------------------------
        # Stage 2: Run inference (branching on media_type)
        # ------------------------------------------------------------------
        guidance_scale = generation_params.get(
            "guidance_scale",
            config.default_guidance_scale,
        )
        num_inference_steps = generation_params.get(
            "num_inference_steps",
            config.default_num_inference_steps,
        )
        seed = generation_params.get("seed", int(uuid.uuid4().int % (2**32)))
        generator = torch.Generator(device=config.device).manual_seed(seed)

        result_image: Image.Image | None = None
        video_output_path: str | None = None

        if media_type == "image":
            width = generation_params.get("width", config.image_resolution)
            height = generation_params.get("height", config.image_resolution)

            pipe = _load_image_pipeline(
                base_model_id=config.base_model_id,
                lora_local_path=lora_local_dir,
                device=config.device,
            )

            logger.info(
                "Running image inference: steps=%d guidance=%.1f size=%dx%d seed=%d",
                num_inference_steps,
                guidance_scale,
                width,
                height,
                seed,
            )
            output = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
            result_image = output.images[0]

            # Free pipeline from GPU
            del pipe
            torch.cuda.empty_cache()

            # Stage 3: Post-processing (image)
            logger.info("Post-processing image")
            result_image = _postprocess_image(result_image)

        elif media_type == "video":
            duration_seconds = generation_params.get(
                "duration_seconds",
                DEFAULT_VIDEO_DURATION_SECONDS,
            )
            fps = generation_params.get("fps", DEFAULT_VIDEO_FPS)
            num_frames = duration_seconds * fps
            target_fps = generation_params.get("target_fps", fps)

            pipe = _load_video_pipeline(
                base_model_id=config.base_model_id,
                lora_local_path=lora_local_dir,
                device=config.device,
            )

            logger.info(
                "Running video inference: frames=%d fps=%d guidance=%.1f seed=%d",
                num_frames,
                fps,
                guidance_scale,
                seed,
            )
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            # output.frames is a list of lists of PIL Images
            pil_frames: list[Image.Image] = output.frames[0]

            # Free pipeline from GPU
            del pipe
            torch.cuda.empty_cache()

            # Convert PIL frames to BGR numpy arrays for post-processing
            bgr_frames = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) for f in pil_frames]

            # Stage 3: Post-processing (video)
            logger.info("Post-processing video frames")
            bgr_frames = _postprocess_video_frames(bgr_frames)

            # Frame interpolation if target FPS exceeds generation FPS
            if target_fps > fps:
                logger.info(
                    "Interpolating frames from %d fps to %d fps",
                    fps,
                    target_fps,
                )
                bgr_frames = _interpolate_frames(bgr_frames, fps, target_fps)
                fps = target_fps

            # Encode to MP4
            video_output_path = str(Path(tmp_dir) / "output.mp4")
            logger.info("Encoding %d frames to MP4 at %d fps", len(bgr_frames), fps)
            _encode_video(bgr_frames, video_output_path, fps)

            # Extract first frame as the reference image for quality validation
            first_frame_rgb = cv2.cvtColor(bgr_frames[0], cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(first_frame_rgb)

        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

        # ------------------------------------------------------------------
        # Stage 4: Quality validation
        # ------------------------------------------------------------------
        assert result_image is not None, "result_image must be set after inference"

        logger.info("Running quality validation")
        clip_score = _compute_clip_score(result_image, prompt, config.device)
        logger.info("CLIP score: %.4f (threshold: %.4f)", clip_score, CLIP_SCORE_THRESHOLD)

        if clip_score < CLIP_SCORE_THRESHOLD:
            raise RuntimeError(
                f"Quality validation failed: CLIP score {clip_score:.4f} "
                f"below threshold {CLIP_SCORE_THRESHOLD:.4f} for job {job_id}"
            )

        brisque_score = _compute_brisque(result_image)
        logger.info("BRISQUE score: %.2f (threshold: %.2f)", brisque_score, BRISQUE_THRESHOLD)

        if brisque_score > BRISQUE_THRESHOLD:
            raise RuntimeError(
                f"Quality validation failed: BRISQUE score {brisque_score:.2f} "
                f"above threshold {BRISQUE_THRESHOLD:.2f} for job {job_id}"
            )

        # ------------------------------------------------------------------
        # Stage 5: Upload to S3
        # ------------------------------------------------------------------
        output_prefix = f"outputs/{creator_id}/{job_id}"

        if media_type == "image":
            # Save image to temp file and upload
            image_path = str(Path(tmp_dir) / "output.png")
            result_image.save(image_path, format="PNG", optimize=True)
            output_key = f"{output_prefix}/output.png"
            logger.info("Uploading image to s3://%s/%s", config.s3_bucket, output_key)
            s3.upload_file(image_path, output_key)
        else:
            assert video_output_path is not None
            output_key = f"{output_prefix}/output.mp4"
            logger.info("Uploading video to s3://%s/%s", config.s3_bucket, output_key)
            s3.upload_file(video_output_path, output_key)

        # Upload thumbnail
        thumbnail = (
            _generate_thumbnail(result_image) if media_type == "image" else _extract_video_thumbnail(video_output_path)  # type: ignore[arg-type]
        )
        thumbnail_path = str(Path(tmp_dir) / "thumbnail.png")
        thumbnail.save(thumbnail_path, format="PNG")
        thumbnail_key = f"{output_prefix}/thumbnail.png"
        logger.info("Uploading thumbnail to s3://%s/%s", config.s3_bucket, thumbnail_key)
        s3.upload_file(thumbnail_path, thumbnail_key)

        # Upload metadata JSON
        elapsed = time.monotonic() - start_time
        metadata = {
            "job_id": job_id,
            "creator_id": creator_id,
            "media_type": media_type,
            "prompt": prompt,
            "generation_params": {
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                **({"width": width, "height": height} if media_type == "image" else {}),
                **(
                    {
                        "duration_seconds": duration_seconds,
                        "fps": fps,
                        "num_frames": num_frames,
                    }
                    if media_type == "video"
                    else {}
                ),
            },
            "quality_metrics": {
                "clip_score": clip_score,
                "brisque_score": brisque_score,
            },
            "output_s3_key": output_key,
            "thumbnail_s3_key": thumbnail_key,
            "generation_time_seconds": round(elapsed, 2),
        }
        metadata_path = str(Path(tmp_dir) / "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        metadata_key = f"{output_prefix}/metadata.json"
        logger.info("Uploading metadata to s3://%s/%s", config.s3_bucket, metadata_key)
        s3.upload_file(metadata_path, metadata_key)

    logger.info(
        "Generation complete for job=%s media_type=%s elapsed=%.2fs",
        job_id,
        media_type,
        elapsed,
    )

    return {
        "job_id": job_id,
        "status": "COMPLETED",
        "media_type": media_type,
        "output_s3_key": output_key,
        "thumbnail_s3_key": thumbnail_key,
        "metadata_s3_key": metadata_key,
        "quality_metrics": {
            "clip_score": clip_score,
            "brisque_score": brisque_score,
        },
        "generation_time_seconds": round(elapsed, 2),
    }
