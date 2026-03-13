"""Step Functions handler for LoRA/DreamBooth fine-tuning pipeline.

Orchestrates the full avatar training pipeline:
1. Download training images from S3
2. Preprocess (face detection, alignment, augmentation, captioning)
3. Train LoRA adapters on SDXL base model
4. Quality validation via CLIP similarity
5. Upload artifacts to S3
6. Register model in DynamoDB
7. Send completion event to SQS
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from transformers import BlipForConditionalGeneration, BlipProcessor

from shared.config import Config
from shared.dynamodb_client import DynamoDBClient
from shared.logger import get_logger
from shared.s3_client import S3Client
from shared.sqs_consumer import SqsConsumer

logger = get_logger(__name__)

# Minimum number of training images required to start a fine-tuning job.
MIN_TRAINING_IMAGES = 10

# Image file extensions accepted for training.
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class TrainingPipelineError(Exception):
    """Raised when any stage of the training pipeline fails."""


# ---------------------------------------------------------------------------
# Stage 1: Download training images from S3
# ---------------------------------------------------------------------------


def download_training_images(
    s3_client: S3Client,
    s3_prefix: str,
    local_dir: Path,
) -> list[Path]:
    """Download training images from S3 and validate minimum count.

    Args:
        s3_client: Configured S3 client.
        s3_prefix: S3 key prefix where training images are stored.
        local_dir: Local directory to save downloaded images.

    Returns:
        List of local file paths to the downloaded images.

    Raises:
        TrainingPipelineError: If fewer than ``MIN_TRAINING_IMAGES`` valid
            images are found under the prefix.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    keys = s3_client.list_objects(prefix=s3_prefix)

    image_keys = [k for k in keys if Path(k).suffix.lower() in VALID_IMAGE_EXTENSIONS]

    logger.info(
        "Found training images in S3",
        extra={"prefix": s3_prefix, "total_keys": len(keys), "image_keys": len(image_keys)},
    )

    if len(image_keys) < MIN_TRAINING_IMAGES:
        raise TrainingPipelineError(
            f"Insufficient training images: found {len(image_keys)}, minimum required is {MIN_TRAINING_IMAGES}"
        )

    downloaded: list[Path] = []
    for key in image_keys:
        filename = Path(key).name
        local_path = local_dir / filename
        s3_client.download_file(s3_key=key, local_path=str(local_path))
        downloaded.append(local_path)

    logger.info("Downloaded training images", extra={"count": len(downloaded)})
    return downloaded


# ---------------------------------------------------------------------------
# Stage 2: Preprocess training images
# ---------------------------------------------------------------------------


def _detect_and_align_face(
    image_path: Path,
    face_app: FaceAnalysis,
    target_size: int,
) -> np.ndarray | None:
    """Detect the dominant face in an image, crop, and resize.

    Uses InsightFace ArcFace for face detection and alignment. If no face is
    detected with sufficient confidence, the image is skipped.

    Args:
        image_path: Path to the source image file.
        face_app: Initialised InsightFace application.
        target_size: Output resolution (square).

    Returns:
        A numpy array (H, W, 3) BGR of the aligned face crop, or ``None``
        if no suitable face was found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image", extra={"path": str(image_path)})
        return None

    faces = face_app.get(img)
    if not faces:
        logger.warning("No face detected", extra={"path": str(image_path)})
        return None

    # Use the face with the highest detection score.
    best_face = max(faces, key=lambda f: f.det_score)
    if best_face.det_score < 0.7:
        logger.warning(
            "Face confidence too low",
            extra={"path": str(image_path), "score": float(best_face.det_score)},
        )
        return None

    # Crop to bounding box with padding.
    bbox = best_face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    # Add 20% padding around the face for context.
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return resized


def _generate_caption(
    image: Image.Image,
    blip_processor: BlipProcessor,
    blip_model: BlipForConditionalGeneration,
    device: str,
    subject_token: str = "sks person",
) -> str:
    """Generate an identity-conditioned caption for a training image.

    Uses BLIP for a base description, then prepends an identity token
    so the LoRA learns to associate the token with the performer's
    appearance (DreamBooth-style).

    Args:
        image: PIL Image to caption.
        blip_processor: BLIP preprocessor.
        blip_model: BLIP captioning model.
        device: Torch device string.
        subject_token: Unique subject identifier for identity conditioning.

    Returns:
        Caption string prefixed with the subject token.
    """
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)
    raw_caption: str = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    # Prefix with subject token for identity conditioning.
    return f"a photo of {subject_token}, {raw_caption}"


def preprocess_images(
    image_paths: list[Path],
    output_dir: Path,
    target_size: int,
    device: str,
    subject_token: str = "sks person",
) -> list[dict[str, Any]]:
    """Preprocess training images: face detect, align, augment, and caption.

    Args:
        image_paths: Paths to raw training images.
        output_dir: Directory to write preprocessed images and captions.
        target_size: Target resolution for training images (square).
        device: Torch device for captioning model.
        subject_token: Unique identifier for identity-conditioned captions.

    Returns:
        List of dicts with keys ``image_path`` and ``caption`` for each
        successfully preprocessed image.

    Raises:
        TrainingPipelineError: If fewer than ``MIN_TRAINING_IMAGES`` images
            pass preprocessing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise InsightFace.
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    # Initialise BLIP captioning model.
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()

    # Augmentation transforms (applied to create additional variants).
    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    )

    processed: list[dict[str, Any]] = []

    for idx, img_path in enumerate(image_paths):
        aligned = _detect_and_align_face(img_path, face_app, target_size)
        if aligned is None:
            continue

        # Convert BGR (OpenCV) to RGB PIL Image.
        pil_image = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

        # Save the original aligned image.
        out_path = output_dir / f"img_{idx:04d}.png"
        pil_image.save(str(out_path))

        # Generate identity-conditioned caption.
        caption = _generate_caption(pil_image, blip_processor, blip_model, device, subject_token)

        # Save caption alongside image.
        caption_path = output_dir / f"img_{idx:04d}.txt"
        caption_path.write_text(caption, encoding="utf-8")

        processed.append({"image_path": str(out_path), "caption": caption})

        # Create one augmented variant per original.
        aug_image = augmentation(pil_image)
        aug_path = output_dir / f"img_{idx:04d}_aug.png"
        aug_image.save(str(aug_path))
        aug_caption_path = output_dir / f"img_{idx:04d}_aug.txt"
        aug_caption_path.write_text(caption, encoding="utf-8")

        processed.append({"image_path": str(aug_path), "caption": caption})

    logger.info("Preprocessing complete", extra={"processed_count": len(processed)})

    if len(processed) < MIN_TRAINING_IMAGES:
        raise TrainingPipelineError(
            f"Too few images passed preprocessing: {len(processed)} (minimum {MIN_TRAINING_IMAGES})"
        )

    return processed


# ---------------------------------------------------------------------------
# Stage 3: Train LoRA model
# ---------------------------------------------------------------------------


def train_lora(
    preprocessed: list[dict[str, Any]],
    config: Config,
    output_dir: Path,
    training_params: dict[str, Any],
) -> dict[str, Any]:
    """Fine-tune LoRA adapters on the SDXL base model.

    Produces safetensors weights that work for both image and video generation
    (shared LoRA model as per dual-track strategy).

    Args:
        preprocessed: List of dicts with ``image_path`` and ``caption``.
        config: ML pipeline configuration.
        output_dir: Directory to save LoRA weights.
        training_params: Override hyper-parameters from the Step Functions event.

    Returns:
        Dict of training metrics (``final_loss``, ``steps_completed``,
        ``learning_rate``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config.device

    lora_rank = training_params.get("lora_rank", config.lora_rank)
    training_steps = training_params.get("training_steps", config.training_steps)
    learning_rate = training_params.get("learning_rate", config.learning_rate)
    resolution = training_params.get("resolution", config.image_resolution)

    logger.info(
        "Starting LoRA training",
        extra={
            "lora_rank": lora_rank,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "num_images": len(preprocessed),
        },
    )

    # Load base SDXL pipeline.
    base_model_id = training_params.get("base_model", config.base_model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )

    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    vae = pipe.vae.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    # Freeze base model parameters.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Attach LoRA adapters to the UNet.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Set up optimizer for LoRA parameters only.
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    # Cosine annealing LR scheduler for smoother convergence.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps, eta_min=learning_rate * 0.01
    )

    # Training batch size — higher gives better gradient estimates on GPU.
    batch_size = training_params.get("batch_size", 4 if device == "cuda" else 1)

    # Build a simple dataset from preprocessed images.
    image_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images_tensors: list[torch.Tensor] = []
    captions: list[str] = []
    for item in preprocessed:
        img = Image.open(item["image_path"]).convert("RGB")
        images_tensors.append(image_transform(img))
        captions.append(item["caption"])

    dataset_size = len(images_tensors)

    # Training loop with mixed precision and batching.
    unet.train()
    total_loss = 0.0
    loss_count = 0
    use_amp = device == "cuda"

    for step in range(training_steps):
        # Build a batch by sampling from the dataset.
        batch_indices = [(step * batch_size + i) % dataset_size for i in range(batch_size)]
        pixel_values = torch.stack([images_tensors[i] for i in batch_indices]).to(device, dtype=vae.dtype)
        batch_captions = [captions[i] for i in batch_indices]

        # Encode images to latent space.
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise and timesteps.
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode text prompts with both SDXL text encoders.
        tokens_1 = tokenizer(
            batch_captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        tokens_2 = tokenizer_2(
            batch_captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(tokens_1, output_hidden_states=True)
            encoder_hidden_states_2 = text_encoder_2(tokens_2, output_hidden_states=True)
            pooled_output = encoder_hidden_states_2[0]
            text_embeds = torch.cat(
                [
                    encoder_hidden_states.hidden_states[-2],
                    encoder_hidden_states_2.hidden_states[-2],
                ],
                dim=-1,
            )

        # Build added_cond_kwargs for SDXL.
        add_time_ids = torch.zeros((batch_size, 6), device=device, dtype=text_embeds.dtype)
        added_cond_kwargs = {
            "text_embeds": pooled_output,
            "time_ids": add_time_ids,
        }

        # Forward pass with mixed precision (fp16 on GPU).
        with torch.autocast(device_type="cuda", enabled=use_amp):
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Compute MSE loss against the target noise.
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        loss_count += 1

        if (step + 1) % 100 == 0 or step == 0:
            avg_loss = total_loss / loss_count
            current_lr = lr_scheduler.get_last_lr()[0]
            logger.info(
                "Training progress",
                extra={
                    "step": step + 1,
                    "total_steps": training_steps,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                },
            )

    # Save LoRA weights as safetensors.
    unet.save_pretrained(str(output_dir / "lora_weights"))

    # Save training config for reproducibility.
    training_config = {
        "base_model_id": base_model_id,
        "lora_rank": lora_rank,
        "training_steps": training_steps,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "num_training_images": dataset_size,
    }
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    final_loss = total_loss / max(loss_count, 1)
    metrics = {
        "final_loss": final_loss,
        "steps_completed": training_steps,
        "learning_rate": learning_rate,
    }

    logger.info("LoRA training complete", extra=metrics)
    return metrics


# ---------------------------------------------------------------------------
# Stage 4: Quality validation
# ---------------------------------------------------------------------------


def validate_model_quality(
    lora_dir: Path,
    config: Config,
    training_params: dict[str, Any],
    num_samples: int = 4,
) -> dict[str, Any]:
    """Generate sample images and compute CLIP similarity for quality gating.

    Args:
        lora_dir: Directory containing the saved LoRA weights.
        config: ML pipeline configuration.
        training_params: Hyper-parameters (may contain ``base_model``).
        num_samples: Number of sample images to generate for validation.

    Returns:
        Dict with ``clip_score`` (mean), ``sample_paths``, and ``passed``
        boolean.

    Raises:
        TrainingPipelineError: If quality validation fails critically.
    """
    from transformers import CLIPModel, CLIPProcessor

    device = config.device
    base_model_id = training_params.get("base_model", config.base_model_id)

    logger.info("Starting quality validation", extra={"num_samples": num_samples})

    # Load base pipeline with trained LoRA weights.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    ).to(device)
    pipe.load_lora_weights(str(lora_dir / "lora_weights"))

    # Load CLIP for similarity scoring.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    validation_prompts = [
        "a photo of sks person, high quality, detailed face",
        "a portrait of sks person, studio lighting",
        "sks person looking at camera, natural light",
        "a close-up photo of sks person, sharp focus",
    ]

    sample_dir = lora_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_paths: list[str] = []
    clip_scores: list[float] = []

    for i in range(min(num_samples, len(validation_prompts))):
        prompt = validation_prompts[i]

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=config.default_num_inference_steps,
                guidance_scale=config.default_guidance_scale,
                height=config.image_resolution,
                width=config.image_resolution,
            )
        generated_image = result.images[0]

        sample_path = sample_dir / f"sample_{i:02d}.png"
        generated_image.save(str(sample_path))
        sample_paths.append(str(sample_path))

        # Compute CLIP similarity between prompt and generated image.
        inputs = clip_processor(text=[prompt], images=generated_image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # Cosine similarity normalised to [0, 1].
            score = outputs.logits_per_image.item() / 100.0
            clip_scores.append(score)

        logger.info(
            "Generated validation sample",
            extra={"sample_index": i, "clip_score": score, "prompt": prompt},
        )

    # Clean up pipeline to free GPU memory.
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    mean_clip = float(np.mean(clip_scores)) if clip_scores else 0.0
    # A CLIP similarity threshold of 0.25 is a reasonable baseline for
    # identity-conditioned generation.
    passed = mean_clip >= 0.25

    metrics = {
        "clip_score": mean_clip,
        "clip_scores": clip_scores,
        "sample_paths": sample_paths,
        "passed": passed,
    }

    logger.info("Quality validation complete", extra=metrics)

    if not passed:
        raise TrainingPipelineError(
            f"Quality validation failed: mean CLIP score {mean_clip:.4f} is below threshold 0.25"
        )

    return metrics


# ---------------------------------------------------------------------------
# Stage 5: Upload artifacts to S3
# ---------------------------------------------------------------------------


def upload_artifacts(
    s3_client: S3Client,
    local_dir: Path,
    creator_id: str,
    job_id: str,
) -> str:
    """Upload trained model artifacts to S3.

    Uploads LoRA weights, training config, and sample outputs under the
    deterministic key scheme ``models/{creator_id}/{job_id}/``.

    Args:
        s3_client: Configured S3 client.
        local_dir: Local directory containing the artifacts.
        creator_id: Creator UUID.
        job_id: Training job UUID.

    Returns:
        The S3 key prefix where artifacts were uploaded.
    """
    s3_prefix = f"models/{creator_id}/{job_id}/"

    uploaded_count = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}{relative.as_posix()}"
            s3_client.upload_file(local_path=str(file_path), s3_key=s3_key)
            uploaded_count += 1

    logger.info(
        "Uploaded artifacts to S3",
        extra={"s3_prefix": s3_prefix, "file_count": uploaded_count},
    )
    return s3_prefix


# ---------------------------------------------------------------------------
# Stage 6: Register model in DynamoDB
# ---------------------------------------------------------------------------


def register_model(
    dynamodb_client: DynamoDBClient,
    model_id: str,
    creator_id: str,
    job_id: str,
    s3_path: str,
    quality_metrics: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    """Register the trained model in the DynamoDB model registry.

    Sets initial status to ``validating`` (pending performer approval).
    Tags with ``consent_status=active`` per the consent gate invariant.
    Records ``supported_media_types`` as both image and video since LoRA
    weights are shared across tracks.

    Args:
        dynamodb_client: Configured DynamoDB client.
        model_id: Unique model identifier.
        creator_id: Creator/performer UUID.
        job_id: Training job UUID.
        s3_path: S3 prefix where model artifacts are stored.
        quality_metrics: Quality validation results (CLIP scores, etc.).
        training_config: Training hyper-parameters used.
    """
    dynamodb_client.put_item(
        {
            "model_id": model_id,
            "performer_id": creator_id,
            "job_id": job_id,
            "version": 1,
            "status": "validating",
            "consent_status": "active",
            "supported_media_types": ["image", "video"],
            "s3_path": s3_path,
            "quality_metrics": quality_metrics,
            "training_config": training_config,
        }
    )

    logger.info(
        "Model registered in registry",
        extra={"model_id": model_id, "status": "validating"},
    )


# ---------------------------------------------------------------------------
# Stage 7: Send completion event to SQS
# ---------------------------------------------------------------------------


def send_completion_event(
    sqs_client: SqsConsumer,
    job_id: str,
    creator_id: str,
    model_id: str,
    s3_path: str,
    metrics: dict[str, Any],
) -> None:
    """Publish a training-complete event to SQS.

    Downstream services (API, notification) subscribe to this queue to
    update job status and notify the performer.

    Args:
        sqs_client: SQS client configured with the training-complete queue.
        job_id: Training job UUID.
        creator_id: Creator/performer UUID.
        model_id: Unique model identifier.
        s3_path: S3 prefix of the uploaded artifacts.
        metrics: Training and quality validation metrics.
    """
    message = {
        "event_type": "training.completed",
        "job_id": job_id,
        "creator_id": creator_id,
        "model_id": model_id,
        "model_s3_path": s3_path,
        "supported_media_types": ["image", "video"],
        "metrics": {
            "final_loss": metrics.get("final_loss"),
            "clip_score": metrics.get("clip_score"),
            "steps_completed": metrics.get("steps_completed"),
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    sqs_client.send_message(message)
    logger.info(
        "Sent completion event to SQS",
        extra={"job_id": job_id, "model_id": model_id},
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Step Functions task handler for model fine-tuning.

    Orchestrates a LoRA/DreamBooth fine-tuning run from training images
    uploaded by the creator.

    Args:
        event: Step Functions input containing at minimum:
            - job_id: Unique identifier for this training job.
            - creator_id: ID of the creator requesting the model.
            - s3_image_prefix: S3 prefix where training images are stored.
            - base_model: Base diffusion model identifier.
            - training_params: Dict of hyper-parameters (steps, lr, rank, etc.).
        context: Lambda/Step Functions context object.

    Returns:
        Dict with job status and output artifact locations.
    """
    job_id: str = event["job_id"]
    creator_id: str = event["creator_id"]
    s3_image_prefix: str = event["s3_image_prefix"]
    base_model: str = event.get("base_model", "")
    training_params: dict[str, Any] = event.get("training_params", {})

    if base_model:
        training_params.setdefault("base_model", base_model)

    model_id = str(uuid.uuid4())

    logger.info(
        "Training pipeline started",
        extra={
            "job_id": job_id,
            "creator_id": creator_id,
            "model_id": model_id,
            "s3_image_prefix": s3_image_prefix,
        },
    )

    # Initialise shared clients.
    config = Config()
    s3_client = S3Client(bucket=config.s3_bucket, region=config.aws_region)
    dynamodb_client = DynamoDBClient(region=config.aws_region)
    sqs_client = SqsConsumer(queue_url=config.sqs_training_queue_url, region=config.aws_region)

    # Use a temporary directory for all intermediate artifacts.
    work_dir = Path(tempfile.mkdtemp(prefix=f"training_{job_id}_"))

    try:
        # Stage 1: Download training images from S3.
        raw_dir = work_dir / "raw"
        image_paths = download_training_images(s3_client, s3_image_prefix, raw_dir)

        # Stage 2: Preprocess images with identity-conditioned captions.
        preprocessed_dir = work_dir / "preprocessed"
        subject_token = training_params.get("subject_token", "sks person")
        preprocessed = preprocess_images(
            image_paths,
            preprocessed_dir,
            target_size=training_params.get("resolution", config.image_resolution),
            device=config.device,
            subject_token=subject_token,
        )

        # Stage 3: Train LoRA model.
        model_dir = work_dir / "model"
        training_metrics = train_lora(preprocessed, config, model_dir, training_params)

        # Stage 4: Quality validation.
        quality_metrics = validate_model_quality(
            model_dir,
            config,
            training_params,
        )

        # Stage 5: Upload artifacts to S3.
        s3_path = upload_artifacts(s3_client, model_dir, creator_id, job_id)

        # Stage 6: Register model in DynamoDB.
        training_config = {
            "base_model": training_params.get("base_model", config.base_model_id),
            "lora_rank": training_params.get("lora_rank", config.lora_rank),
            "training_steps": training_params.get("training_steps", config.training_steps),
            "learning_rate": training_params.get("learning_rate", config.learning_rate),
        }
        register_model(
            dynamodb_client,
            model_id=model_id,
            creator_id=creator_id,
            job_id=job_id,
            s3_path=s3_path,
            quality_metrics={
                "clip_score": quality_metrics["clip_score"],
                "passed": quality_metrics["passed"],
            },
            training_config=training_config,
        )

        # Stage 7: Send completion event to SQS.
        combined_metrics = {**training_metrics, **quality_metrics}
        send_completion_event(
            sqs_client,
            job_id=job_id,
            creator_id=creator_id,
            model_id=model_id,
            s3_path=s3_path,
            metrics=combined_metrics,
        )

        logger.info(
            "Training pipeline completed successfully",
            extra={"job_id": job_id, "model_id": model_id, "s3_path": s3_path},
        )

        return {
            "job_id": job_id,
            "model_id": model_id,
            "creator_id": creator_id,
            "status": "COMPLETED",
            "model_s3_path": s3_path,
            "metrics": {
                "final_loss": training_metrics["final_loss"],
                "clip_score": quality_metrics["clip_score"],
                "steps_completed": training_metrics["steps_completed"],
            },
        }

    except TrainingPipelineError:
        logger.exception("Training pipeline failed", extra={"job_id": job_id})
        raise

    except Exception:
        logger.exception("Unexpected error in training pipeline", extra={"job_id": job_id})
        raise TrainingPipelineError(f"Training pipeline failed for job {job_id}") from None

    finally:
        # Clean up temporary working directory.
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info("Cleaned up working directory", extra={"path": str(work_dir)})
