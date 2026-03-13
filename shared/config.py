"""Configuration loader for ML pipeline environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Config(BaseModel):
    """ML pipeline configuration populated from environment variables."""

    # AWS
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    sqs_training_queue_url: str = os.getenv("SQS_TRAINING_QUEUE_URL", "")
    sqs_generation_queue_url: str = os.getenv("SQS_GENERATION_QUEUE_URL", "")

    # Consent Service
    consent_service_url: str = os.getenv("CONSENT_SERVICE_URL", "http://consent-service:3002")

    # Model defaults
    base_model_id: str = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    lora_rank: int = int(os.getenv("LORA_RANK", "16"))
    training_steps: int = int(os.getenv("TRAINING_STEPS", "1000"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "1e-4"))

    # Generation
    default_guidance_scale: float = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5"))
    default_num_inference_steps: int = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "50"))
    image_resolution: int = int(os.getenv("IMAGE_RESOLUTION", "1024"))

    # Moderation
    nsfw_threshold: float = float(os.getenv("NSFW_THRESHOLD", "0.85"))

    # Watermark
    watermark_key: str = os.getenv("WATERMARK_KEY", "")
    watermark_strength: float = float(os.getenv("WATERMARK_STRENGTH", "0.1"))
    videoseal_model_path: str = os.getenv("VIDEOSEAL_MODEL_PATH", "")

    # Runtime
    device: str = os.getenv("DEVICE", "cuda")
