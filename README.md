# ForYou ML Pipeline

Python-based machine learning pipeline for avatar training, content generation (image + video), moderation, and watermarking. Orchestrated by AWS Step Functions and runs on GPU nodes in EKS.

Part of the [ForYou Platform](https://github.com/tstratis1123/foryou-platform). Extracted as a standalone repository for independent development, deployment, and GPU-specific CI.

## Structure

```
training/                # LoRA/DreamBooth fine-tuning pipeline
  pipeline.py            # 7-stage training orchestration
generation/              # Inference serving and content synthesis
  pipeline.py            # Dual-track generation (image + video)
moderation/              # Content safety scanning
  scanner.py             # NSFW, face detection, policy checks (image + video)
watermark/               # Per-customer watermark embedding
  image_watermark.py     # DwtDctSvd invisible steganography (CPU, <1s)
  video_watermark.py     # VideoSeal neural watermarking (GPU)
  train_videoseal.py     # VideoSeal training pipeline (adversarial, with robustness augmentations)
shared/                  # Shared utilities
  config.py              # Pydantic config from environment variables
  s3_client.py           # S3 upload/download helpers
  sqs_consumer.py        # SQS message consumer base class
  dynamodb_client.py     # Model registry CRUD (DynamoDB)
  consent_client.py      # HTTP consent verification client
  logger.py              # Structured JSON logging
tests/                   # Unit tests
  test_config.py         # Config loader tests
  test_watermark_robustness.py  # Robustness tests for image + video watermarks
pyproject.toml           # Python project config (dependencies, linting, testing)
requirements.txt         # Flat dependency list (used by Dockerfile)
Dockerfile               # GPU-enabled image (CUDA 12.1)
```

## Dual-Track Pipeline

Both image and video generation share the same LoRA avatar model and preprocessing steps. The pipeline branches at the synthesis stage based on `media_type`:

- **Image**: Single-frame SDXL diffusion (5-15 seconds), DwtDctSvd watermark, delivery via CloudFront signed URLs
- **Video**: Multi-frame AnimateDiff temporal diffusion (2-10 minutes), VideoSeal watermark, DRM encryption (Widevine + FairPlay + PlayReady), streaming delivery

## Setup

```bash
# Install with dev dependencies (recommended)
pip install -e ".[dev]"

# Or flat requirements (used by Dockerfile)
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
```

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy . --ignore-missing-imports

# Tests
pytest tests/ -v
```

## Docker

```bash
# Build GPU-enabled image
docker build -t foryou/ml-pipeline .

# Run specific pipeline stage
docker run --gpus all foryou/ml-pipeline python -m training.pipeline
docker run --gpus all foryou/ml-pipeline python -m generation.pipeline
```

## Integration with Backend

The ML pipeline communicates with the NestJS backend asynchronously via SQS queues and S3:

- **Input**: SQS messages with job payloads (training requests, generation requests)
- **Output**: Generated content uploaded to S3, completion events published to SQS
- **Model Registry**: DynamoDB table (`foryou-model-registry`) tracks model lifecycle
- **Shared Contracts**: Type definitions in [`libs/ml-contracts/`](https://github.com/tstratis1123/foryou-platform/tree/main/libs/ml-contracts) (TypeScript npm package) mirror Python types
