# ForYou ML Pipeline

Python-based machine learning pipeline for avatar training, content generation (image + video), moderation, and watermarking. Orchestrated by AWS Step Functions and runs on GPU nodes in EKS.

Part of the [ForYou Platform](https://github.com/tstratis1123/foryou-platform). Extracted as a standalone repository for independent development, deployment, and GPU-specific CI.

## Structure

```
training/                # LoRA/DreamBooth fine-tuning pipeline
  pipeline.py            # 8-stage training orchestration
  image_validator.py     # CPU pre-validation (resolution, sharpness, lighting, faces, duplicates)
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
  test_image_validator.py       # Image pre-validation tests
  test_watermark_robustness.py  # Robustness tests for image + video watermarks
pyproject.toml           # Python project config (dependencies, linting, testing)
requirements.txt         # Flat dependency list (used by Dockerfile)
Dockerfile               # GPU-enabled image (CUDA 12.1)
```

## Dual-Track Pipeline

Both image and video generation share the same LoRA avatar model and preprocessing steps. The pipeline branches at the synthesis stage based on `media_type`:

- **Image**: Single-frame SDXL diffusion (5-15 seconds), DwtDctSvd watermark, delivery via CloudFront signed URLs
- **Video**: Multi-frame AnimateDiff temporal diffusion (2-10 minutes), VideoSeal watermark, DRM encryption (Widevine + FairPlay + PlayReady), streaming delivery

## Key Features

### Safety & Reliability
- **Consent verification** — Every generation and moderation job verifies performer consent via the Consent Service before processing. Revoked consent aborts the job immediately (Critical Invariant #1).
- **SQS fault tolerance** — Failed messages stay on the queue for retry/DLQ instead of being silently deleted. Structured error logging for every failure.
- **S3 retry with backoff** — Exponential backoff (1s → 2s → 4s, 3 attempts) with non-retryable error detection (NoSuchKey, AccessDenied skip retries). Connection/read timeouts configured (10s/30s).

### Training Pipeline
- **Image pre-validation** — CPU-only quality checks before GPU training: minimum resolution (512x512), sharpness (Laplacian variance), lighting (exposure), single-face enforcement, face-size ratio, and near-duplicate detection (perceptual hashing). Performers get specific rejection reasons per image.
- **Minimum 20 images** — Recommended minimum for LoRA/DreamBooth identity fidelity. Pipeline aborts with detailed feedback if too few pass validation.
- **Cosine annealing LR scheduler** — Learning rate decays smoothly to 1% of initial value over the training run, avoiding sudden drops.
- **Mixed precision training** — `torch.autocast` with fp16 on CUDA for ~2x training speed with minimal quality loss.
- **Batched training** — Configurable batch size (default 4 on GPU, 1 on CPU) with gradient accumulation.
- **Identity-conditioned captions** — BLIP-generated captions prefixed with DreamBooth subject token (`"a photo of sks person, {caption}"`) for stronger identity binding.

### Video Watermark Training
- **Adversarial training pipeline** (`watermark/train_videoseal.py`) — Trains encoder-decoder jointly: encoder minimises frame distortion (MSE) while decoder maximises payload recovery (BCE).
- **Robustness augmentations** — JPEG compression, Gaussian noise, scaling, and cropping applied between encoder and decoder during training via straight-through estimation.
- **Curriculum warmup** — First N epochs train without augmentations so networks learn basic embed/recover before facing distortions.
- **Training features** — Cosine LR scheduling, AdamW optimiser, gradient clipping, early stopping, periodic checkpointing.

## Tests

```bash
pytest tests/ -v
```

| Test file | Coverage |
|-----------|----------|
| `test_config.py` | Config defaults and env var loading |
| `test_consent_client.py` | Consent check success, denial (403/404/422), unavailability (503/network/timeout) |
| `test_sqs_consumer.py` | Message processing, failure handling, DLQ behaviour |
| `test_s3_client.py` | Upload/download, retry logic, non-retryable errors, pagination |
| `test_dynamodb_client.py` | CRUD operations, validation, queries |
| `test_image_validator.py` | Resolution, sharpness, lighting, duplicate detection, report structure |
| `test_watermark_robustness.py` | Image watermark survival (JPEG/scaling/noise), VideoSeal encode/decode shapes, augmentation pipeline |

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
