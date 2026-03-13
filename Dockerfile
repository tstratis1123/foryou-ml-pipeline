# ==============================================================================
# ForYou ML Pipeline Dockerfile
# GPU-enabled image for training, inference, moderation, and watermarking.
#
# Usage:
#   docker build -t foryou/ml-pipeline .
# ==============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies + Python 3.11
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (cached layer — only rebuilds if requirements change)
# ---------------------------------------------------------------------------
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# ML source code
# ---------------------------------------------------------------------------
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Security: run as non-root
# ---------------------------------------------------------------------------
RUN groupadd -g 1001 mlgroup && \
    useradd -u 1001 -g mlgroup -m mluser && \
    chown -R mluser:mlgroup /app

USER mluser

# Default entrypoint — override in k8s/Step Functions per pipeline stage
CMD ["python", "-m", "shared"]
