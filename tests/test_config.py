"""Unit tests for ML pipeline configuration."""

from __future__ import annotations

import os
from unittest.mock import patch


def test_config_defaults():
    """Config should populate defaults when env vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        from shared.config import Config

        config = Config()
        assert config.aws_region == "us-east-1"
        assert config.lora_rank == 16
        assert config.training_steps == 1000
        assert config.image_resolution == 1024
        assert config.nsfw_threshold == 0.85
        assert config.device == "cuda"


def test_config_from_env():
    """Config should read values from environment variables."""
    env = {
        "AWS_REGION": "eu-west-1",
        "S3_BUCKET": "test-bucket",
        "LORA_RANK": "32",
        "DEVICE": "cpu",
    }
    with patch.dict(os.environ, env, clear=True):
        from shared.config import Config

        config = Config()
        assert config.aws_region == "eu-west-1"
        assert config.s3_bucket == "test-bucket"
        assert config.lora_rank == 32
        assert config.device == "cpu"
