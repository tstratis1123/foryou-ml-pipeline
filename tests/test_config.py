"""Unit tests for ML pipeline configuration."""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import patch


def _fresh_config():
    """Reimport shared.config to pick up patched env vars."""
    mod_name = "shared.config"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    return mod.Config


def test_config_defaults():
    """Config should populate defaults when env vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        config_cls = _fresh_config()
        config = config_cls()
        assert config.aws_region == "us-east-1"
        assert config.lora_rank == 16
        assert config.training_steps == 1000
        assert config.image_resolution == 1024
        assert config.nsfw_threshold == 0.85
        assert config.device == "cuda"
        assert config.consent_service_url == "http://consent-service:3002"


def test_config_from_env():
    """Config should read values from environment variables."""
    env = {
        "AWS_REGION": "eu-west-1",
        "S3_BUCKET": "test-bucket",
        "LORA_RANK": "32",
        "DEVICE": "cpu",
        "CONSENT_SERVICE_URL": "http://localhost:3002",
    }
    with patch.dict(os.environ, env, clear=True):
        config_cls = _fresh_config()
        config = config_cls()
        assert config.aws_region == "eu-west-1"
        assert config.s3_bucket == "test-bucket"
        assert config.lora_rank == 32
        assert config.device == "cpu"
        assert config.consent_service_url == "http://localhost:3002"
