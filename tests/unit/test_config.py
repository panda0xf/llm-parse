"""LLMConfig / ServiceConfig 单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_parse.config import LLMConfig, ServiceConfig, get_config, get_service_config


class TestLLMConfig:
    def test_default_values(self) -> None:
        cfg = LLMConfig(openai_api_key="test-key", _env_file=None)
        assert cfg.openai_base_url == "https://api.openai.com/v1"
        assert cfg.model_name == "gpt-4o-mini"
        assert cfg.openai_api_key == "test-key"

    def test_custom_values(self) -> None:
        cfg = LLMConfig(
            openai_base_url="https://custom.api.com/v1",
            openai_api_key="custom-key",
            model_name="gpt-4o",
        )
        assert cfg.openai_base_url == "https://custom.api.com/v1"
        assert cfg.openai_api_key == "custom-key"
        assert cfg.model_name == "gpt-4o"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.api.com/v1")
        monkeypatch.setenv("MODEL_NAME", "env-model")
        cfg = LLMConfig(_env_file=None)
        assert cfg.openai_api_key == "env-key"
        assert cfg.openai_base_url == "https://env.api.com/v1"
        assert cfg.model_name == "env-model"


class TestServiceConfig:
    def test_default_values(self) -> None:
        cfg = ServiceConfig()
        assert cfg.output_retries == 3
        assert cfg.http_max_retries == 5
        assert cfg.http_retry_min_wait == 1.0
        assert cfg.http_retry_max_wait == 60.0
        assert cfg.max_agent_concurrency == 10
        assert cfg.max_agent_queued == 100
        assert cfg.max_model_concurrency == 20
        assert cfg.service_semaphore_limit == 50
        assert cfg.batch_chunk_size == 20
        assert cfg.request_timeout == 120.0

    def test_custom_values(self) -> None:
        cfg = ServiceConfig(
            output_retries=5,
            http_max_retries=10,
            http_retry_min_wait=2.0,
            http_retry_max_wait=120.0,
            max_agent_concurrency=20,
            max_agent_queued=200,
            max_model_concurrency=40,
            service_semaphore_limit=100,
            batch_chunk_size=50,
            request_timeout=60.0,
        )
        assert cfg.output_retries == 5
        assert cfg.http_max_retries == 10
        assert cfg.http_retry_min_wait == 2.0
        assert cfg.http_retry_max_wait == 120.0
        assert cfg.max_agent_concurrency == 20
        assert cfg.max_agent_queued == 200
        assert cfg.max_model_concurrency == 40
        assert cfg.service_semaphore_limit == 100
        assert cfg.batch_chunk_size == 50
        assert cfg.request_timeout == 60.0

    def test_from_env_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARSE_OUTPUT_RETRIES", "7")
        monkeypatch.setenv("PARSE_REQUEST_TIMEOUT", "45.0")
        monkeypatch.setenv("PARSE_BATCH_CHUNK_SIZE", "10")
        cfg = ServiceConfig(_env_file=None)
        assert cfg.output_retries == 7
        assert cfg.request_timeout == 45.0
        assert cfg.batch_chunk_size == 10

    def test_validation_constraints(self) -> None:
        with pytest.raises(ValidationError):
            ServiceConfig(output_retries=-1)
        with pytest.raises(ValidationError):
            ServiceConfig(max_agent_concurrency=0)
        with pytest.raises(ValidationError):
            ServiceConfig(request_timeout=0)
        with pytest.raises(ValidationError):
            ServiceConfig(http_retry_min_wait=0)


class TestFactoryFunctions:
    def test_get_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "factory-key")
        get_config.cache_clear()
        cfg = get_config()
        assert isinstance(cfg, LLMConfig)
        assert cfg.openai_api_key == "factory-key"

    def test_get_service_config(self) -> None:
        cfg = get_service_config()
        assert isinstance(cfg, ServiceConfig)
