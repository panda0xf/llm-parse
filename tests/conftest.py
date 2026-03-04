"""共享测试 fixtures。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from llm_parse.config import LLMConfig, ServiceConfig


class SimpleOutput(BaseModel):
    """测试用简单输出模型。"""

    answer: str = Field(description="回答")
    score: int = Field(description="评分", ge=1, le=10)


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        openai_base_url="https://api.test.com/v1",
        openai_api_key="test-key",
        model_name="test-model",
    )


@pytest.fixture
def service_config() -> ServiceConfig:
    return ServiceConfig(
        output_retries=2,
        http_max_retries=2,
        max_agent_concurrency=5,
        max_agent_queued=10,
        max_model_concurrency=5,
        service_semaphore_limit=10,
        batch_chunk_size=3,
        request_timeout=30.0,
    )
