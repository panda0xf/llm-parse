"""配置管理模块。

通过 Pydantic Settings 从 .env 或环境变量加载、验证配置。
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM 服务连接配置。"""

    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API 的 Base URL",
    )
    openai_api_key: str = Field(
        description="API 密钥",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="要使用的模型名称",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class ServiceConfig(BaseSettings):
    """解析服务运行时配置，控制重试、并发与超时行为。"""

    output_retries: int = Field(
        default=3,
        ge=0,
        description="pydantic-ai 层输出校验重试次数",
    )
    http_max_retries: int = Field(
        default=5,
        ge=0,
        description="HTTP 层（429/5xx）tenacity 最大重试次数",
    )
    http_retry_min_wait: float = Field(
        default=1.0,
        gt=0,
        description="HTTP 重试最小等待秒数",
    )
    http_retry_max_wait: float = Field(
        default=60.0,
        gt=0,
        description="HTTP 重试最大等待秒数",
    )
    max_agent_concurrency: int = Field(
        default=10,
        ge=1,
        description="Agent 级别最大并发运行数",
    )
    max_agent_queued: int = Field(
        default=100,
        ge=0,
        description="Agent 级别最大排队数（0 表示无限排队）",
    )
    max_model_concurrency: int = Field(
        default=20,
        ge=1,
        description="Model 级别最大并发 HTTP 请求数",
    )
    service_semaphore_limit: int = Field(
        default=50,
        ge=1,
        description="服务级全局信号量上限",
    )
    batch_chunk_size: int = Field(
        default=20,
        ge=1,
        description="批量解析分片大小",
    )
    request_timeout: float = Field(
        default=120.0,
        gt=0,
        description="单次 LLM 请求超时时间（秒）",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "PARSE_",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_config() -> LLMConfig:
    """获取 LLM 配置实例。"""
    return LLMConfig()


@lru_cache(maxsize=1)
def get_service_config() -> ServiceConfig:
    """获取服务配置实例。"""
    return ServiceConfig()
