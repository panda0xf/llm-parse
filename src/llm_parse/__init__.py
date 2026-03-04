"""LLM 结构化解析服务。

提供通用的 LLM 结构化输出能力：
输入 Pydantic Model 类型 + Prompt，输出填充后的 Pydantic Model 实例。
"""

from llm_parse.config import LLMConfig, ServiceConfig, get_config, get_service_config
from llm_parse.exceptions import (
    ConcurrencyExhaustedError,
    LLMConnectionError,
    LLMTimeoutError,
    OutputValidationError,
    ParseServiceError,
    RateLimitError,
)
from llm_parse.models import BatchResult, ParseRequest, ParseResult, ParseStatus
from llm_parse.service import LLMParseService

__all__ = [
    "LLMParseService",
    "ParseResult",
    "ParseRequest",
    "ParseStatus",
    "BatchResult",
    "LLMConfig",
    "ServiceConfig",
    "get_config",
    "get_service_config",
    "ParseServiceError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "OutputValidationError",
    "RateLimitError",
    "ConcurrencyExhaustedError",
]
