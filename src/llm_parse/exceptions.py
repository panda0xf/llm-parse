"""解析服务异常层级。"""


class ParseServiceError(Exception):
    """解析服务基础异常，所有子异常的基类。"""


class LLMConnectionError(ParseServiceError):
    """无法连接到 LLM 后端服务。"""


class LLMTimeoutError(ParseServiceError):
    """LLM 请求超时。"""


class OutputValidationError(ParseServiceError):
    """LLM 返回结果无法通过 Pydantic 校验。"""

    def __init__(self, message: str, last_raw_output: str | None = None) -> None:
        super().__init__(message)
        self.last_raw_output = last_raw_output


class RateLimitError(ParseServiceError):
    """LLM 后端返回 429，重试策略耗尽。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ConcurrencyExhaustedError(ParseServiceError):
    """服务并发资源耗尽，无法接受更多请求。"""
