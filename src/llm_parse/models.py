"""解析服务通用数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ParseStatus(str, Enum):
    """解析结果状态枚举。"""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ParseResult(Generic[T]):
    """通用解析结果（Result Pattern）。

    批量场景下单条失败不影响整体。
    """

    status: ParseStatus
    output: T | None = None
    error_message: str | None = None
    prompt: str = ""
    elapsed_seconds: float = 0.0
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        return self.status == ParseStatus.SUCCESS

    def unwrap(self) -> T:
        """获取输出值，失败时抛出 ValueError。"""
        if self.output is None:
            raise ValueError(
                f"解析失败 [{self.status.value}]: {self.error_message}"
            )
        return self.output


@dataclass
class ParseRequest(Generic[T]):
    """批量解析请求项。"""

    prompt: str
    output_type: type[T]
    system_prompt: str = ""
    request_id: str = ""


@dataclass
class BatchResult(Generic[T]):
    """批量解析结果汇总。"""

    results: list[ParseResult[T]] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.is_success)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total > 0 else 0.0
