"""异常层级单元测试。"""

from __future__ import annotations

from llm_parse.exceptions import (
    ConcurrencyExhaustedError,
    LLMConnectionError,
    LLMTimeoutError,
    OutputValidationError,
    ParseServiceError,
    RateLimitError,
)


class TestExceptionHierarchy:
    def test_all_subclasses_inherit_from_base(self) -> None:
        for exc_cls in [
            LLMConnectionError,
            LLMTimeoutError,
            OutputValidationError,
            RateLimitError,
            ConcurrencyExhaustedError,
        ]:
            assert issubclass(exc_cls, ParseServiceError)
            assert issubclass(exc_cls, Exception)

    def test_base_exception_message(self) -> None:
        exc = ParseServiceError("基础错误")
        assert str(exc) == "基础错误"

    def test_llm_connection_error(self) -> None:
        exc = LLMConnectionError("连接失败")
        assert str(exc) == "连接失败"
        assert isinstance(exc, ParseServiceError)

    def test_llm_timeout_error(self) -> None:
        exc = LLMTimeoutError("超时")
        assert str(exc) == "超时"
        assert isinstance(exc, ParseServiceError)

    def test_concurrency_exhausted_error(self) -> None:
        exc = ConcurrencyExhaustedError("并发耗尽")
        assert str(exc) == "并发耗尽"
        assert isinstance(exc, ParseServiceError)


class TestOutputValidationError:
    def test_with_raw_output(self) -> None:
        exc = OutputValidationError("校验失败", last_raw_output='{"bad": "data"}')
        assert exc.last_raw_output == '{"bad": "data"}'
        assert "校验失败" in str(exc)

    def test_without_raw_output(self) -> None:
        exc = OutputValidationError("校验失败")
        assert exc.last_raw_output is None
        assert "校验失败" in str(exc)


class TestRateLimitError:
    def test_with_retry_after(self) -> None:
        exc = RateLimitError("限流", retry_after=30.0)
        assert exc.retry_after == 30.0
        assert "限流" in str(exc)

    def test_without_retry_after(self) -> None:
        exc = RateLimitError("限流")
        assert exc.retry_after is None

    def test_catchable_as_base(self) -> None:
        try:
            raise RateLimitError("限流", retry_after=10.0)
        except ParseServiceError as exc:
            assert isinstance(exc, RateLimitError)
