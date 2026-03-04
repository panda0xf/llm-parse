"""ParseResult / BatchResult / ParseRequest 单元测试。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from llm_parse.models import BatchResult, ParseRequest, ParseResult, ParseStatus


class SimpleOutput(BaseModel):
    answer: str = Field(description="回答")
    score: int = Field(description="评分", ge=1, le=10)


class TestParseResult:
    def test_success_result(self) -> None:
        output = SimpleOutput(answer="hello", score=8)
        result = ParseResult(
            status=ParseStatus.SUCCESS,
            output=output,
            prompt="test",
            elapsed_seconds=1.5,
        )
        assert result.is_success is True
        assert result.unwrap() == output
        assert result.elapsed_seconds == 1.5
        assert result.error_message is None

    def test_failed_result_is_not_success(self) -> None:
        result = ParseResult[SimpleOutput](
            status=ParseStatus.TIMEOUT,
            error_message="超时了",
            prompt="test",
        )
        assert result.is_success is False
        assert result.output is None

    def test_failed_result_unwrap_raises_valueerror(self) -> None:
        result = ParseResult[SimpleOutput](
            status=ParseStatus.TIMEOUT,
            error_message="超时了",
            prompt="test",
        )
        with pytest.raises(ValueError, match="超时"):
            result.unwrap()

    def test_all_status_types(self) -> None:
        for status in ParseStatus:
            result = ParseResult[SimpleOutput](status=status, prompt="test")
            if status == ParseStatus.SUCCESS:
                assert result.is_success is True
            else:
                assert result.is_success is False

    def test_default_values(self) -> None:
        result = ParseResult[SimpleOutput](status=ParseStatus.SUCCESS)
        assert result.output is None
        assert result.error_message is None
        assert result.prompt == ""
        assert result.elapsed_seconds == 0.0
        assert result.retry_count == 0

    def test_retry_count(self) -> None:
        result = ParseResult[SimpleOutput](
            status=ParseStatus.SUCCESS,
            retry_count=3,
            prompt="test",
        )
        assert result.retry_count == 3

    def test_unwrap_with_none_output_on_success(self) -> None:
        """status=SUCCESS 但 output=None 时 unwrap 也应抛异常。"""
        result = ParseResult[SimpleOutput](
            status=ParseStatus.SUCCESS,
            output=None,
            prompt="test",
        )
        with pytest.raises(ValueError, match="解析失败"):
            result.unwrap()


class TestBatchResult:
    def test_empty_batch(self) -> None:
        batch: BatchResult[SimpleOutput] = BatchResult()
        assert batch.total == 0
        assert batch.succeeded == 0
        assert batch.failed == 0
        assert batch.success_rate == 0.0

    def test_mixed_results(self) -> None:
        results = [
            ParseResult(
                status=ParseStatus.SUCCESS,
                output=SimpleOutput(answer="ok", score=5),
                prompt="q1",
            ),
            ParseResult[SimpleOutput](
                status=ParseStatus.TIMEOUT,
                error_message="超时",
                prompt="q2",
            ),
            ParseResult(
                status=ParseStatus.SUCCESS,
                output=SimpleOutput(answer="good", score=9),
                prompt="q3",
            ),
        ]
        batch = BatchResult(results=results, total_elapsed_seconds=3.0)
        assert batch.total == 3
        assert batch.succeeded == 2
        assert batch.failed == 1
        assert batch.success_rate == pytest.approx(2 / 3, rel=1e-3)
        assert batch.total_elapsed_seconds == 3.0

    def test_all_success(self) -> None:
        results = [
            ParseResult(
                status=ParseStatus.SUCCESS,
                output=SimpleOutput(answer="a", score=1),
                prompt=f"q{i}",
            )
            for i in range(5)
        ]
        batch = BatchResult(results=results)
        assert batch.total == 5
        assert batch.succeeded == 5
        assert batch.failed == 0
        assert batch.success_rate == 1.0

    def test_all_failed(self) -> None:
        results = [
            ParseResult[SimpleOutput](
                status=ParseStatus.UNKNOWN_ERROR,
                error_message="err",
                prompt=f"q{i}",
            )
            for i in range(3)
        ]
        batch = BatchResult(results=results)
        assert batch.total == 3
        assert batch.succeeded == 0
        assert batch.failed == 3
        assert batch.success_rate == 0.0

    def test_default_elapsed(self) -> None:
        batch: BatchResult[SimpleOutput] = BatchResult()
        assert batch.total_elapsed_seconds == 0.0


class TestParseRequest:
    def test_basic_fields(self) -> None:
        req: ParseRequest[SimpleOutput] = ParseRequest(
            prompt="测试",
            output_type=SimpleOutput,
            system_prompt="你是助手",
            request_id="req-001",
        )
        assert req.prompt == "测试"
        assert req.output_type is SimpleOutput
        assert req.system_prompt == "你是助手"
        assert req.request_id == "req-001"

    def test_default_values(self) -> None:
        req: ParseRequest[SimpleOutput] = ParseRequest(
            prompt="测试",
            output_type=SimpleOutput,
        )
        assert req.system_prompt == ""
        assert req.request_id == ""
