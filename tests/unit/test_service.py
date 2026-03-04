"""LLMParseService 单元测试。

使用 mock 测试所有解析路径，无需真实 LLM 连接。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPStatusError, Request, Response
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import UnexpectedModelBehavior

from llm_parse.config import LLMConfig, ServiceConfig
from llm_parse.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    OutputValidationError,
    ParseServiceError,
    RateLimitError,
)
from llm_parse.models import ParseRequest, ParseStatus
from llm_parse.service import LLMParseService


class SimpleOutput(BaseModel):
    answer: str = Field(description="回答")
    score: int = Field(description="评分", ge=1, le=10)


@pytest.fixture
def mock_service(service_config: ServiceConfig, llm_config: LLMConfig) -> LLMParseService:
    """创建带 mock 内部组件的 LLMParseService。"""
    with patch.object(LLMParseService, "_create_rate_limit_client") as mock_client, \
         patch.object(LLMParseService, "_create_model") as mock_model:
        mock_client.return_value = AsyncMock()
        mock_model.return_value = MagicMock()
        svc = LLMParseService(llm_config=llm_config, service_config=service_config)
    return svc


class TestParse:
    @pytest.mark.asyncio
    async def test_parse_success(self, mock_service: LLMParseService) -> None:
        output = SimpleOutput(answer="test", score=7)
        mock_run_result = MagicMock()
        mock_run_result.output = output

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test prompt", SimpleOutput)

        assert result.is_success
        assert result.output == output
        assert result.elapsed_seconds > 0
        assert result.prompt == "test prompt"

    @pytest.mark.asyncio
    async def test_parse_timeout(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.TIMEOUT
        assert "超时" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_parse_validation_error(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=UnexpectedModelBehavior("bad output")
        )

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.VALIDATION_ERROR
        assert "校验失败" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_parse_rate_limited(self, mock_service: LLMParseService) -> None:
        request = Request("POST", "https://api.test.com/v1/chat/completions")
        response = Response(429, headers={"Retry-After": "30"}, request=request)
        exc = HTTPStatusError("rate limited", request=request, response=response)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=exc)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.RATE_LIMITED
        assert "429" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_parse_http_5xx_error(self, mock_service: LLMParseService) -> None:
        request = Request("POST", "https://api.test.com/v1/chat/completions")
        response = Response(500, request=request)
        exc = HTTPStatusError("server error", request=request, response=response)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=exc)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.CONNECTION_ERROR
        assert "500" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_parse_connection_error(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=ConnectionError("refused"))

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.CONNECTION_ERROR
        assert "连接错误" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_parse_os_error(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=OSError("network unreachable"))

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.CONNECTION_ERROR

    @pytest.mark.asyncio
    async def test_parse_unknown_error(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse("test", SimpleOutput)

        assert result.status == ParseStatus.UNKNOWN_ERROR
        assert "RuntimeError" in (result.error_message or "")


class TestParseBatch:
    @pytest.mark.asyncio
    async def test_batch_success(self, mock_service: LLMParseService) -> None:
        output = SimpleOutput(answer="ok", score=5)
        mock_run_result = MagicMock()
        mock_run_result.output = output
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            requests = [
                ParseRequest(prompt=f"q{i}", output_type=SimpleOutput)
                for i in range(5)
            ]
            batch = await mock_service.parse_batch(requests)

        assert batch.total == 5
        assert batch.succeeded == 5
        assert batch.failed == 0
        assert batch.success_rate == 1.0
        assert batch.total_elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_batch_chunking(self, mock_service: LLMParseService) -> None:
        """batch_chunk_size=3，5 个请求应分 2 批。"""
        call_count = 0

        async def counting_parse(prompt: str, output_type: type, system_prompt: str = "") -> MagicMock:
            nonlocal call_count
            call_count += 1
            from llm_parse.models import ParseResult
            return ParseResult(
                status=ParseStatus.SUCCESS,
                output=SimpleOutput(answer="ok", score=5),
                prompt=prompt,
                elapsed_seconds=0.1,
            )

        with patch.object(mock_service, "parse", side_effect=counting_parse):
            requests = [
                ParseRequest(prompt=f"q{i}", output_type=SimpleOutput)
                for i in range(5)
            ]
            batch = await mock_service.parse_batch(requests)

        assert call_count == 5
        assert batch.total == 5

    @pytest.mark.asyncio
    async def test_batch_mixed_results(self, mock_service: LLMParseService) -> None:
        """批量解析中部分成功部分失败。"""
        outputs = [
            SimpleOutput(answer="ok", score=5),
            None,
            SimpleOutput(answer="good", score=8),
        ]
        call_idx = 0

        async def alternate_parse(prompt: str, output_type: type, system_prompt: str = "") -> MagicMock:
            nonlocal call_idx
            from llm_parse.models import ParseResult
            idx = call_idx
            call_idx += 1
            if outputs[idx] is not None:
                return ParseResult(
                    status=ParseStatus.SUCCESS,
                    output=outputs[idx],
                    prompt=prompt,
                    elapsed_seconds=0.1,
                )
            return ParseResult(
                status=ParseStatus.TIMEOUT,
                error_message="超时",
                prompt=prompt,
                elapsed_seconds=0.1,
            )

        with patch.object(mock_service, "parse", side_effect=alternate_parse):
            requests = [
                ParseRequest(prompt=f"q{i}", output_type=SimpleOutput)
                for i in range(3)
            ]
            batch = await mock_service.parse_batch(requests)

        assert batch.total == 3
        assert batch.succeeded == 2
        assert batch.failed == 1


class TestParseOrRaise:
    @pytest.mark.asyncio
    async def test_success_returns_output(self, mock_service: LLMParseService) -> None:
        output = SimpleOutput(answer="test", score=7)
        mock_run_result = MagicMock()
        mock_run_result.output = output
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            result = await mock_service.parse_or_raise("test", SimpleOutput)

        assert result == output

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            with pytest.raises(LLMTimeoutError):
                await mock_service.parse_or_raise("test", SimpleOutput)

    @pytest.mark.asyncio
    async def test_validation_error_raises(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=UnexpectedModelBehavior("bad")
        )

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            with pytest.raises(OutputValidationError):
                await mock_service.parse_or_raise("test", SimpleOutput)

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self, mock_service: LLMParseService) -> None:
        request = Request("POST", "https://api.test.com/v1/chat/completions")
        response = Response(429, headers={"Retry-After": "30"}, request=request)
        exc = HTTPStatusError("rate limited", request=request, response=response)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=exc)

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            with pytest.raises(RateLimitError):
                await mock_service.parse_or_raise("test", SimpleOutput)

    @pytest.mark.asyncio
    async def test_connection_error_raises(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=ConnectionError("refused"))

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            with pytest.raises(LLMConnectionError):
                await mock_service.parse_or_raise("test", SimpleOutput)

    @pytest.mark.asyncio
    async def test_unknown_error_raises_base(self, mock_service: LLMParseService) -> None:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch.object(mock_service, "_get_or_create_agent", return_value=mock_agent):
            with pytest.raises(ParseServiceError):
                await mock_service.parse_or_raise("test", SimpleOutput)

    @pytest.mark.asyncio
    async def test_success_but_none_output_raises(self, mock_service: LLMParseService) -> None:
        """parse 返回 SUCCESS 但 output=None 时应抛 ParseServiceError。"""
        from llm_parse.models import ParseResult

        with patch.object(
            mock_service,
            "parse",
            return_value=ParseResult(status=ParseStatus.SUCCESS, output=None, prompt="test"),
        ):
            with pytest.raises(ParseServiceError, match="输出为空"):
                await mock_service.parse_or_raise("test", SimpleOutput)


class TestAgentCache:
    def test_same_key_returns_same_agent(self, mock_service: LLMParseService) -> None:
        with patch("llm_parse.service.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            agent1 = mock_service._get_or_create_agent(SimpleOutput, "prompt1")
            agent2 = mock_service._get_or_create_agent(SimpleOutput, "prompt1")
            assert agent1 is agent2
            assert MockAgent.call_count == 1

    def test_different_prompt_returns_different_agent(self, mock_service: LLMParseService) -> None:
        with patch("llm_parse.service.Agent") as MockAgent:
            MockAgent.side_effect = [MagicMock(), MagicMock()]
            agent1 = mock_service._get_or_create_agent(SimpleOutput, "prompt1")
            agent2 = mock_service._get_or_create_agent(SimpleOutput, "prompt2")
            assert agent1 is not agent2
            assert MockAgent.call_count == 2

    def test_different_type_returns_different_agent(self, mock_service: LLMParseService) -> None:
        class AnotherOutput(BaseModel):
            value: str = ""

        with patch("llm_parse.service.Agent") as MockAgent:
            MockAgent.side_effect = [MagicMock(), MagicMock()]
            agent1 = mock_service._get_or_create_agent(SimpleOutput, "prompt")
            agent2 = mock_service._get_or_create_agent(AnotherOutput, "prompt")
            assert agent1 is not agent2
            assert MockAgent.call_count == 2

    def test_lru_eviction(self, mock_service: LLMParseService) -> None:
        """缓存超过上限时应淘汰最旧条目。"""
        from llm_parse.service import _MAX_AGENT_CACHE_SIZE

        with patch("llm_parse.service.Agent") as MockAgent:
            agents = [MagicMock() for _ in range(_MAX_AGENT_CACHE_SIZE + 1)]
            MockAgent.side_effect = agents

            # 填满缓存
            for i in range(_MAX_AGENT_CACHE_SIZE):
                mock_service._get_or_create_agent(SimpleOutput, f"p{i}")

            assert len(mock_service._agent_cache) == _MAX_AGENT_CACHE_SIZE

            # 再添加一个，应淘汰最旧的 (SimpleOutput, "p0")
            mock_service._get_or_create_agent(SimpleOutput, "overflow")
            assert len(mock_service._agent_cache) == _MAX_AGENT_CACHE_SIZE
            assert (SimpleOutput, "p0") not in mock_service._agent_cache
            assert (SimpleOutput, "overflow") in mock_service._agent_cache

    def test_cache_hit_prevents_eviction(self, mock_service: LLMParseService) -> None:
        """缓存命中应将条目移至末尾，防止被淘汰。"""
        from llm_parse.service import _MAX_AGENT_CACHE_SIZE

        with patch("llm_parse.service.Agent") as MockAgent:
            agents = [MagicMock() for _ in range(_MAX_AGENT_CACHE_SIZE + 1)]
            MockAgent.side_effect = agents

            # 填满缓存
            for i in range(_MAX_AGENT_CACHE_SIZE):
                mock_service._get_or_create_agent(SimpleOutput, f"p{i}")

            # 命中 "p0" 使其移至末尾
            mock_service._get_or_create_agent(SimpleOutput, "p0")

            # 添加新条目，应淘汰 "p1"（现在最旧的）
            mock_service._get_or_create_agent(SimpleOutput, "new")
            assert (SimpleOutput, "p0") in mock_service._agent_cache
            assert (SimpleOutput, "p1") not in mock_service._agent_cache


class TestInternalCreation:
    def test_create_rate_limit_client(
        self, service_config: ServiceConfig, llm_config: LLMConfig
    ) -> None:
        """_create_rate_limit_client 应返回 AsyncClient。"""
        with patch.object(LLMParseService, "_create_model") as mock_model:
            mock_model.return_value = MagicMock()
            svc = LLMParseService(llm_config=llm_config, service_config=service_config)
        from httpx import AsyncClient
        assert isinstance(svc._http_client, AsyncClient)

    def test_create_model(
        self, service_config: ServiceConfig, llm_config: LLMConfig
    ) -> None:
        """_create_model 应返回 ConcurrencyLimitedModel。"""
        from httpx import AsyncClient as RealAsyncClient
        from pydantic_ai import ConcurrencyLimitedModel

        real_client = RealAsyncClient()
        with patch.object(LLMParseService, "_create_rate_limit_client") as mock_client:
            mock_client.return_value = real_client
            svc = LLMParseService(llm_config=llm_config, service_config=service_config)
        assert isinstance(svc._model, ConcurrencyLimitedModel)


class TestServiceLifecycle:
    @pytest.mark.asyncio
    async def test_close(self, mock_service: LLMParseService) -> None:
        await mock_service.close()
        mock_service._http_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager(
        self, service_config: ServiceConfig, llm_config: LLMConfig
    ) -> None:
        with patch.object(LLMParseService, "_create_rate_limit_client") as mock_client, \
             patch.object(LLMParseService, "_create_model") as mock_model:
            mock_http = AsyncMock()
            mock_client.return_value = mock_http
            mock_model.return_value = MagicMock()

            async with LLMParseService(
                llm_config=llm_config, service_config=service_config
            ) as svc:
                assert svc is not None

            mock_http.aclose.assert_awaited_once()
