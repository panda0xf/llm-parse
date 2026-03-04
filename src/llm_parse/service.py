"""LLM 结构化解析服务核心模块。

提供通用的 LLM 结构化输出能力：
输入 Pydantic Model 类型 + Prompt，输出填充后的 Pydantic Model 实例。

稳定性保障：pydantic-ai output_retries → HTTP tenacity 重试 → asyncio 超时 → Result Pattern 降级。
并发保障：Agent ConcurrencyLimit → ConcurrencyLimitedModel → asyncio.Semaphore 兜底。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, TypeVar

from httpx import AsyncClient, HTTPStatusError
from pydantic import BaseModel
from pydantic_ai import Agent, ConcurrencyLimit, ConcurrencyLimitedModel
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_AGENT_CACHE_SIZE = 128


class LLMParseService:
    """通用 LLM 结构化解析服务。

    Usage::

        async with LLMParseService() as service:
            result = await service.parse("分析这篇文章...", ArticleSummary)
            if result.is_success:
                print(result.output.title)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        service_config: ServiceConfig | None = None,
    ) -> None:
        self._llm_config = llm_config or get_config()
        self._svc_config = service_config or get_service_config()
        self._semaphore = asyncio.Semaphore(self._svc_config.service_semaphore_limit)
        self._http_client = self._create_rate_limit_client()
        self._model = self._create_model()
        self._agent_cache: OrderedDict[
            tuple[type[BaseModel], str], Agent[None, Any]
        ] = OrderedDict()

        logger.info(
            "LLMParseService 已初始化 | model=%s | agent_concurrency=%d | "
            "model_concurrency=%d | semaphore=%d | output_retries=%d",
            self._llm_config.model_name,
            self._svc_config.max_agent_concurrency,
            self._svc_config.max_model_concurrency,
            self._svc_config.service_semaphore_limit,
            self._svc_config.output_retries,
        )

    async def parse(
        self,
        prompt: str,
        output_type: type[T],
        system_prompt: str = "",
    ) -> ParseResult[T]:
        """解析单条 prompt，返回结构化结果。"""
        start = time.monotonic()
        agent = self._get_or_create_agent(output_type, system_prompt)

        try:
            async with self._semaphore:
                raw_result = await asyncio.wait_for(
                    agent.run(prompt),
                    timeout=self._svc_config.request_timeout,
                )

            elapsed = time.monotonic() - start
            logger.debug(
                "解析成功 | output_type=%s | elapsed=%.2fs",
                output_type.__name__,
                elapsed,
            )
            return ParseResult(
                status=ParseStatus.SUCCESS,
                output=raw_result.output,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            msg = f"请求超时 ({self._svc_config.request_timeout}s)"
            logger.warning("解析超时 | output_type=%s | %s", output_type.__name__, msg)
            return ParseResult(
                status=ParseStatus.TIMEOUT,
                error_message=msg,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

        except UnexpectedModelBehavior as exc:
            elapsed = time.monotonic() - start
            msg = f"LLM 输出校验失败: {exc}"
            logger.warning("输出校验失败 | output_type=%s | %s", output_type.__name__, msg)
            return ParseResult(
                status=ParseStatus.VALIDATION_ERROR,
                error_message=msg,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

        except HTTPStatusError as exc:
            elapsed = time.monotonic() - start
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                msg = f"速率限制 (429), Retry-After={retry_after}"
                logger.warning("速率限制 | %s", msg)
                return ParseResult(
                    status=ParseStatus.RATE_LIMITED,
                    error_message=msg,
                    prompt=prompt,
                    elapsed_seconds=elapsed,
                )
            msg = f"HTTP 错误 {exc.response.status_code}: {exc}"
            logger.error("HTTP 错误 | %s", msg)
            return ParseResult(
                status=ParseStatus.CONNECTION_ERROR,
                error_message=msg,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

        except (ConnectionError, OSError) as exc:
            elapsed = time.monotonic() - start
            msg = f"连接错误: {exc}"
            logger.error("连接错误 | %s", msg)
            return ParseResult(
                status=ParseStatus.CONNECTION_ERROR,
                error_message=msg,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            msg = f"未知错误 ({type(exc).__name__}): {exc}"
            logger.error("未知错误 | output_type=%s | %s", output_type.__name__, msg)
            return ParseResult(
                status=ParseStatus.UNKNOWN_ERROR,
                error_message=msg,
                prompt=prompt,
                elapsed_seconds=elapsed,
            )

    async def parse_batch(
        self,
        requests: list[ParseRequest[T]],
    ) -> BatchResult[T]:
        """批量并发解析，通过 Semaphore 控制并发上限。

        所有请求一次性提交，由 parse() 内部的 Semaphore 自动限流，
        避免分片串行等待导致的吞吐量损失。
        """
        batch_start = time.monotonic()

        tasks = [
            self.parse(
                prompt=req.prompt,
                output_type=req.output_type,
                system_prompt=req.system_prompt,
            )
            for req in requests
        ]
        all_results: list[ParseResult[T]] = list(await asyncio.gather(*tasks))

        batch_elapsed = time.monotonic() - batch_start
        batch_result = BatchResult(
            results=all_results,
            total_elapsed_seconds=batch_elapsed,
        )

        logger.info(
            "批量解析完成 | total=%d | succeeded=%d | failed=%d | "
            "success_rate=%.1f%% | elapsed=%.2fs",
            batch_result.total,
            batch_result.succeeded,
            batch_result.failed,
            batch_result.success_rate * 100,
            batch_elapsed,
        )
        return batch_result

    async def parse_or_raise(
        self,
        prompt: str,
        output_type: type[T],
        system_prompt: str = "",
    ) -> T:
        """解析并直接返回输出，失败时抛出对应异常。

        Raises:
            OutputValidationError: 输出校验失败。
            LLMTimeoutError: 请求超时。
            RateLimitError: 速率限制。
            LLMConnectionError: 连接错误。
            ParseServiceError: 其他错误。
        """
        result = await self.parse(prompt, output_type, system_prompt)

        if result.is_success:
            if result.output is None:
                raise ParseServiceError("解析成功但输出为空")
            return result.output

        error_map: dict[ParseStatus, type[ParseServiceError]] = {
            ParseStatus.VALIDATION_ERROR: OutputValidationError,
            ParseStatus.TIMEOUT: LLMTimeoutError,
            ParseStatus.RATE_LIMITED: RateLimitError,
            ParseStatus.CONNECTION_ERROR: LLMConnectionError,
        }
        exc_cls = error_map.get(result.status, ParseServiceError)
        raise exc_cls(result.error_message or "解析失败")

    def _create_rate_limit_client(self) -> AsyncClient:
        """创建带 429/5xx 重试的 httpx 客户端。"""
        cfg = self._svc_config
        transport = AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(
                    fallback_strategy=wait_exponential(
                        multiplier=1,
                        min=cfg.http_retry_min_wait,
                        max=cfg.http_retry_max_wait,
                    ),
                    max_wait=300,
                ),
                stop=stop_after_attempt(cfg.http_max_retries),
                reraise=True,
            ),
            validate_response=lambda r: r.raise_for_status(),
        )
        return AsyncClient(transport=transport)

    def _create_model(self) -> ConcurrencyLimitedModel:
        """创建带并发限制的 Model。"""
        provider = OpenAIProvider(
            base_url=self._llm_config.openai_base_url,
            api_key=self._llm_config.openai_api_key,
            http_client=self._http_client,
        )
        base_model = OpenAIChatModel(
            self._llm_config.model_name,
            provider=provider,
        )
        return ConcurrencyLimitedModel(
            base_model,
            limiter=self._svc_config.max_model_concurrency,
        )

    def _get_or_create_agent(
        self,
        output_type: type[T],
        system_prompt: str,
    ) -> Agent[None, T]:
        """获取或创建 Agent（按 output_type + system_prompt 缓存，LRU 淘汰）。"""
        cache_key = (output_type, system_prompt)

        if cache_key in self._agent_cache:
            self._agent_cache.move_to_end(cache_key)
            return self._agent_cache[cache_key]

        cfg = self._svc_config
        concurrency_limit = ConcurrencyLimit(
            max_running=cfg.max_agent_concurrency,
            max_queued=cfg.max_agent_queued,
        )

        agent: Agent[None, T] = Agent(
            self._model,
            output_type=output_type,
            system_prompt=system_prompt,
            output_retries=cfg.output_retries,
            max_concurrency=concurrency_limit,
        )

        if len(self._agent_cache) >= _MAX_AGENT_CACHE_SIZE:
            evicted_key, _ = self._agent_cache.popitem(last=False)
            logger.debug("Agent 缓存淘汰 | key=%s", evicted_key)

        self._agent_cache[cache_key] = agent
        logger.debug(
            "创建 Agent | output_type=%s | system_prompt_len=%d | cache_size=%d",
            output_type.__name__,
            len(system_prompt),
            len(self._agent_cache),
        )

        return agent

    async def close(self) -> None:
        """关闭服务，释放底层资源。"""
        await self._http_client.aclose()
        logger.info("LLMParseService 已关闭")

    async def __aenter__(self) -> LLMParseService:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
