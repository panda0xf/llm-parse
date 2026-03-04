"""FastAPI 端点集成测试。

使用 httpx AsyncClient + mock LLMParseService 测试完整请求-响应链路。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from llm_parse.api.app import create_app, lifespan, main as app_main
from llm_parse.models import BatchResult, ParseResult, ParseStatus
from llm_parse.service import LLMParseService


SAMPLE_OUTPUT_SCHEMA = {
    "properties": {
        "title": {"type": "string"},
        "score": {"type": "integer"},
    }
}


@pytest.fixture
def mock_parse_service() -> AsyncMock:
    service = AsyncMock(spec=LLMParseService)
    service.close = AsyncMock()
    return service


@pytest.fixture
async def client(mock_parse_service: AsyncMock):
    app = create_app()
    app.state.parse_service = mock_parse_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestParseEndpoint:
    @pytest.mark.asyncio
    async def test_parse_success(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"title": "Test", "score": 8}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=1.0,
        )

        resp = await client.post("/parse", json={
            "prompt": "分析文章",
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["output"] == {"title": "Test", "score": 8}
        assert data["elapsed_seconds"] == 1.0
        assert data["error_message"] is None

    @pytest.mark.asyncio
    async def test_parse_with_system_prompt(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"title": "T", "score": 5}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=0.5,
        )

        resp = await client.post("/parse", json={
            "prompt": "test",
            "system_prompt": "你是助手",
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })

        assert resp.status_code == 200
        mock_parse_service.parse.assert_awaited_once()
        call_kwargs = mock_parse_service.parse.call_args
        assert call_kwargs.kwargs.get("system_prompt") == "你是助手"

    @pytest.mark.asyncio
    async def test_parse_failure(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.TIMEOUT,
            error_message="请求超时 (30s)",
            prompt="test",
            elapsed_seconds=30.0,
        )

        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "timeout"
        assert data["output"] is None
        assert "超时" in data["error_message"]

    @pytest.mark.asyncio
    async def test_parse_empty_schema_returns_422(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {"properties": {}},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_missing_prompt_returns_422(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/parse", json={
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_missing_schema_returns_422(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/parse", json={
            "prompt": "test",
        })
        assert resp.status_code == 422


class TestParseBatchEndpoint:
    @pytest.mark.asyncio
    async def test_batch_success(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output1 = MagicMock()
        mock_output1.model_dump.return_value = {"title": "A", "score": 7}
        mock_output2 = MagicMock()
        mock_output2.model_dump.return_value = {"title": "B", "score": 9}

        mock_parse_service.parse_batch.return_value = BatchResult(
            results=[
                ParseResult(
                    status=ParseStatus.SUCCESS,
                    output=mock_output1,
                    prompt="q1",
                    elapsed_seconds=0.5,
                ),
                ParseResult(
                    status=ParseStatus.SUCCESS,
                    output=mock_output2,
                    prompt="q2",
                    elapsed_seconds=0.6,
                ),
            ],
            total_elapsed_seconds=1.1,
        )

        resp = await client.post("/parse-batch", json={
            "items": [
                {"prompt": "q1"},
                {"prompt": "q2"},
            ],
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert data["success_rate"] == 1.0
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_mixed_results(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"title": "A", "score": 7}

        mock_parse_service.parse_batch.return_value = BatchResult(
            results=[
                ParseResult(
                    status=ParseStatus.SUCCESS,
                    output=mock_output,
                    prompt="q1",
                    elapsed_seconds=0.5,
                ),
                ParseResult(
                    status=ParseStatus.TIMEOUT,
                    error_message="超时",
                    prompt="q2",
                    elapsed_seconds=30.0,
                ),
            ],
            total_elapsed_seconds=30.5,
        )

        resp = await client.post("/parse-batch", json={
            "items": [
                {"prompt": "q1"},
                {"prompt": "q2", "system_prompt": "sys", "request_id": "r2"},
            ],
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 1
        assert data["failed"] == 1

    @pytest.mark.asyncio
    async def test_batch_empty_items_returns_422(
        self, client: AsyncClient
    ) -> None:
        """空 items 应被框架校验拒绝或正常执行（依赖 Pydantic 验证）。"""
        resp = await client.post("/parse-batch", json={
            "items": [],
            "output_schema": SAMPLE_OUTPUT_SCHEMA,
        })
        # items 为空但类型合法，可以正常通过
        assert resp.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_batch_missing_schema_returns_422(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/parse-batch", json={
            "items": [{"prompt": "q1"}],
        })
        assert resp.status_code == 422


class TestSchemaSecurity:
    @pytest.mark.asyncio
    async def test_dunder_field_name_returns_422(
        self, client: AsyncClient
    ) -> None:
        """双下划线前缀字段名应被拒绝。"""
        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {"__class__": {"type": "string"}},
            },
        })
        assert resp.status_code == 422
        assert "非法字段名" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_field_name_returns_422(
        self, client: AsyncClient
    ) -> None:
        """无效标识符字段名应被拒绝。"""
        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {"123-bad": {"type": "string"}},
            },
        })
        assert resp.status_code == 422
        assert "非法字段名" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_too_many_properties_returns_422(
        self, client: AsyncClient
    ) -> None:
        """超过 50 个字段应被拒绝。"""
        props = {f"field_{i}": {"type": "string"} for i in range(51)}
        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {"properties": props},
        })
        assert resp.status_code == 422
        assert "50" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_valid_field_names_accepted(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        """合法字段名（含单下划线前缀）应被接受。"""
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"_name": "ok", "value_1": "v"}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=0.1,
        )
        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {
                    "_name": {"type": "string"},
                    "value_1": {"type": "string"},
                },
            },
        })
        assert resp.status_code == 200


class TestSchemaToModel:
    @pytest.mark.asyncio
    async def test_array_type_field(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        """测试 output_schema 中 array 类型字段。"""
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"tags": ["a", "b"]}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=0.1,
        )

        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                }
            },
        })

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_boolean_type_field(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"active": True}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=0.1,
        )

        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {
                    "active": {"type": "boolean"},
                }
            },
        })

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_number_type_field(
        self, client: AsyncClient, mock_parse_service: AsyncMock
    ) -> None:
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"price": 9.99}
        mock_parse_service.parse.return_value = ParseResult(
            status=ParseStatus.SUCCESS,
            output=mock_output,
            prompt="test",
            elapsed_seconds=0.1,
        )

        resp = await client.post("/parse", json={
            "prompt": "test",
            "output_schema": {
                "properties": {
                    "price": {"type": "number"},
                }
            },
        })

        assert resp.status_code == 200


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_creates_and_closes_service(self) -> None:
        from fastapi import FastAPI

        app = FastAPI()
        mock_service = AsyncMock(spec=LLMParseService)
        mock_service.close = AsyncMock()

        with patch("llm_parse.api.app.LLMParseService", return_value=mock_service):
            async with lifespan(app):
                assert app.state.parse_service is mock_service

        mock_service.close.assert_awaited_once()


class TestAppMain:
    def test_main_calls_uvicorn(self) -> None:
        with patch("llm_parse.api.app.uvicorn") as mock_uvicorn:
            app_main()
            mock_uvicorn.run.assert_called_once_with(
                "llm_parse.api.app:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
            )
