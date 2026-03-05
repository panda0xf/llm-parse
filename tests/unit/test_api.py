from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from llm_parse.api.app import app, main
from llm_parse.models import BatchResult, ParseResult, ParseStatus

pytestmark = pytest.mark.asyncio


@patch("llm_parse.api.app.uvicorn.run")
async def test_main(mock_run):
    main()
    mock_run.assert_called_once()


async def test_lifespan():
    from llm_parse.api.app import lifespan

    async with lifespan(app):
        assert hasattr(app.state, "parse_service")


async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


async def test_parse_schema_validity():
    # Set dummy state for these tests
    app.state.parse_service = AsyncMock()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing properties
        response = await client.post("/parse", json={"prompt": "hi", "output_schema": {}})
        assert response.status_code == 422
        assert "不能为空" in response.text

        # Too many properties
        long_props = {f"field_{i}": {"type": "string"} for i in range(51)}
        response = await client.post("/parse", json={"prompt": "hi", "output_schema": {"properties": long_props}})
        assert response.status_code == 422
        assert "不能超过 50" in response.text

        # Invalid property name
        response = await client.post(
            "/parse",
            json={"prompt": "hi", "output_schema": {"properties": {"__private": {"type": "string"}}}},
        )
        assert response.status_code == 422
        assert "双下划线" in response.text


async def test_parse_success():
    class DummyOutput(BaseModel):
        name: str

    dummy_result = ParseResult(
        prompt="hello", output=DummyOutput(name="test"), status=ParseStatus.SUCCESS, elapsed_seconds=0.1
    )

    mock_instance = AsyncMock()
    mock_instance.parse.return_value = dummy_result
    app.state.parse_service = mock_instance

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # We must use proper output schema
        schema = {
            "properties": {
                "name": {"type": "string", "default": ""},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "is_active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }
        body = {"prompt": "hello", "output_schema": schema}
        response = await client.post("/parse", json=body)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["output"]["name"] == "test"


async def test_parse_fallback_types():
    dummy_result: ParseResult[Any] = ParseResult(
        prompt="hello", output=None, status=ParseStatus.SUCCESS, elapsed_seconds=0.1
    )
    mock_instance = AsyncMock()
    mock_instance.parse.return_value = dummy_result
    app.state.parse_service = mock_instance

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Schema with unsupported types and default types
        schema = {
            "properties": {
                "custom_obj": {"type": "object"},
                "arr_custom": {"type": "array", "items": {"type": "object"}},
                "arr_no_items": {"type": "array"},
                "arr_str": {"type": "array", "items": {"type": "string"}},
                "no_type": {},
            }
        }
        body = {"prompt": "hello", "output_schema": schema}
        response = await client.post("/parse", json=body)
        assert response.status_code == 200


async def test_parse_batch_success():
    class DummyOutput(BaseModel):
        name: str

    dummy_result = ParseResult(
        prompt="hello", output=DummyOutput(name="test"), status=ParseStatus.SUCCESS, elapsed_seconds=0.1
    )
    batch_result = BatchResult(
        results=[dummy_result],
        total_elapsed_seconds=0.1,
    )

    mock_instance = AsyncMock()
    mock_instance.parse_batch.return_value = batch_result
    app.state.parse_service = mock_instance

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        schema = {"properties": {"name": {"type": "string"}}}
        body = {"items": [{"prompt": "hello"}], "output_schema": schema}
        response = await client.post("/parse-batch", json=body)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["results"][0]["output"]["name"] == "test"
