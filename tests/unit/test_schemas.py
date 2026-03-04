"""API Schema 序列化/反序列化测试。"""

from __future__ import annotations

from llm_parse.api.schemas import (
    BatchItemSchema,
    BatchRequestSchema,
    BatchResponseSchema,
    HealthResponse,
    ParseRequestSchema,
    ParseResponseSchema,
)


class TestParseRequestSchema:
    def test_required_fields(self) -> None:
        schema = ParseRequestSchema(
            prompt="test",
            output_schema={"properties": {"name": {"type": "string"}}},
        )
        assert schema.prompt == "test"
        assert schema.system_prompt == ""

    def test_full_fields(self) -> None:
        schema = ParseRequestSchema(
            prompt="test",
            system_prompt="you are helpful",
            output_schema={"properties": {"x": {"type": "integer"}}},
        )
        assert schema.system_prompt == "you are helpful"

    def test_serialization(self) -> None:
        schema = ParseRequestSchema(
            prompt="test",
            output_schema={"properties": {"name": {"type": "string"}}},
        )
        data = schema.model_dump()
        assert "prompt" in data
        assert "output_schema" in data
        assert "system_prompt" in data


class TestParseResponseSchema:
    def test_success_response(self) -> None:
        resp = ParseResponseSchema(
            status="success",
            output={"name": "test"},
            elapsed_seconds=1.5,
        )
        assert resp.status == "success"
        assert resp.output == {"name": "test"}
        assert resp.error_message is None
        assert resp.elapsed_seconds == 1.5

    def test_error_response(self) -> None:
        resp = ParseResponseSchema(
            status="timeout",
            error_message="请求超时",
            elapsed_seconds=30.0,
        )
        assert resp.output is None
        assert resp.error_message == "请求超时"

    def test_serialization_roundtrip(self) -> None:
        resp = ParseResponseSchema(
            status="success",
            output={"answer": "hello"},
            elapsed_seconds=0.5,
        )
        data = resp.model_dump()
        restored = ParseResponseSchema(**data)
        assert restored == resp


class TestBatchSchemas:
    def test_batch_item(self) -> None:
        item = BatchItemSchema(prompt="q1")
        assert item.system_prompt == ""
        assert item.request_id == ""

    def test_batch_item_full(self) -> None:
        item = BatchItemSchema(
            prompt="q1",
            system_prompt="sys",
            request_id="r1",
        )
        assert item.prompt == "q1"
        assert item.system_prompt == "sys"
        assert item.request_id == "r1"

    def test_batch_request(self) -> None:
        req = BatchRequestSchema(
            items=[BatchItemSchema(prompt="q1"), BatchItemSchema(prompt="q2")],
            output_schema={"properties": {"x": {"type": "string"}}},
        )
        assert len(req.items) == 2

    def test_batch_response(self) -> None:
        resp = BatchResponseSchema(
            results=[
                ParseResponseSchema(status="success", output={"x": "1"}, elapsed_seconds=0.1),
                ParseResponseSchema(status="timeout", error_message="超时", elapsed_seconds=30.0),
            ],
            total=2,
            succeeded=1,
            failed=1,
            success_rate=0.5,
            total_elapsed_seconds=30.1,
        )
        assert resp.total == 2
        assert resp.succeeded == 1
        assert resp.failed == 1
        assert resp.success_rate == 0.5


class TestHealthResponse:
    def test_default(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"

    def test_serialization(self) -> None:
        data = HealthResponse().model_dump()
        assert data == {"status": "ok"}
