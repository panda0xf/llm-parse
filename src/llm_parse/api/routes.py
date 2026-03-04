"""FastAPI 路由定义。"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import create_model

from llm_parse.api.dependencies import get_parse_service
from llm_parse.api.schemas import (
    BatchRequestSchema,
    BatchResponseSchema,
    HealthResponse,
    ParseRequestSchema,
    ParseResponseSchema,
)
from llm_parse.models import ParseRequest
from llm_parse.service import LLMParseService

router = APIRouter()

_PYTHON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_MAX_PROPERTIES = 50
_FIELD_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _schema_to_model(schema: dict[str, Any]) -> type:
    """将简化版 JSON Schema 转换为动态 Pydantic Model。

    支持 properties 中的基础类型（string/integer/number/boolean）
    以及 array of string。不支持的类型回退为 Any。

    安全限制：字段数量上限 50，字段名必须为合法 Python 标识符且不允许双下划线前缀。
    """
    properties = schema.get("properties", {})
    if not properties:
        raise HTTPException(
            status_code=422,
            detail="output_schema.properties 不能为空",
        )
    if len(properties) > _MAX_PROPERTIES:
        raise HTTPException(
            status_code=422,
            detail=f"output_schema.properties 字段数量不能超过 {_MAX_PROPERTIES}",
        )
    for name in properties:
        if name.startswith("__") or not _FIELD_NAME_PATTERN.match(name):
            raise HTTPException(
                status_code=422,
                detail=f"非法字段名: {name!r}，字段名必须为合法标识符且不能以双下划线开头",
            )

    field_definitions: dict[str, Any] = {}
    for name, prop in properties.items():
        prop_type = prop.get("type", "string")
        if prop_type == "array":
            item_type = prop.get("items", {}).get("type", "string")
            python_type = list[_PYTHON_TYPE_MAP.get(item_type, str)]  # type: ignore[index]
        else:
            python_type = _PYTHON_TYPE_MAP.get(prop_type, Any)  # type: ignore[assignment]
        default = prop.get("default", ...)
        field_definitions[name] = (python_type, default)

    return create_model("DynamicOutput", **field_definitions)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()


@router.post("/parse", response_model=ParseResponseSchema)
async def parse(
    body: ParseRequestSchema,
    service: LLMParseService = Depends(get_parse_service),
) -> ParseResponseSchema:
    """单条结构化解析。"""
    output_model = _schema_to_model(body.output_schema)
    result = await service.parse(
        prompt=body.prompt,
        output_type=output_model,
        system_prompt=body.system_prompt,
    )
    return ParseResponseSchema(
        status=result.status.value,
        output=result.output.model_dump() if result.output else None,
        error_message=result.error_message,
        elapsed_seconds=result.elapsed_seconds,
    )


@router.post("/parse-batch", response_model=BatchResponseSchema)
async def parse_batch(
    body: BatchRequestSchema,
    service: LLMParseService = Depends(get_parse_service),
) -> BatchResponseSchema:
    """批量并发解析。"""
    output_model = _schema_to_model(body.output_schema)
    requests = [
        ParseRequest(
            prompt=item.prompt,
            output_type=output_model,
            system_prompt=item.system_prompt,
            request_id=item.request_id,
        )
        for item in body.items
    ]
    batch = await service.parse_batch(requests)

    return BatchResponseSchema(
        results=[
            ParseResponseSchema(
                status=r.status.value,
                output=r.output.model_dump() if r.output else None,
                error_message=r.error_message,
                elapsed_seconds=r.elapsed_seconds,
            )
            for r in batch.results
        ],
        total=batch.total,
        succeeded=batch.succeeded,
        failed=batch.failed,
        success_rate=batch.success_rate,
        total_elapsed_seconds=batch.total_elapsed_seconds,
    )
