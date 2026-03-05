"""FastAPI 路由定义。"""

from __future__ import annotations

import re
from typing import Any, Literal, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, create_model

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
_MAX_RECURSION_DEPTH = 10
_FIELD_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _to_model_name(field_name: str) -> str:
    """将 snake_case / kebab-case 字段名转为 PascalCase 模型名。"""
    parts = re.split(r"[_\-\s]+", field_name)
    return "".join(part.capitalize() for part in parts if part) or "Model"


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    """解析 JSON Schema ``$ref`` 内部引用指针。

    仅支持 ``#/$defs/Name`` 和 ``#/definitions/Name`` 格式。
    """
    if not ref.startswith("#/"):
        raise HTTPException(
            status_code=422,
            detail=f"不支持的 $ref 格式: {ref!r}，仅支持内部引用 (#/$defs/... 或 #/definitions/...)",
        )
    parts = ref[2:].split("/")
    current: Any = root_schema
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise HTTPException(
                status_code=422,
                detail=f"$ref 引用路径不存在: {ref!r}",
            )
        current = current[part]
    if not isinstance(current, dict):
        raise HTTPException(
            status_code=422,
            detail=f"$ref 引用目标不是有效的 Schema 对象: {ref!r}",
        )
    return current


def _validate_field_names(properties: dict[str, Any]) -> None:
    """校验字段名合法性：合法标识符且禁止双下划线前缀。"""
    for name in properties:
        if name.startswith("__") or not _FIELD_NAME_PATTERN.match(name):
            raise HTTPException(
                status_code=422,
                detail=f"非法字段名: {name!r}，字段名必须为合法标识符且不能以双下划线开头",
            )


def _resolve_type(
    prop: dict[str, Any],
    root_schema: dict[str, Any],
    field_name: str,
    depth: int,
) -> Any:
    """将单个 JSON Schema 属性定义解析为 Python 类型。

    递归处理 ``$ref``、``enum``、``oneOf``/``anyOf``、嵌套 ``object``、``array``，
    基础类型回退到 :data:`_PYTHON_TYPE_MAP`，无法识别的类型回退为 ``Any``。
    """
    if depth > _MAX_RECURSION_DEPTH:
        raise HTTPException(
            status_code=422,
            detail=f"Schema 嵌套层级超过上限 ({_MAX_RECURSION_DEPTH})",
        )

    if "$ref" in prop:
        ref_name = prop["$ref"].rsplit("/", 1)[-1]
        resolved = _resolve_ref(prop["$ref"], root_schema)
        return _resolve_type(resolved, root_schema, ref_name, depth + 1)

    if "enum" in prop:
        values = prop["enum"]
        if not values:
            raise HTTPException(
                status_code=422,
                detail=f"字段 {field_name!r} 的 enum 不能为空列表",
            )
        if len(values) == 1:
            return Literal[values[0]]  # type: ignore[valid-type]
        return Literal[tuple(values)]  # type: ignore[valid-type]

    for key in ("oneOf", "anyOf"):
        if key in prop:
            sub_schemas: list[dict[str, Any]] = prop[key]
            if not sub_schemas:
                raise HTTPException(
                    status_code=422,
                    detail=f"字段 {field_name!r} 的 {key} 不能为空列表",
                )
            sub_types: list[Any] = []
            for i, sub_schema in enumerate(sub_schemas):
                sub_name = (
                    sub_schema["$ref"].rsplit("/", 1)[-1]
                    if "$ref" in sub_schema
                    else f"{field_name}_option{i}"
                )
                sub_types.append(
                    _resolve_type(sub_schema, root_schema, sub_name, depth + 1),
                )
            if len(sub_types) == 1:
                return sub_types[0]
            return Union[tuple(sub_types)]  # noqa: UP007

    prop_type = prop.get("type", "string")

    if prop_type == "object":
        if "properties" in prop:
            return _build_object_model(
                prop, root_schema, _to_model_name(field_name), depth + 1,
            )
        return dict[str, Any]

    if prop_type == "array":
        items_schema = prop.get("items", {})
        if items_schema:
            item_type = _resolve_type(
                items_schema, root_schema, f"{field_name}_item", depth + 1,
            )
        else:
            item_type: Any = str  # type: ignore[no-redef]
        return list[item_type]

    return _PYTHON_TYPE_MAP.get(prop_type, Any)


def _build_object_model(
    obj_schema: dict[str, Any],
    root_schema: dict[str, Any],
    model_name: str,
    depth: int,
) -> Any:
    """根据 object 类型的 JSON Schema 构建 Pydantic Model。"""
    properties: dict[str, Any] = obj_schema.get("properties", {})
    if len(properties) > _MAX_PROPERTIES:
        raise HTTPException(
            status_code=422,
            detail=f"output_schema.properties 字段数量不能超过 {_MAX_PROPERTIES}",
        )
    _validate_field_names(properties)

    field_definitions: dict[str, Any] = {}
    for name, prop in properties.items():
        python_type = _resolve_type(prop, root_schema, name, depth)
        default = prop.get("default", ...)
        description = prop.get("description")
        if description is not None:
            field_definitions[name] = (
                python_type,
                Field(default=default, description=description),
            )
        else:
            field_definitions[name] = (python_type, default)

    return create_model(model_name, **field_definitions)


def _schema_to_model(schema: dict[str, Any]) -> Any:
    """将 JSON Schema 转换为动态 Pydantic Model。

    支持：
    - 基础类型（string / integer / number / boolean）
    - array（含嵌套类型的数组）
    - 嵌套 object（递归解析为子 Model）
    - enum 约束（转为 ``Literal`` 类型）
    - oneOf / anyOf 联合类型（转为 ``Union``）
    - ``$ref`` 引用（``#/$defs/...`` 和 ``#/definitions/...``）
    - description 透传至 Pydantic ``Field``

    安全限制：
    - 字段数量上限 50
    - 字段名必须为合法 Python 标识符且不允许双下划线前缀
    - 嵌套深度上限 10 层
    """
    properties = schema.get("properties", {})
    if not properties:
        raise HTTPException(
            status_code=422,
            detail="output_schema.properties 不能为空",
        )
    return _build_object_model(schema, schema, "DynamicOutput", depth=0)


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
    result: Any = await service.parse(
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
    requests: list[ParseRequest[Any]] = [
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
