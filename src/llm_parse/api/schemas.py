"""FastAPI 请求/响应 Schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParseRequestSchema(BaseModel):
    """单条解析请求。"""

    prompt: str = Field(description="用户输入文本")
    system_prompt: str = Field(default="", description="系统提示词")
    output_schema: dict[str, Any] = Field(
        description="期望输出的 JSON Schema，服务端据此动态创建 Pydantic Model"
    )


class ParseResponseSchema(BaseModel):
    """单条解析响应。"""

    status: str = Field(description="解析状态")
    output: dict[str, Any] | None = Field(default=None, description="结构化输出")
    error_message: str | None = Field(default=None, description="错误详情")
    elapsed_seconds: float = Field(default=0.0, description="耗时（秒）")


class BatchItemSchema(BaseModel):
    """批量请求中的单个子请求。"""

    prompt: str = Field(description="用户输入文本")
    system_prompt: str = Field(default="", description="系统提示词")
    request_id: str = Field(default="", description="请求标识")


class BatchRequestSchema(BaseModel):
    """批量解析请求。"""

    items: list[BatchItemSchema] = Field(description="子请求列表")
    output_schema: dict[str, Any] = Field(
        description="所有子请求共用的输出 JSON Schema"
    )


class BatchResponseSchema(BaseModel):
    """批量解析响应。"""

    results: list[ParseResponseSchema] = Field(description="各子请求的结果")
    total: int = Field(description="总数")
    succeeded: int = Field(description="成功数")
    failed: int = Field(description="失败数")
    success_rate: float = Field(description="成功率")
    total_elapsed_seconds: float = Field(description="总耗时（秒）")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
