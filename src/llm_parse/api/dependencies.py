"""FastAPI 依赖注入。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from llm_parse.service import LLMParseService


def get_parse_service(request: Request) -> LLMParseService:
    """从应用 state 中获取 LLMParseService 实例。"""
    from typing import cast

    return cast("LLMParseService", request.app.state.parse_service)
