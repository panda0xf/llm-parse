"""FastAPI 应用工厂与生命周期管理。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from llm_parse.api.routes import router
from llm_parse.service import LLMParseService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """管理 LLMParseService 的创建与销毁。"""
    service = LLMParseService()
    app.state.parse_service = service
    logger.info("LLMParseService 已就绪")
    try:
        yield
    finally:
        await service.close()
        logger.info("LLMParseService 已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(
        title="LLM Parse Service",
        description="通用 LLM 结构化解析服务",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """uvicorn 启动入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    uvicorn.run(
        "llm_parse.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
