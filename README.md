# LLM Parse Service

通用 LLM 结构化解析服务 -- 输入 Prompt + JSON Schema，输出结构化数据。

基于 [pydantic-ai](https://github.com/pydantic/pydantic-ai) 构建，提供多层重试、并发控制和超时保护。

## 功能特性

- **结构化输出**: 通过 JSON Schema 定义期望格式，自动校验 LLM 返回结果
- **多层稳定性**: pydantic-ai 输出校验重试 → HTTP 429/5xx 退避重试 → 请求超时
- **并发控制**: Agent 级 / Model 级 / 服务级三层并发限制
- **Result Pattern**: 单条失败不影响批量解析的其他请求
- **FastAPI 服务**: 提供 REST API，支持单条和批量解析
- **类型安全**: 全面的 type hints，PEP 561 兼容

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (推荐)

### 安装

```bash
# 克隆项目
git clone <repo-url> && cd parse

# 安装依赖
uv sync

# 安装开发依赖
uv pip install -e ".[dev]"
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY 等配置
```

### 启动服务

```bash
# 方式一：直接运行
uv run python main.py

# 方式二：使用 uvicorn
uv run uvicorn llm_parse.api.app:app --reload

# 方式三：安装后使用命令行
llm-parse
```

服务默认监听 `http://0.0.0.0:8000`。

## API 接口

### 健康检查

```
GET /health
```

### 单条解析

```
POST /parse
```

请求体:

```json
{
  "prompt": "分析这篇文章：...",
  "system_prompt": "你是内容分析助手",
  "output_schema": {
    "properties": {
      "title": {"type": "string"},
      "summary": {"type": "string"},
      "keywords": {"type": "array", "items": {"type": "string"}},
      "score": {"type": "integer"}
    }
  }
}
```

响应:

```json
{
  "status": "success",
  "output": {
    "title": "...",
    "summary": "...",
    "keywords": ["..."],
    "score": 8
  },
  "error_message": null,
  "elapsed_seconds": 2.35
}
```

### 批量解析

```
POST /parse-batch
```

请求体:

```json
{
  "items": [
    {"prompt": "文章1...", "request_id": "a1"},
    {"prompt": "文章2...", "request_id": "a2"}
  ],
  "output_schema": {
    "properties": {
      "title": {"type": "string"},
      "score": {"type": "integer"}
    }
  }
}
```

### 作为 Python 库使用

```python
from llm_parse import LLMParseService, ParseRequest
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str = Field(description="标题")
    keywords: list[str] = Field(description="关键词")

async with LLMParseService() as service:
    # 单条解析
    result = await service.parse("分析这篇文章...", Summary, system_prompt="你是分析助手")
    if result.is_success:
        print(result.output.title)

    # 批量解析
    batch = await service.parse_batch([
        ParseRequest(prompt="文章1", output_type=Summary),
        ParseRequest(prompt="文章2", output_type=Summary),
    ])
    print(f"成功率: {batch.success_rate:.1%}")
```

## 配置说明

通过 `.env` 文件或环境变量配置，详见 [.env.example](.env.example)。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 端点 |
| `OPENAI_API_KEY` | (必填) | API 密钥 |
| `MODEL_NAME` | `gpt-4o-mini` | 模型名称 |
| `PARSE_OUTPUT_RETRIES` | `3` | 输出校验重试次数 |
| `PARSE_HTTP_MAX_RETRIES` | `5` | HTTP 重试次数 |
| `PARSE_MAX_AGENT_CONCURRENCY` | `10` | Agent 并发数 |
| `PARSE_MAX_MODEL_CONCURRENCY` | `20` | Model HTTP 并发数 |
| `PARSE_SERVICE_SEMAPHORE_LIMIT` | `50` | 全局信号量上限 |
| `PARSE_BATCH_CHUNK_SIZE` | `20` | 批量分片大小 |
| `PARSE_REQUEST_TIMEOUT` | `120.0` | 请求超时（秒） |

## 开发

### 运行测试

```bash
# 运行全部测试
uv run pytest

# 带覆盖率报告
uv run pytest --cov=llm_parse --cov-report=term-missing

# 仅运行单元测试
uv run pytest tests/unit/

# 仅运行集成测试
uv run pytest tests/integration/
```

### 运行示例脚本

```bash
uv run python scripts/demo.py
```

## 项目结构

```
├── main.py                    # 服务启动入口
├── pyproject.toml
├── scripts/
│   └── demo.py                # 演示脚本
├── src/
│   └── llm_parse/             # 核心包
│       ├── __init__.py
│       ├── config.py           # 配置管理
│       ├── exceptions.py       # 异常层级
│       ├── models.py           # 数据模型
│       ├── service.py          # LLMParseService
│       └── api/                # FastAPI 服务层
│           ├── app.py          # 应用工厂 + lifespan
│           ├── routes.py       # 路由
│           ├── schemas.py      # 请求/响应 schema
│           └── dependencies.py # 依赖注入
└── tests/
    ├── conftest.py
    ├── unit/                   # 单元测试
    └── integration/            # 集成测试
```
