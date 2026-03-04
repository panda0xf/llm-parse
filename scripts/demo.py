"""LLM 结构化解析服务演示脚本。

演示 LLMParseService 的三种使用模式：
1. 单次解析（Result Pattern）
2. 异常模式（parse_or_raise）
3. 批量并发解析

使用方法:
    1. 复制 .env.example 为 .env 并填写配置
    2. uv run python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pydantic import BaseModel, Field

from llm_parse import LLMParseService, ParseRequest


class ArticleSummary(BaseModel):
    """文章摘要结构化输出。"""

    title: str = Field(description="文章标题")
    summary: str = Field(description="100 字以内的文章摘要")
    keywords: list[str] = Field(description="3-5 个关键词")
    sentiment: str = Field(description="情感倾向: positive / negative / neutral")


class CodeReview(BaseModel):
    """代码审查结构化输出。"""

    language: str = Field(description="编程语言")
    issues: list[str] = Field(description="发现的问题列表")
    suggestions: list[str] = Field(description="改进建议列表")
    quality_score: int = Field(description="代码质量评分 1-10", ge=1, le=10)
    explanation: str = Field(description="总体评价说明")


SUMMARIZER_SYSTEM_PROMPT = """\
你是一位专业的内容分析助手。你的任务是：
1. 提取文章的核心标题
2. 用不超过 100 字概括文章要点
3. 提炼 3-5 个关键词
4. 判断文章的情感倾向（positive / negative / neutral）

请始终返回结构化的 JSON 格式数据。
"""

CODE_REVIEWER_SYSTEM_PROMPT = """\
你是一位经验丰富的高级软件工程师和代码审查专家。你的任务是：
1. 识别代码使用的编程语言
2. 找出代码中的潜在问题（bug、安全隐患、性能问题等）
3. 提供具体的改进建议
4. 给出 1-10 的代码质量评分
5. 给出总体评价

请基于 SOLID 原则、Clean Code 和工程最佳实践进行评审。
"""

SAMPLE_ARTICLE = """\
近日，OpenAI 发布了 GPT-5 系列模型。该模型在推理能力、多模态理解和代码生成方面
取得了显著突破。GPT-5 的上下文窗口扩展至 100 万 token，并且在数学推理基准测试
中首次超越人类专家水平。业界专家认为，这标志着通用人工智能（AGI）研究迈入了
一个全新阶段。然而，也有研究者对模型的安全性和能源消耗表示担忧。
"""

SAMPLE_CODE = """\
def get_user(id):
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def process_data(data):
    output = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] == data[j] and i != j:
                output.append(data[i])
    return output
"""

SAMPLE_ARTICLES = [
    "苹果公司发布了最新的 iPhone 16 系列，搭载 A18 芯片，AI 功能成为最大亮点。",
    "全球气候变化加剧，多国遭遇极端天气，联合国呼吁加快碳减排步伐。",
    "SpaceX 星舰第七次试飞取得成功，成功实现了助推器回收和轨道飞行。",
]


async def demo_single_parse(service: LLMParseService) -> None:
    """演示单次解析（Result Pattern）。"""
    print("=" * 60)
    print("单次解析演示 - 文章摘要")
    print("=" * 60)

    result = await service.parse(
        prompt=f"请分析以下文章:\n\n{SAMPLE_ARTICLE}",
        output_type=ArticleSummary,
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
    )

    if result.is_success:
        output = result.unwrap()
        print(f"\n标题:     {output.title}")
        print(f"摘要:     {output.summary}")
        print(f"关键词:   {', '.join(output.keywords)}")
        print(f"情感倾向: {output.sentiment}")
    else:
        print(f"\n解析失败: [{result.status.value}] {result.error_message}")

    print(f"耗时: {result.elapsed_seconds:.2f}s")


async def demo_parse_or_raise(service: LLMParseService) -> None:
    """演示异常模式（parse_or_raise）。"""
    print("\n" + "=" * 60)
    print("异常模式演示 - 代码审查")
    print("=" * 60)

    try:
        output = await service.parse_or_raise(
            prompt=f"请审查以下代码:\n\n```python\n{SAMPLE_CODE}\n```",
            output_type=CodeReview,
            system_prompt=CODE_REVIEWER_SYSTEM_PROMPT,
        )
        print(f"\n语言: {output.language}")
        print(f"质量评分: {output.quality_score}/10")
        print("\n发现的问题:")
        for i, issue in enumerate(output.issues, 1):
            print(f"   {i}. {issue}")
        print("\n改进建议:")
        for i, suggestion in enumerate(output.suggestions, 1):
            print(f"   {i}. {suggestion}")
        print(f"\n总体评价: {output.explanation}")
    except Exception as e:
        print(f"\n运行出错: {e}")


async def demo_batch_parse(service: LLMParseService) -> None:
    """演示批量并发解析。"""
    print("\n" + "=" * 60)
    print("批量并发解析演示 - 多篇文章")
    print("=" * 60)

    requests = [
        ParseRequest(
            prompt=f"请分析以下文章:\n\n{article}",
            output_type=ArticleSummary,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            request_id=f"article-{i}",
        )
        for i, article in enumerate(SAMPLE_ARTICLES, 1)
    ]

    batch = await service.parse_batch(requests)

    print(f"\n批量结果统计:")
    print(f"   总数: {batch.total}")
    print(f"   成功: {batch.succeeded}")
    print(f"   失败: {batch.failed}")
    print(f"   成功率: {batch.success_rate:.1%}")
    print(f"   总耗时: {batch.total_elapsed_seconds:.2f}s")

    for i, result in enumerate(batch.results, 1):
        if result.is_success and result.output is not None:
            print(f"\n   [{i}] {result.output.title}")
            print(f"       {result.output.summary[:50]}...")
        else:
            print(f"\n   [{i}] 失败: {result.error_message}")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    print("LLM 结构化解析服务演示")
    print("=" * 60)

    try:
        async with LLMParseService() as service:
            await demo_single_parse(service)
            await demo_parse_or_raise(service)
            await demo_batch_parse(service)
    except Exception as e:
        print(f"\n运行出错: {e}", file=sys.stderr)
        print("请检查 .env 文件中的配置是否正确。", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
