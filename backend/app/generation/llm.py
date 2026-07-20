"""LLM 客户端 (DeepSeek / OpenAI 兼容接口)"""
from typing import AsyncIterator

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..logger import log


class LLMClient:
    """LLM 客户端 - 支持 DeepSeek 和 OpenAI"""

    def __init__(self):
        self.sync_client = OpenAI(api_key=settings.chat_api_key, base_url=settings.chat_base_url)
        self.async_client = AsyncOpenAI(api_key=settings.chat_api_key, base_url=settings.chat_base_url)
        self.model = settings.chat_model
        log.info(f"LLM 客户端初始化: provider={settings.llm_provider}, model={self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """同步对话"""
        resp = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def stream_chat(
        self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """流式对话 (SSE 用)"""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
