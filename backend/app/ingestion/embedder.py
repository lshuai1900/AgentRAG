"""Embedding 客户端 (阿里 DashScope OpenAI 兼容接口)"""
from typing import Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..logger import log


class Embedder:
    """Embedding 客户端 - 走 DashScope OpenAI 兼容协议"""

    def __init__(self):
        # DashScope 提供 OpenAI 兼容接口,可直接复用 openai SDK
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self.model = settings.dashscope_embedding_model
        self.dimension = settings.embedding_dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed(self, text: str) -> list[float]:
        """单条文本 embedding"""
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量 embedding (OpenAI 单次最多 2048 条)"""
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = list(texts)[i : i + batch_size]
            log.info(f"Embedding batch {i // batch_size + 1}: {len(batch)} 条")
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # 按 index 排序
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            results.extend([d.embedding for d in sorted_data])
        return results


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
