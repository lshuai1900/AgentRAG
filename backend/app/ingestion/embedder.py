"""Embedding 客户端 (阿里 DashScope OpenAI 兼容接口)

DashScope OpenAI 兼容接口限制:
- base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
- text-embedding-v3 默认 1024 维 (可选 512/768/1024)
- 单次批量最多 25 条 (此处取 10 更稳妥)
- 单条文本不超过 8192 tokens
- 需显式传 dimensions 与 encoding_format
"""
from typing import Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..logger import log


# DashScope 单次批量上限 25 条,此处取 10 兼顾稳定与吞吐
BATCH_SIZE = 10
# 单条文本最大字符数 (粗略对应 8192 tokens 上限)
MAX_TEXT_CHARS = 20000


class Embedder:
    """Embedding 客户端 - 走 DashScope OpenAI 兼容协议"""

    def __init__(self):
        if not settings.dashscope_api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未配置,请在 .env 中设置阿里 DashScope API Key"
            )
        # DashScope 提供 OpenAI 兼容接口,可直接复用 openai SDK
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self.model = settings.dashscope_embedding_model
        self.dimension = settings.embedding_dimension

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        """单条文本 embedding"""
        text = self._normalize_text(text)
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension,
            encoding_format="float",
        )
        return resp.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量 embedding

        Args:
            texts: 文本列表

        Returns:
            vectors: 与 texts 等长的向量列表 (空文本会返回零向量占位,
                     保持索引一致,但调用方应在入库前过滤)
        """
        if not texts:
            return []

        # 规范化: 空字符串替换为单空格,避免 DashScope 拒绝空 input
        normalized = [self._normalize_text(t) for t in texts]

        results: list[list[float]] = [None] * len(normalized)  # type: ignore[list-item]
        for i in range(0, len(normalized), BATCH_SIZE):
            batch = normalized[i : i + BATCH_SIZE]
            log.info(f"Embedding batch {i // BATCH_SIZE + 1}: {len(batch)} 条")
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension,
                    encoding_format="float",
                )
            except Exception as e:
                # 把 API 错误的细节(状态码/响应体)打到日志,方便排查
                self._log_api_error(e, batch)
                raise

            # 按 index 排序后写回原位置
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            for j, item in enumerate(sorted_data):
                results[i + j] = item.embedding

        # 兜底:理论上不会出现,但保险起见
        if any(r is None for r in results):
            missing = [i for i, r in enumerate(results) if r is None]
            raise RuntimeError(f"Embedding 部分结果缺失,索引: {missing}")

        return results  # type: ignore[return-value]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """规范化文本: 空字符串 → 单空格; 超长截断"""
        if text is None:
            return " "
        text = text.strip()
        if not text:
            return " "
        if len(text) > MAX_TEXT_CHARS:
            log.warning(f"Embedding 文本过长 ({len(text)} chars),截断到 {MAX_TEXT_CHARS}")
            text = text[:MAX_TEXT_CHARS]
        return text

    @staticmethod
    def _log_api_error(exc: Exception, batch: list[str]):
        """记录 DashScope API 错误的详细信息"""
        # openai.APIStatusError 含 response 属性
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", "?")
            try:
                body = resp.text[:500] if hasattr(resp, "text") else str(resp)
            except Exception:
                body = "<unreadable>"
            log.error(f"DashScope API 错误 status={status} body={body}")
        log.error(
            f"Embedding 调用异常: type={type(exc).__name__} msg={exc} "
            f"batch_size={len(batch)}"
        )


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def reset_embedder():
    """重置单例 (配置变更或测试时使用)"""
    global _embedder
    _embedder = None
