"""Rerank 重排模块

走阿里 DashScope 兼容接口 (gte-rerank-v2 / gte-rerank),按相关性对候选 chunks 重新排序。

DashScope 兼容协议下,rerank 暂未走 OpenAI 标准化路径,使用原生 /services/rerank 接口。
"""
from typing import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..logger import log


class Reranker:
    """阿里 DashScope Rerank 客户端"""

    def __init__(self):
        self.api_key = settings.dashscope_api_key
        self.model = settings.rerank_model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置,无法启用 rerank")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[dict]:
        """对候选 documents 相对 query 重排

        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_n: 返回前 N 条 (None 表示返回全部,按重排分数降序)

        Returns:
            results: [{"index": 原始下标, "relevance_score": 分数}]
        """
        if not documents:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": list(documents),
            },
            "parameters": {
                "top_n": top_n if top_n is not None else len(documents),
                "return_documents": False,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/services/rerank",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.exception(f"Rerank HTTP 错误: {e.response.status_code} {e.response.text[:300]}")
            raise RuntimeError(f"Rerank 接口返回 {e.response.status_code}: {e.response.text[:200]}") from e
        except httpx.HTTPError as e:
            log.exception("Rerank 网络错误")
            raise RuntimeError(f"Rerank 网络错误: {e}") from e

        data = resp.json()
        # DashScope rerank 返回结构: {"output": {"results": [{"index": int, "relevance_score": float}]}}
        results = data.get("output", {}).get("results", [])
        log.info(f"Rerank 完成: 输入 {len(documents)} 条, 返回 {len(results)} 条")
        return [
            {"index": r["index"], "relevance_score": float(r["relevance_score"])}
            for r in results
        ]


def rerank_hits(
    query: str,
    hits: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """对检索结果 hits 重排

    Args:
        query: 查询文本
        hits: [{"id", "text", ...}] 候选结果
        top_n: 返回前 N 条

    Returns:
        reranked: [{"id", "text", "doc_id", ..., "score": relevance_score}]
            (score 字段被替换为 rerank 分数)
    """
    if not hits:
        return []

    if not settings.dashscope_api_key:
        log.warning("DASHSCOPE_API_KEY 未配置,跳过 rerank,返回原顺序")
        return hits[:top_n] if top_n else hits

    reranker = _get_reranker()
    if reranker is None:
        return hits[:top_n] if top_n else hits

    documents = [h["text"] for h in hits]
    try:
        results = reranker.rerank(query, documents, top_n=top_n)
    except Exception as e:
        log.warning(f"Rerank 失败,降级使用原顺序: {e}")
        return hits[:top_n] if top_n else hits

    reranked = []
    for r in results:
        idx = r["index"]
        if idx >= len(hits):
            continue
        h = dict(hits[idx])
        h["score"] = r["relevance_score"]
        reranked.append(h)
    return reranked


_reranker: Reranker | None = None


def _get_reranker() -> Reranker | None:
    global _reranker
    if _reranker is None:
        try:
            _reranker = Reranker()
        except Exception as e:
            log.warning(f"Reranker 初始化失败: {e}")
            return None
    return _reranker
