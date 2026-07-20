"""检索编排器:统一检索入口

流程 (M2):
1. 向量召回 (Milvus) → top_k = retrieval_top_k
2. (可选) BM25 关键词召回 → top_k = retrieval_top_k
3. RRF 融合两路结果 → top_k = retrieval_top_k
4. (可选) Rerank 重排 → top_k = top_k (最终返回数量)
"""
from typing import Optional

from ..config import settings
from ..logger import log
from .bm25 import get_bm25
from .rerank import rerank_hits
from .rrf import rrf_fuse
from .vector_store import get_milvus


def rebuild_bm25_index(doc_ids: list[int] | None = None):
    """重建 BM25 索引

    文档上传/删除后调用,从 Milvus 重新拉取全部 chunks 构建索引。
    """
    bm25 = get_bm25()
    try:
        milvus = get_milvus()
        chunks = milvus.list_all_chunks(doc_ids=doc_ids)
        bm25.rebuild(chunks)
    except Exception as e:
        log.warning(f"BM25 索引重建失败 (后续检索将仅使用向量召回): {e}")


def retrieve(
    query: str,
    query_vector: list[float],
    top_k: Optional[int] = None,
    doc_ids: Optional[list[int]] = None,
) -> list[dict]:
    """统一检索入口

    Args:
        query: 用户原始问题文本 (供 BM25 / Rerank 使用)
        query_vector: 问题的 embedding 向量 (供向量召回使用)
        top_k: 最终返回数量 (None 用 settings.top_k)
        doc_ids: 限定文档 ID 范围

    Returns:
        hits: [{"id", "text", "doc_id", "chunk_idx", "page", "source", "score"}]
    """
    final_top_k = top_k or settings.top_k
    retrieval_top_k = settings.retrieval_top_k

    milvus = get_milvus()

    # 1. 向量召回
    vector_hits = milvus.search(query_vector, top_k=retrieval_top_k, doc_ids=doc_ids)
    log.info(f"[retrieve] 向量召回 {len(vector_hits)} 条")

    # 2. (可选) BM25 召回 + RRF 融合
    if settings.enable_bm25:
        bm25 = get_bm25()
        if bm25.is_empty():
            # 首次或索引为空时尝试重建
            log.info("[retrieve] BM25 索引为空,触发重建")
            rebuild_bm25_index()
        bm25_hits = bm25.search(query, top_k=retrieval_top_k, doc_ids=doc_ids)
        log.info(f"[retrieve] BM25 召回 {len(bm25_hits)} 条")

        if bm25_hits:
            fused = rrf_fuse([vector_hits, bm25_hits], k=settings.rrf_k, top_k=retrieval_top_k)
            log.info(f"[retrieve] RRF 融合后 {len(fused)} 条")
            candidates = fused
        else:
            candidates = vector_hits
    else:
        candidates = vector_hits

    # 3. (可选) Rerank 重排
    if settings.enable_rerank and candidates:
        reranked = rerank_hits(query, candidates, top_n=final_top_k)
        log.info(f"[retrieve] Rerank 后 {len(reranked)} 条")
        return reranked

    # 4. 不启用 Rerank 时直接取前 final_top_k
    return candidates[:final_top_k]
