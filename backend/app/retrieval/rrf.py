"""RRF (Reciprocal Rank Fusion) 融合算法

将多路召回结果按排名倒数加权融合,常用于向量检索 + BM25 检索的合并。
公式: score(d) = Σ 1 / (k + rank_i(d))
其中 k 是平滑常数 (通常取 60),rank_i(d) 是文档 d 在第 i 路召回中的排名 (从 1 开始)
"""
from typing import Sequence


def rrf_fuse(
    result_lists: Sequence[list[dict]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """多路召回结果 RRF 融合

    Args:
        result_lists: 多路召回结果列表,每路是 [{"id", "text", "doc_id", ...}]
            相同 id 视为同一文档
        k: RRF 平滑常数,默认 60
        top_k: 返回 top-K 条 (None 表示返回全部)

    Returns:
        fused: [{"id", "text", "doc_id", "chunk_idx", "page", "source", "score"}]
            score 为 RRF 分数 (非原始相似度)
    """
    if not result_lists:
        return []

    # 累加每个 id 的 RRF 分数,并保留元数据 (以第一次出现为准)
    fused_scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in metadata:
                # 保留第一路出现的元数据
                metadata[doc_id] = {
                    "id": hit["id"],
                    "text": hit["text"],
                    "doc_id": hit["doc_id"],
                    "chunk_idx": hit["chunk_idx"],
                    "page": hit["page"],
                    "source": hit["source"],
                }

    # 按融合分数降序
    sorted_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    return [
        {**metadata[d], "score": fused_scores[d]}
        for d in sorted_ids
    ]
