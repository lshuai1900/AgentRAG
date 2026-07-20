"""BM25 关键词检索 (基于 rank-bm25)

设计要点:
- 单例索引:首次调用时从 Milvus 加载所有 chunks 构建索引
- 增量更新:文档上传/删除后调用 rebuild() 重建
- 支持 doc_ids 过滤
- 中文用简单字符级 + 空白切分 (M2 阶段先不引入 jieba,避免额外依赖)
"""
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from ..logger import log


# 简单中英文 tokenizer:英文按非字母数字切,中文按字切
# 不引入 jieba 以保持依赖精简;后续可替换
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """简易中英文分词

    - 英文/数字: 连续字母数字作为一个 token (小写化)
    - 中文: 每个汉字作为一个 token
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    """BM25 索引,内存态"""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []  # 与 bm25 corpus 一一对应的 chunk 元数据
        self._doc_ids: set[int] = set()  # 当前索引包含的 doc_id 集合

    def is_empty(self) -> bool:
        return self._bm25 is None or len(self._docs) == 0

    def rebuild(self, chunks: Sequence[dict]):
        """用 chunks 重建索引

        Args:
            chunks: [{"id", "text", "doc_id", "chunk_idx", "page", "source"}]
        """
        self._docs = list(chunks)
        self._doc_ids = {c["doc_id"] for c in self._docs}
        corpus = [tokenize(c["text"]) for c in self._docs]
        if not corpus:
            self._bm25 = None
            log.info("BM25 索引重建完成 (空)")
            return
        self._bm25 = BM25Okapi(corpus)
        log.info(f"BM25 索引重建完成: {len(self._docs)} 条 chunks, doc_ids={sorted(self._doc_ids)}")

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_ids: list[int] | None = None,
    ) -> list[dict]:
        """关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量
            doc_ids: 限定文档 ID (None 表示全部)

        Returns:
            hits: [{"id", "text", "doc_id", "chunk_idx", "page", "source", "score"}]
        """
        if self.is_empty():
            log.info("BM25 索引为空,跳过关键词检索")
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        # 配对 (idx, score) 并过滤 doc_ids
        pairs = []
        for idx, score in enumerate(scores):
            doc_id = self._docs[idx]["doc_id"]
            if doc_ids and doc_id not in doc_ids:
                continue
            if score > 0:  # 跳过 0 分文档
                pairs.append((idx, float(score)))

        # 按分数降序
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_k]

        hits = []
        for idx, score in pairs:
            doc = self._docs[idx]
            hits.append({
                "id": doc["id"],
                "score": score,
                "text": doc["text"],
                "doc_id": doc["doc_id"],
                "chunk_idx": doc["chunk_idx"],
                "page": doc["page"],
                "source": doc["source"],
            })
        log.info(f"BM25 检索 top_k={top_k}, 命中 {len(hits)} 条, 最高分: {hits[0]['score'] if hits else 'N/A'}")
        return hits


_bm25_store: BM25Store | None = None


def get_bm25() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
