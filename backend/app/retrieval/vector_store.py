"""Milvus 向量库封装"""
from typing import Sequence

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ..config import settings
from ..logger import log
from ..ingestion.splitter import Chunk

COLLECTION_NAME = "rag_chunks"
VECTOR_FIELD = "vector"
TEXT_FIELD = "text"
DOC_ID_FIELD = "doc_id"
CHUNK_IDX_FIELD = "chunk_idx"
PAGE_FIELD = "page"
SOURCE_FIELD = "source"
PK_FIELD = "id"


class MilvusStore:
    """Milvus 客户端封装"""

    _connected = False

    def __init__(self):
        self.connect()
        self._ensure_collection()

    def connect(self):
        if MilvusStore._connected:
            return
        log.info(f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=str(settings.milvus_port),
        )
        MilvusStore._connected = True
        log.info("Milvus 连接成功")

    def _ensure_collection(self):
        """创建 collection (如果不存在)"""
        if utility.has_collection(COLLECTION_NAME):
            log.info(f"Milvus collection 已存在: {COLLECTION_NAME}")
            return

        log.info(f"创建 Milvus collection: {COLLECTION_NAME}")
        fields = [
            FieldSchema(name=PK_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dimension),
            FieldSchema(name=TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=DOC_ID_FIELD, dtype=DataType.INT64),
            FieldSchema(name=CHUNK_IDX_FIELD, dtype=DataType.INT64),
            FieldSchema(name=PAGE_FIELD, dtype=DataType.INT64),
            FieldSchema(name=SOURCE_FIELD, dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields=fields, description="RAG chunks collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema, using="default")

        # 创建向量索引 (IVF_FLAT + COSINE)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name=VECTOR_FIELD, index_params=index_params)
        log.info("Milvus 索引创建完成 (IVF_FLAT, COSINE)")

    def get_collection(self) -> Collection:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection

    def insert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> list[int]:
        """插入 chunks 与对应向量,返回生成的 id 列表"""
        assert len(chunks) == len(vectors), f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致"
        collection = self.get_collection()

        data = [
            vectors,  # vector
            [c.text for c in chunks],  # text
            [c.doc_id for c in chunks],  # doc_id
            [c.chunk_idx for c in chunks],  # chunk_idx
            [c.page for c in chunks],  # page
            [c.source for c in chunks],  # source
        ]
        result = collection.insert(data)
        collection.flush()
        ids = result.primary_keys
        log.info(f"插入 {len(chunks)} 条 chunks 到 Milvus, ids: {ids[:5]}...")
        return ids

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        doc_ids: list[int] | None = None,
    ) -> list[dict]:
        """向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            doc_ids: 限定文档 ID (可选)

        Returns:
            hits: [{"id", "text", "doc_id", "chunk_idx", "page", "source", "score"}]
        """
        top_k = top_k or settings.top_k
        collection = self.get_collection()

        # 构造过滤表达式
        expr = ""
        if doc_ids:
            expr = f"doc_id in {doc_ids}"

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_vector],
            anns_field=VECTOR_FIELD,
            param=search_params,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=[TEXT_FIELD, DOC_ID_FIELD, CHUNK_IDX_FIELD, PAGE_FIELD, SOURCE_FIELD],
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "text": hit.entity.get(TEXT_FIELD),
                    "doc_id": hit.entity.get(DOC_ID_FIELD),
                    "chunk_idx": hit.entity.get(CHUNK_IDX_FIELD),
                    "page": hit.entity.get(PAGE_FIELD),
                    "source": hit.entity.get(SOURCE_FIELD),
                }
            )
        log.info(f"向量检索 top_k={top_k}, 命中 {len(hits)} 条, 最高分: {hits[0]['score'] if hits else 'N/A'}")
        return hits

    def delete_by_doc(self, doc_id: int):
        """删除指定文档的所有 chunks"""
        collection = self.get_collection()
        collection.delete(expr=f"doc_id == {doc_id}")
        log.info(f"已删除 doc_id={doc_id} 的所有 chunks")

    def list_all_chunks(self, doc_ids: list[int] | None = None) -> list[dict]:
        """列出所有 chunks (供 BM25 重建索引用)

        Args:
            doc_ids: 限定文档 ID (None 表示全部)

        Returns:
            chunks: [{"id", "text", "doc_id", "chunk_idx", "page", "source"}]
        """
        collection = self.get_collection()
        expr = f"doc_id in {doc_ids}" if doc_ids else None
        results = collection.query(
            expr=expr,
            output_fields=[TEXT_FIELD, DOC_ID_FIELD, CHUNK_IDX_FIELD, PAGE_FIELD, SOURCE_FIELD],
            limit=16384,  # 单次最多取 16k 条,后续如需可分页
        )
        chunks = []
        for r in results:
            chunks.append({
                "id": r.get("id"),
                "text": r.get(TEXT_FIELD, ""),
                "doc_id": r.get(DOC_ID_FIELD),
                "chunk_idx": r.get(CHUNK_IDX_FIELD),
                "page": r.get(PAGE_FIELD),
                "source": r.get(SOURCE_FIELD, ""),
            })
        log.info(f"从 Milvus 读取 {len(chunks)} 条 chunks 用于 BM25 索引")
        return chunks


_store: MilvusStore | None = None


def get_milvus() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
