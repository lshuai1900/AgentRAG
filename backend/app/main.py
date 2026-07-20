"""FastAPI 应用入口"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .config import settings
from .db import init_db
from .logger import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭生命周期"""
    log.info(f"=== RAG Backend 启动 ===")
    log.info(f"LLM Provider: {settings.llm_provider}")
    log.info(f"Chat Model: {settings.chat_model}")
    log.info(f"Embedding Model: {settings.openai_embedding_model} (dim={settings.embedding_dimension})")
    log.info(f"Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    log.info(f"Top K: {settings.top_k}")
    log.info(f"Data dir: {settings.data_dir}")

    # 初始化数据库表
    try:
        init_db()
        log.info("Postgres 表已就绪")
    except Exception as e:
        log.error(f"Postgres 初始化失败: {e}")

    # 初始化 Milvus (在请求时延迟连接,避免启动失败)
    try:
        from .retrieval.vector_store import get_milvus

        get_milvus()
        log.info("Milvus 已就绪")
    except Exception as e:
        log.warning(f"Milvus 初始化失败 (将在首次请求重试): {e}")

    yield

    log.info("=== RAG Backend 关闭 ===")


app = FastAPI(
    title="RAG System API",
    description="自研 RAG 系统 - FastAPI + Milvus + DeepSeek",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (允许前端跨域)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/api/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "chat_model": settings.chat_model,
        "embedding_model": settings.openai_embedding_model,
    }


@app.get("/")
def root():
    return {"message": "RAG System API", "docs": "/docs"}
