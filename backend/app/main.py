"""FastAPI 应用入口"""
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import api_router
from .config import settings
from .db import init_db
from .logger import log


def _sanitize_error(msg: str) -> str:
    """脱敏错误信息,避免向前端暴露 API Key/密码/DSN 等敏感信息"""
    msg = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-]+", r"\1***", msg)
    msg = re.sub(r"(sk-[A-Za-z0-9]{6})[A-Za-z0-9]*", r"\1***", msg)
    msg = re.sub(r"(://[^:\s]+:)[^@\s]+(@)", r"\1***\2", msg)
    return msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭生命周期"""
    log.info(f"=== RAG Backend 启动 ===")
    log.info(f"LLM Provider: {settings.llm_provider}")
    log.info(f"Chat Model: {settings.chat_model}")
    log.info(f"Embedding Model: {settings.dashscope_embedding_model} (dim={settings.embedding_dimension})")
    log.info(f"Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    log.info(f"Top K: {settings.top_k}")
    log.info(
        f"Retrieval: bm25={settings.enable_bm25}, rerank={settings.enable_rerank}, "
        f"retrieval_top_k={settings.retrieval_top_k}, rrf_k={settings.rrf_k}"
    )
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


# 全局异常处理器:保证任何未捕获异常都返回合法 JSON,而非纯文本 "Internal Server Error"
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"未捕获异常: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": _sanitize_error(f"服务器内部错误: {exc}")},
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
        "embedding_model": settings.dashscope_embedding_model,
    }


@app.get("/")
def root():
    return {"message": "RAG System API", "docs": "/docs"}
