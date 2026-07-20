"""应用配置 - 通过环境变量加载"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ===== LLM =====
    llm_provider: Literal["deepseek", "openai"] = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"

    # ===== Embedding (阿里 DashScope) =====
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliy.com/compatible-mode/v1"
    dashscope_embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1024

    # ===== DB =====
    postgres_user: str = "rag"
    postgres_password: str = "rag123"
    postgres_db: str = "rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_host: str = "localhost"
    redis_port: int = 6379

    # ===== Milvus =====
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # ===== App =====
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    data_dir: str = "/data"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 5  # 最终返回给 LLM 的 chunks 数量

    # ===== 检索策略 (M2) =====
    enable_bm25: bool = True  # 是否启用 BM25 关键词召回 + RRF 融合
    enable_rerank: bool = False  # 是否启用 Rerank 重排 (默认关闭,需要额外 API 调用)
    rerank_model: str = "gte-rerank-v2"  # DashScope rerank 模型
    retrieval_top_k: int = 20  # 向量/BM25 各路召回数量 (融合前)
    rrf_k: int = 60  # RRF 平滑常数

    @property
    def chat_model(self) -> str:
        return self.deepseek_chat_model

    @property
    def chat_api_key(self) -> str:
        return self.deepseek_api_key

    @property
    def chat_base_url(self) -> str:
        return self.deepseek_base_url

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
