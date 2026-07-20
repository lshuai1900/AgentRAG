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

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

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
    top_k: int = 5

    @property
    def chat_model(self) -> str:
        return self.deepseek_chat_model if self.llm_provider == "deepseek" else "gpt-4o-mini"

    @property
    def chat_api_key(self) -> str:
        return self.deepseek_api_key if self.llm_provider == "deepseek" else self.openai_api_key

    @property
    def chat_base_url(self) -> str:
        return self.deepseek_base_url if self.llm_provider == "deepseek" else self.openai_base_url

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
