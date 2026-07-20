"""数据库连接"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base

# 同步引擎 (M1 阶段先用同步,简化代码)
engine = create_engine(settings.postgres_dsn, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """创建所有表"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖:获取 DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
