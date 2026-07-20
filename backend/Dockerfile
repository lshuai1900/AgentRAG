FROM python:3.11-slim

WORKDIR /app

# 系统依赖 (unstructured/pypdf 等需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.3.0" \
    "python-multipart>=0.0.9" \
    "sqlalchemy>=2.0.30" \
    "psycopg2-binary>=2.9.9" \
    "redis>=5.0.0" \
    "pymilvus>=2.4.0" \
    "openai>=1.30.0" \
    "tiktoken>=0.7.0" \
    "python-docx>=1.1.0" \
    "pypdf>=4.2.0" \
    "markdown>=3.6" \
    "rank-bm25>=0.2.2" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0" \
    "python-dotenv>=1.0.0" \
    "loguru>=0.7.2" \
    "tenacity>=8.3.0"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
