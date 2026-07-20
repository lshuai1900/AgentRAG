# AgentRAG - 自研 RAG 系统

基于 FastAPI + Milvus + DeepSeek 的自研检索增强生成 (RAG) 系统。

## 功能特性

- **多格式文档**：PDF / Word (.docx) / Markdown
- **混合检索**：向量检索 (Milvus) + BM25 关键词召回 (M2 阶段) + RRF 融合 + Rerank
- **流式响应**：SSE 流式输出，打字机效果
- **引用溯源**：每条回答附带引用，可点击查看原文片段
- **Web UI**：Next.js 14 完整界面，含文档管理与对话页
- **DeepSeek 默认**：中文场景性价比高，可切 OpenAI

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + SQLAlchemy + PyMilvus + OpenAI SDK |
| 向量库 | Milvus 2.4 (IVF_FLAT + COSINE) |
| 元数据库 | PostgreSQL 16 |
| 缓存 | Redis 7 |
| 前端 | Next.js 14 + TypeScript + Tailwind CSS |
| LLM | DeepSeek `deepseek-chat` (默认) / OpenAI |
| Embedding | OpenAI `text-embedding-3-small` (1536 维) |
| 部署 | Docker Compose 一键启动 |

## 目录结构

```
.
├── backend/                # FastAPI 后端
│   ├── app/
│   │   ├── main.py         # 入口
│   │   ├── config.py       # 配置
│   │   ├── models.py       # SQLAlchemy 模型
│   │   ├── db.py           # 数据库
│   │   ├── ingestion/      # 文档摄入 (parser/splitter/embedder)
│   │   ├── retrieval/      # 检索 (vector_store)
│   │   ├── generation/     # 生成 (llm/prompt)
│   │   └── api/            # API 路由 (documents/chat)
│   └── Dockerfile
├── frontend/              # Next.js 前端
│   ├── app/
│   │   ├── page.tsx        # 首页
│   │   ├── documents/      # 文档管理
│   │   └── chat/           # 对话页
│   └── Dockerfile
└── docker-compose.yml     # 7 服务编排
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env,填入 DEEPSEEK_API_KEY 和 OPENAI_API_KEY
```

> DeepSeek 暂无 Embedding API，故 Embedding 使用 OpenAI。
> 如需完全本地，可在 `app/ingestion/embedder.py` 替换为本地模型 (如 `bge-large-zh-v1.1`)。

### 2. 启动服务

```bash
docker compose up -d --build
```

等待所有服务启动 (Milvus + Postgres + Redis + Backend + Frontend)：

```bash
docker compose ps
```

### 3. 访问

- 前端：http://localhost:3000
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

### 4. 使用

1. 打开 http://localhost:3000/documents
2. 上传一个 PDF / Word / Markdown 文档，等待状态变为 `ready`
3. 进入 http://localhost:3000/chat
4. 提问，获得带引用的回答

## 里程碑

- [x] **M1**：项目脚手架 + 单文档摄入 + 单轮问答
- [ ] **M2**：BM25 混合检索 + RRF 融合 + Rerank
- [ ] **M3**：引用溯源前端交互完善 + 流式响应优化
- [ ] **M4**：多知识库管理 + 异步摄入 + 评估指标

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/documents/upload` | 上传文档 (multipart) |
| GET | `/api/documents` | 列出所有文档 |
| GET | `/api/documents/{id}` | 获取文档详情 |
| DELETE | `/api/documents/{id}` | 删除文档 (含向量) |
| POST | `/api/chat/ask` | 非流式问答 |
| POST | `/api/chat/stream` | SSE 流式问答 |
| GET | `/api/chat/sessions` | 列出会话 |
| GET | `/api/chat/sessions/{id}/messages` | 会话消息历史 |
| GET | `/api/health` | 健康检查 |

## 配置说明

主要配置项（`.env`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | LLM 提供方 |
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `OPENAI_API_KEY` | - | OpenAI API Key (用于 embedding) |
| `CHUNK_SIZE` | `500` | 切分块大小 (字符) |
| `CHUNK_OVERLAP` | `80` | 切分块重叠 |
| `TOP_K` | `5` | 检索 top-K |

## 本地开发 (不用 Docker)

### 后端

```bash
cd backend
pip install -e .
uvicorn app.main:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

## License

MIT
