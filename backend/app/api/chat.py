"""对话 API (含 SSE 流式)"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..generation.llm import get_llm
from ..generation.prompt import build_messages
from ..ingestion.embedder import get_embedder
from ..logger import log
from ..models import ChatMessage, ChatSession
from ..retrieval.vector_store import get_milvus

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    doc_ids: list[int] | None = None  # 可选:限定检索文档范围
    top_k: int | None = None
    session_id: int | None = None  # 可选:关联会话


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]


@router.post("/ask")
def ask(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """非流式问答"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 检索
    log.info(f"用户提问: {req.question[:80]}...")
    embedder = get_embedder()
    query_vec = embedder.embed(req.question)

    milvus = get_milvus()
    hits = milvus.search(query_vec, top_k=req.top_k, doc_ids=req.doc_ids)

    # 2. 生成
    messages = build_messages(req.question, hits)
    llm = get_llm()
    answer = llm.chat(messages)

    # 3. 保存到 DB
    citations = [
        {
            "doc_id": h["doc_id"],
            "chunk_idx": h["chunk_idx"],
            "page": h["page"],
            "source": h["source"],
            "score": h["score"],
            "text": h["text"][:200] + "..." if len(h["text"]) > 200 else h["text"],
        }
        for h in hits
    ]
    session_id = req.session_id
    if session_id:
        session = db.get(ChatSession, session_id)
        if session is None:
            session = ChatSession(id=session_id, title=req.question[:50])
            db.add(session)
            db.commit()
            db.refresh(session)
        msg = ChatMessage(
            session_id=session.id,
            role="user",
            content=req.question,
        )
        db.add(msg)
        msg2 = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            citations=json.dumps(citations, ensure_ascii=False),
        )
        db.add(msg2)
        db.commit()

    return ChatResponse(answer=answer, citations=citations)


@router.post("/stream")
async def stream_answer(req: ChatRequest, db: Session = Depends(get_db)):
    """SSE 流式问答

    返回 text/event-stream,每条事件:
    - {"type":"citations","data":[...]}
    - {"type":"token","data":"..."}
    - {"type":"done","data":""}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    log.info(f"[stream] 用户提问: {req.question[:80]}...")

    # 1. 检索
    embedder = get_embedder()
    query_vec = embedder.embed(req.question)
    milvus = get_milvus()
    hits = milvus.search(query_vec, top_k=req.top_k, doc_ids=req.doc_ids)

    citations = [
        {
            "doc_id": h["doc_id"],
            "chunk_idx": h["chunk_idx"],
            "page": h["page"],
            "source": h["source"],
            "score": round(h["score"], 4),
            "text": h["text"][:300] + ("..." if len(h["text"]) > 300 else ""),
        }
        for h in hits
    ]

    messages = build_messages(req.question, hits)
    llm = get_llm()

    async def event_generator():
        # 先推送引用
        yield f"data: {json.dumps({'type': 'citations', 'data': citations}, ensure_ascii=False)}\n\n"

        # 流式推送 token
        full_answer = ""
        try:
            async for token in llm.stream_chat(messages):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.exception("流式生成失败")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
            return

        # 推送完成
        yield f"data: {json.dumps({'type': 'done', 'data': full_answer}, ensure_ascii=False)}\n\n"

        # 保存到 DB
        if req.session_id:
            try:
                session = db.get(ChatSession, req.session_id)
                if session is None:
                    session = ChatSession(id=req.session_id, title=req.question[:50])
                    db.add(session)
                    db.commit()
                    db.refresh(session)
                db.add(ChatMessage(session_id=session.id, role="user", content=req.question))
                db.add(
                    ChatMessage(
                        session_id=session.id,
                        role="assistant",
                        content=full_answer,
                        citations=json.dumps(citations, ensure_ascii=False),
                    )
                )
                db.commit()
            except Exception as e:
                log.warning(f"保存对话历史失败: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """列出所有会话"""
    sessions = db.execute(select(ChatSession).order_by(ChatSession.created_at.desc())).scalars().all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: int, db: Session = Depends(get_db)):
    """获取会话的消息历史"""
    messages = (
        db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id))
        .scalars()
        .all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "citations": json.loads(m.citations) if m.citations else None,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]
