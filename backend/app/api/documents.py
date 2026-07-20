"""文档管理 API"""
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..ingestion.embedder import get_embedder
from ..ingestion.parser import detect_file_type, parse_document
from ..ingestion.splitter import split_pages
from ..logger import log
from ..models import Document
from ..retrieval.vector_store import get_milvus

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 允许的文件扩展名
ALLOWED_EXTS = {"pdf", "docx", "doc", "md", "markdown"}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """上传并处理文档:解析 → 切分 → embedding → 入库 Milvus"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    file_type = detect_file_type(file.filename)
    if file_type is None:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型,仅支持: {ALLOWED_EXTS}")

    # 保存到 /data/uploads
    upload_dir = Path(settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    file_size = file_path.stat().st_size
    log.info(f"文件已保存: {file_path} ({file_size} bytes)")

    # 创建文档记录
    doc = Document(
        filename=file.filename,
        file_path=str(file_path),
        file_type=file_type,
        file_size=file_size,
        status="parsing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    log.info(f"创建文档记录: id={doc.id}, filename={doc.filename}")

    try:
        # 1. 解析
        doc.status = "parsing"
        db.commit()
        _, pages = parse_document(str(file_path))

        # 2. 切分
        doc.status = "embedding"
        db.commit()
        chunks = split_pages(pages, doc_id=doc.id, source=doc.filename)
        log.info(f"切分完成: {len(chunks)} 个 chunks")
        if not chunks:
            raise ValueError("文档切分后无可用 chunks")

        # 3. embedding
        embedder = get_embedder()
        vectors = embedder.embed_batch([c.text for c in chunks])

        # 4. 入库 Milvus
        milvus = get_milvus()
        milvus.insert_chunks(chunks, vectors)

        # 5. 更新状态
        doc.chunk_count = len(chunks)
        doc.status = "ready"
        db.commit()
        log.info(f"文档处理完成: id={doc.id}, chunks={doc.chunk_count}")

        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "created_at": doc.created_at.isoformat(),
        }
    except Exception as e:
        log.exception(f"文档处理失败: id={doc.id}")
        doc.status = "failed"
        doc.error_msg = str(e)
        db.commit()
        return JSONResponse(
            status_code=500,
            content={"detail": f"文档处理失败: {e}", "doc_id": doc.id},
        )


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    """列出所有文档"""
    docs = db.execute(select(Document).order_by(Document.created_at.desc())).scalars().all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "chunk_count": d.chunk_count,
            "status": d.status,
            "error_msg": d.error_msg,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """删除文档:同时删除文件、Milvus chunks、DB 记录"""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 1. 删除 Milvus 中的 chunks
    try:
        milvus = get_milvus()
        milvus.delete_by_doc(doc_id)
    except Exception as e:
        log.warning(f"删除 Milvus chunks 失败 (继续): {e}")

    # 2. 删除文件
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception as e:
        log.warning(f"删除文件失败 (继续): {e}")

    # 3. 删除 DB 记录
    db.delete(doc)
    db.commit()
    log.info(f"文档已删除: id={doc_id}")
    return {"detail": "已删除", "doc_id": doc_id}


@router.get("/{doc_id}")
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """获取单个文档详情"""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "chunk_count": doc.chunk_count,
        "status": doc.status,
        "error_msg": doc.error_msg,
        "created_at": doc.created_at.isoformat(),
    }
