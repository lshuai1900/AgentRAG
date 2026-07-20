"""文档管理 API"""
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..ingestion.embedder import get_embedder
from ..ingestion.parser import detect_file_type, parse_document
from ..ingestion.splitter import split_pages
from ..logger import log
from ..models import Document
from ..retrieval.orchestrator import rebuild_bm25_index
from ..retrieval.vector_store import get_milvus

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 允许的文件扩展名 (.doc 旧版格式不支持,需提示用户转换为 .docx)
ALLOWED_EXTS = {"pdf", "docx", "md", "markdown"}


def _sanitize_error(msg: str) -> str:
    """脱敏错误信息,避免向前端暴露 API Key/密码/DSN 等敏感信息"""
    import re

    # 屏蔽 Bearer xxx / sk-xxx / password=xxx 等
    msg = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-]+", r"\1***", msg)
    msg = re.sub(r"(sk-[A-Za-z0-9]{6})[A-Za-z0-9]*", r"\1***", msg)
    msg = re.sub(r"(://[^:\s]+:)[^@\s]+(@)", r"\1***\2", msg)
    return msg


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
        # 对 .doc 给出明确提示
        ext = Path(file.filename).suffix.lower().lstrip(".")
        if ext == "doc":
            raise HTTPException(
                status_code=400,
                detail="暂不支持旧版 .doc 格式,请先在 Word 中另存为 .docx 后再上传",
            )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 .{ext},仅支持: pdf / docx / md",
        )

    # 保存到 /data/uploads
    upload_dir = Path(settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    try:
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        log.exception(f"保存上传文件失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}") from e
    finally:
        await file.close()

    file_size = file_path.stat().st_size
    log.info(f"文件已保存: {file_path} ({file_size} bytes)")

    # 空文件检查
    if file_size == 0:
        try:
            file_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="文件为空,无法处理")

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
            raise ValueError("文档切分后无可用 chunks (可能内容过短或为空白)")

        # 3. embedding
        try:
            embedder = get_embedder()
            vectors = embedder.embed_batch([c.text for c in chunks])
        except Exception as e:
            log.exception(f"Embedding 调用失败: doc_id={doc.id}")
            raise ValueError(
                "Embedding 接口调用失败,请检查 DASHSCOPE_API_KEY 是否正确配置 "
                f"或网络是否可达 (详情见后端日志)"
            ) from e

        # 4. 入库 Milvus
        try:
            milvus = get_milvus()
            milvus.insert_chunks(chunks, vectors)
        except Exception as e:
            log.exception(f"Milvus 写入失败: doc_id={doc.id}")
            raise ValueError(
                "向量库写入失败,请确认 Milvus 服务已启动 "
                "(docker compose ps 查看 milvus-standalone 状态)"
            ) from e

        # 5. 更新状态
        doc.chunk_count = len(chunks)
        doc.status = "ready"
        db.commit()
        log.info(f"文档处理完成: id={doc.id}, chunks={doc.chunk_count}")

        # 6. 触发 BM25 索引重建 (异步执行可后续优化,当前同步重建)
        if settings.enable_bm25:
            try:
                rebuild_bm25_index()
            except Exception as e:
                log.warning(f"BM25 索引重建失败 (不影响向量检索): {e}")

        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "created_at": doc.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except ValueError as e:
        # 用户可理解的错误 (解析失败/切分失败/embedding 配置等)
        log.warning(f"文档处理失败 (业务错误): id={doc.id}, err={e}")
        doc.status = "failed"
        doc.error_msg = str(e)
        db.commit()
        raise HTTPException(status_code=400, detail=_sanitize_error(str(e))) from e
    except Exception as e:
        # 未预期错误,完整堆栈进日志,前端只看到脱敏后的简短信息
        log.exception(f"文档处理失败 (未预期): id={doc.id}")
        doc.status = "failed"
        doc.error_msg = str(e)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=_sanitize_error(f"文档处理失败: {e}"),
        ) from e


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

    # 4. 触发 BM25 索引重建
    if settings.enable_bm25:
        try:
            rebuild_bm25_index()
        except Exception as e:
            log.warning(f"BM25 索引重建失败 (不影响向量检索): {e}")

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
