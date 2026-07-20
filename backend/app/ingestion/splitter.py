"""文本切分器:递归字符切分"""
from dataclasses import dataclass

from ..config import settings


@dataclass
class Chunk:
    """切分后的文本块"""

    text: str
    doc_id: int
    chunk_idx: int
    page: int = 1
    source: str = ""  # 文件名


# 中文优先分隔符 (从长到短,先按段落,再按句子)
SEPARATORS = ["\n\n", "\n", "。", ".", "!", "?", "；", ";", "，", ",", " ", ""]


def split_text(
    text: str,
    doc_id: int,
    source: str = "",
    page: int = 1,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """递归字符切分 (参考 LangChain RecursiveCharacterTextSplitter 简化实现)

    Args:
        text: 待切分文本
        doc_id: 所属文档 ID
        source: 文件名 (用于引用溯源)
        page: 页码
        chunk_size: 单块最大字符数 (默认从配置读)
        chunk_overlap: 块之间重叠字符数

    Returns:
        chunks: Chunk 列表
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text.strip():
        return []

    # 递归切分
    pieces = _recursive_split(text, SEPARATORS, chunk_size)

    # 合并过小块,并添加 overlap
    chunks: list[Chunk] = []
    current = ""
    idx = 0
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(
                    Chunk(
                        text=current.strip(),
                        doc_id=doc_id,
                        chunk_idx=idx,
                        page=page,
                        source=source,
                    )
                )
                idx += 1
                # overlap:保留上一块末尾一部分
                current = current[-chunk_overlap:] + piece if chunk_overlap > 0 else piece
            else:
                current = piece
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                doc_id=doc_id,
                chunk_idx=idx,
                page=page,
                source=source,
            )
        )

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """递归切分:按分隔符从前往后依次尝试"""
    if len(text) <= chunk_size:
        return [text]

    # 找到第一个能切分的分隔符
    for i, sep in enumerate(separators):
        if sep == "":
            continue
        if sep in text:
            parts = text.split(sep)
            result = []
            for part in parts:
                if not part:
                    continue
                if len(part) > chunk_size:
                    # 递归用剩余分隔符切
                    result.extend(_recursive_split(part, separators[i + 1 :], chunk_size))
                else:
                    result.append(part + sep)
            return result
    # 没有分隔符,硬切
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def split_pages(
    pages: list[dict],
    doc_id: int,
    source: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """对多页文本切分,保留页码信息"""
    all_chunks: list[Chunk] = []
    idx = 0
    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        if not page_text.strip():
            continue
        pieces = _recursive_split(page_text, SEPARATORS, chunk_size or settings.chunk_size)

        current = ""
        for piece in pieces:
            if not piece:
                continue
            if len(current) + len(piece) <= (chunk_size or settings.chunk_size):
                current += piece
            else:
                if current.strip():
                    all_chunks.append(
                        Chunk(
                            text=current.strip(),
                            doc_id=doc_id,
                            chunk_idx=idx,
                            page=page_num,
                            source=source,
                        )
                    )
                    idx += 1
                    overlap = chunk_overlap or settings.chunk_overlap
                    current = current[-overlap:] + piece if overlap > 0 else piece
                else:
                    current = piece
        if current.strip():
            all_chunks.append(
                Chunk(
                    text=current.strip(),
                    doc_id=doc_id,
                    chunk_idx=idx,
                    page=page_num,
                    source=source,
                )
            )
            idx += 1

    return all_chunks
