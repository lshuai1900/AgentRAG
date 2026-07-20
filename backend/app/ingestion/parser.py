"""文档解析器:支持 PDF / Word / Markdown"""
from pathlib import Path
from typing import Optional

from docx import Document as DocxDocument
from pypdf import PdfReader

from ..logger import log


def parse_pdf(file_path: str) -> tuple[str, list[dict]]:
    """解析 PDF,返回 (全文, 分页文本列表)

    Returns:
        full_text: 全文拼接
        pages: [{"page": 1, "text": "..."}]
    """
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"PDF 解析失败,文件可能损坏或非标准 PDF: {e}") from e
    pages = []
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
            parts.append(text)
    return "\n\n".join(parts), pages


def parse_docx(file_path: str) -> tuple[str, list[dict]]:
    """解析 Word docx"""
    try:
        doc = DocxDocument(file_path)
    except Exception as e:
        raise ValueError(f"DOCX 解析失败,文件可能损坏或非标准 .docx: {e}") from e
    pages = []
    parts = []
    current_page = 1
    # docx 没有真正的"页"概念,按段落分块近似
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    full = "\n\n".join(parts)
    if full:
        pages.append({"page": current_page, "text": full})
    return full, pages


def parse_markdown(file_path: str) -> tuple[str, list[dict]]:
    """解析 Markdown (直接读文本)"""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    return text, [{"page": 1, "text": text}]


PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "md": parse_markdown,
    "markdown": parse_markdown,
}


def parse_document(file_path: str) -> tuple[str, list[dict]]:
    """根据扩展名选择解析器

    Args:
        file_path: 文件路径

    Returns:
        (full_text, pages): 全文 与 分页文本列表
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    parser = PARSERS.get(ext)
    if parser is None:
        if ext == "doc":
            raise ValueError(
                "暂不支持旧版 .doc 格式,请先转换为 .docx 后上传 "
                "(Word 中另存为 .docx,或使用在线转换工具)"
            )
        raise ValueError(f"暂不支持的文件类型: .{ext} (支持 pdf/docx/md)")

    log.info(f"开始解析文档: {file_path} (类型: {ext})")
    full_text, pages = parser(file_path)

    if not full_text.strip():
        raise ValueError("文档解析结果为空,可能为扫描件 PDF 或空白文档")
    if not pages:
        raise ValueError("文档解析后无有效页内容")

    log.info(f"解析完成: 共 {len(pages)} 页, {len(full_text)} 字符")
    return full_text, pages


def detect_file_type(filename: str) -> Optional[str]:
    """从文件名推断类型 (注意: .doc 不被支持,需提示用户转换)"""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in PARSERS:
        return ext
    return None
