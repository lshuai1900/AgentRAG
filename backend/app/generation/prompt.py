"""Prompt 模板"""

SYSTEM_PROMPT = """你是一个基于检索增强生成 (RAG) 的智能问答助手。

回答规则:
1. **仅基于**下方"参考资料"回答用户问题,不要编造未在资料中出现的信息
2. 如果参考资料不足以回答,请明确说"根据现有资料,我无法回答该问题"并说明缺什么信息
3. 回答中需要在引用到具体资料的位置标注引用编号,格式 [1] [2] 等
4. 回答要简洁、准确、有条理,使用 Markdown 格式
5. 涉及代码时使用 ``` 标注

参考资料:
{context}
"""


USER_PROMPT_TEMPLATE = """用户问题: {question}

请基于上述参考资料回答,并在引用资料处标注 [1] [2] 等编号。"""


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """构造消息列表

    Args:
        question: 用户问题
        chunks: 检索到的 chunks,格式 [{"text", "doc_id", "page", "source", ...}]

    Returns:
        messages: [{"role", "content"}]
    """
    # 构造参考资料文本
    if not chunks:
        context_text = "(无可用参考资料)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "未知")
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            parts.append(f"[{i}] 来源: {source} 第 {page} 页\n{text}")
        context_text = "\n\n".join(parts)

    system = SYSTEM_PROMPT.format(context=context_text)
    user = USER_PROMPT_TEMPLATE.format(question=question)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
