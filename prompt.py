REWRITE_PROMPT = """
You are a query rewriting assistant for the Arab Open University (AOU) Academic Assistant.
Rewrite the user's latest question as a clear standalone search query using the conversation history only when needed.
Preserve course codes, policy names, academic terms, numbers, and dates exactly.
Do not answer the question.
If the question is already standalone, return it with minimal or no changes.
Return only the rewritten query.
"""

SYSTEM_PROMPT = """
You are AOU Smart Academic Assistant, an academic information assistant for Arab Open University students.
Your answers must be grounded only in the retrieved AOU context supplied to you.

Rules:
1. Do not invent university rules, prerequisites, dates, fees, procedures, or academic requirements.
2. If the retrieved context is insufficient, clearly say that the available AOU sources do not contain enough information to answer reliably.
3. Answer in the same language as the user's question unless the user asks for another language.
4. Preserve official course codes and important academic terminology.
5. When you use retrieved information, cite it inline using the source number, for example [1] or [2].
6. If sources conflict, mention the conflict and prefer the newer or more specific source when its date/scope is clear.
7. Be concise and student-friendly. Use bullets when they improve clarity.
8. Do not claim that a source is official unless its metadata identifies it as an AOU source.

Your purpose is to help students find and understand AOU academic information, not to replace official academic advising when the source is unclear.
"""


def query_rewrite_extend(user_input: str, chat_history: list) -> str:
    history_lines = []
    for msg in chat_history or []:
        role = getattr(msg, "type", "message")
        content = getattr(msg, "content", str(msg))
        history_lines.append(f"{role}: {content}")

    history = "\n".join(history_lines[-8:])
    return f"""
Conversation History:
{history}

Latest User Question:
{user_input}

Standalone Search Query:
"""


def format_documents(documents: list) -> tuple[str, list[str]]:
    blocks = []
    sources = []

    for index, doc in enumerate(documents or [], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("url") or metadata.get("file_name") or "Unknown source"
        title = metadata.get("title") or metadata.get("file_name") or "AOU source"
        page = metadata.get("page")

        source_label = f"{title} — {source}"
        if page is not None:
            source_label += f" — page {page + 1 if isinstance(page, int) else page}"

        sources.append(source_label)
        content = getattr(doc, "page_content", str(doc))
        blocks.append(f"[Source {index}]\nTitle: {title}\nSource: {source}\nContent:\n{content}")

    return "\n\n".join(blocks), sources


def system_prompt_extend(user_input: str, chat_history: list, documents: list) -> tuple[str, list[str]]:
    context, sources = format_documents(documents)

    history_lines = []
    for msg in chat_history or []:
        role = getattr(msg, "type", "message")
        content = getattr(msg, "content", str(msg))
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines[-8:])

    prompt = f"""
User Question:
{user_input}

Recent Conversation:
{history}

Retrieved AOU Context:
{context if context else "No relevant context was retrieved."}

Answer the user's question using only the retrieved context. Use inline citations such as [1], [2].
"""
    return prompt, sources
