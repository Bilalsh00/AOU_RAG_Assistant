from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TOP_K
from models import State
from prompt import REWRITE_PROMPT, SYSTEM_PROMPT, query_rewrite_extend, system_prompt_extend


llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
    )


def rewritten_query_agent(state: State) -> dict:
    user_input = state.get("user_input", "")
    chat_history = state.get("chat_history", [])

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=query_rewrite_extend(user_input, chat_history)),
    ]

    response = llm.invoke(messages)
    return {"rewritten_query": response.content.strip()}


def retriever_agent(state: State) -> dict:
    rewritten_query = state.get("rewritten_query") or state.get("user_input", "")
    retriever = get_vector_db().as_retriever(search_kwargs={"k": TOP_K})
    documents = retriever.invoke(rewritten_query)
    return {"context": documents}


def response_agent(state: State) -> dict:
    user_input = state.get("user_input", "")
    chat_history = state.get("chat_history", [])
    documents = state.get("context", []) or []

    user_prompt, sources = system_prompt_extend(user_input, chat_history, documents)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return {
        "response": response.content.strip(),
        "sources": sources,
    }
