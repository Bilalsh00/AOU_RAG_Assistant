import os

LLM_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHROMA_DIR = os.getenv("CHROMA_DIR", "aou_chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "aou_academic_assistant")
TOP_K = int(os.getenv("TOP_K", "6"))
