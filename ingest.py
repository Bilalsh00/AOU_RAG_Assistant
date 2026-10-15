import argparse
import shutil
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
SOURCES_FILE = DATA_DIR / "aou_sources.txt"


def load_web_pages() -> list[Document]:
    documents = []
    if not SOURCES_FILE.exists():
        return documents

    urls = [
        line.strip()
        for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    headers = {"User-Agent": "Mozilla/5.0 AOU-Academic-Assistant/1.0"}

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title = soup.title.get_text(" ", strip=True) if soup.title else url
            main = soup.find("main") or soup.find(id="DeltaPlaceHolderMain") or soup.body or soup
            text = main.get_text("\n", strip=True)

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url, "title": title, "type": "web"},
                    )
                )
                print(f"Loaded web page: {url}")
        except Exception as exc:
            print(f"Skipped {url}: {exc}")

    return documents


def load_local_files() -> list[Document]:
    documents = []
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                reader = PdfReader(str(path))
                for page_number, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": str(path),
                                    "file_name": path.name,
                                    "title": path.stem,
                                    "page": page_number,
                                    "type": "pdf",
                                },
                            )
                        )
                print(f"Loaded PDF: {path.name}")

            elif suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "file_name": path.name,
                                "title": path.stem,
                                "type": "text",
                            },
                        )
                    )
                print(f"Loaded text file: {path.name}")
        except Exception as exc:
            print(f"Skipped {path}: {exc}")

    return documents


def build_database(reset: bool = False):
    if reset and Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"Removed old database: {CHROMA_DIR}")

    raw_documents = load_web_pages() + load_local_files()

    if not raw_documents:
        raise RuntimeError(
            "No documents were loaded. Add URLs to data/aou_sources.txt or PDFs/TXT files to data/documents/."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "، ", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. Loaded {len(raw_documents)} documents and stored {len(chunks)} chunks in {CHROMA_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the AOU RAG knowledge base.")
    parser.add_argument("--reset", action="store_true", help="Delete the old Chroma database before rebuilding.")
    args = parser.parse_args()
    build_database(reset=args.reset)
