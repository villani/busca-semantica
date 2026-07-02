from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONNECTION = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
DEFAULT_COLLECTION = "documents"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
TOP_K = 10


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def require_google_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada. Crie um arquivo .env a partir de .env.example."
        )


def get_connection_string() -> str:
    return os.getenv("POSTGRES_CONNECTION", DEFAULT_CONNECTION)


def get_collection_name() -> str:
    return os.getenv("PGVECTOR_COLLECTION", DEFAULT_COLLECTION)


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    require_google_api_key()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(model=model)


def create_vector_store() -> PGVector:
    return PGVector(
        embeddings=create_embeddings(),
        collection_name=get_collection_name(),
        connection=get_connection_string(),
        use_jsonb=True,
    )


def search_documents(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    vector_store = create_vector_store()
    return vector_store.similarity_search_with_score(query, k=k)


def format_context(results: Iterable[tuple[Document, float]]) -> str:
    context_parts: list[str] = []

    for index, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source", "document.pdf")
        page = document.metadata.get("page")
        page_label = f", página {page + 1}" if isinstance(page, int) else ""
        content = document.page_content.strip()

        if content:
            context_parts.append(
                f"[Trecho {index} | fonte: {source}{page_label} | score: {score}]\n{content}"
            )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    load_environment()
    question = input("Pergunta: ").strip()

    if not question:
        raise SystemExit("Informe uma pergunta para buscar no banco vetorial.")

    matches = search_documents(question, k=TOP_K)
    print(format_context(matches))
