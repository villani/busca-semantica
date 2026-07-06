from __future__ import annotations

import os
import time
from pathlib import Path
from collections.abc import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import (
    PROJECT_ROOT,
    create_embeddings,
    get_collection_name,
    get_connection_string,
    load_environment,
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_BATCH_SLEEP_SECONDS = 65


def get_pdf_path() -> Path:
    configured_path = os.getenv("PDF_PATH", "document.pdf")
    pdf_path = Path(configured_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado em {pdf_path}. Coloque document.pdf na raiz do projeto "
            "ou configure PDF_PATH no .env."
        )

    return pdf_path


def load_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_embedding_batch_size() -> int:
    return int(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE))


def get_embedding_batch_sleep_seconds() -> int:
    return int(
        os.getenv(
            "EMBEDDING_BATCH_SLEEP_SECONDS",
            DEFAULT_EMBEDDING_BATCH_SLEEP_SECONDS,
        )
    )


def batches(documents: Sequence[Document], batch_size: int):
    for start in range(0, len(documents), batch_size):
        yield start, documents[start : start + batch_size]


def persist_chunks(chunks: list[Document]) -> None:
    batch_size = get_embedding_batch_size()
    sleep_seconds = get_embedding_batch_sleep_seconds()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    vector_store: PGVector | None = None

    print(f"Persistindo em {total_batches} lote(s) de até {batch_size} chunk(s).")

    for batch_index, (start, batch) in enumerate(batches(chunks, batch_size), start=1):
        print(
            f"Gerando embeddings do lote {batch_index}/{total_batches} "
            f"({start + 1}-{start + len(batch)} de {len(chunks)})."
        )

        if vector_store is None:
            vector_store = PGVector.from_documents(
                documents=batch,
                embedding=create_embeddings(),
                collection_name=get_collection_name(),
                connection=get_connection_string(),
                pre_delete_collection=True,
                use_jsonb=True,
            )
        else:
            vector_store.add_documents(batch)

        if batch_index < total_batches and sleep_seconds > 0:
            print(f"Aguardando {sleep_seconds}s para respeitar limites da API.")
            time.sleep(sleep_seconds)


def ingest() -> None:
    load_environment()
    pdf_path = get_pdf_path()

    print(f"Carregando PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"Páginas carregadas: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks gerados: {len(chunks)}")

    if not chunks:
        raise RuntimeError("Nenhum texto foi extraído do PDF.")

    persist_chunks(chunks)

    print(f"Ingestão concluída na collection '{get_collection_name()}'.")


if __name__ == "__main__":
    ingest()
