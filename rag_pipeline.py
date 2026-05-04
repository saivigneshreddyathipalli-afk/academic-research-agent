import os
import uuid
from pathlib import Path

import chromadb
import numpy as np
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ResearchRAG:
    """Local RAG pipeline for PDF reference documents using Chroma."""

    def __init__(self, collection_name: str = "research_docs"):
        self.collection_name = collection_name
        self.db_path = str(Path(__file__).parent / "chroma_db")
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self.embedder.passage_embed(texts))
        return [e.tolist() for e in embeddings]

    def _embed_query(self, query: str) -> list[float]:
        embeddings = list(self.embedder.query_embed(query))
        return embeddings[0].tolist()

    def ingest_pdf(self, pdf_path: str) -> int:
        """Extract, chunk, embed, and store a PDF. Returns chunk count."""
        docs = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        embeddings = self._embed_texts(texts)
        ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in texts]
        metadatas = [
            {"source": c.metadata.get("source", ""), "page": c.metadata.get("page", 0)}
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def ingest_pdf_bytes(self, pdf_bytes: bytes) -> int:
        """Ingest a PDF from in-memory bytes. Returns chunk count."""
        tmp_path = Path(__file__).parent / "_temp_upload.pdf"
        tmp_path.write_bytes(pdf_bytes)
        try:
            return self.ingest_pdf(str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def query(self, query: str, top_k: int = 5) -> str:
        """Query the vector store and return formatted results."""
        query_embedding = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            page = meta.get("page", "?")
            source = meta.get("source", "uploaded PDF")
            output.append(
                f"[RAG Result {i+1}] (Page {page}, distance: {dist:.3f})\n"
                f"Source: {source}\n"
                f"Content: {doc[:400]}...\n"
            )
        return "\n---\n".join(output) if output else "[No matching documents found]"

    def is_ready(self) -> bool:
        """Check if the collection has any documents."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def clear(self):
        """Delete all documents from the collection."""
        try:
            ids = self.collection.get()["ids"]
            if ids:
                self.collection.delete(ids=ids)
        except Exception:
            pass
