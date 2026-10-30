"""
Knowledge Base Ingestion Script.

Usage:
    python scripts/ingest_knowledge_base.py

This script:
1. Reads the ML textbook PDF from backend/knowledge_base/source_documents/
2. Extracts text page by page using PyMuPDF
3. Chunks the text into overlapping segments
4. Generates sentence-transformer embeddings
5. Saves FAISS / NumPy vector index to data/vector_store/
"""
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.utils.pdf_parser import PDFParser
from backend.app.rag.ingestion import DocumentChunker
from backend.app.rag.embeddings import EmbeddingGenerator
from backend.app.rag.vector_store import VectorStore


def ingest():
    print("=" * 60)
    print("  AI Interviewer - Knowledge Base Ingestion")
    print("=" * 60)

    pdf_path = settings.KNOWLEDGE_BASE_PDF
    if not os.path.exists(pdf_path):
        print(f"\n[ERROR] PDF not found at: {pdf_path}")
        print("Please place the ML textbook PDF at that path and retry.")
        sys.exit(1)

    print(f"\n[1/4] Extracting text from: {os.path.basename(pdf_path)}")
    t0 = time.time()
    pages = PDFParser.extract_text_with_pages(pdf_path)
    print(f"      Extracted {len(pages)} pages in {time.time()-t0:.1f}s")

    print(f"\n[2/4] Chunking text (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
    chunker = DocumentChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = chunker.chunk_pages(pages, source_doc=os.path.basename(pdf_path))
    print(f"      Generated {len(chunks)} chunks")

    print(f"\n[3/4] Generating embeddings (model: {settings.EMBEDDING_MODEL})")
    t1 = time.time()
    embedder = EmbeddingGenerator(settings.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)
    print(f"      Embedded {len(vectors)} vectors of dim {vectors.shape[1]} in {time.time()-t1:.1f}s")

    print(f"\n[4/4] Building and saving vector index to: {settings.VECTOR_STORE_PATH}")
    vector_store = VectorStore(settings.VECTOR_STORE_PATH)
    vector_store.build_index(chunks, vectors)
    vector_store.save()

    print("\n" + "=" * 60)
    print(f"  Ingestion complete! {len(chunks)} chunks indexed.")
    print("  Run the backend server to start using the RAG pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
