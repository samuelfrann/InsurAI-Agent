import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
)


def build_vector_database():

    # ── 1. CIIN Textbooks (PDFs) ──────────────────────────────────────
    print("\n📚 Loading CIIN textbooks...")
    textbook_docs = PyPDFDirectoryLoader('./insurance_docs/ciin_books').load()
    print(f"   {len(textbook_docs)} pages found")

    for doc in textbook_docs:
        doc.metadata["source_type"] = "textbook"

    # ── 2. NAICOM Regulatory PDFs ─────────────────────────────────────
    print("\n📋 Loading NAICOM regulatory PDFs...")
    naicom_pdf_docs = PyPDFDirectoryLoader('./insurance_docs/naicom_docs').load()
    print(f"   {len(naicom_pdf_docs)} pages found")

    for doc in naicom_pdf_docs:
        doc.metadata["source_type"] = "regulatory"

   # ── 3. NIA Text files ─────────────────────────────────────────────────
    print("\n📄 Loading NIA text files...")
    import glob as _glob
    from langchain_core.documents import Document as _Doc

    naicom_txt_docs = []
    for filepath in _glob.glob('./insurance_docs/nia_txt/*.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            naicom_txt_docs.append(_Doc(
                page_content=content,
                metadata={"source": filepath, "source_type": "regulatory"}
            ))
        except Exception as e:
            print(f"  Failed: {filepath} — {e}")
    print(f"   {len(naicom_txt_docs)} files found")

    for doc in naicom_txt_docs:
        doc.metadata["source_type"] = "regulatory"

    # ── 4. Combine everything ─────────────────────────────────────────
    all_docs = textbook_docs + naicom_pdf_docs + naicom_txt_docs
    print(f"\n   Total: {len(all_docs)} documents combined")

    # ── 5. Split into chunks ──────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"   {len(chunks)} valid chunks after splitting")

    if not chunks:
        print("ERROR: No valid chunks. Are the PDFs scanned images?")
        return

    # ── 6. Embed and store ────────────────────────────────────────────
    print("\n🔄 Building embeddings — this takes a few minutes...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory='./chroma_db',
    )

    print(f"\n✅ Done. {len(chunks)} chunks stored in ChromaDB.")
    print(f"   Textbook chunks: {sum(1 for c in chunks if c.metadata.get('source_type') == 'textbook')}")
    print(f"   Regulatory chunks: {sum(1 for c in chunks if c.metadata.get('source_type') == 'regulatory')}")


if __name__ == '__main__':
    build_vector_database()