import os
from functools import lru_cache
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from backend.config import settings

if settings.hf_hub_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def _build_hybrid_retriever():
    vector_store = get_vector_store()
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    raw = vector_store.get()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"📚 BM25 index built on {len(docs)} chunks.")

    bm25 = BM25Retriever.from_documents(docs, k=3)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.6, 0.4],
    )


@lru_cache(maxsize=1)
def _get_rewrite_llm():
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=80)


class _SmartRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            llm = _get_rewrite_llm()
            response = llm.invoke(
                f"Rewrite as a precise insurance textbook search query using "
                f"technical insurance terminology. Return ONLY the rewritten "
                f"query, nothing else.\n\nOriginal: {query}\nRewritten:"
            )
            rewritten = response.content.strip()
            print(f"🔍 '{query[:60]}' -> '{rewritten}'")
        except Exception:
            rewritten = query

        return _build_hybrid_retriever().invoke(rewritten)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


@lru_cache(maxsize=1)
def get_rag_tool():
    _build_hybrid_retriever()
    return create_retriever_tool(
        _SmartRetriever(),
        name="insurance_policy_lookup",
        description=(
            "Search this tool to find official definitions, principles, and rules "
            "from foundational CIIN insurance textbooks. Always use this before "
            "answering claims or underwriting questions."
        ),
    )