from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.config import settings
from backend.services.agent import Agent, llm, MAIN_PROMPT, FRAUD_CHAT_PROMPT
from backend.services.rag import get_rag_tool
from backend.services.tools import search_tool, wiki_tool, pdf_reader_tool
from backend.services.fraud_model import fraud_detection_tool, get_fraud_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load ML model at startup so the first request isn't slow
    get_fraud_model()

    # Build RAG tool (loads embeddings + Chroma)
    ciin_tool = get_rag_tool()

    main_tools = [search_tool, wiki_tool, ciin_tool, fraud_detection_tool, pdf_reader_tool]
    fraud_chat_tools = [search_tool, wiki_tool, pdf_reader_tool]

    async with AsyncSqliteSaver.from_conn_string(settings.memory_db_path) as memory:
        app.state.bot = Agent(llm, llm, main_tools, checkpointer=memory, system=MAIN_PROMPT)
        app.state.fraud_chat_bot = Agent(
            llm, llm, fraud_chat_tools, checkpointer=memory, system=FRAUD_CHAT_PROMPT
        )
        yield