import json
import base64
import uuid as _uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.core.security import get_current_user
from backend.db.database import conn as _sdb
from backend.services.agent import llm          # ← fixed
from backend.services.tools import file_store   # ← fixed

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────────────────
class RenameRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    query:      str
    thread_id:  str        = "insurai_staff_default"
    file_data:  str | None = None
    file_type:  str | None = None
    file_name:  str | None = None

# ── File handling helper (was copy-pasted twice) ──────────────────────────────
def _build_message_content(request: ChatRequest, default_prompt: str = "Please analyse this.") -> str | list:
    if not (request.file_data and request.file_type):
        return request.query

    fname    = (request.file_name or "").lower()
    is_image = request.file_type.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
    is_pdf   = request.file_type == "application/pdf" or fname.endswith(".pdf")

    if is_image:
        if "/" in request.file_type and not request.file_type.endswith("octet-stream"):
            media_type = request.file_type
        elif fname.endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif fname.endswith(".png"):
            media_type = "image/png"
        else:
            media_type = "image/webp"
        return [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": request.file_data}},
            {"type": "text",  "text": request.query or default_prompt},
        ]

    if is_pdf:
        file_key = str(_uuid.uuid4())
        file_store[file_key] = {
            "bytes":    base64.b64decode(request.file_data),
            "filename": request.file_name or "document.pdf",
        }
        return (
            f"{request.query or 'Please read and summarise this document.'}\n\n"
            f"[PDF uploaded: '{request.file_name}'. "
            f"Call pdf_reader_tool with file_key='{file_key}' to read its contents.]"
        )

    # Plain text / CSV / other
    raw = base64.b64decode(request.file_data).decode("utf-8", errors="ignore")[:4000]
    return f"File '{request.file_name}':\n\n{raw}\n\nQuestion: {request.query}"


# ── Session Management ────────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(current_user: str = Depends(get_current_user)):
    rows = _sdb.execute(
        "SELECT thread_id, title, updated_at, COALESCE(session_type, 'chat') "
        "FROM sessions WHERE username = ? ORDER BY updated_at DESC",
        (current_user,),
    ).fetchall()
    return [{"thread_id": r[0], "title": r[1], "updated_at": r[2], "session_type": r[3]} for r in rows]

@router.patch("/sessions/{thread_id}")
async def rename_session(thread_id: str, body: RenameRequest, current_user: str = Depends(get_current_user)):
    _sdb.execute(
        "UPDATE sessions SET title = ? WHERE thread_id = ? AND username = ?",
        (body.title.strip()[:60], thread_id, current_user),
    )
    _sdb.commit()
    return {"ok": True}

@router.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str, current_user: str = Depends(get_current_user)):
    _sdb.execute("DELETE FROM sessions WHERE thread_id = ? AND username = ?", (thread_id, current_user))
    _sdb.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    _sdb.execute("DELETE FROM fraud_assessments WHERE thread_id = ? AND username = ?", (thread_id, current_user))
    _sdb.commit()
    return {"ok": True}

@router.get("/sessions/{thread_id}/history")
async def get_session_history(thread_id: str, current_user: str = Depends(get_current_user)):
    rows = _sdb.execute(
        "SELECT role, content, timestamp FROM messages WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


# ── Shared streaming logic ────────────────────────────────────────────────────
_STATUS_LABELS = {
    "search":                  "Searching the web",
    "wikipedia":               "Looking up Wikipedia",
    "insurance_policy_lookup": "Checking policy documents",
    "fraud_detection_tool":    "Running fraud model",
    "pdf_reader_tool":         "Reading uploaded document",
}

async def _stream(bot, request: ChatRequest, current_user: str, session_type: str = "chat", default_prompt: str = "Please analyse this."):
    now = datetime.now(timezone.utc).isoformat()
    _sdb.execute(
        "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (request.thread_id, "user", request.query, now),
    )
    _sdb.commit()

    async def generate():
        full_response = ""
        try:
            msg_content = _build_message_content(request, default_prompt)
            messages    = [HumanMessage(content=msg_content)]
            config      = {"configurable": {"thread_id": request.thread_id}}

            async for event in bot.graph.astream_events({"messages": messages}, config):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                if kind == "on_tool_start":
                    label = _STATUS_LABELS.get(event.get("name", ""), f"Using {event.get('name', '')}")
                    yield f"data: {json.dumps({'status': label})}\n\n"
                elif kind == "on_chat_model_stream" and node == "claude_summarizer":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'token': content})}\n\n"
        except Exception as e:
            import traceback; traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            ai_now = datetime.now(timezone.utc).isoformat()
            _sdb.execute(
                "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (request.thread_id, "ai", full_response, ai_now),
            )
            exists = _sdb.execute("SELECT 1 FROM sessions WHERE thread_id = ?", (request.thread_id,)).fetchone()
            if exists:
                _sdb.execute("UPDATE sessions SET updated_at = ? WHERE thread_id = ?", (ai_now, request.thread_id))
            else:
                # Auto-generate title for brand-new sessions
                if session_type == "chat":
                    try:
                        title_msg = await llm.ainvoke([HumanMessage(content=(
                            f"Write a short 3-6 word title for a conversation that starts with this message. "
                            f"Return ONLY the title, no punctuation, no quotes:\n\n{request.query[:200]}"
                        ))])
                        title = title_msg.content.strip().strip('"').strip("'")[:60]
                    except Exception:
                        title = request.query[:55] + ("…" if len(request.query) > 55 else "")
                else:
                    title = "New Assessment"
                _sdb.execute(
                    "INSERT INTO sessions (thread_id, username, title, session_type, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (request.thread_id, current_user, title, session_type, now, ai_now),
                )
            _sdb.commit()
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, fastapi_req: Request, current_user: str = Depends(get_current_user)):
    return await _stream(fastapi_req.app.state.bot, request, current_user, session_type="chat")

@router.post("/fraud/chat/stream")
async def fraud_chat_stream(request: ChatRequest, fastapi_req: Request, current_user: str = Depends(get_current_user)):
    return await _stream(
        fastapi_req.app.state.fraud_chat_bot, request, current_user,
        session_type="fraud",
        default_prompt="Please analyse this image and extract any claim fields.",
    )

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, fastapi_req: Request, current_user: str = Depends(get_current_user)):
    """Non-streaming fallback. Kept for compatibility."""
    try:
        bot          = fastapi_req.app.state.bot
        messages     = [HumanMessage(content=request.query)]
        config       = {"configurable": {"thread_id": request.thread_id}}  # ← bug fixed
        full_response = ""
        async for event in bot.graph.astream_events({"messages": messages}, config):
            if (event["event"] == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "claude_summarizer"):
                content = event["data"]["chunk"].content
                if content:
                    full_response += content
        return {"response": full_response or "I wasn't able to generate a response. Please try again."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))