import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sqlite3
import bcrypt
from datetime import datetime, timezone

from backend.core.lifespan import lifespan
from backend.api import auth, chat, dashboard, fraud
from backend.db.database import init_db, seed_default_admin
from backend.config import settings

init_db()
seed_default_admin(settings.default_password)

def seed_demo_user():
    demo_user = os.getenv("DEMO_USER", "demo")
    demo_pass = settings.default_password
    if not demo_pass:
        return
    db_path = "insurai_memory/insurai_sessions.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT
            )
        """)
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (demo_user,)).fetchone()
        if not existing:
            hashed = bcrypt.hashpw(demo_pass.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (demo_user, hashed, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
    except Exception as e:
        print(f"Error seeding demo user: {e}")
    finally:
        conn.close()

seed_demo_user()

app = FastAPI(title="InsurAI Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(dashboard.router)
app.include_router(fraud.router)

_DIST = Path(__file__).parent / "frontend" / "dist"
if (_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="vite-assets")

def _spa():
    dist_index = _DIST / "index.html"
    return FileResponse(str(dist_index)) if dist_index.exists() else FileResponse("index.html")

for _path in ["/", "/login", "/chat", "/dashboard", "/analytics"]:
    app.add_api_route(_path, _spa, methods=["GET"])

@app.get("/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)