import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.core.lifespan import lifespan
from backend.api import auth, chat, dashboard, fraud
from backend.db.database import init_db, seed_default_admin
from backend.config import settings

init_db()
seed_default_admin(settings.default_password)

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

# SPA fallback for React frontend
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