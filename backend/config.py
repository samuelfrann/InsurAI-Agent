from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

class Settings(BaseSettings):
    # API
    anthropic_api_key: str

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24h

    # App
    default_password: str = "insurai123"
    cors_origins: list[str] = [
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

    # Paths (relative to project root — no more hardcoded /home/frank/...)
    model_path: str = str(BASE_DIR / "models" / "catboost_fraud_model.cbm")
    chroma_db_path: str = str(BASE_DIR / "chroma_db")
    memory_db_path: str = str(BASE_DIR / "insurai_memory" / "insurai_memory")

    # Offline HuggingFace (set True if running without internet for embeddings)
    hf_hub_offline: bool = True
    transformers_offline: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"
settings = Settings()