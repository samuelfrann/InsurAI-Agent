import os
import sqlite3
import bcrypt
from datetime import datetime, timezone

# 1. Define Paths (Navigate up from backend/db/ to the root folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "insurai_memory")
os.makedirs(DB_PATH, exist_ok=True)

MEMORY_DB_PATH   = os.path.join(DB_PATH, "insurai_memory.db")
SESSIONS_DB_PATH = os.path.join(DB_PATH, "insurai_sessions.db")

# 2. Establish Global Connection
conn = sqlite3.connect(SESSIONS_DB_PATH, check_same_thread=False)

# 3. Database Initialization
def init_db():
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username      TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id  TEXT PRIMARY KEY,
            username   TEXT,
            title      TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id  TEXT,
            role       TEXT,
            content    TEXT,
            timestamp  TEXT
        );
        CREATE TABLE IF NOT EXISTS fraud_assessments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id    TEXT NOT NULL,
            username     TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            form_data    TEXT NOT NULL,
            result       TEXT NOT NULL,
            risk_level   TEXT NOT NULL,
            probability  REAL NOT NULL,
            vehicle_ngn  REAL,
            vehicle_make TEXT,
            base_policy  TEXT
        );
    """)

    # Apply schema migrations safely
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN username TEXT DEFAULT 'admin'")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT 'chat'")
    except sqlite3.OperationalError:
        pass

    conn.commit()

# 4. Default Admin Seeding
def seed_default_admin(default_pw: str = "insurai123"):
    if not conn.execute("SELECT 1 FROM users").fetchone():
        hashed = bcrypt.hashpw(default_pw.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            ("admin", hashed, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        print(f"\n✅  Default admin created — username: admin  password: {default_pw}\n")