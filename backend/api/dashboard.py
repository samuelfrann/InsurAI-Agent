from fastapi import APIRouter, Depends
from datetime import date, timedelta, datetime, timezone
import re
import httpx
import os

from backend.db.database import conn as _sdb
from backend.core.security import get_current_user

router = APIRouter()

# ── FX rate cache ──
_fx_cache = {"rate": None, "fetched_at": None}
FX_CACHE_TTL = timedelta(hours=6)

async def _fetch_live_ngn_rate() -> float | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("https://api.exchangerate-api.com/v4/latest/USD")
            r.raise_for_status()
            return float(r.json()["rates"]["NGN"])
    except Exception as e:
        print(f"⚠️ FX live fetch failed: {e}")
        return None

@router.get('/config/fx-rate')
async def get_fx_rate(current_user: str = Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    if (_fx_cache["rate"] and _fx_cache["fetched_at"] and (now - _fx_cache["fetched_at"]) < FX_CACHE_TTL):
        return {"ngn_per_usd": _fx_cache["rate"], "source": "cache"}

    live = await _fetch_live_ngn_rate()
    if live:
        _fx_cache["rate"] = live
        _fx_cache["fetched_at"] = now
        return {"ngn_per_usd": live, "source": "live"}

    return {"ngn_per_usd": float(os.getenv("NGN_USD_RATE", "1600")), "source": "fallback"}

@router.get('/dashboard/stats')
async def dashboard_stats(current_user: str = Depends(get_current_user)):
    total_sessions = _sdb.execute("SELECT COUNT(*) FROM sessions WHERE username = ?", (current_user,)).fetchone()[0]

    fraud_msgs = _sdb.execute("""
        SELECT m.content FROM messages m
        INNER JOIN sessions s ON m.thread_id = s.thread_id
        WHERE s.username = ? AND m.role = 'ai'
          AND m.content LIKE '%Fraud Probability%' AND m.content LIKE '%Risk Level%'
    """, (current_user,)).fetchall()

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW-MEDIUM": 0, "LOW": 0}
    for (content,) in fraud_msgs:
        m = re.search(r'Risk Level[^:]*:\s*([\w\s-]+)', content, re.IGNORECASE)
        if m:
            r = m.group(1).strip().split('\n')[0].replace('*', '').replace('_', '').strip().upper()
            if 'HIGH' in r and 'LOW' not in r: risk_counts['HIGH'] += 1
            elif 'LOW' in r and 'MEDIUM' in r: risk_counts['LOW-MEDIUM'] += 1
            elif 'MEDIUM' in r: risk_counts['MEDIUM'] += 1
            else: risk_counts['LOW'] += 1

    recent = _sdb.execute(
        "SELECT thread_id, title, updated_at FROM sessions WHERE username = ? AND (session_type = 'chat' OR session_type IS NULL) ORDER BY updated_at DESC LIMIT 8",
        (current_user,)
    ).fetchall()

    return {
        "total_sessions": total_sessions,
        "total_fraud_runs": len(fraud_msgs),
        "flagged": risk_counts['HIGH'] + risk_counts['MEDIUM'],
        "clear": risk_counts['LOW-MEDIUM'] + risk_counts['LOW'],
        "risk_breakdown": risk_counts,
        "recent_sessions": [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in recent]
    }

@router.get('/analytics/data')
async def analytics_data(current_user: str = Depends(get_current_user)):
    rows = _sdb.execute("""
        SELECT date(created_at) AS day, COUNT(*) AS cnt
        FROM sessions WHERE username = ? AND date(created_at) >= date('now', '-13 days')
        GROUP BY day ORDER BY day
    """, (current_user,)).fetchall()
    sessions_map = {r[0]: r[1] for r in rows}

    rows2 = _sdb.execute("""
        SELECT date(m.timestamp) AS day, COUNT(*) AS cnt
        FROM messages m
        INNER JOIN sessions s ON m.thread_id = s.thread_id
        WHERE s.username = ? AND m.role = 'ai' AND m.content LIKE '%Fraud Probability%' AND date(m.timestamp) >= date('now', '-13 days')
        GROUP BY day ORDER BY day
    """, (current_user,)).fetchall()
    fraud_map = {r[0]: r[1] for r in rows2}

    today = date.today()
    labels, sessions_data, fraud_data = [], [], []
    for i in range(13, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        labels.append(d[5:])
        sessions_data.append(sessions_map.get(d, 0))
        fraud_data.append(fraud_map.get(d, 0))

    fraud_msgs = _sdb.execute("""
        SELECT m.content FROM messages m
        INNER JOIN sessions s ON m.thread_id = s.thread_id
        WHERE s.username = ? AND m.role = 'ai' AND m.content LIKE '%Fraud Probability%' AND m.content LIKE '%Risk Level%'
    """, (current_user,)).fetchall()

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW-MEDIUM": 0, "LOW": 0}
    for (content,) in fraud_msgs:
        m = re.search(r'Risk Level[^:]*:\s*([\w\s-]+)', content, re.IGNORECASE)
        if m:
            r = m.group(1).strip().split('\n')[0].replace('*', '').replace('_', '').strip().upper()
            if 'HIGH' in r and 'LOW' not in r: risk_counts['HIGH'] += 1
            elif 'LOW' in r and 'MEDIUM' in r: risk_counts['LOW-MEDIUM'] += 1
            elif 'MEDIUM' in r: risk_counts['MEDIUM'] += 1
            else: risk_counts['LOW'] += 1

    return {
        "labels": labels, "sessions_by_day": sessions_data, "fraud_by_day": fraud_data, "risk_distribution": risk_counts
    }