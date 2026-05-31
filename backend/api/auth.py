from fastapi import APIRouter, Depends, HTTPException

from backend.core.security import verify_password, create_access_token, get_current_user
from backend.db.database import conn as _sdb
from backend.models.schemas import LoginRequest, LoginResponse

router = APIRouter(tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    row = _sdb.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (body.username.strip(),),
    ).fetchone()
    if not row or not verify_password(body.password, row[0]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(body.username)
    return LoginResponse(access_token=token, username=body.username, success=True)


@router.get("/auth/me")
async def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


@router.post("/logout")
async def logout():
    return {"success": True}