from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    username: str
    success: bool


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionRenameRequest(BaseModel):
    name: str