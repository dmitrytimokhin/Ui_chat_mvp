# app/auth.py
import hashlib
import time
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from .storage import load_users  # ← импортируем из storage

def hash_password(password: str) -> str:
    salt = "local_salt_2026_secure"
    return hashlib.sha256((salt + password).encode()).hexdigest()

def verify_user(username: str, password: str) -> bool:
    users = load_users()
    return username in users and users[username] == hash_password(password)

def create_user(username: str, password: str) -> bool:
    from .storage import save_users
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public_paths = {"/login", "/register", "/docs", "/openapi.json", "/health"}
        if request.url.path in public_paths:
            return await call_next(request)

        session = request.cookies.get("session")
        if not session or ":" not in session:
            return RedirectResponse("/login")

        username, _ = session.split(":", 1)
        request.state.username = username
        response = await call_next(request)
        return response
