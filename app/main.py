# app/main.py
import time
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse

from app.auth import AuthMiddleware, verify_user, create_user, load_users
from app.storage import get_user_conversations, save_user_conversations
from utils.utils import configure_logging

from ollama_client.endpoint.ollama_router import ollama_router
from transformers_client.endpoint.qwen3_router import qwen3_router

MAX_USERS = 10
configure_logging()

app = FastAPI(title="Multi-User LLM Chat API")
app.add_middleware(AuthMiddleware)

app.include_router(ollama_router)
app.include_router(qwen3_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# === Вход: только проверка, без редиректа ===
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    password = password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Логин и пароль обязательны")
    if not verify_user(username, password):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    # Устанавливаем cookie даже в JSON-ответе
    response = JSONResponse(content={"status": "success", "username": username})
    session = f"{username}:{int(time.time())}"
    response.set_cookie(
        key="session",
        value=session,
        httponly=True,
        max_age=86400,
        secure=False,  # True только для HTTPS
        samesite="lax"
    )
    return response


# === Регистрация ===
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    password = password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="Логин и пароль обязательны")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Пароль должен быть не короче 4 символов")
    if not username.replace("_", "").replace("-", "").isalnum() or len(username) < 2 or len(username) > 20:
        raise HTTPException(status_code=400, detail="Некорректный формат логина")

    users = load_users()
    if len(users) >= MAX_USERS:
        raise HTTPException(status_code=403, detail=f"Достигнут лимит в {MAX_USERS} пользователей")
    if username in users:
        raise HTTPException(status_code=400, detail="Пользователь с таким именем уже существует")

    if create_user(username, password):
        return {"status": "success", "message": "Пользователь создан"}
    raise HTTPException(status_code=500, detail="Ошибка при создании пользователя")


# === API для диалогов (опционально) ===
@app.get("/api/conversations")
async def get_conversations(request: Request):
    username = getattr(request.state, "username", None)
    if not username:
        raise HTTPException(status_code=401)
    return get_user_conversations(username)


@app.post("/api/conversations")
async def save_conversations(request: Request, data: dict):
    username = getattr(request.state, "username", None)
    if not username:
        raise HTTPException(status_code=401)
    save_user_conversations(username, data)
    return {"status": "saved"}
