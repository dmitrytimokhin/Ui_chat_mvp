# endpoint/qwen3_lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers_client.client.qwen3_client import Qwen3Client
from transformers_client.endpoint.qwen3_settings import get_qwen3_settings


@asynccontextmanager
async def qwen3_lifespan(app: FastAPI):
    """
    Lifespan для Qwen3-роутера.
    Инициализирует клиент при старте и сохраняет его в app.state.
    """

    settings = get_qwen3_settings()
    client = Qwen3Client(settings)
    if not client.connect():
        raise RuntimeError("Не удалось загрузить Qwen3")

    # Сохраняем клиент в состоянии приложения под уникальным ключом
    app.state.qwen3_client = client

    yield