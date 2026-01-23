from contextlib import asynccontextmanager
from fastapi import FastAPI
from ollama_client.client.ollama_client import OllamaClient
from ollama_client.endpoint.ollama_settings import get_ollama_settings

@asynccontextmanager
async def ollama_lifespan(app: FastAPI):
    """
    Lifespan для Ollama-роутера.
    Инициализирует клиент при старте и сохраняет его в app.state.
    """

    settings = get_ollama_settings()
    client = OllamaClient(settings)

    if not client.connect():
        raise RuntimeError(
            f"Не удалось подключиться к Ollama по адресу {settings.ollama_url}. "
            "Убедитесь, что сервер запущен и доступен."
        )

    # Сохраняем клиент в состоянии приложения под уникальным ключом
    app.state.ollama_client = client

    yield