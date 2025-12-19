import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse
from .llm_ollama import query_ollama
from .llm_qwen import query_qwen, init_models
from .utils import configure_logging, cleanup_memory, EngineError

# Настройка логгера через utils
configure_logging()
logger = logging.getLogger(__name__)


# Event при старте приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения.
    
    На старте: инициализирует все модели.
    На завершение: логирует остановку.
    """
    # Startup: предварительная очистка кэша и инициализация моделей
    logger.info("🚀 Приложение стартует... Очищаем кэш и инициализируем модели...")
    try:
        cleanup_memory()
    except Exception:
        logger.warning("Не удалось очистить кэш перед стартом, продолжаем")
    init_models()
    logger.info("✅ Приложение полностью готово к работе!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Приложение завершает работу...")


app = FastAPI(
    title="Hybrid LLM Gateway (Ollama + Qwen)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit по умолчанию
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Получен запрос к модели: {request.model_alias}")
    try:
        if request.model_alias == "ollama":
            # model variant can be provided in request.ollama_model
            model_name = request.ollama_model or "phi"
            # run blocking I/O in threadpool
            response_text = await asyncio.to_thread(
                query_ollama,
                request.prompt,
                request.history,
                request.temperature,
                request.max_tokens,
                model_name,
            )
        elif request.model_alias == "qwen_transformers" or request.model_alias == "Qwen3":
            response_text = await asyncio.to_thread(
                query_qwen,
                request.prompt,
                request.history,
                request.temperature,
                request.max_tokens,
            )
        else:
            raise ValueError("Неподдерживаемая модель")

        return ChatResponse(response=response_text)

    except EngineError as e:
        # Предсказуемые ошибки от адаптеров — помечаем как пользовательские ошибки
        error_msg = str(e)
        logger.warning(f"Движок вернул ошибку: {error_msg}")
        return ChatResponse(response="", error=error_msg)

    except Exception as e:
        error_msg = str(e)
        logger.exception("Непредвиденная ошибка в /chat")
        return ChatResponse(response="", error=error_msg)
        