"""
Клиент для взаимодействия с локальным Ollama-сервером.
Отправляет запросы к модели phi3, развернутой на хосте.
Учитывает лимит контекста через token_utils.
"""

import os
import logging
import requests
from typing import List, Dict
from requests.exceptions import RequestException, Timeout, ConnectionError

from .models import ChatMessage
from .utils import truncate_and_build_messages, log_request_start, EngineError

# === Константы ===
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
REQUEST_TIMEOUT_SECONDS: int = 120
MODEL_NAME: str = "phi3"
# phi3 поддерживает до ~4096 токенов
MAX_CONTEXT_LENGTH: int = 4096
RESERVED_TOKENS_FOR_RESPONSE: int = 512

logger = logging.getLogger(__name__)


def query_ollama(
    prompt: str,
    history: List[ChatMessage],
    temperature: float,
    max_tokens: int,
    model_name: str = None,
) -> str:
    """
    Отправляет запрос к локальному Ollama-серверу и возвращает сгенерированный ответ.
    Автоматически обрезает историю, чтобы уложиться в лимит контекста phi3.
    """
    engine_name = f"Ollama/{model_name or MODEL_NAME}"
    log_request_start(engine_name, temperature, max_tokens)

    # === ОБРЕЗКА ИСТОРИИ ПО ТОКЕНАМ И ПОДГОТОВКА СООБЩЕНИЙ ===
    messages, safe_history = truncate_and_build_messages(
        prompt=prompt,
        history=history,
        max_total_tokens=MAX_CONTEXT_LENGTH - RESERVED_TOKENS_FOR_RESPONSE,
        reserved_for_response=max_tokens,
    )
    payload = {
        "model": model_name or MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        logger.debug(f"Отправка POST-запроса к {OLLAMA_URL}")
        response = requests.post(
            url=OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        response_data = response.json()
        content = response_data.get("message", {}).get("content", "").strip()

        if not content:
            raise ValueError("Пустой ответ от Ollama")

        logger.info(f"Успешно получен ответ от Ollama (длина: {len(content)} символов)")
        return content

    except ConnectionError as e:
        error_msg = "Не удалось подключиться к Ollama (ConnectionError)"
        logger.error(f"{error_msg}: {e}")
        raise EngineError(f"Ollama недоступна: {error_msg}") from e

    except Timeout as e:
        error_msg = "Таймаут при обращении к Ollama"
        logger.error(f"{error_msg}: {e}")
        raise EngineError(f"Ollama не ответила за {REQUEST_TIMEOUT_SECONDS} секунд") from e

    except RequestException as e:
        status = e.response.status_code if e.response else "unknown"
        error_msg = f"Ошибка HTTP-запроса к Ollama: {status}"
        logger.error(f"{error_msg}: {e}")
        raise EngineError(f"Ошибка связи с Ollama: {error_msg}") from e

    except (ValueError, KeyError, TypeError) as e:
        error_msg = "Некорректный формат ответа от Ollama"
        logger.error(f"{error_msg}: {e}")
        raise EngineError(f"{error_msg}: {e}") from e

    except Exception as e:
        error_msg = "Неожиданная ошибка при работе с Ollama"
        logger.error(f"{error_msg}: {e}")
        raise EngineError(f"{error_msg}: {e}") from e
        