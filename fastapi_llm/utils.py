"""
Утилиты для LLM-адаптеров: токенизация, обрезка истории,
логирование и общие исключения.

Этот модуль объединяет ранее разнесённую логику из
`llm_common.py` и `token_utils.py` для удобства и единообразия.
"""
from typing import List, Dict, Tuple, Optional
import logging
import os

from .models import ChatMessage

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency at import-time
    tiktoken = None

logger = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """Базовое исключение, которое должны бросать адаптеры движков.

    Используется для единообразной обработки ошибок в `main`.
    """


SYSTEM_PROMPT: str = (
    "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
    "Если вопрос неясен — уточни. Не выдумывай фактов."
)


def configure_logging(level: int = logging.INFO) -> None:
    """Настраивает корневой логгер приложения.

    Вызывается при старте приложения (в `main`).
    """
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=level, format=fmt)


def count_tokens(text: str) -> int:
    """Подсчитывает токены текста с использованием `tiktoken`.

    Если `tiktoken` недоступен — возвращает оценку по количеству символов.
    """
    if tiktoken is None:
        # Падает обратно на приближённую метрику для безопасной работы
        return max(1, len(text) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def truncate_history(
    history: List[ChatMessage],
    prompt: str,
    max_total_tokens: int = 30720,
    reserved_for_response: int = 512,
) -> List[ChatMessage]:
    """Обрезает историю так, чтобы вместиться в лимит токенов.

    Сохраняются самые свежие сообщения.
    """
    available_tokens = max_total_tokens - reserved_for_response - count_tokens(prompt)
    if available_tokens <= 0:
        return []

    truncated: List[ChatMessage] = []
    total = 0
    # Идём с конца — сохраняем новые сообщения
    for msg in reversed(history):
        msg_tokens = count_tokens(msg.text) + 3
        if total + msg_tokens > available_tokens:
            break
        truncated.append(msg)
        total += msg_tokens

    return list(reversed(truncated))


def truncate_and_build_messages(
    prompt: str,
    history: List[ChatMessage],
    max_total_tokens: int,
    reserved_for_response: int,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[ChatMessage]]:
    """Обрезает историю и формирует список сообщений для send-пейлоада.

    Возвращает кортеж `(messages, safe_history)`.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    safe_history = truncate_history(
        history=history,
        prompt=prompt,
        max_total_tokens=max_total_tokens,
        reserved_for_response=reserved_for_response,
    )

    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})

    for msg in safe_history:
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.text})

    messages.append({"role": "user", "content": prompt})

    logger.debug(f"Prepared messages (count={len(messages)}) after truncation")
    return messages, safe_history


def log_request_start(engine_name: str, temperature: float, max_tokens: int) -> None:
    logger.info(f"Запрос к движку: {engine_name}")
    logger.debug(f"Параметры генерации: temperature={temperature}, max_tokens={max_tokens}")
