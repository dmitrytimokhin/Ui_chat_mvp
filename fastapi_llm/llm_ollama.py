import requests
import logging
from typing import List
from .models import ChatMessage

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

_SYSTEM_PROMPT = (
    "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
    "Если вопрос неясен — уточни. Не выдумывай факты."
)

def query_ollama(
    prompt: str,
    history: List[ChatMessage],
    temperature: float,
    max_tokens: int
) -> str:
    """
    Отправляет запрос к локальному Ollama (модель phi3).
    Автоматически обрезает историю при превышении лимита контекста.
    """
    from .token_utils import truncate_history

    # Ограничиваем историю (phi3 имеет контекст ~4096 токенов)
    safe_history = truncate_history(
        history=history,
        prompt=prompt,
        max_total_tokens=3584,  # оставляем запас
        reserved_for_response=max_tokens
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in safe_history:
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.text
        })
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "phi3",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        logger.debug("Отправка запроса в Ollama...")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        logger.debug("Получен ответ от Ollama")
        return content
    except Exception as e:
        logger.error(f"Ошибка при вызове Ollama: {e}")
        raise RuntimeError(f"Ollama недоступна: {e}")
        