import tiktoken
from typing import List
from .models import ChatMessage

# Поддерживаемые энкодеры (phi3 и qwen примерно совпадают по токенизации для русского)
ENCODER = tiktoken.get_encoding("cl100k_base")  # близок к phi/Qwen

def count_tokens(text: str) -> int:
    """Подсчитывает количество токенов в тексте."""
    return len(ENCODER.encode(text))

def truncate_history(
    history: List[ChatMessage],
    prompt: str,
    max_total_tokens: int = 30720,
    reserved_for_response: int = 512
) -> List[ChatMessage]:
    """
    Обрезает историю диалога, чтобы общее число токенов (история + промпт + ответ)
    не превышало max_total_tokens.
    """
    available_tokens = max_total_tokens - reserved_for_response - count_tokens(prompt)
    if available_tokens <= 0:
        return []

    truncated = []
    total = 0
    # Проходим с конца — сохраняем самые свежие сообщения
    for msg in reversed(history):
        msg_tokens = count_tokens(msg.text) + 3  # ~3 токена на роль ("user"/"assistant")
        if total + msg_tokens > available_tokens:
            break
        truncated.append(msg)
        total += msg_tokens

    return list(reversed(truncated))
    