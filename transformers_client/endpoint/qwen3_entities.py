from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    """
    Представление одного сообщения в диалоге.
    """
    role: str = Field(..., description="Должно быть 'user' или 'assistant'")
    text: str


class ChatRequest(BaseModel):
    """
    Запрос к endpoint.
    """
    prompt: str
    history: List[ChatMessage]
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Температура модели"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Максимальное количество токенов на ответ"
    )


class ChatResponse(BaseModel):
    """
    Ответ от LLM модели
    """
    response: str
    error: Optional[str] = None