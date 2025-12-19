from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    """Представление одного сообщения в диалоге."""
    role: str = Field(..., description="'user' или 'assistant'")
    text: str

class ChatRequest(BaseModel):
    """Запрос к /chat эндпоинту."""
    prompt: str
    history: List[ChatMessage] = Field(default_factory=list)
    model_alias: str = Field(..., description="'ollama' или 'qwen_transformers'")
    # Для варианта 'ollama' можно указать конкретную модель, например 'phi', 'qwen_lite', 'qwen_pro'
    ollama_model: Optional[str] = Field(None, description="вариант модели для Ollama (phi, qwen_lite, qwen_pro)")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=4096)

class ChatResponse(BaseModel):
    """Ответ от LLM-движка."""
    response: str
    error: Optional[str] = None
