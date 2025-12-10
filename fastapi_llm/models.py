from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" или "assistant"
    text: str

class ChatRequest(BaseModel):
    prompt: str
    history: List[ChatMessage]
    mode: str  # "yandex" или "local"
    model: Optional[str] = "gpt_pro"  # только для Yandex
    temperature: float = 0.0
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    