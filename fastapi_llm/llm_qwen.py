# fastapi_llm/llm_qwen.py
import logging
from pathlib import Path
from llama_cpp import Llama

from .models import ChatMessage

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/qwen3-1.7b-Q4_K_M.gguf")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

_qwen_llm = None

def _get_qwen():
    global _qwen_llm
    if _qwen_llm is None:
        logger.info("Загрузка Qwen3-1.7B (GGUF, Q4) через llama.cpp с Metal...")
        _qwen_llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=32768,
            n_threads=6,          # M1: 6–8 потоков
            n_gpu_layers=1,       # Важно! Даже 1 слой включает Metal
            verbose=False,
            chat_format="qwen"
        )
        logger.info("✅ Qwen3-1.7B загружена (Metal ускорение)")
    return _qwen_llm

def query_qwen(prompt: str, history: List[ChatMessage], temperature: float, max_tokens: int) -> str:
    from .token_utils import truncate_history

    safe_history = truncate_history(history, prompt, max_total_tokens=30720, reserved_for_response=max_tokens)

    messages = [{"role": "system", "content": (
        "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
        "Если вопрос неясен — уточни. Не выдумывай факты."
    )}]
    for msg in safe_history:
        messages.append({"role": "user" if msg.role == "user" else "assistant", "content": msg.text})
    messages.append({"role": "user", "content": prompt})

    try:
        response = _get_qwen().create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ошибка в Qwen3-1.7B (GGUF): {e}")
        raise RuntimeError(f"Ошибка Qwen: {e}")