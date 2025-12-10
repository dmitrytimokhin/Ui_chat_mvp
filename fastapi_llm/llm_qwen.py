# fastapi_llm/llm_qwen.py
import logging
import torch
from typing import List, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .models import ChatMessage

logger = logging.getLogger(__name__)

_QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"
_tokenizer = None
_model = None

def _load_qwen() -> Tuple:
    """Ленивая загрузка Qwen3-1.7B с оптимизацией под MPS (M1/M2)."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return _tokenizer, _model

    logger.info("Загрузка Qwen3-1.7B (1.7B, float16) через transformers + accelerate...")

    # На M1: MPS поддерживает float16, но не int8/4 (bitsandbytes не работает)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Используем устройство: {device}")

    # Загрузка токенизатора
    _tokenizer = AutoTokenizer.from_pretrained(
        _QWEN_MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # === ЗАГРУЗКА МОДЕЛИ С ОПТИМИЗАЦИЕЙ ===
    try:
        # Вариант 1: Прямая загрузка с device_map (работает на M1)
        _model = AutoModelForCausalLM.from_pretrained(
            _QWEN_MODEL_NAME,
            torch_dtype=torch.float16,          # float16 → 2× меньше RAM
            device_map="auto",                  # MPS или CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True              # критично для M1
        )
    except Exception as e1:
        logger.warning(f"Прямая загрузка не удалась: {e1}. Пробуем через accelerate dispatch...")
        # Вариант 2: Через accelerate (если не хватает памяти)
        try:
            with init_empty_weights():
                _model = AutoModelForCausalLM.from_pretrained(
                    _QWEN_MODEL_NAME,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            _model = load_checkpoint_and_dispatch(
                _model,
                checkpoint=_QWEN_MODEL_NAME,
                device_map="auto",
                dtype=torch.float16,
                no_split_module_classes=["Qwen2DecoderLayer"]  # важно для Qwen
            )
        except Exception as e2:
            logger.error(f"Обе попытки загрузки провалились: {e2}")
            raise RuntimeError("Не удалось загрузить Qwen3-1.7B")

    logger.info("✅ Qwen3-1.7B успешно загружена (float16, device_map=auto)")
    return _tokenizer, _model


def query_qwen(
    prompt: str,
    history: List[ChatMessage],
    temperature: float,
    max_tokens: int
) -> str:
    """
    Генерирует ответ с использованием Qwen3-1.7B (1.7B параметров).
    История обрезается по токенам (до 28k контекста).
    """
    from .token_utils import truncate_history

    safe_history = truncate_history(
        history=history,
        prompt=prompt,
        max_total_tokens=28672,      # оставляем запас под ответ
        reserved_for_response=max_tokens
    )

    # Формируем диалог в формате Qwen
    messages = [{"role": "system", "content": (
        "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
        "Если вопрос неясен — уточни. Не выдумывай факты."
    )}]
    for msg in safe_history:
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.text
        })
    messages.append({"role": "user", "content": prompt})

    tokenizer, model = _load_qwen()

    # Применяем шаблон чата
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    ).to(model.device)

    # Генерация
    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.95

    try:
        logger.debug("Запуск генерации через Qwen3-8B...")
        with torch.no_grad():  # экономия памяти
            outputs = model.generate(inputs, **generate_kwargs)

        input_len = inputs.shape[-1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Очистка от артефактов Qwen3
        if ":</think>" in decoded:
            decoded = decoded.split(":</think>")[-1].strip()
        elif "</think>" in decoded:
            decoded = decoded.split("</think>")[-1].strip()

        logger.debug("✅ Ответ от Qwen3-1.7B получен")
        return decoded

    except Exception as e:
        logger.error(f"Ошибка генерации в Qwen3-1.7B: {e}")
        raise RuntimeError(f"Ошибка Qwen3-1.7B: {e}")
