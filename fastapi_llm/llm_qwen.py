# fastapi_llm/llm_qwen.py
import logging
import torch
import gc
from typing import List, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .models import ChatMessage
from .utils import truncate_and_build_messages, log_request_start, cleanup_memory, EngineError, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"
_tokenizer = None
_model = None
_models_initialized = False

def init_models() -> None:
    """Инициализирует модели при старте приложения.
    
    Вызывается один раз при запуске FastAPI приложения.
    """
    global _models_initialized
    if _models_initialized:
        logger.info("Модели уже инициализированы, пропускаем...")
        return
    
    logger.info("=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ПРИ СТАРТЕ ПРИЛОЖЕНИЯ")
    logger.info("=" * 60)
    
    _load_qwen()
    
    _models_initialized = True
    logger.info("=" * 60)
    logger.info("✅ ВСЕ МОДЕЛИ УСПЕШНО ЗАГРУЖЕНЫ И ГОТОВЫ К ИСПОЛЬЗОВАНИЮ")
    logger.info("=" * 60)


def _load_qwen() -> Tuple:
    """Ленивая загрузка Qwen3 с оптимизацией под MPS (M1/M2)."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return _tokenizer, _model

    logger.info("Загрузка Qwen3 (float16) через transformers + accelerate...")

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
            raise RuntimeError("Не удалось загрузить Qwen3")

    logger.info("✅ Qwen3 успешно загружена (float16, device_map=auto)")
    return _tokenizer, _model


def query_qwen(
    prompt: str,
    history: List[ChatMessage],
    temperature: float,
    max_tokens: int
) -> str:
    """
    Генерирует ответ с использованием Qwen3.
    История обрезается по токенам (до 28k контекста).
    """
    log_request_start("Qwen/transformers", temperature, max_tokens)

    # Для Qwen добавляем специальный маркер, отключающий внутренний "think"-режим
    messages, safe_history = truncate_and_build_messages(
        prompt=prompt,
        history=history,
        max_total_tokens=28672,      # оставляем запас под ответ
        reserved_for_response=max_tokens,
        system_prompt=SYSTEM_PROMPT + " /no_think",
    )

    tokenizer, model = _load_qwen()

    # Применяем шаблон чата — tokenizer.apply_chat_template возвращает словарь токенов
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )

    # Извлекаем input_ids и внимание и переносим на устройство модели
    if isinstance(tokenized, dict):
        input_ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
    else:
        # Безопасный fallback: если библиотека вернула тензор
        input_ids = tokenized
        attention_mask = None

    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    else:
        # Если attention_mask не задан, строим его вручную (1 для ненулевых токенов)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_mask = (input_ids != pad_id).long().to(model.device)

    # Подготовка конфигурации генерации (совместимо с новыми версиями transformers)
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0.0),
        temperature=float(temperature) if temperature > 0.0 else None,
        top_p=0.95 if temperature > 0.0 else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    try:
        logger.debug("Запуск генерации через Qwen3...")
        with torch.no_grad():  # экономия памяти
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )

        input_len = input_ids.shape[-1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Очистка от артефактов Qwen3
        if ":</think>" in decoded:
            decoded = decoded.split(":</think>")[-1].strip()
        elif "</think>" in decoded:
            decoded = decoded.split("</think>")[-1].strip()

        logger.debug("✅ Ответ от Qwen3 получен")
        
        # Очистка памяти после генерации
        cleanup_memory()
        
        return decoded

    except Exception as e:
        logger.error(f"Ошибка генерации в Qwen3: {e}")
        # Очистка памяти даже при ошибке
        cleanup_memory()
        raise EngineError(f"Ошибка Qwen3: {e}") from e
