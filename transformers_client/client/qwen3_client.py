import logging
import torch
import gc
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from transformers_client.endpoint.qwen3_entities import ChatMessage
from transformers_client.endpoint.qwen3_settings import Qwen3Settings
from transformers_client.client.qwen3_utils import truncate_and_build_messages

logger = logging.getLogger(__name__)


class Qwen3Client:
    def __init__(self, settings: Qwen3Settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

    def connect(self) -> bool:
        """
        Загружает модель и токенизатор в память.
        """

        if self.is_loaded:
            logger.info("Qwen3 уже загружена, пропускаем...")
            return True

        logger.info("=" * 60)
        logger.info("ИНИЦИАЛИЗАЦИЯ QWEN3 ЧЕРЕЗ TRANSFORMERS")
        logger.info("=" * 60)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Обнаруженное устройство: {device}")

        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Прямая загрузка
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e1:
            logger.warning(f"Прямая загрузка не удалась: {e1}. Пробуем через accelerate...")
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.settings.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                self.model = load_checkpoint_and_dispatch(
                    self.model,
                    checkpoint=self.settings.model_name,
                    device_map="auto",
                    dtype=torch.float16,
                    no_split_module_classes=["Qwen2DecoderLayer"]
                )
            except Exception as e2:
                logger.error(f"Обе попытки загрузки провалились: {e2}")
                raise RuntimeError("Не удалось загрузить Qwen3")

        self.is_loaded = True
        logger.info("Qwen3 успешно загружена (float16, device_map=auto)")
        return True

    def _cleanup_memory(self):
        """
        Очистка GPU/MPS/CPU памяти после генерации.
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def query(
        self,
        prompt: str,
        history: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("Qwen3 не загружена")

        logger.info(f"Запрос к Qwen3 (temperature={temperature}, max_tokens={max_tokens})")

        messages, _ = truncate_and_build_messages(
            prompt=prompt,
            history=history,
            max_total_tokens=self.settings.max_context_length,
            reserved_for_response=max_tokens,
        )

        # Применяем шаблон чата
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )

        # Обработка результата apply_chat_template
        if isinstance(tokenized, dict):
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask")
        else:
            input_ids = tokenized
            attention_mask = None

        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        else:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            attention_mask = (input_ids != pad_id).long().to(self.model.device)

        # Конфигурация генерации
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=0.95 if temperature > 0.0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config,
                )

            input_len = input_ids.shape[-1]
            decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            # Очистка от артефактов Qwen
            if ":</think>" in decoded:
                decoded = decoded.split(":</think>")[-1].strip()
            elif "</think>" in decoded:
                decoded = decoded.split("</think>")[-1].strip()

            logger.info(f"Ответ от Qwen3 получен (длина: {len(decoded)} символов)")
            self._cleanup_memory()
            return decoded

        except Exception as e:
            logger.error(f"Ошибка генерации в Qwen3: {e}")
            self._cleanup_memory()
            raise RuntimeError(f"Ошибка Qwen3: {e}") from e