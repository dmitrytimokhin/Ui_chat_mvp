# client/ollama_client.py
import logging
from typing import List, Optional
from requests.exceptions import RequestException, Timeout, ConnectionError
import requests

from ollama_client.endpoint.ollama_entities import ChatMessage
from ollama_client.endpoint.ollama_settings import OllamaSettings
from ollama_client.client.ollama_connection import ollama_connection
from ollama_client.client.ollama_utils import truncate_and_build_messages

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, settings: OllamaSettings):
        self.settings = settings
        self.is_connected = False

    def connect(self) -> bool:
        """Проверяет подключение к Ollama и устанавливает флаг готовности."""
        self.is_connected = ollama_connection(self.settings)
        return self.is_connected

    def query(
        self,
        prompt: str,
        history: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        model_name: Optional[str] = None,
    ) -> str:
        if not self.is_connected:
            raise RuntimeError("Ollama client is not connected")

        engine_name = f"Ollama/{model_name or self.settings.model_name}"
        logger.info(f"Запрос к движку: {engine_name}")
        logger.debug(f"Параметры генерации: temperature={temperature}, max_tokens={max_tokens}")

        messages, _ = truncate_and_build_messages(
            prompt=prompt,
            history=history,
            max_total_tokens=self.settings.max_context_length,
            reserved_for_response=max_tokens,
        )

        payload = {
            "model": model_name or self.settings.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            url = f"{self.settings.ollama_url}/api/chat"
            logger.debug(f"Отправка POST-запроса к {url}")
            response = requests.post(
                url=url,
                json=payload,
                timeout=self.settings.request_timeout_seconds
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()

            if not content:
                raise ValueError("Пустой ответ от Ollama")

            logger.info(f"Успешно получен ответ от Ollama (длина: {len(content)} символов)")
            return content

        except ConnectionError as e:
            error_msg = "Не удалось подключиться к Ollama (ConnectionError)"
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e

        except Timeout as e:
            error_msg = f"Таймаут при обращении к Ollama (таймаут: {self.settings.request_timeout_seconds} сек)"
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e

        except RequestException as e:
            status = e.response.status_code if e.response else "unknown"
            error_msg = f"Ошибка HTTP-запроса к Ollama: статус {status}"
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e

        except (ValueError, KeyError, TypeError) as e:
            error_msg = "Некорректный формат ответа от Ollama"
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = "Неожиданная ошибка при работе с Ollama"
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e
