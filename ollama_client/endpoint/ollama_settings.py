from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class OllamaSettings(BaseSettings):
    ollama_url: str = Field(
        default= "http://localhost:11434",
        description="URL хостинга модели Ollama",
        alias="ollama_url"
    )

    model_name: str = Field(
        default="phi3",
        description="Наименование модели",
        alias="phi"
    )

    request_timeout_seconds: int = Field(
        default=120,
        description="Таймаут на запрос к хостингу"
    )

    max_context_length: int = Field(
        default=4096,
        description="Максимальное число токенов в контексте модели",
        alias="max_context_length"
    )

    reserved_tokens_for_response: int = Field(
        default=512,
        description="Зарезервировано под ответ (используется в truncate)",
        alias="reserved_tokens_for_response"
    )

    model_config = SettingsConfigDict(
        frozen=True,
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env"
    )


@lru_cache()
def get_ollama_settings() -> OllamaSettings:
    """
    Получить настройки приложения

    Используют @lru_cache для кэширования - настройки загружаются один раз
    и переиспользуются при последующих вызовах

    :return: экземпляр OllamaSettings
    """
    return OllamaSettings()