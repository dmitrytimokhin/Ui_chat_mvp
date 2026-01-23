from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Qwen3Settings(BaseSettings):
    model_name: str = Field(
        default= "Qwen/Qwen3-1.7B",
        description="Наименование модели",
        alias="model_name"
    )

    max_context_length: int = Field(
        default=28672,
        description="Максимальное число токенов в контексте модели",
        alias="max_context_length"
    )

    reserved_tokens_for_response: int = Field(
        default=1024,
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
def get_qwen3_settings() -> Qwen3Settings:
    """
    Получить настройки приложения

    Используют @lru_cache для кэширования - настройки загружаются один раз
    и переиспользуются при последующих вызовах

    :return: экземпляр Qwen3Settings
    """
    return Qwen3Settings()