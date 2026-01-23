import logging
import os
import gc

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Настраивает корневой логгер приложения.

    Вызывается при старте приложения (в `main`).
    """
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=level, format=fmt)
    # Подавляем шум от Kubernetes readiness/liveness probe'ов (uvicorn access logs)
    try:
        from utils.log_filters import HealthcheckFilter

        filt = HealthcheckFilter()
        # Apply to uvicorn access logger (HTTP access lines)
        logging.getLogger("uvicorn.access").addFilter(filt)

        # Also add to any existing root handlers so startup messages are filtered
        root = logging.getLogger()
        for h in list(root.handlers):
            h.addFilter(filt)
    except Exception as e:
        # If anything goes wrong, do not break application startup — logging remains configured
        logging.getLogger(__name__).debug(f"Не удалось установить HealthcheckFilter для логов: {e}")


def log_request_start(engine_name: str, temperature: float, max_tokens: int) -> None:
    logger.info(f"Запрос к движку: {engine_name}")
    logger.debug(f"Параметры генерации: temperature={temperature}, max_tokens={max_tokens}")


def cleanup_memory() -> None:
    """Очищает память: освобождает неиспользуемые объекты и кэш CUDA."""
    try:
        # Очистка Python кэша мусора
        gc.collect()
        
        # Если доступна CUDA, очищаем её кэш
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Ошибка с torch: {e}")
            pass
        
        # Если доступна MPS (Apple Silicon), очищаем её память
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"Ошибка с torch: {e}")
            pass
        
        logger.debug("✅ Память успешно очищена (gc + cuda/mps cache)")
    except Exception as e:
        logger.warning(f"Не удалось полностью очистить память: {e}")
