# Используем официальный Python 3.11 образ (multi-arch, включая arm64 для M1)
FROM python:3.11-slim

# Установка зависимостей ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка torch с поддержкой CPU (MPS не работает в Docker, но float16 + CPU — да)
# В Docker на M1 torch будет использовать CPU, но это приемлемо для тестирования
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONT_WRITE_BYTECODE=1

WORKDIR /app

# Копируем зависимости в контейнер
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаём папку для модели (на случай, если не монтируем)
RUN mkdir -p /app/models

# Экспонируем порты
EXPOSE 8000 8501

# Запуск обоих сервисов через простой скрипт
CMD ["sh", "-c", "uvicorn fastapi_llm.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit.py --server.port=8501 --server.address=0.0.0.0"]