FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopenblas-dev \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONT_WRITE_BYTECODE=1

# Подавляем ненужные FutureWarning от некоторых HF-библиотек
ENV PYTHONWARNINGS="ignore::FutureWarning"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn fastapi_llm.main:app --host 0.0.0.0 --port 8000 & streamlit run chat_ui.py --server.port=8501 --server.address=0.0.0.0"]