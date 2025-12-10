# chat_ui.py
import streamlit as st
import requests
from datetime import datetime

FASTAPI_URL = "http://localhost:8000"

# === Инициализация состояния ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Навигация ===
st.sidebar.title("🧠 LLM Чат")
page = st.sidebar.radio(
    "Навигация",
    options=[ "О проекте", "Чат", "История диалога"],
    index=0
)

# === Страница: О проекте ===
if page == "О проекте":
    st.title("📖 О проекте")
    st.markdown("""
    ### 🧠 Гибридный локальный LLM-чат

    Этот проект реализует **полностью автономный чат с языковыми моделями**, работающий без подключения к облачным сервисам.

    #### 🔑 Основные возможности:
    - **Два режима работы**:
      - `phi3_ollama` — лёгкая модель через локальный сервер Ollama (быстро, мало RAM).
      - `qwen_transformers` — мощная модель **Qwen3-8B** с 4/8-битной квантизацией (точнее, требует больше ресурсов).
    - **Автоматическое управление контекстом**: история обрезается при превышении лимита токенов.
    - **Полная приватность**: все данные остаются на вашем устройстве.
    - **Поддержка MacBook Pro M1/M2** через MPS и GGUF (при использовании llama.cpp).

    #### 🛠️ Архитектура:
    ```
    Streamlit (UI) → FastAPI (бэкенд) → LLM (Ollama / Qwen)
    ```

    #### 📦 Технологии:
    - Python 3.11
    - FastAPI + Streamlit
    - Ollama, Hugging Face Transformers, llama.cpp (опционально)
    - bitsandbytes (на Linux с GPU)

    #### 📂 Структура проекта:
    ```
    ./chat_ui.py
    ./fastapi_llm/
      ├── main.py
      ├── models.py
      ├── llm_ollama.py
      └── llm_qwen.py
    ```

    Автор: **Dmitry**  
    GitHub: [github.com/dmitrytimokhin](https://github.com/dmitrytimokhin)
    """)

# === Страница: Чат ===
elif page == "Чат":
    st.title("💬 Чат с локальной LLM")

    model_choice = st.sidebar.selectbox(
        "Модель:",
        options=["phi3_ollama", "qwen_transformers"],
        index=0,
        help="phi3_ollama → быстро, qwen_transformers → мощнее (Qwen3-8B)"
    )
    temperature = st.sidebar.slider("Температура", 0.0, 1.0, 0.0, 0.1)
    max_tokens_response = st.sidebar.slider("Макс. токенов ответа", 1, 4096, 512, 64)

    if st.sidebar.button("🗑️ Сбросить диалог"):
        st.session_state.chat_history = []
        st.rerun()

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.chat_history.append({"role": "user", "text": prompt})

        payload = {
            "prompt": prompt,
            "history": st.session_state.chat_history[:-1],
            "model_alias": model_choice,
            "temperature": temperature,
            "max_tokens": max_tokens_response
        }

        with st.spinner("Генерация ответа..."):
            try:
                resp = requests.post(f"{FASTAPI_URL}/chat", json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "") or f"❌ Ошибка: {data.get('error', 'неизвестно')}"
                else:
                    response = f"❌ Ошибка API: {resp.status_code}"
            except Exception as e:
                response = f"❌ Нет связи с бэкендом: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "text": response})

    # Отображение чата
    for msg in st.session_state.chat_history[-30:]:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    st.caption(f"Модель: {model_choice} | Температура: {temperature}")

# === Страница: История диалога ===
elif page == "История диалога":
    st.title("📜 История диалога")

    if not st.session_state.chat_history:
        st.info("Диалог пуст. Перейдите на страницу «Чат» и начните общение.")
    else:
        # Отображаем историю
        for i, msg in enumerate(st.session_state.chat_history):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_emoji} {msg['role'].title()}:** {msg['text']}")

        # Генерация TXT
        history_text = ""
        for msg in st.session_state.chat_history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            history_text += f"[{role}]: {msg['text']}\n"

        # Кнопка скачивания
        st.download_button(
            label="📥 Скачать историю в TXT",
            data=history_text,
            file_name=f"llm_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
