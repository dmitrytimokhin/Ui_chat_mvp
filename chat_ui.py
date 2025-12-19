import streamlit as st
import requests
from datetime import datetime

FASTAPI_URL = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("🧠 LLM Чат")
page = st.sidebar.radio("Навигация", ["О проекте", "Чат", "История диалога"], index=0)

if page == "О проекте":
    st.title("📖 О проекте")
    st.markdown("""
    ### 🧠 Гибридный локальный LLM-чат

    - **phi3_ollama**: через локальный Ollama (быстро).
    - **qwen_vllm**: через vLLM (CPU) с моделью Qwen3-4B (медленно, но мощно).
    - Все данные остаются локально.
    - Архитектура: Streamlit → FastAPI → vLLM/Ollama.
    """)

elif page == "Чат":
    st.title("💬 Чат с локальной LLM")
    model_choice = st.sidebar.selectbox("Модель:", ["ollama", "Qwen3"], index=0)
    # Если выбрали ollama — показываем вариант (phi / qwen_lite / qwen_pro)
    ollama_variant = None
    if model_choice == "ollama":
        ollama_variant = st.sidebar.selectbox("Ollama модель:", ["phi", "qwen_lite", "qwen_pro"], index=0)

    temperature = st.sidebar.slider("Температура", 0.0, 1.0, 0.0, 0.1)
    # Числовой ввод максимального количества токенов (целое число)
    max_tokens_response = int(st.sidebar.number_input("Макс. токенов ответа", min_value=1, max_value=4096, value=512, step=1))

    if st.sidebar.button("🗑️ Сбросить диалог"):
        st.session_state.chat_history = []
        st.rerun()

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        payload = {
            "prompt": prompt,
            "history": st.session_state.chat_history[:-1],
            "model_alias": model_choice,
            "ollama_model": ollama_variant,
            "temperature": temperature,
            "max_tokens": max_tokens_response
        }

        with st.spinner("Генерация ответа..."):
            try:
                # Qwen on CPU can be slow; increase timeout to 10 minutes
                resp = requests.post(f"{FASTAPI_URL}/chat", json=payload, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "") or f"❌ Ошибка: {data.get('error', 'неизвестно')}"
                else:
                    response = f"❌ Ошибка API: {resp.status_code}"
            except Exception as e:
                response = f"❌ Нет связи с бэкендом: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "text": response})

    for msg in st.session_state.chat_history[-30:]:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
    extra = f" (variant={ollama_variant})" if ollama_variant else ""
    st.caption(f"Модель: {model_choice}{extra} | Температура: {temperature}")

elif page == "История диалога":
    st.title("📜 История диалога")
    if not st.session_state.chat_history:
        st.info("Диалог пуст.")
    else:
        for msg in st.session_state.chat_history:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_emoji} {msg['role'].title()}:** {msg['text']}")
        history_text = "\n".join(
            f"[{'Пользователь' if m['role'] == 'user' else 'Ассистент'}]: {m['text']}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            "📥 Скачать историю в TXT",
            history_text,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )