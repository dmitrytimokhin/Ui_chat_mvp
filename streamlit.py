import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("Настройки")
model_choice = st.sidebar.selectbox(
    "Модель:",
    options=["phi3_ollama", "qwen_transformers"],
    index=0,
    help="phi3_ollama → быстрее, qwen_transformers → точнее (но требует GPU/CPU+RAM)"
)
temperature = st.sidebar.slider("Температура", 0.0, 1.0, 0.0, 0.1)
max_tokens_response = st.sidebar.slider("Макс. токенов ответа", 1, 4096, 512, 64)

if st.sidebar.button("🗑️ Сбросить диалог"):
    st.session_state.chat_history = []
    st.rerun()

st.title("💬 Чат с локальной LLM")

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.chat_history.append({"role": "user", "text": prompt})

    payload = {
        "prompt": prompt,
        "history": st.session_state.chat_history[:-1],  # без текущего промпта
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

# Отображаем последние 20 сообщений
for msg in st.session_state.chat_history[-20:]:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

st.caption(f"Модель: {model_choice} | Температура: {temperature}")
