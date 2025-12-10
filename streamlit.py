import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

# Инициализация истории
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Навигация
st.sidebar.title("Настройки")
model_choice = st.sidebar.selectbox(
    "Модель:",
    options=["phi3_ollama", "qwen_transformers"],
    index=0,
    help="phi3_ollama → phi3 (быстро), qwen_transformers → qwen3:0.6b (точнее)"
)
temperature = st.sidebar.slider("Температура", 0.0, 1.0, 0.0, 0.1)
max_tokens_response = st.sidebar.slider("Макс. токенов", 1, 4096, 512, 64)

if st.sidebar.button("🗑️ Сбросить диалог"):
    st.session_state.chat_history = []
    st.rerun()

# Заголовок
st.title("💬 Чат с Ollama (через FastAPI)")

# Обработка ввода
if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.chat_history.append({"role": "user", "text": prompt})

    # Подготовка запроса к FastAPI
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
                if data.get("error"):
                    response = f"❌ Ошибка: {data['error']}"
                else:
                    response = data["response"]
            else:
                response = f"❌ Ошибка API: {resp.status_code}"
        except Exception as e:
            response = f"❌ Нет связи с бэкендом: {str(e)}"

    st.session_state.chat_history.append({"role": "system", "text": response})

# Отображение чата
for msg in st.session_state.chat_history[-10:]:  # последние 10
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["text"])

st.caption(f"Модель: {model_choice} | Температура: {temperature}")
