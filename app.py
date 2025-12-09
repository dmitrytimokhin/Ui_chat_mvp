import streamlit as st
import requests

# ========== Настройки ==========
MAX_WORDS = 32000

SYSTEM_INSTRUCTION = (
    "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
    "Если вопрос неясен — уточни. Не выдумывай факты."
)

# ========== Инициализация ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== Вспомогательные функции ==========

def count_words_in_history(history):
    return sum(len(msg["content"].split()) for msg in history)

def trim_history_to_max_words(history, max_words):
    while history and count_words_in_history(history) > max_words:
        history.pop(0)
    return history

def query_ollama(prompt, history):
    try:
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for msg in history:
            messages.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": msg["content"]
            })
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "phi3", "messages": messages, "stream": False},
            timeout=60
        )
        if response.ok:
            return response.json().get("message", {}).get("content", "").strip()
        else:
            return "❌ Ollama: ошибка генерации."
    except requests.exceptions.ConnectionError:
        return "❌ Ollama не запущен. Выполните: `ollama run phi3`"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

# ========== Навигация ==========
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Перейти на:",
    ["Описание проекта", "Прототип чата", "Полный диалог"]
)

if page == "Описание проекта":
    st.session_state.current_page = "home"
elif page == "Прототип чата":
    st.session_state.current_page = "chat"
else:
    st.session_state.current_page = "full_dialog"

# ========== Страница: Описание ==========
if st.session_state.current_page == "home":
    st.title("Вопрос-ответная система (локальный MVP)")
    st.markdown("""
    ### Только локальный режим через Ollama
    
    - Модель: `phi3`
    - Системная инструкция задана в коде.
    - Контекст: до 32 000 слов.
    """)

# ========== Страница: Чат ==========
elif st.session_state.current_page == "chat":
    st.title("💬 Локальный чат (Ollama + phi3)")
    st.info("Убедитесь, что запущена команда: `ollama run phi3` и модель хостится черезе `ollama serve`")

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        history_so_far = st.session_state.chat_history[:-1]

        with st.spinner("Генерация ответа..."):
            response = query_ollama(prompt, history_so_far)

        st.session_state.chat_history.append({"role": "system", "content": response})
        st.session_state.chat_history = trim_history_to_max_words(
            st.session_state.chat_history, MAX_WORDS
        )

    # Отображаем последние 10 сообщений
    start_idx = max(0, len(st.session_state.chat_history) - 10)
    for msg in st.session_state.chat_history[start_idx:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    total_words = count_words_in_history(st.session_state.chat_history)
    st.caption(f"📝 Контекст: {total_words} / {MAX_WORDS} слов")

# ========== Страница: Полный диалог ==========
elif st.session_state.current_page == "full_dialog":
    st.title("📜 Полный диалог")

    if not st.session_state.chat_history:
        st.info("Диалог пока пуст. Перейдите на страницу «Прототип чата» и начните общение.")
    else:
        # Формируем plain text (без Markdown)
        text_lines = []
        for msg in st.session_state.chat_history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            text_lines.append(f"{role}: {msg['content']}")
        
        full_text = "\n".join(text_lines)

        # Отображаем как обычный текст (моноширинный для читаемости)
        st.text_area("Вся история диалога:", value=full_text, height=500, disabled=True)

        # Кнопка скачивания как .txt
        st.download_button(
            label="📥 Скачать диалог как .txt",
            data=full_text,
            file_name="dialog.txt",
            mime="text/plain"
        )

        st.caption(f"Всего сообщений: {len(st.session_state.chat_history)}")