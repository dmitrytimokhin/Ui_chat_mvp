import streamlit as st

MAX_WORDS = 20

# Инициализация
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def count_words_in_history(history):
    return sum(len(msg["content"].split()) for msg in history)

def trim_history_to_max_words(history, max_words):
    while history and count_words_in_history(history) > max_words:
        history.pop(0)
    return history

# ========== Навигация ==========
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти на:", ["Описание проекта", "Прототип чата"], key="sidebar_nav")

if page == "Описание проекта":
    st.session_state.current_page = "home"
else:
    st.session_state.current_page = "chat"

# ========== Страница: Описание проекта ==========
if st.session_state.current_page == "home":
    st.title("Вопрос-ответная система на основе RAG")
    st.markdown("""
    ### MVP проекта
    - Чат с контекстом до 32 000 слов.
    - Старые сообщения удаляются при превышении лимита.
    - На экране — последние 10 сообщений.
    """)

# ========== Страница: Прототип чата ==========
elif st.session_state.current_page == "chat":
    st.title("💬 Прототип чата")

    # === 1. Обработка нового ввода (если есть) ===
    if prompt := st.chat_input("Напишите сообщение..."):
        # Добавляем сообщение пользователя
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Заглушка: эхо
        st.session_state.chat_history.append({"role": "system", "content": prompt})
        # Обрезаем под лимит
        st.session_state.chat_history = trim_history_to_max_words(
            st.session_state.chat_history, MAX_WORDS
        )

    # === 2. ОТОБРАЖЕНИЕ чата (после обновления!) ===
    # Показываем последние 10 сообщений
    start_idx = max(0, len(st.session_state.chat_history) - 10)
    for msg in st.session_state.chat_history[start_idx:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    # === Отладка (опционально) ===
    total_words = count_words_in_history(st.session_state.chat_history)
    st.caption(f"📝 Всего слов в контексте: {total_words} / {MAX_WORDS}\nВесь накопленный контекст: {st.session_state.chat_history}")