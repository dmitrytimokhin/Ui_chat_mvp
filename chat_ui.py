# chat_ui.py
import streamlit as st
import requests
import json
import os
from datetime import datetime

# === Настройка путей ===
DATA_DIR = "data"
CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

FASTAPI_URL = "http://localhost:8000"


def get_conv_file_path(username: str) -> str:
    """Возвращает путь к файлу диалогов пользователя."""
    return os.path.join(CONV_DIR, f"{username}.json")


def load_conversations(username: str) -> dict:
    """Загружает диалоги пользователя из файла."""
    conv_file = get_conv_file_path(username)
    if os.path.exists(conv_file):
        try:
            with open(conv_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                migrated = {}
                default_meta = {
                    "model_choice": "ollama",
                    "ollama_variant": "phi3",
                    "temperature": 0.0,
                    "max_tokens": 512
                }
                for name, val in data.items():
                    if isinstance(val, dict) and "messages" in val:
                        migrated[name] = val
                    else:
                        migrated[name] = {
                            "messages": val if isinstance(val, list) else [],
                            "meta": default_meta
                        }
                return migrated
        except Exception:
            pass
    return {
        "Диалог 1": {
            "messages": [],
            "meta": {
                "model_choice": "ollama",
                "ollama_variant": "phi3",
                "temperature": 0.0,
                "max_tokens": 512
            }
        }
    }


def save_conversations(username: str, data: dict) -> None:
    """Сохраняет диалоги пользователя в файл."""
    conv_file = get_conv_file_path(username)
    try:
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# === Экран авторизации ===
if "logged_in" not in st.session_state or not st.session_state.get("logged_in", False):
    st.title("🔐 LLM Чат — Вход или регистрация")
    action = st.radio("Выберите действие:", ["Войти", "Зарегистрироваться"], horizontal=True)

    with st.form("auth_form"):
        username = st.text_input("Имя пользователя", help="2–20 символов, буквы/цифры/_/-")
        password = st.text_input("Пароль", type="password", help="Минимум 4 символа")
        submit = st.form_submit_button("Отправить")

        if submit:
            username = username.strip()
            password = password.strip()
            if not username or not password:
                st.error("❌ Логин и пароль обязательны")
            else:
                try:
                    if action == "Войти":
                        resp = requests.post(
                            f"{FASTAPI_URL}/login",
                            data={"username": username, "password": password}
                        )
                        if resp.status_code == 200:
                            session_cookie = resp.cookies.get("session")
                            if session_cookie:
                                # Инициализация чистого состояния
                                st.session_state.clear()
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.session_cookie = session_cookie
                                st.session_state.conversations = load_conversations(username)
                                st.session_state.active_convo = list(st.session_state.conversations.keys())[0]
                                st.rerun()
                            else:
                                st.error("❌ Не получена сессия от сервера")
                        else:
                            st.error("❌ Неверный логин или пароль")
                    else:  # Регистрация
                        resp = requests.post(
                            f"{FASTAPI_URL}/register",
                            data={"username": username, "password": password}
                        )
                        if resp.status_code == 200:
                            st.success("✅ Регистрация успешна! Теперь войдите.")
                        else:
                            error = resp.json().get("detail", "Неизвестная ошибка")
                            st.error(f"❌ {error}")
                except Exception as e:
                    st.error(f"⚠️ Ошибка подключения: {e}")
    st.stop()

# === Кнопка выхода (всегда доступна) ===
if st.sidebar.button("🚪 Выйти"):
    st.session_state.clear()
    st.rerun()

# === Основной интерфейс ===
st.sidebar.title(f"🧠 LLM Чат ({st.session_state.username})")
page = st.sidebar.radio("Навигация", ["О проекте", "Чат", "История диалога"], index=0)

if page == "О проекте":
    st.title("📖 О проекте")
    st.markdown("""
    ### 🧠 Гибридный локальный LLM-чат
    - **Ollama**: `phi3` — быстро.
    - **Qwen3**: локальная модель — мощно.
    - Все данные хранятся локально в `data/conversations/`.
    - Полная изоляция между пользователями.
    """)


elif page == "Чат":
    st.title("💬 Чат с локальной LLM")

    # Управление диалогами
    convo_names = list(st.session_state.conversations.keys())
    current_index = convo_names.index(
        st.session_state.active_convo) if st.session_state.active_convo in convo_names else 0
    selected = st.sidebar.selectbox("Диалог:", convo_names, index=current_index)
    st.session_state.active_convo = selected

    # Загрузка метаданных текущего диалога
    default_meta = {
        "model_choice": "unset",  # ← ключевое изменение!
        "ollama_variant": None,
        "temperature": 0.7,
        "max_tokens": 512
    }
    convo_entry = st.session_state.conversations.get(selected, {"messages": [], "meta": default_meta})
    if isinstance(convo_entry, list):
        convo_entry = {"messages": convo_entry, "meta": default_meta}
        st.session_state.conversations[selected] = convo_entry

    meta = convo_entry.get("meta", default_meta)

    # === Выбор модели с опцией "не выбрано" ===
    model_opts = ["unset", "ollama", "qwen3"]
    model_labels = {"unset": "— Выберите модель —", "ollama": "Ollama", "qwen3": "Qwen3"}
    model_choice = st.sidebar.selectbox(
        "Модель:",
        options=model_opts,
        format_func=lambda x: model_labels.get(x, x),
        index=model_opts.index(meta.get("model_choice", "unset")) if meta.get("model_choice") in model_opts else 0,
        key=f"model_{selected}"
    )

    ollama_variant = None
    if model_choice == "ollama":
        ollama_opts = ["phi3"]
        ov_default = meta.get("ollama_variant", "phi3")
        ov_index = ollama_opts.index(ov_default) if ov_default in ollama_opts else 0
        ollama_variant = st.sidebar.selectbox(
            "Ollama модель:",
            ollama_opts,
            index=ov_index,
            key=f"ollama_{selected}"
        )

    temperature = st.sidebar.slider(
        "Температура",
        0.0, 1.0,
        value=float(meta.get("temperature", 0.7)),
        step=0.05,
        key=f"temp_{selected}"
    )
    max_tokens_response = int(st.sidebar.number_input(
        "Макс. токенов ответа",
        min_value=1, max_value=4096,
        value=int(meta.get("max_tokens", 512)),
        step=1,
        key=f"max_{selected}"
    ))

    # Сохраняем обновлённые метаданные
    st.session_state.conversations[selected]["meta"] = {
        "model_choice": model_choice,
        "ollama_variant": ollama_variant,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens_response),
    }
    save_conversations(st.session_state.username, st.session_state.conversations)

    # === Управление диалогами ===
    rename_value = st.sidebar.text_input("Переименовать текущий диалог", value=st.session_state.active_convo)
    if st.sidebar.button("✏️ Переименовать диалог"):
        newn = rename_value.strip()
        old = st.session_state.active_convo
        if not newn:
            st.sidebar.warning("Имя не может быть пустым")
        elif newn == old:
            pass
        elif newn in st.session_state.conversations:
            st.sidebar.warning("Диалог с таким именем уже существует")
        else:
            st.session_state.conversations[newn] = st.session_state.conversations.pop(old)
            st.session_state.active_convo = newn
            st.rerun()  # ← перезагрузка для отображения нового имени

    if st.sidebar.button("🗑️ Удалить диалог"):
        current_name = st.session_state.active_convo
        st.session_state.conversations.pop(current_name, None)

        # Генерация имени нового диалога
        if st.session_state.conversations:
            existing_numbers = []
            for name in st.session_state.conversations.keys():
                if name.startswith("Диалог "):
                    try:
                        num = int(name.split(" ")[1])
                        existing_numbers.append(num)
                    except (ValueError, IndexError):
                        pass
            next_num = max(existing_numbers) + 1 if existing_numbers else len(st.session_state.conversations) + 1
            new_name = f"Диалог {next_num}"
        else:
            new_name = "Диалог 1"

        # Создаём новый чистый диалог
        st.session_state.conversations[new_name] = {
            "messages": [],
            "meta": {"model_choice": "unset", "ollama_variant": None, "temperature": 0.7, "max_tokens": 512}
        }
        st.session_state.active_convo = new_name

        save_conversations(st.session_state.username, st.session_state.conversations)
        st.rerun()

    new_name = st.sidebar.text_input("Имя нового диалога", "")
    if st.sidebar.button("➕ Новый диалог"):
        name = new_name.strip() or f"Диалог {len(st.session_state.conversations) + 1}"
        if name in st.session_state.conversations:
            st.sidebar.warning("Такое имя уже существует")
        else:
            if len(st.session_state.conversations) >= 50:
                oldest = next(iter(st.session_state.conversations))
                st.sidebar.info(f"Лимит 50 диалогов — удаляю: {oldest}")
                st.session_state.conversations.pop(oldest, None)
            # Новый диалог — модель НЕ ВЫБРАНА
            st.session_state.conversations[name] = {
                "messages": [],
                "meta": {"model_choice": "unset", "ollama_variant": None, "temperature": 0.7, "max_tokens": 512}
            }
            st.session_state.active_convo = name
            save_conversations(st.session_state.username, st.session_state.conversations)
            st.rerun()

    # === Отправка запроса ===
    if prompt := st.chat_input("Ваш вопрос..."):
        if model_choice == "unset":
            st.error("❌ Сначала выберите модель в боковой панели!")
        else:
            convo_msgs = st.session_state.conversations[st.session_state.active_convo].setdefault("messages", [])
            convo_msgs.append({"role": "user", "text": prompt})

            payload = {
                "prompt": prompt,
                "history": convo_msgs[:-1],
                "temperature": temperature,
                "max_tokens": max_tokens_response,
            }
            if model_choice == "ollama":
                payload["model_name"] = ollama_variant

            endpoint = f"{FASTAPI_URL}/ollama/chat" if model_choice == "ollama" else f"{FASTAPI_URL}/qwen3/chat"

            cookies = {"session": st.session_state.session_cookie}

            with st.spinner("Генерация ответа..."):
                try:
                    resp = requests.post(endpoint, json=payload, cookies=cookies, timeout=600)
                    if resp.status_code == 200:
                        response = resp.json().get("response", "").strip() or "❌ Пустой ответ от модели"
                    else:
                        response = f"❌ Ошибка API: {resp.status_code}"
                except Exception as e:
                    response = f"❌ Нет связи с бэкендом: {str(e)}"

            convo_msgs.append({"role": "assistant", "text": response})
            save_conversations(st.session_state.username, st.session_state.conversations)

    # Отображение истории
    convo = st.session_state.conversations[st.session_state.active_convo].get("messages", [])
    for msg in convo[-30:]:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    # Подпись
    if model_choice == "unset":
        st.caption("⚠️ Выберите модель для генерации ответов")
    else:
        extra = f" (variant={ollama_variant})" if ollama_variant else ""
        st.caption(
            f"Модель: {model_labels[model_choice]}{extra} | Диалог: {st.session_state.active_convo} | Температура: {temperature}")

elif page == "История диалога":
    st.title("📜 История диалогов")
    if not st.session_state.conversations:
        st.info("Диалоги отсутствуют.")
    else:
        convo_names = list(st.session_state.conversations.keys())
        default_idx = convo_names.index(
            st.session_state.active_convo) if st.session_state.active_convo in convo_names else 0
        sel = st.selectbox("Выберите диалог для просмотра:", convo_names, index=default_idx)
        msgs = st.session_state.conversations.get(sel, {}).get("messages", [])
        st.subheader(sel)
        if not msgs:
            st.write("(пустой)")
        for msg in msgs:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_emoji} {msg['role'].title()}:** {msg['text']}")

        history_text = "\n".join(
            f"[{'Пользователь' if m['role'] == 'user' else 'Ассистент'}]: {m['text']}" for m in msgs
        )
        st.download_button(
            "📥 Скачать диалог в TXT",
            history_text,
            f"chat_{sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )