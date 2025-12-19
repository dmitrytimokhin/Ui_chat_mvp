import streamlit as st
import requests
import json
import os
from datetime import datetime

FASTAPI_URL = "http://localhost:8000"
CONV_FILE = "conversations.json"


def load_conversations() -> dict:
    if os.path.exists(CONV_FILE):
        try:
            with open(CONV_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # migrate old format (name -> list of messages) to new format
                migrated = {}
                default_meta = {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}
                for name, val in data.items():
                    if isinstance(val, dict) and "messages" in val:
                        migrated[name] = val
                    else:
                        migrated[name] = {"messages": val if isinstance(val, list) else [], "meta": default_meta}
                return migrated
        except Exception:
            return {"Диалог 1": {"messages": [], "meta": {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}}}
    return {"Диалог 1": {"messages": [], "meta": {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}}}


def save_conversations(data: dict) -> None:
    try:
        with open(CONV_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "conversations" not in st.session_state:
    # load from disk if available
    st.session_state.conversations = load_conversations()
    st.session_state.active_convo = list(st.session_state.conversations.keys())[0]

if "conversations" in st.session_state and "active_convo" not in st.session_state:
    keys = list(st.session_state.conversations.keys())
    st.session_state.active_convo = keys[0] if keys else "Диалог 1"

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
    # Conversation selector + management
    convo_names = list(st.session_state.conversations.keys())
    try:
        current_index = convo_names.index(st.session_state.active_convo)
    except Exception:
        current_index = 0
        st.session_state.active_convo = convo_names[0]

    selected = st.sidebar.selectbox("Диалог:", convo_names, index=current_index)
    st.session_state.active_convo = selected

    # read existing meta for selected conversation (migrate if needed)
    default_meta = {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}
    convo_entry = st.session_state.conversations.get(selected)
    # if old-format (list), migrate it
    if isinstance(convo_entry, list):
        st.session_state.conversations[selected] = {"messages": convo_entry, "meta": default_meta}
        convo_entry = st.session_state.conversations[selected]

    meta = convo_entry.get("meta", default_meta)

    # Model + meta controls (per-conversation keys)
    model_opts = ["ollama", "Qwen3"]
    try:
        model_idx = model_opts.index(meta.get("model_choice", "ollama"))
    except Exception:
        model_idx = 0
    model_choice = st.sidebar.selectbox("Модель:", model_opts, index=model_idx, key=f"model_{selected}")

    ollama_variant = None
    if model_choice == "ollama":
        ollama_opts = ["phi", "qwen_lite", "qwen_pro"]
        try:
            ov_idx = ollama_opts.index(meta.get("ollama_variant", "phi"))
        except Exception:
            ov_idx = 0
        ollama_variant = st.sidebar.selectbox("Ollama модель:", ollama_opts, index=ov_idx, key=f"ollama_{selected}")

    temperature = st.sidebar.slider("Температура", 0.0, 1.0, value=float(meta.get("temperature", 0.0)), step=0.05, key=f"temp_{selected}")
    max_tokens_response = int(st.sidebar.number_input("Макс. токенов ответа", min_value=1, max_value=4096, value=int(meta.get("max_tokens", 512)), step=1, key=f"max_{selected}"))

    # persist meta changes immediately
    st.session_state.conversations[selected]["meta"] = {
        "model_choice": model_choice,
        "ollama_variant": ollama_variant,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens_response),
    }
    save_conversations(st.session_state.conversations)

    # Rename current conversation
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
            save_conversations(st.session_state.conversations)

    # Delete current conversation (remove key)
    if st.sidebar.button("🗑️ Удалить диалог"):
        rm = st.session_state.active_convo
        st.session_state.conversations.pop(rm, None)
        # ensure at least one convo exists
        if st.session_state.conversations:
            st.session_state.active_convo = list(st.session_state.conversations.keys())[0]
        else:
            st.session_state.conversations = {"Диалог 1": {"messages": [], "meta": {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}}}
            st.session_state.active_convo = "Диалог 1"
        save_conversations(st.session_state.conversations)

    new_name = st.sidebar.text_input("Имя нового диалога", "")
    if st.sidebar.button("➕ Новый диалог"):
        name = new_name.strip() or f"Диалог {len(st.session_state.conversations) + 1}"
        if name in st.session_state.conversations:
            st.sidebar.warning("Такое имя уже существует")
        else:
            # enforce max 50 conversations
            default_meta = {"model_choice": "ollama", "ollama_variant": "phi", "temperature": 0.0, "max_tokens": 512}
            if len(st.session_state.conversations) >= 50:
                # remove the oldest conversation (preserve insertion order)
                oldest = next(iter(st.session_state.conversations))
                st.sidebar.info(f"Достигнут лимит в 50 диалогов — удаляю самый старый: {oldest}")
                st.session_state.conversations.pop(oldest, None)

            st.session_state.conversations[name] = {"messages": [], "meta": default_meta}
            st.session_state.active_convo = name
            save_conversations(st.session_state.conversations)

    if prompt := st.chat_input("Ваш вопрос..."):
        convo_msgs = st.session_state.conversations[st.session_state.active_convo].setdefault("messages", [])
        convo_msgs.append({"role": "user", "text": prompt})
        payload = {
            "prompt": prompt,
            "history": convo_msgs[:-1],
            "model_alias": model_choice,
            "ollama_model": ollama_variant,
            "temperature": temperature,
            "max_tokens": max_tokens_response
        }

        with st.spinner("Генерация ответа..."):
            try:
                resp = requests.post(f"{FASTAPI_URL}/chat", json=payload, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "") or f"❌ Ошибка: {data.get('error', 'неизвестно')}"
                else:
                    response = f"❌ Ошибка API: {resp.status_code}"
            except Exception as e:
                response = f"❌ Нет связи с бэкендом: {str(e)}"
        st.session_state.conversations[st.session_state.active_convo].setdefault("messages", []).append({"role": "assistant", "text": response})
        save_conversations(st.session_state.conversations)

    # show only active conversation messages
    convo = st.session_state.conversations[st.session_state.active_convo].get("messages", [])
    for msg in convo[-30:]:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
    extra = f" (variant={ollama_variant})" if ollama_variant else ""
    st.caption(f"Модель: {model_choice}{extra} | Диалог: {st.session_state.active_convo} | Температура: {temperature}")

elif page == "История диалога":
    st.title("📜 История диалогов")
    if not st.session_state.conversations:
        st.info("Диалоги отсутствуют.")
    else:
        convo_names = list(st.session_state.conversations.keys())
        try:
            default_idx = convo_names.index(st.session_state.active_convo)
        except Exception:
            default_idx = 0
        sel = st.selectbox("Выберите диалог для просмотра:", convo_names, index=default_idx)
        entry = st.session_state.conversations.get(sel, {})
        msgs = entry.get("messages", [])
        st.subheader(sel)
        if not msgs:
            st.write("(пустой)")
        for msg in msgs:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_emoji} {msg['role'].title()}:** {msg['text']}")

        # скачать выбранный диалог
        history_text = "\n".join(
            f"[{'Пользователь' if m['role'] == 'user' else 'Ассистент'}]: {m['text']}" for m in msgs
        )
        st.download_button(
            "📥 Скачать диалог в TXT",
            history_text,
            f"chat_{sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )