import requests

def query_ollama(
    prompt: str,
    history: list,
    temperature: float,
    max_tokens: int
) -> str:
    messages = [{"role": "system", "content": (
        "Ты — вежливый и точный ассистент. Отвечай кратко, по делу и только на русском языке. "
        "Если вопрос неясен — уточни. Не выдумывай факты."
    )}]

    for msg in history:
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.text
        })
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "phi3",  # или "llama3.2:1b", "gemma2:2b" — ваш выбор для gpt_lite
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
        else:
            raise Exception(f"Ollama error {resp.status_code}: {resp.text}")
    except Exception as e:
        raise Exception(f"Ollama недоступна: {e}")
        