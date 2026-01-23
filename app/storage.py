import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CONV_DIR = DATA_DIR / "conversations"
USERS_FILE = DATA_DIR / "users.json"

DATA_DIR.mkdir(exist_ok=True)
CONV_DIR.mkdir(exist_ok=True)

def load_users() -> Dict[str, str]:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users: Dict[str, str]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def get_user_conversations(username: str) -> Dict[str, Any]:
    path = CONV_DIR / f"{username}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Ошибка открытия файла диалога: {e}")
            pass
    return {"Диалог 1": {"messages": [], "meta": {"model_choice": "ollama", "ollama_variant": "phi3", "temperature": 0.0, "max_tokens": 512}}}

def save_user_conversations(username: str, data: Dict[str, Any]) -> None:
    path = CONV_DIR / f"{username}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        