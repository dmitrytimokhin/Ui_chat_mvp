import requests
from requests.exceptions import ConnectionError, Timeout
from ollama_client.endpoint.ollama_settings import OllamaSettings

def ollama_connection(settings: OllamaSettings) -> bool:
    """
    Проверяет доступность Ollama сервера по endpoint.
    """
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except (ConnectionError, Timeout):
        return False