<!-- Copilot / agent guidance for Ui_chat_mvp -->
# Copilot instructions — Ui_chat_mvp

This file gives actionable, project-specific guidance for coding agents working on the Ui_chat_mvp repository.

- Purpose: local LLM chat app (Streamlit UI + FastAPI backend) that routes to Ollama or vLLM/Qwen.

1) Big picture
- Frontend: `chat_ui.py` (Streamlit) — user interacts here and posts JSON to `/chat`.
- Backend: `fastapi_llm/main.py` — single endpoint `POST /chat` (request model: `fastapi_llm/models.py`).
- Engines: `fastapi_llm/llm_ollama.py` (Ollama HTTP) and `fastapi_llm/llm_qwen.py` (vLLM/OpenAI-compatible).
- Token management: `fastapi_llm/token_utils.py` uses `tiktoken` and enforces truncation rules.

2) How requests flow (concrete)
- UI sends JSON to `http://localhost:8000/chat` containing: `prompt`, `history` (array of `{role,text}`), `model_alias`, `temperature`, `max_tokens` — see `fastapi_llm/models.py`.
- `main.py` dispatches by `model_alias`: `phi3_ollama` -> `query_ollama()`, `qwen_vllm` -> `query_qwen()`.
- Both engine adapters call `truncate_history()` to ensure context fits model limits.

3) Running the project (dev examples)
- Start backend (reload): `uvicorn fastapi_llm.main:app --host 0.0.0.0 --port 8000 --reload`
- Start UI: `streamlit run chat_ui.py --server.port=8501`
- Containerized: `Dockerfile.app` builds both backend and UI (CMD runs uvicorn + streamlit). `Dockerfile.vllm` builds a separate vLLM container.

4) Important conventions & gotchas
- Model aliases: use `phi3_ollama` and `qwen_vllm` exactly (backend matches these strings).
- Ollama URL is controlled by env `OLLAMA_URL` (default points to host.docker.internal). See `fastapi_llm/llm_ollama.py`.
- vLLM endpoint assumed at `http://vllm:8000/v1/chat/completions` inside Docker network — adjust when running locally without compose.
- Tokenization: `tiktoken.get_encoding('cl100k_base')` is used; do not change encoder without verifying token counts for both engines.
- Error handling pattern: engine modules raise `RuntimeError` on connectivity/format errors; `main.py` returns `ChatResponse` with empty `response` and `error` filled.

5) Tests & debugging tips
- To reproduce issues quickly, enable uvicorn `--reload` and tail logs — backend logs to standard output (uses `logging`).
- For Ollama connectivity, verify `ollama serve` and `ollama pull phi3` locally; check `OLLAMA_URL` in environment.
- For vLLM, ensure the vLLM container exposes OpenAI-compatible API at `vllm:8000` or change `VLLM_URL` in `fastapi_llm/llm_qwen.py`.

6) Where to change behavior
- System prompt used by both adapters: `_SYSTEM_PROMPT` in `fastapi_llm/llm_ollama.py` and `fastapi_llm/llm_qwen.py`.
- Context limits and reserved tokens: constants at the top of `llm_*.py` files (`MAX_CONTEXT_LENGTH`, `RESERVED_TOKENS_FOR_RESPONSE`). Adjust only with tests.

7) Quick examples to use in edits or PRs
- Add a new model alias `qwen_small`: follow `llm_qwen.py` patterns, add mapping in `main.py`, and document alias in `chat_ui.py` selectbox.
- When updating truncation logic, write a small unit test that asserts `truncate_history()` preserves newest messages and that token counts + prompt + reserved <= max.

If any part is unclear or you want more examples (tests, Docker compose notes, or API contract samples), tell me which section to expand.
