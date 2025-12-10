from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse
from .llm_ollama import query_ollama
from .llm_qwen import query_qwen

app = FastAPI(title="Hybrid LLM Gateway (Ollama + Qwen)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if request.model_alias == "phi3_ollama":
            response = query_ollama(
                prompt=request.prompt,
                history=request.history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        elif request.model_alias == "qwen_transformers":
            response = query_qwen(
                prompt=request.prompt,
                history=request.history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            raise ValueError("model_alias должен быть 'phi3_ollama' или 'qwen_transformers'")
        
        return ChatResponse(response=response)

    except Exception as e:
        return ChatResponse(response="", error=str(e))
