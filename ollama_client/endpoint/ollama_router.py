from fastapi import APIRouter, Request, HTTPException
from ollama_client.endpoint.ollama_entities import ChatRequest, ChatResponse
from ollama_client.endpoint.ollama_lifespan import ollama_lifespan

ollama_router = APIRouter(
    prefix="/ollama",
    tags=["Ollama"],
    lifespan=ollama_lifespan
)

@ollama_router.post("/chat", response_model=ChatResponse)
async def chat_with_ollama(request: ChatRequest, req: Request):
    client = getattr(req.app.state, "ollama_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Ollama client not initialized")

    try:
        response_text = client.query(
            prompt=request.prompt,
            history=request.history,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
#            model_name=request.model_name,
        )
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
