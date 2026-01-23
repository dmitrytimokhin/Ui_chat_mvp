from fastapi import APIRouter, Request, HTTPException
from transformers_client.endpoint.qwen3_entities import ChatRequest, ChatResponse
from transformers_client.endpoint.qwen3_lifespan import qwen3_lifespan

qwen3_router = APIRouter(
    prefix="/qwen3",
    tags=["Qwen3"],
    lifespan=qwen3_lifespan
)

@qwen3_router.post("/chat", response_model=ChatResponse)
async def chat_with_qwen3(request: ChatRequest, req: Request):
    client = getattr(req.app.state, "qwen3_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Qwen3 client not initialized")

    try:
        response_text = client.query(
            prompt=request.prompt,
            history=request.history,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))