from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_tokenizer = None
_model = None

def load_qwen():
    global _tokenizer, _model
    if _tokenizer is None:
        print("⏳ Загрузка Qwen3-0.6B...")
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    return _tokenizer, _model

def query_qwen(
    prompt: str,
    history: list,
    temperature: float,
    max_tokens: int
) -> str:
    tokenizer, model = load_qwen()

    messages = []
    for msg in history:
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.text})
    messages.append({"role": "user", "content": prompt + " [/no_think]"})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    input_len = inputs.shape[-1]
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    # Убираем [/no_think] и </think>
    if ":</think>" in decoded:
        decoded = decoded.split(":</think>")[-1].strip()
    elif "</think>" in decoded:
        decoded = decoded.split("</think>")[-1].strip()
    return decoded
    