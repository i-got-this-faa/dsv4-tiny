"""DSV4-Tiny inference server with OpenAI-compatible API.

Provides /v1/chat/completions endpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.model import DSV4TinyForCausalLM

app = FastAPI(title="DSV4-Tiny Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
_model: Optional[DSV4TinyForCausalLM] = None
_tokenizer: Optional[object] = None
_device: torch.device = torch.device("cpu")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "dsv4-tiny"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dsv4-tiny"


@app.on_event("startup")
async def load_model():
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {_device}...")

    cfg = DSV4TinyConfig()

    # Build model from scratch (no pretrained weights for v1)
    _model = DSV4TinyForCausalLM(cfg)
    _model = _model.to(device=_device, dtype=torch.bfloat16)
    _model.eval()

    print("Model loaded successfully!")

    # Load tokenizer (Qwen3.5 tokenizer)
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(cfg._BASE_MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        _tokenizer = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": "dsv4-tiny"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id="dsv4-tiny",
                created=int(time.time()),
            )
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format prompt from messages
    if _tokenizer is not None:
        prompt = _tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_device)
    else:
        # Fallback: use last message content directly
        prompt = request.messages[-1].content
        # Simple byte tokenization for demo
        input_ids = torch.tensor(
            [[ord(c) for c in prompt[:1024]]], device=_device
        ).long()

    # Generate
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

    # Decode
    if _tokenizer is not None:
        generated_text = _tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
    else:
        generated_text = "".join(chr(min(c, 127)) for c in output_ids[0][input_ids.shape[1]:])

    prompt_tokens = input_ids.shape[1]
    completion_tokens = output_ids.shape[1] - prompt_tokens

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": "length",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="DSV4-Tiny Inference Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(
        "src.dsv4_tiny.inference:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
