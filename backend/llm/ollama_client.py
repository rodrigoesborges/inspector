# backend/llm/ollama_client.py
import os
import requests
from typing import Dict, Any, Optional
from openai import OpenAI

OLLAMA_URL = os.environ.get("OLLAMA_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

def ollama_generate(prompt: str, model: str = "gpt-5-mini", stream: bool = False, max_tokens: int = 1024) -> Dict[str, Any]:
    """
    Call local Ollama generate endpoint. Returns the generated text or raises.
    """
    if not OLLAMA_URL:
        raise RuntimeError("OLLAMA_URL not set")
    
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": stream}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    js = resp.json()
    if isinstance(js, dict) and "text" in js:
        return {"text": js["text"], "raw": js}
    if isinstance(js, dict) and "content" in js:
        return {"text": js["content"], "raw": js}
    return {"text": str(js), "raw": js}

def openai_generate(prompt: str, model: str = "gpt-5-mini", max_tokens: int = 1024) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": "Você é um assistente inteligente."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    return {"text": text, "raw": response}

def generate_answer(prompt: str, preferred: str = "ollama", **kwargs) -> Dict[str, Any]:
    """
    Composite helper: try preferred (ollama) then fallback to OpenAI if available.
    """
    if preferred == "ollama":
        if OLLAMA_URL:
            try:
                return ollama_generate(prompt, **kwargs)
            except Exception:
                if OPENAI_API_KEY:
                    return openai_generate(prompt, **kwargs)
                raise
        else:
            if OPENAI_API_KEY:
                return openai_generate(prompt, **kwargs)
            else:
                raise RuntimeError("No LLM backend available")
    elif preferred == "openai":
        if OPENAI_API_KEY:
            return openai_generate(prompt, **kwargs)
        else:
            if OLLAMA_URL:
                return ollama_generate(prompt, **kwargs)
            else:
                raise RuntimeError("No LLM backend available")
    else:
        if OPENAI_API_KEY:
            return openai_generate(prompt, **kwargs)
        else:
            raise RuntimeError("No LLM backend available")
