"""Minimal local Ollama client for the required Llama 3:8B model."""

import json
import requests

from config.config import LLM_MODEL, OLLAMA_URL

OLLAMA_TIMEOUT = 600


def _base_url():
    return OLLAMA_URL.rsplit("/", 1)[0]


def check_ollama():
    try:
        response = requests.get(f"{_base_url()}/tags", timeout=5)
        response.raise_for_status()
        models = [model.get("name", "") for model in response.json().get("models", [])]
        installed = LLM_MODEL in models
        return {
            "ok": True,
            "model": LLM_MODEL,
            "model_installed": installed,
            "models": models,
            "message": (
                f"Ollama is running and {LLM_MODEL} is installed."
                if installed
                else f"Ollama is running, but {LLM_MODEL} is not installed."
            ),
        }
    except requests.exceptions.ConnectionError:
        return {"ok": False, "model": LLM_MODEL, "model_installed": False, "models": [],
                "message": "Cannot connect to Ollama at http://localhost:11434."}
    except requests.exceptions.Timeout:
        return {"ok": False, "model": LLM_MODEL, "model_installed": False, "models": [],
                "message": "Ollama health check timed out."}
    except Exception as exc:
        return {"ok": False, "model": LLM_MODEL, "model_installed": False, "models": [],
                "message": str(exc)}


def generate_stream(prompt, temperature=0.1):
    health = check_ollama()
    if not health["ok"]:
        raise RuntimeError(health["message"])
    if not health["model_installed"]:
        raise RuntimeError(
            f"Model '{LLM_MODEL}' is not installed. Run: ollama pull {LLM_MODEL}"
        )

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": 180},
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, OLLAMA_TIMEOUT),
        )
        response.raise_for_status()
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError("Ollama timed out while generating the answer.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not contact Ollama: {exc}") from exc

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            chunk = data.get("response", "")
            if chunk:
                yield chunk
            if data.get("done", False):
                break
    finally:
        response.close()
