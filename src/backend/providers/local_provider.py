import requests

def call_local(model: str, prompt: str) -> str:
    response = requests.post(
        f"http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
