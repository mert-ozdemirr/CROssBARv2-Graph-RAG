import requests

def call_mistral(model: str, prompt: str, api_key: str) -> str:
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
