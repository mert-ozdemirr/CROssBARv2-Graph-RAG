import requests

def call_openrouter(model: str, prompt: str, api_key: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourapp.com"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
