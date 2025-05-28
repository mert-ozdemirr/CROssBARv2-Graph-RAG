from providers.openai_provider import call_openai
from providers.openrouter_provider import call_openrouter

def generate_response(model: str, prompt: str, api_key: str) -> str:
    if model.startswith("openai"):
        return call_openai(model, prompt, api_key)
    elif model.startswith("gemini") or model.startswith("co") or model.startswith("o"):
        return call_openrouter(model, prompt, api_key)
    else:
        raise ValueError(f"Unsupported model: {model}")
