rom providers.openai_provider import call_openai
from providers.openrouter_provider import call_openrouter
from providers.anthropic_provider import call_anthropic
from providers.mistral_provider import call_mistral
from providers.local_provider import call_local

def generate_response(model: str, prompt: str, api_key: str) -> str:
    if model.startswith("openai"):
        return call_openai(model, prompt, api_key)
    elif model.startswith("gemini") or model.startswith("co") or model.startswith("o"):
        return call_openrouter(model, prompt, api_key)
    elif model.startswith("claude"):
        return call_anthropic(model, prompt, api_key)
    elif model.startswith("mistral"):
        return call_mistral(model, prompt, api_key)
    elif model.startswith("local"):
        return call_local(model, prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")
