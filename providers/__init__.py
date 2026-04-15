from .base import BaseProvider, Message, Role

__all__ = [
    "BaseProvider",
    "Message",
    "Role",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
]

def __getattr__(name: str):
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider
        return AnthropicProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider
        return GeminiProvider
    if name == "OllamaProvider":
        from .ollama import OllamaProvider
        return OllamaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
