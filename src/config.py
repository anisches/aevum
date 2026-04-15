import os
from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["anthropic", "gemini", "ollama"]


@dataclass
class Config:
    provider: Provider = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_tokens: int = 4096
    ollama_base_url: str = "http://localhost:11434"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            provider=os.environ.get("AGENT_PROVIDER", "anthropic"),
            model=os.environ.get("AGENT_MODEL", "claude-sonnet-4-6"),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "4096")),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
        )
