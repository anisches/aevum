from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    output: Any
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseTool(ABC):
    """Abstract base for all agent tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a result."""

    def schema(self) -> dict:
        """Return JSON schema for tool parameters (override as needed)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}},
        }
