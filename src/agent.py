import os
from typing import AsyncIterator, Callable

from providers.base import (
    BaseProvider,
    CompletionOptions,
    CompletionResponse,
    Message,
    Role,
)
from tools.base import BaseTool
from tools.read_write import ReadFileTool, WriteFileTool
from tools.web_search import WebSearchTool

from .config import Config


def _build_provider(config: Config) -> BaseProvider:
    match config.provider:
        case "anthropic":
            from providers.anthropic import AnthropicProvider
            return AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        case "gemini":
            from providers.gemini import GeminiProvider
            return GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
        case "ollama":
            from providers.ollama import OllamaProvider
            return OllamaProvider(base_url=config.ollama_base_url)
        case _:
            raise ValueError(f"Unknown provider: {config.provider!r}")


class Agent:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.provider: BaseProvider = _build_provider(self.config)
        self.tools: dict[str, BaseTool] = {}
        self._history: list[Message] = []

        self.register_tool(ReadFileTool())
        self.register_tool(WriteFileTool())
        self.register_tool(WebSearchTool())

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def _options(self, max_tokens: int | None = None) -> CompletionOptions:
        return CompletionOptions(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            extra=self.config.extra,
        )

    def _build_messages(self, user_input: str, system: str = "") -> list[Message]:
        msgs = []
        if system:
            msgs.append(Message(role=Role.SYSTEM, content=system))
        msgs.extend(self._history)
        msgs.append(Message(role=Role.USER, content=user_input))
        return msgs

    async def _llm(self, prompt: str, max_tokens: int = 100) -> str:
        result = await self.provider.complete(
            [Message(role=Role.USER, content=prompt)],
            self._options(max_tokens=max_tokens),
        )
        return result.content.strip()

    async def chat(self, user_input: str) -> CompletionResponse:
        messages = self._build_messages(user_input)
        response = await self.provider.complete(messages, self._options())
        self._history.append(Message(role=Role.USER, content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=response.content))
        return response

    async def stream(
        self,
        user_input: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(user_input)
        response = await self.provider.complete(messages, self._options())
        self._history.append(Message(role=Role.USER, content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=response.content))
        yield response.content

    def reset(self) -> None:
        self._history.clear()
