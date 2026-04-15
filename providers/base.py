from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Callable

if TYPE_CHECKING:
    from tools.base import BaseTool

StepCallback = Callable[[str, str], None] | None


class Role(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class CompletionOptions:
    model: str
    temperature: float = 0.7
    max_tokens: int    = 4096
    stream: bool       = False
    extra: dict        = field(default_factory=dict)


@dataclass
class CompletionResponse:
    content: str
    model: str
    input_tokens: int       = 0
    output_tokens: int      = 0
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseProvider(ABC):

    name: str = ""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> CompletionResponse:
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> AsyncIterator[str]:
        ...

    async def complete_with_tools(
        self,
        messages: list[Message],
        options: CompletionOptions,
        tools: dict[str, "BaseTool"],
        on_step: "StepCallback" = None,
    ) -> CompletionResponse:
        return await self.complete(messages, options)

    @abstractmethod
    def list_models(self) -> list[str]:
        ...
