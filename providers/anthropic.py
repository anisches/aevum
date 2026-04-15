from typing import TYPE_CHECKING, AsyncIterator, Callable

import anthropic

from .base import BaseProvider, CompletionOptions, CompletionResponse, Message, Role

if TYPE_CHECKING:
    from tools.base import BaseTool

_ROLE_MAP = {
    Role.USER:      "user",
    Role.ASSISTANT: "assistant",
}

DEFAULT_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def _tool_schemas(tools: dict[str, "BaseTool"]) -> list[dict]:
    out = []
    for tool in tools.values():
        s = tool.schema()
        out.append({
            "name":         s["name"],
            "description":  s["description"],
            "input_schema": s["parameters"],
        })
    return out


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def _to_api_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        system: str | None = None
        chat: list[dict]   = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            else:
                chat.append({"role": _ROLE_MAP[msg.role], "content": msg.content})
        return system, chat

    async def complete(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> CompletionResponse:
        system, chat = self._to_api_messages(messages)
        kwargs: dict = dict(
            model=options.model,
            max_tokens=options.max_tokens,
            messages=chat,
            **options.extra,
        )
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        content  = response.content[0].text if response.content else ""
        return CompletionResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def complete_with_tools(
        self,
        messages: list[Message],
        options: CompletionOptions,
        tools: dict[str, "BaseTool"],
        on_step: Callable[[str, str], None] | None = None,
    ) -> CompletionResponse:
        if not tools:
            return await self.complete(messages, options)

        system, chat = self._to_api_messages(messages)
        kwargs: dict = dict(
            model=options.model,
            max_tokens=options.max_tokens,
            messages=chat,
            tools=_tool_schemas(tools),
        )
        if system:
            kwargs["system"] = system

        total_in = total_out = 0

        while True:
            response   = await self._client.messages.create(**kwargs)
            total_in  += response.usage.input_tokens
            total_out += response.usage.output_tokens

            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            thought = next((b.text for b in response.content if b.type == "text"), "").strip()
            if thought and on_step:
                on_step("thought", thought)

            if not tool_blocks:
                return CompletionResponse(
                    content=thought,
                    model=response.model,
                    input_tokens=total_in,
                    output_tokens=total_out,
                )

            kwargs["messages"].append({
                "role":    "assistant",
                "content": response.content,
            })

            results = []
            for block in tool_blocks:
                args_preview = ", ".join(f"{k}={v!r}" for k, v in list(block.input.items())[:2])
                if on_step:
                    on_step("action", f"{block.name}({args_preview})")
                tool   = tools.get(block.name)
                output = (
                    str((await tool.run(**block.input)).output)
                    if tool else f"unknown tool: {block.name}"
                )
                if on_step:
                    on_step("observation", output[:120])
                results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     output,
                })

            kwargs["messages"].append({"role": "user", "content": results})

    async def stream(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> AsyncIterator[str]:
        system, chat = self._to_api_messages(messages)
        kwargs: dict = dict(
            model=options.model,
            max_tokens=options.max_tokens,
            messages=chat,
            **options.extra,
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as s:
            async for chunk in s.text_stream:
                yield chunk

    def list_models(self) -> list[str]:
        return DEFAULT_MODELS
