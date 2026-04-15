import json
import os
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from .base import BaseProvider, CompletionOptions, CompletionResponse, Message, Role

if TYPE_CHECKING:
    from tools.base import BaseTool

_ROLE_MAP = {
    Role.SYSTEM:    "system",
    Role.USER:      "user",
    Role.ASSISTANT: "assistant",
}


def _tool_schemas(tools: dict[str, "BaseTool"]) -> list[dict]:
    out = []
    for tool in tools.values():
        s = tool.schema()
        out.append({
            "type": "function",
            "function": {
                "name":        s["name"],
                "description": s["description"],
                "parameters":  s["parameters"],
            },
        })
    return out


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        base_url       = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._base_url = base_url.rstrip("/")
        self._client   = httpx.AsyncClient(base_url=self._base_url, timeout=120)

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {"role": _ROLE_MAP[msg.role], "content": msg.content}
            for msg in messages
        ]

    async def complete(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> CompletionResponse:
        payload = {
            "model":    options.model,
            "messages": self._build_messages(messages),
            "stream":   False,
            "options":  {
                "temperature": options.temperature,
                "num_predict": options.max_tokens,
                **options.extra,
            },
        }
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return CompletionResponse(
            content=data["message"]["content"],
            model=data.get("model", options.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def complete_with_tools(
        self,
        messages: list[Message],
        options: CompletionOptions,
        tools: dict[str, "BaseTool"],
        on_step=None,
    ) -> CompletionResponse:
        if not tools:
            return await self.complete(messages, options)

        api_messages = self._build_messages(messages)
        schemas      = _tool_schemas(tools)
        total_in = total_out = 0

        while True:
            payload = {
                "model":    options.model,
                "messages": api_messages,
                "stream":   False,
                "tools":    schemas,
                "options":  {
                    "temperature": options.temperature,
                    "num_predict": options.max_tokens,
                },
            }
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data     = response.json()
            total_in  += data.get("prompt_eval_count", 0)
            total_out += data.get("eval_count", 0)

            message    = data["message"]
            tool_calls = message.get("tool_calls") or []

            thought = message.get("content", "").strip()
            if thought and on_step:
                on_step("thought", thought)

            if not tool_calls:
                return CompletionResponse(
                    content=thought,
                    model=data.get("model", options.model),
                    input_tokens=total_in,
                    output_tokens=total_out,
                )

            api_messages.append(message)

            for tc in tool_calls:
                fn        = tc["function"]
                tool_name = fn["name"]
                tool_args = fn.get("arguments") or {}
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                args_preview = ", ".join(f"{k}={v!r}" for k, v in list(tool_args.items())[:2])
                if on_step:
                    on_step("action", f"{tool_name}({args_preview})")

                tool   = tools.get(tool_name)
                output = (
                    str((await tool.run(**tool_args)).output)
                    if tool else f"unknown tool: {tool_name}"
                )
                if on_step:
                    on_step("observation", output[:120])
                api_messages.append({"role": "tool", "content": output})

    async def stream(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> AsyncIterator[str]:
        payload = {
            "model":    options.model,
            "messages": self._build_messages(messages),
            "stream":   True,
            "options":  {
                "temperature": options.temperature,
                "num_predict": options.max_tokens,
                **options.extra,
            },
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if content := data.get("message", {}).get("content"):
                    yield content

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
