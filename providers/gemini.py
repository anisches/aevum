from typing import TYPE_CHECKING, AsyncIterator

import google.generativeai as genai

from .base import BaseProvider, CompletionOptions, CompletionResponse, Message, Role

if TYPE_CHECKING:
    from tools.base import BaseTool

DEFAULT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

_ROLE_MAP = {
    Role.USER:      "user",
    Role.ASSISTANT: "model",
}

_GEMINI_TYPE_MAP = {
    "string":  "STRING",
    "integer": "INTEGER",
    "number":  "NUMBER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


def _to_gemini_schema(params: dict) -> genai.protos.Schema:
    props = {}
    for name, spec in params.get("properties", {}).items():
        t = _GEMINI_TYPE_MAP.get(spec.get("type", "string"), "STRING")
        props[name] = genai.protos.Schema(type=t, description=spec.get("description", ""))
    return genai.protos.Schema(
        type="OBJECT",
        properties=props,
        required=params.get("required", []),
    )


def _tool_declarations(tools: dict[str, "BaseTool"]) -> list:
    out = []
    for tool in tools.values():
        s = tool.schema()
        out.append(
            genai.protos.FunctionDeclaration(
                name=s["name"],
                description=s["description"],
                parameters=_to_gemini_schema(s["parameters"]),
            )
        )
    return out


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)

    def _build_history(
        self, messages: list[Message]
    ) -> tuple[str, list[dict]]:
        system  = ""
        history: list[dict] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            else:
                history.append({"role": _ROLE_MAP[msg.role], "parts": [msg.content]})
        return system, history

    async def complete(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> CompletionResponse:
        system, history = self._build_history(messages)
        model  = genai.GenerativeModel(options.model, system_instruction=system or None)
        last   = history.pop() if history else {"parts": [""]}
        chat   = model.start_chat(history=history)
        response = await chat.send_message_async(
            last["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=options.temperature,
                max_output_tokens=options.max_tokens,
            ),
        )
        return CompletionResponse(content=response.text, model=options.model)

    async def complete_with_tools(
        self,
        messages: list[Message],
        options: CompletionOptions,
        tools: dict[str, "BaseTool"],
        on_step=None,
    ) -> CompletionResponse:
        if not tools:
            return await self.complete(messages, options)

        system, history = self._build_history(messages)
        gemini_tools    = [genai.protos.Tool(function_declarations=_tool_declarations(tools))]
        model  = genai.GenerativeModel(
            options.model,
            system_instruction=system or None,
            tools=gemini_tools,
        )
        last = history.pop() if history else {"parts": [""]}
        chat = model.start_chat(history=history)

        response = await chat.send_message_async(
            last["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=options.temperature,
                max_output_tokens=options.max_tokens,
            ),
        )

        while True:
            fn_calls = [
                part.function_call
                for part in response.parts
                if hasattr(part, "function_call") and part.function_call.name
            ]

            thought = " ".join(
                p.text for p in response.parts if hasattr(p, "text") and p.text
            ).strip()
            if thought and on_step:
                on_step("thought", thought)

            if not fn_calls:
                return CompletionResponse(content=response.text, model=options.model)

            tool_responses = []
            for fc in fn_calls:
                args_preview = ", ".join(f"{k}={v!r}" for k, v in list(dict(fc.args).items())[:2])
                if on_step:
                    on_step("action", f"{fc.name}({args_preview})")
                tool   = tools.get(fc.name)
                result = (await tool.run(**dict(fc.args))).output if tool else f"unknown tool: {fc.name}"
                if on_step:
                    on_step("observation", str(result)[:120])
                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": str(result)},
                        )
                    )
                )

            response = await chat.send_message_async(tool_responses)

    async def stream(
        self,
        messages: list[Message],
        options: CompletionOptions,
    ) -> AsyncIterator[str]:
        system, history = self._build_history(messages)
        model  = genai.GenerativeModel(options.model, system_instruction=system or None)
        last   = history.pop() if history else {"parts": [""]}
        chat   = model.start_chat(history=history)
        response = await chat.send_message_async(
            last["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=options.temperature,
                max_output_tokens=options.max_tokens,
            ),
            stream=True,
        )
        async for chunk in response:
            yield chunk.text

    def list_models(self) -> list[str]:
        return DEFAULT_MODELS
