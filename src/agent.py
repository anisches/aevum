import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

from providers.base import BaseProvider, CompletionOptions, CompletionResponse, Message, Role
from tools.base import BaseTool
from tools.read_write import ReadFileTool, WriteFileTool
from tools.web_search import WebSearchTool

from .config import Config
from .state import AgentState, Skill

_DEFAULT_STATE_PATH = Path.home() / ".aevum" / "state.json"

_GATE1_PROMPT = """\
Exchange:
user: {user_input}
you:  {response}
turn: {depth}

Reasoning trace:
{trace}

Did something specific happen in this exchange that evidences a skill — something concrete you did, not just that you responded?
Consider both the final answer AND the reasoning trace above (tools called, observations made, how thinking evolved).

If yes, reply in this exact format:
skill: <short name>
evidence: turn {depth} - <what specifically happened>

If nothing specific happened, reply with exactly: none"""

_GATE2_PROMPT = """\
Skill observed: {skill}
Evidence: {evidence}

Give one specific scenario (different from the current exchange) where this skill would naturally apply.
Be concrete — one sentence.
If this cannot generalize to another scenario, reply with exactly: none"""

_GATE3_PROMPT = """\
Exchange:
user: {user_input}
you:  {response}

Skills on record:
{skill_scenarios}

Did you successfully apply any of the skills above in this exchange?
For each that applied and worked, reply:
applied: <exact skill name>

If none applied or worked, reply with exactly: none"""

_VALIDATE_PROMPT = """\
Exchange:
user: {user_input}
you:  {response}

Your evidenced skills:
{skill_state}

Two questions — reply in this exact format:
source: mine | llm
reason: <one sentence>
transfer: yes: <reframed scenario> | no"""


@dataclass
class ResponseMeta:
    source:   str
    reason:   str
    transfer: str


def _parse_candidate(text: str) -> tuple[str, str] | None:
    text = text.strip()
    if text.lower().startswith("none"):
        return None
    skill = evidence = ""
    for line in text.splitlines():
        if line.lower().startswith("skill:"):
            skill = line.split(":", 1)[1].strip()
        elif line.lower().startswith("evidence:"):
            evidence = line.split(":", 1)[1].strip()
    if skill and evidence:
        return skill, evidence
    return None


def _parse_applied(text: str) -> list[str]:
    text = text.strip()
    if text.lower().startswith("none"):
        return []
    applied = []
    for line in text.splitlines():
        if line.lower().startswith("applied:"):
            name = line.split(":", 1)[1].strip()
            if name:
                applied.append(name)
    return applied


def _parse_validation(text: str) -> ResponseMeta:
    source = transfer = reason = ""
    for line in text.splitlines():
        l = line.lower().strip()
        if l.startswith("source:"):
            source = line.split(":", 1)[1].strip()
        elif l.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif l.startswith("transfer:"):
            transfer = line.split(":", 1)[1].strip()
    return ResponseMeta(
        source   = source   or "llm",
        reason   = reason   or "",
        transfer = transfer or "no",
    )


def _relevant_skills(user_input: str, skills: list[Skill]) -> list[Skill]:
    words = set(user_input.lower().split())
    out   = []
    for s in skills:
        scenario_words = set(s.scenario.lower().split())
        skill_words    = set(s.skill.lower().split())
        if words & scenario_words or words & skill_words:
            out.append(s)
    return out


def _prior_encounter(user_input: str, history: list[Message]) -> str | None:
    words = set(user_input.lower().split())
    for msg in history:
        if msg.role != Role.USER:
            continue
        overlap = words & set(msg.content.lower().split())
        if len(overlap) >= 3:
            return msg.content[:80]
    return None


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
    def __init__(
        self,
        config: Config | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.config              = config or Config.from_env()
        self.provider: BaseProvider     = _build_provider(self.config)
        self.tools: dict[str, BaseTool] = {}
        self._history: list[Message]    = []
        self._state_path = state_path or _DEFAULT_STATE_PATH
        self._state      = AgentState.load(self._state_path)
        self.last_meta: ResponseMeta | None = None

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

    def _pre_response_check(self, user_input: str) -> str:
        lines: list[str] = []

        relevant = _relevant_skills(user_input, self._state.skills.skills)
        if relevant:
            lines.append("relevant skills for this question:")
            for s in relevant:
                lines.append(f"  - {s.skill} [{s.confidence}]: {s.scenario}")
        else:
            lines.append("no validated skills map to this question")

        lines.append(f"depth: {self._state.primordial.depth} exchanges have shaped this response")

        prior = _prior_encounter(user_input, self._history)
        if prior:
            lines.append(f"prior encounter: yes — similar question seen before: \"{prior}\"")
        else:
            lines.append("prior encounter: no")

        return "\n".join(lines)

    def _build_messages(self, user_input: str, pre_context: str = "") -> list[Message]:
        system = self._state.build_system_prompt()
        if pre_context:
            system += f"\n\npre-response context:\n{pre_context}"
        return [
            Message(role=Role.SYSTEM, content=system),
            *self._history,
            Message(role=Role.USER, content=user_input),
        ]

    async def _llm(self, prompt: str, max_tokens: int = 100) -> str:
        result = await self.provider.complete(
            [Message(role=Role.USER, content=prompt)],
            self._options(max_tokens=max_tokens),
        )
        return result.content.strip()

    async def _validate_response(self, user_input: str, response: str) -> ResponseMeta:
        if not self._state.skills.skills:
            return ResponseMeta(source="llm", reason="no skill state yet", transfer="no")

        skill_state = "\n".join(
            f"  - {s.skill} [{s.confidence}]: {s.scenario}"
            for s in self._state.skills.skills
        )
        raw = await self._llm(
            _VALIDATE_PROMPT.format(
                user_input=user_input,
                response=response,
                skill_state=skill_state,
            ),
            max_tokens=120,
        )
        return _parse_validation(raw)

    async def _run_skill_gates(
        self, user_input: str, response: str, trace: list[str] | None = None
    ) -> None:
        depth      = self._state.primordial.depth
        trace_text = (
            "\n".join(f"  {s}" for s in trace)
            if trace else "  (direct response — no tool use)"
        )

        gate1 = await self._llm(
            _GATE1_PROMPT.format(
                user_input=user_input, response=response, depth=depth, trace=trace_text
            ),
            max_tokens=120,
        )
        candidate = _parse_candidate(gate1)

        if candidate:
            skill, evidence = candidate
            gate2 = await self._llm(
                _GATE2_PROMPT.format(skill=skill, evidence=evidence),
                max_tokens=100,
            )
            scenario = gate2.strip()
            if not scenario.lower().startswith("none"):
                self._state.skills.record(skill, evidence, scenario)

        if self._state.skills.skills:
            skill_scenarios = "\n".join(
                f"  - {s.skill}: {s.scenario}"
                for s in self._state.skills.skills
            )
            gate3 = await self._llm(
                _GATE3_PROMPT.format(
                    user_input=user_input,
                    response=response,
                    skill_scenarios=skill_scenarios,
                ),
                max_tokens=150,
            )
            for name in _parse_applied(gate3):
                self._state.skills.confirm(name)

    async def _post_exchange(
        self, user_input: str, response: str, trace: list[str] | None = None
    ) -> None:
        self._state.primordial.advance()
        await self._run_skill_gates(user_input, response, trace)
        self._state.save(self._state_path)

    async def _full_pipeline(
        self, user_input: str
    ) -> tuple[CompletionResponse, ResponseMeta, list[str]]:
        pre_context = self._pre_response_check(user_input)
        messages    = self._build_messages(user_input, pre_context)
        trace: list[str] = []

        def _collect(step_type: str, content: str) -> None:
            trace.append(f"{step_type}: {content}")

        response = await self.provider.complete_with_tools(
            messages, self._options(), self.tools, on_step=_collect
        )
        meta = await self._validate_response(user_input, response.content)
        return response, meta, trace

    async def chat(self, user_input: str) -> CompletionResponse:
        response, meta, trace = await self._full_pipeline(user_input)
        self.last_meta        = meta
        self._history.append(Message(role=Role.USER,      content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=response.content))
        await self._post_exchange(user_input, response.content, trace)
        return response

    async def stream(
        self,
        user_input: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AsyncIterator[str]:
        pre_context = self._pre_response_check(user_input)
        messages    = self._build_messages(user_input, pre_context)
        trace: list[str] = []

        def _collecting_step(step_type: str, content: str) -> None:
            trace.append(f"{step_type}: {content}")
            if on_step:
                on_step(step_type, content)

        if self.tools:
            response      = await self.provider.complete_with_tools(
                messages, self._options(), self.tools, on_step=_collecting_step,
            )
            full          = response.content
            meta          = await self._validate_response(user_input, full)
            self.last_meta = meta
            yield full
        else:
            full = ""
            async for chunk in self.provider.stream(messages, self._options()):
                full += chunk
                yield chunk
            meta           = await self._validate_response(user_input, full)
            self.last_meta = meta

        self._history.append(Message(role=Role.USER,      content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=full))
        await self._post_exchange(user_input, full, trace)

    def reset(self) -> None:
        self._history.clear()
