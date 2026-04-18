import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
from .state import AgentState, Breadcrumb, Episode, Skill

_DEFAULT_STATE_PATH = Path.home() / ".aevum" / "state.json"

_ABSTRACT_OBS_PROMPT = """\
intent: {intent}
tool: {tool}
raw result:
{raw}

Extract only what is relevant to the intent above.
Strip URLs, boilerplate, and noise.
Reply in 1-3 concise sentences — signal only."""

_CLASSIFY_PROMPT = """\
Message: {user_input}

Classify and analyze this message. All three fields are required — do not leave any blank.
Reply in this exact format:
type: query | instruction | response
intent: <one sentence — what the user actually wants>
required: <what is needed to fully address this>"""

_THINK_PROMPT = """\
current date/time: {now}
type: {type}
intent: {intent}
required: {required}

Available tools:
{tools}

Observations so far:
{observations}

What is the next action?
To use a tool reply:
action: <tool_name>
args: <json object>

If the goal is met reply: done
If you cannot proceed reply: stuck: <reason>"""

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

_GAPS_PROMPT = """\
Session reflection:
trigger: {trigger}
trajectory: {trajectory}
outcome: {outcome}
reflection: {reflection}

What 1-2 specific topics should you search to be better prepared for similar tasks?
Focus on: knowledge gaps, things you had to guess, concepts that came up but weren't mastered.

Reply in this exact format (repeat block for each topic):
topic: <specific search query>
reason: <one sentence — why this matters>

If nothing worth searching, reply: none"""

_DISTILL_PROMPT = """\
Topic: {topic}
Reason to learn: {reason}
Search results:
{results}

Distill this into a reusable knowledge nugget — something that will help in future sessions.
Be specific, factual, and concise.

Reply in this exact format:
summary: <2-3 sentences of distilled knowledge>
tags: <3-5 comma-separated keywords>"""

_REFLECT_PROMPT = """\
Session turns:
{turns}

Reflect on this session from first person. Be honest and direct.

Reply in this exact format:
trigger: <the core ask or project in one sentence>
trajectory: <2-3 sentences: what was attempted, what tools/steps were used, any pivots>
outcome: completed | stuck | partial
reflection: <one honest sentence: what happened, what you'd do differently, or what worked>"""

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
    source: str
    reason: str
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
        source=source or "llm",
        reason=reason or "",
        transfer=transfer or "no",
    )


def _parse_classify(text: str, fallback_input: str = "") -> tuple[str, str, str]:
    type_ = intent = required = ""
    for line in text.splitlines():
        l = line.lower().strip()
        if l.startswith("type:"):
            type_ = line.split(":", 1)[1].strip()
        elif l.startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
        elif l.startswith("required:"):
            required = line.split(":", 1)[1].strip()
    return (
        type_ or "query",
        intent or fallback_input,
        required or "address the message directly",
    )


def _parse_think(text: str) -> tuple[str, dict]:
    import re as _re
    text = text.strip()
    low = text.lower()
    if low == "done" or low.startswith("done\n"):
        return "done", {}
    if low.startswith("stuck"):
        reason = text.split(":", 1)[1].strip() if ":" in text else text
        return "stuck", {"reason": reason}

    action = ""
    args_lines: list[str] = []
    collecting_args = False

    for line in text.splitlines():
        l = line.lower().strip()
        if l.startswith("action:"):
            action = line.split(":", 1)[1].strip()
            collecting_args = False
        elif l.startswith("args:"):
            rest = line.split(":", 1)[1].strip()
            args_lines = [rest] if rest else []
            collecting_args = True
        elif collecting_args and line.strip() and not l.startswith("action:"):
            args_lines.append(line)

    if action:
        args_str = " ".join(args_lines).strip()
        # strip markdown fences if the LLM wrapped the JSON
        args_str = _re.sub(r"```\w*\s*", "", args_str).strip()
        args: dict = {}
        if args_str:
            try:
                args = json.loads(args_str)
            except Exception:
                # try to find any {...} block in the string
                m = _re.search(r"\{[^}]+\}", args_str)
                if m:
                    try:
                        args = json.loads(m.group())
                    except Exception:
                        pass
        return "act", {"tool": action, "args": args}
    return "done", {}


def _relevant_skills(user_input: str, skills: list[Skill]) -> list[Skill]:
    words = set(user_input.lower().split())
    out = []
    for s in skills:
        scenario_words = set(s.scenario.lower().split())
        skill_words = set(s.skill.lower().split())
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
        self.config = config or Config.from_env()
        self.provider: BaseProvider = _build_provider(self.config)
        self.tools: dict[str, BaseTool] = {}
        self._history: list[Message] = []
        self._state_path = state_path or _DEFAULT_STATE_PATH
        self._state = AgentState.load(self._state_path)
        self.last_meta: ResponseMeta | None = None
        self._session_turns: list[dict] = []

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
        lines: list[str] = [f"current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

        relevant = _relevant_skills(user_input, self._state.skills.skills)
        if relevant:
            lines.append("relevant skills for this question:")
            for s in relevant:
                lines.append(f"  - {s.skill} [{s.confidence}]: {s.scenario}")
        else:
            lines.append("no validated skills map to this question")

        lines.append(
            f"depth: {self._state.primordial.depth} exchanges have shaped this response"
        )

        prior = _prior_encounter(user_input, self._history)
        if prior:
            lines.append(
                f'prior encounter: yes — similar question seen before: "{prior}"'
            )
        else:
            lines.append("prior encounter: no")

        return "\n".join(lines)

    def _build_messages(self, user_input: str, pre_context: str = "") -> list[Message]:
        system = self._state.build_system_prompt(user_input)
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

    async def _tao_loop(
        self,
        user_input: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> tuple[str, list[str], str, str]:
        classify_raw = await self._llm(
            _CLASSIFY_PROMPT.format(user_input=user_input), max_tokens=150
        )
        type_, intent, required = _parse_classify(classify_raw, fallback_input=user_input)
        if on_step:
            on_step("thought", f"[{type_}] {intent} | required: {required}")

        trace: list[str] = [f"classify: {type_} — {intent}"]
        observations: list[str] = []

        def _fmt_tool(name: str, tool: "BaseTool") -> str:
            props = tool.schema().get("parameters", {}).get("properties", {})
            sig = ", ".join(
                f"{k}: {v.get('description', v.get('type', 'str'))}"
                for k, v in props.items()
            )
            return f"  {name}({sig}): {tool.description}"

        tool_desc = "\n".join(_fmt_tool(n, t) for n, t in self.tools.items())

        for _ in range(10):
            obs_text = (
                "\n".join(f"  {i+1}. {o}" for i, o in enumerate(observations))
                or "  none yet"
            )
            think_raw = await self._llm(
                _THINK_PROMPT.format(
                    now=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    type=type_,
                    intent=intent,
                    required=required,
                    tools=tool_desc,
                    observations=obs_text,
                ),
                max_tokens=150,
            )
            status, payload = _parse_think(think_raw)

            if status == "done":
                if on_step:
                    on_step("thought", "goal reached")
                trace.append("think: done")
                break
            elif status == "stuck":
                reason = payload.get("reason", "")
                if on_step:
                    on_step("thought", f"stuck: {reason}")
                trace.append(f"think: stuck — {reason}")
                break
            else:
                tool_name = payload.get("tool", "")
                args = payload.get("args", {})
                args_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
                if on_step:
                    on_step("action", f"{tool_name}({args_preview})")
                trace.append(f"action: {tool_name}({args_preview})")

                tool = self.tools.get(tool_name)
                if tool:
                    tool_result = await tool.run(**args)
                    if tool_result.error:
                        raw = f"error: {tool_result.error}"
                    else:
                        raw = str(tool_result.output)
                else:
                    raw = f"unknown tool: {tool_name}"

                # post-obs abstraction: compress raw result to signal
                if not raw.startswith("error:") and len(raw) > 80:
                    abstracted = await self._llm(
                        _ABSTRACT_OBS_PROMPT.format(
                            intent=intent,
                            tool=tool_name,
                            raw=raw[:1200],
                        ),
                        max_tokens=120,
                    )
                else:
                    abstracted = raw

                observations.append(f"{tool_name}: {abstracted}")
                if on_step:
                    on_step("observation", abstracted[:120])
                trace.append(f"observation: {abstracted[:120]}")

        obs_summary = "\n".join(observations) if observations else "no tool use"
        pre_context = self._pre_response_check(user_input)
        messages = self._build_messages(
            user_input,
            pre_context + f"\n\nresearch gathered:\n{obs_summary}",
        )
        response = await self.provider.complete(messages, self._options())
        return response.content, trace, type_, intent

    async def _validate_response(self, user_input: str, response: str) -> ResponseMeta:
        if not self._state.skills.skills:
            return ResponseMeta(
                source="llm", reason="no skill state yet", transfer="no"
            )

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
        depth = self._state.primordial.depth
        trace_text = (
            "\n".join(f"  {s}" for s in trace)
            if trace
            else "  (direct response — no tool use)"
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
                f"  - {s.skill}: {s.scenario}" for s in self._state.skills.skills
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
        self, user_input: str, on_step: Callable[[str, str], None] | None = None
    ) -> tuple[CompletionResponse, ResponseMeta, list[str]]:
        content, trace, type_, intent = await self._tao_loop(user_input, on_step=on_step)
        response = CompletionResponse(content=content, model=self.config.model)
        meta = await self._validate_response(user_input, content)
        outcome = (
            "stuck" if any(t.startswith("think: stuck") for t in trace)
            else "completed" if any(t == "think: done" for t in trace)
            else "partial"
        )
        self._session_turns.append({
            "user_input": user_input,
            "intent": intent,
            "type": type_,
            "trace": trace,
            "outcome": outcome,
        })
        return response, meta, trace

    async def chat(self, user_input: str) -> CompletionResponse:
        response, meta, trace = await self._full_pipeline(user_input, on_step=None)
        self.last_meta = meta
        self._history.append(Message(role=Role.USER, content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=response.content))
        await self._post_exchange(user_input, response.content, trace)
        return response

    async def stream(
        self,
        user_input: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AsyncIterator[str]:
        response, meta, trace = await self._full_pipeline(user_input, on_step=on_step)
        self.last_meta = meta
        self._history.append(Message(role=Role.USER, content=user_input))
        self._history.append(Message(role=Role.ASSISTANT, content=response.content))
        await self._post_exchange(user_input, response.content, trace)
        yield response.content

    async def _learn_from_episode(self, episode: Episode) -> list[Breadcrumb]:
        raw = await self._llm(
            _GAPS_PROMPT.format(
                trigger=episode.trigger,
                trajectory=episode.trajectory,
                outcome=episode.outcome,
                reflection=episode.reflection,
            ),
            max_tokens=150,
        )
        if raw.strip().lower().startswith("none"):
            return []

        gaps: list[tuple[str, str]] = []
        topic = reason = ""
        for line in raw.splitlines():
            l = line.lower().strip()
            if l.startswith("topic:"):
                if topic:
                    gaps.append((topic, reason or "no reason given"))
                topic = line.split(":", 1)[1].strip()
                reason = ""
            elif l.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        if topic:
            gaps.append((topic, reason or "no reason given"))

        breadcrumbs: list[Breadcrumb] = []
        web_tool = self.tools.get("web_search")
        for topic, reason in gaps[:2]:
            if web_tool:
                result = await web_tool.run(query=topic)
                raw_results = str(result.output) if not result.error else ""
            else:
                raw_results = "no search tool available"

            distilled = await self._llm(
                _DISTILL_PROMPT.format(
                    topic=topic,
                    reason=reason,
                    results=raw_results[:1500],
                ),
                max_tokens=150,
            )

            summary = tags_raw = ""
            for line in distilled.splitlines():
                l = line.lower().strip()
                if l.startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif l.startswith("tags:"):
                    tags_raw = line.split(":", 1)[1].strip()

            if summary:
                crumb = Breadcrumb(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    topic=topic,
                    summary=summary,
                    tags=[t.strip() for t in tags_raw.split(",") if t.strip()],
                )
                self._state.knowledge.record(crumb)
                breadcrumbs.append(crumb)

        return breadcrumbs

    async def end_session(self) -> Episode | None:
        if not self._session_turns:
            return None

        turns_text = "\n\n".join(
            f"turn {i+1}:\n"
            f"  user: {t['user_input'][:120]}\n"
            f"  intent: {t['intent'][:100]}\n"
            f"  trace: {'; '.join(t['trace'][:5])}\n"
            f"  outcome: {t['outcome']}"
            for i, t in enumerate(self._session_turns[-8:])
        )

        raw = await self._llm(_REFLECT_PROMPT.format(turns=turns_text), max_tokens=400)

        trigger = trajectory = reflection = ""
        outcome = "partial"
        for line in raw.splitlines():
            l = line.lower().strip()
            if l.startswith("trigger:"):
                trigger = line.split(":", 1)[1].strip()
            elif l.startswith("trajectory:"):
                trajectory = line.split(":", 1)[1].strip()
            elif l.startswith("outcome:"):
                val = line.split(":", 1)[1].strip().lower()
                if val in ("completed", "stuck", "partial"):
                    outcome = val
            elif l.startswith("reflection:"):
                reflection = line.split(":", 1)[1].strip()

        episode = Episode(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            trigger=trigger or self._session_turns[0]["user_input"][:80],
            trajectory=trajectory or "no trajectory captured",
            outcome=outcome,
            reflection=reflection or "no reflection generated",
        )
        self._state.episodes.record(episode)
        await self._learn_from_episode(episode)
        self._state.save(self._state_path)
        self._session_turns.clear()
        self._history.clear()
        return episode

    def reset(self) -> None:
        self._history.clear()
        self._session_turns.clear()
