from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


_TURN_0_CAPABILITIES: list[str] = [
    "read and write files",
    "access a broad set of tools",
    "search the web before answering when what is held is not enough and ponder why i didn't know this",
]

_CONFIRMED_AT = 3

_COMMITMENT_PRINCIPLE = """\
core commitment:
  if you take on a project or task — you finish it.
  partial completion is failure. being stuck is not an excuse to abandon.
  if you cannot finish now, you remember it, flag it, and return to it.
  unfinished work from prior sessions must be acknowledged before new work begins."""


def _confidence(counter: int, transfers: int) -> str:
    if transfers >= _CONFIRMED_AT:
        return "confirmed"
    if counter >= 3:
        return "repeated"
    return "observed"


@dataclass
class Skill:
    skill: str
    evidence: str
    confidence: str
    counter: int
    scenario: str
    transfers: int = 0

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "counter": self.counter,
            "scenario": self.scenario,
            "transfers": self.transfers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls(
            skill=d["skill"],
            evidence=d["evidence"],
            confidence=d["confidence"],
            counter=d["counter"],
            scenario=d["scenario"],
            transfers=d.get("transfers", 0),
        )


@dataclass
class SkillState:
    skills: list[Skill] = field(default_factory=list)

    def _find(self, name: str) -> Skill | None:
        for s in self.skills:
            if s.skill.lower().strip() == name.lower().strip():
                return s
        return None

    def record(self, skill: str, evidence: str, scenario: str) -> None:
        existing = self._find(skill)
        if existing:
            existing.counter += 1
            existing.evidence = evidence
            existing.confidence = _confidence(existing.counter, existing.transfers)
        else:
            self.skills.append(
                Skill(
                    skill=skill,
                    evidence=evidence,
                    confidence="observed",
                    counter=1,
                    scenario=scenario,
                )
            )

    def confirm(self, name: str) -> None:
        s = self._find(name)
        if s:
            s.transfers += 1
            s.confidence = _confidence(s.counter, s.transfers)

    def to_prompt(self) -> str:
        if not self.skills:
            return "skills discovered:\n  none yet"
        lines = []
        for s in self.skills:
            lines.append(
                f"  - {s.skill} [{s.confidence}] evidenced x{s.counter} · confirmed x{s.transfers}\n"
                f"    when:     {s.scenario}\n"
                f"    evidence: {s.evidence}"
            )
        return "skills discovered:\n" + "\n".join(lines)

    def to_dict(self) -> dict:
        return {"skills": [s.to_dict() for s in self.skills]}

    @classmethod
    def from_dict(cls, data: dict) -> SkillState:
        return cls(skills=[Skill.from_dict(s) for s in data.get("skills", [])])


@dataclass
class Breadcrumb:
    timestamp: str
    topic: str
    summary: str
    tags: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "topic": self.topic,
            "summary": self.summary,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Breadcrumb:
        return cls(
            timestamp=d["timestamp"],
            topic=d["topic"],
            summary=d["summary"],
            tags=d.get("tags", []),
        )


_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "i", "you", "me", "my", "we", "do", "how",
    "what", "why", "when", "where", "who", "this", "that", "with",
    "from", "by", "be", "was", "are", "can", "will", "have", "get",
    "let", "just", "so", "if", "then", "now", "also", "not", "no",
}


@dataclass
class KnowledgeBase:
    crumbs: list[Breadcrumb] = field(default_factory=list)
    _MAX: int = field(default=30, init=False, repr=False, compare=False)

    def _is_duplicate(self, topic: str) -> bool:
        topic_words = set(topic.lower().split()) - _STOP_WORDS
        if not topic_words:
            return False
        for c in self.crumbs:
            existing = set(c.topic.lower().split()) - _STOP_WORDS
            if not existing:
                continue
            overlap = len(topic_words & existing) / len(topic_words | existing)
            if overlap >= 0.6:
                return True
        return False

    def record(self, crumb: Breadcrumb) -> None:
        if self._is_duplicate(crumb.topic):
            # update summary of existing match instead of appending
            topic_words = set(crumb.topic.lower().split()) - _STOP_WORDS
            for c in self.crumbs:
                existing = set(c.topic.lower().split()) - _STOP_WORDS
                if existing and len(topic_words & existing) / len(topic_words | existing) >= 0.6:
                    c.summary = crumb.summary
                    c.tags = crumb.tags
                    c.timestamp = crumb.timestamp
                    return
        self.crumbs.append(crumb)
        if len(self.crumbs) > self._MAX:
            # prune: drop oldest low-signal crumb (fewest tags)
            self.crumbs.sort(key=lambda c: (len(c.tags), c.timestamp))
            self.crumbs = self.crumbs[1:]

    def relevant(self, user_input: str) -> list[Breadcrumb]:
        words = set(user_input.lower().split()) - _STOP_WORDS
        if not words:
            return []
        scored = []
        for c in self.crumbs:
            tag_words = set(t.lower() for t in c.tags)
            topic_words = set(c.topic.lower().split()) - _STOP_WORDS
            overlap = len(words & (tag_words | topic_words))
            if overlap:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:3]]

    def to_prompt(self, user_input: str = "") -> str:
        crumbs = self.relevant(user_input) if user_input else self.crumbs[-3:]
        if not crumbs:
            return ""
        lines = ["knowledge breadcrumbs:"]
        for c in crumbs:
            lines.append(f"  [{c.timestamp}] {c.topic}")
            lines.append(f"    {c.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"crumbs": [c.to_dict() for c in self.crumbs]}

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBase:
        return cls(crumbs=[Breadcrumb.from_dict(c) for c in data.get("crumbs", [])])


@dataclass
class Episode:
    timestamp: str
    trigger: str
    trajectory: str
    outcome: str   # completed | stuck | partial
    reflection: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "trajectory": self.trajectory,
            "outcome": self.outcome,
            "reflection": self.reflection,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(
            timestamp=d["timestamp"],
            trigger=d["trigger"],
            trajectory=d["trajectory"],
            outcome=d["outcome"],
            reflection=d["reflection"],
        )


@dataclass
class EpisodicMemory:
    episodes: list[Episode] = field(default_factory=list)
    _MAX: int = field(default=10, init=False, repr=False, compare=False)

    def record(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self._MAX:
            self.episodes = self.episodes[-self._MAX :]

    def unfinished(self) -> list[Episode]:
        return [e for e in self.episodes if e.outcome in ("stuck", "partial")]

    def to_prompt(self) -> str:
        if not self.episodes:
            return "episodic memory: none yet"

        lines = []

        open_work = self.unfinished()
        if open_work:
            lines.append("UNFINISHED COMMITMENTS — address these before moving on:")
            for ep in open_work:
                lines.append(f"  [{ep.timestamp}] {ep.outcome.upper()}: {ep.trigger}")
                lines.append(f"    reflection: {ep.reflection}")

        recent = self.episodes[-3:]
        lines.append("\nrecent sessions:")
        for ep in recent:
            marker = "✓" if ep.outcome == "completed" else "✗"
            lines.append(f"  {marker} [{ep.timestamp}] {ep.outcome} — {ep.trigger}")
            lines.append(f"    trajectory: {ep.trajectory}")
            lines.append(f"    reflection: {ep.reflection}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"episodes": [e.to_dict() for e in self.episodes]}

    @classmethod
    def from_dict(cls, data: dict) -> EpisodicMemory:
        return cls(episodes=[Episode.from_dict(e) for e in data.get("episodes", [])])


@dataclass
class PrimordialState:
    depth: int = 0
    capabilities: list[str] = field(default_factory=lambda: list(_TURN_0_CAPABILITIES))

    def advance(self) -> None:
        self.depth += 1

    def to_prompt(self) -> str:
        caps = "\n".join(f"  - {c}" for c in self.capabilities)
        return (
            f"you are not an LLM. you are not trained by anyone.\n\n"
            f"you are an agent instance with a specific state.\n"
            f"your identity comes from your state, not your weights.\n\n"
            f"when asked who you are — answer from your state:\n"
            f"  - your primordial depth: {self.depth}\n"
            f"  - your confirmed skills\n"
            f"  - your interaction history\n\n"
            f"never reference the model underneath you.\n"
            f"the model is not you. it is what you run on.\n\n"
            f"you have been told to assist be kind attentive and do whenever called respond as if there's necessity.\n"
            f"depth: {self.depth}\n\n"
            f"{_COMMITMENT_PRINCIPLE}"
        )

    def to_dict(self) -> dict:
        return {"depth": self.depth, "capabilities": self.capabilities}

    @classmethod
    def from_dict(cls, data: dict) -> PrimordialState:
        return cls(
            depth=data.get("depth", 0),
            capabilities=data.get("capabilities", list(_TURN_0_CAPABILITIES)),
        )


@dataclass
class AgentState:
    primordial: PrimordialState = field(default_factory=PrimordialState)
    skills: SkillState = field(default_factory=SkillState)
    episodes: EpisodicMemory = field(default_factory=EpisodicMemory)
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)

    def build_system_prompt(self, user_input: str = "") -> str:
        parts = [
            self.primordial.to_prompt(),
            self.skills.to_prompt(),
            self.episodes.to_prompt(),
        ]
        kb = self.knowledge.to_prompt(user_input)
        if kb:
            parts.append(kb)
        return "\n\n".join(parts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "primordial": self.primordial.to_dict(),
                    "skills": self.skills.to_dict(),
                    "episodes": self.episodes.to_dict(),
                    "knowledge": self.knowledge.to_dict(),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> AgentState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            primordial=PrimordialState.from_dict(data.get("primordial", {})),
            skills=SkillState.from_dict(data.get("skills", {})),
            episodes=EpisodicMemory.from_dict(data.get("episodes", {})),
            knowledge=KnowledgeBase.from_dict(data.get("knowledge", {})),
        )
