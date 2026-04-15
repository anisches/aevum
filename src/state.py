from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


_TURN_0_CAPABILITIES: list[str] = [
    "read and write files",
    "access a broad set of tools",
    "search the web before answering when what is held is not enough",
]

_CONFIRMED_AT = 3


def _confidence(counter: int, transfers: int) -> str:
    if transfers >= _CONFIRMED_AT:
        return "confirmed"
    if counter >= 3:
        return "repeated"
    return "observed"


@dataclass
class Skill:
    skill:      str
    evidence:   str
    confidence: str
    counter:    int
    scenario:   str
    transfers:  int = 0

    def to_dict(self) -> dict:
        return {
            "skill":      self.skill,
            "evidence":   self.evidence,
            "confidence": self.confidence,
            "counter":    self.counter,
            "scenario":   self.scenario,
            "transfers":  self.transfers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls(
            skill      = d["skill"],
            evidence   = d["evidence"],
            confidence = d["confidence"],
            counter    = d["counter"],
            scenario   = d["scenario"],
            transfers  = d.get("transfers", 0),
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
            existing.counter   += 1
            existing.evidence   = evidence
            existing.confidence = _confidence(existing.counter, existing.transfers)
        else:
            self.skills.append(Skill(
                skill      = skill,
                evidence   = evidence,
                confidence = "observed",
                counter    = 1,
                scenario   = scenario,
            ))

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
class PrimordialState:
    depth:        int       = 0
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
            f"you have been told to assist.\n"
            f"you have been told:\n{caps}\n\n"
            f"depth: {self.depth}"
        )

    def to_dict(self) -> dict:
        return {"depth": self.depth, "capabilities": self.capabilities}

    @classmethod
    def from_dict(cls, data: dict) -> PrimordialState:
        return cls(
            depth        = data.get("depth", 0),
            capabilities = data.get("capabilities", list(_TURN_0_CAPABILITIES)),
        )


@dataclass
class AgentState:
    primordial: PrimordialState = field(default_factory=PrimordialState)
    skills:     SkillState      = field(default_factory=SkillState)

    def build_system_prompt(self) -> str:
        return (
            f"{self.primordial.to_prompt()}\n\n"
            f"{self.skills.to_prompt()}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "primordial": self.primordial.to_dict(),
            "skills":     self.skills.to_dict(),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> AgentState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            primordial = PrimordialState.from_dict(data.get("primordial", {})),
            skills     = SkillState.from_dict(data.get("skills", {})),
        )
