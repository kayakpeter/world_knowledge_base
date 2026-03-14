# strategist/schema.py
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional


class NodeStatus(str, Enum):
    PENDING   = "pending"
    EXPANDED  = "expanded"
    PRUNED    = "pruned"
    LEAF      = "leaf"


@dataclass
class AttemptModel:
    """For events that may be attempted multiple times (missile salvos, drone attacks)."""
    p_attempt_72h: float
    p_success_per_attempt: float
    expected_attempts: int

    @property
    def p_cumulative_success(self) -> float:
        """P(at least one attempt succeeds in window)."""
        return 1.0 - (1.0 - self.p_success_per_attempt) ** self.expected_attempts

    def to_dict(self) -> dict:
        return {
            "p_attempt_72h": self.p_attempt_72h,
            "p_success_per_attempt": self.p_success_per_attempt,
            "expected_attempts": self.expected_attempts,
            "p_cumulative_success": self.p_cumulative_success,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttemptModel":
        return cls(
            p_attempt_72h=d["p_attempt_72h"],
            p_success_per_attempt=d["p_success_per_attempt"],
            expected_attempts=d["expected_attempts"],
        )


@dataclass
class SectorImpact:
    sector: str
    direction: str          # "UP", "DOWN", "VOLATILE", "NEUTRAL"
    magnitude: str          # "LOW", "MODERATE", "HIGH", "CRITICAL"
    magnitude_pct: Optional[float]
    tickers: list[str]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "magnitude_pct": self.magnitude_pct,
            "tickers": self.tickers,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SectorImpact":
        return cls(
            sector=d["sector"],
            direction=d["direction"],
            magnitude=d["magnitude"],
            magnitude_pct=d.get("magnitude_pct"),
            tickers=d.get("tickers", []),
            notes=d.get("notes", ""),
        )


@dataclass
class ScenarioNode:
    node_id: str
    description: str
    branch_probability: float
    parent_id: Optional[str]
    depth: int
    parent_joint_probability: float = 1.0
    status: NodeStatus = NodeStatus.PENDING
    attempt_model: Optional[AttemptModel] = None
    sector_impacts: list[SectorImpact] = field(default_factory=list)
    infrastructure_effects: list[dict] = field(default_factory=list)
    time_offset_hours: float = 0.0
    confidence: float = 0.7
    children: list[str] = field(default_factory=list)
    pruned_reason: str = ""

    @property
    def joint_probability(self) -> float:
        return self.parent_joint_probability * self.branch_probability

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "description": self.description,
            "branch_probability": self.branch_probability,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "parent_joint_probability": self.parent_joint_probability,
            "joint_probability": self.joint_probability,
            "status": self.status.value,
            "attempt_model": self.attempt_model.to_dict() if self.attempt_model else None,
            "sector_impacts": [s.to_dict() for s in self.sector_impacts],
            "infrastructure_effects": self.infrastructure_effects,
            "time_offset_hours": self.time_offset_hours,
            "confidence": self.confidence,
            "children": self.children,
            "pruned_reason": self.pruned_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioNode":
        node = cls(
            node_id=d["node_id"],
            description=d["description"],
            branch_probability=d["branch_probability"],
            parent_id=d.get("parent_id"),
            depth=d["depth"],
            parent_joint_probability=d.get("parent_joint_probability", 1.0),
            status=NodeStatus(d.get("status", "pending")),
            time_offset_hours=d.get("time_offset_hours", 0.0),
            confidence=d.get("confidence", 0.7),
            children=d.get("children", []),
            pruned_reason=d.get("pruned_reason", ""),
        )
        if d.get("attempt_model"):
            node.attempt_model = AttemptModel.from_dict(d["attempt_model"])
        node.sector_impacts = [SectorImpact.from_dict(s) for s in d.get("sector_impacts", [])]
        node.infrastructure_effects = d.get("infrastructure_effects", [])
        return node


@dataclass
class ScenarioTree:
    scenario_id: str
    trigger_event: str
    severity: str                           # CRITICAL | HIGH | MEDIUM | LOW
    confirmed: bool
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    window_hours: int = 72
    nodes: dict[str, ScenarioNode] = field(default_factory=dict)
    tombstones: list[dict] = field(default_factory=list)
    infrastructure_deps: list[str] = field(default_factory=list)
    status: str = "active"

    @property
    def expires_at(self) -> str:
        created = datetime.fromisoformat(self.created_at)
        return (created + timedelta(hours=self.window_hours)).isoformat()

    def add_node(self, node: ScenarioNode) -> None:
        """Insert node into tree. Parent MUST be added before children."""
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children:
                parent.children.append(node.node_id)

    def add_tombstone(self, node_id: str, description: str, joint_prob: float, reason: str) -> None:
        self.tombstones.append({
            "node_id": node_id,
            "description": description,
            "joint_probability": joint_prob,
            "pruned_reason": reason,
        })

    def node_count(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "trigger_event": self.trigger_event,
            "severity": self.severity,
            "confirmed": self.confirmed,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "window_hours": self.window_hours,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "tombstones": self.tombstones,
            "infrastructure_deps": self.infrastructure_deps,
            "status": self.status,
        }

    def save(self, base_dir) -> str:
        from pathlib import Path
        path = Path(base_dir) / "active" / f"{self.scenario_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return str(path)

    @classmethod
    def load(cls, path: str) -> "ScenarioTree":
        import pathlib
        d = json.loads(pathlib.Path(path).read_text())
        tree = cls(
            scenario_id=d["scenario_id"],
            trigger_event=d["trigger_event"],
            severity=d["severity"],
            confirmed=d["confirmed"],
            created_at=d["created_at"],
            window_hours=d.get("window_hours", 72),
            tombstones=d.get("tombstones", []),
            infrastructure_deps=d.get("infrastructure_deps", []),
            status=d.get("status", "active"),
        )
        for node_dict in d.get("nodes", {}).values():
            node = ScenarioNode.from_dict(node_dict)
            tree.nodes[node.node_id] = node
        return tree

    @classmethod
    def make_id(cls, trigger_event: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        slug = trigger_event[:30].lower().replace(" ", "-").replace("'", "")
        uid = uuid.uuid4().hex[:6]
        return f"{slug}-{ts}-{uid}"
