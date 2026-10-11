"""Serializable models. All artifact paths are relative to the scan root."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any


def stable_id(kind: str, value: str) -> str:
    return kind + "_" + sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class Artifact:
    id: str
    path: str
    kind: str
    size: int
    modified: float
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Experiment:
    id: str
    name: str
    path: str
    confidence: float
    evidence: list[str]
    kind: str = "experiment"
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    subtree_checkpoint_count: int = 0
    subtree_checkpoint_bytes: int = 0
    roles: dict[str, str] = field(default_factory=dict)
    role_sources: dict[str, list[dict]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    config_sources: dict[str, list[dict]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_sources: dict[str, str] = field(default_factory=dict)
    references: list[dict] = field(default_factory=list)
    script_hashes: list[str] = field(default_factory=list)
    script_sources: dict[str, list[str]] = field(default_factory=dict)
    checkpoint_count: int = 0
    checkpoint_bytes: int = 0
    modified: float = 0
    started: float | None = None
    finished: float | None = None
    time_sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Edge:
    id: str
    source: str
    target: str
    kind: str
    directed: bool
    confidence: float
    evidence: list[dict]
    caveats: list[str] = field(default_factory=list)
    changes: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Report:
    schema_version: str
    tool_version: str
    root: str
    generated_at: str
    options: dict
    statistics: dict
    experiments: list[Experiment]
    edges: list[Edge]
    artifacts: list[Artifact]
    warnings: list[dict]
    assumptions: list[str]

    def to_dict(self):
        return asdict(self)
