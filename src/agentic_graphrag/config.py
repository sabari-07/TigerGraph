"""Central configuration, loaded from environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv optional at runtime
    pass


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "mock"))
    model: str = field(default_factory=lambda: _get("LLM_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: _get("LLM_JUDGE_MODEL", "gpt-4o-mini"))
    api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: _get("OPENAI_BASE_URL"))


@dataclass
class EmbeddingConfig:
    provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "mock"))
    model: str = field(
        default_factory=lambda: _get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dim: int = field(default_factory=lambda: int(_get("EMBEDDING_DIM", "384")))


@dataclass
class TigerGraphConfig:
    host: str = field(default_factory=lambda: _get("TG_HOST"))
    graph_name: str = field(default_factory=lambda: _get("TG_GRAPH_NAME", "AgenticGraphRAG"))
    username: str = field(default_factory=lambda: _get("TG_USERNAME"))
    password: str = field(default_factory=lambda: _get("TG_PASSWORD"))
    secret: str = field(default_factory=lambda: _get("TG_SECRET"))
    gs_port: int = field(default_factory=lambda: int(_get("TG_GS_PORT", "14240")))
    rest_port: int = field(default_factory=lambda: int(_get("TG_REST_PORT", "9000")))


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_get("DATA_DIR", ".")))
    artifacts_dir: Path = field(default_factory=lambda: Path(_get("ARTIFACTS_DIR", "artifacts")))
    graph_backend: str = field(default_factory=lambda: _get("GRAPH_BACKEND", "local"))
    # Agent planner: 'rule' (fast deterministic policy) or 'llm' (LLM decides
    # the next tool each step - genuinely agentic, robust to novel phrasing).
    planner: str = field(default_factory=lambda: _get("PLANNER", "rule"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tigergraph: TigerGraphConfig = field(default_factory=TigerGraphConfig)

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus" / "corpus.jsonl"

    @property
    def public_questions_path(self) -> Path:
        return self.data_dir / "questions" / "eval_public.jsonl"

    @property
    def hidden_questions_path(self) -> Path:
        return self.data_dir / "questions" / "eval_hidden.jsonl"

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
