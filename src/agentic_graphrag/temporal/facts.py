"""Bi-temporal fact store with conflict resolution and uncertainty.

Round 2 requires the agent to reason over facts that change and disagree:
detect conflicting versions of a fact, decide what supersedes what, weigh
source authority, and express uncertainty. Real-world facts (launch dates,
records, rosters) get revised, and sources conflict.

Model
-----
A `FactAssertion` is a claim that (subject, predicate) == object, made by a
`source` at an `asserted_at` time, optionally valid over a window. Sources have
an authority weight. Multiple assertions about the same (subject, predicate)
form a conflict set. The `resolve` policy chooses the surviving value by:

  1. Supersession: a later assertion supersedes an earlier one from the same
     or lower-authority source (facts get corrected over time).
  2. Authority: when timestamps tie or are absent, higher authority wins.
  3. Uncertainty: if the top two candidates are close in combined weight, the
     result is flagged low-confidence and both are reported.

This is source-agnostic and plugs into the agent as a `conflict_resolution`
tool; it also underpins temporal "as of <date>" queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# Default authority weights by source type; higher is more trusted.
DEFAULT_AUTHORITY = {
    "official": 1.0,
    "wikipedia": 0.8,
    "news": 0.6,
    "blog": 0.3,
    "social": 0.15,
    "unknown": 0.2,
}


@dataclass
class FactAssertion:
    subject: str
    predicate: str
    object: str
    source: str = "unknown"          # source identifier
    source_type: str = "unknown"     # official|wikipedia|news|blog|social
    asserted_at: Optional[date] = None   # when the claim was made/published
    valid_from: Optional[date] = None    # when the fact became true
    valid_to: Optional[date] = None      # when it stopped being true
    doc_id: str = ""

    def authority(self) -> float:
        return DEFAULT_AUTHORITY.get(self.source_type, DEFAULT_AUTHORITY["unknown"])


@dataclass
class Resolution:
    subject: str
    predicate: str
    value: str
    confidence: float
    winning: FactAssertion
    superseded: list[FactAssertion] = field(default_factory=list)
    conflicts: list[FactAssertion] = field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "winning_source": f"{self.winning.source} ({self.winning.source_type})",
            "num_conflicts": len(self.conflicts),
            "num_superseded": len(self.superseded),
            "rationale": self.rationale,
        }


class TemporalFactStore:
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], list[FactAssertion]] = {}

    def add(self, fact: FactAssertion) -> None:
        self._facts.setdefault((fact.subject, fact.predicate), []).append(fact)

    def assertions(self, subject: str, predicate: str) -> list[FactAssertion]:
        return list(self._facts.get((subject, predicate), []))

    # ------------------------------------------------------------------ #
    def resolve(
        self, subject: str, predicate: str, as_of: Optional[date] = None
    ) -> Optional[Resolution]:
        """Resolve the current (or as-of) value among conflicting assertions."""
        facts = self._facts.get((subject, predicate), [])
        if not facts:
            return None

        # Temporal filter: keep facts valid at `as_of` (if a window is set).
        if as_of is not None:
            def valid(f: FactAssertion) -> bool:
                if f.valid_from and as_of < f.valid_from:
                    return False
                if f.valid_to and as_of > f.valid_to:
                    return False
                # also ignore assertions made after the as-of moment
                if f.asserted_at and f.asserted_at > as_of:
                    return False
                return True

            candidates = [f for f in facts if valid(f)] or facts
        else:
            candidates = facts

        # Group by object value; each value gets a combined score.
        def score(f: FactAssertion) -> float:
            # recency component (newer asserted_at scores higher) + authority
            recency = 0.0
            if f.asserted_at:
                recency = f.asserted_at.toordinal() / 1_000_000.0  # small tie-breaker weight
            return f.authority() + recency

        ranked = sorted(candidates, key=score, reverse=True)
        winner = ranked[0]

        # Superseded = same-or-lower authority, earlier assertions with a
        # different value than the winner.
        superseded = [
            f for f in ranked[1:]
            if f.object != winner.object
            and (f.asserted_at and winner.asserted_at and f.asserted_at < winner.asserted_at)
        ]
        conflicts = [f for f in candidates if f.object != winner.object]

        # Confidence: gap between winner and best conflicting value.
        best_conflict_score = max(
            (score(f) for f in candidates if f.object != winner.object), default=0.0
        )
        win_score = score(winner)
        gap = win_score - best_conflict_score
        # Normalize gap into (0,1]; small gap => low confidence (uncertainty).
        confidence = min(1.0, 0.5 + gap) if conflicts else 1.0

        rationale = _explain(winner, conflicts, superseded, as_of)
        return Resolution(
            subject=subject, predicate=predicate, value=winner.object,
            confidence=confidence, winning=winner,
            superseded=superseded, conflicts=conflicts, rationale=rationale,
        )


def _explain(
    winner: FactAssertion,
    conflicts: list[FactAssertion],
    superseded: list[FactAssertion],
    as_of: Optional[date],
) -> str:
    bits = []
    if as_of:
        bits.append(f"as of {as_of.isoformat()}")
    if not conflicts:
        bits.append(f"single uncontested value from {winner.source_type} source")
    else:
        bits.append(
            f"chose '{winner.object}' from {winner.source} "
            f"({winner.source_type}, authority={winner.authority():.2f}"
            + (f", asserted {winner.asserted_at.isoformat()}" if winner.asserted_at else "")
            + ")"
        )
        if superseded:
            older = ", ".join(
                f"'{f.object}'({f.asserted_at.isoformat() if f.asserted_at else '?'})"
                for f in superseded
            )
            bits.append(f"superseding older claim(s): {older}")
        other = ", ".join(
            f"'{f.object}'<{f.source_type}>" for f in conflicts if f not in superseded
        )
        if other:
            bits.append(f"over competing claim(s): {other}")
    return "; ".join(bits)
