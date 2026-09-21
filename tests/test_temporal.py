"""Round 2 conflict/temporal reasoning tests."""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_graphrag.temporal.facts import FactAssertion, TemporalFactStore  # noqa: E402


def _store():
    s = TemporalFactStore()
    return s


def test_supersession_prefers_later_higher_authority():
    s = _store()
    s.add(FactAssertion("E", "date", "10 Aug", source_type="news", asserted_at=date(2023, 1, 1)))
    s.add(FactAssertion("E", "date", "17 Aug", source_type="official", asserted_at=date(2024, 6, 1)))
    r = s.resolve("E", "date")
    assert r.value == "17 Aug"
    assert r.superseded  # older claim recorded as superseded


def test_authority_breaks_ties():
    s = _store()
    s.add(FactAssertion("R", "holder", "A", source_type="blog", asserted_at=date(2024, 5, 1)))
    s.add(FactAssertion("R", "holder", "B", source_type="official", asserted_at=date(2024, 5, 1)))
    r = s.resolve("R", "holder")
    assert r.value == "B"
    assert r.confidence == 1.0


def test_uncertainty_flagged_on_close_conflict():
    s = _store()
    s.add(FactAssertion("F", "count", "5", source_type="news", asserted_at=date(2024, 3, 1)))
    s.add(FactAssertion("F", "count", "6", source_type="news", asserted_at=date(2024, 3, 2)))
    r = s.resolve("F", "count")
    assert r.confidence <= 0.6  # uncertain
    assert set(f.object for f in r.conflicts) == {"5"} or {"6"}


def test_as_of_temporal_window():
    s = _store()
    s.add(FactAssertion("T", "cap", "P1", source_type="official",
                        valid_from=date(2020, 1, 1), valid_to=date(2022, 12, 31),
                        asserted_at=date(2020, 1, 1)))
    s.add(FactAssertion("T", "cap", "P2", source_type="official",
                        valid_from=date(2023, 1, 1), asserted_at=date(2023, 1, 1)))
    assert s.resolve("T", "cap", as_of=date(2021, 6, 1)).value == "P1"
    assert s.resolve("T", "cap", as_of=date(2024, 6, 1)).value == "P2"


def test_conflict_tool_via_agent():
    from agentic_graphrag.agents.tools import Tools
    from agentic_graphrag.agents.state import AgentState
    from agentic_graphrag.graph.entity_linking import ParsedQuestion

    s = _store()
    s.add(FactAssertion("E", "date", "10 Aug", source_type="news", asserted_at=date(2023, 1, 1)))
    s.add(FactAssertion("E", "date", "17 Aug", source_type="official", asserted_at=date(2024, 6, 1)))
    tools = Tools(graph=None, vector_store=None, fact_store=s)
    state = AgentState(question="q", parsed=ParsedQuestion(raw="q"))
    tr = tools.conflict_resolution(state, subject="E", predicate="date")
    assert tr.proposed_answer == "17 Aug"
    assert tr.evidence
