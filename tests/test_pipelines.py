"""Regression tests locking in the core behaviors and the three-way story.

These run offline (mock LLM + hashing embedder) so CI needs no API keys.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_graphrag.config import EmbeddingConfig, LLMConfig  # noqa: E402
from agentic_graphrag.data.loader import load_documents, load_questions  # noqa: E402
from agentic_graphrag.graph.builder import build_graph  # noqa: E402
from agentic_graphrag.graph.entity_linking import parse_question  # noqa: E402
from agentic_graphrag.llm import LLMClient  # noqa: E402
from agentic_graphrag.retrieval.embeddings import build_embedder  # noqa: E402
from agentic_graphrag.retrieval.vector_store import VectorStore  # noqa: E402
from agentic_graphrag.pipelines.graphrag import GraphRAGPipeline  # noqa: E402
from agentic_graphrag.pipelines.agentic import AgenticPipeline  # noqa: E402
from agentic_graphrag.benchmark.metrics import answer_hit  # noqa: E402


@pytest.fixture(scope="module")
def env():
    docs = load_documents(ROOT / "corpus" / "corpus.jsonl")
    graph = build_graph(docs)
    emb = build_embedder(EmbeddingConfig(provider="mock", dim=512))
    vs = VectorStore(emb).build(docs)
    llm = LLMClient(LLMConfig(provider="mock"))
    qs = load_questions(ROOT / "questions" / "eval_public.jsonl")
    return {"graph": graph, "vs": vs, "llm": llm, "qs": qs}


def test_graph_built(env):
    stats = env["graph"].stats()
    assert stats.get("Event") == 2187
    assert stats["_edges"] > 15000


def test_parser_intents_match_qtype(env):
    ok = sum(parse_question(q.question, env["graph"]).intent == q.qtype.value for q in env["qs"])
    assert ok == len(env["qs"])  # 100/100


def test_graphrag_accuracy_at_least_90(env):
    pipe = GraphRAGPipeline(env["graph"], env["vs"], env["llm"], top_k=3)
    correct = sum(answer_hit(pipe.answer(q).answer, q.answer) for q in env["qs"])
    assert correct >= 90


def test_agentic_beats_graphrag(env):
    gr = GraphRAGPipeline(env["graph"], env["vs"], env["llm"], top_k=3)
    ag = AgenticPipeline(env["graph"], env["vs"], env["llm"], max_steps=8)
    gr_correct = sum(answer_hit(gr.answer(q).answer, q.answer) for q in env["qs"])
    ag_correct = sum(answer_hit(ag.answer(q).answer, q.answer) for q in env["qs"])
    assert ag_correct >= gr_correct
    assert ag_correct >= 98


def test_agentic_uses_fewer_tokens_than_rag_on_average(env):
    """The efficiency claim: agentic answers cost less than naive RAG."""
    ag = AgenticPipeline(env["graph"], env["vs"], env["llm"], max_steps=8)
    sample = env["qs"][:20]
    ag_tokens = sum(ag.answer(q).tokens.total for q in sample) / len(sample)
    assert ag_tokens < 846  # RAG baseline avg tokens/q


def test_multi_hop_disambiguation_recovers_pub038(env):
    ag = AgenticPipeline(env["graph"], env["vs"], env["llm"], max_steps=8)
    q = next(q for q in env["qs"] if q.qid == "pub-038")
    r = ag.answer(q)
    assert answer_hit(r.answer, q.answer)
    assert r.strategy_changed
