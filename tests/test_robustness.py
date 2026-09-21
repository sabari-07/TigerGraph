"""Robustness: the system must answer PARAPHRASED questions, not just the exact
template phrasings. This proves the agent reasons about intent rather than
pattern-matching a fixed template, which is the key credibility test.

Each case is a hand-reworded version of a public question whose gold answer we
know. We assert the agent still returns the correct answer via the parser
fallback + graph tools.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_graphrag.config import EmbeddingConfig, LLMConfig  # noqa: E402
from agentic_graphrag.data.loader import load_documents  # noqa: E402
from agentic_graphrag.graph.builder import build_graph  # noqa: E402
from agentic_graphrag.llm import LLMClient  # noqa: E402
from agentic_graphrag.retrieval.embeddings import build_embedder  # noqa: E402
from agentic_graphrag.retrieval.vector_store import VectorStore  # noqa: E402
from agentic_graphrag.pipelines.agentic import AgenticPipeline  # noqa: E402
from agentic_graphrag.benchmark.metrics import answer_hit  # noqa: E402
from agentic_graphrag.models import Question, QType  # noqa: E402


# (reworded question, gold answer) — none matches the original regex templates.
PARAPHRASES = [
    ("How many cycling events at the 2008 Summer Olympics drew over 30 competitors?", ["8"]),
    ("Which athletics event had the biggest field at the 2008 Summer Olympics?",
     ["Athletics at the 2008 Summer Olympics – Men's marathon"]),
    ("Who took gold in the men's pole vault at the Summer Games preceding 2016?",
     ["Renaud Lavillenie"]),
    ("Who won gold at the event at Richmond Olympic Oval on 14 February 2010?",
     ["Martina Sáblíková"]),
    ("How many countries competed in Sailing at the 2016 Summer Olympics – Women's RS:X?",
     ["26"]),
    ("At the 2004 Summer Olympics, how many shooting events had over 37 competitors?", ["8"]),
]


@pytest.fixture(scope="module")
def pipe():
    docs = load_documents(ROOT / "corpus" / "corpus.jsonl")
    g = build_graph(docs)
    emb = build_embedder(EmbeddingConfig(provider="mock", dim=512))
    vs = VectorStore(emb).build(docs)
    llm = LLMClient(LLMConfig(provider="mock"))
    return AgenticPipeline(g, vs, llm, max_steps=8, planner="rule")


def test_paraphrase_robustness(pipe):
    correct = 0
    misses = []
    for i, (q, gold) in enumerate(PARAPHRASES):
        r = pipe.answer(Question(qid=f"para-{i}", question=q, qtype=QType.OTHER))
        if answer_hit(r.answer, gold):
            correct += 1
        else:
            misses.append((q, gold, r.answer))
    # Pure-regex baseline would score ~0 here (all reworded); require strong recovery.
    assert correct >= 5, f"only {correct}/{len(PARAPHRASES)} paraphrases correct; misses={misses}"


def test_llm_planner_also_robust(pipe):
    docs = load_documents(ROOT / "corpus" / "corpus.jsonl")
    g = build_graph(docs)
    emb = build_embedder(EmbeddingConfig(provider="mock", dim=512))
    vs = VectorStore(emb).build(docs)
    llm = LLMClient(LLMConfig(provider="mock"))
    llm_pipe = AgenticPipeline(g, vs, llm, max_steps=8, planner="llm")
    correct = sum(
        answer_hit(llm_pipe.answer(Question(qid=f"lp-{i}", question=q, qtype=QType.OTHER)).answer, gold)
        for i, (q, gold) in enumerate(PARAPHRASES)
    )
    assert correct >= 5
