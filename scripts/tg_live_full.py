"""Full live run: TigerGraph graph + TigerGraph Vector DB, agent end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from agentic_graphrag.data.loader import load_documents, load_questions
from agentic_graphrag.graph.tigergraph_backend import TigerGraphBackend
from agentic_graphrag.llm import LLMClient
from agentic_graphrag.retrieval.embeddings import build_embedder
from agentic_graphrag.retrieval.vector_store import VectorStore
from agentic_graphrag.pipelines.agentic import AgenticPipeline
from agentic_graphrag.benchmark.metrics import answer_hit

cfg = load_config()
print("connecting to live TigerGraph...")
graph = TigerGraphBackend(cfg.tigergraph)
emb = build_embedder(cfg.embedding)
# Local vector store kept only as a fallback; TigerGraph Vector DB is primary.
vs = VectorStore(emb)  # empty; agent uses graph.vector_search via embedder
llm = LLMClient(cfg.llm)
pipe = AgenticPipeline(graph, vs, llm, max_steps=8, planner="rule", embedder=emb)

qs = load_questions(cfg.public_questions_path)
picks, seen = [], set()
for q in qs:
    if q.qtype.value not in seen:
        picks.append(q); seen.add(q.qtype.value)
    if len(seen) == 5:
        break

print("\n=== LIVE: TigerGraph graph + TigerGraph Vector DB ===")
for q in picks:
    r = pipe.answer(q)
    ok = answer_hit(r.answer, q.answer)
    tools = [t.agent for t in r.trace]
    print(f"[{'HIT' if ok else 'miss'}] {q.qtype.value:12} got={r.answer!r} "
          f"steps={r.num_steps} tools={tools}")
