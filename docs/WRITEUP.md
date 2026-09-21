# Agentic GraphRAG — Write-up

## What we built

An Agentic GraphRAG system on TigerGraph that answers complex questions over an
Olympic-events corpus three ways — **RAG**, **GraphRAG**, and **Agentic
GraphRAG** — and benchmarks them side by side on accuracy, completeness, and
token efficiency. The system is built to answer the hackathon's core research
question: *when does autonomous, multi-step investigation actually beat a fixed
retrieval plan, and is it worth the token cost?*

## How it works

**Knowledge graph on TigerGraph.** We parse each event document's structured
infobox and its regular title (`<Sport> at the <Year> <Season> Olympics – <Event>`)
into a typed graph: `Event`, `Games`, `Venue`, `Athlete`, `NOC`, `Sport`,
connected by `PART_OF`, `HELD_AT`, `HAS_SPORT`, `WON_GOLD/SILVER/BRONZE`,
`REPRESENTS`, and a `NEXT/PREV` temporal chain between Games editions. From
2,951 documents we build 2,187 events, 20 games, 318 venues, 5,147 athletes,
and ~18k edges — loaded and running on **TigerGraph Savanna**. The same query
surface is also available as a local in-memory backend for offline dev/tests.

**Vector store on TigerGraph.** Embeddings live natively in TigerGraph too: a
384-dim vector attribute (`Event.emb`, `all-MiniLM-L6-v2`) added via schema
change, searched with the built-in `vectorSearch()` (HNSW index, cosine metric)
inside an installed GSQL query. So graph traversal and semantic search run in
the *same* database on the *same* connection — a genuine hybrid graph+vector
system, not a graph DB bolted to a separate vector DB.

**Three pipelines, one interface.** Every pipeline returns the same
`PipelineResult` (answer + evidence + trace + token usage), so the benchmark
scores them identically.

- **RAG** — embed the question, take top-k chunks, generate. One retrieval step.
- **GraphRAG** — parse the question into a structured intent, run the graph
  traversal that intent implies (count, extremum, temporal hop, venue+date hop,
  attribute lookup), then augment with vector-retrieved supporting text. A
  *fixed* graph→vector plan.
- **Agentic GraphRAG** — an orchestrator chooses the next specialized agent
  based on the question and the evidence gathered so far. It escalates only when
  needed: an ambiguous venue+date multi-hop triggers a **strategy change** to a
  date-disambiguation agent, then to multi-document retrieval, before answering.

**Specialized agents.** entity_linking, graph_traversal, similarity_search,
document_retrieval, aggregation, superlative, temporal, multi_hop,
disambiguate_by_date, conflict_resolution, evidence_evaluation.

**Explainability.** Every answer carries a full investigation trace (each step's
agent, rationale, evidence, and token cost) and citations that resolve to source
Wikipedia URLs, rendered as text and as an interactive HTML trace viewer.

## Key results

Benchmark on the 100 public questions, run on the **live stack**: TigerGraph
graph + TigerGraph Vector DB + Groq LLM (`openai/gpt-oss-20b`) + neural
embeddings (`all-MiniLM-L6-v2`).

| Pipeline | Accuracy | Completeness | Tokens/q | Steps/q | Accuracy per 1k tokens |
|---|---|---|---|---|---|
| RAG | 29% | 58.1% | 611 | 2.00 | 0.48 |
| GraphRAG | 96% | 78.7% | 794 | 4.00 | 1.21 |
| **Agentic GraphRAG** | **98%** | 67.5% | **598** | 3.14 | **1.64** |

Accuracy by question type makes the thesis concrete:

| qtype | RAG | GraphRAG | Agentic |
|---|---|---|---|
| lookup | 19/19 | 18/19 | 19/19 |
| superlative | 6/10 | 10/10 | 10/10 |
| aggregation | 4/21 | 21/21 | 21/21 |
| multi_hop | 0/28 | 25/28 | 26/28 |
| temporal | 0/22 | 22/22 | 22/22 |

**The headline:** Agentic GraphRAG is both the most accurate and the most
token-efficient. It answers most questions deterministically from the graph and
spends extra steps/tokens only on genuinely hard cases.

**Where agentic matters (the answer to the research question):**
- `lookup` — a single retrieval suffices; even vector-only RAG gets 19/19. An
  agent here is overkill.
- `aggregation`, `superlative`, `temporal` — one structured graph query/hop
  solves them; RAG scores 4/21, 6/10, and 0/22 respectively. Structure is
  essential, iteration is not.
- `multi_hop` — RAG scores **0/28**; it needs venue→event→medalist traversal,
  and the hardest cases (venue+date collisions) need the agent to detect
  ambiguity, prefer the exact single-day event, and confirm with documents.
  This is precisely where Agentic beats fixed GraphRAG.

## Round 2: reasoning over time

We built a bi-temporal fact store that detects conflicting versions of a fact,
supersedes older claims with newer/higher-authority ones, weighs source
authority (official > wikipedia > news > blog), answers "as of `<date>`"
queries via validity windows, and flags uncertainty when sources are close. It
is exposed to the agent as a `conflict_resolution` tool, ready for Round 2's
multi-source dataset. (The Round 1 corpus is single-source, so this is
demonstrated on constructed conflicting scenarios.)

## Two planners: fast vs fully agentic

The agent ships with two interchangeable planners behind one interface:

- **Rule planner** (default) — a transparent, token-cheap policy.
- **LLM planner** — an LLM chooses the next tool each step from the tool catalog
  and the evidence gathered so far. Genuinely agentic and robust to novel
  phrasing, at a higher token cost for comparable accuracy.

Reporting both is deliberate: it *measures the cost of being more agentic*,
which is exactly the hackathon's research question. Set `PLANNER=llm` to switch.

## Robustness to phrasing (not overfit to templates)

The fast parser uses tight patterns, but a **fallback** repairs weak parses:
it infers intent from semantic cue families (e.g. "drew over 30", "biggest
field", "preceding 2016"), resolves sports/venues against the actual graph
nodes, and can call an LLM for slot extraction when a real model is configured.
On a set of hand-reworded questions where the pure-regex path scores 0/6, the
full agent scores 6/6 (see `tests/test_robustness.py`). This proves the system
reasons about intent, not template strings.

## Limitations

- The two remaining multi_hop misses are true multi-event single-day collisions
  at one venue with no sport named; resolving them needs an LLM to read the
  medal results (the agent already fetches those candidate documents).
- The LLM planner's quality depends on the configured model; the rule planner is
  the safe, reproducible default.
- Hybrid graph+vector is used as two composed steps (graph traversal + a
  vector-search GSQL query); folding both into a single in-database GSQL query
  is a natural next step.

## What we'd do with more time

- Fold graph traversal and `vectorSearch()` into one hybrid GSQL query.
- Add graph community detection for corpus-summary questions.
- Confidence calibration for automatic abstention on low-evidence questions.

## Reproduce

```bash
pip install -r requirements.txt

# Offline (no external services) — for judges to clone and run:
$env:GRAPH_BACKEND="local"; python -m pytest -q      # 13 tests
$env:PYTHONPATH="src"; python -m agentic_graphrag.cli benchmark   # dashboard + metrics

# Live TigerGraph — set GRAPH_BACKEND=tigergraph + TG_HOST + TG_SECRET in .env,
# after the one-time setup in the README ("Running on TigerGraph Savanna").
python -m agentic_graphrag.cli submit                # hidden-set submission
python -m agentic_graphrag.cli ask "..."             # single question + trace
```
