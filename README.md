# Agentic GraphRAG on TigerGraph

An Agentic GraphRAG system built for the **TigerGraph Agentic GraphRAG Hackathon**.
It answers complex questions over an Olympic-events corpus three ways — **RAG**,
**GraphRAG**, and **Agentic GraphRAG** — and benchmarks them side by side on
**accuracy, completeness, and token efficiency**.

The guiding question of the hackathon: *when does a complex question actually
need an autonomous multi-step investigation, and when is a single retrieval
enough?* This project answers that with evidence.

The graph **and** the vector store run natively on **TigerGraph Savanna**: the
knowledge graph as vertices/edges, and the embeddings as a vector attribute on
`Event` searched with TigerGraph's built-in `vectorSearch()` (HNSW, cosine).

## The dataset (common benchmark)

| Path | Contents |
|---|---|
| `corpus/corpus.jsonl` | 2,951 docs (~5.47M tokens). 2,187 are Olympic-event infoboxes; the rest are distractors (e.g. films). |
| `questions/eval_public.jsonl` | 100 questions with gold answers and `gold_doc_ids`. |
| `questions/eval_hidden.jsonl` | 50 questions, answers withheld (scored by organizers). |

Question types: `lookup`, `multi_hop`, `aggregation`, `superlative`, `temporal`.
The corpus is the only source of truth; answers are defined over these
documents, not the real world.

## Results (100 public questions, live TigerGraph + Groq LLM + neural embeddings)

| Pipeline | Accuracy | Completeness | Tokens/q | Steps/q | Acc per 1k tokens |
|---|---|---|---|---|---|
| RAG | 29% | 58.1% | 611 | 2.00 | 0.48 |
| GraphRAG | 96% | 78.7% | 794 | 4.00 | 1.21 |
| **Agentic GraphRAG** | **98%** | 67.5% | **598** | 3.14 | **1.64** |

Per question type, RAG collapses exactly where structure is needed:

| qtype | RAG | GraphRAG | Agentic |
|---|---|---|---|
| lookup | 19/19 | 18/19 | 19/19 |
| superlative | 6/10 | 10/10 | 10/10 |
| aggregation | 4/21 | 21/21 | 21/21 |
| multi_hop | 0/28 | 25/28 | 26/28 |
| temporal | 0/22 | 22/22 | 22/22 |

**The finding:** vector-only RAG handles simple lookups but scores **0** on
multi-hop and temporal questions — those need graph structure. Agentic
GraphRAG is both the most accurate *and* the most token-efficient: it answers
most questions deterministically from the graph and escalates to extra
retrieval/reasoning steps only for genuinely hard cases (e.g. venue+date
collisions in multi-hop). See `docs/WRITEUP.md` for the full analysis.

## Architecture

```
   question ---> Benchmark Harness ---> metrics dashboard (3-way compare)
                       |
        +--------------+--------------+
        |              |              |
      RAG          GraphRAG       Agentic GraphRAG
    (vector)     (graph+vector)   (orchestrator + specialized agents)
        |              |              |
        +------+-------+------+-------+
               |              |
        TigerGraph Vector DB  TigerGraph graph (GSQL traversal)
        (Event.emb, HNSW)     (Event/Games/Venue/Athlete/NOC/Sport)
```

- **Agent harness** — manages state, tools, evidence, token budget, stopping criteria.
- **Orchestrator** — two interchangeable planners: `rule` (fast, deterministic) and `llm` (an LLM picks the next tool each step). Selectable via `PLANNER`.
- **Specialized agents** — entity linking, graph traversal, similarity search (TigerGraph Vector DB), document retrieval, aggregation, superlative, multi-hop, date disambiguation, conflict resolution, evidence evaluation.
- **Every result carries citations (with source URLs) and a full investigation trace** for explainability.

## Project layout

```
corpus/  questions/            dataset (source of truth)
tigergraph/                    GSQL: 01_schema, 02_vector_attribute, 03_vector_query
scripts/                       TigerGraph setup + verification scripts
docs/                          ARCHITECTURE, WRITEUP, DEMO_SCRIPT, GUIDEBOOK
src/agentic_graphrag/
    config.py                  env-driven config (LLM, embeddings, graph backend, planner)
    models.py                  shared types (Document, Question, Evidence, TraceStep, PipelineResult)
    llm.py                     provider-agnostic LLM client + token accounting
    data/loader.py             JSONL loading + infobox parser
    retrieval/                 embeddings, local vector store, TigerGraph vector store
    graph/                     ontology, builder, backends (local + tigergraph), entity linking, factory
    agents/                    harness, orchestrator, llm_planner, tools, state
    pipelines/                 rag / graphrag / agentic
    benchmark/                 runner + metrics + dashboard + explain
    temporal/                  Round 2 conflict/temporal fact reasoning
tests/                         13 offline tests (pytest)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env            # then fill in LLM + (optional) TigerGraph creds
```

Two run modes:

- **Local mode** (`GRAPH_BACKEND=local`) — builds an in-memory graph + local
  vector index from the corpus. Zero external services; ideal for judges to
  clone and run, and for the offline test suite.
- **TigerGraph mode** (`GRAPH_BACKEND=tigergraph`) — queries a live Savanna
  graph + TigerGraph Vector DB. This is the intended deployment.

## Usage

```bash
python -m pytest -q                                            # 13 tests (set GRAPH_BACKEND=local)

$env:PYTHONPATH="src"                                          # Windows PowerShell
python -m agentic_graphrag.cli benchmark                       # all 3 pipelines -> dashboard
python -m agentic_graphrag.cli submit                          # hidden-set submission jsonl
python -m agentic_graphrag.cli ask "who won gold at ... on ...?"   # single question + trace
python -m agentic_graphrag.cli explain --qids eval-004         # HTML trace viewer
```

Outputs land in `artifacts/`: `dashboard_public.html`, `traces.html`,
`summary_public.json`, `results_raw_public.jsonl`, `submission_hidden.jsonl`.

LLM: any OpenAI-compatible provider via `.env` (`OPENAI_BASE_URL` + `OPENAI_API_KEY`).
Tested with **Groq** (`openai/gpt-oss-20b`). Embeddings: `sentence-transformers`
(`all-MiniLM-L6-v2`, 384-dim), local and free.

## Running on TigerGraph Savanna

One-time setup (see `docs/ARCHITECTURE.md` for detail):

```bash
# 1. In Savanna: create an empty graph named AgenticGraphRAG
# 2. Query Editor: run  tigergraph/01_schema.gsql
python scripts/tigergraph_ingest.py         # generate vertex/edge CSVs
python scripts/tigergraph_load.py           # load graph (needs TG_HOST + TG_SECRET in .env)
python scripts/tigergraph_add_vector_attr.py   # add Event.emb (or run tigergraph/02_vector_attribute.gsql)
python scripts/tigergraph_load_vectors.py   # embed + upsert 384-dim vectors
python scripts/tigergraph_install_vquery.py # install event_vector_search (or run tigergraph/03_vector_query.gsql)
python scripts/tigergraph_verify.py         # sanity-check counts + a traversal
```

Then set `GRAPH_BACKEND=tigergraph` + `TG_HOST` + `TG_SECRET` in `.env`. The
graph query surface is identical across backends, so pipelines and agents run
unchanged.

## Docs

- `docs/ARCHITECTURE.md` — system, ontology, and agentic-loop diagrams
- `docs/WRITEUP.md` — what/how/results/limitations/future
- `docs/DEMO_SCRIPT.md` — 3–5 min demo walkthrough
- `docs/GUIDEBOOK.md` — saved hackathon guidebook

## Attribution

Corpus text derives from English Wikipedia, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each document
carries its source URL.
