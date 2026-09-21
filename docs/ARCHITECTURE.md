# Architecture

## System overview

```mermaid
flowchart TB
    Q[Question] --> BE[Benchmark Harness]
    BE --> RAG[Pipeline 1: RAG]
    BE --> GR[Pipeline 2: GraphRAG]
    BE --> AG[Pipeline 3: Agentic GraphRAG]

    RAG --> VS[(TigerGraph Vector DB<br/>Event.emb, HNSW/cosine)]
    GR --> VS
    GR --> KG[(Knowledge Graph)]
    AG --> ORCH

    subgraph Agentic core
      ORCH[Orchestrator] -->|plans next action| HARNESS[Agent Harness]
      HARNESS --> EL[entity_linking]
      HARNESS --> GT[graph_traversal]
      HARNESS --> AGG[aggregation]
      HARNESS --> SUP[superlative]
      HARNESS --> TMP[temporal]
      HARNESS --> MH[multi_hop]
      HARNESS --> DIS[disambiguate_by_date]
      HARNESS --> SS[similarity_search]
      HARNESS --> DR[document_retrieval_multi]
      HARNESS --> CR[conflict_resolution]
      HARNESS --> EE[evidence_evaluation]
    end

    EL --> KG
    GT --> KG
    AGG --> KG
    SUP --> KG
    TMP --> KG
    MH --> KG
    DIS --> KG
    SS --> VS
    DR --> KG
    CR --> FS[(Temporal Fact Store)]

    RAG --> OUT[PipelineResult: answer + citations + trace + tokens]
    GR --> OUT
    AG --> OUT
    OUT --> DASH[Metrics Dashboard + Trace Viewer]

    KG -.->|GRAPH_BACKEND=tigergraph| TG[TigerGraph Savanna: GSQL + Vector DB]
    KG -.->|GRAPH_BACKEND=local| LOCAL[In-memory LocalGraph + local vectors offline]
    VS -.->|tigergraph| TG
    VS -.->|local| LOCAL
```

On TigerGraph, both the graph and the vector store are the **same database**,
reached with **one connection / one secret**. The vector store is the
`Event.emb` attribute searched by the installed `event_vector_search` GSQL
query (`vectorSearch()`, HNSW, cosine).

## Knowledge graph ontology

```mermaid
erDiagram
    EVENT ||--|| GAMES : PART_OF
    EVENT ||--|| VENUE : HELD_AT
    EVENT ||--|| SPORT : HAS_SPORT
    EVENT ||--o{ ATHLETE : "WON_GOLD / WON_SILVER / WON_BRONZE"
    ATHLETE ||--|| NOC : REPRESENTS
    GAMES ||--|| GAMES : "NEXT / PREV"

    EVENT {
      string id
      string title
      int competitors
      int nations
      int year
      string season
      string date_raw
    }
    GAMES { string id  int year  string season }
    VENUE { string name }
    ATHLETE { string name }
    NOC { string code }
    SPORT { string name }
```

## The agentic loop

```mermaid
sequenceDiagram
    participant Q as Question
    participant O as Orchestrator
    participant T as Tools
    participant S as AgentState

    Q->>O: investigate
    loop until confident or budget reached
      O->>S: read evidence + facts + history
      O->>O: plan_next (choose tool + rationale)
      O->>T: dispatch tool
      T->>S: add evidence / facts / candidates / proposed answer
      Note over O,S: if a plan stalls (e.g. ambiguous multi-hop)<br/>orchestrator CHANGES STRATEGY
    end
    O->>Q: answer + full trace + citations + tokens
```

## Why the design maps to the scoring

| Criterion | Where it lives |
|---|---|
| Investigation accuracy (30%) | Graph-grounded traversals per intent; agentic disambiguation |
| Evidence quality & explainability (15%) | `TraceStep` per action + citations with source URLs; `explain` trace viewer |
| Agentic effectiveness & efficiency (15%) | Orchestrator escalates only when needed; `accuracy_per_1k_tokens` |
| Design, engineering & code quality (15%) | Pluggable backends, uniform `PipelineResult`, tests, reproducible offline |
| Innovation (15%) | Exact single-day disambiguation, bi-temporal conflict store, TigerGraph hybrid graph+vector |
| Presentation & Q&A (10%) | Dashboard + trace viewer + demo script |

## TigerGraph Savanna setup (one-time)

The graph and vector store are created and loaded in five steps. GSQL lives in
`tigergraph/`; loaders live in `scripts/`.

| Step | Command / file | What it does |
|---|---|---|
| 1. Schema | `tigergraph/01_schema.gsql` (Query Editor) | 6 vertex types + 9 edge types on graph `AgenticGraphRAG` |
| 2. Load graph | `scripts/tigergraph_ingest.py` then `scripts/tigergraph_load.py` | generate CSVs, upsert ~7.8k vertices + ~18k edges |
| 3. Vector attribute | `scripts/tigergraph_add_vector_attr.py` (or `tigergraph/02_vector_attribute.gsql`) | add `Event.emb(DIMENSION=384, METRIC="COSINE")` |
| 4. Load vectors | `scripts/tigergraph_load_vectors.py` | embed 2,187 events, upsert vectors |
| 5. Vector query | `scripts/tigergraph_install_vquery.py` (or `tigergraph/03_vector_query.gsql`) | install `event_vector_search` |

`scripts/tigergraph_verify.py` confirms counts + a sample traversal.
Connection uses `TG_HOST` + `TG_SECRET` from `.env` (Savanna gsqlSecret auth).
