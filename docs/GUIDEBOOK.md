# Agentic GraphRAG Hackathon Guidebook

> Saved copy of the official hackathon guidebook (Notion). Source of truth for rules, dataset, evaluation, and deliverables.

## About

Build an AI agent that autonomously investigates complex questions using graph, vector, and document evidence. Benchmark three approaches side by side (RAG, GraphRAG, Agentic GraphRAG) and prove exactly where agentic reasoning adds real value.

Headline goal: figure out which questions need an agent and which don't. Show where Agentic GraphRAG measurably improves accuracy/reasoning over simpler approaches, and where it is overkill.

## Getting Started

1. TigerGraph environment: Savanna (recommended, tgcloud.io, credits provided) or Community Edition (dl.tigergraph.com).
2. Clone the GraphRAG repo: github.com/tigergraph/graphrag
3. Get an LLM API key (any provider; free tiers are enough for hackathon scale).

## Problem Statement

- RAG retrieves similar text chunks.
- GraphRAG adds structure: entities, relationships, multi-hop reasoning.
- Agentic GraphRAG adds autonomous planning: the system decides its own retrieval path based on what it finds.

Big question: when does a complex question require an agentic, multi-step investigation rather than a single GraphRAG or RAG retrieval?

## Rounds

- Round 1 (everyone): build a working Agentic GraphRAG system with the three-way benchmark. Must include an orchestrator agent, specialised retrieval/reasoning agents, and a benchmarking pipeline. Top 15 advance.
- Round 2 (top 15): Sep 25 -> Oct 1, 2026. Extend to reason over evolving, conflicting, uncertain facts. Detect conflicting fact versions, determine what supersedes what, identify authoritative sources, handle uncertainty. Submit demo video, writeup, metrics dashboard. Top teams present live.

## What You Build

Three pipelines answering the same questions plus a comparison:

1. RAG: similarity search over text, then generate.
2. GraphRAG: use graph structure (entities, relationships, supporting content) to retrieve context and answer.
3. Agentic GraphRAG: an agent plans the investigation, selects retrieval methods, evaluates intermediate results, and performs additional retrieval/reasoning steps as needed.

System must include:

- Agent harness: manage state, tools, context, evidence, stopping criteria.
- Orchestrator agent: decides what to investigate and picks the next action (not a fixed sequence).
- Specialised agents: entity linking, graph traversal, similarity search, document retrieval, aggregation, multi-hop reasoning, evidence evaluation.

Orchestrator's next move depends on: the original question, the graph/available entities, evidence returned so far, and information still needed.

Example flows:
- Entity Linking -> Graph Traversal -> Answer
- Similarity Search -> Identify Entity -> Graph Traversal -> Retrieve Supporting Documents -> Answer
- Complex questions may need several iterations.

## Dataset

- Corpus: document collection to ingest and reason over.
- 100 public evaluation questions (with answers): test, tune, benchmark. Run all three pipelines and include results in the metrics dashboard.
- 50 hidden evaluation questions (no answers): run the system on all 50, submit raw outputs (tokens used, answers, agentic trace). Scored against held-out ground truth.
- Own datasets welcome as a bonus; provided dataset is the common benchmark.

## Evaluation

For every question and pipeline measure:

- Accuracy: correctness, completeness, grounding in available evidence.
- Token efficiency: context tokens, LLM input tokens, LLM output tokens, total tokens per answer.
- Trace & agentic behavior (Agentic GraphRAG): number of retrieval/reasoning steps, retrieval methods selected, specialised agents invoked, tools called, time per operation, tokens per operation, total tokens, number of chunks and citations, whether strategy changed mid-investigation, when and why it stopped.

Objective: not just whether Agentic GraphRAG gives a better answer, but whether the extra steps are worth the extra complexity and token cost.

## Judging

| Criteria | Weight | What they look for |
|---|---|---|
| Investigation accuracy | 30% | Correct, complete answers using the right evidence |
| Evidence quality & explainability | 15% | Grounded answers, clear citations, clear investigation path |
| Agentic effectiveness & efficiency | 15% | Right retrieval methods, agentic steps where they add value, accuracy vs token cost |
| Agentic design, engineering & code quality | 15% | Architecture, tool use, reliability, reproducibility, repo quality |
| Innovation | 15% | Novel investigation methods, graph reasoning, UX |
| Final presentation & Q&A | 10% | Demo quality, technical clarity, Q&A |

## Deliverables

Round 1: working system, GitHub repo, architecture diagram, demo video, metrics dashboard (tokens, accuracy, completeness across three pipelines). Optional social post (tag @TigerGraph).

Round 2 (finalists): refined system, updated repo, architecture diagram, 3-5 min demo video, metrics dashboard, short writeup (what, how, key results, limitations, future work), live presentation for top teams.

## Timeline

- Sep 2: registration opens, guidebook + dataset live
- Sep 14: registration closes
- Sep 24: Round 1 submission deadline
- Oct 1: Round 2 / final submission deadline
- Oct 2-5: judging
- Oct 7: results announced

## Key Links

- GraphRAG repo: github.com/tigergraph/graphrag
- TigerGraph MCP: github.com/tigergraph/tigergraph-mcp
- Savanna: tgcloud.io
- Community Edition: dl.tigergraph.com
- Docs: https://www.tigergraph.com/docs/home/
- Discord: https://discord.com/invite/eKWm3mbkw2
- Dataset: https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing
