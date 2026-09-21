# Demo Video Script (3–5 minutes)

Goal: show that Agentic GraphRAG beats RAG and GraphRAG *where it matters*, with
a clear investigation path and lower token cost. Keep it tight and evidence-led.

---

## 0:00–0:30 — The question
"RAG retrieves text. GraphRAG adds structure. But some questions need a system
that plans, checks what it found, spots gaps, and keeps digging. We built that
on TigerGraph, and we benchmarked all three approaches to find out exactly when
the agent is worth it."

Show: the title slide + the one-line research question.

## 0:30–1:15 — The data and the graph
- Show `corpus/corpus.jsonl`: 2,951 Olympic event docs with structured infoboxes.
- Show the ontology diagram (docs/ARCHITECTURE.md): Event–Games–Venue–Athlete–NOC–Sport,
  with the NEXT/PREV temporal chain.
- One line: "We turn every infobox into a typed graph — 2,187 events, ~18k edges —
  loaded on TigerGraph Savanna. The embeddings live in the same database as a
  vector attribute, searched with TigerGraph's native vectorSearch."
- Optionally show the Savanna schema view + `scripts/tigergraph_verify.py` counts.

## 1:15–2:15 — The three pipelines + the benchmark
- Open `artifacts/dashboard_public.html` (from a live run).
- Walk the summary table and the charts. Land on the headline:
  "RAG 29%, GraphRAG 96%, Agentic 98% — and the agent uses the fewest tokens per
  question. It's both the most accurate and the most efficient."
- Point at the per-qtype table: "Vector-only RAG scores ZERO on multi_hop and
  temporal — those need graph structure. lookup it gets 19/19. That's the whole
  point: structure matters where a single retrieval can't reach."

## 2:15–3:30 — Where the agent earns its keep (the money shot)
- Run: `python -m agentic_graphrag.cli ask "Who won the gold medal in the event
  held at Sydney Convention and Exhibition Centre on 23 September 2000?"`
- Read the trace aloud:
  1. entity_linking resolves the venue
  2. multi_hop finds 57 events → 2 share the date → **AMBIGUOUS**
  3. **strategy change** → disambiguate_by_date → the event on *exactly* that
     single day → Pyrros Dimas
  4. synthesize with citations to Wikipedia
- "GraphRAG's fixed plan picks the wrong one here. The agent notices the
  ambiguity and changes strategy — that's the whole thesis, live."
- Open `artifacts/traces.html` to show the visual investigation path + clickable
  citations.

## 3:30–4:15 — Round 2 preview + engineering
- "For evolving facts we built a bi-temporal store (`temporal/facts.py`): it
  supersedes stale claims, weighs source authority, answers as-of-date, and
  flags uncertainty — exposed to the agent as a conflict_resolution tool."
- Mention: fully TigerGraph-native (graph + vector DB, one connection), two
  interchangeable planners (rule vs LLM), 13 passing tests, reproducible offline
  for anyone who clones the repo.

## 4:15–4:45 — Close
"So: agents don't help everywhere. They help precisely on the hard, ambiguous,
multi-hop questions — and when you design them to escalate only when needed,
they're cheaper *and* more accurate. That's the answer to the hackathon's
question, backed by the benchmark."

Show: final dashboard + GitHub repo URL.

---

### Shot list / assets
- dashboard_public.html (metrics)
- traces.html (investigation paths)
- docs/ARCHITECTURE.md diagrams
- Savanna console (graph schema + vector attribute) for the "on TigerGraph" proof
- terminal for the live `ask` run
