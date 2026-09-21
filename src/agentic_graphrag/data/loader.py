"""Load corpus and questions from JSONL, and parse Olympic-event infoboxes.

The infobox is the structured heart of each event document. Parsing it once
into a dict lets the graph builder create clean typed nodes/edges and lets the
GraphRAG/agentic aggregation logic run over numeric fields deterministically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from ..models import Document, Question, QType

INFOBOX_MARKER = "[Infobox Olympic event]"
# Infobox lines look like:  "  competitors: 24"  (two leading spaces, key: value)
_FIELD_RE = re.compile(r"^\s{2}([A-Za-z_]+):\s*(.*)$")


def parse_infobox(text: str) -> dict[str, str]:
    """Extract key/value pairs from the infobox block at the top of a doc."""
    fields: dict[str, str] = {}
    if INFOBOX_MARKER not in text:
        return fields
    # The infobox is the run of "  key: value" lines after the marker.
    started = False
    for line in text.splitlines():
        if INFOBOX_MARKER in line:
            started = True
            continue
        if not started:
            continue
        m = _FIELD_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            # First occurrence wins (infobox precedes result tables).
            if key not in fields:
                fields[key] = value
        elif line.strip() == "":
            # Blank line ends the infobox block.
            if fields:
                break
    return fields


def load_documents(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "")
            docs.append(
                Document(
                    doc_id=d.get("doc_id") or d.get("wikidata_qid", ""),
                    title=d.get("title", ""),
                    text=text,
                    url=d.get("url", ""),
                    wikidata_qid=d.get("wikidata_qid", ""),
                    approx_tokens=int(d.get("approx_tokens", 0) or 0),
                    infobox=parse_infobox(text),
                )
            )
    return docs


def iter_documents(path: str | Path) -> Iterator[Document]:
    """Streaming variant for large corpora."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "")
            yield Document(
                doc_id=d.get("doc_id") or d.get("wikidata_qid", ""),
                title=d.get("title", ""),
                text=text,
                url=d.get("url", ""),
                wikidata_qid=d.get("wikidata_qid", ""),
                approx_tokens=int(d.get("approx_tokens", 0) or 0),
                infobox=parse_infobox(text),
            )


def _coerce_qtype(raw: str) -> QType:
    try:
        return QType(raw)
    except ValueError:
        return QType.OTHER


def load_questions(path: str | Path) -> list[Question]:
    questions: list[Question] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            ans = q.get("answer", [])
            if isinstance(ans, str):
                ans = [ans]
            questions.append(
                Question(
                    qid=q["qid"],
                    question=q["question"],
                    qtype=_coerce_qtype(q.get("qtype", "other")),
                    gold_doc_ids=list(q.get("gold_doc_ids", [])),
                    answer=list(ans),
                    answer_named_in_question=bool(q.get("answer_named_in_question", False)),
                    answer_verified=bool(q.get("answer_verified", False)),
                )
            )
    return questions
