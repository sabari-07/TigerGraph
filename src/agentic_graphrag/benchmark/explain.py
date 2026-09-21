"""Explainability: turn a PipelineResult into a human-readable investigation.

Two renderers:
  - explain_text(result): a plain-text investigation path with cited evidence,
    suitable for logs, the CLI, and the demo.
  - render_trace_html(results, out): an interactive HTML viewer showing, per
    question, the ordered steps the agent took, why (rationale), what evidence
    each step produced (linked to source URLs), tokens spent, and the stopping
    decision.

This directly supports the "evidence quality & explainability" criterion:
grounded answers with clear citations and a clear investigation path.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

from ..models import PipelineResult


def explain_text(result: PipelineResult) -> str:
    lines: list[str] = []
    lines.append(f"Q[{result.qid}] pipeline={result.pipeline}")
    lines.append(f"Answer: {result.answer}")
    if result.meta.get("intent"):
        lines.append(f"Intent: {result.meta['intent']}  "
                     f"confidence: {result.meta.get('confidence', '-')}")
    lines.append("")
    lines.append("Investigation path:")
    for step in result.trace:
        lines.append(f"  {step.step_index}. [{step.agent}] {step.action}")
        if step.rationale:
            lines.append(f"      why: {step.rationale}")
        if step.outputs_summary:
            lines.append(f"      -> {step.outputs_summary}")
        if step.evidence_ids:
            lines.append(f"      evidence: {', '.join(step.evidence_ids[:6])}")
        if step.tokens.total:
            lines.append(f"      tokens: {step.tokens.total}")
    lines.append("")
    if result.strategy_changed:
        lines.append("Note: the agent CHANGED STRATEGY mid-investigation.")
    lines.append(f"Stopped because: {result.stopped_reason}")
    lines.append(f"Total tokens: {result.tokens.total}  steps: {result.num_steps}")
    lines.append("")
    lines.append("Citations:")
    for ev in result.evidence:
        url = f"  <{ev.url}>" if ev.url else ""
        lines.append(f"  [{ev.doc_id}] {ev.title}{url}")
    return "\n".join(lines)


def _step_html(step) -> str:
    ev = ""
    if step.evidence_ids:
        ev = ("<div class='ev'>evidence: "
              + ", ".join(html.escape(e) for e in step.evidence_ids[:8]) + "</div>")
    toks = f"<span class='tok'>{step.tokens.total} tok</span>" if step.tokens.total else ""
    return (
        f"<div class='step'>"
        f"<div class='head'><span class='idx'>{step.step_index}</span>"
        f"<span class='agent'>{html.escape(step.agent)}</span>"
        f"<span class='action'>{html.escape(step.action)}</span>{toks}</div>"
        f"<div class='why'>{html.escape(step.rationale)}</div>"
        f"<div class='out'>{html.escape(step.outputs_summary)}</div>{ev}"
        f"</div>"
    )


def render_trace_html(results: Iterable[PipelineResult], out_path: str | Path,
                      title: str = "Agentic Investigation Traces") -> str:
    out_path = Path(out_path)
    cards = []
    for r in results:
        strat = "<span class='badge change'>strategy change</span>" if r.strategy_changed else ""
        steps = "".join(_step_html(s) for s in r.trace)
        cites = "".join(
            f"<li><a href='{html.escape(e.url)}' target='_blank'>[{html.escape(e.doc_id)}]</a> "
            f"{html.escape(e.title)}</li>" if e.url else
            f"<li>[{html.escape(e.doc_id)}] {html.escape(e.title)}</li>"
            for e in r.evidence
        )
        cards.append(
            f"<div class='card'>"
            f"<div class='q'>[{html.escape(r.qid)}] <b>{html.escape(r.pipeline)}</b> {strat}</div>"
            f"<div class='ans'>Answer: <b>{html.escape(r.answer)}</b></div>"
            f"<div class='meta'>intent={html.escape(str(r.meta.get('intent','-')))} "
            f"&middot; {r.num_steps} steps &middot; {r.tokens.total} tokens &middot; "
            f"stop: {html.escape(r.stopped_reason)}</div>"
            f"<div class='steps'>{steps}</div>"
            f"<details><summary>citations ({len(r.evidence)})</summary><ul>{cites}</ul></details>"
            f"</div>"
        )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e8eef5;margin:0;padding:28px;}}
 h1{{font-size:22px;}}
 .card{{background:#141b24;border:1px solid #1e2733;border-radius:10px;padding:16px 20px;margin-bottom:18px;}}
 .q{{font-size:15px;color:#9fb3c8;margin-bottom:4px;}}
 .ans{{font-size:16px;margin-bottom:6px;}}
 .meta{{color:#7d8ba0;font-size:12px;margin-bottom:12px;}}
 .step{{border-left:2px solid #2b3a4a;padding:6px 0 6px 14px;margin:6px 0;}}
 .head{{display:flex;gap:10px;align-items:center;}}
 .idx{{background:#22303d;border-radius:4px;padding:0 7px;font-size:12px;}}
 .agent{{color:#4c8bf5;font-weight:600;font-size:13px;}}
 .action{{color:#9fb3c8;font-size:13px;}}
 .tok{{margin-left:auto;color:#7d8ba0;font-size:11px;}}
 .why{{color:#cfd8e3;font-size:13px;margin:3px 0;}}
 .out{{color:#8fd19e;font-size:12px;}}
 .ev{{color:#7d8ba0;font-size:11px;margin-top:2px;}}
 .badge.change{{background:#f5a623;color:#111;border-radius:4px;padding:1px 7px;font-size:11px;}}
 a{{color:#4c8bf5;}} summary{{cursor:pointer;color:#9fb3c8;font-size:13px;margin-top:8px;}}
 ul{{font-size:12px;color:#cfd8e3;}}
</style></head><body>
<h1>{html.escape(title)}</h1>{''.join(cards)}</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return str(out_path)
