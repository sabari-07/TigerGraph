"""Render the metrics dashboard as a self-contained HTML file.

Shows the three-way comparison the hackathon requires: accuracy, completeness,
token efficiency, agentic steps, and a per-question-type breakdown. The chart is
drawn with inline SVG so the file opens anywhere with no network access.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_PIPELINE_ORDER = ["rag", "graphrag", "agentic"]
_PIPELINE_LABEL = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}
_COLORS = {"rag": "#9aa7b5", "graphrag": "#4c8bf5", "agentic": "#f5a623"}


def _bar_chart(summaries: dict[str, Any], metric: str, title: str, fmt: str = "pct") -> str:
    """Inline SVG bar chart comparing pipelines on one metric."""
    rows = [(p, summaries[p][metric]) for p in _PIPELINE_ORDER if p in summaries]
    if not rows:
        return ""
    max_val = max(v for _, v in rows) or 1.0
    width, bar_h, gap, left = 520, 34, 18, 150
    height = len(rows) * (bar_h + gap) + 20
    bars = []
    for i, (p, v) in enumerate(rows):
        y = 10 + i * (bar_h + gap)
        w = int((v / max_val) * (width - left - 70))
        label = f"{v*100:.0f}%" if fmt == "pct" else (f"{v:,.0f}" if fmt == "int" else f"{v:.2f}")
        bars.append(
            f'<text x="{left-10}" y="{y+bar_h*0.65}" text-anchor="end" '
            f'font-size="14" fill="#cfd8e3">{_PIPELINE_LABEL[p]}</text>'
            f'<rect x="{left}" y="{y}" width="{w}" height="{bar_h}" rx="5" '
            f'fill="{_COLORS[p]}"></rect>'
            f'<text x="{left+w+8}" y="{y+bar_h*0.65}" font-size="14" '
            f'fill="#e8eef5">{label}</text>'
        )
    return (
        f'<div class="chart"><h3>{html.escape(title)}</h3>'
        f'<svg width="{width}" height="{height}" role="img">{"".join(bars)}</svg></div>'
    )


def _summary_table(summaries: dict[str, Any]) -> str:
    metrics = [
        ("accuracy", "Accuracy", "pct"),
        ("avg_completeness", "Avg completeness (citation grounding)", "pct"),
        ("avg_tokens", "Avg tokens / question", "int"),
        ("avg_steps", "Avg steps / question", "num"),
        ("accuracy_per_1k_tokens", "Accuracy per 1k tokens", "num"),
        ("strategy_changes", "Strategy changes", "int"),
    ]
    header = "".join(f"<th>{_PIPELINE_LABEL[p]}</th>" for p in _PIPELINE_ORDER if p in summaries)
    rows = []
    for key, label, fmt in metrics:
        cells = []
        for p in _PIPELINE_ORDER:
            if p not in summaries:
                continue
            v = summaries[p].get(key, 0)
            if fmt == "pct":
                cells.append(f"<td>{v*100:.1f}%</td>")
            elif fmt == "int":
                cells.append(f"<td>{v:,.0f}</td>")
            else:
                cells.append(f"<td>{v:.2f}</td>")
        rows.append(f"<tr><td class='rowlabel'>{label}</td>{''.join(cells)}</tr>")
    return (
        f"<table class='summary'><thead><tr><th>Metric</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _qtype_table(summaries: dict[str, Any]) -> str:
    qtypes: list[str] = []
    for p in summaries.values():
        for qt in p.get("by_qtype", {}):
            if qt not in qtypes:
                qtypes.append(qt)
    qtypes.sort()
    header = "".join(f"<th>{_PIPELINE_LABEL[p]}</th>" for p in _PIPELINE_ORDER if p in summaries)
    rows = []
    for qt in qtypes:
        cells = []
        for p in _PIPELINE_ORDER:
            if p not in summaries:
                continue
            d = summaries[p].get("by_qtype", {}).get(qt)
            if d:
                cells.append(f"<td>{d['correct']}/{d['n']} <span class='pct'>({d['accuracy']*100:.0f}%)</span></td>")
            else:
                cells.append("<td>-</td>")
        rows.append(f"<tr><td class='rowlabel'>{qt}</td>{''.join(cells)}</tr>")
    return (
        f"<table class='summary'><thead><tr><th>Question type</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_dashboard(summaries: dict[str, Any], out_path: str | Path, label: str = "public") -> str:
    out_path = Path(out_path)
    charts = (
        _bar_chart(summaries, "accuracy", "Accuracy", "pct")
        + _bar_chart(summaries, "avg_completeness", "Citation completeness", "pct")
        + _bar_chart(summaries, "avg_tokens", "Avg tokens per question", "int")
        + _bar_chart(summaries, "accuracy_per_1k_tokens", "Accuracy per 1k tokens (efficiency)", "num")
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic GraphRAG Benchmark - {html.escape(label)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0d1117;
         color:#e8eef5; margin:0; padding:32px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:32px 0 12px; color:#9fb3c8; }}
  h3 {{ font-size:14px; margin:0 0 8px; color:#9fb3c8; font-weight:600; }}
  .sub {{ color:#7d8ba0; margin-bottom:24px; }}
  table.summary {{ border-collapse:collapse; width:100%; max-width:900px; margin-bottom:16px; }}
  table.summary th, table.summary td {{ padding:10px 14px; text-align:left;
         border-bottom:1px solid #1e2733; }}
  table.summary th {{ color:#9fb3c8; font-weight:600; font-size:13px; }}
  td.rowlabel {{ color:#cfd8e3; }}
  .pct {{ color:#7d8ba0; font-size:12px; }}
  .charts {{ display:flex; flex-wrap:wrap; gap:28px; }}
  .chart {{ background:#141b24; border:1px solid #1e2733; border-radius:10px; padding:16px 20px; }}
  .legend span {{ display:inline-block; margin-right:16px; font-size:13px; }}
  .dot {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px;
          vertical-align:middle; }}
  .headline {{ background:#141b24; border:1px solid #22303d; border-left:4px solid #f5a623;
          border-radius:8px; padding:16px 20px; max-width:900px; margin-bottom:24px; }}
</style></head>
<body>
  <h1>Agentic GraphRAG - Three-Pipeline Benchmark</h1>
  <div class="sub">Dataset split: <b>{html.escape(label)}</b> &middot; RAG vs GraphRAG vs Agentic GraphRAG</div>

  <div class="headline">
    <b>The question this answers:</b> does autonomous multi-step investigation beat a fixed
    retrieval plan, and is it worth the token cost? Compare accuracy against
    <i>accuracy per 1k tokens</i> below.
  </div>

  <div class="legend">
    <span><span class="dot" style="background:{_COLORS['rag']}"></span>RAG</span>
    <span><span class="dot" style="background:{_COLORS['graphrag']}"></span>GraphRAG</span>
    <span><span class="dot" style="background:{_COLORS['agentic']}"></span>Agentic GraphRAG</span>
  </div>

  <h2>Summary metrics</h2>
  {_summary_table(summaries)}

  <h2>Comparison charts</h2>
  <div class="charts">{charts}</div>

  <h2>Accuracy by question type</h2>
  {_qtype_table(summaries)}

  <h2>Raw summary (JSON)</h2>
  <pre style="background:#141b24;border:1px solid #1e2733;border-radius:8px;padding:16px;overflow:auto;max-width:900px;color:#9fb3c8;font-size:12px;">{html.escape(json.dumps(summaries, indent=2))}</pre>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return str(out_path)
