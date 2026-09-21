"""Command-line entrypoint.

Usage:
  python -m agentic_graphrag.cli benchmark [--limit N] [--pipelines rag,graphrag,agentic]
  python -m agentic_graphrag.cli submit    [--pipeline agentic]
  python -m agentic_graphrag.cli ask "your question here"
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .data.loader import load_questions
from .benchmark.runner import BenchmarkEngine
from .benchmark.dashboard import render_dashboard
from .benchmark.explain import explain_text, render_trace_html
from .models import Question, QType


def _print_summary(summaries: dict) -> None:
    print("\n=== BENCHMARK SUMMARY ===")
    for name in ("rag", "graphrag", "agentic"):
        if name not in summaries:
            continue
        s = summaries[name]
        print(f"\n{name.upper():>16}: acc={s['accuracy']*100:.1f}%  "
              f"complete={s['avg_completeness']*100:.1f}%  "
              f"tokens/q={s['avg_tokens']:.0f}  steps/q={s['avg_steps']:.2f}  "
              f"acc/1k-tok={s['accuracy_per_1k_tokens']:.3f}")
        for qt, d in s["by_qtype"].items():
            print(f"                    {qt:12} {d['correct']}/{d['n']}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config()
    engine = BenchmarkEngine.build(cfg, top_k=args.top_k, max_steps=args.max_steps)
    questions = load_questions(cfg.public_questions_path)
    if args.limit:
        questions = questions[: args.limit]
    which = args.pipelines.split(",") if args.pipelines else None
    out = engine.run(questions, which=which, label="public")
    _print_summary(out["summaries"])
    dash = render_dashboard(out["summaries"], cfg.artifacts_dir / "dashboard_public.html", "public")
    print(f"\nDashboard: {dash}")
    print(f"Artifacts in: {cfg.artifacts_dir}")


def cmd_submit(args: argparse.Namespace) -> None:
    cfg = load_config()
    engine = BenchmarkEngine.build(cfg, max_steps=args.max_steps)
    hidden = load_questions(cfg.hidden_questions_path)
    path = engine.emit_submission(hidden, pipeline=args.pipeline, label="hidden")
    print(f"Submission written: {path}")


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config()
    engine = BenchmarkEngine.build(cfg, max_steps=args.max_steps)
    q = Question(qid="adhoc", question=args.question, qtype=QType.OTHER)
    result = engine.pipelines[args.pipeline].answer(q)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(explain_text(result))


def cmd_explain(args: argparse.Namespace) -> None:
    """Render an HTML trace viewer for selected questions (agentic pipeline)."""
    cfg = load_config()
    engine = BenchmarkEngine.build(cfg, max_steps=args.max_steps)
    questions = load_questions(cfg.public_questions_path)
    if args.qids:
        wanted = set(args.qids.split(","))
        questions = [q for q in questions if q.qid in wanted]
    elif args.limit:
        questions = questions[: args.limit]
    results = [engine.pipelines[args.pipeline].answer(q) for q in questions]
    out = render_trace_html(results, cfg.artifacts_dir / "traces.html")
    print(f"Trace viewer: {out}  ({len(results)} questions)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic_graphrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run all pipelines on the public set")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--pipelines", type=str, default="")
    b.add_argument("--top-k", type=int, default=5)
    b.add_argument("--max-steps", type=int, default=8)
    b.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("submit", help="emit hidden-set submission")
    s.add_argument("--pipeline", type=str, default="agentic")
    s.add_argument("--max-steps", type=int, default=8)
    s.set_defaults(func=cmd_submit)

    a = sub.add_parser("ask", help="answer a single question with explanation")
    a.add_argument("question", type=str)
    a.add_argument("--pipeline", type=str, default="agentic")
    a.add_argument("--max-steps", type=int, default=8)
    a.add_argument("--json", action="store_true", help="print full JSON instead of narrative")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("explain", help="render HTML trace viewer for questions")
    e.add_argument("--qids", type=str, default="", help="comma-separated qids")
    e.add_argument("--limit", type=int, default=10)
    e.add_argument("--pipeline", type=str, default="agentic")
    e.add_argument("--max-steps", type=int, default=8)
    e.set_defaults(func=cmd_explain)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
