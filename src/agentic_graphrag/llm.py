"""LLM client abstraction with token accounting.

Every LLM call routes through here so token usage is measured consistently
across all three pipelines (token efficiency is a scored metric). A
deterministic 'mock' provider lets the full system + benchmark run offline; it
extracts answers from the provided context so the pipelines are exercised
end to end without an API key.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import LLMConfig
from .models import TokenUsage


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
class TokenCounter:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._enc = None
        try:
            import tiktoken

            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Fallback heuristic: ~4 chars per token.
        return max(1, len(text) // 4)


@dataclass
class LLMResponse:
    text: str
    tokens: TokenUsage


class LLMClient:
    """Provider-agnostic chat client. Returns text + measured token usage."""

    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.counter = TokenCounter(self.cfg.model)
        self._client = None
        if self.cfg.provider in ("openai", "azure"):
            self._init_openai()

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI

            kwargs = {}
            if self.cfg.api_key:
                kwargs["api_key"] = self.cfg.api_key
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            print(f"[llm] OpenAI init failed, using mock provider: {exc}")
            self._client = None

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        context_tokens = self.counter.count(system)
        input_tokens = self.counter.count(user)

        # Reasoning models (e.g. gpt-oss, o1/o3) spend part of the token budget
        # on hidden reasoning, so a tight max_tokens can leave no visible answer.
        # Raise the floor for these while keeping small budgets for plain models.
        effective_max = max_tokens
        model_lc = (self.cfg.model or "").lower()
        if any(tag in model_lc for tag in ("gpt-oss", "o1", "o3", "reason", "qwen3")):
            effective_max = max(max_tokens, 4096)

        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=effective_max,
                )
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    tokens = TokenUsage(
                        context_tokens=context_tokens,
                        input_tokens=max(0, usage.prompt_tokens - context_tokens),
                        output_tokens=usage.completion_tokens,
                    )
                else:
                    tokens = TokenUsage(
                        context_tokens=context_tokens,
                        input_tokens=input_tokens,
                        output_tokens=self.counter.count(text),
                    )
                return LLMResponse(text=text, tokens=tokens)
            except Exception as exc:
                print(f"[llm] completion failed, using mock: {exc}")

        # ---- mock provider ---- #
        text = _mock_answer(system, user)
        return LLMResponse(
            text=text,
            tokens=TokenUsage(
                context_tokens=context_tokens,
                input_tokens=input_tokens,
                output_tokens=self.counter.count(text),
            ),
        )


# --------------------------------------------------------------------------- #
# Mock answer extraction (offline mode)
# --------------------------------------------------------------------------- #
_GOLD_RE = re.compile(r"^\s{2}gold:\s*(.+)$", re.M)
_NATIONS_RE = re.compile(r"^\s{2}nations:\s*(\d+)", re.M)
_ANSWER_HINT_RE = re.compile(r"FINAL_ANSWER:\s*(.+)", re.I)


def _mock_planner_decision(user: str) -> str:
    """Simulate an agentic planner's next-action choice offline.

    Reads the parsed-intent hint and the tools-already-run list from the planner
    prompt, and returns a JSON decision that mirrors sound tool selection. This
    lets the LLM-planner path run deterministically without an API key while
    still exercising real branching (including strategy changes).
    """
    intent_m = re.search(r"Parsed intent \(hint\):\s*(\w+)", user)
    intent = intent_m.group(1) if intent_m else "unknown"
    run_m = re.search(r"Tools already run:\s*(.+)", user)
    ran = run_m.group(1).strip() if run_m else "none"
    cand_m = re.search(r"Candidate events under consideration:\s*(\d+)", user)
    n_cand = int(cand_m.group(1)) if cand_m else 0
    conf_m = re.search(r"confidence\s+([0-9.]+)\)", user)
    conf = float(conf_m.group(1)) if conf_m else 0.0

    primary = {
        "aggregation": "aggregation", "superlative": "superlative",
        "temporal": "temporal", "multi_hop": "multi_hop", "lookup": "lookup",
    }.get(intent)

    def dec(tool, rationale, stop=False, args=None):
        return json.dumps({"tool": tool, "args": args or {}, "rationale": rationale, "stop": stop})

    if primary and primary not in ran:
        return dec(primary, f"Run {primary} for intent {intent}.")
    # multi_hop escalation
    if intent == "multi_hop" and conf < 0.75 and n_cand > 1:
        if "disambiguate_by_date" not in ran:
            return dec("disambiguate_by_date", "Several candidates share the date; pick exact day.")
        if "document_retrieval_multi" not in ran:
            return dec("document_retrieval_multi", "Fetch candidate docs to ground the choice.",
                       args={"limit": 4})
    if conf >= 0.6:
        return dec("", "Confident grounded answer reached.", stop=True)
    if "similarity_search" not in ran:
        return dec("similarity_search", "No structured path; try semantic retrieval.", args={"k": 5})
    return dec("", "No further useful action.", stop=True)


def _mock_answer(system: str, user: str) -> str:
    """Best-effort answer from context so offline runs exercise the pipeline.

    Real reasoning comes from a configured LLM. This heuristic pulls the most
    likely answer token out of the provided context based on the question,
    keeping the mock honest about what evidence it saw.
    """
    # Planner prompt -> return a JSON action decision.
    if "You are the orchestrator" in system or "What is the single best next action?" in user:
        return _mock_planner_decision(user)

    # If an agent pre-computed a deterministic answer, honor it.
    hint = _ANSWER_HINT_RE.search(user)
    if hint:
        return hint.group(1).strip()

    q = user.lower()
    # "who won gold" style -> pull a gold field from context.
    if "gold" in q or "who won" in q:
        m = _GOLD_RE.search(user)
        if m:
            return m.group(1).strip()
    if "how many nations" in q:
        m = _NATIONS_RE.search(user)
        if m:
            return m.group(1).strip()
    # Fallback: first non-empty context line.
    for line in user.splitlines():
        s = line.strip()
        if s and not s.endswith("?") and len(s) > 3:
            return s[:120]
    return "unknown"
