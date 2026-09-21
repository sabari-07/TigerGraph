"""Verify the configured LLM (Groq) responds. Does not print the API key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from agentic_graphrag.llm import LLMClient

cfg = load_config()
key = cfg.llm.api_key or ""
print(f"provider={cfg.llm.provider}  base_url={cfg.llm.base_url!r}  model={cfg.llm.model}")
print(f"api_key set: {bool(key) and key != 'PASTE_YOUR_GROQ_KEY_HERE'}  (len={len(key)})")

client = LLMClient(cfg.llm)
print(f"real client initialized: {client._client is not None}")

resp = client.complete(
    system="You answer in one word.",
    user="Reply with the single word: pong",
    max_tokens=8,
)
print(f"response text: {resp.text!r}")
print(f"tokens: {resp.tokens.as_dict()}")
print("LIVE LLM OK" if resp.text.strip() else "NO RESPONSE (fell back to mock?)")
