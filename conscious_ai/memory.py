from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .events import read_recent_jsonl


# Rough heuristic; we have no tokenizer dependency in the zero-dep core.
# ~4 chars/token is the usual approximation for English + JSON punctuation.
CHARS_PER_TOKEN = 4


def estimate_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return len(text) // CHARS_PER_TOKEN + 1


def build_context_window(
    journal_path: Path,
    events_path: Path,
    token_budget: int,
    scan_limit: int = 5000,
) -> tuple[list[dict[str, Any]], int]:
    """Assemble the agent's episodic memory: its past thoughts and the
    conversation, merged in time order and trimmed (oldest first) to fit
    `token_budget`. Returns (records_oldest_first, approx_tokens_used).
    """
    records: list[dict[str, Any]] = []

    for entry in read_recent_jsonl(journal_path, scan_limit):
        text = str(entry.get("journal", "")).strip()
        if not text:
            continue
        record: dict[str, Any] = {
            "time": entry.get("time"),
            "cycle": entry.get("cycle"),
            "type": "thought",
            "text": text,
        }
        if entry.get("seed_word"):
            record["seed_word"] = entry["seed_word"]
        reflection = entry.get("reflection")
        if isinstance(reflection, dict) and reflection.get("familiarity") is not None:
            record["familiarity"] = reflection.get("familiarity")
        records.append(record)

    for entry in read_recent_jsonl(events_path, scan_limit):
        kind = entry.get("kind")
        payload = entry.get("payload") or {}
        if kind == "chat":
            text = str(payload.get("text", "")).strip()
            if text:
                records.append({"time": entry.get("time"), "type": "human", "text": text})
        elif kind == "model":
            reply = str(payload.get("reply", "")).strip()
            if reply:
                records.append({"time": entry.get("time"), "type": "reply", "text": reply})

    records.sort(key=lambda r: r.get("time") or "")

    # Accumulate from the newest backwards until the budget is spent, then
    # restore chronological order so the model reads oldest -> newest.
    selected: list[dict[str, Any]] = []
    used = 0
    for record in reversed(records):
        cost = estimate_tokens(record)
        if selected and used + cost > token_budget:
            break
        used += cost
        selected.append(record)
    selected.reverse()
    return selected, used
