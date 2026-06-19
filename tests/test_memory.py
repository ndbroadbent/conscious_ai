from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conscious_ai.memory import build_context_window, estimate_tokens


class MemoryTests(unittest.TestCase):
    def _write(self, lines: list[dict]) -> Path:
        path = Path(tempfile.mkdtemp()) / "log.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        return path

    def test_merges_thoughts_and_chat_in_time_order(self) -> None:
        journal = self._write([
            {"time": "2026-01-01T00:00:02", "cycle": 1, "journal": "thinking about rivers", "seed_word": "river"},
        ])
        events = self._write([
            {"time": "2026-01-01T00:00:01", "kind": "chat", "payload": {"text": "hello"}},
            {"time": "2026-01-01T00:00:03", "kind": "model", "payload": {"reply": "hi there"}},
            {"time": "2026-01-01T00:00:03", "kind": "sensor", "payload": {"summary": {}}},
        ])
        records, used = build_context_window(journal, events, token_budget=100_000)
        types = [r["type"] for r in records]
        self.assertEqual(types, ["human", "thought", "reply"])  # chronological, sensors excluded
        self.assertGreater(used, 0)

    def test_budget_trims_oldest_first(self) -> None:
        # Many fat thoughts; a tiny budget should keep only the newest few.
        lines = [
            {"time": f"2026-01-01T00:00:{i:02d}", "cycle": i, "journal": "x" * 400}
            for i in range(20)
        ]
        journal = self._write(lines)
        empty = self._write([])
        records, used = build_context_window(journal, empty, token_budget=300)
        self.assertLess(len(records), 20)
        self.assertGreaterEqual(len(records), 1)
        # kept the newest entries (highest cycle numbers), in order
        cycles = [r["cycle"] for r in records]
        self.assertEqual(cycles, sorted(cycles))
        self.assertEqual(cycles[-1], 19)

    def test_estimate_tokens_monotonic(self) -> None:
        self.assertGreater(estimate_tokens("a" * 400), estimate_tokens("a" * 4))


if __name__ == "__main__":
    unittest.main()
