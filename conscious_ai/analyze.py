from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_config
from .events import read_recent_jsonl


def _nums(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def summarize(data_dir: Path) -> str:
    metrics = read_recent_jsonl(data_dir / "metrics.jsonl", 100_000)
    journal = read_recent_jsonl(data_dir / "journal.jsonl", 100_000)
    lines: list[str] = []

    lines.append(f"cycles recorded: {len(metrics)}")
    if not metrics:
        lines.append("(no metrics yet — run the loop first)")
        return "\n".join(lines)

    errors = _nums(metrics, "prediction_error")
    if len(errors) >= 4:
        half = len(errors) // 2
        first, second = mean(errors[:half]), mean(errors[half:])
        trend = "falling ↓ (learning)" if second < first - 0.01 else "rising ↑" if second > first + 0.01 else "flat →"
        lines.append(
            f"prediction error: mean {mean(errors):.3f} · first-half {first:.3f} → second-half {second:.3f}  [{trend}]"
        )
    elif errors:
        lines.append(f"prediction error: mean {mean(errors):.3f} (need more cycles for a trend)")

    valence = _nums(metrics, "valence")
    arousal = _nums(metrics, "arousal")
    if valence:
        lines.append(f"valence: range {min(valence):+.2f}..{max(valence):+.2f} · mean {mean(valence):+.2f}")
    if arousal:
        lines.append(f"arousal: range {min(arousal):.2f}..{max(arousal):.2f} · mean {mean(arousal):.2f}")

    triggers: dict[str, int] = {}
    for row in metrics:
        for kind in str(row.get("trigger", "")).split(","):
            if kind:
                triggers[kind] = triggers.get(kind, 0) + 1
    if triggers:
        lines.append("wakeups: " + ", ".join(f"{k}×{v}" for k, v in sorted(triggers.items(), key=lambda x: -x[1])))

    meditations = [j for j in journal if j.get("seed_word")]
    lines.append(f"meditations: {len(meditations)} of {len(journal)} journal entries")
    fams = [j.get("reflection", {}).get("familiarity") for j in meditations]
    fams = [f for f in fams if isinstance(f, (int, float)) and not isinstance(f, bool)]
    if fams:
        lines.append(f"avg self-rated familiarity with seed words: {mean(fams):.2f}")

    if meditations:
        lines.append("\nrecent meditations:")
        for entry in meditations[-3:]:
            text = " ".join(str(entry.get("journal", "")).split())
            if len(text) > 160:
                text = text[:159] + "…"
            lines.append(f"  · [{entry.get('seed_word')}] {text}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a conscious_ai run.")
    parser.add_argument("--data-dir", default=None, help="Override data directory.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else load_config().data_dir
    print(summarize(data_dir))


if __name__ == "__main__":
    main()
