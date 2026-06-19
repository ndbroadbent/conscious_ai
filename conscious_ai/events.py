from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "time": utc_now(),
        "kind": kind,
        "payload": payload or {},
    }


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_recent_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []

    recent: deque[dict[str, Any]] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                recent.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(recent)

