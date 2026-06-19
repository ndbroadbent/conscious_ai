from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from typing import Any

from .events import utc_now


DEFAULT_STATE: dict[str, Any] = {
    "schema_version": 1,
    "created_at": None,
    "updated_at": None,
    "cycle": 0,
    "identity": {
        "name": "Loop",
        "self_description": "A local experimental agent with persistent state and sensory inputs.",
    },
    "mood": {
        "valence": 0.0,
        "arousal": 0.1,
        "labels": ["new"],
    },
    "attention": {
        "focus": "initializing",
        "salience": [],
    },
    "memory": {
        "short_term": [],
        "long_term": [],
    },
    "goals": [
        {
            "name": "maintain continuity",
            "status": "active",
            "notes": "Preserve useful context between cycles.",
        }
    ],
    "sensory_summary": {},
    "predicted_next_sensory": {},
    "last_prediction_error": None,
    "last_response": "",
}

ALLOWED_PATCH_PREFIXES = (
    "/identity",
    "/mood",
    "/attention",
    "/memory",
    "/goals",
    "/sensory_summary",
    "/last_response",
)
MAX_LIST_ITEMS = 80
MAX_STRING_LENGTH = 4000


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        now = utc_now()
        state = deepcopy(DEFAULT_STATE)
        state["created_at"] = now
        state["updated_at"] = now
        save_state(path, state)
        return state
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as file:
        file.write(rendered)
        temp_name = file.name
    Path(temp_name).replace(path)


def diff_states(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        patches: list[dict[str, Any]] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys):
            patches.append({"op": "remove", "path": join_pointer(path, key)})
        for key in sorted(after_keys - before_keys):
            patches.append({"op": "add", "path": join_pointer(path, key), "value": after[key]})
        for key in sorted(before_keys & after_keys):
            patches.extend(diff_states(before[key], after[key], join_pointer(path, key)))
        return patches

    if before != after:
        return [{"op": "replace", "path": path or "/", "value": after}]
    return []


def apply_model_patch(state: dict[str, Any], patch: list[dict[str, Any]]) -> dict[str, Any]:
    next_state = deepcopy(state)
    for op in patch:
        validate_patch_op(op)
        operation = op["op"]
        pointer = op["path"]
        if operation == "remove":
            remove_pointer(next_state, pointer)
        elif operation in {"add", "replace"}:
            set_pointer(next_state, pointer, sanitize_value(op.get("value")), create=operation == "add")
    next_state["cycle"] = int(state.get("cycle", 0)) + 1
    next_state["updated_at"] = utc_now()
    return next_state


def validate_patch_op(op: dict[str, Any]) -> None:
    if op.get("op") not in {"add", "replace", "remove"}:
        raise ValueError(f"Unsupported patch op: {op!r}")
    path = op.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"Patch path must be a JSON pointer: {op!r}")
    if not any(path == prefix or path.startswith(prefix + "/") for prefix in ALLOWED_PATCH_PREFIXES):
        raise ValueError(f"Patch path is not model-editable: {path}")
    if op["op"] in {"add", "replace"} and "value" not in op:
        raise ValueError(f"Patch op requires value: {op!r}")


def join_pointer(base: str, key: str) -> str:
    encoded = key.replace("~", "~0").replace("/", "~1")
    return f"{base}/{encoded}" if base else f"/{encoded}"


def split_pointer(pointer: str) -> list[str]:
    if pointer == "/":
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.strip("/").split("/")]


def set_pointer(doc: Any, pointer: str, value: Any, create: bool) -> None:
    parts = split_pointer(pointer)
    if not parts:
        raise ValueError("Replacing the whole state is not allowed")
    parent = doc
    for part in parts[:-1]:
        parent = descend(parent, part)
    key = parts[-1]
    if isinstance(parent, dict):
        if not create and key not in parent:
            raise ValueError(f"Cannot replace missing key at {pointer}")
        parent[key] = value
        return
    if isinstance(parent, list):
        index = len(parent) if key == "-" else int(key)
        if create:
            parent.insert(index, value)
        else:
            parent[index] = value
        trim_list(parent)
        return
    raise ValueError(f"Cannot set pointer on {type(parent).__name__}")


def remove_pointer(doc: Any, pointer: str) -> None:
    parts = split_pointer(pointer)
    if not parts:
        raise ValueError("Removing the whole state is not allowed")
    parent = doc
    for part in parts[:-1]:
        parent = descend(parent, part)
    key = parts[-1]
    if isinstance(parent, dict):
        parent.pop(key, None)
        return
    if isinstance(parent, list):
        del parent[int(key)]
        return
    raise ValueError(f"Cannot remove pointer on {type(parent).__name__}")


def descend(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key not in value:
            raise ValueError(f"Missing key in patch path: {key}")
        return value[key]
    if isinstance(value, list):
        return value[int(key)]
    raise ValueError(f"Cannot descend into {type(value).__name__}")


def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if isinstance(value, list):
        sanitized = [sanitize_value(item) for item in value[:MAX_LIST_ITEMS]]
        return sanitized
    if isinstance(value, dict):
        return {str(key)[:128]: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:MAX_STRING_LENGTH]


def trim_list(items: list[Any]) -> None:
    if len(items) > MAX_LIST_ITEMS:
        del items[:-MAX_LIST_ITEMS]

