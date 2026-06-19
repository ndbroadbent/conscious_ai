from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    data_dir: Path = Path("data")
    agent_interval_seconds: float = 10.0
    sensor_interval_seconds: float = 3.0
    sensor_cooldown_seconds: float = 8.0
    max_recent_events: int = 24
    timeout_seconds: float = 60.0
    enable_mic: bool = True
    mic_window_seconds: float = 3.0
    enable_prediction: bool = True
    seed_word_file: str | None = None
    dashboard_port: int = 8765


def load_config() -> Config:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing. Add it to .env or export it.")

    return Config(
        api_key=api_key,
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        agent_interval_seconds=float(os.environ.get("AGENT_INTERVAL_SECONDS", "10")),
        sensor_interval_seconds=float(os.environ.get("SENSOR_INTERVAL_SECONDS", "3")),
        sensor_cooldown_seconds=float(os.environ.get("SENSOR_COOLDOWN_SECONDS", "8")),
        max_recent_events=int(os.environ.get("MAX_RECENT_EVENTS", "24")),
        timeout_seconds=float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "60")),
        enable_mic=_env_bool("ENABLE_MIC", True),
        mic_window_seconds=float(os.environ.get("MIC_WINDOW_SECONDS", "3")),
        enable_prediction=_env_bool("ENABLE_PREDICTION", True),
        seed_word_file=os.environ.get("SEED_WORD_FILE") or None,
        dashboard_port=int(os.environ.get("DASHBOARD_PORT", "8765")),
    )
