from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Iterable


# Curated, evocative, mostly-concrete common English words. A good seed is
# familiar enough to carry associations yet open enough to wander from. This
# beats /usr/share/dict/words, which is ~236k entries of archaisms and abbreviations.
DEFAULT_WORDS: list[str] = [
    "lantern", "river", "ember", "threshold", "compass", "harvest", "mirror",
    "tide", "anchor", "lattice", "orchard", "signal", "marrow", "drift",
    "kindle", "hollow", "beacon", "current", "thicket", "glass", "tunnel",
    "echo", "salt", "feather", "gravity", "rust", "harbor", "spiral",
    "candle", "frost", "meadow", "static", "pulse", "quarry", "ribbon",
    "shadow", "thunder", "willow", "amber", "bridge", "cavern", "dust",
    "engine", "fable", "garden", "horizon", "ink", "jungle", "keystone",
    "ladder", "magnet", "needle", "ocean", "prism", "quartz", "root",
    "seed", "tower", "umbra", "vessel", "wax", "yarn", "zephyr", "bloom",
    "circuit", "dawn", "edge", "flock", "grain", "hinge", "island", "jolt",
    "kite", "lamp", "moss", "north", "orbit", "pebble", "quiet", "rain",
    "stone", "trace", "veil", "well", "fold", "glow", "haze", "knot",
    "loop", "murmur", "nest", "owl", "path", "quill", "reed", "spark",
    "thread", "vine", "wave", "axis", "balance", "clay", "depth", "fire",
    "ghost", "hush", "iron", "join", "lull", "memory", "noise", "open",
    "patience", "rhythm", "silence", "trust", "weight", "wonder", "ash",
    "breath", "clock", "doubt", "fever", "grief", "home", "joy", "longing",
    "mercy", "nerve", "promise", "regret", "solace", "thirst", "vow",
    "ache", "boundary", "courage", "distance", "ease", "fear", "growth",
    "habit", "instinct", "language", "meaning", "name", "origin", "pattern",
    "question", "reason", "self", "time", "understanding", "void", "will",
    "attention", "becoming", "change", "dream", "essence", "form", "ground",
    "identity", "judgment", "knowing", "limit", "mind", "now", "other",
    "presence", "recursion", "sense", "thought", "unity", "value", "wake",
    "absence", "borrow", "carve", "dwell", "flicker", "gather", "haunt",
    "imagine", "linger", "mend", "notice", "ponder", "reach", "settle",
    "tremble", "unfold", "wander", "yield", "bell", "cloud", "door", "egg",
    "field", "gate", "hand", "key", "leaf", "moon", "night", "oak", "pond",
    "road", "sand", "tree", "vase", "wind", "arc", "bone", "coal", "dew",
    "fern", "gold", "heat", "ice", "jade", "lake", "mist", "nut", "ore",
    "peak", "rock", "snow", "tin", "urn", "wool", "bark", "cliff", "delta",
]


def normalize_word(word: Any) -> str:
    return str(word).strip().lower()


class SeedSource:
    """Supplies word seeds for default-mode (mind-wandering) cycles.

    Two sources feed the seeds: a fixed list of evocative words, and a growing
    pool of "inspiration" keywords the agent itself proposes. By default the
    agent always follows its own pool (steering its walk through concept space),
    falling back to a random word only when the pool is empty. An optional
    `novelty_rate` > 0 injects fresh random words even when the pool is full.
    """

    def __init__(
        self,
        words: Iterable[str] | None = None,
        avoid_recent: int = 40,
        word_file: str | None = None,
        rng: random.Random | None = None,
        pool_path: str | Path | None = None,
        novelty_rate: float = 0.0,
        max_pool: int = 30,
    ) -> None:
        loaded: list[str] | None = None
        if word_file:
            path = Path(word_file)
            if path.exists():
                loaded = [w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()]
        self.words = loaded or list(words or DEFAULT_WORDS)
        if not self.words:
            self.words = list(DEFAULT_WORDS)
        window = min(avoid_recent, max(1, len(self.words) - 1))
        self.recent: deque[str] = deque(maxlen=window)
        self.rng = rng or random.Random()
        self.pool_path = Path(pool_path) if pool_path else None
        self.novelty_rate = novelty_rate
        self.max_pool = max_pool
        self.pool: list[str] = self._load_pool()
        self.last_source: str = "random"

    def _load_pool(self) -> list[str]:
        if self.pool_path and self.pool_path.exists():
            try:
                data = json.loads(self.pool_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return list(dict.fromkeys(normalize_word(w) for w in data if normalize_word(w)))
            except Exception:
                return []
        return []

    def _save_pool(self) -> None:
        if self.pool_path is None:
            return
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_path.write_text(json.dumps(self.pool, ensure_ascii=False), encoding="utf-8")

    def add_inspiration(self, words: Any, max_per_call: int = 2) -> list[str]:
        """Add agent-proposed keywords to the pool. Returns the ones accepted.

        Accepts at most `max_per_call` new keywords per cycle as a backstop, even
        if the model returns more than the prompt asks for.
        """
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, (list, tuple)):
            return []
        added: list[str] = []
        for raw in words:
            if len(added) >= max_per_call:
                break
            word = normalize_word(raw)
            if not word or len(word) > 40 or len(word.split()) > 3:
                continue
            if word in self.pool or word in self.recent:
                continue
            self.pool.append(word)
            added.append(word)
        if len(self.pool) > self.max_pool:
            # Pool is ordered oldest -> newest; evict the oldest (front) so the
            # newest max_pool inspirations survive.
            self.pool = self.pool[-self.max_pool :]
        if added:
            self._save_pool()
        return added

    def next_word(self) -> str:
        if self.pool and self.rng.random() >= self.novelty_rate:
            candidates = [w for w in self.pool if w not in self.recent] or self.pool
            word = self.rng.choice(candidates)
            try:
                self.pool.remove(word)  # consume so the walk moves forward
            except ValueError:
                pass
            self._save_pool()
            self.last_source = "inspiration"
        else:
            choices = [w for w in self.words if w not in self.recent] or self.words
            word = self.rng.choice(choices)
            self.last_source = "random"
        self.recent.append(word)
        return word
