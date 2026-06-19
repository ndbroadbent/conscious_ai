from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Iterable


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


class SeedSource:
    """Supplies random word seeds for default-mode (mind-wandering) cycles."""

    def __init__(
        self,
        words: Iterable[str] | None = None,
        avoid_recent: int = 40,
        word_file: str | None = None,
        rng: random.Random | None = None,
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

    def next_word(self) -> str:
        choices = [w for w in self.words if w not in self.recent] or self.words
        word = self.rng.choice(choices)
        self.recent.append(word)
        return word
