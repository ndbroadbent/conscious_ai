from __future__ import annotations

import random
import unittest

from conscious_ai.seeds import DEFAULT_WORDS, SeedSource


class SeedTests(unittest.TestCase):
    def test_next_word_comes_from_list(self) -> None:
        source = SeedSource(rng=random.Random(0))
        for _ in range(20):
            self.assertIn(source.next_word(), DEFAULT_WORDS)

    def test_avoids_recent_repeats(self) -> None:
        words = ["a", "b", "c", "d"]
        source = SeedSource(words=words, avoid_recent=3, rng=random.Random(1))
        seen = [source.next_word() for _ in range(4)]
        # With a 3-slot avoid window over 4 words, any 3 consecutive picks are distinct.
        self.assertEqual(len(set(seen[:3])), 3)

    def test_deterministic_with_seeded_rng(self) -> None:
        a = SeedSource(rng=random.Random(42))
        b = SeedSource(rng=random.Random(42))
        self.assertEqual([a.next_word() for _ in range(10)], [b.next_word() for _ in range(10)])

    def test_falls_back_when_word_file_missing(self) -> None:
        source = SeedSource(word_file="/nonexistent/path/words.txt", rng=random.Random(0))
        self.assertTrue(source.words)
        self.assertIn(source.next_word(), DEFAULT_WORDS)

    def test_inspiration_pool_is_preferred_and_consumed(self) -> None:
        source = SeedSource(rng=random.Random(0), novelty_rate=0.0)
        added = source.add_inspiration(["Peanut", "bolt", "peanut", "  ", "shell"])
        self.assertEqual(added, ["peanut", "bolt", "shell"])  # normalized + deduped
        picked = source.next_word()
        self.assertIn(picked, ["peanut", "bolt", "shell"])
        self.assertEqual(source.last_source, "inspiration")
        self.assertNotIn(picked, source.pool)  # consumed

    def test_random_when_pool_empty(self) -> None:
        source = SeedSource(rng=random.Random(0), novelty_rate=0.0)
        self.assertIn(source.next_word(), DEFAULT_WORDS)
        self.assertEqual(source.last_source, "random")

    def test_pool_persists_to_disk(self) -> None:
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "inspiration.json"
        a = SeedSource(pool_path=path)
        a.add_inspiration(["river", "delta"])
        b = SeedSource(pool_path=path)
        self.assertEqual(set(b.pool), {"river", "delta"})

    def test_add_inspiration_handles_non_list(self) -> None:
        source = SeedSource(rng=random.Random(0))
        self.assertEqual(source.add_inspiration(None), [])
        self.assertEqual(source.add_inspiration("solo"), ["solo"])


if __name__ == "__main__":
    unittest.main()
