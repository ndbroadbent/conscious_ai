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


if __name__ == "__main__":
    unittest.main()
