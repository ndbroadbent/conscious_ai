from __future__ import annotations

import unittest

from conscious_ai.state import apply_model_patch, diff_states


class StateTests(unittest.TestCase):
    def test_diff_states_records_nested_replace(self) -> None:
        before = {"mood": {"valence": 0.0}, "goals": ["observe"]}
        after = {"mood": {"valence": 0.5}, "goals": ["observe"]}

        self.assertEqual(
            diff_states(before, after),
            [{"op": "replace", "path": "/mood/valence", "value": 0.5}],
        )


    def test_apply_model_patch_allows_state_subtrees_only(self) -> None:
        state = {
            "cycle": 3,
            "updated_at": "old",
            "attention": {"focus": "before"},
            "memory": {"short_term": []},
        }

        patched = apply_model_patch(
            state,
            [{"op": "replace", "path": "/attention/focus", "value": "listening"}],
        )

        self.assertEqual(patched["cycle"], 4)
        self.assertEqual(patched["attention"]["focus"], "listening")
        self.assertEqual(state["attention"]["focus"], "before")


    def test_apply_model_patch_rejects_metadata_edits(self) -> None:
        state = {"cycle": 0, "updated_at": "old", "attention": {"focus": "before"}}

        with self.assertRaisesRegex(ValueError, "not model-editable"):
            apply_model_patch(state, [{"op": "replace", "path": "/cycle", "value": 99}])


if __name__ == "__main__":
    unittest.main()
