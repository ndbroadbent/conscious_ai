from __future__ import annotations

import unittest

from conscious_ai.agent import clean_prediction, compute_prediction_error
from conscious_ai.sensors import MockSensorSource


class SensorTests(unittest.TestCase):
    def test_mock_sample_has_expected_fields(self) -> None:
        frame = MockSensorSource().sample()
        for key in ("cpu_load_percent", "ram_load_percent", "load_avg_1m", "mic_rms_avg", "mic_peak"):
            self.assertIn(key, frame)

    def test_first_event_is_initial_sample(self) -> None:
        source = MockSensorSource()
        event = source.event_if_changed()
        self.assertIsNotNone(event)
        self.assertEqual(event["kind"], "sensor")
        self.assertEqual(event["payload"].get("reason"), "initial sample")


class PredictionTests(unittest.TestCase):
    def test_perfect_prediction_has_zero_error(self) -> None:
        frame = {"cpu_load_percent": 40.0, "ram_load_percent": 55.0, "mic_rms_avg": 0.1}
        error = compute_prediction_error(frame, frame)
        self.assertIsNotNone(error)
        self.assertEqual(error["value"], 0.0)

    def test_error_scales_per_key(self) -> None:
        predicted = {"cpu_load_percent": 0.0}
        actual = {"cpu_load_percent": 100.0}
        error = compute_prediction_error(predicted, actual)
        self.assertEqual(error["value"], 1.0)

    def test_no_shared_numeric_keys_returns_none(self) -> None:
        self.assertIsNone(compute_prediction_error({"foo": 1}, {"bar": 2}))

    def test_clean_prediction_keeps_only_known_numeric_keys(self) -> None:
        raw = {"cpu_load_percent": 30, "mic_rms_avg": 0.2, "garbage": "x", "flag": True}
        cleaned = clean_prediction(raw)
        self.assertEqual(cleaned, {"cpu_load_percent": 30.0, "mic_rms_avg": 0.2})


if __name__ == "__main__":
    unittest.main()
