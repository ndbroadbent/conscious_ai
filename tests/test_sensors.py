from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
