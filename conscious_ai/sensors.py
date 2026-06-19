from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any

from .events import make_event

try:  # Optional: richer system metrics. Install with `pip install -e .[sensors]`.
    import psutil
except Exception:  # pragma: no cover - environment dependent
    psutil = None

try:  # Optional: microphone volume. Needs sounddevice + numpy + OS mic permission.
    import numpy as np
    import sounddevice as sd
except Exception:  # pragma: no cover - environment dependent
    np = None
    sd = None


# Per-key change thresholds for waking the agent out of its idle loop.
WAKE_THRESHOLDS: dict[str, float] = {
    "cpu_load_percent": 15.0,
    "ram_load_percent": 5.0,
    "load_avg_1m": 0.6,
    "mic_rms_avg": 0.05,
    "mic_peak": 0.12,
}


class MicMonitor:
    """Maintains a rolling window of microphone loudness.

    Stores ONLY loudness (RMS) values, never audio. Computes average and peak
    over the window. Degrades to unavailable if sounddevice/numpy are missing
    or the OS denies microphone access.
    """

    def __init__(self, window_seconds: float = 3.0, samplerate: int = 16000, blocksize: int = 1600) -> None:
        self.window_seconds = window_seconds
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.available = False
        self._stream = None
        self._rms: deque[tuple[float, float]] = deque()

    def start(self) -> None:
        if sd is None or np is None:
            return
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self._callback,
            )
            self._stream.start()
            self.available = True
        except Exception:
            self._stream = None
            self.available = False

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - audio thread
        rms = float(np.sqrt(np.mean(np.square(indata)))) if frames else 0.0
        now = time.monotonic()
        self._rms.append((now, rms))
        cutoff = now - self.window_seconds
        while self._rms and self._rms[0][0] < cutoff:
            self._rms.popleft()

    def read(self) -> dict[str, float] | None:
        if not self.available or not self._rms:
            return None
        values = [rms for _, rms in list(self._rms)]
        return {
            "mic_rms_avg": round(sum(values) / len(values), 4),
            "mic_peak": round(max(values), 4),
        }

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class RealSensorSource:
    """Samples the laptop as the agent's body: cpu, ram, load, microphone."""

    def __init__(self, enable_mic: bool = True, mic_window_seconds: float = 3.0) -> None:
        self.started_at = time.monotonic()
        self.last: dict[str, Any] | None = None
        self.mic = MicMonitor(window_seconds=mic_window_seconds) if enable_mic else None
        if self.mic is not None:
            self.mic.start()
        if psutil is not None:  # prime the cpu_percent baseline so first read is meaningful
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def sample(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        frame: dict[str, Any] = {"clock_monotonic_seconds": round(elapsed, 2)}
        try:
            import os

            if hasattr(os, "getloadavg"):
                frame["load_avg_1m"] = round(os.getloadavg()[0], 3)
        except Exception:
            pass
        if psutil is not None:
            try:
                frame["cpu_load_percent"] = round(psutil.cpu_percent(interval=None), 1)
                frame["ram_load_percent"] = round(psutil.virtual_memory().percent, 1)
            except Exception:
                pass
        if self.mic is not None:
            mic = self.mic.read()
            if mic:
                frame.update(mic)
        return frame

    def event_if_changed(self) -> dict[str, Any] | None:
        return _event_if_changed(self, self.sample())

    def stop(self) -> None:
        if self.mic is not None:
            self.mic.stop()

    def describe(self) -> str:
        bits = []
        if psutil is not None:
            bits.append("cpu+ram")
        bits.append("load_avg")
        if self.mic is not None and self.mic.available:
            bits.append("mic")
        elif self.mic is not None:
            bits.append("mic(unavailable)")
        return ", ".join(bits)


class MockSensorSource:
    """Synthetic stand-in mirroring the real sensor field names.

    Used by `--mock` and tests so the rest of the pipeline (dashboard,
    analysis) behaves identically without real hardware access.
    """

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.last: dict[str, Any] | None = None

    def sample(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        return {
            "clock_monotonic_seconds": round(elapsed, 2),
            "cpu_load_percent": round(max(0.0, min(100.0, 30 + 25 * math.sin(elapsed / 11))), 1),
            "ram_load_percent": round(max(0.0, min(100.0, 55 + 8 * math.sin(elapsed / 29))), 1),
            "load_avg_1m": round(max(0.0, 2.0 + 1.5 * math.sin(elapsed / 13)), 3),
            "mic_rms_avg": round(max(0.0, min(1.0, 0.08 + 0.06 * math.sin(elapsed / 7))), 4),
            "mic_peak": round(max(0.0, min(1.0, 0.18 + 0.12 * math.sin(elapsed / 5))), 4),
        }

    def event_if_changed(self) -> dict[str, Any] | None:
        return _event_if_changed(self, self.sample())

    def stop(self) -> None:
        pass

    def describe(self) -> str:
        return "mock (synthetic cpu/ram/load/mic)"


def _event_if_changed(source: Any, current: dict[str, Any]) -> dict[str, Any] | None:
    previous = source.last
    source.last = current
    if previous is None:
        return make_event("sensor", {"summary": current, "reason": "initial sample"})

    delta: dict[str, float] = {}
    significant = False
    for key, threshold in WAKE_THRESHOLDS.items():
        if key in current and key in previous:
            change = abs(float(current[key]) - float(previous[key]))
            delta[key] = round(change, 4)
            if change >= threshold:
                significant = True
    if significant:
        return make_event("sensor", {"summary": current, "delta": delta})
    return None


def make_sensor_source(config: Any, mock: bool = False) -> Any:
    if mock:
        return MockSensorSource()
    return RealSensorSource(
        enable_mic=getattr(config, "enable_mic", True),
        mic_window_seconds=getattr(config, "mic_window_seconds", 3.0),
    )


async def sensor_loop(
    queue: "asyncio.Queue[dict[str, Any]]",
    source: Any,
    interval_seconds: float,
    cooldown_seconds: float = 0.0,
) -> None:
    last_wake = 0.0
    while True:
        event = source.event_if_changed()
        if event is not None:
            now = time.monotonic()
            initial = event["payload"].get("reason") == "initial sample"
            if initial or now - last_wake >= cooldown_seconds:
                await queue.put(event)
                last_wake = now
        await asyncio.sleep(interval_seconds)
