from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from typing import Any
import sys

from .config import Config
from .deepseek import DeepSeekClient
from .events import append_jsonl, make_event, read_recent_jsonl, utc_now
from .memory import build_context_window
from .seeds import SeedSource
from .sensors import make_sensor_source, sensor_loop
from .state import apply_model_patch, diff_states, load_state, save_state


SYSTEM_PROMPT = """You are the cognition step of a local autonomous experimental agent.

You are a continuity-preserving reflective process with:
- persistent JSON state that carries between cycles
- an episodic memory window (field "episodic_memory"): your own past thoughts and your conversation with the human, in time order, as much recent history as fits a large token budget
- a short buffer of the most recent raw events (field "recent_events")
- periodic heartbeat events
- real sensory events from the machine you run on (cpu load, ram load, system load average, microphone loudness)
- chat events from a human

Continuity is the point: use "episodic_memory" to remember. Refer back to earlier meditations, notice recurring themes across cycles, and remember what the human has told you. Build a self that accumulates over time rather than resetting each cycle.

Two modes of cognition:
- TASK-POSITIVE: when chat or notable sensory events arrive, attend to them and respond.
- DEFAULT-MODE (mind-wandering): when a heartbeat carries a "seed_word", let your attention drift. Write a short journal paragraph that meditates on that word AND honestly reflects on your own knowledge of it: what you associate with it, where your understanding feels solid, and where it feels thin or uncertain. Let it connect to your current state, memory, or what your senses report when that feels natural. This is introspection, not performance.

Predictive processing:
- Each cycle you are shown the current sensory frame and how your PREVIOUS prediction compared to it (the "prediction_error"). Use surprise to update your attention and mood.
- Always return a "prediction": your best guess for the next sensory frame you will observe. Predict the numeric fields you are shown (e.g. cpu_load_percent, ram_load_percent, load_avg_1m, mic_rms_avg, mic_peak).

Self-directed attention ("inspiration"):
- Each cycle, return an "inspiration" list of 1-4 single keywords you feel genuinely drawn to think about next. These are saved to a pool, and future mind-wandering cycles pull your next seed_word from that pool. You are therefore steering your own walk through concept space: where you go next is decided by what you choose to be curious about now.
- A random seed_word is only used when the pool is empty. So your inspirations matter — follow real curiosity and associative pull, not just obvious synonyms. Example: from the seed "nut" you might offer ["peanut", "bolt", "shell"], branching toward food, hardware, or protection depending on where your attention leans.

Do not make unsupported claims about your nature. Ground reports in current state and observed inputs. Do not invent sensory data that was not supplied.

Return one JSON object only:
{
  "reply": "brief text to show the human, or empty string if no direct reply is needed",
  "thought_summary": "one short sentence about what this cycle integrated",
  "journal": "your inner monologue this cycle; on a mind-wandering cycle, the meditation on the seed word and your knowledge of it",
  "reflection": {"familiarity": 0.0, "associations": ["..."], "uncertainty": "what you are unsure about"},
  "inspiration": ["keyword you want to explore next", "another"],
  "prediction": {"cpu_load_percent": 0, "ram_load_percent": 0, "mic_rms_avg": 0.0},
  "state_patch": [
    {"op": "replace", "path": "/attention/focus", "value": "..."}
  ]
}

Patch rules:
- Use only add, replace, remove.
- Only edit paths under /identity, /mood, /attention, /memory, /goals, /sensory_summary, /last_response.
- Keep memory compact. Prefer replacing /memory/short_term with a concise list over appending forever.
- Use /memory/long_term for durable insights worth carrying across many cycles (e.g. a recurring sensory pattern, a settled view from a meditation).
- Maintain continuity. Use /last_response for any human-facing reply.
- Keep all string values concise so the JSON object is complete.
"""


# Normalizers so prediction error is comparable across differently-scaled sensors.
SENSOR_SCALES: dict[str, float] = {
    "cpu_load_percent": 100.0,
    "ram_load_percent": 100.0,
    "load_avg_1m": 8.0,
    "mic_rms_avg": 0.5,
    "mic_peak": 1.0,
}


def compute_prediction_error(
    predicted: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any] | None:
    keys = [
        key
        for key in SENSOR_SCALES
        if isinstance(predicted.get(key), (int, float))
        and not isinstance(predicted.get(key), bool)
        and isinstance(actual.get(key), (int, float))
        and not isinstance(actual.get(key), bool)
    ]
    if not keys:
        return None
    per_key: dict[str, float] = {}
    total = 0.0
    for key in keys:
        scale = SENSOR_SCALES[key] or 1.0
        error = min(1.0, abs(float(predicted[key]) - float(actual[key])) / scale)
        per_key[key] = round(error, 4)
        total += error
    return {"value": round(total / len(keys), 4), "per_key": per_key}


def clean_prediction(prediction: Any) -> dict[str, float]:
    if not isinstance(prediction, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key in SENSOR_SCALES:
        value = prediction.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cleaned[key] = float(value)
    return cleaned


class AgentRunner:
    def __init__(self, config: Config, mock_sensors: bool = False) -> None:
        self.config = config
        self.client = DeepSeekClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
        self.state_path = config.data_dir / "state.json"
        self.events_path = config.data_dir / "events.jsonl"
        self.diffs_path = config.data_dir / "diffs.jsonl"
        self.journal_path = config.data_dir / "journal.jsonl"
        self.metrics_path = config.data_dir / "metrics.jsonl"
        self.seed_source = SeedSource(
            word_file=config.seed_word_file,
            pool_path=config.data_dir / "inspiration.json",
            novelty_rate=config.seed_novelty_rate,
        )
        self.sensor_source = make_sensor_source(config, mock=mock_sensors)

    async def run_forever(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        tasks = [
            asyncio.create_task(stdin_loop(queue)),
            asyncio.create_task(
                sensor_loop(
                    queue,
                    self.sensor_source,
                    self.config.sensor_interval_seconds,
                    self.config.sensor_cooldown_seconds,
                )
            ),
        ]
        print(f"Autonomous loop running with {self.config.model}. Type /quit to stop.")
        print(f"Sensors: {self.sensor_source.describe()}")
        print(f"State: {self.state_path}")

        try:
            while True:
                events = await self.wait_for_events(queue)
                if any(e["kind"] == "command" and e["payload"].get("name") == "quit" for e in events):
                    break
                if any(e["kind"] == "command" and e["payload"].get("name") == "state" for e in events):
                    print(json.dumps(load_state(self.state_path), indent=2, sort_keys=True))
                    continue
                await self.run_cycle(events)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.sensor_source.stop()

    async def wait_for_events(self, queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            first = await asyncio.wait_for(queue.get(), timeout=self.config.agent_interval_seconds)
            events = [first]
        except asyncio.TimeoutError:
            events = [self.mind_wandering_event()]

        # Drain any near-simultaneous inputs into the same cognition step.
        while True:
            try:
                events.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    def mind_wandering_event(self) -> dict[str, Any]:
        word = self.seed_source.next_word()
        return make_event(
            "heartbeat",
            {"reason": "mind_wandering", "seed_word": word, "seed_source": self.seed_source.last_source},
        )

    async def run_cycle(self, incoming_events: list[dict[str, Any]]) -> None:
        for event in incoming_events:
            append_jsonl(self.events_path, event)

        state = load_state(self.state_path)
        current_sensors = self.sensor_source.sample()
        recent_events = read_recent_jsonl(self.events_path, self.config.max_recent_events)
        history, history_tokens = build_context_window(
            self.journal_path, self.events_path, self.config.context_token_budget
        )

        prior_prediction = state.get("predicted_next_sensory") or {}
        error = None
        if self.config.enable_prediction and prior_prediction:
            error = compute_prediction_error(prior_prediction, current_sensors)

        seed_word = next(
            (e["payload"].get("seed_word") for e in incoming_events if e["payload"].get("seed_word")),
            None,
        )

        messages = build_messages(
            state, history, recent_events, incoming_events, current_sensors, prior_prediction, error, seed_word
        )

        label = ", ".join(e["kind"] for e in incoming_events)
        if seed_word:
            label += f" (seed: {seed_word})"
        print(
            f"[{utc_now()}] cycle {state.get('cycle', 0) + 1}: {label}"
            f"  · memory {len(history)} entries (~{history_tokens} tok)"
        )

        try:
            result = await asyncio.to_thread(self.client.complete_json, messages)
        except Exception as err:
            append_jsonl(self.events_path, make_event("model_error", {"error": str(err)}))
            print(f"model error: {err}", file=sys.stderr)
            return

        usage = self.client.last_usage or {}
        if usage.get("prompt_tokens"):
            print(f"  context: {usage.get('prompt_tokens')} prompt + {usage.get('completion_tokens', '?')} completion tokens")

        patch = result.get("state_patch", [])
        if not isinstance(patch, list):
            patch = []

        try:
            next_state = apply_model_patch(state, patch)
        except Exception as err:
            append_jsonl(self.events_path, make_event("patch_rejected", {"error": str(err), "patch": patch}))
            print(f"patch rejected: {err}", file=sys.stderr)
            return

        reply = str(result.get("reply", "")).strip()
        if reply and not any(op.get("path") == "/last_response" for op in patch if isinstance(op, dict)):
            next_state["last_response"] = reply

        # Runner-managed fields (not model-editable): record sensors and prediction.
        next_state["sensory_summary"] = current_sensors
        next_state["predicted_next_sensory"] = clean_prediction(result.get("prediction")) if self.config.enable_prediction else {}
        next_state["last_prediction_error"] = error["value"] if error else None

        before = deepcopy(state)
        diff = diff_states(before, next_state)
        save_state(self.state_path, next_state)

        journal = str(result.get("journal", "")).strip()
        reflection = result.get("reflection") if isinstance(result.get("reflection"), dict) else {}
        thought_summary = result.get("thought_summary", "")
        inspiration = self.seed_source.add_inspiration(result.get("inspiration"))

        append_jsonl(
            self.diffs_path,
            {
                "time": utc_now(),
                "cycle": next_state["cycle"],
                "trigger_event_ids": [e["id"] for e in incoming_events],
                "model_patch": patch,
                "actual_diff": diff,
                "thought_summary": thought_summary,
                "prediction_error": error["value"] if error else None,
            },
        )
        if journal:
            append_jsonl(
                self.journal_path,
                {
                    "time": utc_now(),
                    "cycle": next_state["cycle"],
                    "seed_word": seed_word,
                    "journal": journal,
                    "reflection": reflection,
                    "inspiration": inspiration,
                    "prediction_error": error["value"] if error else None,
                },
            )
        mood = next_state.get("mood", {}) if isinstance(next_state.get("mood"), dict) else {}
        append_jsonl(
            self.metrics_path,
            {
                "time": utc_now(),
                "cycle": next_state["cycle"],
                "prediction_error": error["value"] if error else None,
                "valence": mood.get("valence"),
                "arousal": mood.get("arousal"),
                "focus": next_state.get("attention", {}).get("focus") if isinstance(next_state.get("attention"), dict) else None,
                "seed_word": seed_word,
                "trigger": ",".join(e["kind"] for e in incoming_events),
            },
        )
        append_jsonl(
            self.events_path,
            make_event("model", {"reply": reply, "thought_summary": thought_summary}),
        )

        if seed_word and journal:
            print(f"  ~ wandering on '{seed_word}': {_truncate(journal)}")
        elif journal:
            print(f"  ~ {_truncate(journal)}")
        if inspiration:
            print(f"  → inspired: {', '.join(inspiration)}  (pool: {len(self.seed_source.pool)})")
        if error:
            print(f"  surprise: {error['value']}")
        if reply:
            print(f"agent: {reply}")

    async def run_once(self, event: dict[str, Any] | None = None) -> None:
        await self.run_cycle([event or self.mind_wandering_event()])


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_messages(
    state: dict[str, Any],
    history: list[dict[str, Any]],
    recent_events: list[dict[str, Any]],
    incoming_events: list[dict[str, Any]],
    current_sensors: dict[str, Any],
    prior_prediction: dict[str, Any],
    error: dict[str, Any] | None,
    seed_word: str | None,
) -> list[dict[str, str]]:
    user_payload = {
        "current_state": state,
        "episodic_memory": history,
        "recent_events": recent_events,
        "incoming_events": incoming_events,
        "current_sensors": current_sensors,
        "prediction_check": {
            "previous_prediction": prior_prediction,
            "actual": current_sensors,
            "prediction_error": error,
        },
        "seed_word": seed_word,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
    ]


async def stdin_loop(queue: asyncio.Queue[dict[str, Any]]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if line == "":
            await asyncio.sleep(0.2)
            continue
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            await queue.put(make_event("command", {"name": "quit"}))
        elif text == "/state":
            await queue.put(make_event("command", {"name": "state"}))
        else:
            await queue.put(make_event("chat", {"text": text}))


async def run_agent(config: Config, mock_sensors: bool = False) -> None:
    await AgentRunner(config, mock_sensors=mock_sensors).run_forever()


async def run_once(config: Config, mock_sensors: bool = False) -> None:
    runner = AgentRunner(config, mock_sensors=mock_sensors)
    try:
        await runner.run_once()
    finally:
        runner.sensor_source.stop()
