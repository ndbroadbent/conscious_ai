# Conscious AI Local Experiment

A small local loop for experimenting with a persistent, stateful AI process —
and an open question: if you give a model continuity, a body, and a default
mode, and let it run, what emerges? Consciousness?

Consciousness isn't directly measurable, so rather than assert or deny it,
the loop instruments what we *can* observe — continuity of identity, self-modeling,
the texture of its mind-wandering, the concepts it chooses to chase — and leaves the interpretation to you.
The code gives the model a body, a default mode, and a measurable inner life:

- a persistent JSON state file (identity, mood, attention, memory, goals)
- an append-only event log, diff log, journal, and metrics log
- a **rolling episodic memory**: its own past thoughts + the conversation, fed
  back each cycle up to a ~100k-token budget (oldest trimmed) — real continuity
- **self-directed attention**: each cycle it emits "inspiration" keywords it
  wants to explore; those seed future mind-wandering, so it walks concept space
  in directions it chooses rather than purely at random
- **real sensory input from your laptop** — cpu load, ram load, system load
  average, and microphone loudness (RMS average + peak over a window) — the
  agent's "body" that it senses and reacts to
- a **default mode / mind-wandering** behavior: when idle, the agent is given a
  random English word as a seed and meditates on it, reflecting on the limits of
  its own knowledge
- autonomous wakeups, immediate wakeups from chat, and wakeups when the senses
  change enough to matter

DeepSeek is used through its OpenAI-compatible `/chat/completions` endpoint.

## Setup

Your `.env` already has:

```sh
DEEPSEEK_API_KEY=...
```

The core runs with **zero dependencies**. For richer sensors install the extra:

```sh
pip install -e '.[sensors]'   # psutil (cpu/ram) + sounddevice + numpy (mic)
```

- Without `psutil`, the agent falls back to load average only.
- Without `sounddevice`, the microphone is simply absent.
- The **first run that uses the mic triggers a macOS microphone-permission
  prompt** for your terminal. Only loudness (RMS) is ever computed — no audio is
  recorded or stored. Disable the mic any time with `ENABLE_MIC=false`.

Optional `.env` values:

```sh
DEEPSEEK_MODEL=deepseek-v4-flash
AGENT_INTERVAL_SECONDS=10      # idle heartbeat cadence
SENSOR_INTERVAL_SECONDS=3      # how often sensors are sampled
SENSOR_COOLDOWN_SECONDS=8      # min gap between sensor-triggered wakeups (cost control)
ENABLE_MIC=true
MIC_WINDOW_SECONDS=3
CONTEXT_TOKEN_BUDGET=100000    # episodic memory window resent each cycle
SEED_WORD_FILE=               # optional: newline-separated word list for seeds
SEED_NOVELTY_RATE=0.15        # chance of a fresh random seed even when the inspiration pool is full
DASHBOARD_PORT=8765
DATA_DIR=data
```

## Run

```sh
python3 -m conscious_ai                 # real sensors
python3 -m conscious_ai --dashboard     # + live dashboard at http://127.0.0.1:8765
python3 -m conscious_ai --mock          # synthetic sensors, no hardware/mic needed
```

Type messages and press Enter to talk to it. The agent also wakes on its own and
mind-wanders on a seed word; loud noise or a CPU spike can wake it earlier.

> Cost note: this is an always-on loop making one model call per wakeup, and
> each wakeup resends the episodic memory (up to ~`CONTEXT_TOKEN_BUDGET` tokens),
> so input cost grows as memory fills. Lower the budget or raise the intervals to
> spend less.

Useful commands while running:

```text
/state
/quit
```

## Watch it think

```sh
python3 -m conscious_ai.dashboard
```

A zero-dependency dashboard (stdlib `http.server`) showing the current focus and
mood, live sensor gauges, the valence sparkline, the inspiration pool, and the
stream of consciousness with each meditation's self-rated familiarity.

## Analyze a run

```sh
python3 -m conscious_ai.analyze
```

Prints summary stats over the logs: the concept walk (distinct inspirations,
most-revisited keywords, pending pool), mood range, wakeup breakdown, and recent
meditations with average self-rated familiarity.

## Data files (under `data/`)

- `state.json` — current agent state
- `events.jsonl` — sensory, chat, heartbeat, and model events
- `diffs.jsonl` — state patches applied per cycle
- `journal.jsonl` — the inner monologue / meditations + self-reflection + inspiration
- `metrics.jsonl` — per-cycle mood, focus, trigger (drives the charts)
- `inspiration.json` — the pool of keywords it chose to explore next (its concept walk)

## Tests

```sh
python3 -m unittest discover -s tests
```

Run a single cognition cycle (one API call) and exit:

```sh
python3 -m conscious_ai --once --mock
```

## What's next

Architected so the next steps drop in cleanly:

- **A friend** — a second agent with a distinct disposition that shares this same
  body (sensors); when both are idle they discuss the seed word instead of
  meditating alone. The event model already supports a `peer_message` route.
- Memory **retrieval** over the full log, and a "sleep" consolidation cycle.
