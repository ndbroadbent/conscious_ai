from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .events import read_recent_jsonl


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>conscious_ai · loop</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0b0d10; color: #d7dde3;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 14px 20px; border-bottom: 1px solid #1b2027;
           display: flex; align-items: baseline; gap: 14px; }
  header h1 { font-size: 15px; margin: 0; letter-spacing: .5px; color: #8ab4f8; }
  header .focus { color: #9aa4af; }
  header .dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; display: inline-block; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 20px; max-width: 1100px; }
  .card { background: #11151a; border: 1px solid #1b2027; border-radius: 10px; padding: 14px 16px; }
  .card h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
             color: #6b7682; margin: 0 0 10px; }
  .gauges { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; }
  .gauge { font-size: 13px; }
  .gauge .k { color: #8b95a1; }
  .gauge .v { float: right; color: #e6edf3; }
  .bar { height: 5px; background: #1b2027; border-radius: 3px; margin-top: 4px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: linear-gradient(90deg, #2f81f7, #56d364); }
  .mood { display: flex; gap: 22px; }
  .mood .big { font-size: 26px; color: #e6edf3; }
  .mood .lbl { color: #6b7682; font-size: 11px; text-transform: uppercase; }
  .stream { grid-column: 1 / -1; max-height: 360px; overflow-y: auto; }
  .entry { border-top: 1px solid #161b22; padding: 10px 0; }
  .entry:first-of-type { border-top: 0; }
  .entry .meta { color: #6b7682; font-size: 11px; margin-bottom: 3px; }
  .entry .seed { color: #d29922; }
  .entry .body { color: #c9d1d9; }
  .entry .refl { color: #7d8590; font-size: 12px; margin-top: 4px; }
  svg { width: 100%; height: 60px; display: block; }
  .empty { color: #6b7682; font-style: italic; }
  .tag { font-size: 11px; color: #6b7682; }
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <h1 id="name">loop</h1>
  <span class="focus" id="focus">—</span>
  <span class="tag" id="cycle"></span>
</header>
<main>
  <section class="card">
    <h2>Mood</h2>
    <div class="mood">
      <div><div class="big" id="valence">0</div><div class="lbl">valence</div></div>
      <div><div class="big" id="arousal">0</div><div class="lbl">arousal</div></div>
      <div><div class="big" id="surprise">—</div><div class="lbl">surprise</div></div>
    </div>
  </section>
  <section class="card">
    <h2>Senses (laptop body)</h2>
    <div class="gauges" id="gauges"></div>
  </section>
  <section class="card">
    <h2>Prediction error</h2>
    <svg id="errchart" viewBox="0 0 300 60" preserveAspectRatio="none"></svg>
  </section>
  <section class="card">
    <h2>Valence over time</h2>
    <svg id="valchart" viewBox="0 0 300 60" preserveAspectRatio="none"></svg>
  </section>
  <section class="card stream">
    <h2>Stream of consciousness</h2>
    <div id="journal"><div class="empty">waiting for the loop to think…</div></div>
  </section>
</main>
<script>
const GAUGES = [
  ["cpu_load_percent", "cpu", 100],
  ["ram_load_percent", "ram", 100],
  ["load_avg_1m", "load avg", 8],
  ["mic_rms_avg", "mic avg", 0.5],
  ["mic_peak", "mic peak", 1],
];
function spark(id, series, color, lo, hi) {
  const el = document.getElementById(id);
  const pts = series.filter(v => v !== null && v !== undefined);
  if (pts.length < 2) { el.innerHTML = '<text x="6" y="34" fill="#6b7682" font-size="11">no data yet</text>'; return; }
  const min = lo ?? Math.min(...pts), max = hi ?? Math.max(...pts);
  const span = (max - min) || 1;
  const d = pts.map((v, i) => {
    const x = (i / (pts.length - 1)) * 300;
    const y = 58 - ((v - min) / span) * 54;
    return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
  }).join(" ");
  el.innerHTML = '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.5"/>';
}
async function tick() {
  let s;
  try { s = await (await fetch("/api/snapshot")).json(); } catch (e) { return; }
  const st = s.state || {};
  const id = st.identity || {}, mood = st.mood || {}, att = st.attention || {};
  document.getElementById("name").textContent = id.name || "loop";
  document.getElementById("focus").textContent = att.focus ? "▸ " + att.focus : "—";
  document.getElementById("cycle").textContent = "cycle " + (st.cycle ?? 0);
  document.getElementById("valence").textContent = (mood.valence ?? 0).toFixed ? (mood.valence ?? 0).toFixed(2) : mood.valence;
  document.getElementById("arousal").textContent = (mood.arousal ?? 0).toFixed ? (mood.arousal ?? 0).toFixed(2) : mood.arousal;
  const err = st.last_prediction_error;
  document.getElementById("surprise").textContent = (err === null || err === undefined) ? "—" : err.toFixed(2);

  const sens = st.sensory_summary || {};
  document.getElementById("gauges").innerHTML = GAUGES.map(([k, lbl, max]) => {
    const v = sens[k];
    if (v === undefined) return "";
    const pct = Math.max(0, Math.min(100, (v / max) * 100));
    const shown = max <= 1 ? v.toFixed(3) : v.toFixed(1);
    return '<div class="gauge"><span class="k">' + lbl + '</span><span class="v">' + shown +
           '</span><div class="bar"><i style="width:' + pct + '%"></i></div></div>';
  }).join("") || '<div class="empty">no sensors</div>';

  spark("errchart", (s.metrics || []).map(m => m.prediction_error), "#f85149", 0, 1);
  spark("valchart", (s.metrics || []).map(m => m.valence), "#3fb950", -1, 1);

  const j = s.journal || [];
  document.getElementById("journal").innerHTML = j.length ? j.slice().reverse().map(e => {
    const r = e.reflection || {};
    const refl = (r.familiarity !== undefined || (r.associations && r.associations.length))
      ? '<div class="refl">familiarity ' + (r.familiarity ?? "?") +
        (r.associations && r.associations.length ? " · " + r.associations.join(", ") : "") +
        (r.uncertainty ? " · unsure: " + r.uncertainty : "") + '</div>'
      : "";
    return '<div class="entry"><div class="meta">cycle ' + e.cycle +
      (e.seed_word ? ' · seed <span class="seed">' + e.seed_word + '</span>' : "") +
      (e.prediction_error != null ? ' · surprise ' + e.prediction_error : "") +
      '</div><div class="body">' + escapeHtml(e.journal) + '</div>' + refl + '</div>';
  }).join("") : '<div class="empty">waiting for the loop to think…</div>';
}
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
tick(); setInterval(tick, 1500);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    data_dir: Path = Path("data")

    def log_message(self, *args: Any) -> None:  # silence per-request logging
        pass

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/snapshot"):
            self._send(json.dumps(self._snapshot()).encode("utf-8"), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _snapshot(self) -> dict[str, Any]:
        state_path = self.data_dir / "state.json"
        state: dict[str, Any] = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        return {
            "state": state,
            "journal": read_recent_jsonl(self.data_dir / "journal.jsonl", 30),
            "metrics": read_recent_jsonl(self.data_dir / "metrics.jsonl", 120),
        }


def serve(data_dir: Path, port: int, background: bool = False) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (_Handler,), {"data_dir": data_dir})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    else:
        server.serve_forever()
    return server


def run() -> None:
    from .config import load_config

    config = load_config()
    print(f"Dashboard: http://127.0.0.1:{config.dashboard_port}  (data: {config.data_dir})")
    serve(config.data_dir, config.dashboard_port, background=False)


if __name__ == "__main__":
    run()
