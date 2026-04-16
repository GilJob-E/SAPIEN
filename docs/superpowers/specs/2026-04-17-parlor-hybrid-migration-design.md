# Parlor-style Hybrid Multimodal Migration — Design Spec

**Date:** 2026-04-17
**Author:** hoddukzoa (brainstorming partner: Claude)
**Status:** Design approved, pending implementation plan
**Supersedes:** Partial — maintains Legacy Gemini-cloud path as fallback

---

## 1. Context

### Current architecture
The giljob-e interview platform currently uses a 4-stage serial cloud pipeline:

```
User speech → [Gemini STT upload] → [Gemini Vision upload] → [Gemini LLM] → [ElevenLabs TTS] → Audio
```

End-to-end latency: **≈ 5 s** (measured baseline on step5-1 branch). Primary bottlenecks are:

1. Entire WAV file upload after utterance ends (blocking)
2. Separate Gemini Vision API call for each webcam frame (separate RTT)
3. Four independent network round-trips per turn
4. Per-call model cold-path overhead

### Goal
Reduce end-to-end latency to **≤ 3 s** (p95) while preserving Korean interview-quality output.

### Non-goals
- Full offline operation (ElevenLabs remains cloud)
- Kokoro TTS (risk to Korean voice quality)
- Complete Flask → FastAPI port (scope creep)
- Multi-user concurrent sessions (single-user demo scope)

---

## 2. Decisions Locked

| Axis | Decision | Rationale |
|------|----------|-----------|
| Primary motivation | **Latency ≤ 3 s** | Demo UX; other benefits are secondary |
| Architecture approach | **Parlor-style unified multimodal pipeline** | Collapses 4 serial API calls into one streaming inference |
| Model strategy | **Hybrid: Gemma 4 E2B local (STT+Vision+LLM) + ElevenLabs cloud TTS** | Gemma bundles three stages into one pass; ElevenLabs keeps Korean voice quality safe |
| Rewrite scope | **Surgical Swap — Flask untouched, new FastAPI service on :8000** | Minimizes risk to existing OAuth/DB/tests; independent rollback |
| Hardware target | **M2 MacBook (dev = demo machine)** | Apple Silicon Metal GPU via LiteRT-LM; same machine for demo eliminates portability risk |
| Validation gate | **Pre-migration smoke tests must pass** | No-go criteria prevent sunk-cost loss |

---

## 3. Architecture Overview

### 3.1 Two-process topology

```
                           ┌──────────────────────────────────┐
                           │         Browser (기존 UI)        │
                           │  ┌─────────────────────────────┐ │
                           │  │ chat.html / meeting UI      │ │
                           │  │ - Google OAuth button        │ │
                           │  │ - mic/camera permissions     │ │
                           │  │ - Silero VAD (WASM)  ← NEW   │ │
                           │  │ - PCM + JPEG WS send ← NEW   │ │
                           │  │ - TTS audio playback         │ │
                           │  └─────────────────────────────┘ │
                           └────────┬──────────────┬──────────┘
                                    │              │
                    HTTP/HTTPS      │              │ WSS (binary)
                    cookie session  │              │ PCM + JPEG + ctrl
                                    ▼              ▼
          ┌──────────────────────────┐   ┌──────────────────────────┐
          │  Flask @ :5001 (existing) │   │  FastAPI @ :8000 (NEW)   │
          │  ─────────────────        │   │  ─────────────────       │
          │  · Google OAuth           │   │  · WebSocket /ws/iv/{id} │
          │  · DB (meeting, user)     │   │  · Flask session verify  │
          │  · HTML templates         │   │  · Gemma 4 E2B loaded    │
          │  · /feedback, /admin      │   │  · Multimodal loop       │
          │  · Existing 34 tests      │   │  · ElevenLabs streaming  │
          │  · Fallback /meeting/msg  │   │  · Dialog history in-mem │
          └──────────────┬───────────┘   └──────────┬───────────────┘
                         │                          │
                         └──────┬───────────────────┘
                                │ Internal HTTP + HMAC
                                │ (/internal/meeting/...)
                                ▼
                    ┌──────────────────────┐
                    │ Shared state          │
                    │ · SECRET_KEY (env)    │
                    │ · DB (Flask-owned)    │
                    │ · INTERNAL_API_SECRET │
                    └──────────────────────┘
```

### 3.2 Key properties

- **Independent processes**: Flask and FastAPI run as separate supervised processes. Local dev: `make dev` starts both.
- **Browser dual-connection**: HTTP to Flask for pages/auth; WSS to FastAPI for realtime I/O.
- **Session bridging**: Shared Flask `SECRET_KEY` lets FastAPI verify Flask's session cookie without hitting Flask on every request.
- **Graceful fallback**: Browser falls back to Legacy Flask path automatically on FastAPI failure.
- **Gemma cold-load once**: FastAPI loads Gemma 4 E2B at startup (~3 GB RAM), runs one warmup inference.

---

## 4. Components

### 4.1 Directory structure

```
giljob-e/SAPIEN/
├── start_app/                        # Existing Flask, minimal edits
│   ├── dialogue_manager/
│   │   ├── meeting.py                # Export session history API
│   │   ├── llm.py                    # Retained for Legacy fallback
│   │   └── speech2text.py,
│   │       text2speech.py            # Retained for Legacy fallback
│   ├── templates/chat.html           # WS client hook
│   ├── static/js/chat.js             # Feature flag dispatch
│   └── tests/                        # All 34 tests preserved
│
├── start_app_ws/                     # 🆕 FastAPI multimodal server
│   ├── __init__.py
│   ├── main.py                       # FastAPI app + WS route
│   ├── gemma_engine.py               # LiteRT-LM wrapper
│   ├── elevenlabs_streamer.py        # ElevenLabs streaming TTS
│   ├── session_manager.py            # Dialog history + Flask bridge
│   ├── audio_protocol.py             # PCM/JPEG/Control frame parser
│   ├── auth_bridge.py                # Flask session cookie verify
│   └── tests/
│       ├── unit/
│       └── integration/
│
├── static/js/realtime/               # 🆕 Browser realtime modules
│   ├── vad-worker.js                 # Silero VAD WASM wrapper
│   ├── audio-capture.js              # AudioWorklet 16 kHz PCM
│   ├── frame-capture.js              # video → canvas → JPEG
│   ├── ws-client.js                  # FastAPI WS connect/reconnect
│   └── audio-player.js               # Web Audio gapless playback
│
├── models/                           # 🆕 gitignored
│   └── gemma-4-e2b/                  # LiteRT-LM model files
│
├── scripts/                          # 🆕 smoke tests, bench, preflight
│   ├── smoke_gemma_korean.py
│   ├── smoke_gemma_multimodal.py
│   ├── bench_latency.py
│   ├── preflight.sh
│   └── e2e_local.py
│
├── docs/superpowers/specs/           # 🆕 this file
│
├── requirements.txt                  # Existing (Flask side)
└── requirements-ws.txt               # 🆕 FastAPI side isolated
```

### 4.2 New Python dependencies (`requirements-ws.txt`)

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP + WebSocket framework |
| `uvicorn[standard]` | ASGI server with `websockets` |
| `litert-lm` | Gemma 4 E2B local inference (Apple Silicon Metal) |
| `numpy` | Audio array processing |
| `Pillow` | JPEG decode |
| `pydantic` | WS message schema |
| `itsdangerous` | Flask session cookie verification |
| `httpx` | Async HTTP client for Flask internal API |
| `elevenlabs` (existing) | TTS streaming client |

### 4.3 New browser dependencies

| Package | Purpose |
|---------|---------|
| `@ricky0123/vad-web` | Silero VAD (WASM) |
| AudioWorklet (builtin) | 16 kHz mono PCM capture |
| Canvas + MediaStream (builtin) | Camera → JPEG |

Bundling: CDN `<script type="module">` (no build step) for MVP.

### 4.4 FastAPI endpoints

- `GET /health` — JSON `{"gemma_loaded": bool, "version": "..."}`
- `WS /ws/interview/{session_id}` — main multimodal channel
- `GET /metrics` (optional) — rolling p50/p95 latency

Out of scope for FastAPI: OAuth, DB, HTML rendering, feedback. All remain Flask-owned.

### 4.5 Gemma engine interface

```python
# start_app_ws/gemma_engine.py
class GemmaEngine:
    def __init__(self, model_path: str, device: str = "metal"):
        self.model = LiteRT.load(model_path)  # cold load, ~3 GB RAM
        self._warmup()

    async def stream_respond(
        self,
        audio_pcm: bytes,           # 16 kHz mono int16 PCM (full utterance)
        latest_frame_jpeg: bytes,   # Most recent webcam frame
        history: list[Turn],        # Dialog history
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Streams tokens. Caller detects sentence boundaries."""
        async for token in self.model.stream_generate(...):
            yield token
```

### 4.6 Browser module responsibilities

| Module | Responsibility |
|--------|----------------|
| `vad-worker.js` | Mic audio realtime VAD → utterance start/end events |
| `audio-capture.js` | AudioWorklet → 16 kHz mono PCM chunks (20 ms = 640 B) |
| `frame-capture.js` | Capture JPEG on utterance end (quality 70, 512×384) |
| `ws-client.js` | Binary frame send, JSON control, exponential backoff reconnect |
| `audio-player.js` | ElevenLabs chunks → Web Audio gapless playback |

---

## 5. Data Flow & Protocol

### 5.1 End-to-end timeline (latency budget)

```
Time(s)  Browser                        FastAPI :8000                   Gemma 4 E2B      ElevenLabs
0.00     User utterance end (VAD)
         → WS {type:"utterance_end"}
         → binary [0x02][final JPEG]
0.05                                    Assemble PCM + JPEG + history
0.10                                    Gemma.stream_respond() →        inference start
1.00                                                                    ← first sentence
1.05                                    Sentence boundary detected
                                        → ElevenLabs streaming TTS ──────────────────→
1.30                                                                                   ← first audio chunk
1.35                                    ← binary [0x03][audio] ───
1.40     Playback begins 🔊
         ▶ Perceived latency ≈ 1.4 s  ✅ (target 3 s, margin 1.6 s)
1.40~3.50                               Continue generating → pipe into TTS
                                        Forward audio chunks to browser continuously
3.50     AI utterance ends
         ← WS {type:"turn_end"}
```

**Note:** M2 first-token time is an estimate. Gate #3 validates the actual value on target hardware.

### 5.2 WebSocket protocol

**Binary frames** (high-volume data):

| Type tag | Direction | Content |
|----------|-----------|---------|
| `0x01` | Browser → Server | PCM chunk (16 kHz mono int16, 20 ms = 640 B) |
| `0x02` | Browser → Server | JPEG frame (variable length) |
| `0x03` | Server → Browser | AI audio chunk (mp3_44100_128 from ElevenLabs) |

**JSON text frames** (control):

Browser → Server:
```json
{"type": "session_init", "session_id": "abc123", "language": "ko"}
{"type": "utterance_start"}
{"type": "utterance_end"}
{"type": "interrupt"}
```

Server → Browser:
```json
{"type": "ready"}
{"type": "response_text", "sentence": "안녕하세요!"}
{"type": "response_audio_start"}
{"type": "turn_end", "latency_ms": 1423}
{"type": "error", "code": "gemma_timeout", "msg": "..."}
{"type": "shutdown", "resume_in_sec": 30}
```

### 5.3 Sentence boundary detection

```python
buf = []
async for token in gemma.stream_respond(...):
    buf.append(token)
    sent = "".join(buf)
    if re.search(r"[.!?。?!\n]$", sent) or len(buf) > 20:
        await elevenlabs.speak_streaming(sent)
        await ws.send_json({"type": "response_text", "sentence": sent})
        buf.clear()
if buf:
    await elevenlabs.speak_streaming("".join(buf))
```

Threshold `20 tokens` is an initial value, tuned during M4 benchmarking.

### 5.4 Barge-in (user interruption) — v1 supported

Flow when user speaks while AI is playing:

1. Browser VAD detects user voice → `audio-player.js` pauses, queue cleared
2. Browser sends WS `{type: "interrupt"}`
3. Server cancels in-flight `gemma.stream_respond()` task
4. Server cancels ElevenLabs streaming request
5. Server drops buffered audio chunks
6. Browser resumes normal capture loop

ElevenLabs streaming API is WebSocket-based and supports cancellation. Gemma checks a cancellation token inside the generate loop.

### 5.5 Frame sampling strategy — Option A (final frame only)

Capture one JPEG at the moment of utterance end. Rationale: final facial expression carries the most emotional signal for interview context; bandwidth and complexity are minimized. Options B (periodic) and C (continuous) are future enhancements if emotion precision proves insufficient.

### 5.6 Dialog history management

- FastAPI in-memory `dict[session_id, list[Turn]]`, where `Turn = {role, text, emotion?, timestamp}`
- On WS close: POST `/internal/meeting/{sid}/flush` (Flask persists to DB)
- On WS reconnect: GET `/internal/meeting/{sid}/history` (Flask returns turns)
- Long-session memory: after 8 turns, oldest turns summarized into a single `system` turn

---

## 6. Authentication & Session Bridging

### 6.1 Problem
User authenticates on Flask (Google OAuth → session cookie). WebSocket connects to FastAPI on different port. FastAPI must verify user is authenticated without duplicating OAuth logic.

### 6.2 Solution: shared `SECRET_KEY`, cookie verification

`.env` (both processes read):
```
FLASK_SECRET_KEY=<strong random value>
INTERNAL_API_SECRET=<separate strong random value>
```

- Dev: `localhost:5001` and `localhost:8000` share the `localhost` host; browsers send `session` cookie to both per RFC 6265.
- Production: nginx reverse proxy routes `/` → Flask and `/ws/*` → FastAPI, making everything same-origin.

### 6.3 FastAPI cookie verify (`auth_bridge.py`)

```python
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from flask.sessions import session_json_serializer

_SERIALIZER_SALT = "cookie-session"
_MAX_AGE_SECONDS = 31 * 24 * 3600

def verify_flask_cookie(cookie_value: str, secret_key: str) -> dict:
    s = URLSafeTimedSerializer(
        secret_key,
        salt=_SERIALIZER_SALT,
        serializer=session_json_serializer,
    )
    return s.loads(cookie_value, max_age=_MAX_AGE_SECONDS)

async def authenticate_ws(websocket) -> dict:
    cookie_header = websocket.headers.get("cookie", "")
    session_cookie = parse_cookie_header(cookie_header).get("session")
    if not session_cookie:
        await websocket.close(code=1008, reason="unauthorized")
        raise HTTPException(401)
    try:
        payload = verify_flask_cookie(session_cookie, os.environ["FLASK_SECRET_KEY"])
    except (BadSignature, SignatureExpired):
        await websocket.close(code=1008, reason="invalid_session")
        raise HTTPException(401)
    if not payload.get("user_id"):
        await websocket.close(code=1008, reason="no_user")
        raise HTTPException(401)
    return payload
```

### 6.4 Flask internal API (HMAC-protected)

New Flask endpoints (minimal additions):

- `POST /internal/meeting/{session_id}/flush` — save turns to DB
- `GET /internal/meeting/{session_id}/history` — fetch turns

Protected by `@require_internal_hmac` decorator. Request must carry header `X-Internal-Signature` equal to `HMAC-SHA256(INTERNAL_API_SECRET, body)`. Signature mismatch returns 403. `hmac.compare_digest()` used for timing-safe comparison.

### 6.5 Session lifecycle
- **Expiry:** 31-day Flask cookie TTL. On expire, WS close 1008, browser redirects to login.
- **Logout:** FastAPI detects on next WS reconnect attempt; already-open WS lives until natural close. Acceptable for MVP.

### 6.6 Security checklist

| Item | Mitigation |
|------|------------|
| `FLASK_SECRET_KEY` leak | `.env` gitignored, CI secret, never in code (already fixed per CSO audit) |
| `INTERNAL_API_SECRET` leak | Separate env var; never share with frontend |
| WS public exposure | Dev: localhost bind only; Prod: nginx reverse proxy |
| CSRF on WS | Validate `Origin` header (only `http://localhost:5001` allowed in dev) |
| HMAC timing attack | `hmac.compare_digest()` |
| Internal endpoint exposure | `@require_internal_hmac` + localhost bind |

---

## 7. Error Handling & Fallback

### 7.1 Failure mode matrix

| # | Failure | Detection | Response / Fallback | Demo impact |
|---|---------|-----------|---------------------|-------------|
| 1 | Gemma model load fails (startup) | FastAPI `startup_event` exception → `/health` fail | Process exits; Flask UI banner "Realtime mode unavailable"; browser auto-uses Legacy | 🔴 Critical — preflight prevents |
| 2 | Gemma inference timeout (> 8 s no tokens) | `asyncio.wait_for(timeout=8s)` | Cancel current turn → `{type: "error", code: "gemma_timeout"}` → browser retries in Legacy for that turn | 🟡 Single-turn degrade |
| 3 | Gemma runtime error (OOM, kernel panic) | `try/except OSError` | Cancel turn; after 2 consecutive failures, force session into Legacy mode | 🟡 Recoverable |
| 4 | ElevenLabs API failure (one-off) | HTTP 4xx/5xx | Level 1: browser SpeechSynthesis fallback + `{type: "tts_degraded"}` signal | 🟡 Quality degradation, continues |
| 5 | ElevenLabs sustained failure (3 in a row) | Rolling counter | Level 2: full Legacy mode | 🟡 Legacy mode |
| 6 | WS disconnect | `onclose` event | Exponential backoff reconnect (1s → 16s max); after 3 tries → Legacy | 🟡 Brief reconnect |
| 7 | Browser mic/camera denied | `getUserMedia` exception | Clear error UI + retry button | 🔴 User action required |
| 8 | VAD false positive | Utterance < 0.5 s | Client filters short utterances, does not send `utterance_end` | 🟢 Transparent |
| 9 | Gemma Korean quality degrades (runtime) | User dissatisfaction | Keyboard shortcut `Cmd+Shift+L` toggles Legacy immediately | 🟡 Demo rescue |
| 10 | Memory leak (long session) | FastAPI memory > 6 GB | After 8 turns, summarize oldest turns into single system turn | 🟢 Transparent |

### 7.2 Two-level fallback

**Level 1: Degraded (single-feature regression)**
- ElevenLabs fail → Web Speech API TTS
- Latency over budget → skip frame sampling
- Single Gemma turn fail → retry that turn on Legacy endpoint
- UX: minor quality/latency impact, session continues

**Level 2: Full Legacy (session-wide switchover)**
- Browser: `sessionStorage["use_legacy"] = "1"`
- Disconnect WS, use HTTP `POST /meeting/message` (existing)
- Pipeline: Flask → Gemini 2.0 Flash → ElevenLabs (current production)
- UI banner: "호환 모드로 진행 중"
- UX: original ≈5 s latency, full stability

**Triggers for Level 2:**
- Gemma startup failure
- 2 consecutive Gemma runtime failures
- 3 consecutive ElevenLabs failures
- Manual `Cmd+Shift+L`
- Browser observes no `/health` response for 10 s

### 7.3 Reconnection strategy (browser `ws-client.js`)

Exponential backoff: 1 s → 2 s → 4 s → 8 s → 16 s (max). After 3 failed attempts, enter Legacy mode. On close code 1008 (unauthorized), redirect to login.

### 7.4 Graceful shutdown (FastAPI)

On `SIGTERM`:
1. Reject new WS connections (503)
2. Broadcast `{type: "shutdown", resume_in_sec: 30}` to active sessions
3. Wait up to 10 s for active turns to complete
4. Flush all histories to Flask `/internal/meeting/.../flush`
5. Exit

Browsers interpret `shutdown` message as "enter Legacy mode" (do not attempt reconnect).

### 7.5 Demo-specific safeguards

| Safeguard | Implementation | Purpose |
|-----------|----------------|---------|
| Manual Legacy toggle | `Cmd+Shift+L` shortcut | Instant recovery mid-demo |
| Preflight health check | `scripts/preflight.sh` | 5-minute pre-demo validation |
| Warmup automation | FastAPI runs dummy turn 30 s after startup | Eliminates cold-start latency |
| Realtime latency HUD | Dev console shows p50/p95 | Mid-demo tuning visibility |

### 7.6 Logging

- FastAPI: `structlog` with fields `session_id`, `user_id`, `turn_idx`, `latency_ms`, `error_code`
- Flask: existing logs preserved; internal API calls at distinct level
- Error rate > 2 % triggers local alert (Slack integration out of scope for MVP)

---

## 8. Testing & Validation

### 8.1 Pre-migration Smoke Gates (GO/NO-GO)

Three gates must pass before any `start_app_ws/` implementation begins. Failure halts migration and re-evaluates approach.

**Gate #1 — Gemma 4 E2B Korean dialogue quality**
- Script: `scripts/smoke_gemma_korean.py`
- Inputs: current `initial_system_messages` + 10 Korean interview prompts
- Comparison: Gemini 2.0 Flash responses side-by-side (blind)
- Ratings: fluency / relevance / formal register, 1–5 each
- **Pass:** mean ≥ 3.5/5, no response ≤ 2.0/5
- **Fail:** abandon full migration → reconsider keeping Gemini cloud LLM even in hybrid

**Gate #2 — Gemma 4 E2B multimodal (audio + vision)**
- Script: `scripts/smoke_gemma_multimodal.py`
- Inputs: 5 recorded Korean audio samples + photograph
- Verify: transcription accuracy ≥ 90 %; emotion inference matches GPT-4 judge
- **Fail:** fall back to separated audio/vision pipelines (abandon Parlor structure)

**Gate #3 — M2 latency benchmark**
- Script: `scripts/bench_latency.py`
- Measure: audio+frame input → first audio chunk output
- 20 iterations, 3-turn history
- **Pass:** p50 < 2.5 s AND p95 < 4 s
- **Fail:** revisit quantization, revisit target, or abandon

### 8.2 Unit tests (`start_app_ws/tests/unit/`)

| File | Target | Coverage |
|------|--------|----------|
| `test_audio_protocol.py` | Binary frame parsing, type tags | 100 % |
| `test_auth_bridge.py` | Cookie verify, HMAC signing | 100 % (security path) |
| `test_gemma_engine.py` | LiteRT load/infer interface (mocked) | 90 % |
| `test_elevenlabs_streamer.py` | Sentence boundary, stream trigger | 95 % |
| `test_session_manager.py` | In-memory history, flush, resume | 90 % |

Unit tests have **zero external dependencies** (no real Gemma load, no real API calls). Must complete in CI under 5 s.

### 8.3 Integration tests (`start_app_ws/tests/integration/`)

Use FastAPI `TestClient` with injected fakes. Scenarios:

1. Normal turn (connect → audio → response → `turn_end`)
2. Reconnect + history resume
3. Barge-in mid-playback
4. Auth failure (missing cookie → 1008)
5. Gemma timeout (fake returns nothing → error frame)
6. ElevenLabs failure (fake returns 500 → Level 1 signal)
7. History flush on close (verify Flask `/internal/flush` called)

### 8.4 E2E tests (local only)

`scripts/e2e_local.py`:
- Starts both Flask and FastAPI
- Loads real Gemma model
- Uses real ElevenLabs API key
- Plays 5 scripted interview scenarios from pre-recorded audio
- Records per-turn latency → `e2e_results.json`

Not run in CI. Executed manually before each milestone PR merge.

### 8.5 CI strategy

GitHub Actions runs:
- `flake8 start_app/ start_app_ws/`
- `mypy start_app/ start_app_ws/`
- `pytest start_app/tests/` (existing 34 tests)
- `pytest start_app_ws/tests/unit/`
- `pytest start_app_ws/tests/integration/`

E2E with real Gemma is local/M2 only.

### 8.6 Regression tests (Legacy path preservation)

Every migration PR must:
1. Pass all 34 existing Flask tests
2. Pass new `test_legacy_fallback.py`, which simulates a browser that sets `sessionStorage["use_legacy"] = "1"`, skips WS, hits `POST /meeting/message`, and verifies response matches current production behavior

### 8.7 Preflight (`scripts/preflight.sh`)

```bash
#!/bin/bash
# Run within 5 minutes of demo start
curl -f localhost:5001/health
curl -f localhost:8000/health | grep '"gemma_loaded":true'
python scripts/check_elevenlabs.py
python scripts/check_db.py
python scripts/warmup_turn.py
echo "✅ Demo ready"
```

---

## 9. Migration Sequence

### 9.1 Preparation: resolve current in-flight work

Before migration begins, current `step5-1/latency-optimization` branch must be finalized:
- Remove debug `print` added for `finish_reason`
- Commit outstanding changes (concise-response prompt, max_tokens=500, Korean punctuation handling)
- Open PR, merge to main, close issue #15
- Result: Legacy path stabilized with tuning gains preserved as safety net

### 9.2 Milestones

Each milestone is independently shippable; Legacy path stays functional throughout. `FASTAPI_WS_ENABLED=false` until M7.

| # | Milestone | Duration | Deliverables | Done criteria |
|---|-----------|----------|--------------|---------------|
| **M0** | Pre-migration Smoke Gates | 1–2 d | `scripts/smoke_gemma_*.py`, `bench_latency.py`, `smoke_results.md` | All 3 gates pass; GO decision documented |
| **M1** | FastAPI skeleton + Auth | 1–2 d | `start_app_ws/` scaffold, `/health`, WS echo, `auth_bridge.py` | Authenticated browser round-trips WS echo |
| **M2** | Gemma engine integration | 2–3 d | `gemma_engine.py`, text-in/text-out streaming | Integration test: text prompt → Gemma token stream |
| **M3** | Audio/frame pipeline | 2–3 d | `audio-capture.js`, `frame-capture.js`, binary protocol | Real mic + webcam → Gemma → text response (no TTS) |
| **M4** | ElevenLabs streaming TTS | 1–2 d | `elevenlabs_streamer.py`, `audio-player.js`, sentence boundary | Full E2E voice round-trip; latency measured |
| **M5** | VAD + Barge-in | 1–2 d | `vad-worker.js`, interrupt protocol | Hands-free interview session completes |
| **M6** | Error handling + fallback | 2 d | Auto-Legacy triggers, reconnect, `preflight.sh` | All 10 failure matrix scenarios pass |
| **M7** | Demo prep + rehearsal | 2 d | Demo script, recording, `FASTAPI_WS_ENABLED=true` | 3 consecutive clean rehearsals |

Total estimate: 12–16 working days.

### 9.3 Branching

```
main (production, Legacy preserved)
  ├── step5-1/latency-optimization    (merge/close before M0)
  └── parlor-hybrid/
        ├── m0-smoke-gates            # scripts only, zero prod impact
        ├── m1-fastapi-skeleton       # start_app_ws/ new, Flask untouched
        ├── m2-gemma-integration
        ├── m3-audio-pipeline
        ├── m4-elevenlabs-streaming
        ├── m5-vad-bargein
        ├── m6-error-fallback
        └── m7-demo-ready             # flag flip, final merge
```

One PR per milestone, merged to main. Main remains deployable at every step.

### 9.4 Feature flag

```python
# start_app/app.py (small addition)
@app.route("/config")
def client_config():
    return jsonify({
        "fastapi_ws_enabled": os.environ.get("FASTAPI_WS_ENABLED") == "true",
        "fastapi_ws_url": os.environ.get("FASTAPI_WS_URL", "ws://localhost:8000"),
    })
```

```javascript
// static/js/chat.js (small addition)
const config = await fetch("/config").then(r => r.json());
if (config.fastapi_ws_enabled && !sessionStorage.getItem("use_legacy")) {
  initRealtimeMode(config.fastapi_ws_url);
} else {
  initLegacyMode();  // existing path
}
```

Effects:
- Default `false` → zero user impact if accidentally deployed
- Flip via `.env` one-liner at M7
- Instant rollback by reverting env var

### 9.5 GitHub issue mapping

New **Milestone: `v2.0 Parlor Hybrid Migration`** with 8 new issues:

| Issue | Title | Depends on |
|-------|-------|------------|
| #16 | [M0] Pre-migration Smoke Gates | — |
| #17 | [M1] FastAPI skeleton + Auth Bridge | M0 pass |
| #18 | [M2] Gemma 4 E2B LiteRT-LM integration | M1 |
| #19 | [M3] Audio/frame binary protocol + browser capture | M2 |
| #20 | [M4] ElevenLabs streaming TTS + sentence boundary | M3 |
| #21 | [M5] Silero VAD + Barge-in | M4 |
| #22 | [M6] Error handling + Legacy auto-fallback + preflight | M5 |
| #23 | [M7] Demo rehearsal + flag activation | M6 |

Existing issues #2 and #6 (currently OPEN but implementation-complete) are closed.

### 9.6 Rollback scenarios

| Stage | Failure | Response |
|-------|---------|----------|
| M0 fail | Korean quality below threshold | Halt migration; ship Step 5-1 optimizations only |
| M1–M6 fail | Development slip | Keep flag `false`; Legacy only; zero user impact |
| Post-M7 rehearsal fail | Latency/quality regression | `FASTAPI_WS_ENABLED=false` → Legacy within 1 s |
| Demo runtime failure | Realtime degradation | `Cmd+Shift+L` toggle or env flag flip |

### 9.7 Visibility

- This document committed to `docs/superpowers/specs/`
- `CHANGELOG.md` entry per milestone
- `README.md` gains "Realtime Mode (Experimental)" section at M7
- Each PR body includes milestone number + Done-criteria checklist

---

## 10. Open Questions

1. **Gemma 4 E2B Korean tokenizer behavior** — sentence boundary regex `[.!?。?!\n]` assumes Latin + CJK punctuation. Verify against actual Gemma output on Korean during M2.
2. **ElevenLabs streaming WebSocket vs HTTP chunked** — both are supported; choose based on first-byte latency measurement in M4.
3. **AudioWorklet Safari compatibility** — acceptable as M2 MacBook uses Chrome for demo, but document as browser constraint.
4. **Gemma memory pressure during long sessions** — the history-summarization threshold (8 turns) is an educated guess. Tune in M6.

## 11. Appendix

### 11.1 Environment variables

| Variable | Consumer | Purpose |
|----------|----------|---------|
| `FLASK_SECRET_KEY` | Flask + FastAPI | Session cookie signing/verification |
| `INTERNAL_API_SECRET` | Flask + FastAPI | HMAC for internal API |
| `FASTAPI_WS_ENABLED` | Flask (`/config`) | Feature flag dispatch |
| `FASTAPI_WS_URL` | Flask (`/config`) | Browser WS endpoint |
| `GOOGLE_API_KEY` | Legacy Flask path | Gemini cloud fallback |
| `ELEVENLABS_API_KEY` | FastAPI + Legacy Flask | Streaming + fallback TTS |
| `ELEVENLABS_VOICE_ID` | FastAPI + Legacy Flask | Voice selection |
| `GEMMA_MODEL_PATH` | FastAPI | LiteRT-LM model file location |

### 11.2 New files created (summary)

```
docs/superpowers/specs/2026-04-17-parlor-hybrid-migration-design.md    (this file)
start_app_ws/                                                          (new package, ~8 files)
static/js/realtime/                                                    (new JS modules, 5 files)
scripts/smoke_gemma_korean.py
scripts/smoke_gemma_multimodal.py
scripts/bench_latency.py
scripts/preflight.sh
scripts/e2e_local.py
scripts/check_elevenlabs.py
scripts/check_db.py
scripts/warmup_turn.py
requirements-ws.txt
models/                                                                (gitignored)
```

### 11.3 Existing files modified (summary)

```
start_app/app.py                           (add /config, /internal/meeting/*)
start_app/dialogue_manager/meeting.py      (expose history API)
start_app/templates/chat.html              (realtime JS hooks)
start_app/static/js/chat.js                (feature flag dispatch)
.gitignore                                 (add models/)
.env.example                               (document new env vars)
CLAUDE.md                                  (update health stack to include start_app_ws/)
requirements.txt                           (no change expected)
```
