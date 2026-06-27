# asrd realtime ASR notes

Date: 2026-06-19

This document records the current implemented behavior and the smoke tests used for `asrd`. It intentionally omits old exploration notes and unused candidate strategies.

## Current service

`asrd` is a local OpenAI-compatible ASR daemon powered by `mlx-qwen3-asr`.

Endpoints:

```txt
HTTP: http://127.0.0.1:5100/v1/audio/transcriptions
WS:   ws://127.0.0.1:5100/v1/realtime
```

Current internal deployment:

```txt
LaunchAgent: com.sid.asrd
Logs:
  ~/Library/Logs/com.sid.asrd.out.log
  ~/Library/Logs/com.sid.asrd.err.log
Debug dir:
  ~/Library/Logs/com.sid.asrd.debug
API key: password
```

Service commands:

```bash
uv run asrd service install
uv run asrd service restart
uv run asrd service status
uv run asrd service logs
uv run asrd service uninstall
```

## Models

```txt
Preview model:    mlx-community/Qwen3-ASR-0.6B-6bit
Correction/final: mlx-community/Qwen3-ASR-1.7B-8bit
Sample rate:      16000 Hz
```

Routing is fixed by role:

```txt
preview/realtime partials -> 0.6B
correction/final          -> 1.7B
```

Client-provided model names only need to be non-empty on HTTP. WS model is optional.

## Realtime WebSocket behavior

Canonical route:

```txt
/v1/realtime
```

Aliyun Bailian/Qwen-compatible event flow:

```txt
receive: session.update
send:    session.updated
receive: input_audio_buffer.append
send:    conversation.item.input_audio_transcription.text
receive: session.finish
send:    conversation.item.input_audio_transcription.completed
send:    session.finished
```

Current preview policy:

```txt
0.6B last-20s chunk preview
WS update cadence: 2s
```

Current rolling correction policy:

```txt
W = 40s
stride = 10s
commit_position = 20s
commit_span = 10s
```

For steady-state window:

```txt
window: [t, t+40]
commit: [t+20, t+30]
```

First window is allowed to commit from the start:

```txt
window: [0, 40]
commit: [0, 30]
```

Finalization behavior:

```txt
1. On session.finish, stop scheduling preview updates.
2. Immediately send current best partial as an interim text event when available.
3. Wait up to 4.5s for 1.7B final-tail/catch-up correction.
4. Always send completed + session.finished using corrected text if ready, otherwise best available text.
```

This avoids VoxT/client disconnects caused by long silent final waits.

## HTTP Whisper/OpenAI behavior

Route:

```txt
/v1/audio/transcriptions
```

VoxT chunk preview detection:

```txt
filename starts with voxt-openai-preview-
```

HTTP preview policy:

```txt
0.6B last-20s chunk preview
HTTP preview cadence: 1.4s
```

Preview requests also advance the 1.7B rolling correction cache.

Final request behavior:

```txt
If final audio matches the cached preview session:
  cached stable prefix + bounded final-tail/catch-up

If no matching preview cache exists:
  full one-shot 1.7B transcription
```

Cache matching uses:

```txt
language/prompt match
head audio match
tail audio match at the expected position
final audio not shorter than cached preview beyond tolerance
```

## Language and context

Current language behavior:

```txt
Use request-provided language when present.
Do not force a default language.
If no language is provided, pass None/auto to Qwen3-ASR.
```

Current context/hotword behavior:

```txt
Do not inject default context or hotwords.
Pass through request-provided prompt/context/hotword terms only.
```

## Health check

```bash
curl -H 'Authorization: Bearer password' http://127.0.0.1:5100/health
```

Expected notable fields:

```json
{
  "status": "ok",
  "sample_rate": 16000,
  "chunk_preview_window_sec": 20.0,
  "ws_chunk_preview_update_sec": 2.0,
  "http_chunk_preview_update_sec": 1.4,
  "rolling_correction_window_sec": 40.0,
  "rolling_correction_stride_sec": 10.0,
  "rolling_correction_commit_position_sec": 20.0,
  "rolling_correction_commit_span_sec": 10.0,
  "final_catchup_max_window_sec": 60.0,
  "realtime_final_wait_timeout_sec": 4.5
}
```

## Smoke tests

### Syntax/import

```bash
for f in $(find src/asrd -name '*.py' -print); do
  uv run python -m ast "$f" >/dev/null || exit 1
done

PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY'
import asrd.server, asrd.cli, asrd.audio, asrd.util
print('import-ok')
PY
```

### CLI/package

```bash
uv run asrd --help
uv run asrd serve --readme
uv build --out-dir /tmp/asrd-build-check
```

### Service

```bash
uv run asrd service restart --host 0.0.0.0 --port 5100 --unload-after-sec 30
uv run asrd service status
```

### Empty WebSocket final

```bash
uv run python - <<'PY'
import asyncio, json, websockets

async def main():
    async with websockets.connect(
        'ws://127.0.0.1:5100/v1/realtime',
        additional_headers={'Authorization': 'Bearer password'},
        ping_interval=None,
    ) as ws:
        await ws.send(json.dumps({
            'type': 'session.update',
            'session': {
                'input_audio_format': 'pcm',
                'input_audio_transcription': {'language': 'zh'},
                'sample_rate': 16000,
                'modalities': ['text'],
            },
        }))
        print(await asyncio.wait_for(ws.recv(), timeout=10))
        await ws.send(json.dumps({'type': 'session.finish'}))
        while True:
            try:
                print(await asyncio.wait_for(ws.recv(), timeout=10))
            except Exception as exc:
                print('recv-end', type(exc).__name__, exc)
                break

asyncio.run(main())
PY
```

Expected result:

```txt
session.updated
conversation.item.input_audio_transcription.completed
session.finished
ConnectionClosedOK
```

### Logs to inspect

```bash
tail -f ~/Library/Logs/com.sid.asrd.out.log
tail -f ~/Library/Logs/com.sid.asrd.err.log
```

Useful markers:

```txt
Realtime WS session.update ... language='Chinese'
Realtime WS final interim text sent ...
Realtime WS final correction timeout ...
Realtime WS final completed sent ...
Realtime WS final session.finished sent ...
HTTP realtime final strategy=rolling-final-tail matched_session=True
HTTP realtime final strategy=full-one-shot matched_session=False
```

## Known limitations

- 0.6B last-20s preview is replaceable and may visibly jump, especially before the first stable correction or on mixed-language/noisy audio.
- 1.7B correction worker is the main latency and memory cost.
- Long no-cache HTTP uploads use full one-shot 1.7B transcription and can be slow.
- HTTP preview cache is currently single-session, so concurrent Whisper chunk sessions may fall back to full one-shot final.
- Debug mode preserves compact audio/JSON artifacts under `debug_dir`; WS JSON stores chunk boundaries for replay, but not separate per-chunk audio files.
