# asrd

[![CI](https://github.com/sidkang/asrd/actions/workflows/ci.yml/badge.svg)](https://github.com/sidkang/asrd/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

A local OpenAI-compatible ASR daemon for Apple Silicon, powered by MLX and Qwen3-ASR.

`asrd` exposes a small local speech-to-text service with:

- OpenAI-compatible HTTP transcription: `POST /v1/audio/transcriptions`
- Realtime WebSocket transcription: `GET /v1/realtime`
- macOS LaunchAgent installation via the built-in CLI
- VoxT-friendly HTTP chunk preview and realtime WebSocket behavior

> `asrd` stands on the shoulders of [`mlx-qwen3-asr`](https://github.com/moona3k/mlx-qwen3-asr) and the Qwen3-ASR models. Model loading, transcription, generation, and forced alignment come from that ecosystem. This project is a local service wrapper, gateway, deployment helper, and realtime/chunking policy layer around it. Many thanks to the MLX/Qwen3-ASR maintainers and contributors for making this practical on Apple Silicon.

## Requirements

- macOS on Apple Silicon
- Python `>=3.10`
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` available on `PATH` for non-WAV audio input

```bash
brew install uv ffmpeg
```

## Installation

`asrd` is not published to PyPI. Install or run it directly from Git.

Run once without installing:

```bash
uvx --from git+https://github.com/sidkang/asrd.git asrd --help
```

Install as a persistent user tool:

```bash
uv tool install git+https://github.com/sidkang/asrd.git
asrd --help
```

Upgrade later:

```bash
uv tool upgrade asrd
```

## Quick start

Check the local environment:

```bash
asrd doctor
```

Run the server in the foreground:

```bash
asrd serve
```

Default endpoints:

```text
HTTP: http://127.0.0.1:5100/v1/audio/transcriptions
WS:   ws://127.0.0.1:5100/v1/realtime
```

Default API key:

```text
password
```

The default bind address is `127.0.0.1`. Use `--host 0.0.0.0` only when you intentionally want to expose the service on your LAN, and change the default API key for shared machines or untrusted networks.

Health check:

```bash
curl -H 'Authorization: Bearer password' http://127.0.0.1:5100/health
```

Transcribe a file:

```bash
curl http://127.0.0.1:5100/v1/audio/transcriptions \
  -H 'Authorization: Bearer password' \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F file=@sample.wav
```

## macOS service

Install and start a user LaunchAgent:

```bash
asrd service install
```

Manage it:

```bash
asrd service status
asrd service restart
asrd service logs
asrd service uninstall
```

Service files and logs:

```text
Label:  com.sid.asrd
Plist:  ~/Library/LaunchAgents/com.sid.asrd.plist
Stdout: ~/Library/Logs/com.sid.asrd.out.log
Stderr: ~/Library/Logs/com.sid.asrd.err.log
Debug:  ~/Library/Logs/com.sid.asrd.debug
```

Install with custom options:

```bash
asrd service install \
  --host 127.0.0.1 \
  --port 5100 \
  --api-key password \
  --unload-after-sec 30
```

For LAN access, pass `--host 0.0.0.0` and use a non-default API key.

## Configuration

Useful `serve` options:

```bash
# keep models loaded
asrd serve --unload-after-sec 0

# unload after two minutes idle
asrd serve --unload-after-sec 120

# preserve debug audio artifacts and compact JSON sidecars
asrd serve --debug

# use the official Hugging Face endpoint instead of the default mirror
HF_ENDPOINT=https://huggingface.co asrd serve
```

Defaults:

| Setting | CLI | Default |
|---|---|---|
| Host | `--host` | `127.0.0.1` |
| Port | `--port` | `5100` |
| API key | `--api-key` | `password` |
| Preview model | `--preview-model` | `mlx-community/Qwen3-ASR-0.6B-6bit` |
| Final model | `--model` | `mlx-community/Qwen3-ASR-1.7B-8bit` |
| Model cache | `--cache-dir` | `~/.cache` |
| Hugging Face endpoint | `HF_ENDPOINT` | `https://hf-mirror.com` |
| Idle unload | `--unload-after-sec` | `30` |
| Debug | `--debug` | off |
| Debug dir | `--debug-dir` | `~/Library/Logs/com.sid.asrd.debug` |

With `--debug`, each captured audio file is saved directly under `debug_dir` with a same-basename compact `.json` sidecar.

## Realtime WebSocket

The canonical realtime route is:

```text
ws://127.0.0.1:5100/v1/realtime
```

The protocol is compatible with Aliyun Bailian/Qwen-style realtime ASR clients:

```text
receive: session.update
send:    session.updated
receive: input_audio_buffer.append
send:    conversation.item.input_audio_transcription.text
receive: session.finish
send:    conversation.item.input_audio_transcription.completed
send:    session.finished
```

`session.finish` sends an interim partial, waits briefly for 1.7B final correction, then always sends `completed` and `session.finished` using corrected or best-available text.

## VoxT setup

`asrd` is a general local ASR daemon, but it includes behavior tuned for VoxT.

### OpenAI Transcribe provider

```text
Provider: OpenAI Transcribe
Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions
API Key: password
Model: mlx-community/Qwen3-ASR-1.7B-8bit
Chunk Pseudo Realtime Preview: On
```

### Aliyun Bailian ASR provider

```text
Provider: Aliyun Bailian ASR
Endpoint: ws://127.0.0.1:5100/v1/realtime
API Key: password
Model: qwen3-asr-flash-realtime
Audio: PCM16 mono 16k
```

For LAN use, replace `127.0.0.1` with the server machine IP.

## Behavior notes

- HTTP/WS preview uses the 0.6B model; final/correction uses the 1.7B model.
- VoxT HTTP preview uploads are detected by `voxt-openai-preview-*.wav`.
- Matching HTTP final uploads reuse the preview correction cache; plain uploads use full one-shot transcription.
- Request-provided language/context/hotwords are passed through; no defaults are injected.
- Logs include latency, speed, and RTF summaries.

## Resource usage

Models are loaded lazily and unloaded after 30 seconds idle by default. Realtime use loads the 0.6B preview model in the main process and starts a separate 1.7B correction worker when rolling or final correction is needed.

Expect several GB of memory use during active transcription. The 1.7B correction worker is the main cost and can peak around 7–8 GB on longer clips. Use `--unload-after-sec 0` to keep models warm, or keep the default idle unload to free memory between requests.

## License

Apache-2.0. See [LICENSE](LICENSE).
