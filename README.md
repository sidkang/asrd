# voxt-qwen3-asr

Local OpenAI-compatible ASR server for VoxT on Apple Silicon/MLX, powered by `mlx-qwen3-asr`.

## Run

```bash
uv run qwen3_asr_server.py
```

Defaults:

- endpoint: `http://127.0.0.1:5100/v1/audio/transcriptions`
- realtime endpoint: `ws://127.0.0.1:5100/v1/realtime`
- bind: `0.0.0.0:5100`
- API key: `password`
- final model: `mlx-community/Qwen3-ASR-1.7B-6bit`
- preview model: `mlx-community/Qwen3-ASR-0.6B-4bit`
- model cache: `./.cache`
- Hugging Face endpoint: `https://hf-mirror.com`
- idle unload: `30s`

## VoxT settings

Use VoxT's remote ASR provider:

```text
Provider: OpenAI Transcribe
Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions
API Key: password
Model: mlx-community/Qwen3-ASR-1.7B-6bit
Chunk Pseudo Realtime Preview: On
```

Or use VoxT's realtime ASR provider:

```text
Provider: Aliyun Qwen Realtime ASR
Endpoint: ws://127.0.0.1:5100/v1/realtime
API Key: password
Model: qwen3-asr-flash-realtime
Audio: PCM16 mono 16k
```

For LAN use, replace `127.0.0.1` with the server machine IP.

## Behavior

- VoxT preview uploads named `voxt-openai-preview-*.wav` use the preview model.
- Final uploads use the final model.
- Realtime WebSocket streams use the final model in-process and accept VoxT Aliyun Qwen realtime model names such as `qwen3-asr-flash-realtime`.
- ASR runs in a worker process, loaded lazily on first use.
- Realtime ASR uses a separate lazily-loaded in-process session.
- Preview uploads transcribe only the last 30 seconds by default.
- The worker process exits after 30 seconds idle by default.
- Requests are serialized to avoid MLX thread/stream issues.
- Busy preview requests are skipped with an empty result to avoid backlog.
- Logs include speed and RTF, for example `speed=14.11x rtf=0.071`.

## Useful options

```bash
# keep models loaded
uv run qwen3_asr_server.py --unload-after-sec 0

# unload sooner/later
uv run qwen3_asr_server.py --unload-after-sec 120

# use one model for both preview and final
uv run qwen3_asr_server.py \
  --model mlx-community/Qwen3-ASR-1.7B-6bit \
  --preview-model mlx-community/Qwen3-ASR-1.7B-6bit

# faster preview
uv run qwen3_asr_server.py \
  --preview-model mlx-community/Qwen3-ASR-0.6B-4bit \
  --preview-max-new-tokens 64

# use official Hugging Face instead of mirror
HF_ENDPOINT=https://huggingface.co uv run qwen3_asr_server.py
```

## Health check

```bash
curl http://127.0.0.1:5100/health
```
