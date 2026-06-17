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
- final model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- preview model: `mlx-community/Qwen3-ASR-0.6B-6bit`
- model cache: `~/.cache/huggingface/hub`
- Hugging Face endpoint: `https://hf-mirror.com`
- idle unload: `30s`

## VoxT settings

Use VoxT's remote ASR provider:

```text
Provider: OpenAI Transcribe
Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions
API Key: password
Model: mlx-community/Qwen3-ASR-1.7B-8bit
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

- Request model names only need to be non-empty; service-side routing chooses the actual configured models.
- VoxT preview uploads named `voxt-openai-preview-*.wav` use the preview model.
- Final uploads use the final model.
- Realtime WebSocket streams use the final model in-process and accept VoxT Aliyun Qwen realtime model names such as `qwen3-asr-flash-realtime`.
- ASR runs in a worker process, loaded lazily on first use.
- Realtime ASR uses a separate lazily-loaded in-process session.
- Preview uploads are handled as stateful streaming/delta ASR with a 30-second context window by default.
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
  --model mlx-community/Qwen3-ASR-1.7B-8bit \
  --preview-model mlx-community/Qwen3-ASR-1.7B-8bit

# faster preview
uv run qwen3_asr_server.py \
  --preview-model mlx-community/Qwen3-ASR-0.6B-6bit \
  --preview-max-new-tokens 64

# use official Hugging Face instead of mirror
HF_ENDPOINT=https://huggingface.co uv run qwen3_asr_server.py
```

## VoxT-style LLM reference API

`voxt_llm_reference.py` is an independent reference API for VoxT-style ASR and ASR → LLM chains:

- Whisper/OpenAI final transcription call
- Whisper/OpenAI chunk preview call
- transcription enhancement
- translation
- rewrite

It keeps VoxT-like default prompts and only includes LLM runtime fields VoxT sets itself for these chains: `max_tokens`, `temperature`, `top_p`, plus DeepSeek JSON response format for structured rewrite. It also includes reference implementations for ASR hint compilation, visible-output cleanup, structured rewrite parsing, and long-text segmentation.

```bash
uv run voxt_llm_reference.py --port 5112

curl http://127.0.0.1:5112/v1/voxt/transcription \
  -H 'content-type: application/json' \
  -d '{"text":"um hello world","dry_run":true}'
```

Reference Whisper/OpenAI calls:

```bash
# final transcription-style call
curl http://127.0.0.1:5112/v1/voxt/whisper/transcribe \
  -F file=@sample.wav

# VoxT chunk pseudo-realtime preview-style call
curl http://127.0.0.1:5112/v1/voxt/whisper/chunk \
  -F file=@sample.wav
```

ASR hint compilation:

```bash
curl http://127.0.0.1:5112/v1/voxt/asr-hint/compile \
  -H 'content-type: application/json' \
  -d '{"target":"openai_whisper","user_main_language":"Chinese","user_other_languages":["English","Japanese"],"dictionary_terms":"VoxT\nQwen3-ASR"}'
```

Supported Jinja variables:

```txt
{{USER_MAIN_LANGUAGE}}
{{USER_OTHER_LANGUAGES}}
{{TARGET_LANGUAGE}}
{{DICTIONARY_TERMS}}
```

`dictionary_terms` feeds `{{DICTIONARY_TERMS}}`; `dictionary_glossary` is added as a separate LLM context block. Long inputs are segmented by default after 300 characters.

Set `upstream_url`, `upstream_api_key`, and `model` in the request body to call an OpenAI-compatible LLM. Set multipart fields `endpoint`, `api_key`, and `model` to override the Whisper/OpenAI ASR target.

## Health check

```bash
curl http://127.0.0.1:5100/health
```
