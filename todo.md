# TODO

## 多语言 preview 可选默认 ASR prompt

现象：VoxT 使用多语言设置（如中文 + 英文 + 日文）时，OpenAI Transcribe preview 请求通常不传 `language`，日志为：

```txt
language=None->None
prompt_chars=0
```

因此 Qwen3-ASR 会每次自动识别语言，日文内容可能在 preview 中跳成中文/英文/日文。

后续可考虑：服务端增加默认 prompt，只在请求未传 prompt 时使用。

建议内容：

```txt
Chinese, English, and Japanese may appear. Transcribe exactly as spoken. Do not translate. Preserve mixed-language content, names, terms, and code-like text.
```

实现要点：

- 增加 `--default-prompt` / `ASR_DEFAULT_PROMPT`
- `Config` 增加 `default_prompt`
- HTTP：`effective_prompt = prompt or config.default_prompt`
- Realtime：默认使用 `config.default_prompt`，客户端传 prompt 时覆盖

注意：不要强制固定 `language=zh/ja/en`，因为内容可能确实是混合语言。
