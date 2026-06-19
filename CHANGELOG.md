# Changelog

## 0.2.2 - 2026-06-19

- Add request/session ids to operational logs.
- Keep multiple HTTP preview correction sessions in cache for better final matching.
- Add `asrd service show`.
- Add `asrd service uninstall --purge-logs`.

## 0.2.1 - 2026-06-19

- Remove ASR-specific environment-variable configuration; use CLI options for service configuration.
- Keep `HF_ENDPOINT` as the only documented environment override.

## 0.2.0 - 2026-06-19

- Rename and package the project as `asrd` with a `src/asrd` layout and `asrd` CLI.
- Add `asrd serve`, `asrd service ...`, and `asrd doctor` commands.
- Add macOS LaunchAgent management under `com.sid.asrd`.
- Add Apache-2.0 license and GitHub Actions CI.
- Add unit and mock integration tests for audio helpers, CLI, launchd, HTTP, and WebSocket behavior.
- Simplify debug artifacts: flat `debug_dir`, same-basename audio/JSON sidecars, compact aligned HTTP/WS metadata, and WS chunk manifests.
- Default service/debug behavior is safer: localhost bind by default and debug is opt-in.
