from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Compatibility entry point; the maintained implementation is the single uv script."""
    from start_voxt_asr_server import main as script_main

    return script_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
