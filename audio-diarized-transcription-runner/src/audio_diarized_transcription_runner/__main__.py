"""Module entry point for running the diarized transcription runner."""

from __future__ import annotations

import uvicorn

from .settings import RunnerSettings


def main() -> None:
    settings = RunnerSettings()
    uvicorn.run(
        "audio_diarized_transcription_runner.app:app",
        host="0.0.0.0",
        port=settings.runner_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
