# Agent Notes

This is a standalone Livepeer module repo for `openai:audio-transcriptions`.

Start with:

- `README.md` for operator-facing build, run, and request details.
- `RUNNERS.md` for the runner-specific config and API contract.
- `audio-diarized-transcription-runner/src/audio_diarized_transcription_runner/` for implementation.
- `infra/` for Docker, compose, env, and offering metadata.

Keep this repo self-contained. Do not add imports, Docker `COPY` statements, or
runtime paths that reach back into the old combined `livepeer-roboflow` repo.
