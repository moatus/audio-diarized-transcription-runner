# audio-diarized-transcription-runner

Workload binary that serves the Livepeer `audio:diarized-transcription@v0`
capability over HTTP, backed by NVIDIA NeMo offline diarization-with-ASR. One
Docker image per capability; one process per broker-dispatched container.

> **For agents:** start at [`AGENTS.md`](./AGENTS.md).

## What this repo ships

| Image | Language | Capability |
|---|---|---|
| `audio-diarized-transcription-runner` | Python | `audio:diarized-transcription@v0` |

The runner exposes `POST /v1/audio/diarized-transcriptions`, accepts multipart
audio uploads, and returns speaker-labeled transcripts with segments, optional
word timestamps, subtitle artifacts, and usage. It also exposes an additive
stateful live API under `/v1/audio/diarized-transcriptions/live/sessions` backed
by NeMo `OnlineClusteringDiarizer`. Audio is normalized with `ffmpeg`; models
and caches live under `MODEL_CACHE_DIR`.

The Docker image is based on `nvcr.io/nvidia/pytorch:24.07-py3` and installs
NeMo Toolkit ASR dependencies during build.

## Build

Every normal operator gesture is Docker-first:

```bash
./build-images.sh build                                      # build all images
./build-images.sh build audio-diarized-transcription-runner  # build the runner
./build-images.sh validate                                   # validate compose overlays
./build-images.sh push                                       # push to ${REGISTRY}
./build-images.sh clean                                      # remove locally-built images
./build-images.sh help                                       # show subcommands
```

No host Python is required for Docker build/run.

## Quick start

```bash
# 1. Build the runner.
./build-images.sh build

# 2. Run the runner. The first real request may download NeMo model weights.
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up

# 3. Transcribe and diarize an audio file.
curl -s http://localhost:8080/v1/audio/diarized-transcriptions \
  -F file=@./sample.wav \
  -F model=nemo-diarized-transcription-meeting-v0 \
  -F language=en \
  -F preset=meeting \
  -F max_speakers=8 \
  -F response_format=json | jq
```

For offline / no-egress hosts, pre-stage NeMo, Hugging Face, and torch caches
into the `audio-diarized-transcription-models` volume before starting the
runner. See [`RUNNERS.md`](./RUNNERS.md) for environment variables and request
details.

## Repo layout

```text
.
├── AGENTS.md, README.md, RUNNERS.md
├── LICENSE, NOTICE
├── build-images.sh
├── infra/
│   ├── compose/        # docker-compose overlay
│   ├── dockerfiles/    # runner Dockerfile
│   ├── env/            # .env.example template
│   └── offerings/      # Livepeer offering manifest
├── audio-diarized-transcription-runner/
│   ├── pyproject.toml
│   └── src/audio_diarized_transcription_runner/
│       ├── app.py
│       ├── pipeline.py
│       ├── settings.py
│       └── configs/diar_infer_meeting.yaml
├── scripts/
│   └── audio_diarized_transcription_smoke.py
└── tests/
```

## Configuration

Common env vars:

- `RUNNER_PORT` default `8080`
- `DEVICE` default `cuda`
- `MODEL_CACHE_DIR` default `/models`
- `WORK_DIR` default `/tmp/audio-diarized-transcription-runner`
- `MAX_QUEUE_SIZE` default `1`
- `MAX_AUDIO_MB` default `100`
- `METRICS_ENABLED` default `false`
- `LIVE_HISTORY_BUFFER_SIZE` default `256`
- `LIVE_CURRENT_BUFFER_SIZE` default `256`
- `LIVE_SESSION_TTL_SECONDS` default `3600`
- `LIVE_CLOSED_SESSION_TTL_SECONDS` default `300`

The default model stack is `vad_multilingual_marblenet`, `titanet_large`, and
`stt_en_conformer_ctc_large`. The packaged NeMo meeting config can be overridden
with `NEMO_DIARIZER_CONFIG`.

## Validation

Fast host checks that avoid model downloads:

```bash
PYTHONPATH=audio-diarized-transcription-runner/src pytest -q
./build-images.sh validate
```

Full runtime validation requires a CUDA-capable Docker host and enough disk for
NeMo model caches:

```bash
./build-images.sh build
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
python scripts/audio_diarized_transcription_smoke.py ./sample.wav --require-text
```

## License

**Apache-2.0** — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

This repo's runner code is Apache-2.0. The image installs NVIDIA NeMo Toolkit,
PyTorch, and uses an NVIDIA NGC base image; review those upstream licenses and
NGC container terms before redistributing images.
