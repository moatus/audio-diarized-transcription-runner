# Runner Reference

## `audio-diarized-transcription-runner`

Serves the Livepeer `audio:diarized-transcription@v0` capability over HTTP.
The runner accepts one multipart audio upload, decodes it with `ffmpeg`, converts
it to 16 kHz mono WAV, and runs NVIDIA NeMo offline diarization-with-ASR.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Runtime readiness and CUDA facts |
| `GET` | `/audio:diarized-transcription@v0/options` | Capability defaults and limits |
| `POST` | `/v1/audio/diarized-transcriptions` | Multipart diarized transcription request |
| `GET` | `/metrics` | Prometheus text metrics when `METRICS_ENABLED=true` |

### Request

`POST /v1/audio/diarized-transcriptions` requires multipart form data:

| Field | Required | Default | Notes |
|---|---:|---|---|
| `file` | yes |  | Audio file. Anything decodable by `ffmpeg` is accepted. |
| `model` | no | `nemo-diarized-transcription-meeting-v0` | Logical model id exposed to Livepeer. |
| `language` | no | `en` | Only English is supported in this phase. |
| `preset` | no | `meeting` | Only the meeting preset is supported in this phase. |
| `num_speakers` | no |  | Exact speaker count, if known. |
| `max_speakers` | no | `8` | Upper bound for clustering. |
| `response_format` | no | `json` | `json`, `text`, `srt`, or `vtt`. |
| `include_words` | no | `true` | Include normalized word timestamps in JSON. |
| `include_artifacts` | no | `true` | Include local artifact paths in JSON. |

Unsupported language, preset, or response format returns a structured `422`.
Oversized uploads return `413`. A full local queue returns `429`. CUDA OOM
returns `507`.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `RUNNER_PORT` | `8080` | HTTP listen port |
| `DEVICE` | `cuda` | `cuda` or `cpu`; CUDA is expected for production |
| `CAPABILITY_NAME` | `audio:diarized-transcription@v0` | Capability advertised in responses |
| `METRICS_ENABLED` | `false` | Expose `/metrics` |
| `MAX_QUEUE_SIZE` | `1` | Concurrent in-process request slots |
| `MAX_AUDIO_MB` | `100` | Upload limit |
| `MODEL_CACHE_DIR` | `/models` | NeMo, Hugging Face, and torch cache root |
| `WORK_DIR` | `/tmp/audio-diarized-transcription-runner` | Per-request uploads and artifacts |
| `DEFAULT_VAD_MODEL` | `vad_multilingual_marblenet` | NeMo VAD model |
| `DEFAULT_SPEAKER_MODEL` | `titanet_large` | NeMo speaker embedding model |
| `DEFAULT_ASR_MODEL` | `stt_en_conformer_ctc_large` | NeMo English ASR model |
| `DEFAULT_MAX_SPEAKERS` | `8` | Default clustering speaker upper bound |
| `DEFAULT_PRESET` | `meeting` | Default preset |
| `NEMO_DIARIZER_CONFIG` | packaged config | Optional override for the NeMo diarization YAML |

### Response Shape

JSON responses include:

- `id`, `status`, `capability`, and `mode`
- `source` filename/content type
- `models` used for VAD, speaker embeddings, and ASR
- `duration_seconds`, `text`, and `speaker_count`
- `speakers`, `segments`, and optional `words`
- optional artifact paths for NeMo JSON/RTTM/CTM/TXT/Gecko plus generated SRT/VTT
- `usage.audio_seconds` and `usage.work_units`

`text`, `srt`, and `vtt` response formats return plain text.

