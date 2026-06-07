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
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions` | Create a stateful online diarization session |
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}/audio` | Ingest one audio chunk into a live session |
| `GET` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}` | Read the latest live diarization snapshot |
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}/finish` | Close a live session, optionally running final offline ASR+diarization |
| `DELETE` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}` | Close a live session without final transcription |
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
| `LIVE_FINAL_ASR_MODEL` | `stt_en_fastconformer_ctc_large` | NeMo English ASR model used only by live-session final transcription |
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

### Live Session Path

The live path is additive and keeps one NeMo `OnlineClusteringDiarizer` instance
per session. Audio chunks are normalized to 16 kHz mono WAV, appended to the
session timeline, diarized with a rolling audio buffer, and returned as
cumulative speaker turn events. The default live VAD strategy is an energy VAD
that feeds absolute speech intervals into NeMo; callers that already have VAD
can create the session with `vad_strategy=provided` and pass
`vad_segments_json` on each chunk upload.

Create a session:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions \
  -H 'content-type: application/json' \
  -d '{"session_id":"live_demo","max_speakers":4,"vad_strategy":"energy"}' | jq
```

Ingest one chunk:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions/live_demo/audio \
  -F file=@chunk-000.wav \
  -F sequence_index=0 | jq
```

Finish without ASR:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions/live_demo/finish \
  -H 'content-type: application/json' \
  -d '{"run_final_transcription":false}' | jq
```

Near-live responses currently contain diarization turns only (`text` is empty on
segments). The pragmatic transcript path is `run_final_transcription=true` on
finish, which reuses the existing verified offline NeMo ASR+diarization path
over the accumulated session WAV. Live-session final transcription uses
`LIVE_FINAL_ASR_MODEL`, which defaults to FastConformer CTC for cleaner meeting
audio while preserving `DEFAULT_ASR_MODEL` for existing non-live requests.
