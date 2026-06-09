#!/usr/bin/env python3
"""Smoke helper for the true streaming WebSocket transcription path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from array import array
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

import websockets


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the true streaming runner WebSocket.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Optional audio file. If omitted, sends a synthetic sine wave.",
    )
    parser.add_argument(
        "--runner-url",
        default=os.getenv("AUDIO_DIARIZED_TRANSCRIPTION_RUNNER_URL", "http://localhost:8080"),
    )
    parser.add_argument("--session-id", default=f"stream_smoke_{uuid.uuid4().hex[:8]}")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--chunk-seconds", type=float, default=0.08)
    parser.add_argument("--realtime", action="store_true", help="Pace audio sends by chunk duration.")
    parser.add_argument("--recv-timeout", type=float, default=30.0)
    parser.add_argument("--finish-timeout", type=float, default=120.0)
    parser.add_argument("--events-jsonl", help="Optional path to write received event JSONL.")
    parser.add_argument("--summary-json", help="Optional path to write latency/event summary JSON.")
    parser.add_argument("--require-transcript", action="store_true")
    parser.add_argument("--require-speaker", action="store_true")
    args = parser.parse_args()

    audio = _load_audio(args.audio_path, seconds=args.seconds)
    events, summary = asyncio.run(_run_smoke(args, audio))
    if args.events_jsonl:
        _write_jsonl(Path(args.events_jsonl), events)
    if args.summary_json:
        _write_json(Path(args.summary_json), summary)
    for event in events:
        print(json.dumps(event, sort_keys=True))
    if args.require_transcript and not any(
        event.get("event_type") == "transcript.segment" and str(event.get("text", "")).strip()
        for event in events
    ):
        raise RuntimeError("streaming runner returned no transcript.segment text")
    if args.require_speaker and not any(
        event.get("event_type") == "speaker.update" and event.get("speaker")
        for event in events
    ):
        raise RuntimeError("streaming runner returned no speaker.update speaker")
    return 0


async def _run_smoke(args, audio: bytes) -> tuple[list[dict], dict]:
    events: list[dict] = []
    uri = _ws_uri(args.runner_url, args.session_id)
    chunk_bytes = max(2, int(16000 * args.chunk_seconds) * 2)
    if chunk_bytes % 2:
        chunk_bytes += 1
    started_monotonic = time.monotonic()
    audio_started_monotonic: float | None = None
    send_count = 0
    receive_done = asyncio.Event()

    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as websocket:
        async def receive_until_finished(timeout: float) -> None:
            while True:
                try:
                    raw_event = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                except asyncio.TimeoutError as error:
                    raise RuntimeError(f"timed out waiting {timeout:.1f}s for streaming event") from error
                received_at_epoch = time.time()
                received_at_elapsed = time.monotonic() - started_monotonic
                event = json.loads(raw_event)
                event["client_received_at_epoch"] = received_at_epoch
                event["client_received_at_elapsed_seconds"] = round(received_at_elapsed, 6)
                if audio_started_monotonic is not None and isinstance(event.get("end"), (int, float)):
                    event["client_audio_timeline_latency_seconds"] = round(
                        time.monotonic() - audio_started_monotonic - float(event["end"]),
                        6,
                    )
                events.append(event)
                if event.get("event_type") == "transcript.session.finished":
                    receive_done.set()
                    return

        receiver = asyncio.create_task(receive_until_finished(args.recv_timeout))
        await _wait_for_event(events, "session.snapshot", timeout=args.recv_timeout)
        audio_started_monotonic = time.monotonic()
        for offset in range(0, len(audio), chunk_bytes):
            await websocket.send(audio[offset : offset + chunk_bytes])
            send_count += 1
            if args.realtime:
                await asyncio.sleep(len(audio[offset : offset + chunk_bytes]) / 2 / 16000.0)
        await websocket.send(json.dumps({"type": "finish"}))
        await asyncio.wait_for(receive_done.wait(), timeout=args.finish_timeout)
        await receiver
    summary = _summarize_events(
        events,
        audio_bytes=len(audio),
        chunk_bytes=chunk_bytes,
        sent_frame_count=send_count,
        wall_seconds=time.monotonic() - started_monotonic,
    )
    return events, summary


async def _wait_for_event(events: list[dict], event_type: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(event.get("event_type") == event_type for event in events):
            return
        await asyncio.sleep(0.01)
    raise RuntimeError(f"timed out waiting {timeout:.1f}s for {event_type}")


def _ws_uri(runner_url: str, session_id: str) -> str:
    parsed = urlparse(runner_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"session_id": session_id})
    return urlunparse((scheme, parsed.netloc, "/v1/audio/transcriptions/stream", "", query, ""))


def _load_audio(audio_path: str | None, *, seconds: float) -> bytes:
    if not audio_path:
        return _synthetic_pcm(seconds=seconds)
    path = Path(audio_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg produced no PCM audio from {path}")
    return completed.stdout


def _synthetic_pcm(*, seconds: float, sample_rate: int = 16000) -> bytes:
    sample_count = int(sample_rate * seconds)
    samples = array(
        "h",
        [
            int(0.25 * 32767 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            for index in range(sample_count)
        ],
    )
    return samples.tobytes()


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_events(
    events: list[dict],
    *,
    audio_bytes: int,
    chunk_bytes: int,
    sent_frame_count: int,
    wall_seconds: float,
) -> dict:
    event_counts: dict[str, int] = {}
    transcript_texts: list[str] = []
    speakers: set[str] = set()
    latencies: list[float] = []
    latencies_by_event_type: dict[str, list[float]] = {}
    processing_seconds: list[float] = []
    for event in events:
        event_type = str(event.get("event_type") or "<missing>")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if event_type == "transcript.segment" and str(event.get("text", "")).strip():
            transcript_texts.append(str(event["text"]).strip())
        if event_type == "speaker.update" and event.get("speaker"):
            speakers.add(str(event["speaker"]))
        latency = event.get("client_audio_timeline_latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
            latencies_by_event_type.setdefault(event_type, []).append(float(latency))
        processing = event.get("processing_seconds")
        if isinstance(processing, (int, float)):
            processing_seconds.append(float(processing))
    transcript_word_counts = [_word_count(text) for text in transcript_texts]
    transcript_char_counts = [len(text) for text in transcript_texts]
    return {
        "audio_duration_seconds": round(audio_bytes / 2 / 16000.0, 6),
        "chunk_bytes": chunk_bytes,
        "sent_frame_count": sent_frame_count,
        "received_event_count": len(events),
        "event_counts": event_counts,
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "transcript_segment_count": len(transcript_texts),
        "transcript_fragmentation": {
            "one_word_or_less_segments": sum(
                1 for word_count in transcript_word_counts if word_count <= 1
            ),
            "two_words_or_less_segments": sum(
                1 for word_count in transcript_word_counts if word_count <= 2
            ),
            "subword_like_segments": sum(
                1
                for text, word_count in zip(transcript_texts, transcript_word_counts)
                if word_count <= 1 and re.fullmatch(r"[A-Za-z]{1,4}", text)
            ),
            "words_per_segment": _stats(transcript_word_counts),
            "chars_per_segment": _stats(transcript_char_counts),
        },
        "transcript_preview": " ".join(transcript_texts)[:500],
        "wall_seconds": round(wall_seconds, 6),
        "latency_seconds": _stats(latencies),
        "latency_seconds_by_event_type": {
            event_type: _stats(values)
            for event_type, values in sorted(latencies_by_event_type.items())
        },
        "processing_seconds": _stats(processing_seconds),
        "models": next((event.get("models") for event in events if event.get("models")), None),
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+|[^\w\s]", text))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": round(sorted_values[0], 6),
        "p50": round(_percentile(sorted_values, 0.5), 6),
        "p95": round(_percentile(sorted_values, 0.95), 6),
        "max": round(sorted_values[-1], 6),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1)
