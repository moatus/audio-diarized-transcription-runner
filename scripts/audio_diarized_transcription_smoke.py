#!/usr/bin/env python3
"""Smoke helper for the audio diarized transcription runner."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the local diarized transcription runner.")
    parser.add_argument("audio_path")
    parser.add_argument(
        "--runner-url",
        default=os.getenv("AUDIO_DIARIZED_TRANSCRIPTION_RUNNER_URL", "http://localhost:8080"),
    )
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--response-format", default="json", choices=["json", "text", "srt", "vtt"])
    parser.add_argument("--require-text", action="store_true")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    boundary = f"----livepeer-{uuid.uuid4().hex}"
    body = _multipart_body(
        boundary=boundary,
        audio_path=audio_path,
        fields={
            "model": "nemo-diarized-transcription-meeting-v0",
            "language": "en",
            "preset": "meeting",
            "max_speakers": str(args.max_speakers),
            "response_format": "verbose_json" if args.response_format == "json" else args.response_format,
            "diarization": "true",
            "timestamp_granularities[]": "segment,word",
            "include_words": "true",
            "include_artifacts": "true",
            **({"num_speakers": str(args.num_speakers)} if args.num_speakers else {}),
        },
    )
    request = urllib.request.Request(
        f"{args.runner_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        sys.stderr.write(error.read().decode("utf-8") + "\n")
        return error.code

    if args.response_format == "json":
        payload = json.loads(response_body)
        print(json.dumps(payload, indent=2))
        if args.require_text and not str(payload.get("text", "")).strip():
            raise RuntimeError("runner returned empty text")
    else:
        print(response_body)
        if args.require_text and not response_body.strip():
            raise RuntimeError("runner returned empty text")
    return 0


def _multipart_body(boundary: str, audio_path: Path, fields: dict[str, str]) -> bytes:
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode(),
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
