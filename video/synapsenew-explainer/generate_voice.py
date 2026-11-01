#!/usr/bin/env python3
"""Generate the requested ElevenLabs narration and measure scene durations."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "voice"
OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("ELEVENLABS_API_KEY")
if not KEY:
    raise SystemExit("ELEVENLABS_API_KEY is required (requested provider: ElevenLabs)")

VOICE_ID = "SAz9YHcvj6GT2YYXdXww"  # River — relaxed, neutral, informative
URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128"
lines = json.loads((ROOT / "narration.json").read_text())
records = []
for line in lines:
    dest = OUT / f"{line['id']}.mp3"
    payload = json.dumps({
        "text": line["text"],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.48,
            "similarity_boost": 0.78,
            "style": 0.12,
            "use_speaker_boost": True,
            "speed": 0.98,
        },
    }).encode()
    req = urllib.request.Request(URL, data=payload, method="POST", headers={
        "xi-api-key": KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    })
    print(f"ElevenLabs: {line['id']} ({len(line['text'].split())} words)", flush=True)
    with urllib.request.urlopen(req, timeout=180) as response:
        audio = response.read()
    if len(audio) < 1000 or audio.startswith(b"{"):
        raise RuntimeError(f"Unexpected ElevenLabs response for {line['id']}: {audio[:200]!r}")
    dest.write_bytes(audio)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(dest),
    ], text=True)
    duration = float(probe.strip())
    records.append({**line, "path": str(dest.relative_to(ROOT)), "duration": round(duration, 3)})
    print(f"  -> {dest.name}: {duration:.2f}s", flush=True)

start = 0.0
for record in records:
    record["start"] = round(start, 3)
    record["scene_duration"] = round(record["duration"] + (0.85 if record["id"] == "hook" else 0.60), 3)
    start += record["scene_duration"]
(ROOT / "timeline.json").write_text(json.dumps({
    "duration": round(start, 3), "scenes": records,
}, indent=2) + "\n")
print(f"Total: {start:.2f}s")
