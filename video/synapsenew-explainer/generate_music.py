#!/usr/bin/env python3
"""Generate an instrumental ElevenLabs music bed for the explainer."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import urllib.error

root = Path(__file__).resolve().parent
out = root / "assets" / "music" / "documentary.mp3"
out.parent.mkdir(parents=True, exist_ok=True)
duration_ms = round(json.loads((root / "timeline.json").read_text())["duration"] * 1000)
payload = json.dumps({
    "prompt": (
        "Instrumental only. A restrained minimal documentary pulse for a technical explainer: "
        "warm marimba, soft analog keys, light muted percussion, precise and curious mood, "
        "96 BPM, steady energy with a gentle lift near the end. No vocals, no voices, "
        "no cinematic drums, no dramatic drops, no trailer effects, no silence."
    ),
    "music_length_ms": duration_ms,
    "model_id": "music_v1",
    "force_instrumental": True,
    "sign_with_c2pa": True,
}).encode()
req = urllib.request.Request(
    "https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192",
    data=payload,
    method="POST",
    headers={
        "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    },
)
print(f"ElevenLabs music: {duration_ms/1000:.1f}s instrumental bed", flush=True)
try:
    with urllib.request.urlopen(req, timeout=600) as response:
        audio = response.read()
except urllib.error.HTTPError as exc:
    raise SystemExit(f"ElevenLabs music API {exc.code}: {exc.read().decode(errors='replace')}")
if len(audio) < 10000 or audio.startswith(b"{"):
    raise SystemExit(f"Unexpected music response: {audio[:300]!r}")
out.write_bytes(audio)
probe = subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", str(out),
], text=True)
print(f"Wrote {out} ({len(audio)/1e6:.1f} MB), duration {float(probe.strip()):.2f}s")
