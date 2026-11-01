# Synapse 2.1B — HyperFrames explainer

A 116.147-second, 1920×1080 explainer of the Synapse 2.1B architecture and the
synapsenew SFT → LoRA → DPO pipeline, with ElevenLabs narration and real Q&A
from `reports/dashboard_results.json`.

## Watch / edit

- Render: `../synapsenew_explainer.mp4`
- Studio: `npx hyperframes@0.8.226 preview --background`
- Director spec: `SPEC.md`
- Alternative storylines: `STORYLINES.md`
- Review notes: `REVIEW.md`

## Scenes

1. Real arithmetic answer: pretrain base vs big-LoRA
2. Exact tokenizer output and 2,560-dimensional embeddings
3. 28 transformer blocks, GQA + RoPE, SwiGLU and residuals
4. Next-token decoding and KV caching
5. Assistant-only masked loss in supervised fine-tuning
6. Frozen bf16 base + fp32 LoRA adapters (r=256, 302.8M trainable)
7. DPO chosen/rejected pairs and the adapters-off reference
8. Real Q&A: change, algebra, exact-format PONG
9. Validation loss, chat bench, GSM8K and ARC-Challenge results

## Regenerate narration

Requires `ELEVENLABS_API_KEY` in the environment. The requested provider is
ElevenLabs (River, `eleven_multilingual_v2`).

```bash
python3 generate_voice.py
```

This rewrites `assets/voice/*.mp3` and `timeline.json`. Scene times in
`index.html` and the compositions use the current measured durations; if the
script changes, update those timings to match.

## Music

ElevenLabs does provide a music endpoint (`POST /v1/music`), and
`generate_music.py` is ready to create the instrumental bed. During this build
the available API key returned:

```text
missing_permissions: music_generation
```

Enable that permission on the key, then run:

```bash
python3 generate_music.py
```

Add the resulting `assets/music/documentary.mp3` as a root `<audio>` clip at
low volume (about 0.12–0.14 under the voice). The delivered cut therefore uses
narration + sound effects, without a music bed.

## Validate / render

```bash
npx hyperframes@0.8.226 check --json
npx hyperframes@0.8.226 snapshot --review
npx hyperframes@0.8.226 render --quality delivery --output ../synapsenew_explainer.mp4
```

All visual and audio media used by the composition are local. The only network
calls in the build scripts are the explicit ElevenLabs generation requests.
