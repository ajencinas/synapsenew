# Review — Synapse 2.1B explainer

## Creative calls

Built storyline 1, **“One question, all the way through”**: a real arithmetic
prompt carries the viewer from tokenization through the transformer and into
SFT, LoRA and DPO, then returns to real Q&A evidence. The alternative storylines
left unbuilt are **“The 28-layer machine”** (spatial cutaway / endless zoom) and
**“Two answers, one training night”** (split-screen replay led by the training
log). Both remain in `STORYLINES.md`.

The opening changed from a side-by-side hold to a full-width before/after wipe
so the base output stays readable on a phone. Evidence answers use formatted
excerpts of actual generations and are labeled as such; no improved-model
answer was invented. The DPO pair is explicitly labeled as pair-construction
illustration rather than a claimed dataset row.

## Measured and inspected

- `npx hyperframes@0.8.226 check --json`: **zero errors** across lint, runtime,
  layout, motion and contrast.
- `snapshot --review`: inspected beat sheets, scene seams and 360 px phone
  sheets; independent reviewer inspected two revisions.
- Fixed a transformer text collision, clipped LoRA parameter count, footer
  placement, small essential text, the SFT mask hierarchy, DPO lane text and
  base-model Q&A excerpts. Added Q/K/V and residual paths to the transformer
  cutaway.
- Final decoded MP4 frames inspected at the hook, tokenization, SFT, LoRA, DPO,
  all three Q&A examples and the final card. No clipped text remains.
- Final file: H.264, 1920×1080, 30 fps, AAC stereo 48 kHz, duration
  **116.200 s** (root composition duration 116.147 s plus encoder rounding).
- Final audio: **−16.1 LUFS integrated**, true peak **−1.0 dBTP**, zero clips.
  The sound was measured, not heard.

## Warnings kept on purpose

Seven `line_unread` craft warnings remain for text arriving during a wipe or
clip-path entrance. Those are short entrance transients; the text then remains
static and readable for several seconds. There are no layout, motion or
contrast warnings.

## Weakest shot

The transformer cutaway is still the densest scene: exact architecture requires
more labels than the other scenes. It now has a visible dataflow, Q/K/V and
residual paths, but a longer version could spend an extra 3–4 seconds on the
attention calculation itself rather than the head counts.

## Music

ElevenLabs narration was generated normally. ElevenLabs music was also tried,
but this API key lacks `music_generation` permission. This cut therefore uses
narration plus local transition/answer sound effects, with no music bed.
`generate_music.py` is included for a key with that permission enabled.
