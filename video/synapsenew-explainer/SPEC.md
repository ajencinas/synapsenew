GOAL: A viewer can explain how Synapse turns tokens into an answer, how SFT + LoRA + DPO improve it, and can cite real Q&A evidence.
OUTPUT: HyperFrames, 1920 x 1080, 30 fps, MP4, ElevenLabs narration.
TIME: 116.147 seconds x 30 = 3484 frames; scene durations measured from ElevenLabs narration + 0.60 s (0.85 s hook).
LOOK:
  Ground #EEF2F6 with a 64 px graph grid at 4% ink opacity; no gradients.
  Ink #101828; secondary #475467; accent #079455 only on the improved path, chosen answer and verified result.
  Rejected/base ink #B42318, used only on failure excerpts and rejected preference lane.
  Display Inter 900, 64–104 px, tracking -0.035 em; body Inter 400/700, 30–38 px; token/code JetBrains Mono 30–40 px.
  Medium: technical workbench / architectural cutaway. White #FFFFFF work surfaces with 2 px #D0D5DD borders, 14 px radius, 0 12 px 30 px rgba(16,24,40,.08).
  Depth: one flat shadow per work surface; grain 0%, vignette 0%, glow 0%.
OBJECTS:
  Real Q&A transcript panels; token chips with real text/IDs; 2560-wide vector represented by 12 bars (dimension label carries exact value);
  28-layer stack represented by 7 visible layers plus “× 28”; attention Q/K/V paths; SwiGLU block; next-token probability ruler;
  SFT ChatML transcript with assistant-only loss mask; LoRA base/adapters with exact parameter counts; DPO chosen/rejected lanes;
  three real Q&A examples from dashboard_results.json; validation-loss bars with exact measured values.
MOTION:
  Enter expo.out 0.45–0.65 s; camera/diagram moves power2.inOut 0.6–0.9 s; exits power3.in 0.2–0.3 s.
  Hero move one per shot; secondary reveals trail 0.12 s; token selection linear, 0.12 s per step.
  Q&A wipe uses installed before-after-wipe mechanic (clip-path, 1.05 s, sine.inOut).
SHOTS (target times; adjust to narration):
  0–7.119 s Hook: real arithmetic answer comparison; base excerpt wipes into big-LoRA “17 + 25 = 42.”
  7.119–18.351 s Tokens: the same question becomes token chips, IDs, then a 2560-d vector.
  18.351–33.841 s Transformer: vector enters the 28-block stack; one block opens into GQA attention + SwiGLU.
  33.841–44.289 s Decode: next-token scores select “ =”, append it, and repeat.
  44.289–59.779 s SFT: ChatML transcript builds; question labels mask to −100, assistant labels remain active.
  59.779–77.698 s LoRA: frozen bf16 base stays; fp32 low-rank adapters train; merged output leaves the machine.
  77.698–89.426 s DPO: chosen curated answer versus rejected corrupted answer; preference margin moves the policy.
  89.426–102.225 s Evidence: three actual Q&A pairs (change, algebra, PONG) resolve one after another.
  102.225–116.147 s Results: measured bars (2.746 → 1.198; bench 32% → 70%, DPO 75%) and final line “Same base. Better answers.”
BANNED: dark navy void, floating particles, glow/bloom/neon, purple/blue gradients, glass cards, cream-paper-serif styling, pill-chip decoration, three equal KPI cards, all-at-once fades, repeated fade+slide entrances, corner logos, decorative HUD labels, moving titles after landing, narrated text repeated verbatim on screen, essential Q&A or loss-mask text below 48 px at 1080, dense formula rows without a visible dataflow path.
QA: frames at 0.2, 4, 12, 24, 35, 45, 57, 69, 80, 86, 94 s and final-0.2 s; every real Q&A readable at 360 px; no answer text clipped; every number matches the source; voice and SFX present; no render-time network dependency.
SOURCES: synapse_model.py; lora.py; train_sft.py; train_dpo.py; reports/dashboard_results.json; reports/eval_loss_*.json; reports/bench_*/; reports/sft_training_latest_biglora.json.
AUDIO: ElevenLabs River narration generated and measured per scene. Music generation attempted with ElevenLabs POST /v1/music; the available key lacks music_generation permission, so this cut uses narration + local sound effects only.
