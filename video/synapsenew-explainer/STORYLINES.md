# Storylines — Synapse 2.1B explainer

## 1. One question, all the way through (RECOMMENDED)

**Device / form:** one real prompt becomes the object of the film; a continuous worked example grows into the architecture and then into before/after evidence.

**What the viewer sees and feels:** the improved model answers “What is 17 + 25?” while the raw base rambles. The question then travels through token IDs, embeddings, transformer blocks and next-token selection before the film reveals how SFT, LoRA and DPO changed those answers. It feels like watching one question being understood, then watching the model learn to answer it well.

**First two seconds:** a full-frame question, “What is 17 + 25?”, with the base model’s noisy answer on the left and the improved model’s concise “17 + 25 = 42” wiping across it.

**Beats:** real answer first → tokenization → transformer → next-token loop → supervised fine-tuning → LoRA on a laptop → DPO preferences → three real Q&A comparisons → measured results.

**Look:** a light technical workbench: pale blue-grey graph grid, dark ink, green only for the improved path and verified answers; Inter plus JetBrains Mono.

**Sound:** clear ElevenLabs narration, subtle transition effects, correct-answer ticks; voice carries the clock.

## 2. The 28-layer machine

**Device / form:** the model is a physical machine on a bench; an endless zoom dives from its outer shell into attention, adapters and preference gates.

**What the viewer sees and feels:** tokens enter a 28-stage machine as labeled blocks. The camera dives into one block, then into a small LoRA side-module, and finally through a two-lane “chosen / rejected” gate. It feels mechanical and spatial rather than instructional.

**First two seconds:** the machine finishes printing “42” while a counter clicks from token 1 to token 2.

**Beats:** output first → zoom into the machine → attention internals → adapter cutaway → preference gate → Q&A output strip → final stamp.

**Look:** ink-blue machine casings, white cutaway faces, green adapter modules, hard contact shadows, no photographic assets.

**Sound:** precise mechanical clicks and one restrained impact at the final validation stamp beneath ElevenLabs narration.

## 3. Two answers, one training night

**Device / form:** split-screen before/after scenes replay the same prompts while a training log runs between them; each comparison causes the next training stage to appear.

**What the viewer sees and feels:** the base model fails in its own words, the new model answers, and the film rewinds to show which stage caused the improvement. The viewer experiences the contrast first and understands the mechanism second.

**First two seconds:** the base model’s long repeated arithmetic answer compresses into a single red “unstructured” stamp as “42” lands on the improved side.

**Beats:** answer contrast → rewind to pretraining → SFT mask → LoRA memory budget → DPO pair choice → three more contrasts → results ledger.

**Look:** bright editorial split screens, red only for rejected/base output, green for chosen output, oversized answer text.

**Sound:** sparse ElevenLabs narration, wipe effects on each comparison, music only if a suitable local/catalog track is available.

---

**Chosen for delivery:** storyline 1, because the user asked both how the model works and for Q&A examples; one real prompt provides the clearest spine for both.
