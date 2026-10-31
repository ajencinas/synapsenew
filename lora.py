#!/usr/bin/env python3
"""LoRA for SynapseGPT — the memory trick that lets a 2.1B model fine-tune on a
16 GB laptop GPU.

Why LoRA here (and not full FT):
  - fp32 full FT needs params (8.4 GB) + grads (8.4 GB) + Adam states (16.8 GB)
    = 33 GB — an A100-class budget, not a laptop one.
  - bf16 full FT with Adafactor would fit memory-wise, but bf16 parameter ULP at
    weight scale (~0.01) is ~4e-5 while an SFT step at lr=1.5e-5 moves weights
    ~1e-6..1e-5 per step: updates round away (silent no-op training).
  - So: freeze the bf16 base (no update-rounding problem on it), train fp32 LoRA
    adapters (rank 64 ≈ 76M params ≈ 1.2 GB with Adam states).  Base weights stay
    exactly the pretrained ones; only the additive low-rank deltas move.

Adapters are merged into the base weights when checkpoints are saved, so every
saved .pth is directly loadable by the existing SynapseInfer / bench tooling
(no adapter plumbing needed at inference time).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2")


def raw_model(model):
    """Unwrap torch.compile's OptimizedModule so state_dict/named_modules use
    the original key names (no '_orig_mod.' prefix)."""
    return getattr(model, "_orig_mod", model)


class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(drop(x))).  `enabled=False` gives the frozen
    base output — used for the DPO reference pass (ref = policy with the
    in-training adapters switched off, i.e. the frozen start-of-DPO policy)."""

    def __init__(self, base: nn.Linear, r=64, alpha=64, dropout=0.05,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.enabled = True
        self.drop = nn.Dropout(dropout)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.02)
        self.base.weight.requires_grad_(False)

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.drop(x).to(self.lora_A.dtype)
        delta = F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def delta_weight(self):
        """fp32 (out,in) additive weight delta."""
        return (self.lora_B.float() @ self.lora_A.float()) * self.scale


def inject_lora(model, r=64, alpha=64, dropout=0.05, targets=DEFAULT_TARGETS,
                device=None, dtype=torch.float32):
    """Wrap every Linear whose leaf name is in `targets`; freeze the rest."""
    n_wrapped = 0
    for parent_name, parent in raw_model(model).named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(parent, child_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout,
                                   device=device, dtype=dtype))
                n_wrapped += 1
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad_(True)
            n_train += p.numel()
    print(f"[lora] wrapped {n_wrapped} linears | trainable {n_train/1e6:.1f}M params "
          f"(r={r}, alpha={alpha}, targets={targets})")
    return model


def lora_parameters(model):
    return [p for name, p in raw_model(model).named_parameters()
            if ".lora_A" in name or ".lora_B" in name]


def set_lora_enabled(model, enabled: bool):
    for m in raw_model(model).modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_state_dict(model):
    out = {}
    for name, mod in raw_model(model).named_modules():
        if isinstance(mod, LoRALinear):
            out[name + ".lora_A"] = mod.lora_A.detach().cpu()
            out[name + ".lora_B"] = mod.lora_B.detach().cpu()
    return out


def load_lora_state_dict(model, state):
    names = dict(raw_model(model).named_modules())
    for name, mod in names.items():
        if isinstance(mod, LoRALinear):
            a = state.get(name + ".lora_A")
            b = state.get(name + ".lora_B")
            if a is None or b is None:
                raise SystemExit(f"[lora] missing adapter tensors for {name}")
            mod.lora_A.data.copy_(a.to(mod.lora_A.device, mod.lora_A.dtype))
            mod.lora_B.data.copy_(b.to(mod.lora_B.device, mod.lora_B.dtype))


def num_lora_params(model):
    return sum(p.numel() for p in lora_parameters(model))


def merged_state_dict(model, base_dtype=torch.float32):
    """Full state_dict with LoRA deltas merged into base weights, under the
    ORIGINAL (unwrapped) key names — byte-compatible with SynapseGPT /
    SynapseInfer.checkpoint loading.  All float tensors converted to fp32
    (matching the existing checkpoint format; loaders cast on their side)."""
    model = raw_model(model)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            w = mod.base.weight.detach().to("cpu", torch.float32)
            w = w + mod.delta_weight().cpu()
            out[name + ".weight"] = w
    sd = model.state_dict()
    for k, v in sd.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B") or k.endswith(".base.weight"):
            continue
        out[k] = v.detach().to("cpu", torch.float32) if v.is_floating_point() else v.detach().cpu().clone()
    return out
