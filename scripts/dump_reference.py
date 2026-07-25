#!/usr/bin/env python3
"""Dump reference activations from the transformers implementation of Qwen3.

This is the oracle the Rust forward pass is checked against. It exists because
the three things Qwen3 does differently from Llama (RMS norm applied to Q and K
per head before RoPE, a head width decoupled from hidden_size / num_heads, and
grouped-query attention) all fail silently: an implementation that gets any of
them wrong still runs and still produces fluent text.

Two scales are dumped, and they answer different questions.

The tiny model is two layers at toy widths, with weights from a fixed seed. Its
fixture is a few hundred kilobytes, so it is committed and CI runs it, and it is
what proves the structure: at a relative tolerance of 1e-5 any of the mistakes
above is unmissable.

The real Qwen3-0.6B is dumped at the same widths it ships with and its fixture
is not committed. It is what says the implementation still agrees across 28
layers, where rounding has had time to accumulate.

Both are dumped in float32, including the real model, whose checkpoint is
bfloat16. candle has no bf16 matmul on the CPU, so the comparison that isolates
the implementation from the backend has to run in f32 on both sides. What the
serving dtype costs is then its own measurement, taken in bf16 on Metal against
this same f32 reference.

    python scripts/dump_reference.py tiny  crates/pagedllm/tests/fixtures/tiny
    python scripts/dump_reference.py real  models/Qwen3-0.6B  models/reference

Both write a safetensors file of named activations plus a JSON manifest naming
the prompt and the relative tolerance the dump was taken at.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

# A fixed prompt. Token ids rather than text, so the fixture does not depend on
# the tokenizer, which is checked separately.
REAL_PROMPT = [151644, 872, 198, 3838, 374, 264, 4013, 30, 151645, 198]

SEED = 20260725


def tiny_config() -> Qwen3Config:
    """A Qwen3 small enough to commit, shaped to exercise what breaks.

    Every width is chosen against a specific mistake. head_dim is 16 while
    hidden_size / num_attention_heads is 8, so an implementation that derives
    the head width instead of reading it produces wrong shapes. Four query heads
    over two key heads make the grouping ratio 2, so a repeat_kv that broadcasts
    in the wrong order lands on the wrong head. Nothing here is square, so a
    transposed weight cannot pass by accident.
    """
    return Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
        torch_dtype="float32",
    )


def captured_modules(num_layers: int) -> list[str]:
    """Module names to hook, in the order the forward pass reaches them."""
    names = ["model.embed_tokens"]
    for i in range(num_layers):
        p = f"model.layers.{i}"
        names += [
            f"{p}.input_layernorm",
            f"{p}.self_attn.q_proj",
            f"{p}.self_attn.k_proj",
            f"{p}.self_attn.v_proj",
            f"{p}.self_attn.q_norm",
            f"{p}.self_attn.k_norm",
            f"{p}.self_attn.o_proj",
            f"{p}.self_attn",
            f"{p}.post_attention_layernorm",
            f"{p}.mlp.gate_proj",
            f"{p}.mlp.up_proj",
            f"{p}.mlp.down_proj",
            f"{p}.mlp",
            p,
        ]
    names += ["model.norm", "lm_head"]
    return names


def first_tensor(value):
    """Unwrap what a module returned down to its first tensor."""
    while isinstance(value, (tuple, list)):
        value = next(v for v in value if v is not None)
    return value


def dump(model: Qwen3ForCausalLM, prompt: list[int], out_dir: Path, tolerance: float) -> None:
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    for name in captured_modules(model.config.num_hidden_layers):
        module = modules[name]

        # Cloned, not just detached: a module that returns its child's output
        # unchanged (self_attn around o_proj, lm_head into logits) would
        # otherwise store one buffer under two names, which safetensors refuses.
        def hook(_module, args, output, name=name):
            captured[f"{name}.out"] = first_tensor(output).detach().to(torch.float32).clone()
            # o_proj and down_proj take the result of the attention and of the
            # gated product, which no other hook sees.
            if name.endswith(("o_proj", "down_proj")):
                captured[f"{name}.in"] = first_tensor(args).detach().to(torch.float32).clone()

        handles.append(module.register_forward_hook(hook))

    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    for h in handles:
        h.remove()

    captured["logits"] = out.logits.detach().to(torch.float32).contiguous()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(captured, str(out_dir / "activations.safetensors"))
    manifest = {
        "prompt": prompt,
        "relative_tolerance": tolerance,
        "dtype": str(next(model.parameters()).dtype),
        "num_hidden_layers": model.config.num_hidden_layers,
        "tensors": sorted(captured),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(captured)} tensors to {out_dir}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode = sys.argv[1]

    if mode == "tiny":
        out_dir = Path(sys.argv[2])
        torch.manual_seed(SEED)
        config = tiny_config()
        model = Qwen3ForCausalLM(config).to(torch.float32)
        # Default initialisation leaves the norms at exactly one, which would
        # let an implementation that drops the norm weight pass.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("norm.weight"):
                    param.copy_(torch.rand_like(param) * 0.5 + 0.75)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        prompt = [3, 17, 42, 8, 99, 1, 64, 7]
        dump(model, prompt, out_dir, tolerance=1e-5)
        return 0

    if mode == "real":
        model_dir, out_dir = Path(sys.argv[2]), Path(sys.argv[3])
        # Defaults to float32, not the checkpoint's bfloat16, because candle has
        # no bf16 matmul on the CPU and the comparison that isolates the
        # implementation from the backend has to run in f32 on both sides.
        # Dumping bfloat16 as well is what makes the GPU path checkable against
        # something that ran in the same dtype it does.
        wanted = sys.argv[4] if len(sys.argv) > 4 else "float32"
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[wanted]
        model = Qwen3ForCausalLM.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="eager"
        )
        # f32 measured at 9.4e-6 across all 452 tensors on an M4 Pro, so five
        # times that and no looser: a threshold two orders of magnitude above
        # what the code does would pass a regression as easily as a match. bf16
        # carries roughly three decimal digits, and its threshold is set from
        # the same rule against its own measurement.
        # bf16 carries about three decimal digits, and two implementations that
        # order their reductions differently diverge by percents after a few
        # layers. Its threshold is a break detector, not a precision claim: the
        # correctness gate in bf16 is that the greedy tokens still match.
        tolerance = 5e-5 if dtype is torch.float32 else 1e-1
        dump(model, REAL_PROMPT, out_dir, tolerance=tolerance)
        return 0

    print(f"unknown mode {mode!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
