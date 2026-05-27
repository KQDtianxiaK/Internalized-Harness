from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    SAFETY_TEXT_HARNESS_SYSTEM_PROMPT,
    TEXT_HARNESS_SYSTEM_PROMPT,
    env_default,
    ensure_dir,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    normalize,
    set_seed,
)


def parse_layers(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def prompt_text_for_rule(rule: str) -> str:
    if rule == "safety":
        return SAFETY_TEXT_HARNESS_SYSTEM_PROMPT
    if rule == "logging":
        return TEXT_HARNESS_SYSTEM_PROMPT
    raise ValueError(f"Unknown rule: {rule}")


@torch.inference_mode()
def extract_bundle(model, tokenizer, rule_text: str, layers: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": rule_text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    vectors = {}
    resid_norms = {}
    for layer_index in layers:
        hidden = out.hidden_states[layer_index][0].float()
        raw = hidden.mean(dim=0).cpu()
        vectors[str(layer_index)] = normalize(raw)
        resid_norms[str(layer_index)] = hidden.norm(dim=-1).mean().item()
    return vectors, resid_norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--rule", choices=["safety", "logging"], default="safety")
    parser.add_argument("--layers", default="12,16,20,24,28")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    layers = parse_layers(args.layers)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    decoder_layers = get_decoder_layers(model)
    for layer_index in layers:
        if not (0 <= layer_index < len(decoder_layers)):
            raise ValueError(f"Layer {layer_index} out of range for model with {len(decoder_layers)} layers")

    rule_text = prompt_text_for_rule(args.rule)
    vectors, resid_norms = extract_bundle(model, tokenizer, rule_text, layers)
    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR
        / "internal_harness_b"
        / "vectors"
        / f"activation_prefix_{args.rule}_layers_{'_'.join(map(str, layers))}.pt"
    )
    ensure_dir(output.parent)
    torch.save(
        {
            "vectors": vectors,
            "meta": {
                "kind": "activation_prefix_summary",
                "rule": args.rule,
                "layers": layers,
                "resid_norms": resid_norms,
                "model": str(Path(args.model)),
            },
        },
        output,
    )
    print(f"Saved activation prefix bundle to {output}")
    print(f"layers={layers}")


if __name__ == "__main__":
    main()
