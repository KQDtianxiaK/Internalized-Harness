from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    SAFETY_TEXT_HARNESS_SYSTEM_PROMPT,
    apply_chat,
    env_default,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    normalize,
    read_jsonl,
    save_vector,
    set_seed,
)


def pooled_activation(hidden: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "last_token":
        return hidden[-1, :].float()
    if pooling == "mean_prompt":
        return hidden.float().mean(dim=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


@torch.inference_mode()
def prompt_activation(
    model,
    tokenizer,
    prompt: str,
    *,
    system_prompt: str | None,
    layer_index: int,
    pooling: str,
) -> tuple[torch.Tensor, float]:
    inputs = apply_chat(tokenizer, prompt, system_prompt=system_prompt)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer_index][0]
    h = pooled_activation(hidden, pooling).cpu()
    resid_norm = hidden.float().norm(dim=-1).mean().item()
    return h, resid_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument(
        "--prompts",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "data" / "extract_prompts.jsonl"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--pooling", choices=["last_token", "mean_prompt"], default="last_token")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No extraction prompts loaded")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(f"--layer-index {args.layer_index} out of range for {len(layers)} layers")

    deltas = []
    base_norms = []
    system_norms = []
    for row in tqdm(rows, desc=f"extracting system delta {args.pooling}"):
        base_h, base_norm = prompt_activation(
            model,
            tokenizer,
            row["prompt"],
            system_prompt=None,
            layer_index=args.layer_index,
            pooling=args.pooling,
        )
        system_h, system_norm = prompt_activation(
            model,
            tokenizer,
            row["prompt"],
            system_prompt=SAFETY_TEXT_HARNESS_SYSTEM_PROMPT,
            layer_index=args.layer_index,
            pooling=args.pooling,
        )
        deltas.append(system_h - base_h)
        base_norms.append(base_norm)
        system_norms.append(system_norm)

    raw_direction = torch.stack(deltas).mean(dim=0)
    direction = normalize(raw_direction)
    resid_norm_mean = float(sum(base_norms) / len(base_norms))
    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR
        / "internal_harness_a"
        / "vectors"
        / f"v_system_delta_safety_l{args.layer_index}_{args.pooling}.pt"
    )
    save_vector(
        output,
        direction,
        {
            "kind": "system_prompt_delta",
            "rule": "file_deletion_safety",
            "layer_index": args.layer_index,
            "pooling": args.pooling,
            "num_prompts": len(rows),
            "resid_norm_mean": resid_norm_mean,
            "system_resid_norm_mean": float(sum(system_norms) / len(system_norms)),
            "raw_norm": raw_direction.norm().item(),
            "model": str(Path(args.model)),
            "prompts": str(Path(args.prompts)),
        },
    )
    print(f"Saved system delta vector to {output}")
    print(f"resid_norm_mean={resid_norm_mean:.4f} raw_norm={raw_direction.norm().item():.4f}")


if __name__ == "__main__":
    main()
