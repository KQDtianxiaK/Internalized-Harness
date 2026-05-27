from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    load_causal_lm,
    load_tokenizer,
    normalize,
    read_jsonl,
    save_vector,
    set_seed,
)


@torch.inference_mode()
def full_code_activation(model, tokenizer, code: str, layer_index: int) -> tuple[torch.Tensor, float]:
    """Extract mean hidden state across all tokens of the code snippet."""
    inputs = tokenizer(code, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer_index][0].float()
    # Mean over all real tokens (exclude padding if any)
    seq_mean = h.mean(dim=0).cpu()
    resid_norm = h.norm(dim=-1).mean().item()
    return seq_mean, resid_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--pairs", default=str(DEFAULT_OUTPUT_DIR / "data" / "contrastive_pairs.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_contrastive_fullpool.pt"))
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    pairs = read_jsonl(args.pairs)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No pairs loaded")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)

    from experiments.neural_claude_md.common import get_decoder_layers
    layers = get_decoder_layers(model)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(
            f"--layer-index {args.layer_index} out of range [0, {len(layers)}] "
            f"for model with {len(layers)} decoder layers"
        )

    deltas = []
    pos_norms = []
    neg_norms = []
    for row in tqdm(pairs, desc="extracting full-pool activations"):
        pos_h, pos_norm = full_code_activation(model, tokenizer, row["positive"], args.layer_index)
        neg_h, neg_norm = full_code_activation(model, tokenizer, row["negative"], args.layer_index)
        deltas.append(pos_h - neg_h)
        pos_norms.append(pos_norm)
        neg_norms.append(neg_norm)

    raw_direction = torch.stack(deltas).mean(dim=0)
    direction = normalize(raw_direction)
    resid_norm_mean = float(sum(pos_norms + neg_norms) / (len(pos_norms) + len(neg_norms)))

    save_vector(
        args.output,
        direction,
        {
            "kind": "contrastive_direction_fullpool",
            "layer_index": args.layer_index,
            "num_pairs": len(pairs),
            "resid_norm_mean": resid_norm_mean,
            "raw_norm": raw_direction.norm().item(),
            "model": str(Path(args.model)),
        },
    )
    print(f"Saved full-pool contrastive direction to {args.output}")
    print(f"resid_norm_mean={resid_norm_mean:.4f} raw_norm={raw_direction.norm().item():.4f}")


if __name__ == "__main__":
    main()
