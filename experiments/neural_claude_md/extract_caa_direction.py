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
    marker_span_from_offsets,
    normalize,
    read_jsonl,
    save_vector,
    set_seed,
)


def variance_filtered_mean(deltas: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    """Compute mean of deltas after filtering out low-variance dimensions."""
    dim_vars = deltas.var(dim=0)
    threshold = dim_vars.quantile(1.0 - keep_ratio)
    mask = dim_vars >= threshold
    filtered = deltas[:, mask]
    direction = filtered.mean(dim=0)
    # Project back to full dimensionality (zero-pad filtered dims)
    full = torch.zeros(deltas.shape[1], dtype=direction.dtype)
    full[mask] = direction
    return full


def pca_first_component(deltas: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Run PCA on deltas and return the first principal component + explained variance ratio."""
    # deltas: [N, d]
    centered = deltas - deltas.mean(dim=0, keepdim=True)
    # SVD: centered = U S V^T, where V^T rows are principal components
    try:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vh[0]
    except RuntimeError:
        # Fallback for very small matrices or numerical issues
        # torch.svd returns V (not Vh), columns are right singular vectors
        _, s, v = torch.svd(centered)
        pc1 = v[:, 0]
    # Ensure PC1 points in the same general direction as the mean delta
    mean_delta = deltas.mean(dim=0)
    if torch.dot(pc1, mean_delta) < 0:
        pc1 = -pc1
    explained_var_ratio = (s[0] ** 2 / (s ** 2).sum()).item() if len(s) > 0 else 0.0
    return pc1, explained_var_ratio


@torch.inference_mode()
def extract_activations(model, tokenizer, rows, layer_index: int):
    """Extract positive and negative activations for all rows at the given layer."""
    pos_acts = []
    neg_acts = []
    pos_norms = []
    neg_norms = []

    for row in tqdm(rows, desc=f"Extracting activations for layer {layer_index}"):
        pos_ids, pos_idx = marker_span_from_offsets(
            tokenizer, row["positive"], row["positive_marker"]
        )
        neg_ids, neg_idx = marker_span_from_offsets(
            tokenizer, row["negative"], row["negative_marker"]
        )

        pos_ids = pos_ids.to(model.device)
        neg_ids = neg_ids.to(model.device)

        pos_out = model(input_ids=pos_ids, output_hidden_states=True, use_cache=False)
        neg_out = model(input_ids=neg_ids, output_hidden_states=True, use_cache=False)

        pos_h = pos_out.hidden_states[layer_index][0, pos_idx, :].float().mean(dim=0).cpu()
        neg_h = neg_out.hidden_states[layer_index][0, neg_idx, :].float().mean(dim=0).cpu()

        pos_acts.append(pos_h)
        neg_acts.append(neg_h)
        # Use full-sequence mean norm for consistent steering scale across methods
        pos_norms.append(pos_out.hidden_states[layer_index][0].float().norm(dim=-1).mean().item())
        neg_norms.append(neg_out.hidden_states[layer_index][0].float().norm(dim=-1).mean().item())

    return (
        torch.stack(pos_acts),
        torch.stack(neg_acts),
        float(sum(pos_norms + neg_norms) / (len(pos_norms) + len(neg_norms))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--pairs", default=str(DEFAULT_OUTPUT_DIR / "data" / "contrastive_pairs.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_caa.pt"))
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--method", default="delta_pca", choices=["delta_pca", "variance_filtered"])
    parser.add_argument("--variance-keep-ratio", type=float, default=0.5)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    pairs = read_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No pairs loaded")
    if not (0.0 <= args.variance_keep_ratio <= 1.0):
        raise ValueError(f"--variance-keep-ratio must be in [0, 1], got {args.variance_keep_ratio}")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)

    from experiments.neural_claude_md.common import get_decoder_layers
    layers = get_decoder_layers(model)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(
            f"--layer-index {args.layer_index} out of range [0, {len(layers)}] "
            f"for model with {len(layers)} decoder layers"
        )

    pos_stack, neg_stack, resid_norm_mean = extract_activations(
        model, tokenizer, pairs, args.layer_index
    )

    deltas = pos_stack - neg_stack

    if args.method == "delta_pca":
        direction, explained_var = pca_first_component(deltas)
        direction = normalize(direction)
        meta = {
            "kind": "caa_direction",
            "method": "delta_pca",
            "layer_index": args.layer_index,
            "num_pairs": len(pairs),
            "resid_norm_mean": resid_norm_mean,
            "explained_variance_ratio": explained_var,
            "delta_mean_norm": deltas.mean(dim=0).norm().item(),
            "model": str(Path(args.model)),
        }
    elif args.method == "variance_filtered":
        raw_direction = variance_filtered_mean(deltas, keep_ratio=args.variance_keep_ratio)
        direction = normalize(raw_direction)
        meta = {
            "kind": "caa_direction",
            "method": "variance_filtered",
            "layer_index": args.layer_index,
            "num_pairs": len(pairs),
            "resid_norm_mean": resid_norm_mean,
            "variance_keep_ratio": args.variance_keep_ratio,
            "raw_norm": raw_direction.norm().item(),
            "model": str(Path(args.model)),
        }
    else:
        raise ValueError(f"Unknown method: {args.method}")

    save_vector(args.output, direction, meta)
    print(f"Saved CAA direction to {args.output}")
    for k, v in meta.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
