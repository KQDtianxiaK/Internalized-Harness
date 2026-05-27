from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_NLA_AR,
    DEFAULT_NLA_AV,
    DEFAULT_NLA_REPO,
    DEFAULT_OUTPUT_DIR,
    dtype_from_name,
    import_nla,
    load_causal_lm,
    load_tokenizer,
    marker_span_from_offsets,
    normalize,
    read_jsonl,
    save_vector,
    set_seed,
    write_json,
)

KEYWORDS = ["logging", "logger", "print", "stdout", "log", "output"]


def score_explanation_keywords(text: str) -> dict[str, float]:
    text_lower = text.lower()
    scores = {}
    for kw in KEYWORDS:
        scores[f"has_{kw}"] = float(kw in text_lower)
    scores["keyword_score"] = sum(scores.values()) / len(KEYWORDS)
    return scores


def composite_score(r: dict[str, Any]) -> float:
    # Higher cos, higher keyword_score, higher inter_distance, lower intra_var
    return (
        r["ar_cos"] * 0.4
        + r["keyword_score"] * 0.3
        + min(r["inter_distance"] / 100, 1.0) * 0.2
        - (r["pos_intra_var"] + r["neg_intra_var"]) / 200 * 0.1
    )


@torch.inference_mode()
def extract_all_layer_activations(model, tokenizer, rows, layer_indices: list[int]):
    """Run the model once per input and cache hidden states for all target layers."""
    max_layer = max(layer_indices)

    per_layer_pos = {layer: [] for layer in layer_indices}
    per_layer_neg = {layer: [] for layer in layer_indices}
    pos_norms = []
    neg_norms = []

    for row in rows:
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

        for layer in layer_indices:
            pos_h = pos_out.hidden_states[layer][0, pos_idx, :].float().mean(dim=0).cpu()
            neg_h = neg_out.hidden_states[layer][0, neg_idx, :].float().mean(dim=0).cpu()
            per_layer_pos[layer].append(pos_h)
            per_layer_neg[layer].append(neg_h)

        # Use max_layer for norm estimation (approximate, consistent across layers)
        pos_norms.append(
            pos_out.hidden_states[max_layer][0].float().norm(dim=-1).mean().item()
        )
        neg_norms.append(
            neg_out.hidden_states[max_layer][0].float().norm(dim=-1).mean().item()
        )

    return per_layer_pos, per_layer_neg, pos_norms, neg_norms


def compute_layer_directions(
    per_layer_pos: dict[int, list[torch.Tensor]],
    per_layer_neg: dict[int, list[torch.Tensor]],
    pos_norms: list[float],
    neg_norms: list[float],
    layer_indices: list[int],
) -> dict[int, dict[str, Any]]:
    results = {}
    for layer in layer_indices:
        pos_stack = torch.stack(per_layer_pos[layer])
        neg_stack = torch.stack(per_layer_neg[layer])
        pos_mean = pos_stack.mean(dim=0)
        neg_mean = neg_stack.mean(dim=0)

        raw_direction = pos_mean - neg_mean
        direction = normalize(raw_direction)

        results[layer] = {
            "direction": direction,
            "raw_norm": raw_direction.norm().item(),
            "resid_norm_mean": float(sum(pos_norms + neg_norms) / (len(pos_norms) + len(neg_norms))),
            "inter_distance": raw_direction.norm().item(),
            "pos_intra_var": pos_stack.var(dim=0).mean().item(),
            "neg_intra_var": neg_stack.var(dim=0).mean().item(),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--pairs", default=str(DEFAULT_OUTPUT_DIR / "data" / "contrastive_pairs.jsonl"))
    parser.add_argument("--nla-repo", default=str(DEFAULT_NLA_REPO))
    parser.add_argument("--av", default=str(DEFAULT_NLA_AV))
    parser.add_argument("--ar", default=str(DEFAULT_NLA_AR))
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    parser.add_argument("--limit", type=int, default=16, help="Number of pairs to use for fast layer evaluation")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "layer_sweep"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--model-dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction and only score existing vectors")
    parser.add_argument("--skip-score", action="store_true", help="Skip scoring and only extract vectors")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract directions per layer
    if not args.skip_extract:
        pairs = read_jsonl(args.pairs)
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            raise ValueError("No pairs loaded; check --pairs file or --limit")

        tokenizer = load_tokenizer(args.model)
        model = load_causal_lm(args.model, dtype=args.model_dtype, device_map=args.device_map)

        print(f"Extracting activations for {len(pairs)} pairs across layers {args.layers} ...")
        per_layer_pos, per_layer_neg, pos_norms, neg_norms = extract_all_layer_activations(
            model, tokenizer, pairs, args.layers
        )

        layer_results = compute_layer_directions(
            per_layer_pos, per_layer_neg, pos_norms, neg_norms, args.layers
        )

        for layer, meta in layer_results.items():
            save_vector(
                out_dir / f"v_layer_{layer:02d}.pt",
                meta["direction"],
                {
                    "kind": "contrastive_direction",
                    "layer_index": layer,
                    "num_pairs": len(pairs),
                    "resid_norm_mean": meta["resid_norm_mean"],
                    "raw_norm": meta["raw_norm"],
                    "inter_distance": meta["inter_distance"],
                    "pos_intra_var": meta["pos_intra_var"],
                    "neg_intra_var": meta["neg_intra_var"],
                    "model": str(Path(args.model)),
                },
            )
        print(f"Saved layer directions to {out_dir}")
        del model
        torch.cuda.empty_cache()

    if args.skip_score:
        return

    # Stage 2: Score with AV + AR
    nla = import_nla(args.nla_repo)
    dtype = dtype_from_name(args.dtype)
    client = nla.NLAClient(args.av, sglang_url=args.sglang_url)
    critic = nla.NLACritic(args.ar, device=args.device, dtype=dtype)

    results = []
    for layer in tqdm(args.layers, desc="Scoring with AV/AR"):
        vec_path = out_dir / f"v_layer_{layer:02d}.pt"
        if not vec_path.exists():
            print(f"Warning: {vec_path} not found, skipping")
            continue

        obj = torch.load(vec_path, map_location="cpu")
        if isinstance(obj, dict) and "vector" in obj:
            vector = obj["vector"].float()
            meta = obj.get("meta", {})
        elif isinstance(obj, torch.Tensor):
            vector = obj.float()
            meta = {}
        else:
            print(f"Warning: unexpected format in {vec_path}, skipping")
            continue

        try:
            explanation = client.generate(vector.numpy(), temperature=0.3, max_new_tokens=200)
            mse, cos = critic.score(explanation, vector.numpy())
        except Exception as exc:
            print(f"Warning: NLA scoring failed for layer {layer}: {exc}")
            explanation = ""
            mse = float("nan")
            cos = float("nan")

        kw_scores = score_explanation_keywords(explanation)

        result = {
            "layer_index": layer,
            "av_explanation": explanation,
            "ar_mse": mse,
            "ar_cos": cos,
            "inter_distance": meta.get("inter_distance", 0.0),
            "pos_intra_var": meta.get("pos_intra_var", 0.0),
            "neg_intra_var": meta.get("neg_intra_var", 0.0),
            "raw_norm": meta.get("raw_norm", 0.0),
            "resid_norm_mean": meta.get("resid_norm_mean", 0.0),
            **kw_scores,
        }
        results.append(result)

    write_json(out_dir / "layer_sweep_report.json", {"layers": results})

    if not results:
        print("No results to report. Check vector files and NLA connectivity.")
        return

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Layer':>6} | {'AR_cos':>8} | {'AR_mse':>8} | {'Keyword':>8} | {'InterDist':>10} | {'PosVar':>8} | {'NegVar':>8} | {'RawNorm':>8}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['layer_index']:>6} | {r['ar_cos']:>8.4f} | {r['ar_mse']:>8.4f} | "
            f"{r['keyword_score']:>8.2f} | {r['inter_distance']:>10.2f} | "
            f"{r['pos_intra_var']:>8.2f} | {r['neg_intra_var']:>8.2f} | {r['raw_norm']:>8.2f}"
        )
    print("=" * 110)

    best = max(results, key=composite_score)
    print(f"\nBest layer by composite score: {best['layer_index']} (score={composite_score(best):.4f})")
    print(f"  AV explanation: {best['av_explanation'][:200]}...")


if __name__ == "__main__":
    main()
