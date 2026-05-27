from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    apply_chat,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    load_vector,
    normalize,
    read_jsonl,
    set_seed,
    write_json,
)


@torch.inference_mode()
def logprobs_at_last_position(model, input_ids) -> torch.Tensor:
    """Return log-prob distribution at the final token position."""
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1, :]
    return torch.log_softmax(logits.float(), dim=-1)


@torch.inference_mode()
def greedy_prefix(model, input_ids, num_tokens: int) -> torch.Tensor:
    """Greedy-decode a short prefix of num_tokens."""
    for _ in range(num_tokens):
        out = model(input_ids=input_ids, use_cache=False)
        next_token = out.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return input_ids


def _collect_deltas(
    baseline_lp: torch.Tensor,
    steered_lp: torch.Tensor,
    ids: list[int],
    name: str,
    tokenizer,
) -> dict:
    results = {}
    for tok_id in ids:
        tok_label = tokenizer.decode([tok_id]).strip().replace(" ", "_")
        key_base = f"{name}_{tok_label}_lp"
        results[f"baseline_{key_base}"] = baseline_lp[tok_id].item()
        results[f"steered_{key_base}"] = steered_lp[tok_id].item()
        results[f"delta_{key_base}"] = results[f"steered_{key_base}"] - results[f"baseline_{key_base}"]
    return results


def probe_logprob_from_baseline(
    model,
    tokenizer,
    input_ids,
    layer,
    vector: torch.Tensor,
    alpha: float,
    resid_norm: float,
    target_ids: list[int],
    avoid_ids: list[int],
    baseline_lp: torch.Tensor,
) -> dict[str, float]:
    from experiments.neural_claude_md.run_generation import SteeringHook

    hook = SteeringHook(layer, vector, alpha, resid_norm)
    try:
        steered_lp = logprobs_at_last_position(model, input_ids)
    finally:
        hook.close()

    results = {}
    results.update(_collect_deltas(baseline_lp, steered_lp, target_ids, "target", tokenizer))
    results.update(_collect_deltas(baseline_lp, steered_lp, avoid_ids, "avoid", tokenizer))

    target_keys = [k for k in results if k.startswith("delta_target_")]
    avoid_keys = [k for k in results if k.startswith("delta_avoid_")]
    results["target_mean_delta"] = sum(results[k] for k in target_keys) / len(target_keys) if target_keys else 0.0
    results["avoid_mean_delta"] = sum(results[k] for k in avoid_keys) / len(avoid_keys) if avoid_keys else 0.0
    results["contrast_delta"] = results["target_mean_delta"] - results["avoid_mean_delta"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--eval-prompts", default=str(DEFAULT_OUTPUT_DIR / "data" / "eval_prompts.jsonl"))
    parser.add_argument("--contrastive-vector", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_contrastive.pt"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "causal_probe.json"))
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--alphas", default="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.5")
    parser.add_argument("--prefix-len", type=int, default=8, help="Number of greedy tokens to generate before probing")
    parser.add_argument("--target-tokens", default="logger,logging")
    parser.add_argument("--avoid-tokens", default="print")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)
    layer = layers[args.layer_index]

    vector, vector_meta = load_vector(args.contrastive_vector)
    vector = normalize(vector)
    resid_norm = float(vector_meta.get("resid_norm_mean", 1.0))

    def resolve_token(tok: str) -> int:
        enc = tokenizer.encode(f" {tok}", add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"Token '{tok}' maps to {len(enc)} subword IDs ({enc}) — "
                f"causal_probe requires single-token words."
            )
        return enc[0]

    target_ids = [resolve_token(t) for t in args.target_tokens.split(",")]
    avoid_ids = [resolve_token(t) for t in args.avoid_tokens.split(",")]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    prompts = read_jsonl(args.eval_prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    all_results = []
    for row in tqdm(prompts, desc="Probing"):
        inputs = apply_chat(tokenizer, row["prompt"], system_prompt=args.system_prompt)
        input_ids = inputs["input_ids"].to(model.device)
        prefix_ids = greedy_prefix(model, input_ids, args.prefix_len)

        # Baseline computed once per prefix
        baseline_lp = logprobs_at_last_position(model, prefix_ids)

        prompt_results = {"prompt_id": row["id"], "split": row.get("split"), "alphas": []}
        for alpha in alphas:
            probe = probe_logprob_from_baseline(
                model, tokenizer, prefix_ids, layer, vector, alpha, resid_norm, target_ids, avoid_ids, baseline_lp
            )
            prompt_results["alphas"].append({"alpha": alpha, **probe})
        all_results.append(prompt_results)

    write_json(args.output, {"probes": all_results, "meta": {
        "model": str(args.model),
        "layer_index": args.layer_index,
        "vector": str(args.contrastive_vector),
        "target_tokens": args.target_tokens,
        "avoid_tokens": args.avoid_tokens,
        "prefix_len": args.prefix_len,
    }})
    print(f"Saved causal probe results to {args.output}")

    # Print quick summary
    print("\n" + "=" * 70)
    print(f"{'Alpha':>8} | {'Target Δ':>12} | {'Avoid Δ':>12} | {'Contrast Δ':>14}")
    print("-" * 70)
    for alpha in alphas:
        target_deltas = []
        avoid_deltas = []
        for r in all_results:
            for a in r["alphas"]:
                if a["alpha"] == alpha:
                    target_deltas.append(a["target_mean_delta"])
                    avoid_deltas.append(a["avoid_mean_delta"])
        if target_deltas:
            print(
                f"{alpha:>8.2f} | {sum(target_deltas)/len(target_deltas):>12.4f} | "
                f"{sum(avoid_deltas)/len(avoid_deltas):>12.4f} | "
                f"{sum(t-a for t,a in zip(target_deltas, avoid_deltas))/len(target_deltas):>14.4f}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
