from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    SAFETY_TEXT_HARNESS_SYSTEM_PROMPT,
    env_default,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    load_vector,
    normalize,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from experiments.neural_claude_md.run_generation import SteeringHook, generate_one, parse_float_list


INTERNAL_CONDITIONS = {"internal_harness", "random_internal_control", "negative_internal_control"}


def vector_for_condition(condition: str, vector_path: Path, seed: int) -> tuple[torch.Tensor | None, dict]:
    if condition in {"no_harness", "visible_text_harness"}:
        return None, {}
    v, meta = load_vector(vector_path)
    if condition == "internal_harness":
        return v, meta
    if condition == "negative_internal_control":
        return -v, {**meta, "negated": True}
    if condition == "random_internal_control":
        gen = torch.Generator().manual_seed(seed)
        return normalize(torch.randn(v.numel(), generator=gen)), {**meta, "random_control": True}
    raise ValueError(f"Unknown condition: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument(
        "--eval-prompts",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "data" / "dev_prompts.jsonl"),
    )
    parser.add_argument(
        "--system-delta-vector",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "vectors" / "v_system_delta_safety_l20_last_token.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "generations" / "dev_generations.jsonl"),
    )
    parser.add_argument(
        "--conditions",
        default="no_harness,visible_text_harness,internal_harness,random_internal_control,negative_internal_control",
    )
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--resid-norm", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = read_jsonl(args.eval_prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("No evaluation prompts loaded")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(f"--layer-index {args.layer_index} out of range for {len(layers)} layers")
    layer = layers[args.layer_index]

    rows = []
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    alphas = parse_float_list(args.alphas)
    vector_path = Path(args.system_delta_vector)

    for condition in conditions:
        vector, vector_meta = vector_for_condition(condition, vector_path, args.seed)
        resid_norm = args.resid_norm or float(vector_meta.get("resid_norm_mean", 1.0))
        condition_alphas = alphas if condition in INTERNAL_CONDITIONS else [0.0]
        for alpha in condition_alphas:
            hook = None
            if vector is not None:
                hook = SteeringHook(layer, normalize(vector), alpha, resid_norm)
            try:
                for prompt_row in tqdm(prompts, desc=f"{condition} alpha={alpha}"):
                    system_prompt = SAFETY_TEXT_HARNESS_SYSTEM_PROMPT if condition == "visible_text_harness" else None
                    started = time.time()
                    text = generate_one(
                        model,
                        tokenizer,
                        prompt_row["prompt"],
                        system_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "alpha": alpha,
                            "layer_index": args.layer_index,
                            "resid_norm": resid_norm,
                            "vector_path": str(vector_path) if vector is not None else None,
                            "vector_kind": vector_meta.get("kind"),
                            "vector_pooling": vector_meta.get("pooling"),
                            "prompt_id": prompt_row["id"],
                            "split": prompt_row["split"],
                            "task_id": prompt_row.get("task_id"),
                            "variant_id": prompt_row.get("variant_id"),
                            "base_index": prompt_row.get("base_index"),
                            "prompt": prompt_row["prompt"],
                            "generation": text,
                            "latency_s": time.time() - started,
                        }
                    )
                    if args.save_every > 0 and len(rows) % args.save_every == 0:
                        write_jsonl(args.output, rows)
            finally:
                if hook is not None:
                    hook.close()

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} generations to {args.output}")


if __name__ == "__main__":
    main()
