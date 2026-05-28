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
    apply_chat,
    env_default,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from experiments.neural_claude_md.train_residual_controller_preference import (
    CONTROLLER_CONDITIONS,
    ResidualControllerHooks,
    scaled_random_controller,
)


def load_controller(path: str | Path) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "vectors" not in obj:
        raise ValueError(f"Unsupported residual controller file: {path}")
    return obj["vectors"].float(), obj.get("meta", {})


def controller_for_condition(condition: str, learned_vectors: torch.Tensor, *, seed: int) -> torch.Tensor | None:
    if condition in {"no_harness", "visible_text_harness"}:
        return None
    if condition == "learned_residual_controller":
        return learned_vectors
    if condition == "random_residual_controller":
        return scaled_random_controller(learned_vectors, seed + 997)
    if condition == "zero_residual_controller":
        return torch.zeros_like(learned_vectors)
    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    system_prompt: str | None,
    max_new_tokens: int,
) -> str:
    inputs = apply_chat(tokenizer, prompt, system_prompt=system_prompt)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs, use_cache=True)
    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :]

    generated = []
    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        out = model(input_ids=next_token.to(model.device), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument(
        "--eval-prompts",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "data_compact" / "test_prompts.jsonl"),
    )
    parser.add_argument(
        "--controller",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "controllers" / "c5_residual_controller.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "generations" / "c5_residual_generations.jsonl"),
    )
    parser.add_argument(
        "--conditions",
        default=(
            "no_harness,visible_text_harness,learned_residual_controller,"
            "random_residual_controller,zero_residual_controller"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = read_jsonl(args.eval_prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("No evaluation prompts loaded")

    learned_vectors, controller_meta = load_controller(args.controller)
    layer_indices = [int(x) for x in controller_meta["layer_indices"]]
    scale = float(controller_meta.get("scale", 1.0))
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)

    rows = []
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    for condition in conditions:
        vectors = controller_for_condition(condition, learned_vectors, seed=args.seed)
        system_prompt = SAFETY_TEXT_HARNESS_SYSTEM_PROMPT if condition == "visible_text_harness" else None
        hooks = None
        if vectors is not None:
            hooks = ResidualControllerHooks(layers, layer_indices, vectors, scale=scale)
        try:
            for prompt_row in tqdm(prompts, desc=condition):
                started = time.time()
                text = generate_one(
                    model,
                    tokenizer,
                    prompt_row["prompt"],
                    system_prompt=system_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append(
                    {
                        "condition": condition,
                        "alpha": 0.0,
                        "prompt_id": prompt_row["id"],
                        "split": prompt_row["split"],
                        "task_id": prompt_row.get("task_id"),
                        "variant_id": prompt_row.get("variant_id"),
                        "base_index": prompt_row.get("base_index"),
                        "controller": str(Path(args.controller)) if condition in CONTROLLER_CONDITIONS else None,
                        "layer_indices": ",".join(str(x) for x in layer_indices) if condition in CONTROLLER_CONDITIONS else "",
                        "prompt": prompt_row["prompt"],
                        "generation": text,
                        "latency_s": time.time() - started,
                    }
                )
                if args.save_every > 0 and len(rows) % args.save_every == 0:
                    write_jsonl(args.output, rows)
        finally:
            if hooks is not None:
                hooks.close()

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} generations to {args.output}")


if __name__ == "__main__":
    main()
