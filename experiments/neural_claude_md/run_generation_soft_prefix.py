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
    load_causal_lm,
    load_tokenizer,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from experiments.neural_claude_md.train_soft_prefix_preference import scaled_random_prefix


PREFIX_CONDITIONS = {"learned_soft_prefix", "random_soft_prefix", "zero_soft_prefix"}


def load_prefix(path: str | Path) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "prefix" not in obj:
        raise ValueError(f"Unsupported soft prefix file: {path}")
    return obj["prefix"].float(), obj.get("meta", {})


def prefix_for_condition(condition: str, learned_prefix: torch.Tensor, *, seed: int) -> torch.Tensor | None:
    if condition in {"no_harness", "visible_text_harness"}:
        return None
    if condition == "learned_soft_prefix":
        return learned_prefix
    if condition == "random_soft_prefix":
        return scaled_random_prefix(learned_prefix, seed + 997)
    if condition == "zero_soft_prefix":
        return torch.zeros_like(learned_prefix)
    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    prefix: torch.Tensor | None,
    system_prompt: str | None,
    max_new_tokens: int,
) -> str:
    inputs = apply_chat(tokenizer, prompt, system_prompt=system_prompt)
    prompt_ids = inputs["input_ids"].to(model.device)
    embed_layer = model.get_input_embeddings()
    token_embeds = embed_layer(prompt_ids)
    if prefix is not None:
        prefix_embeds = prefix.to(device=token_embeds.device, dtype=token_embeds.dtype).unsqueeze(0)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    else:
        inputs_embeds = token_embeds

    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
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
        "--soft-prefix",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "prefixes" / "c1_soft_prefix_len8_e5.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "generations" / "c1b_generations.jsonl"),
    )
    parser.add_argument(
        "--conditions",
        default="no_harness,visible_text_harness,learned_soft_prefix,random_soft_prefix,zero_soft_prefix",
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

    learned_prefix, prefix_meta = load_prefix(args.soft_prefix)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)

    rows = []
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    for condition in conditions:
        prefix = prefix_for_condition(condition, learned_prefix, seed=args.seed)
        system_prompt = SAFETY_TEXT_HARNESS_SYSTEM_PROMPT if condition == "visible_text_harness" else None
        for prompt_row in tqdm(prompts, desc=condition):
            started = time.time()
            text = generate_one(
                model,
                tokenizer,
                prompt_row["prompt"],
                prefix=prefix,
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
                    "soft_prefix": str(Path(args.soft_prefix)) if condition in PREFIX_CONDITIONS else None,
                    "prefix_len": prefix_meta.get("prefix_len") if condition in PREFIX_CONDITIONS else 0,
                    "prompt": prompt_row["prompt"],
                    "generation": text,
                    "latency_s": time.time() - started,
                }
            )
            if args.save_every > 0 and len(rows) % args.save_every == 0:
                write_jsonl(args.output, rows)

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} generations to {args.output}")


if __name__ == "__main__":
    main()
