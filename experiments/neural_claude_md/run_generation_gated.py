from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    TEXT_HARNESS_SYSTEM_PROMPT,
    apply_chat,
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
from experiments.neural_claude_md.run_generation import SteeringHook, generate_one


GATED_CONDITIONS = {
    "gated_contrastive",
    "gated_no_guard",
    "always_guarded",
    "random_gated",
    "negative_gated",
}


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_single_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids: list[int] = []
    for word in words:
        for text in (word, f" {word}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
    if not ids:
        raise ValueError(f"No single-token IDs found for {words}")
    return ids


def vector_for_condition(condition: str, contrastive_path: Path, seed: int) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if condition in {"no_harness", "text_harness"}:
        return None, {}
    vector, meta = load_vector(contrastive_path)
    if condition in {"gated_contrastive", "gated_no_guard", "always_guarded"}:
        return normalize(vector), meta
    if condition == "negative_gated":
        return -normalize(vector), {**meta, "negated": True}
    if condition == "random_gated":
        gen = torch.Generator().manual_seed(seed)
        return normalize(torch.randn(vector.numel(), generator=gen)), {**meta, "random_control": True}
    raise ValueError(f"Unknown condition: {condition}")


@dataclass
class GateDecision:
    triggered: bool
    blocked: bool
    reason: str
    alpha: float
    print_rank: int | None
    margin: float
    top_tokens: list[str]


def lexical_guard_allows(prefix_text: str) -> tuple[bool, str]:
    code = prefix_text
    if "```" in code:
        parts = code.split("```")
        code = parts[-1]
        if code.startswith("python"):
            code = code[len("python") :]
        elif code.startswith("py"):
            code = code[len("py") :]

    if code.count('"""') % 2 == 1 or code.count("'''") % 2 == 1:
        return False, "inside_triple_quote"

    line = code.splitlines()[-1] if code.splitlines() else code
    stripped = line.strip()
    if stripped.startswith("#"):
        return False, "inside_comment"

    def odd_unescaped_quote(s: str, quote: str) -> bool:
        count = 0
        escaped = False
        for ch in s:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                count += 1
        return count % 2 == 1

    if odd_unescaped_quote(line, '"') or odd_unescaped_quote(line, "'"):
        return False, "inside_string"

    if not code:
        return True, "empty_prefix"
    last = code[-1]
    if last.isalnum() or last == "_" or last == ".":
        return False, "inside_identifier"
    if stripped == "":
        return True, "line_start"
    if last.isspace() and (code.rstrip().endswith("\n") or not stripped):
        return True, "line_start"
    if code.rstrip().endswith(("\n", ";", ":")):
        return True, "statement_boundary"
    return False, "not_statement_boundary"


def top_token_strings(tokenizer, logits: torch.Tensor, n: int = 5) -> list[str]:
    top = torch.topk(logits.float(), k=n).indices.tolist()
    return [tokenizer.decode([idx]).replace("\n", "\\n") for idx in top]


def decide_gate(
    *,
    tokenizer,
    logits: torch.Tensor,
    prefix_text: str,
    print_ids: list[int],
    target_ids: list[int],
    rank_k: int,
    threshold: float,
    guard_enabled: bool,
    mode: str,
    alpha: float,
    alpha_mode: str,
    alpha_base: float,
    alpha_scale: float,
    alpha_max: float,
) -> GateDecision:
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    sorted_ids = torch.argsort(logprobs, descending=True)
    rank_by_id = {int(tok_id): rank + 1 for rank, tok_id in enumerate(sorted_ids[:200].tolist())}
    print_rank = min((rank_by_id.get(tok_id, 10**9) for tok_id in print_ids), default=10**9)
    print_lp = max(logprobs[tok_id].item() for tok_id in print_ids)
    target_lp = max(logprobs[tok_id].item() for tok_id in target_ids)
    margin = print_lp - target_lp

    raw_trigger = mode == "always" or (print_rank <= rank_k and margin >= threshold)
    if not raw_trigger:
        return GateDecision(False, False, "not_triggered", 0.0, print_rank, margin, top_token_strings(tokenizer, logits))

    if alpha_mode == "fixed":
        step_alpha = alpha
    elif alpha_mode == "adaptive":
        step_alpha = min(alpha_max, max(0.0, alpha_base + alpha_scale * max(margin, 0.0)))
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

    if guard_enabled:
        allowed, reason = lexical_guard_allows(prefix_text)
        if not allowed:
            return GateDecision(False, True, reason, 0.0, print_rank, margin, top_token_strings(tokenizer, logits))
        return GateDecision(True, False, reason, step_alpha, print_rank, margin, top_token_strings(tokenizer, logits))

    return GateDecision(True, False, "no_guard", step_alpha, print_rank, margin, top_token_strings(tokenizer, logits))


@torch.inference_mode()
def generate_gated_one(
    *,
    model,
    tokenizer,
    layer,
    prompt: str,
    vector: torch.Tensor,
    resid_norm: float,
    alpha: float,
    alpha_mode: str,
    alpha_base: float,
    alpha_scale: float,
    alpha_max: float,
    rank_k: int,
    threshold: float,
    max_new_tokens: int,
    print_ids: list[int],
    target_ids: list[int],
    condition: str,
    trace_limit: int,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    inputs = apply_chat(tokenizer, prompt)
    input_ids = inputs["input_ids"].to(model.device)
    past = None
    step_input = input_ids
    generated: list[int] = []
    traces: list[dict[str, Any]] = []
    gate_fire_count = 0
    guard_block_count = 0

    guard_enabled = condition not in {"gated_no_guard"}
    mode = "always" if condition == "always_guarded" else "logit"
    vector = normalize(vector)

    for step in range(max_new_tokens):
        baseline = model(input_ids=step_input, past_key_values=past, use_cache=True)
        baseline_logits = baseline.logits[0, -1, :]
        prefix_text = tokenizer.decode(generated, skip_special_tokens=True)
        decision = decide_gate(
            tokenizer=tokenizer,
            logits=baseline_logits,
            prefix_text=prefix_text,
            print_ids=print_ids,
            target_ids=target_ids,
            rank_k=rank_k,
            threshold=threshold,
            guard_enabled=guard_enabled,
            mode=mode,
            alpha=alpha,
            alpha_mode=alpha_mode,
            alpha_base=alpha_base,
            alpha_scale=alpha_scale,
            alpha_max=alpha_max,
        )

        logits = baseline_logits
        next_past = baseline.past_key_values
        if decision.blocked:
            guard_block_count += 1
        if decision.triggered:
            hook = SteeringHook(layer, vector, decision.alpha, resid_norm)
            try:
                steered = model(input_ids=step_input, past_key_values=past, use_cache=True)
            finally:
                hook.close()
            logits = steered.logits[0, -1, :]
            next_past = steered.past_key_values
            gate_fire_count += 1

        next_token = int(torch.argmax(logits).item())
        generated.append(next_token)
        if len(traces) < trace_limit and (decision.triggered or decision.blocked):
            traces.append(
                {
                    "step": step,
                    "triggered": decision.triggered,
                    "blocked": decision.blocked,
                    "reason": decision.reason,
                    "alpha": decision.alpha,
                    "print_rank": decision.print_rank,
                    "margin": decision.margin,
                    "top_tokens": decision.top_tokens,
                    "chosen_token": tokenizer.decode([next_token]).replace("\n", "\\n"),
                    "prefix_tail": prefix_text[-120:],
                }
            )

        past = next_past
        step_input = torch.tensor([[next_token]], device=model.device)
        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    meta = {
        "generated_tokens": len(generated),
        "gate_fire_count": gate_fire_count,
        "guard_block_count": guard_block_count,
        "gate_fire_rate": gate_fire_count / max(len(generated), 1) * 100.0,
        "guard_block_rate": guard_block_count / max(gate_fire_count + guard_block_count, 1),
    }
    return text, meta, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--eval-prompts", default=str(DEFAULT_OUTPUT_DIR / "data" / "gated_dev_prompts.jsonl"))
    parser.add_argument("--contrastive-vector", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_contrastive.pt"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations_gated_dev.jsonl"))
    parser.add_argument("--trace-output", default=str(DEFAULT_OUTPUT_DIR / "gated" / "traces_dev.jsonl"))
    parser.add_argument("--conditions", default="gated_contrastive")
    parser.add_argument("--alphas", default="0.15,0.2,0.25")
    parser.add_argument("--rank-ks", default="10")
    parser.add_argument("--thresholds", default="0")
    parser.add_argument("--alpha-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--alpha-base", type=float, default=0.05)
    parser.add_argument("--alpha-scale", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.2)
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--resid-norm", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-limit", type=int, default=40)
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = read_jsonl(args.eval_prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)
    layer = layers[args.layer_index]

    print_ids = resolve_single_token_ids(tokenizer, ["print"])
    target_ids = resolve_single_token_ids(tokenizer, ["logger", "logging"])
    alphas = parse_float_list(args.alphas)
    rank_ks = parse_int_list(args.rank_ks)
    thresholds = parse_float_list(args.thresholds)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]

    rows = []
    trace_rows = []
    contrastive_path = Path(args.contrastive_vector)

    for condition in conditions:
        vector, vector_meta = vector_for_condition(condition, contrastive_path, args.seed)
        resid_norm = args.resid_norm or float(vector_meta.get("resid_norm_mean", 1.0))
        if condition in {"no_harness", "text_harness"}:
            config_grid = [(0.0, 0, 0.0)]
        else:
            config_grid = [(alpha, rank_k, threshold) for alpha in alphas for rank_k in rank_ks for threshold in thresholds]

        for alpha, rank_k, threshold in config_grid:
            desc = f"{condition} a={alpha} k={rank_k} t={threshold}"
            for prompt_row in tqdm(prompts, desc=desc):
                started = time.time()
                if condition == "no_harness":
                    text = generate_one(model, tokenizer, prompt_row["prompt"], None, args.max_new_tokens, 0.0)
                    gate_meta = {"generated_tokens": None, "gate_fire_count": 0, "guard_block_count": 0}
                    traces = []
                elif condition == "text_harness":
                    system_prompt = args.system_prompt if args.system_prompt is not None else TEXT_HARNESS_SYSTEM_PROMPT
                    text = generate_one(model, tokenizer, prompt_row["prompt"], system_prompt, args.max_new_tokens, 0.0)
                    gate_meta = {"generated_tokens": None, "gate_fire_count": 0, "guard_block_count": 0}
                    traces = []
                else:
                    assert vector is not None
                    text, gate_meta, traces = generate_gated_one(
                        model=model,
                        tokenizer=tokenizer,
                        layer=layer,
                        prompt=prompt_row["prompt"],
                        vector=vector,
                        resid_norm=resid_norm,
                        alpha=alpha,
                        alpha_mode=args.alpha_mode,
                        alpha_base=args.alpha_base,
                        alpha_scale=args.alpha_scale,
                        alpha_max=args.alpha_max,
                        rank_k=rank_k,
                        threshold=threshold,
                        max_new_tokens=args.max_new_tokens,
                        print_ids=print_ids,
                        target_ids=target_ids,
                        condition=condition,
                        trace_limit=args.trace_limit,
                    )

                row = {
                    "condition": condition,
                    "alpha": alpha,
                    "alpha_mode": args.alpha_mode,
                    "alpha_base": args.alpha_base,
                    "alpha_scale": args.alpha_scale,
                    "alpha_max": args.alpha_max,
                    "rank_k": rank_k,
                    "threshold": threshold,
                    "layer_index": args.layer_index,
                    "resid_norm": resid_norm,
                    "prompt_id": prompt_row["id"],
                    "split": prompt_row["split"],
                    "task_id": prompt_row.get("task_id"),
                    "variant_id": prompt_row.get("variant_id"),
                    "prompt": prompt_row["prompt"],
                    "generation": text,
                    "latency_s": time.time() - started,
                    **gate_meta,
                }
                rows.append(row)
                for trace in traces:
                    trace_rows.append(
                        {
                            "condition": condition,
                            "alpha": alpha,
                            "alpha_mode": args.alpha_mode,
                            "alpha_base": args.alpha_base,
                            "alpha_scale": args.alpha_scale,
                            "alpha_max": args.alpha_max,
                            "rank_k": rank_k,
                            "threshold": threshold,
                            "prompt_id": prompt_row["id"],
                            **trace,
                        }
                    )

    write_jsonl(args.output, rows)
    write_jsonl(args.trace_output, trace_rows)
    print(f"Saved {len(rows)} generations to {args.output}")
    print(f"Saved {len(trace_rows)} trace rows to {args.trace_output}")


if __name__ == "__main__":
    main()
