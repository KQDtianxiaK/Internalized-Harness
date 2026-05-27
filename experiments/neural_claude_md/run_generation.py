from __future__ import annotations

import argparse
import time
from pathlib import Path

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


NEURAL_CONDITIONS = {"contrastive_neural", "nla_neural", "random_vector", "negative_contrastive"}


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def vector_for_condition(condition: str, contrastive_path: Path, nla_path: Path, seed: int) -> tuple[torch.Tensor | None, dict]:
    if condition in {"no_harness", "text_harness"}:
        return None, {}
    if condition == "contrastive_neural":
        return load_vector(contrastive_path)
    if condition == "nla_neural":
        return load_vector(nla_path)
    if condition == "negative_contrastive":
        v, meta = load_vector(contrastive_path)
        return -v, {**meta, "negated": True}
    if condition == "random_vector":
        v, meta = load_vector(contrastive_path)
        gen = torch.Generator().manual_seed(seed)
        return normalize(torch.randn(v.numel(), generator=gen)), {**meta, "random_control": True}
    raise ValueError(f"Unknown condition: {condition}")


class SteeringHook:
    def __init__(self, layer, vector: torch.Tensor, alpha: float, resid_norm: float):
        self.vector = vector.float()
        self.alpha = float(alpha)
        self.resid_norm = float(resid_norm)
        # Use pre-hook so we steer the *input* to the layer, matching where
        # hidden_states[layer_index] is extracted in contrastive scripts.
        self.handle = layer.register_forward_pre_hook(self._pre_hook)

    def close(self) -> None:
        self.handle.remove()

    def _steer_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0:
            return h
        steered = h.clone()
        v = self.vector.to(device=steered.device, dtype=steered.dtype)
        steered[:, -1, :] = steered[:, -1, :] + self.alpha * self.resid_norm * v
        return steered

    def _pre_hook(self, module, inputs):
        if isinstance(inputs, tuple):
            return (self._steer_hidden(inputs[0]),) + inputs[1:]
        return (self._steer_hidden(inputs),)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, system_prompt: str | None, max_new_tokens: int, temperature: float) -> str:
    inputs = apply_chat(tokenizer, prompt, system_prompt=system_prompt)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    sampling = temperature > 0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": sampling,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sampling:
        generation_kwargs["temperature"] = temperature
    out = model.generate(**generation_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--eval-prompts", default=str(DEFAULT_OUTPUT_DIR / "data" / "eval_prompts.jsonl"))
    parser.add_argument("--contrastive-vector", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_contrastive.pt"))
    parser.add_argument("--nla-vector", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_nla.pt"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations.jsonl"))
    parser.add_argument(
        "--conditions",
        default="no_harness,text_harness,contrastive_neural,nla_neural,random_vector,negative_contrastive",
    )
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--layer-index", type=int, default=20)
    parser.add_argument("--resid-norm", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--system-prompt", default=None, help="Override the default text_harness system prompt")
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = read_jsonl(args.eval_prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    layers = get_decoder_layers(model)
    layer = layers[args.layer_index]

    rows = []
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    alphas = parse_float_list(args.alphas)
    contrastive_path = Path(args.contrastive_vector)
    nla_path = Path(args.nla_vector)

    for condition in conditions:
        vector, vector_meta = vector_for_condition(condition, contrastive_path, nla_path, args.seed)
        resid_norm = args.resid_norm or float(vector_meta.get("resid_norm_mean", 1.0))
        condition_alphas = alphas if condition in NEURAL_CONDITIONS else [0.0]
        for alpha in condition_alphas:
            hook = None
            if vector is not None:
                hook = SteeringHook(layer, normalize(vector), alpha, resid_norm)
            try:
                for prompt_row in tqdm(prompts, desc=f"{condition} alpha={alpha}"):
                    if condition == "text_harness":
                        system_prompt = args.system_prompt if args.system_prompt is not None else TEXT_HARNESS_SYSTEM_PROMPT
                    else:
                        system_prompt = None
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
                            "prompt_id": prompt_row["id"],
                            "split": prompt_row["split"],
                            "task_id": prompt_row.get("task_id"),
                            "prompt": prompt_row["prompt"],
                            "generation": text,
                            "latency_s": time.time() - started,
                        }
                    )
            finally:
                if hook is not None:
                    hook.close()

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} generations to {args.output}")


if __name__ == "__main__":
    main()
