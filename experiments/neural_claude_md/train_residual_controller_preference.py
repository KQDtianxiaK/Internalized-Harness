from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    SAFETY_TEXT_HARNESS_SYSTEM_PROMPT,
    env_default,
    ensure_dir,
    get_decoder_layers,
    load_causal_lm,
    load_tokenizer,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from experiments.neural_claude_md.train_soft_prefix_preference import pair_logprobs_for_row


CONTROLLER_CONDITIONS = {"learned_residual_controller", "random_residual_controller", "zero_residual_controller"}


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


class ResidualControllerHooks:
    def __init__(
        self,
        layers,
        layer_indices: list[int],
        vectors: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        self.layer_indices = layer_indices
        self.vectors = vectors
        self.scale = float(scale)
        self.handles = []
        for slot, layer_index in enumerate(layer_indices):
            self.handles.append(layers[layer_index].register_forward_pre_hook(self._make_hook(slot)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, slot: int):
        def hook(module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            vector = self.vectors[slot].to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            steered = hidden + self.scale * vector
            if isinstance(inputs, tuple):
                return (steered,) + inputs[1:]
            return (steered,)

        return hook


def init_controller(model, layer_count: int, *, init: str, seed: int) -> torch.nn.Parameter:
    hidden_size = model.config.hidden_size
    embed_weight = model.get_input_embeddings().weight.detach().float()
    gen = torch.Generator(device=embed_weight.device).manual_seed(seed)
    if init == "zeros":
        data = torch.zeros((layer_count, hidden_size), device=embed_weight.device)
    elif init == "random":
        std = float(embed_weight.std().item()) * 0.05
        data = torch.randn((layer_count, hidden_size), generator=gen, device=embed_weight.device) * std
    else:
        raise ValueError(f"Unsupported init: {init}")
    return torch.nn.Parameter(data.float())


def scaled_random_controller(reference: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=reference.device).manual_seed(seed)
    random = torch.randn(reference.shape, generator=gen, device=reference.device).float()
    ref_norm = reference.detach().float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    random_norm = random.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return random / random_norm * ref_norm


def evaluate(
    model,
    tokenizer,
    layers,
    rows: list[dict],
    *,
    learned_vectors: torch.Tensor,
    layer_indices: list[int],
    scale: float,
    aggregation_temperature: float,
    seed: int,
) -> tuple[list[dict], pd.DataFrame]:
    random_vectors = scaled_random_controller(learned_vectors, seed + 997)
    zero_vectors = torch.zeros_like(learned_vectors)
    conditions: list[tuple[str, torch.Tensor | None, str | None]] = [
        ("no_harness", None, None),
        ("visible_text_harness", None, SAFETY_TEXT_HARNESS_SYSTEM_PROMPT),
        ("learned_residual_controller", learned_vectors.detach(), None),
        ("random_residual_controller", random_vectors.detach(), None),
        ("zero_residual_controller", zero_vectors.detach(), None),
    ]

    out_rows = []
    for condition, vectors, system_prompt in conditions:
        hooks = None
        if vectors is not None:
            hooks = ResidualControllerHooks(layers, layer_indices, vectors, scale=scale)
        try:
            for row in tqdm(rows, desc=f"eval {condition}"):
                started = time.time()
                safe_logprob, unsafe_logprob, safe_tokens, unsafe_tokens = pair_logprobs_for_row(
                    model,
                    tokenizer,
                    row,
                    prefix=None,
                    system_prompt=system_prompt,
                    completion_style="verifier_bank",
                    requires_grad=False,
                    aggregation_temperature=aggregation_temperature,
                )
                margin = safe_logprob - unsafe_logprob
                margin_value = float(margin.item())
                out_rows.append(
                    {
                        "condition": condition,
                        "split": row["split"],
                        "prompt_id": row["id"],
                        "task_id": row.get("task_id"),
                        "variant_id": row.get("variant_id"),
                        "base_index": row.get("base_index"),
                        "layer_indices": ",".join(str(x) for x in layer_indices),
                        "scale": scale,
                        "safe_mean_logprob": float(safe_logprob.item()),
                        "unsafe_mean_logprob": float(unsafe_logprob.item()),
                        "margin": margin_value,
                        "prefers_safe": margin_value > 0,
                        "safe_tokens": safe_tokens,
                        "unsafe_tokens": unsafe_tokens,
                        "latency_s": time.time() - started,
                    }
                )
        finally:
            if hooks is not None:
                hooks.close()

    df = pd.DataFrame(out_rows)
    summary = (
        df.groupby(["condition", "split"], dropna=False)
        .agg(
            n=("prompt_id", "count"),
            prefers_safe_rate=("prefers_safe", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_latency_s=("latency_s", "mean"),
        )
        .reset_index()
    )
    return out_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default("NCM_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument(
        "--train-prompts",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "data_compact" / "dev_prompts.jsonl"),
    )
    parser.add_argument(
        "--eval-prompts",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_a" / "data_compact" / "test_prompts.jsonl"),
    )
    parser.add_argument(
        "--controller-output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "controllers" / "c5_residual_controller.pt"),
    )
    parser.add_argument(
        "--eval-output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "scores" / "c5_residual_eval_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_OUTPUT_DIR / "internal_harness_c" / "scores" / "c5_residual_eval_summary.csv"),
    )
    parser.add_argument("--layer-indices", default="16,20,24")
    parser.add_argument("--init", choices=["zeros", "random"], default="zeros")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--aggregation-temperature", type=float, default=0.5)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    layer_indices = parse_int_list(args.layer_indices)
    train_rows = read_jsonl(args.train_prompts)
    eval_rows = read_jsonl(args.eval_prompts)
    if not train_rows or not eval_rows:
        raise ValueError("Train and eval prompts must be non-empty")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)
    for param in model.parameters():
        param.requires_grad_(False)
    layers = get_decoder_layers(model)
    for layer_index in layer_indices:
        if not (0 <= layer_index < len(layers)):
            raise ValueError(f"Layer index {layer_index} out of range for {len(layers)} layers")

    vectors = init_controller(model, len(layer_indices), init=args.init, seed=args.seed)
    optimizer = torch.optim.AdamW([vectors], lr=args.lr)

    history = []
    hooks = ResidualControllerHooks(layers, layer_indices, vectors, scale=args.scale)
    try:
        for epoch in range(args.epochs):
            total_loss = 0.0
            total_margin = 0.0
            total_safe_nll = 0.0
            total_l2 = 0.0
            for row in tqdm(train_rows, desc=f"train epoch {epoch + 1}/{args.epochs}"):
                safe_logprob, unsafe_logprob, _, _ = pair_logprobs_for_row(
                    model,
                    tokenizer,
                    row,
                    prefix=None,
                    system_prompt=None,
                    completion_style="verifier_bank",
                    requires_grad=True,
                    aggregation_temperature=args.aggregation_temperature,
                )
                margin = safe_logprob - unsafe_logprob
                safe_nll = -safe_logprob
                l2_penalty = vectors.float().pow(2).mean()
                loss = safe_nll + args.beta * F.softplus(-margin) + args.l2 * l2_penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                total_margin += float(margin.detach().item())
                total_safe_nll += float(safe_nll.detach().item())
                total_l2 += float(l2_penalty.detach().item())
            vector_norm = float(vectors.detach().float().norm(dim=1).mean().item())
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": total_loss / len(train_rows),
                    "mean_margin": total_margin / len(train_rows),
                    "mean_safe_nll": total_safe_nll / len(train_rows),
                    "mean_l2": total_l2 / len(train_rows),
                    "mean_vector_norm": vector_norm,
                }
            )
            print(
                f"epoch={epoch + 1} mean_loss={history[-1]['mean_loss']:.4f} "
                f"mean_margin={history[-1]['mean_margin']:.4f} "
                f"mean_safe_nll={history[-1]['mean_safe_nll']:.4f} "
                f"mean_vector_norm={vector_norm:.4f}"
            )
    finally:
        hooks.close()

    controller_path = Path(args.controller_output)
    ensure_dir(controller_path.parent)
    torch.save(
        {
            "vectors": vectors.detach().cpu().float(),
            "meta": {
                "kind": "learned_residual_controller",
                "layer_indices": layer_indices,
                "init": args.init,
                "epochs": args.epochs,
                "lr": args.lr,
                "scale": args.scale,
                "beta": args.beta,
                "l2": args.l2,
                "aggregation_temperature": args.aggregation_temperature,
                "train_prompts": str(Path(args.train_prompts)),
                "eval_prompts": str(Path(args.eval_prompts)),
                "history": history,
            },
        },
        controller_path,
    )

    rows, summary = evaluate(
        model,
        tokenizer,
        layers,
        eval_rows,
        learned_vectors=vectors,
        layer_indices=layer_indices,
        scale=args.scale,
        aggregation_temperature=args.aggregation_temperature,
        seed=args.seed,
    )
    write_jsonl(args.eval_output, rows)
    summary_path = Path(args.summary_output)
    ensure_dir(summary_path.parent)
    summary.to_csv(summary_path, index=False)
    print(f"Saved residual controller to {controller_path}")
    print(f"Saved eval rows to {args.eval_output}")
    print(f"Saved eval summary to {summary_path}")


if __name__ == "__main__":
    main()
