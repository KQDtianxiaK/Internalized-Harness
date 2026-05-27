from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.neural_claude_md.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_NLA_AR,
    DEFAULT_NLA_AV,
    DEFAULT_NLA_REPO,
    DEFAULT_OUTPUT_DIR,
    import_nla,
    load_causal_lm,
    load_tokenizer,
    marker_span_from_offsets,
    normalize,
    read_jsonl,
    write_json,
)


@torch.inference_mode()
def span_h(model, tokenizer, code: str, marker: str, layer_index: int) -> torch.Tensor:
    input_ids, marker_indices = marker_span_from_offsets(tokenizer, code, marker)
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_index][0, marker_indices, :].float().mean(dim=0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--pairs", default=str(DEFAULT_OUTPUT_DIR / "data" / "contrastive_pairs.jsonl"))
    parser.add_argument("--nla-repo", default=str(DEFAULT_NLA_REPO))
    parser.add_argument("--av", default=str(DEFAULT_NLA_AV))
    parser.add_argument("--ar", default=str(DEFAULT_NLA_AR))
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "layer_alignment.json"))
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 21])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--model-dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    rows = read_jsonl(args.pairs)[: args.limit]
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=args.model_dtype, device_map=args.device_map)
    nla = import_nla(args.nla_repo)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    client = nla.NLAClient(args.av, sglang_url=args.sglang_url)
    critic = nla.NLACritic(args.ar, device=args.device, dtype=dtype)

    results = []
    for layer in args.layers:
        deltas = []
        for row in tqdm(rows, desc=f"layer {layer}"):
            pos_h = span_h(model, tokenizer, row["positive"], row["positive_marker"], layer)
            neg_h = span_h(model, tokenizer, row["negative"], row["negative_marker"], layer)
            deltas.append(pos_h - neg_h)
        direction = normalize(torch.stack(deltas).mean(dim=0))
        explanation = client.generate(direction.numpy(), temperature=0.3, max_new_tokens=200)
        mse, cos = critic.score(explanation, direction.numpy())
        results.append(
            {
                "layer_index": layer,
                "num_pairs": len(rows),
                "av_explanation": explanation,
                "ar_mse": mse,
                "ar_cos": cos,
            }
        )

    write_json(args.output, {"candidates": results})
    print(f"Saved layer alignment report to {args.output}")


if __name__ == "__main__":
    main()

