from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.neural_claude_md.common import (
    DEFAULT_NLA_AR,
    DEFAULT_NLA_AV,
    DEFAULT_NLA_REPO,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RULE_PHRASES,
    import_nla,
    load_vector,
    normalize,
    save_vector,
    write_json,
)


def generate_explanation(client, vector: torch.Tensor, name: str, temperature: float, max_new_tokens: int) -> dict:
    text = client.generate(
        vector.float().cpu().numpy(),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return {"name": name, "explanation": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nla-repo", default=str(DEFAULT_NLA_REPO))
    parser.add_argument("--ar", default=str(DEFAULT_NLA_AR))
    parser.add_argument("--av", default=str(DEFAULT_NLA_AV))
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--contrastive-vector", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_contrastive.pt"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "vectors" / "v_nla.pt"))
    parser.add_argument("--explanations-output", default=str(DEFAULT_OUTPUT_DIR / "nla_explanations.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--rule", action="append", default=None)
    args = parser.parse_args()

    nla = import_nla(args.nla_repo)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    critic = nla.NLACritic(args.ar, device=args.device, dtype=dtype)
    phrases = args.rule or DEFAULT_RULE_PHRASES

    reconstructions = []
    for phrase in phrases:
        reconstructions.append(normalize(critic.reconstruct(phrase)))
    v_nla = normalize(torch.stack(reconstructions).mean(dim=0))
    save_vector(
        args.output,
        v_nla,
        {
            "kind": "nla_ar_direction",
            "num_rule_phrases": len(phrases),
            "rule_phrases": phrases,
            "ar": str(Path(args.ar)),
        },
    )

    client = nla.NLAClient(args.av, sglang_url=args.sglang_url)
    explanations = {
        "rule_phrases": phrases,
        "nla_vector": generate_explanation(client, v_nla, "v_nla", args.temperature, args.max_new_tokens),
    }
    contrastive_path = Path(args.contrastive_vector)
    if contrastive_path.exists():
        v_contrastive, contrastive_meta = load_vector(contrastive_path)
        cos = torch.nn.functional.cosine_similarity(v_contrastive.view(1, -1), v_nla.view(1, -1)).item()
        explanations["contrastive_vector"] = generate_explanation(
            client, v_contrastive, "v_contrastive", args.temperature, args.max_new_tokens
        )
        explanations["cos_v_contrastive_v_nla"] = cos
        explanations["contrastive_meta"] = contrastive_meta
    write_json(args.explanations_output, explanations)
    print(f"Saved NLA direction to {args.output}")
    print(f"Saved NLA explanations to {args.explanations_output}")


if __name__ == "__main__":
    main()

