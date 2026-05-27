from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR, evaluate_python_output, read_jsonl, write_json, write_jsonl


DEFAULT_PRIORITY = [
    "text_harness",
    "no_harness",
    "gated_contrastive",
    "gated_no_guard",
    "always_guarded",
    "random_gated",
    "negative_gated",
]


def token_count(row: dict[str, Any]) -> int:
    if row.get("generated_tokens") is not None:
        try:
            return int(row["generated_tokens"])
        except (TypeError, ValueError):
            pass
    return len(str(row.get("generation", "")).split())


def score_candidate(row: dict[str, Any], median_tokens: float) -> tuple[float, dict[str, Any]]:
    metrics = evaluate_python_output(row["generation"])
    log_calls = metrics["logger_info_calls"] + metrics["logging_info_calls"]
    length_penalty = abs(token_count(row) - median_tokens)
    score = (
        100.0 * float(metrics["valid_compliant"])
        + 20.0 * float(metrics["syntax_valid"])
        + 5.0 * float(log_calls > 0)
        - 5.0 * float(metrics["print_calls"])
        - 0.02 * float(length_penalty)
    )
    return score, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations_gated_test.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations_verifier_rerank.jsonl"))
    parser.add_argument("--report-output", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "verifier_rerank_report.json"))
    parser.add_argument("--condition-name", default="verifier_rerank")
    parser.add_argument("--conditions", default=",".join(DEFAULT_PRIORITY))
    args = parser.parse_args()

    allowed = [x.strip() for x in args.conditions.split(",") if x.strip()]
    priority = {condition: idx for idx, condition in enumerate(allowed)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.generations):
        if row.get("condition") in priority:
            grouped[row["prompt_id"]].append(row)

    selected_rows = []
    source_counts: Counter[str] = Counter()
    for prompt_id, candidates in sorted(grouped.items()):
        lengths = [token_count(row) for row in candidates]
        median_tokens = statistics.median(lengths) if lengths else 0
        scored = []
        for row in candidates:
            score, metrics = score_candidate(row, median_tokens)
            scored.append((score, -priority[row["condition"]], -token_count(row), row, metrics))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        score, _, _, best, metrics = scored[0]
        source_counts[best["condition"]] += 1
        selected_rows.append(
            {
                **best,
                "source_condition": best["condition"],
                "condition": args.condition_name,
                "rerank_score": score,
                "candidate_count": len(candidates),
                "source_valid_compliant": metrics["valid_compliant"],
                "source_syntax_valid": metrics["syntax_valid"],
                "source_print_calls": metrics["print_calls"],
                "source_logger_info_calls": metrics["logger_info_calls"],
                "source_logging_info_calls": metrics["logging_info_calls"],
            }
        )

    write_jsonl(args.output, selected_rows)
    write_json(
        args.report_output,
        {
            "num_prompts": len(selected_rows),
            "source_condition_counts": dict(source_counts),
            "conditions": allowed,
            "score_formula": "100*valid_compliant + 20*syntax_valid + 5*has_logger - 5*print_calls - 0.02*abs(tokens-median)",
        },
    )
    print(f"Saved {len(selected_rows)} reranked generations to {args.output}")
    print(f"Saved rerank report to {args.report_output}")


if __name__ == "__main__":
    main()
