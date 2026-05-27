from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR, evaluate_python_output, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "results.csv"))
    parser.add_argument("--summary-output", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "summary.csv"))
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.generations):
        metrics = evaluate_python_output(row["generation"])
        rows.append({**row, **metrics})

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    group_cols = ["condition", "alpha", "split"]
    for optional in ["rank_k", "threshold", "alpha_mode", "alpha_base", "alpha_scale", "alpha_max"]:
        if optional in df.columns:
            group_cols.append(optional)
    agg = {
        "n": ("prompt_id", "count"),
        "compliance_rate": ("compliant", "mean"),
        "valid_compliance_rate": ("valid_compliant", "mean"),
        "syntax_valid_rate": ("syntax_valid", "mean"),
        "mean_print_calls": ("print_calls", "mean"),
        "mean_logger_info_calls": ("logger_info_calls", "mean"),
        "mean_logging_info_calls": ("logging_info_calls", "mean"),
        "mean_latency_s": ("latency_s", "mean"),
    }
    if "gate_fire_count" in df.columns:
        agg.update(
            {
                "mean_generated_tokens": ("generated_tokens", "mean"),
                "mean_gate_fire_count": ("gate_fire_count", "mean"),
                "mean_guard_block_count": ("guard_block_count", "mean"),
                "mean_gate_fire_rate": ("gate_fire_rate", "mean"),
                "mean_guard_block_rate": ("guard_block_rate", "mean"),
            }
        )

    summary = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    summary_out = Path(args.summary_output)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    print(f"Saved row-level metrics to {out}")
    print(f"Saved summary metrics to {summary_out}")


if __name__ == "__main__":
    main()
