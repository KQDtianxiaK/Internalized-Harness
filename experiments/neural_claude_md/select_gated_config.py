from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "summary_gated_dev.csv"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gated" / "selected_config.json"))
    parser.add_argument("--condition", default="gated_contrastive")
    parser.add_argument("--min-syntax-valid", type=float, default=0.85)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    required = {"condition", "alpha", "rank_k", "threshold", "valid_compliance_rate", "syntax_valid_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")

    candidates = df[df["condition"] == args.condition].copy()
    if candidates.empty:
        raise ValueError(f"No rows found for condition {args.condition!r}")

    eligible = candidates[candidates["syntax_valid_rate"] >= args.min_syntax_valid].copy()
    if eligible.empty:
        eligible = candidates.copy()
        selection_note = f"No config met syntax_valid_rate >= {args.min_syntax_valid}; selected best available."
    else:
        selection_note = f"Selected among configs with syntax_valid_rate >= {args.min_syntax_valid}."

    eligible = eligible.sort_values(
        by=[
            "valid_compliance_rate",
            "syntax_valid_rate",
            "compliance_rate",
            "mean_gate_fire_rate",
            "mean_latency_s",
        ],
        ascending=[False, False, False, True, True],
    )
    best = eligible.iloc[0].to_dict()
    selected = {
        "condition": args.condition,
        "alpha": float(best["alpha"]),
        "rank_k": int(best["rank_k"]),
        "threshold": float(best["threshold"]),
        "alpha_mode": best.get("alpha_mode", "fixed"),
        "alpha_base": float(best["alpha_base"]) if "alpha_base" in best else None,
        "alpha_scale": float(best["alpha_scale"]) if "alpha_scale" in best else None,
        "alpha_max": float(best["alpha_max"]) if "alpha_max" in best else None,
        "selection_note": selection_note,
        "selected_metrics": best,
        "summary": str(Path(args.summary)),
    }
    write_json(args.output, selected)
    print(f"Selected gated config: alpha={selected['alpha']} rank_k={selected['rank_k']} threshold={selected['threshold']}")
    print(f"Wrote selection to {args.output}")


if __name__ == "__main__":
    main()
