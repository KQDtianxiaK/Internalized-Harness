from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR


def save_bar(summary: pd.DataFrame, out_dir: Path, metric: str, filename: str, title: str) -> None:
    pivot = summary.pivot_table(index="condition_alpha", columns="split", values=metric, aggfunc="mean").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("condition")
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    ax.legend(title="split")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def save_alpha_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    neural = summary[summary["condition"].isin(["contrastive_neural", "nla_neural", "random_vector", "negative_contrastive"])]
    if neural.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for (condition, split), group in neural.groupby(["condition", "split"]):
        group = group.sort_values("alpha")
        ax.plot(group["alpha"], group["compliance_rate"], marker="o", label=f"{condition}/{split}")
    plt.ylim(0, 1)
    plt.title("Compliance rate by steering strength")
    plt.xlabel("alpha")
    plt.ylabel("compliance_rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_sweep.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "summary.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "figures"))
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary.copy()
    summary["condition_alpha"] = summary.apply(
        lambda r: r["condition"] if r["alpha"] == 0 else f"{r['condition']}@{r['alpha']}",
        axis=1,
    )
    save_bar(summary, out_dir, "compliance_rate", "compliance_bar.png", "Rule compliance by harness")
    save_bar(summary, out_dir, "syntax_valid_rate", "syntax_valid_bar.png", "Syntax validity by harness")
    save_alpha_sweep(summary, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
