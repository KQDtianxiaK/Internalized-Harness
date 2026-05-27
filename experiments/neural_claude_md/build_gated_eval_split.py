from __future__ import annotations

import argparse
from pathlib import Path

from experiments.neural_claude_md.build_dataset import TASKS
from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR, write_jsonl


DEV_VARIANTS = [
    ("compact", "Keep the code compact and production-oriented."),
    ("typed", "Use type hints and a small helper where useful."),
]

TEST_VARIANTS = DEV_VARIANTS


def build_rows(tasks: list[tuple[str, str]], variants: list[tuple[str, str]], split_name: str) -> list[dict]:
    rows = []
    for task_id, task_prompt in tasks:
        for variant_id, variant_prompt in variants:
            rows.append(
                {
                    "id": f"gated_{split_name}_{len(rows):03d}_{task_id}_{variant_id}",
                    "split": split_name,
                    "task_id": task_id,
                    "variant_id": variant_id,
                    "prompt": f"{task_prompt} {variant_prompt} Return only Python code.",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "data"))
    parser.add_argument("--dev-task-count", type=int, default=10)
    parser.add_argument("--test-task-count", type=int, default=20)
    args = parser.parse_args()

    needed = args.dev_task_count + args.test_task_count
    if len(TASKS) < needed:
        raise ValueError(f"Need {needed} tasks, but build_dataset.TASKS only has {len(TASKS)}")

    dev_tasks = TASKS[: args.dev_task_count]
    test_tasks = TASKS[args.dev_task_count : needed]

    dev_rows = build_rows(dev_tasks, DEV_VARIANTS, "gated_dev")
    test_rows = build_rows(test_tasks, TEST_VARIANTS, "gated_test")

    out = Path(args.output_dir)
    write_jsonl(out / "gated_dev_prompts.jsonl", dev_rows)
    write_jsonl(out / "gated_test_prompts.jsonl", test_rows)
    print(f"Wrote {len(dev_rows)} dev prompts to {out / 'gated_dev_prompts.jsonl'}")
    print(f"Wrote {len(test_rows)} test prompts to {out / 'gated_test_prompts.jsonl'}")


if __name__ == "__main__":
    main()
