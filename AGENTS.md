# Repository Guidelines

## Project Structure & Module Organization

This workspace combines a representation-engineering experiment with a local copy of Natural Language Autoencoders.

- `experiments/neural_claude_md/`: dataset building, direction extraction, generation, evaluation, plotting, and shared utilities in `common.py`.
- `natural_language_autoencoders-main/`: NLA package and tooling. Core package code is in `nla/`; docs, configs, patches, tools, and examples live beside it.
- `model/`: local model checkpoints and tokenizer files. Treat these as large binary assets.
- `outputs/neural_claude_md/`: generated datasets, vectors, generations, metrics, and figures.
- `Plan.md` and `toy_experiment_plan.md`: legacy experiment notes and design context.
- `New_Plan.md` and `New_Results.md`: current internalized-harness research plan and results log.

## Build, Test, and Development Commands

- `uv sync`: install root dependencies from `pyproject.toml` and `uv.lock`.
- `uv pip install -e natural_language_autoencoders-main`: install the local NLA package in editable mode.
- `uv run python -m experiments.neural_claude_md.build_dataset`: build the toy experiment dataset.
- `uv run python -m experiments.neural_claude_md.validate_layer_alignment`: run the alignment sanity check.
- `uv run python -m experiments.neural_claude_md.evaluate_generations`: evaluate generated outputs after a run.
- `uv run ruff check .`: lint Python files with the repository Ruff settings.

Full Qwen7B/NLA runs require GPU resources; the README says the local RTX A400 is insufficient.

## Coding Style & Naming Conventions

Use Python 3.10 compatible code. Follow the existing style: scripts callable with `python -m`, explicit constants, and shared paths/helpers in `experiments/neural_claude_md/common.py`. Ruff uses `line-length = 119`; keep imports clean and avoid broad refactors in vendored NLA code.

Name experiment scripts by action, for example `build_dataset.py` or `evaluate_generations_safety.py`. Keep generated files under `outputs/`, not beside source modules.

## Testing Guidelines

There is no dedicated test suite in this checkout. Use script-level checks instead:

- Run `validate_layer_alignment` after changing layer/model handling.
- Run `evaluate_generations` after generation or scoring changes.
- Run `plot_results` after metric schema changes.

For new logic, prefer deterministic helpers that can later have `pytest` files named `test_<behavior>.py`.

## Experiment Logging

For the internalized-harness research track, write new route plans and protocol changes to `New_Plan.md` before implementation.(Using Chinese) After each experiment stage finishes, append objective results, artifact paths, metrics, and failure modes to `New_Results.md`.(Using Chinese) Keep `Plan.md` and `toy_experiment_plan.md` as legacy context unless the user explicitly asks to update them.

## Commit & Pull Request Guidelines

Git history is not accessible here, so no project-specific convention can be inferred. Use concise imperative subjects, such as `Add safety generation evaluator`, and include the experiment scope when helpful.

Pull requests should describe the changed workflow, list commands run, mention required GPUs or checkpoints, and link relevant artifacts under `outputs/neural_claude_md/`. Include screenshots only for plot or figure changes.

## Security & Configuration Tips

Do not commit API keys, private datasets, or new checkpoint files. Keep values such as `CUDA_VISIBLE_DEVICES`, server ports, and API keys in the shell or local scripts rather than reusable modules.
