# Neural CLAUDE.md Toy Experiment

This directory implements the `print() -> logger.info()` neural harness experiment.

## Setup

```bash
uv sync
uv pip install -e natural_language_autoencoders-main
```

Run the full experiment on the H100 server. The local RTX A400 is not sufficient for full Qwen7B/NLA runs.

## Recommended Run Order

Build data:

```bash
uv run python -m experiments.neural_claude_md.build_dataset
```

Launch the NLA AV server:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m sglang.launch_server \
  --model-path model/nla-qwen2.5-7b-L20-av \
  --port 30000 \
  --disable-radix-cache \
  --trust-remote-code \
  --mem-fraction-static 0.85
```

Validate layer alignment:

```bash
uv run python -m experiments.neural_claude_md.validate_layer_alignment
```

Extract the contrastive direction:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.neural_claude_md.extract_contrastive_direction
```

Build the NLA AR direction and AV explanations:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m experiments.neural_claude_md.build_nla_direction
```

Run generations:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.neural_claude_md.run_generation
```

Evaluate and plot:

```bash
uv run python -m experiments.neural_claude_md.evaluate_generations
uv run python -m experiments.neural_claude_md.plot_results
```

