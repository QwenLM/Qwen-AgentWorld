# FAQ

## Model comparison and training lineage

Qwen-AgentWorld benchmark results can be compared with other published model results under the evaluation setup described in the README, blog, technical report, or model card. This repository contains the AgentWorldBench evaluation tooling and prompt/configuration code. It is not the authoritative source for unpublished third-party model lineage or retraining history.

For model-background questions, please rely on the model card, release notes, or maintainer announcements for the specific checkpoint being evaluated. If a benchmark table compares against another model family, treat it as an evaluation result under the stated benchmark setup rather than a statement about that model's training recipe unless the release explicitly says so.

## Official prompt for SimRL/OpenClaw-style evaluation

The evaluation script does not hard-code one universal prompt for SimRL or OpenClaw-style runs. It builds chat messages from each benchmark sample:

- `system_str`, when present, is sent as the system message.
- `current_prompt`, when present, is sent as the current user message.
- If `current_prompt` is absent, the script falls back to the sample's `prompt` list and `turn_idx`.

Use the prompt fields shipped with the benchmark/evaluation data as the canonical prompt source for reproducing AgentWorldBench runs. Domain-specific response markers and prompt file paths are configured in `eval/lwm_eval_utils/task_configs.py`.
