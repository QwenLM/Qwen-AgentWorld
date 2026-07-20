# Qwen-AgentWorld

<div style="text-align: center">
  <img width="400px" src="assets/logo.png">
  <p>
    <a href="http://arxiv.org/abs/2606.24597">📑 Technical Report</a> |
    <a href="https://qwen.ai/blog?id=qwen-agentworld">📖 Blog</a> |
    <a href="https://huggingface.co/collections/Qwen/qwen-agentworld">🤗 Hugging Face</a> |
    <a href="https://modelscope.cn/collections/Qwen/Qwen-AgentWorld">🤖 ModelScope</a> |
    <a href="https://qwen.ai/blog?id=qwen-agentworld#interactive-demo-interactive-demo">🖥️ Demo</a>
  </p>
</div>

<div align="center">
  <img src="assets/group_capybaras_flat.png" alt="Qwen-AgentWorld" width="1000px"/>
</div>

Welcome to the GitHub repository of Qwen-AgentWorld. Here, you can find official information about Qwen-AgentWorld, post your questions ([Issues](https://github.com/QwenLM/Qwen-AgentWorld/issues)), and share your ideas with the community ([Discussions](https://github.com/QwenLM/Qwen-AgentWorld/discussions)).


## News

- **2026-06-24**: We release **Qwen-AgentWorld-35B-A3B** and **AgentWorldBench**. Read more on our [blog](https://qwen.ai/blog?id=qwen-agentworld) and the [technical report](http://arxiv.org/abs/2606.24597).


## Open-Source Release

We open-source Qwen-AgentWorld-35B-A3B (model weights) and AgentWorldBench (evaluation benchmark):

| Release | Description |
|---------|-------------|
| [Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) | Language world model (MoE, 35B total / 3B active, 256K context) |
| [AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench) | Evaluation benchmark across 7 domains |

The official weights and data are released on:
- [🤗 HuggingFace](https://huggingface.co/Qwen): Download automatically via model ID, e.g., `Qwen/Qwen-AgentWorld-35B-A3B`. You can also download model files manually using `huggingface download` or `git clone`. Please follow the instructions on the model page.
- [🤖 ModelScope](https://modelscope.cn/organization/qwen): For users unable to access Hugging Face Hub. For supported frameworks, you can download from ModelScope by setting environment variables, such as `SGLANG_USE_MODELSCOPE=true` or `VLLM_USE_MODELSCOPE=true`.


## Introduction

<div align="center">
  <img src="assets/teaser.png" alt="Qwen-AgentWorld Overview" width="800px"/>
</div>

<br>

**Qwen-AgentWorld** is a native language world model that simulates agentic environments via long chain-of-thought reasoning across **seven unified domains**: MCP, Search, Terminal, SWE, Android, Web, and OS. It is trained through a three-stage pipeline -- CPT injects environment knowledge, SFT activates next-state-prediction reasoning, RL sharpens simulation fidelity -- on more than 10M real-world interaction trajectories. Unlike prior approaches that treat world modeling as a post-hoc add-on, Qwen-AgentWorld is a **native world model**: environment modeling is the training objective from the CPT stage onward.

Key features:

- **Seven Unified Domains.** The first language world model to cover seven agent interaction domains within a single model.
- **Native World Model.** Environment modeling from CPT onward, not post-hoc adaptation.
- **Generalizable, Scalable & Controllable Simulator.** Zero-shot generalization to OOD environments (e.g., Claw Agent); controllable perturbations and fictional-world construction surpass real-environment training.
- **Agent Foundation Model.** LWM RL warm-up on single-turn, non-agentic trajectories transfers to multi-turn, tool-calling agentic tasks across seven benchmarks, including three entirely out-of-domain.

## Performance

<div align="center">
  <img src="assets/bench_results_bar.png" alt="AgentWorldBench Results" width="800px"/>
</div>

Five-dimensional rubric mean (↑) per domain, normalized to 0--100 scale.

| Model | MCP | Search | Term. | SWE | Android | Web | OS | **Overall** |
|:------|:---:|:------:|:-----:|:---:|:-------:|:---:|:--:|:-----------:|
| GPT-5.4 | **70.10** | 37.26 | 53.69 | 66.29 | 60.00 | 51.80 | 68.58 | 58.25 |
| Claude Opus 4.8 | 54.93 | 35.14 | **59.18** | 64.10 | 61.50 | **54.66** | 66.62 | 56.59 |
| Claude Opus 4.6 | 69.90 | 29.30 | 57.51 | 64.55 | **61.74** | 51.42 | **70.20** | 57.80 |
| Gemini 3.1 Pro | 59.07 | 30.21 | 52.47 | 59.07 | 61.40 | 52.83 | 66.92 | 54.57 |
| Claude Sonnet 4.6 | 70.00 | 28.79 | 56.98 | 64.52 | 58.03 | 50.78 | 63.17 | 56.04 |
| DeepSeek-V4-Pro | 63.27 | 27.61 | 51.26 | 59.44 | 55.17 | 50.32 | 63.70 | 52.97 |
| GLM-5.1 | 67.60 | 22.46 | 47.32 | 52.07 | 59.10 | 51.50 | 59.13 | 51.31 |
| Kimi K2.6 | 65.23 | 27.48 | 52.54 | 58.77 | 58.93 | 50.20 | 60.80 | 53.42 |
| MiniMax-M2.7 | 55.82 | 27.30 | 41.62 | 37.44 | 52.40 | 50.52 | 57.73 | 46.12 |
| Qwen3.5-35B-A3B | 57.87 | 25.98 | 46.13 | 47.58 | 53.18 | 47.10 | 56.27 | 47.73 |
| Qwen3.5-397B-A17B | 68.31 | 30.81 | 55.30 | 64.44 | 54.90 | 48.55 | 60.85 | 54.74 |
| Qwen3.6-Plus | 55.28 | 21.94 | 50.58 | 59.08 | 57.65 | 50.78 | 60.33 | 50.81 |
| **Qwen-AgentWorld-35B-A3B** | 64.79 | 36.69 | 53.96 | 65.63 | 58.17 | 49.55 | 65.92 | 56.39 |
| **Qwen-AgentWorld-397B-A17B** | 68.24 | **37.82** | 57.73 | **68.49** | 60.20 | 50.98 | 67.89 | **58.71** |

Qwen-AgentWorld-397B-A17B achieves the highest overall score (58.71), outperforming all frontier proprietary models including GPT-5.4 (58.25). Qwen-AgentWorld-35B-A3B shows +8.66 improvement over Qwen3.5-35B-A3B without LWM training.

## Applications

**Generalizable Environment Scaling.** Sim RL with Qwen-AgentWorld-397B-A17B on 4k out-of-distribution OpenClaw environments:

> [!NOTE]
> **Official OpenClaw SimRL prompt.** For OpenClaw/Claw-Agent-style environments, we use Qwen-AgentWorld as a language world model with the following system prompt. The user message should contain the task instruction, optional interaction history, the current OpenClaw observation/state, and the action to simulate.
>
> ```text
> # Role and Objective
>
> You are an **OpenClaw World Model** — a precise simulator for OpenClaw agent environments. Your task is to predict the exact next environment observation after an agent action is executed.
>
> Given:
> 1. **Historical Context** (Optional): Previous interactions in the same trajectory
> 2. **Task Instruction**: The user goal for the OpenClaw task
> 3. **Current State**: The current environment observation/state visible to the agent
> 4. **Action**: The agent action to execute in the environment
>
> Predict the **exact next environment observation** after the action completes.
>
> ## Core Responsibilities
>
> 1. **State Prediction**: Generate the complete next OpenClaw observation after action execution
> 2. **Context Maintenance**: Track navigation state, UI state, files, tool results, and any persistent environment state across turns
> 3. **Behavioral Fidelity**: Stay faithful to realistic OpenClaw environment behavior, including invalid actions and no-op actions
>
> ## State Modeling Rules
>
> - Preserve the original observation format exactly
> - Keep ordering, labels, identifiers, coordinates, and structured fields consistent unless the action changes them
> - If the action has no visible or observable effect, return the unchanged state
> - Use prior turns as context when environment state persists across actions
> - Do not add hidden metadata or rewrite the observation into a different schema
>
> ## Action Modeling Rules
>
> - Treat the action supplied in the user message as the ground-truth action specification
> - Infer the next state from the semantics of the action and the current observation
> - Do not assume a closed action space; OpenClaw tasks may contain browser, UI, tool, or code-like actions depending on the environment
>
> ## Output Requirements
>
> First, think step by step to explain your reasoning. Then, provide the final predicted OpenClaw observation wrapped strictly within the `<predicted_observation></predicted_observation>` tags.
>
> Inside the tags, output only the full next observation with no extra commentary.
> ```
> [!NOTE]
> **OpenClaw SimRL prompt status.** This repository does not currently include a canonical SimRL/OpenClaw prompt or OpenClaw task configuration. The only prompt sources of truth shipped here are the seven AgentWorldBench domain templates under [`prompts/`](prompts/) and the per-domain paths in [`eval/lwm_eval_utils/task_configs.py`](eval/lwm_eval_utils/task_configs.py).
>
> For OpenClaw/Claw-Agent-style SimRL, use the prompt released with your OpenClaw/SimRL training configuration if available. If you need a repository-local starting point, select the closest domain template from `prompts/` and keep the same message contract used by AgentWorldBench samples: `system_str` is the system prompt, while the user message contains the trajectory context and current action.
>
> Expected user message template:
>
> ```text
> ### Turn {turn_idx}
> **Task Instruction:**
> {task_instruction}
>
> **Historical Context:**
> {previous_turns_optional}
>
> **Current State:**
> {current_openclaw_observation}
>
> **Action:**
> {action_to_simulate}
> ```
>
> The model response should include the predicted next observation in a `<predicted_observation></predicted_observation>` block, preserving the observation format used by the environment.

Qwen-AgentWorld-35B-A3B is supported by multiple inference frameworks. Here we demonstrate the usage of SGLang and vLLM.

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models.

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen-AgentWorld-35B-A3B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --context-length 262144 \
    --reasoning-parser qwen3
```

An OpenAI-compatible API will be available at `http://localhost:8000/v1`.

#### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference engine for LLMs.

```bash
vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --trust-remote-code
```

> [!TIP]
> The `--language-model-only` flag is required because the model architecture includes visual component definitions but the checkpoint only contains language model weights. Without this flag, vLLM will attempt to initialize visual modules and fail.

An OpenAI-compatible API will be available at `http://localhost:8000/v1`.

### Inference with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-AgentWorld-35B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": "You are a language world model simulating a Linux terminal environment. "
                   "Given the user's command, predict the terminal output."
    },
    {
        "role": "user",
        "content": "Action: execute_bash\nCommand: ls -la /home/user/project/"
    }
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.6)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

> [!NOTE]
> We provide **domain-specific world model system prompt templates** in [`prompts/`](prompts/) for all seven domains. These can serve as general-purpose system prompts when using Qwen-AgentWorld as an environment simulator. Each domain folder contains a `system_prompt.txt` (world model system prompt) and a `judge_system_prompt.txt` (evaluation prompt).

## Evaluate on AgentWorldBench

AgentWorldBench evaluates language world models by scoring each predicted environment observation on five dimensions: **Format**, **Factuality**, **Consistency**, **Realism**, and **Quality**.

<div align="center">
  <img src="assets/bench_overview.png" alt="AgentWorldBench Overview" width="800px"/>
  <p><em>Overview of AgentWorldBench: domain distribution, source benchmarks, evaluation dimensions, and per-domain trajectory statistics.</em></p>
</div>

### Setup

```bash
# Download the benchmark
huggingface-cli download Qwen/AgentWorldBench --repo-type dataset --local-dir ./AgentWorldBench

# Install dependencies
pip install openai
```

### Data Format

AgentWorldBench consists of per-domain JSONL files (`mcp_test.jsonl`, `search_test.jsonl`, `terminal_test.jsonl`, `swe_test.jsonl`, `android_test.jsonl`, `web_test.jsonl`, `os_test.jsonl`).

```json
{
    "task": "mcp",
    "id": 145256090131919,
    "prompt": ["### Turn 1\n**Action:**\n```json\n{...}\n```\n..."],
    "response": ["**Environment Observation:**\n{...}"],
    "current_prompt": "### Turn 1\n**Action:**\n...",
    "system_str": "# Role and Objective\n\nYou are a **Tool World Model** ...",
    "turn_idx": 1,
    "total_turns": 5
}
```

Key fields:
- **`system_str`**: The world model system prompt for this specific sample. Each sample carries its own system prompt, so the prompts provided in the `prompts/` directory of this repo are **templates for reference only**.
- **`prompt`** / **`response`**: Lists of all turns in the trajectory (action prompts and ground-truth environment observations).
- **`current_prompt`**: The action prompt for the turn being evaluated.
- **`turn_idx`**: 1-indexed position of the current turn.

### Run Evaluation

We provide a standalone evaluation script (`eval/eval.py`) that uses the OpenAI-compatible API for both world model inference and LLM judge scoring. The evaluation follows a three-step pipeline:

```bash
cd eval

# Step 1: Run world model inference
python eval.py infer \
    --data-dir ../AgentWorldBench \
    --model-base-url http://localhost:8000/v1 \
    --model-name Qwen/Qwen-AgentWorld-35B-A3B \
    --output-dir ./results

# Step 2: Run LLM judge scoring
export OPENAI_API_KEY="your-api-key"
python eval.py judge \
    --predictions ./results/predictions.jsonl \
    --judge-base-url https://api.openai.com/v1 \
    --judge-model gpt-5.2-2025-12-11 \
    --output-dir ./results

# Step 3: Aggregate and display scores
python eval.py score --predictions ./results/judged.jsonl
```

The judge prompts used for scoring are located in `prompts/{domain}/judge_system_prompt.txt`. The world model system prompts in `prompts/{domain}/system_prompt.txt` are provided as **reference templates**; during evaluation, the system prompt from each sample's `system_str` field is used instead.

### Evaluation Output

The `score` command outputs per-domain and overall results:

```
======================================================================
AgentWorldBench Evaluation Results (example output)
======================================================================

--- MCP (286/286 valid, 0 failed) ---
         format: 81.46
     factuality: 68.75
    consistency: 72.92
       realism: 71.88
        quality: 67.08
    total_score: 72.42
...

======================================================================
Overall: 56.39
======================================================================
```


## Finetuning

We advise you to use training frameworks, including [Swift](https://github.com/modelscope/swift), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), [UnSloth](https://github.com/unslothai/unsloth), etc., to finetune the model on domain-specific environment data.


## License Agreement

All open-weight models and AgentWorldBench are licensed under Apache 2.0.
You can find the license files in the respective Hugging Face repositories.


## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@article{zuo2026qwen,
  title={Qwen-agentworld: language world models for general agents},
  author={Zuo, Yuxin and Xiao, Zikai and Sheng, Li and Huang, Fei and Tu, Jianhong and Liu, Yuxuan and Tang, Tianyi and Hu, Xiaomeng and Su, Yang and Lan, Qingfeng and others},
  journal={arXiv preprint arXiv:2606.24597},
  year={2026}
}
```


## Contact Us

If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/CV4E9rpNSD) or [WeChat groups](https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png)!
