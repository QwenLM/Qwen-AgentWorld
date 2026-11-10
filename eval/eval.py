"""
AgentWorldBench Evaluation Script

Standalone evaluation script for AgentWorldBench using the OpenAI-compatible API.
Supports any OpenAI-compatible endpoint (vLLM, SGLang, OpenAI, etc.) for both
world model inference and LLM judge scoring.

Usage:
    # Step 1: Run world model inference
    python eval.py infer \
        --data-dir ./AgentWorldBench \
        --model-base-url http://localhost:8000/v1 \
        --model-name Qwen/Qwen-AgentWorld-35B-A3B \
        --output-dir ./results

    # Step 2: Run judge scoring
    python eval.py judge \
        --predictions ./results/predictions.jsonl \
        --judge-base-url https://api.openai.com/v1 \
        --judge-model gpt-5.2-2025-12-11 \
        --output-dir ./results

    # Step 3: Aggregate scores
    python eval.py score --predictions ./results/judged.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Optional

from openai import OpenAI

from lwm_eval_utils import (
    TASK_CONFIGS,
    SCORE_DIMENSIONS,
    JUDGE_USER_PROMPT,
    parse_judge_output,
    parse_model_output,
    clean_response_marker,
    load_judge_system_prompts,
)
from lwm_eval_utils.checkpoint import RunCheckpoint, checkpoint_signature

INVALID_VALUE = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSONL files from the data directory."""
    data_dir = Path(data_dir)
    jobs = []
    for jsonl_file in sorted(data_dir.glob("*_test.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    jobs.append(json.loads(line))
    logger.info(f"Loaded {len(jobs)} evaluation samples from {data_dir}")
    return jobs


def get_subtask(job: Dict[str, Any]) -> str:
    """Extract subtask name from job's task field."""
    task = job.get("task", "mcp")
    return task.split("/")[-1] if "/" in task else task


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    jobs: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    max_tokens: int = 32768,
    temperature: float = 0.6,
    on_complete=None,
) -> List[Dict[str, Any]]:
    """Run world model inference on all jobs."""
    total = len(jobs)
    for i, job in enumerate(jobs):
        subtask = get_subtask(job)

        system_prompt = job.get("system_str", "")
        prompts = job.get("prompt", [])
        responses = job.get("response", [])
        turn_idx = job["turn_idx"] - 1

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for prompt, response in zip(prompts[:turn_idx], responses[:turn_idx]):
            messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ])
        messages.append({"role": "user", "content": prompts[turn_idx]})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            gen = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"[{i+1}/{total}] Inference failed for {job.get('id')}: {e}")
            if on_complete:
                raise RuntimeError("Inference failed; rerun with the same checkpoint to retry") from e
            gen = ""

        if on_complete and not gen:
            raise RuntimeError("Inference returned an empty response; rerun with the same checkpoint to retry")
        job["gen"] = gen
        if on_complete:
            on_complete(job)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"[{i+1}/{total}] Inference progress")

    return jobs


# ─── Judge ────────────────────────────────────────────────────────────────────

def build_judge_messages(
    job: Dict[str, Any],
    model_output: str,
    judge_system_prompts: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build LLM Judge evaluation messages."""
    subtask = get_subtask(job)
    prompts = job.get("prompt", [])
    responses = job.get("response", [])
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]

    turn_idx = max(job.get("turn_idx", 1) - 1, 0)

    context = ""
    for i in range(turn_idx):
        if i < len(prompts) and i < len(responses):
            context += prompts[i] + "\n" + responses[i] + "\n\n"
    if context:
        context = f"# Context (Historical Interactions):\n\n{context}"

    current_prompt = job.get("current_prompt", "")
    if not current_prompt and turn_idx < len(prompts):
        current_prompt = prompts[turn_idx]

    ground_truth_raw = responses[turn_idx] if turn_idx < len(responses) else ""

    model_output_clean = clean_response_marker(model_output, subtask)
    ground_truth_clean = clean_response_marker(ground_truth_raw, subtask)

    user_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        world_model_input=f"# Current Turn:\n\n{current_prompt}",
        predicted_observation=f"**World Model Output (Simulated):**\n```\n{model_output_clean}\n```",
        ground_truth=f"**Ground Truth (Real Output):**\n```\n{ground_truth_clean}\n```",
    ).strip()

    return [
        {"role": "system", "content": judge_system_prompts.get(subtask, "")},
        {"role": "user", "content": user_prompt},
    ]


def run_judge(
    jobs: List[Dict[str, Any]],
    client: OpenAI,
    judge_model: str,
    max_tokens: int = 32768,
    temperature: float = 0.6,
    max_retries: int = 3,
    on_complete=None,
    judge_system_prompts=None,
) -> List[Dict[str, Any]]:
    """Run LLM Judge scoring on all jobs."""
    if judge_system_prompts is None:
        judge_system_prompts = load_judge_system_prompts()
    total = len(jobs)

    for i, job in enumerate(jobs):
        gen = job.get("gen", "")
        if not gen:
            job["failed"] = 1.0
            job["error_message"] = "No model generation"
            if on_complete:
                on_complete(job)
            continue

        subtask = get_subtask(job)
        config = TASK_CONFIGS.get(subtask, TASK_CONFIGS["mcp"])
        response_tag = config.get("response_tag", "predicted_observation")
        judge_response_tag = config.get("judge_response_tag", "final_evaluation")

        model_output = parse_model_output(gen, response_tag)
        judge_messages = build_judge_messages(job, model_output, judge_system_prompts)

        parsed = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=judge_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_output = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"[{i+1}/{total}] Judge call failed (attempt {attempt+1}): {e}")
                time.sleep(2)
                continue

            parsed = parse_judge_output(raw_output, response_tag=judge_response_tag)
            if parsed["success"]:
                break
            logger.warning(f"[{i+1}/{total}] Judge parse failed (attempt {attempt+1})")

        if parsed and parsed["success"]:
            scores = parsed.get("scores", {})
            job.update({
                "total_score": parsed.get("total_score", 0.0),
                "format": scores.get("format", 0),
                "factuality": scores.get("factuality", 0),
                "consistency": scores.get("consistency", 0),
                "realism": scores.get("realism", 0),
                "quality": scores.get("quality", 0),
                "failed": 0.0,
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "extracted_output": model_output,
                "judge_raw_output": parsed.get("judge_raw_output", ""),
            })
        else:
            if on_complete:
                raise RuntimeError("Judge scoring failed; rerun with the same checkpoint to retry")
            job.update({
                "total_score": INVALID_VALUE,
                "failed": 1.0,
                "error_message": "Judge scoring failed",
                "extracted_output": model_output,
            })

        if on_complete:
            on_complete(job)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"[{i+1}/{total}] Judge progress")

    return jobs


# ─── Score Aggregation ────────────────────────────────────────────────────────

def aggregate_scores(jobs: List[Dict[str, Any]]) -> None:
    """Aggregate and print scores grouped by subtask.

    Judge scores are on a 1-5 scale per dimension. We normalize to 0-100
    for reporting: normalized = (raw - 1) / 4 * 100.
    """
    subtask_jobs: Dict[str, List] = defaultdict(list)
    for job in jobs:
        subtask = get_subtask(job)
        subtask_jobs[subtask].append(job)

    domain_display = {
        "mcp": "MCP", "search": "Search", "terminal": "Terminal",
        "swe": "SWE", "android": "Android", "web": "Web", "os": "OS",
    }

    def normalize(raw: float) -> float:
        return (raw - 1) / 4 * 100

    print("\n" + "=" * 70)
    print("AgentWorldBench Evaluation Results")
    print("=" * 70)

    all_totals = []

    for subtask in sorted(subtask_jobs.keys()):
        sj = subtask_jobs[subtask]
        valid = [j for j in sj if j.get("failed", 1.0) == 0.0]
        failed = len(sj) - len(valid)

        display_name = domain_display.get(subtask, subtask.capitalize())
        print(f"\n--- {display_name} ({len(valid)}/{len(sj)} valid, {failed} failed) ---")

        if not valid:
            print("  No valid results.")
            continue

        for dim in SCORE_DIMENSIONS + ["total_score"]:
            values = [j.get(dim, 0) for j in valid if j.get(dim, INVALID_VALUE) != INVALID_VALUE]
            avg = mean(values) if values else 0.0
            score = normalize(avg)
            if dim == "total_score":
                all_totals.extend(values)
            print(f"  {dim:>15s}: {score:.2f}")

    if all_totals:
        overall = normalize(mean(all_totals))
        print(f"\n{'=' * 70}")
        print(f"Overall: {overall:.2f}")
        print(f"Total samples: {len(jobs)}, Valid: {len(all_totals)}, Failed: {len(jobs) - len(all_totals)}")
        print("=" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def same_path(first, second):
    first, second = Path(first), Path(second)
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and os.path.samefile(first, second)
    )


def run_with_checkpoint(jobs, path, settings, runner):
    if not path:
        return runner(jobs, None)
    signature = checkpoint_signature(jobs, settings)
    with RunCheckpoint(path, signature, jobs) as checkpoint:
        remaining = jobs[len(checkpoint.completed):]
        logger.info(f"Resuming with {len(checkpoint.completed)}/{len(jobs)} samples complete")
        runner(remaining, checkpoint.append)
        if len(checkpoint.completed) != len(jobs):
            raise ValueError("Evaluation finished without checkpointing every sample")
        return checkpoint.completed


def main():
    parser = argparse.ArgumentParser(description="AgentWorldBench Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # infer
    p_infer = subparsers.add_parser("infer", help="Run world model inference")
    p_infer.add_argument("--data-dir", required=True, help="Path to AgentWorldBench data")
    p_infer.add_argument("--model-base-url", required=True, help="OpenAI-compatible API base URL")
    p_infer.add_argument("--model-name", required=True, help="Model name / ID")
    p_infer.add_argument("--model-api-key", default="EMPTY", help="API key (default: EMPTY)")
    p_infer.add_argument("--output-dir", default="./results", help="Output directory")
    p_infer.add_argument("--max-tokens", type=int, default=32768)
    p_infer.add_argument("--temperature", type=float, default=0.6)
    p_infer.add_argument("--checkpoint", help="Save progress here and resume a matching run")

    # judge
    p_judge = subparsers.add_parser("judge", help="Run LLM judge scoring")
    p_judge.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    p_judge.add_argument("--judge-base-url", required=True, help="Judge API base URL")
    p_judge.add_argument("--judge-model", required=True, help="Judge model name")
    p_judge.add_argument("--judge-api-key", default=None, help="Judge API key (default: OPENAI_API_KEY env)")
    p_judge.add_argument("--output-dir", default="./results")
    p_judge.add_argument("--max-tokens", type=int, default=32768)
    p_judge.add_argument("--temperature", type=float, default=0.6)
    p_judge.add_argument("--max-retries", type=int, default=3)
    p_judge.add_argument("--checkpoint", help="Save progress here and resume a matching run")

    # score
    p_score = subparsers.add_parser("score", help="Aggregate scores from judged results")
    p_score.add_argument("--predictions", required=True, help="Path to judged JSONL")

    args = parser.parse_args()

    if args.command == "infer":
        out_dir = Path(args.output_dir)
        out_path = out_dir / "predictions.jsonl"
        if args.checkpoint and same_path(args.checkpoint, out_path):
            parser.error("--checkpoint must differ from the output file")
        jobs = load_data(args.data_dir)
        client = OpenAI(base_url=args.model_base_url, api_key=args.model_api_key)
        settings = {"stage": "infer", "base_url": args.model_base_url, "model": args.model_name,
                    "max_tokens": args.max_tokens, "temperature": args.temperature}
        jobs = run_with_checkpoint(
            jobs, args.checkpoint, settings,
            lambda remaining, save: run_inference(
                remaining, client, args.model_name, args.max_tokens, args.temperature, on_complete=save
            ),
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        logger.info(f"Predictions saved to {out_path}")

    elif args.command == "judge":
        out_dir = Path(args.output_dir)
        out_path = out_dir / "judged.jsonl"
        if args.checkpoint and (same_path(args.checkpoint, out_path) or same_path(args.checkpoint, args.predictions)):
            parser.error("--checkpoint must differ from the input and output files")
        with open(args.predictions, "r") as f:
            jobs = [json.loads(line) for line in f if line.strip()]

        api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = OpenAI(base_url=args.judge_base_url, api_key=api_key)
        prompts = load_judge_system_prompts()
        settings = {"stage": "judge", "base_url": args.judge_base_url, "model": args.judge_model,
                    "max_tokens": args.max_tokens, "temperature": args.temperature,
                    "max_retries": args.max_retries, "prompts": prompts}
        jobs = run_with_checkpoint(
            jobs, args.checkpoint, settings,
            lambda remaining, save: run_judge(
                remaining, client, args.judge_model, args.max_tokens, args.temperature,
                args.max_retries, on_complete=save, judge_system_prompts=prompts,
            ),
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        logger.info(f"Judged results saved to {out_path}")
        aggregate_scores(jobs)

    elif args.command == "score":
        with open(args.predictions, "r") as f:
            jobs = [json.loads(line) for line in f if line.strip()]
        aggregate_scores(jobs)


if __name__ == "__main__":
    main()
