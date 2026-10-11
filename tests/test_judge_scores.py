"""Regression tests for incomplete judge score replies."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from eval import aggregate_scores, run_judge  # noqa: E402
from lwm_eval_utils import parse_judge_output  # noqa: E402


def judge_reply(scores):
    return (
        "<final_evaluation>\n```json\n"
        + '{"strengths": [], "weaknesses": [], "scores": '
        + scores
        + "}\n```\n</final_evaluation>"
    )


PARTIAL = judge_reply('{"format": 5, "factuality": 5, "consistency": 5, "realism": 5}')
COMPLETE = judge_reply(
    '{"format": 5, "factuality": 5, "consistency": 5, "realism": 5, "quality": 1}'
)


class FakeJudgeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.replies)))]
        )


def sample_job():
    return {
        "id": "sample",
        "task": "mcp",
        "prompt": ["Call a tool"],
        "response": ["**Environment Observation:** success"],
        "current_prompt": "Call a tool",
        "turn_idx": 1,
        "gen": "<predicted_observation>success</predicted_observation>",
    }


class JudgeScoreTests(unittest.TestCase):
    def test_complete_reply_is_scored(self):
        parsed = parse_judge_output(COMPLETE, "final_evaluation")
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["total_score"], 4.2)

    def test_partial_reply_is_invalid(self):
        scores = {
            "format": 5,
            "factuality": 5,
            "consistency": 5,
            "realism": 5,
            "quality": 1,
        }
        for missing in scores:
            with self.subTest(missing=missing):
                partial = {key: value for key, value in scores.items() if key != missing}
                parsed = parse_judge_output(judge_reply(json.dumps(partial)), "final_evaluation")
                self.assertFalse(parsed["success"])

    def test_zero_score_is_invalid(self):
        parsed = parse_judge_output(
            judge_reply('{"format": 5, "factuality": 5, "consistency": 5, "realism": 5, "quality": 0}'),
            "final_evaluation",
        )
        self.assertFalse(parsed["success"])

    def test_partial_reply_is_retried_before_scoring(self):
        client = FakeJudgeClient([PARTIAL, COMPLETE])
        jobs = run_judge([sample_job()], client, "test-judge", max_retries=2)
        self.assertEqual(client.calls, 2)
        self.assertEqual(jobs[0]["failed"], 0.0)
        self.assertEqual(jobs[0]["total_score"], 4.2)

    def test_partial_reply_without_retry_is_excluded(self):
        client = FakeJudgeClient([PARTIAL])
        jobs = run_judge([sample_job()], client, "test-judge", max_retries=1)
        self.assertEqual(jobs[0]["failed"], 1.0)
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            aggregate_scores(jobs)
        self.assertIn("MCP (0/1 valid, 1 failed)", report.getvalue())
        self.assertIn("No valid results.", report.getvalue())

    def test_score_command_excludes_prior_incomplete_results(self):
        job = sample_job()
        job.update({
            "failed": 0.0,
            "format": 5,
            "factuality": 5,
            "consistency": 5,
            "realism": 5,
            "quality": 0,
            "total_score": 5.0,
        })
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            aggregate_scores([job])
        self.assertIn("MCP (0/1 valid, 1 failed)", report.getvalue())
        self.assertNotIn("Overall: 100.00", report.getvalue())


if __name__ == "__main__":
    unittest.main()
