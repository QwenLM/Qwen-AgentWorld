import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import eval as evaluator
from lwm_eval_utils.checkpoint import RunCheckpoint, checkpoint_signature


class FakeCompletions:
    def __init__(self, reply, interrupt_after=None):
        self.reply = reply
        self.interrupt_after = interrupt_after
        self.calls = []

    def create(self, **kwargs):
        if self.interrupt_after is not None and len(self.calls) == self.interrupt_after:
            raise KeyboardInterrupt()
        self.calls.append(kwargs)
        content = self.reply(kwargs) if callable(self.reply) else self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def client_for(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def jobs():
    return [
        {"id": i, "task": "mcp", "turn_idx": 1, "prompt": [f"prompt-{i}"],
         "response": [f"observation-{i}"], "system_str": "simulate"}
        for i in range(3)
    ]


class EvaluationCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run.jsonl"
        self.settings = {"stage": "infer", "model": "test-model", "base_url": "https://example.test/v1",
                         "max_tokens": 42, "temperature": 0.6}

    def infer(self, data, fake, path=None, settings=None):
        return evaluator.run_with_checkpoint(
            data, path, settings or self.settings,
            lambda remaining, save: evaluator.run_inference(
                remaining, client_for(fake), "test-model", 42, 0.6, on_complete=save
            ),
        )

    def test_inference_resumes_without_repeating_completed_calls(self):
        def reply(kw):
            return "simulated " + kw["messages"][-1]["content"]
        reference_client = FakeCompletions(reply)
        expected = self.infer(copy.deepcopy(jobs()), reference_client)
        first_client = FakeCompletions(reply, interrupt_after=2)
        with self.assertRaises(KeyboardInterrupt):
            self.infer(copy.deepcopy(jobs()), first_client, str(self.path))
        self.assertEqual(len(first_client.calls), 2)
        second_client = FakeCompletions(reply)
        actual = self.infer(copy.deepcopy(jobs()), second_client, str(self.path))
        self.assertEqual(actual, expected)
        self.assertEqual(len(second_client.calls), 1)
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 4)
        third_client = FakeCompletions(reply)
        self.assertEqual(self.infer(copy.deepcopy(jobs()), third_client, str(self.path)), expected)
        self.assertEqual(len(third_client.calls), 0)

    def test_judge_resumes_and_preserves_failed_samples(self):
        data = jobs()
        for job in data:
            job["gen"] = "<predicted_observation>simulated</predicted_observation>"
        data[0]["gen"] = ""
        reply = json.dumps({"scores": {d: 4 for d in evaluator.SCORE_DIMENSIONS}})
        prompts = {"mcp": "judge prompt"}
        settings = {"stage": "judge", "model": "judge", "base_url": "https://example.test/v1",
                    "temperature": 0.6, "max_tokens": 42, "max_retries": 3, "prompts": prompts}

        def judge(values, fake, path=None, run_settings=None):
            return evaluator.run_with_checkpoint(
                values, path, run_settings or settings,
                lambda remaining, save: evaluator.run_judge(
                    remaining, client_for(fake), "judge", 42, 0.6, 3,
                    on_complete=save, judge_system_prompts=prompts,
                ),
            )

        expected = judge(copy.deepcopy(data), FakeCompletions(reply))
        first = FakeCompletions(reply, interrupt_after=1)
        with self.assertRaises(KeyboardInterrupt):
            judge(copy.deepcopy(data), first, str(self.path))
        resumed = FakeCompletions(reply)
        actual = judge(copy.deepcopy(data), resumed, str(self.path))
        self.assertEqual(actual, expected)
        self.assertEqual(len(resumed.calls), 1)
        self.assertEqual(actual[0]["failed"], 1.0)

    def test_mismatched_input_or_settings_leave_checkpoint_unchanged(self):
        data = jobs()
        signature = checkpoint_signature(data, self.settings)
        with RunCheckpoint(str(self.path), signature, data) as checkpoint:
            checkpoint.append({**data[0], "gen": "complete"})
        original = self.path.read_bytes()
        for modified_jobs, modified_settings in (
            ([{**data[0], "prompt": ["changed"]}, *data[1:]], self.settings),
            (data, {**self.settings, "model": "another"}),
            (data, {**self.settings, "base_url": "https://another.test/v1"}),
            (data, {**self.settings, "temperature": 0.0}),
            (data, {**self.settings, "prompts": {"mcp": "changed"}}),
        ):
            fake = FakeCompletions("response")
            with self.assertRaises(ValueError):
                self.infer(copy.deepcopy(modified_jobs), fake, str(self.path), modified_settings)
            self.assertEqual(len(fake.calls), 0)
            self.assertEqual(self.path.read_bytes(), original)

    def test_malformed_and_out_of_order_records_are_not_modified(self):
        data = jobs()
        header = json.dumps({"version": 1, "signature": checkpoint_signature(data, self.settings)}) + "\n"
        for suffix in (
            '{"index":0',
            json.dumps({"index": 1, "result": data[0]}) + "\n",
            json.dumps({"index": 0, "input_hash": "wrong", "result": data[0]}) + "\n",
        ):
            self.path.write_text(header + suffix, encoding="utf-8")
            original = self.path.read_bytes()
            with self.assertRaises(ValueError):
                self.infer(copy.deepcopy(data), FakeCompletions("response"), str(self.path))
            self.assertEqual(self.path.read_bytes(), original)

    def test_checkpoint_contains_no_client_secret_and_rejects_active_lock(self):
        settings = {**self.settings, "base_url": "https://user:secret@example.test/v1"}
        self.infer(copy.deepcopy(jobs()), FakeCompletions("response"), str(self.path), settings)
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))
        alias = self.path.with_name("hardlink.jsonl")
        os.link(self.path, alias)
        signature = checkpoint_signature(jobs(), settings)
        with RunCheckpoint(str(self.path), signature, jobs()):
            for path in (self.path, alias):
                with self.assertRaisesRegex(ValueError, "locked"):
                    self.infer(copy.deepcopy(jobs()), FakeCompletions("response"), str(path), settings)

    def test_crashed_process_releases_lock(self):
        data = jobs()
        signature = checkpoint_signature(data, self.settings)
        script = """
import json, os, sys
from lwm_eval_utils.checkpoint import RunCheckpoint
path, signature, encoded = sys.argv[1:]
data = json.loads(encoded)
checkpoint = RunCheckpoint(path, signature, data).__enter__()
checkpoint.append({**data[0], 'gen': 'saved'})
os._exit(0)
"""
        env = os.environ.copy()
        eval_dir = str(Path(__file__).resolve().parents[1] / "eval")
        env["PYTHONPATH"] = os.pathsep.join([eval_dir, env.get("PYTHONPATH", "")])
        result = subprocess.run([sys.executable, "-c", script, str(self.path), signature, json.dumps(data)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        fake = FakeCompletions("response")
        actual = self.infer(copy.deepcopy(data), fake, str(self.path))
        self.assertEqual(actual[0]["gen"], "saved")
        self.assertEqual(len(fake.calls), 2)

    def test_empty_checkpoint_is_initialized_after_creation_crash(self):
        self.path.touch()
        fake = FakeCompletions("response")
        self.assertEqual(len(self.infer(copy.deepcopy(jobs()), fake, str(self.path))), 3)
        self.assertEqual(len(fake.calls), 3)

    def test_transient_inference_error_is_retried_after_resume(self):
        class Flaky(FakeCompletions):
            def create(self, **kwargs):
                if len(self.calls) == 1:
                    raise ConnectionError("temporary failure")
                return super().create(**kwargs)

        first = Flaky("response")
        with self.assertRaisesRegex(RuntimeError, "rerun"):
            self.infer(copy.deepcopy(jobs()), first, str(self.path))
        self.assertEqual(len(first.calls), 1)
        resumed = FakeCompletions("response")
        self.assertEqual(len(self.infer(copy.deepcopy(jobs()), resumed, str(self.path))), 3)
        self.assertEqual(len(resumed.calls), 2)

    def test_judge_failure_is_retried_after_resume(self):
        data = jobs()
        for job in data:
            job["gen"] = "simulated"
        prompts = {"mcp": "judge"}
        settings = {"stage": "judge", "prompts": prompts}
        invalid = FakeCompletions("not JSON")
        def run(fake):
            return evaluator.run_with_checkpoint(
                copy.deepcopy(data), str(self.path), settings,
                lambda remaining, save: evaluator.run_judge(
                    remaining, client_for(fake), "judge", max_retries=2,
                    on_complete=save, judge_system_prompts=prompts,
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "rerun"):
            run(invalid)
        self.assertEqual(len(invalid.calls), 2)
        reply = json.dumps({"scores": {d: 4 for d in evaluator.SCORE_DIMENSIONS}})
        valid = FakeCompletions(reply)
        self.assertEqual(len(run(valid)), 3)
        self.assertEqual(len(valid.calls), 3)

    def test_cli_restores_checkpoint_and_preserves_output_format(self):
        data_dir = Path(self.temp.name) / "data"
        data_dir.mkdir()
        (data_dir / "mcp_test.jsonl").write_text(
            "".join(json.dumps(job) + "\n" for job in jobs()), encoding="utf-8"
        )
        out_dir = Path(self.temp.name) / "results"
        args = ["eval.py", "infer", "--data-dir", str(data_dir), "--model-base-url", "https://example.test/v1",
                "--model-name", "test-model", "--model-api-key", "test-secret", "--output-dir", str(out_dir),
                "--checkpoint", str(self.path)]
        with mock.patch.object(sys, "argv", args), mock.patch.object(evaluator, "OpenAI", return_value=client_for(FakeCompletions("response", 1))):
            with self.assertRaises(KeyboardInterrupt):
                evaluator.main()
        resumed = FakeCompletions("response")
        with mock.patch.object(sys, "argv", args), mock.patch.object(evaluator, "OpenAI", return_value=client_for(resumed)):
            evaluator.main()
        records = [json.loads(line) for line in (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["id"] for record in records], [0, 1, 2])
        self.assertEqual(len(resumed.calls), 2)
        self.assertNotIn("test-secret", self.path.read_text(encoding="utf-8"))

    def test_cli_rejects_checkpoint_aliasing_output_or_input(self):
        out_dir = Path(self.temp.name) / "results"
        out_dir.mkdir()
        output = out_dir / "predictions.jsonl"
        output.write_text("existing result", encoding="utf-8")
        alias = Path(self.temp.name) / "alias.jsonl"
        os.link(output, alias)
        args = ["eval.py", "infer", "--data-dir", self.temp.name, "--model-base-url", "https://example.test/v1",
                "--model-name", "test-model", "--output-dir", str(out_dir), "--checkpoint", str(alias)]
        with mock.patch.object(sys, "argv", args), mock.patch.object(evaluator, "OpenAI") as client:
            with self.assertRaises(SystemExit):
                evaluator.main()
        client.assert_not_called()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing result")

        args = ["eval.py", "judge", "--predictions", str(output), "--judge-base-url", "https://example.test/v1",
                "--judge-model", "judge", "--output-dir", self.temp.name, "--checkpoint", str(alias)]
        with mock.patch.object(sys, "argv", args), mock.patch.object(evaluator, "OpenAI") as client:
            with self.assertRaises(SystemExit):
                evaluator.main()
        client.assert_not_called()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing result")


if __name__ == "__main__":
    unittest.main()
