"""Incremental checkpoints for resumable evaluation runs."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _digest(value: Any) -> str:
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256()
    for part in encoder.iterencode(value):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def checkpoint_signature(jobs: List[Dict[str, Any]], settings: Dict[str, Any]) -> str:
    """Hash inputs, settings and evaluator code without storing credentials."""
    evaluator_dir = Path(__file__).resolve().parent.parent
    source_files = [evaluator_dir / "eval.py", *sorted((evaluator_dir / "lwm_eval_utils").glob("*.py"))]
    source_hashes = {str(path.relative_to(evaluator_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    payload = {"jobs": jobs, "settings": settings, "source_hashes": source_hashes}
    return _digest(payload)


def _lock_file(file):
    if os.name == "nt":
        import msvcrt
        os.lseek(file.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file):
    if os.name == "nt":
        import msvcrt
        os.lseek(file.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class RunCheckpoint:
    """Append completed jobs and resume only a validated prefix of the same run."""

    def __init__(self, path: str, signature: str, jobs: List[Dict[str, Any]]):
        self.path = Path(path).resolve()
        self.signature = signature
        self.input_hashes = [_digest(job) for job in jobs]
        self.completed = []
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = self.path.open("x+b", buffering=0)
        except FileExistsError:
            self._file = self.path.open("r+b", buffering=0)
        try:
            _lock_file(self._file)
        except OSError as exc:
            self._file.close()
            self._file = None
            raise ValueError(f"Checkpoint is locked: {self.path}") from exc

        try:
            header = {"version": 1, "signature": self.signature}
            if self._file.seek(0, os.SEEK_END) == 0:
                self._write(json.dumps(header) + "\n")
            else:
                self._file.seek(0)
                first = self._file.readline()
                if not first.endswith(b"\n") or json.loads(first) != header:
                    raise ValueError("Checkpoint does not match the current inputs or settings")
                for line in self._file:
                    if not line.endswith(b"\n"):
                        raise ValueError("Checkpoint ends with an incomplete record")
                    record = json.loads(line)
                    if (not isinstance(record, dict) or record.get("index") != len(self.completed)
                            or not isinstance(record.get("result"), dict)
                            or len(self.completed) >= len(self.input_hashes)
                            or record.get("input_hash") != self.input_hashes[len(self.completed)]):
                        raise ValueError("Checkpoint records are not a valid input prefix")
                    self.completed.append(record["result"])
            self._file.seek(0, os.SEEK_END)
            return self
        except BaseException:
            self._close()
            raise

    def append(self, result: Dict[str, Any]):
        if len(self.completed) >= len(self.input_hashes):
            raise ValueError("Checkpoint has more records than inputs")
        record = {"index": len(self.completed), "input_hash": self.input_hashes[len(self.completed)],
                  "result": result}
        self._write(json.dumps(record, ensure_ascii=False) + "\n")
        self.completed.append(result)

    def _write(self, text: str):
        data = text.encode("utf-8")
        if self._file.write(data) != len(data):
            raise OSError("Incomplete checkpoint write")
        self._file.flush()

    def _close(self):
        if self._file is not None:
            try:
                _unlock_file(self._file)
            finally:
                self._file.close()
                self._file = None

    def __exit__(self, exc_type, exc, tb):
        self._close()
