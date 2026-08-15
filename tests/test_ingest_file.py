"""Offline CLI coverage for ``python -m worker <file> [--dry-run]``.

These tests call :func:`worker.main` / :func:`worker.ingest_local_file`
directly and, for the documented invocation, run ``python -m worker`` with
``REDIS_URL`` unset. They never need a live Redis, Celery worker, or OpenAI key.
"""

import os
import subprocess
import sys
from pathlib import Path

import worker

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_sample_fixture_exits_zero_and_prints_a_title(capsys):
    assert worker.main([str(FIXTURE), "--dry-run"]) == 0
    assert "Complete Entry" in capsys.readouterr().out


def test_missing_file_exits_nonzero(capsys):
    missing = "/definitely/not/a/streamrag-feed.xml"
    assert worker.main([missing]) != 0
    err = capsys.readouterr().err
    assert missing in err


def test_dry_run_does_not_call_embedder():
    def boom(entries):
        raise AssertionError(f"dry-run must not index {entries!r}")

    count = worker.ingest_local_file(str(FIXTURE), dry_run=True, embed_and_index=boom)
    assert count >= 1


def test_ingest_uses_injected_embedder_without_openai():
    received = {}

    def fake_embed(entries):
        received["entries"] = entries
        return len(entries)

    count = worker.ingest_local_file(str(FIXTURE), embed_and_index=fake_embed)
    assert count == len(received["entries"])
    assert received["entries"][0]["title"] == "Complete Entry"


def test_module_cli_dry_run_works_with_redis_url_unset():
    env = os.environ.copy()
    env.pop("REDIS_URL", None)
    result = subprocess.run(
        [sys.executable, "-m", "worker", str(FIXTURE), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Complete Entry" in result.stdout
