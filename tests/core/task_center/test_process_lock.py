from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from vibe.core.task_center import (
    TaskCreate,
    TaskStore,
    _process_lock as process_lock_module,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKER = """
import asyncio
from pathlib import Path
import sys
import time

from vibe.core.task_center import TaskCreate, TaskStore

path = Path(sys.argv[1])
task_id = sys.argv[2]
started = Path(sys.argv[3])
entered = Path(sys.argv[4])
release = None if sys.argv[5] == "-" else Path(sys.argv[5])

def id_factory() -> str:
    entered.touch()
    while release is not None and not release.exists():
        time.sleep(0.01)
    return task_id

async def main() -> None:
    started.touch()
    await TaskStore(path, id_factory=id_factory).create(TaskCreate(title=task_id))

asyncio.run(main())
"""


def _worker(
    path: Path, task_id: str, started: Path, entered: Path, release: Path | None
) -> subprocess.Popen[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER,
            str(path),
            task_id,
            str(started),
            str(entered),
            str(release) if release is not None else "-",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {path}")
        time.sleep(0.01)


def _assert_success(process: subprocess.Popen[str]) -> None:
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"


def test_windows_locking_branch_uses_kernel_managed_lock(monkeypatch, tmp_path) -> None:
    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, _fd: int, mode: int, nbytes: int) -> None:
            self.calls.append((mode, nbytes))

    fake = FakeMsvcrt()
    monkeypatch.setattr(process_lock_module, "is_windows", lambda: True)
    monkeypatch.setattr(
        process_lock_module.importlib, "import_module", lambda _name: fake
    )
    path = tmp_path / "task-center.lock"

    with process_lock_module.process_file_lock(path):
        assert path.stat().st_size == 1

    assert fake.calls == [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)]


@pytest.mark.asyncio
async def test_process_lock_serializes_full_read_modify_replace(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    first_started = tmp_path / "first-started"
    first_entered = tmp_path / "first-entered"
    release_first = tmp_path / "release-first"
    first = _worker(path, f"task_{1:032x}", first_started, first_entered, release_first)
    _wait_for(first_entered)
    second_started = tmp_path / "second-started"
    second_entered = tmp_path / "second-entered"
    second = _worker(path, f"task_{2:032x}", second_started, second_entered, None)
    _wait_for(second_started)
    time.sleep(0.1)
    assert not second_entered.exists()

    release_first.touch()
    _assert_success(first)
    _assert_success(second)

    tasks = await TaskStore(path).load()
    assert {task.task_id for task in tasks} == {f"task_{1:032x}", f"task_{2:032x}"}


@pytest.mark.asyncio
async def test_process_death_does_not_leave_stale_transaction_lock(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    holder = _worker(
        path,
        f"task_{1:032x}",
        tmp_path / "holder-started",
        tmp_path / "holder-entered",
        tmp_path / "never-release",
    )
    _wait_for(tmp_path / "holder-entered")
    holder.kill()
    holder.wait(timeout=5)

    successor = _worker(
        path,
        f"task_{2:032x}",
        tmp_path / "successor-started",
        tmp_path / "successor-entered",
        None,
    )
    _assert_success(successor)

    tasks = await TaskStore(path).load()
    assert [task.task_id for task in tasks] == [f"task_{2:032x}"]


@pytest.mark.asyncio
async def test_waiting_for_process_lock_does_not_block_event_loop(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    release = tmp_path / "release-holder"
    holder = _worker(
        path,
        f"task_{1:032x}",
        tmp_path / "holder-started",
        tmp_path / "holder-entered",
        release,
    )
    _wait_for(tmp_path / "holder-entered")
    waiting = asyncio.create_task(
        TaskStore(path, id_factory=lambda: f"task_{2:032x}").create(
            TaskCreate(title="Second")
        )
    )

    await asyncio.sleep(0.05)
    assert not waiting.done()
    release.touch()
    await asyncio.wait_for(waiting, timeout=5)
    _assert_success(holder)

    assert len(await TaskStore(path).load()) == 2
