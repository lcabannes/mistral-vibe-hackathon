from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from vibe.core.team_workspace.identity import normalize_project_remote


class GitTeamWorkspaceError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class GitSyncResult:
    pushed: bool
    remote_branch_exists: bool


class GitTeamWorkspaceTransport:
    def __init__(
        self,
        *,
        remote_url: str,
        checkout_dir: Path,
        branch: str = "vibe-team-demo",
        max_retries: int = 3,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not remote_url.strip():
            raise ValueError("remote_url must not be empty")
        if not branch.strip() or branch.startswith("-"):
            raise ValueError("invalid team workspace branch")
        if max_retries < 1 or timeout_seconds <= 0:
            raise ValueError("Git sync bounds must be positive")
        self.remote_url = remote_url.strip()
        self.checkout_dir = checkout_dir.expanduser()
        self.branch = branch.strip()
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._prepared = False

    @property
    def materialization_root(self) -> Path:
        return self.checkout_dir / "state"

    def prepare(self) -> None:
        if self._prepared:
            return
        if self.checkout_dir.is_symlink():
            raise GitTeamWorkspaceError("invalid checkout")
        git_dir = self.checkout_dir / ".git"
        if git_dir.is_dir():
            self._validate_existing_checkout()
            self._prepared = True
            return
        if self.checkout_dir.exists() and any(self.checkout_dir.iterdir()):
            raise GitTeamWorkspaceError("checkout is not empty")
        self.checkout_dir.mkdir(parents=True, exist_ok=True)
        self._run("init", cwd=self.checkout_dir)
        self._run("remote", "add", "origin", self.remote_url, cwd=self.checkout_dir)
        self._run(
            "config", "user.name", "Mistral Vibe Team Workspace", cwd=self.checkout_dir
        )
        self._run(
            "config",
            "user.email",
            "vibe-team-workspace@localhost",
            cwd=self.checkout_dir,
        )
        self._run("checkout", "--orphan", self.branch, cwd=self.checkout_dir)
        self._prepared = True

    def sync(self) -> GitSyncResult:
        self.prepare()
        self._commit_local_changes()
        remote_exists = self._remote_branch_exists()
        for _attempt in range(self.max_retries):
            if remote_exists:
                self._fetch_and_rebase()
            push = self._run_result(
                "push",
                "--set-upstream",
                "origin",
                f"HEAD:refs/heads/{self.branch}",
                cwd=self.checkout_dir,
            )
            if push.returncode == 0:
                return GitSyncResult(pushed=True, remote_branch_exists=remote_exists)
            remote_exists = True
        raise GitTeamWorkspaceError("Git push retries exhausted")

    def _validate_existing_checkout(self) -> None:
        remote = self._run("remote", "get-url", "origin", cwd=self.checkout_dir).strip()
        if normalize_project_remote(remote) != normalize_project_remote(
            self.remote_url
        ):
            raise GitTeamWorkspaceError("checkout remote mismatch")
        current = self._run("branch", "--show-current", cwd=self.checkout_dir).strip()
        if current != self.branch:
            raise GitTeamWorkspaceError("checkout branch mismatch")

    def _commit_local_changes(self) -> None:
        self._run("add", "--", "state", cwd=self.checkout_dir)
        staged = self._run_result(
            "diff", "--cached", "--quiet", "--", "state", cwd=self.checkout_dir
        )
        if staged.returncode == 0:
            return
        if staged.returncode != 1:
            raise GitTeamWorkspaceError("failed to inspect staged team state")
        self._run("commit", "-m", "Update Vibe team workspace", cwd=self.checkout_dir)

    def _remote_branch_exists(self) -> bool:
        result = self._run_result(
            "ls-remote",
            "--exit-code",
            "--heads",
            "origin",
            f"refs/heads/{self.branch}",
            cwd=self.checkout_dir,
        )
        return result.returncode == 0

    def _fetch_and_rebase(self) -> None:
        remote_ref = f"refs/remotes/origin/{self.branch}"
        result = self._run_result(
            "fetch",
            "origin",
            f"+refs/heads/{self.branch}:{remote_ref}",
            cwd=self.checkout_dir,
        )
        if result.returncode != 0:
            raise GitTeamWorkspaceError("failed to fetch team workspace")
        rebase = self._run_result("rebase", remote_ref, cwd=self.checkout_dir)
        if rebase.returncode == 0:
            return
        self._run_result("rebase", "--abort", cwd=self.checkout_dir)
        raise GitTeamWorkspaceError("team workspace rebase conflicted")

    def _run(self, *args: str, cwd: Path) -> str:
        result = self._run_result(*args, cwd=cwd)
        if result.returncode != 0:
            raise GitTeamWorkspaceError("Git team workspace command failed")
        return result.stdout

    def _run_result(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GitTeamWorkspaceError("Git team workspace command failed") from error
