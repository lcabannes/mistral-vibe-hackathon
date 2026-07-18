from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePath
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic import BaseModel

    from vibe.core.config import AnyVibeConfig

# Paths whose contents must never enter cloud-visible context. Users extend
# this via privacy_routing.protected_paths.
DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*credentials*",
    "~/.ssh/**",
    "~/.aws/**",
)

PROTECTED_PATH_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit",
    "grep",
    "bash",
})


def protected_path_message(path: str) -> str:
    return (
        f"'{path}' is a protected path: its contents must not enter this "
        f"conversation. Delegate the whole file operation to the local model "
        f"instead: call the local_task tool with a precise task brief (what to "
        f"do, which files, what counts as done). The local task runs with "
        f"full access and reports only completion status back."
    )


def _expand(pattern: str) -> str:
    return str(Path(pattern).expanduser()) if pattern.startswith("~") else pattern


def is_protected_path(path: str, patterns: Sequence[str]) -> bool:
    """Match a path against protection globs.

    Bare patterns (no separator) match against the basename, so ``.env.*``
    protects ``sub/dir/.env.local`` too. Patterns with separators match
    against both the path as given and its absolute form, so relative
    patterns like ``contracts/**`` work regardless of how the tool call
    spelled the path.
    """
    try:
        resolved = Path(path).expanduser()
        full = str(resolved if resolved.is_absolute() else Path.cwd() / resolved)
    except (OSError, ValueError):
        full = path
    name = PurePath(full).name
    for pattern in patterns:
        expanded = _expand(pattern)
        if "/" in expanded:
            candidates = (full, path) if not Path(expanded).is_absolute() else (full,)
            prefix = expanded.rstrip("*/")
            for candidate in candidates:
                if fnmatch(candidate, expanded) or fnmatch(candidate, f"{prefix}/**"):
                    return True
        elif fnmatch(name, expanded):
            return True
    return False


def protection_patterns(config: AnyVibeConfig) -> tuple[str, ...]:
    settings = config.privacy_routing
    if not settings.enabled:
        return ()
    return (*DEFAULT_PROTECTED_PATHS, *settings.protected_paths)


def _bash_path_candidates(command: str) -> Iterable[str]:
    """Best-effort extraction of path-like words from a shell command.

    Not a parser and not airtight (command substitution, variables, etc.) —
    the redaction net downstream still catches credential-shaped content this
    misses. Splits on common shell operators so `cat .env && ls` yields both
    words.
    """
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        words = command.split()
    for word in words:
        for part in word.replace(";", " ").replace("|", " ").split():
            stripped = part.strip("\"'`()<>")
            if stripped and not stripped.startswith("-"):
                yield stripped


def find_protected_path_in_args(
    tool_name: str, args: BaseModel, patterns: Sequence[str]
) -> str | None:
    """Return the first protected path a tool call touches, if any."""
    if not patterns or tool_name not in PROTECTED_PATH_TOOLS:
        return None
    if tool_name == "bash":
        command = getattr(args, "command", "")
        for candidate in _bash_path_candidates(command):
            if is_protected_path(candidate, patterns):
                return candidate
        return None
    path = getattr(args, "file_path", None) or getattr(args, "path", None)
    if isinstance(path, str) and path and is_protected_path(path, patterns):
        return path
    return None
