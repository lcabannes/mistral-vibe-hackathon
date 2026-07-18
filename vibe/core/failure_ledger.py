from __future__ import annotations

import hashlib
import json
from typing import Any


def failure_key(tool_name: str, args: dict[str, Any]) -> str:
    """Stable identity for 'the same command': tool name + exact arguments."""
    digest = hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"{tool_name}:{digest}"


class FailureLedger:
    """Consecutive-failure counter per identical tool call, session-scoped.

    A failure increments the counter; a success resets it to zero. Used to
    warn the user when the agent keeps retrying a command that keeps failing.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record_failure(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def record_success(self, key: str) -> None:
        self._counts.pop(key, None)

    def count(self, key: str) -> int:
        return self._counts.get(key, 0)

    def reset(self) -> None:
        self._counts.clear()
