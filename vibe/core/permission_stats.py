from __future__ import annotations

import logging
from pathlib import Path
import tomllib

import tomli_w

from vibe.core.paths import VIBE_HOME

logger = logging.getLogger(__name__)

PERMISSION_STATS_FILE = "permission_stats.toml"
# Approvals of the same tool+pattern across sessions before suggesting an
# always-allow rule.
SUGGESTION_THRESHOLD = 5


def _stats_path() -> Path:
    return VIBE_HOME.path / PERMISSION_STATS_FILE


class PermissionStats:
    """Cross-session counter of user approvals per (tool, pattern).

    Every manual approval increments a counter keyed by the tool name and the
    matched permission pattern (e.g. ``bash`` + ``pytest *``). When a counter
    crosses ``SUGGESTION_THRESHOLD`` the UI suggests promoting it to a
    permanent allowlist rule — suggested at most once per key.
    """

    def __init__(self, stats_path: Path | None = None) -> None:
        self._path = stats_path or _stats_path()
        self._counts: dict[str, int] = {}
        self._suggested: set[str] = set()
        self._load()

    @staticmethod
    def key(tool_name: str, pattern: str) -> str:
        return f"{tool_name}::{pattern}"

    def record_approval(self, tool_name: str, pattern: str) -> int:
        k = self.key(tool_name, pattern)
        self._counts[k] = self._counts.get(k, 0) + 1
        self._save()
        return self._counts[k]

    def should_suggest(self, tool_name: str, pattern: str) -> bool:
        k = self.key(tool_name, pattern)
        if k in self._suggested:
            return False
        return self._counts.get(k, 0) >= SUGGESTION_THRESHOLD

    def mark_suggested(self, tool_name: str, pattern: str) -> None:
        self._suggested.add(self.key(tool_name, pattern))
        self._save()

    def count(self, tool_name: str, pattern: str) -> int:
        return self._counts.get(self.key(tool_name, pattern), 0)

    def _load(self) -> None:
        try:
            data = tomllib.loads(self._path.read_text())
        except FileNotFoundError:
            return
        except (OSError, tomllib.TOMLDecodeError) as e:
            logger.warning("Cannot read permission stats: %s", e)
            return
        counts = data.get("counts", {})
        if isinstance(counts, dict):
            self._counts = {str(k): int(v) for k, v in counts.items()}
        suggested = data.get("suggested", [])
        if isinstance(suggested, list):
            self._suggested = {str(s) for s in suggested}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                tomli_w.dump(
                    {"counts": self._counts, "suggested": sorted(self._suggested)}, f
                )
        except OSError as e:
            logger.warning("Cannot write permission stats: %s", e)
