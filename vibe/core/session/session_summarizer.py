from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from vibe.core.config import ModelConfig
from vibe.core.llm.types import BackendLike
from vibe.core.logger import logger
from vibe.core.prompts import UtilityPrompt
from vibe.core.session.session_logger import SessionLogger
from vibe.core.types import LLMMessage, Role, SessionSummary
from vibe.core.utils import utc_now
from vibe.core.utils.tokens import truncate_middle_to_tokens

# Keep the rendered transcript comfortably below small-model context limits.
TRANSCRIPT_MAX_TOKENS = 24_000
# Skip trivial sessions: a summary of one exchange adds nothing over the title.
MIN_MESSAGES_TO_SUMMARIZE = 4
# Mid-session checkpoint cadence: refresh once the transcript has grown this
# many messages past the last summary, so a crash loses at most one interval.
CHECKPOINT_MESSAGE_INTERVAL = 20
# Session-end summaries must finish before shutdown; don't hold quit hostage.
FINALIZE_TIMEOUT_SECONDS = 10.0
_SUMMARY_MAX_TOKENS = 512

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_TAGS_RE = re.compile(r"<tags>(.*?)</tags>", re.DOTALL)


def render_transcript(messages: Sequence[LLMMessage]) -> str:
    """Render non-system messages as a compact plain-text transcript."""
    lines: list[str] = []
    for message in messages:
        if message.role == Role.system:
            continue
        content = (message.content or "").strip()
        if message.role == Role.assistant and message.tool_calls:
            calls = ", ".join(tc.function.name or "?" for tc in message.tool_calls)
            content = f"{content}\n[tool calls: {calls}]".strip()
        if not content:
            continue
        lines.append(f"### {message.role.value}\n{content}")
    return truncate_middle_to_tokens("\n\n".join(lines), TRANSCRIPT_MAX_TOKENS)


def parse_summary_response(text: str, message_count: int) -> SessionSummary | None:
    summary_match = _SUMMARY_RE.search(text)
    if summary_match is None:
        return None
    summary = summary_match.group(1).strip()
    if not summary:
        return None

    tags: list[str] = []
    tags_match = _TAGS_RE.search(text)
    if tags_match is not None:
        tags = [t.strip().lower() for t in tags_match.group(1).split(",") if t.strip()]

    return SessionSummary(
        summary=summary,
        tags=tags,
        generated_at=utc_now().isoformat(),
        message_count=message_count,
    )


@dataclass(frozen=True, slots=True)
class _SessionSnapshot:
    messages: list[LLMMessage]
    session_dir: Path
    non_system_count: int


class SessionSummarizer:
    """Generates and persists small-model summaries of the session transcript.

    Two triggers:
    - ``maybe_checkpoint()`` after agent turns — only summarizes once the
      transcript has grown ``CHECKPOINT_MESSAGE_INTERVAL`` messages past the
      last summary, so a killed session keeps a recent-enough summary.
    - ``finalize()`` at session boundaries (app exit, /clear, /compact) —
      summarizes any growth since the last summary.

    Both snapshot the transcript and session directory synchronously — the
    caller may reset the session immediately afterwards — and generate in the
    background. Failures never raise into the caller; the stale (or missing)
    summary simply stays as-is.
    """

    def __init__(
        self,
        *,
        backend: BackendLike,
        model: ModelConfig,
        session_logger: SessionLogger,
        messages_getter: Callable[[], Sequence[LLMMessage]],
        on_summary: Callable[[SessionSummary], None] | None = None,
    ) -> None:
        self._backend = backend
        self._model = model
        self._session_logger = session_logger
        self._messages_getter = messages_getter
        self._on_summary = on_summary
        self._tasks: set[asyncio.Task[Any]] = set()
        # Message count covered by the most recent summary of the *live*
        # session, mirrored here so checkpointing works even before the
        # first summary has been persisted.
        self._summarized_count = 0

    def _snapshot(self) -> _SessionSnapshot | None:
        if not self._session_logger.enabled:
            return None
        session_dir = self._session_logger.session_dir
        if session_dir is None:
            return None
        messages = list(self._messages_getter())
        non_system_count = sum(1 for m in messages if m.role != Role.system)
        return _SessionSnapshot(
            messages=messages,
            session_dir=session_dir,
            non_system_count=non_system_count,
        )

    def _spawn(self, snapshot: _SessionSnapshot) -> asyncio.Task[None]:
        self._summarized_count = snapshot.non_system_count
        task = asyncio.create_task(self._summarize(snapshot))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def maybe_checkpoint(self) -> None:
        """Refresh the summary if the transcript has grown a full interval."""
        snapshot = self._snapshot()
        if snapshot is None:
            return
        if snapshot.non_system_count < MIN_MESSAGES_TO_SUMMARIZE:
            return
        if (
            snapshot.non_system_count - self._summarized_count
            < CHECKPOINT_MESSAGE_INTERVAL
        ):
            return
        self._spawn(snapshot)

    async def finalize(self, *, wait: bool = False) -> None:
        """Summarize the session at a boundary (exit, /clear, /compact).

        Snapshots synchronously so the caller may reset the session right
        after. With ``wait=True`` (app exit) the summary is awaited with a
        timeout so shutdown isn't held hostage by a slow network.
        """
        snapshot = self._snapshot()
        if (
            snapshot is None
            or snapshot.non_system_count < MIN_MESSAGES_TO_SUMMARIZE
            or snapshot.non_system_count <= self._summarized_count
        ):
            return
        task = self._spawn(snapshot)
        if wait:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=FINALIZE_TIMEOUT_SECONDS
                )

    def reset(self) -> None:
        """Forget summary progress; call when a new session starts."""
        self._summarized_count = 0

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _summarize(self, snapshot: _SessionSnapshot) -> None:
        try:
            transcript = render_transcript(snapshot.messages)
            if not transcript:
                return

            result = await self._backend.complete(
                model=self._model,
                messages=[
                    LLMMessage(
                        role=Role.system, content=UtilityPrompt.SESSION_SUMMARY.read()
                    ),
                    LLMMessage(role=Role.user, content=transcript),
                ],
                temperature=0.0,
                tools=None,
                tool_choice=None,
                max_tokens=_SUMMARY_MAX_TOKENS,
                extra_headers=None,
            )

            summary = parse_summary_response(
                result.message.content or "", snapshot.non_system_count
            )
            if summary is None:
                logger.warning("Session summary response had no <summary> block")
                return

            await self._session_logger.persist_summary(
                summary, session_dir=snapshot.session_dir
            )
            if self._on_summary is not None:
                self._on_summary(summary)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Session summary generation failed", exc_info=True)
