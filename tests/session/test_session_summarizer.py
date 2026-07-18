from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import ModelConfig, SessionLoggingConfig
from vibe.core.llm.types import BackendLike
from vibe.core.session.session_loader import METADATA_FILENAME
from vibe.core.session.session_logger import SessionLogger
from vibe.core.session.session_summarizer import (
    CHECKPOINT_MESSAGE_INTERVAL,
    MIN_MESSAGES_TO_SUMMARIZE,
    SessionSummarizer,
    parse_summary_response,
    render_transcript,
)
from vibe.core.types import LLMMessage, Role


@pytest.fixture
def session_config(tmp_path: Path) -> SessionLoggingConfig:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return SessionLoggingConfig(
        save_dir=str(session_dir), session_prefix="test", enabled=True
    )


@pytest.fixture
def model() -> ModelConfig:
    return ModelConfig(name="small", provider="mistral", alias="small")


def _conversation(rounds: int) -> list[LLMMessage]:
    messages: list[LLMMessage] = [LLMMessage(role=Role.system, content="system")]
    for i in range(rounds):
        messages.append(LLMMessage(role=Role.user, content=f"question {i}"))
        messages.append(LLMMessage(role=Role.assistant, content=f"answer {i}"))
    return messages


def _summary_response(text: str = "Fixed the login bug.", tags: str = "bugfix, auth"):
    return mock_llm_chunk(content=f"<summary>\n{text}\n</summary>\n<tags>{tags}</tags>")


class TestParseSummaryResponse:
    def test_parses_summary_and_tags(self) -> None:
        result = parse_summary_response(
            "<summary>Did things.</summary>\n<tags>python, Refactor</tags>",
            message_count=6,
        )
        assert result is not None
        assert result.summary == "Did things."
        assert result.tags == ["python", "refactor"]
        assert result.message_count == 6

    def test_missing_summary_block_returns_none(self) -> None:
        assert parse_summary_response("no blocks here", message_count=2) is None

    def test_empty_summary_returns_none(self) -> None:
        assert parse_summary_response("<summary>  </summary>", message_count=2) is None

    def test_missing_tags_is_tolerated(self) -> None:
        result = parse_summary_response("<summary>Done.</summary>", message_count=2)
        assert result is not None
        assert result.tags == []


class TestRenderTranscript:
    def test_includes_roles_and_excludes_system(self) -> None:
        messages = _conversation(1)
        rendered = render_transcript(messages)
        assert "### user\nquestion 0" in rendered
        assert "### assistant\nanswer 0" in rendered
        assert "system" not in rendered

    def test_empty_conversation_renders_empty(self) -> None:
        assert render_transcript([LLMMessage(role=Role.system, content="s")]) == ""


class TestSessionSummarizer:
    def _make(
        self,
        session_config: SessionLoggingConfig,
        model: ModelConfig,
        messages: list[LLMMessage],
        backend: FakeBackend | None = None,
    ) -> tuple[SessionSummarizer, SessionLogger, list[LLMMessage]]:
        # A mutable message list shared with the summarizer's getter, so tests
        # can grow or reset the conversation the way the agent loop would.
        live_messages = messages
        logger = SessionLogger(session_config, "sess-1234567890")
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)
        (logger.session_dir / METADATA_FILENAME).write_text("{}", encoding="utf-8")
        backend = backend or FakeBackend(_summary_response())
        summarizer = SessionSummarizer(
            backend=cast(BackendLike, backend),
            model=model,
            session_logger=logger,
            messages_getter=lambda: list(live_messages),
        )
        return summarizer, logger, live_messages

    async def _drain(self, summarizer: SessionSummarizer) -> None:
        import asyncio

        tasks = list(summarizer._tasks)  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_checkpoint_skips_short_sessions(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        messages = _conversation(1)
        assert len(messages) - 1 < MIN_MESSAGES_TO_SUMMARIZE
        summarizer, logger, _ = self._make(session_config, model, messages)
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is None

    @pytest.mark.asyncio
    async def test_checkpoint_skips_below_interval(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        # Above the minimum but below the checkpoint interval: no summary yet.
        rounds = (CHECKPOINT_MESSAGE_INTERVAL - 2) // 2
        summarizer, logger, _ = self._make(session_config, model, _conversation(rounds))
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is None

    @pytest.mark.asyncio
    async def test_checkpoint_fires_at_interval(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        rounds = CHECKPOINT_MESSAGE_INTERVAL // 2
        summarizer, logger, _ = self._make(session_config, model, _conversation(rounds))
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        summary = logger.session_metadata.summary
        assert summary is not None
        assert summary.summary == "Fixed the login bug."
        assert summary.message_count == CHECKPOINT_MESSAGE_INTERVAL

    @pytest.mark.asyncio
    async def test_checkpoint_waits_full_interval_before_refreshing(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        rounds = CHECKPOINT_MESSAGE_INTERVAL // 2
        backend = FakeBackend([
            [_summary_response("first")],
            [_summary_response("second")],
        ])
        summarizer, logger, messages = self._make(
            session_config, model, _conversation(rounds), backend=backend
        )
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)

        # A couple more messages: not a full interval past the last summary.
        messages.extend(_conversation(1)[1:])
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is not None
        assert logger.session_metadata.summary.summary == "first"

        # Grow past the next interval: refresh happens.
        messages.extend(_conversation(CHECKPOINT_MESSAGE_INTERVAL // 2)[1:])
        summarizer.maybe_checkpoint()
        await self._drain(summarizer)
        assert logger.session_metadata.summary.summary == "second"

    @pytest.mark.asyncio
    async def test_finalize_summarizes_any_growth(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        # Below the checkpoint interval but above the minimum: finalize fires.
        summarizer, logger, _ = self._make(session_config, model, _conversation(3))
        await summarizer.finalize()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is not None

    @pytest.mark.asyncio
    async def test_finalize_skips_when_nothing_new(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        backend = FakeBackend([
            [_summary_response("first")],
            [_summary_response("second")],
        ])
        summarizer, logger, _ = self._make(
            session_config, model, _conversation(3), backend=backend
        )
        await summarizer.finalize()
        await self._drain(summarizer)
        await summarizer.finalize()
        await self._drain(summarizer)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is not None
        assert logger.session_metadata.summary.summary == "first"

    @pytest.mark.asyncio
    async def test_finalize_wait_blocks_until_persisted(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        summarizer, logger, _ = self._make(session_config, model, _conversation(3))
        await summarizer.finalize(wait=True)
        # No drain: wait=True must have persisted before returning.
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is not None

    @pytest.mark.asyncio
    async def test_finalize_snapshot_survives_session_reset(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        """/clear resets messages and rotates the session dir right after
        finalize; the summary must land in the *old* session's metadata.
        """
        summarizer, logger, messages = self._make(
            session_config, model, _conversation(3)
        )
        old_session_dir = logger.session_dir
        assert old_session_dir is not None

        await summarizer.finalize()
        messages.clear()
        logger.reset_session("sess-new-9876543210")

        await self._drain(summarizer)

        import json

        persisted = json.loads(
            (old_session_dir / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        assert persisted["summary"]["summary"] == "Fixed the login bug."

    @pytest.mark.asyncio
    async def test_backend_errors_are_swallowed(
        self, session_config: SessionLoggingConfig, model: ModelConfig
    ) -> None:
        backend = FakeBackend(exception_to_raise=RuntimeError("boom"))
        summarizer, logger, _ = self._make(
            session_config, model, _conversation(3), backend=backend
        )
        await summarizer.finalize(wait=True)
        assert logger.session_metadata is not None
        assert logger.session_metadata.summary is None
