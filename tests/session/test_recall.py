from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.config import SessionLoggingConfig
from vibe.core.session.recall import render_recall_context, search_session_summaries
from vibe.core.session.session_loader import METADATA_FILENAME


@pytest.fixture
def session_config(tmp_path: Path) -> SessionLoggingConfig:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return SessionLoggingConfig(
        save_dir=str(session_dir), session_prefix="test", enabled=True
    )


def _write_session(
    config: SessionLoggingConfig,
    name: str,
    *,
    session_id: str,
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    end_time: str = "2026-07-01T00:00:00+00:00",
    cwd: str = "/repo",
) -> None:
    session_dir = Path(config.save_dir) / f"{config.session_prefix}_{name}"
    session_dir.mkdir(parents=True)
    metadata: dict[str, object] = {
        "session_id": session_id,
        "title": title,
        "end_time": end_time,
        "environment": {"working_directory": cwd},
    }
    if summary is not None:
        metadata["summary"] = {
            "summary": summary,
            "tags": tags or [],
            "generated_at": end_time,
            "message_count": 6,
        }
    (session_dir / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


class TestSearchSessionSummaries:
    def test_returns_empty_when_no_sessions(
        self, session_config: SessionLoggingConfig
    ) -> None:
        assert search_session_summaries("auth", session_config) == []

    def test_sessions_without_summaries_are_skipped(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(session_config, "a", session_id="s1", title="No summary yet")
        assert search_session_summaries("", session_config) == []

    def test_keyword_matches_summary_body(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(
            session_config,
            "a",
            session_id="s1",
            summary="Fixed the authentication token refresh bug.",
        )
        _write_session(
            session_config, "b", session_id="s2", summary="Refactored the CSS layout."
        )
        results = search_session_summaries("authentication", session_config)
        assert [r.session_id for r in results] == ["s1"]

    def test_tag_matches_rank_above_body_matches(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(
            session_config,
            "a",
            session_id="body-match",
            summary="Some auth work happened.",
        )
        _write_session(
            session_config,
            "b",
            session_id="tag-match",
            summary="General cleanup.",
            tags=["auth"],
        )
        results = search_session_summaries("auth", session_config)
        assert [r.session_id for r in results] == ["tag-match", "body-match"]

    def test_empty_query_lists_most_recent_first(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(
            session_config,
            "old",
            session_id="old",
            summary="old work",
            end_time="2026-01-01T00:00:00+00:00",
        )
        _write_session(
            session_config,
            "new",
            session_id="new",
            summary="new work",
            end_time="2026-07-01T00:00:00+00:00",
        )
        results = search_session_summaries("", session_config)
        assert [r.session_id for r in results] == ["new", "old"]

    def test_excludes_current_session(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(session_config, "a", session_id="current", summary="work")
        results = search_session_summaries(
            "", session_config, exclude_session_id="current"
        )
        assert results == []

    def test_cwd_filter(self, session_config: SessionLoggingConfig) -> None:
        _write_session(
            session_config, "a", session_id="here", summary="work", cwd="/repo"
        )
        _write_session(
            session_config, "b", session_id="there", summary="work", cwd="/other"
        )
        results = search_session_summaries("", session_config, cwd="/repo")
        assert [r.session_id for r in results] == ["here"]

    def test_corrupt_metadata_is_skipped(
        self, session_config: SessionLoggingConfig
    ) -> None:
        session_dir = Path(session_config.save_dir) / "test_corrupt"
        session_dir.mkdir(parents=True)
        (session_dir / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
        assert search_session_summaries("", session_config) == []


class TestRenderRecallContext:
    def test_includes_summary_title_and_tags(
        self, session_config: SessionLoggingConfig
    ) -> None:
        _write_session(
            session_config,
            "a",
            session_id="s1",
            title="Auth fix",
            summary="Fixed the token refresh.",
            tags=["auth", "bugfix"],
        )
        result = search_session_summaries("", session_config)[0]
        rendered = render_recall_context(result)
        assert "Auth fix" in rendered
        assert "Fixed the token refresh." in rendered
        assert "auth, bugfix" in rendered
        assert "<recalled_session_summary>" in rendered
