from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vibe.core.team_workspace import (
    ActivityState,
    ActivitySummary,
    ConnectionState,
    ConversationRole,
    HistoryScope,
    PresenceState,
    PrivacyMode,
    TeamActivityEvent,
    TeamConversationEntry,
    TeamMemberPresence,
    TeamWorkspaceIdentity,
)
from vibe.core.team_workspace.file_store import (
    SharedTeamWorkspaceStore,
    TeamWorkspaceStoreError,
)

NOW = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
IDENTITY = TeamWorkspaceIdentity(
    workspace_id="ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    project_fingerprint="b" * 64,
    display_name="Shared project",
)
MEMBER_A = "member_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MEMBER_B = "member_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
CLIENT_A = "client_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
CLIENT_B = "client_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
RUN_A = "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _store(
    root: Path,
    member_id: str | None = None,
    client_id: str | None = None,
    *,
    privacy_mode: PrivacyMode = PrivacyMode.SUMMARIES,
    ttl: float = 30,
    max_file_bytes: int = 64 * 1024,
    history_scope: HistoryScope = HistoryScope.STATUS,
    history_limit: int = 50,
) -> SharedTeamWorkspaceStore:
    return SharedTeamWorkspaceStore(
        shared_root=root,
        identity=IDENTITY,
        privacy_mode=privacy_mode,
        member_id=member_id,
        client_id=client_id,
        presence_ttl_seconds=ttl,
        max_file_bytes=max_file_bytes,
        history_scope=history_scope,
        history_limit=history_limit,
    )


def _presence(
    member_id: str,
    client_id: str,
    *,
    name: str,
    revision: int = 1,
    seen_at: datetime = NOW,
) -> TeamMemberPresence:
    return TeamMemberPresence(
        workspace_id=IDENTITY.workspace_id,
        member_id=member_id,
        member_display_name=name,
        client_id=client_id,
        branch="feature/team-workspace",
        revision=revision,
        last_seen_at=seen_at,
    )


def _event(
    *,
    sequence: int,
    state: ActivityState,
    event_id: str | None = None,
    member_id: str = MEMBER_A,
    client_id: str = CLIENT_A,
    run_id: str = RUN_A,
    occurred_at: datetime | None = None,
) -> TeamActivityEvent:
    return TeamActivityEvent(
        workspace_id=IDENTITY.workspace_id,
        event_id=event_id or f"event_{sequence:032x}",
        member_id=member_id,
        member_display_name="Ada" if member_id == MEMBER_A else "Grace",
        client_id=client_id,
        sequence=sequence,
        run_id=run_id,
        agent_name="default",
        agent_display_name="Default",
        state=state,
        privacy_mode=PrivacyMode.SUMMARIES,
        summary=ActivitySummary.USING_TOOL,
        occurred_at=occurred_at or NOW + timedelta(seconds=sequence),
    )


def test_two_clients_share_presence_and_sanitized_runs(tmp_path: Path) -> None:
    first = _store(tmp_path, MEMBER_A, CLIENT_A)
    second = _store(tmp_path, MEMBER_B, CLIENT_B)
    first.initialize(NOW)
    second.initialize(NOW)
    first.write_presence(_presence(MEMBER_A, CLIENT_A, name="Ada"))
    second.write_presence(_presence(MEMBER_B, CLIENT_B, name="Grace"))
    first.write_event(_event(sequence=1, state=ActivityState.WORKING))
    second.write_event(
        _event(
            sequence=1,
            state=ActivityState.ATTENTION,
            member_id=MEMBER_B,
            client_id=CLIENT_B,
            run_id="run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
    )

    snapshot = _store(tmp_path).read_snapshot(NOW + timedelta(seconds=2))

    assert snapshot.connection_state is ConnectionState.CONNECTED
    assert {member.display_name for member in snapshot.members} == {"Ada", "Grace"}
    assert {run.state for run in snapshot.runs} == {
        ActivityState.WORKING,
        ActivityState.ATTENTION,
    }
    assert all(member.presence is PresenceState.ONLINE for member in snapshot.members)


def test_atomic_writes_leave_no_temp_files(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A)
    store.initialize(NOW)
    store.write_presence(_presence(MEMBER_A, CLIENT_A, name="Ada"))
    store.write_event(_event(sequence=1, state=ActivityState.RUNNING))

    assert not list(store.workspace_dir.rglob("*.tmp"))
    assert store.manifest_path.is_file()
    assert len(list((store.workspace_dir / "events").rglob("*.json"))) == 1


def test_writer_cannot_write_another_clients_subtree(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A)
    store.initialize(NOW)

    with pytest.raises(TeamWorkspaceStoreError):
        store.write_presence(_presence(MEMBER_B, CLIENT_B, name="Grace"))
    with pytest.raises(TeamWorkspaceStoreError):
        store.write_event(
            _event(
                sequence=1,
                state=ActivityState.RUNNING,
                member_id=MEMBER_B,
                client_id=CLIENT_B,
            )
        )


@pytest.mark.parametrize("content", ["not json", '{"schema_version": 2}'])
def test_malformed_and_unsupported_events_are_ignored_and_degrade_snapshot(
    tmp_path: Path, content: str
) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A)
    store.initialize(NOW)
    event_dir = store.workspace_dir / "events" / MEMBER_A / CLIENT_A
    event_dir.mkdir(parents=True)
    (event_dir / "00000000000000000001-event_bbbbbbbb.json").write_text(
        content, encoding="utf-8"
    )

    snapshot = store.read_snapshot(NOW)

    assert snapshot.connection_state is ConnectionState.DEGRADED
    assert not snapshot.runs


def test_symlink_and_temp_event_files_are_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A)
    store.initialize(NOW)
    event_dir = store.workspace_dir / "events" / MEMBER_A / CLIENT_A
    event_dir.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text(_event(sequence=1, state=ActivityState.RUNNING).model_dump_json())
    (event_dir / "00000000000000000001-event_link.json").symlink_to(target)
    (event_dir / ".event.json.partial.tmp").write_text("partial", encoding="utf-8")

    snapshot = store.read_snapshot(NOW)

    assert snapshot.connection_state is ConnectionState.CONNECTED
    assert not snapshot.runs


def test_oversized_event_is_ignored_and_degrades_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A, max_file_bytes=100)
    store.initialize(NOW)
    event_dir = store.workspace_dir / "events" / MEMBER_A / CLIENT_A
    event_dir.mkdir(parents=True)
    (event_dir / "00000000000000000001-event_bbbbbbbb.json").write_text(
        "x" * 101, encoding="utf-8"
    )

    snapshot = store.read_snapshot(NOW)

    assert snapshot.connection_state is ConnectionState.DEGRADED
    assert not snapshot.runs


def test_stale_reopen_and_duplicate_event_are_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A)
    store.initialize(NOW)
    first_id = "event_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    terminal_id = "event_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    store.write_event(
        _event(sequence=1, state=ActivityState.RUNNING, event_id=first_id)
    )
    store.write_event(
        _event(sequence=2, state=ActivityState.COMPLETED, event_id=terminal_id)
    )
    store.write_event(_event(sequence=3, state=ActivityState.WORKING))
    store.write_event(
        _event(sequence=4, state=ActivityState.FAILED, event_id=terminal_id)
    )

    snapshot = store.read_snapshot(NOW + timedelta(seconds=10))

    assert len(snapshot.runs) == 1
    assert snapshot.runs[0].state is ActivityState.COMPLETED
    assert snapshot.runs[0].sequence == 2


def test_presence_expiry_is_derived_from_ttl(tmp_path: Path) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A, ttl=10)
    store.initialize(NOW)
    store.write_presence(_presence(MEMBER_A, CLIENT_A, name="Ada"))

    online = store.read_snapshot(NOW + timedelta(seconds=10))
    offline = store.read_snapshot(NOW + timedelta(seconds=11))

    assert online.members[0].presence is PresenceState.ONLINE
    assert offline.members[0].presence is PresenceState.OFFLINE


def test_message_history_is_redacted_and_bounded_per_run(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        MEMBER_A,
        CLIENT_A,
        history_scope=HistoryScope.MESSAGES,
        history_limit=2,
    )
    store.initialize(NOW)
    store.write_event(_event(sequence=1, state=ActivityState.WORKING))
    for sequence in range(2, 5):
        store.write_conversation(
            TeamConversationEntry(
                workspace_id=IDENTITY.workspace_id,
                entry_id=f"entry_{sequence:032x}",
                member_id=MEMBER_A,
                client_id=CLIENT_A,
                sequence=sequence,
                run_id=RUN_A,
                role=(
                    ConversationRole.USER
                    if sequence % 2 == 0
                    else ConversationRole.ASSISTANT
                ),
                history_scope=HistoryScope.MESSAGES,
                text=f"turn {sequence} API_KEY=secret-{sequence}",
                occurred_at=NOW + timedelta(seconds=sequence),
            )
        )

    snapshot = _store(
        tmp_path, history_scope=HistoryScope.MESSAGES, history_limit=2
    ).read_snapshot(NOW + timedelta(seconds=10))

    assert [entry.sequence for entry in snapshot.runs[0].history] == [3, 4]
    assert all("secret" not in (entry.text or "") for entry in snapshot.runs[0].history)
