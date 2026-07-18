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
    max_event_files: int = 2_000,
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
        max_event_files=max_event_files,
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
        event_id=event_id
        or f"event_{client_id.removeprefix('client_')[:16]}{sequence:016x}",
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


def test_restarted_client_can_advance_same_run_with_lower_sequence(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path, MEMBER_A, CLIENT_A)
    restarted = _store(tmp_path, MEMBER_A, CLIENT_B)
    first.initialize(NOW)
    first.write_event(
        _event(
            sequence=9,
            state=ActivityState.WORKING,
            event_id=f"event_{'d' * 32}",
            occurred_at=NOW + timedelta(seconds=9),
        )
    )
    restarted.write_event(
        _event(
            sequence=1,
            state=ActivityState.ATTENTION,
            event_id=f"event_{'d' * 32}",
            client_id=CLIENT_B,
            occurred_at=NOW + timedelta(seconds=10),
        )
    )

    snapshot = _store(tmp_path).read_snapshot(NOW + timedelta(seconds=11))

    assert len(snapshot.runs) == 1
    assert snapshot.runs[0].client_id == CLIENT_B
    assert snapshot.runs[0].state is ActivityState.ATTENTION


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


def test_event_cap_selects_latest_state_without_starving_other_clients(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path, MEMBER_A, CLIENT_A)
    second = _store(tmp_path, MEMBER_B, CLIENT_B)
    first.initialize(NOW)
    for sequence in range(1, 5):
        first.write_event(_event(sequence=sequence, state=ActivityState.WORKING))
    first.write_event(_event(sequence=5, state=ActivityState.ATTENTION))
    second.write_event(
        _event(
            sequence=1,
            state=ActivityState.RUNNING,
            member_id=MEMBER_B,
            client_id=CLIENT_B,
            run_id="run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
    )

    snapshot = _store(tmp_path, max_event_files=2).read_snapshot(
        NOW + timedelta(seconds=10)
    )

    assert snapshot.connection_state is ConnectionState.CONNECTED
    assert {run.client_id for run in snapshot.runs} == {CLIENT_A, CLIENT_B}
    latest = next(run for run in snapshot.runs if run.client_id == CLIENT_A)
    assert latest.sequence == 5
    assert latest.state is ActivityState.ATTENTION


def test_conversation_cap_selects_latest_entry_fairly_across_clients(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path, MEMBER_A, CLIENT_A, history_scope=HistoryScope.MESSAGES)
    second = _store(tmp_path, MEMBER_B, CLIENT_B, history_scope=HistoryScope.MESSAGES)
    first.initialize(NOW)
    first.write_event(_event(sequence=1, state=ActivityState.WORKING))
    second.write_event(
        _event(
            sequence=1,
            state=ActivityState.WORKING,
            member_id=MEMBER_B,
            client_id=CLIENT_B,
            run_id="run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
    )
    for sequence in range(2, 5):
        first.write_conversation(
            TeamConversationEntry(
                workspace_id=IDENTITY.workspace_id,
                entry_id=f"entry_{sequence:032x}",
                member_id=MEMBER_A,
                client_id=CLIENT_A,
                sequence=sequence,
                run_id=RUN_A,
                role=ConversationRole.USER,
                history_scope=HistoryScope.MESSAGES,
                text=f"first {sequence}",
                occurred_at=NOW + timedelta(seconds=sequence),
            )
        )
    second.write_conversation(
        TeamConversationEntry(
            workspace_id=IDENTITY.workspace_id,
            entry_id="entry_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            member_id=MEMBER_B,
            client_id=CLIENT_B,
            sequence=2,
            run_id="run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            role=ConversationRole.ASSISTANT,
            history_scope=HistoryScope.MESSAGES,
            text="second latest",
            occurred_at=NOW + timedelta(seconds=2),
        )
    )

    snapshot = _store(
        tmp_path, history_scope=HistoryScope.MESSAGES, max_event_files=2
    ).read_snapshot(NOW + timedelta(seconds=10))

    history = {run.client_id: run.history for run in snapshot.runs}
    assert [entry.sequence for entry in history[CLIENT_A]] == [4]
    assert [entry.sequence for entry in history[CLIENT_B]] == [2]


@pytest.mark.parametrize(
    "terminal", [ActivityState.COMPLETED, ActivityState.FAILED, ActivityState.CANCELLED]
)
def test_event_compaction_preserves_terminal_monotonicity(
    tmp_path: Path, terminal: ActivityState
) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A, max_event_files=1)
    store.initialize(NOW)
    store.write_event(_event(sequence=1, state=terminal))
    store.write_event(_event(sequence=2, state=ActivityState.WORKING))

    snapshot = store.read_snapshot(NOW + timedelta(seconds=3))

    assert snapshot.runs[0].state is terminal
    assert snapshot.runs[0].sequence == 1
    assert len(list((store.workspace_dir / "events").rglob("*.json"))) == 1


def test_compaction_and_restart_preserve_original_run_start_time(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, MEMBER_A, CLIENT_A, max_event_files=1)
    store.initialize(NOW)
    store.write_event(
        _event(
            sequence=1,
            state=ActivityState.RUNNING,
            occurred_at=NOW + timedelta(seconds=1),
        )
    )
    store.write_event(
        _event(
            sequence=2,
            state=ActivityState.WORKING,
            occurred_at=NOW + timedelta(seconds=2),
        )
    )
    store.write_event(
        _event(
            sequence=3,
            state=ActivityState.COMPLETED,
            occurred_at=NOW + timedelta(seconds=3),
        )
    )

    restarted = _store(tmp_path, max_event_files=1)
    snapshot = restarted.read_snapshot(NOW + timedelta(seconds=4))

    assert len(list((store.workspace_dir / "events").rglob("*.json"))) == 1
    assert snapshot.runs[0].state is ActivityState.COMPLETED
    assert snapshot.runs[0].started_at == NOW + timedelta(seconds=1)
    assert snapshot.runs[0].updated_at == NOW + timedelta(seconds=3)


def test_newest_sequence_one_client_survives_stream_overflow(tmp_path: Path) -> None:
    cap = 3
    first = _store(
        tmp_path,
        MEMBER_A,
        "client_00000000000000000000000000000001",
        max_event_files=cap,
    )
    first.initialize(NOW)
    newest_client = ""
    for index in range(1, cap + 2):
        client_id = f"client_{index:032x}"
        newest_client = client_id
        writer = _store(tmp_path, MEMBER_A, client_id, max_event_files=cap)
        writer.write_event(
            _event(
                sequence=1,
                state=ActivityState.WORKING,
                event_id=f"event_{index:032x}",
                client_id=client_id,
                run_id=f"run_{index:032x}",
                occurred_at=NOW + timedelta(seconds=index),
            )
        )

    snapshot = _store(tmp_path, max_event_files=cap).read_snapshot(
        NOW + timedelta(seconds=10)
    )

    assert newest_client in {run.client_id for run in snapshot.runs}
    assert len(list((first.workspace_dir / "events").rglob("*.json"))) == cap


def test_conversation_overflow_keeps_newest_sequence_one_client(tmp_path: Path) -> None:
    cap = 3
    first = _store(
        tmp_path,
        MEMBER_A,
        "client_00000000000000000000000000000001",
        history_scope=HistoryScope.MESSAGES,
        max_event_files=cap,
    )
    first.initialize(NOW)
    newest_client = ""
    for index in range(1, cap + 2):
        client_id = f"client_{index:032x}"
        newest_client = client_id
        writer = _store(
            tmp_path,
            MEMBER_A,
            client_id,
            history_scope=HistoryScope.MESSAGES,
            max_event_files=cap,
        )
        writer.write_event(
            _event(
                sequence=1,
                state=ActivityState.WORKING,
                event_id=f"event_{index:032x}",
                client_id=client_id,
                run_id=f"run_{index:032x}",
                occurred_at=NOW + timedelta(seconds=index),
            )
        )
        writer.write_conversation(
            TeamConversationEntry(
                workspace_id=IDENTITY.workspace_id,
                entry_id=f"entry_{index:032x}",
                member_id=MEMBER_A,
                client_id=client_id,
                sequence=1,
                run_id=f"run_{index:032x}",
                role=ConversationRole.USER,
                history_scope=HistoryScope.MESSAGES,
                text=f"client {index}",
                occurred_at=NOW + timedelta(seconds=index),
            )
        )

    snapshot = _store(
        tmp_path, history_scope=HistoryScope.MESSAGES, max_event_files=cap
    ).read_snapshot(NOW + timedelta(seconds=10))

    represented = {entry.client_id for run in snapshot.runs for entry in run.history}
    assert newest_client in represented
    assert len(list((first.workspace_dir / "conversations").rglob("*.json"))) == cap


@pytest.mark.parametrize("tightened", [HistoryScope.MARKERS, HistoryScope.STATUS])
def test_message_policy_can_tighten_without_manifest_mismatch(
    tmp_path: Path, tightened: HistoryScope
) -> None:
    messages = _store(tmp_path, MEMBER_A, CLIENT_A, history_scope=HistoryScope.MESSAGES)
    messages.initialize(NOW)
    messages.write_event(_event(sequence=1, state=ActivityState.WORKING))
    messages.write_conversation(
        TeamConversationEntry(
            workspace_id=IDENTITY.workspace_id,
            entry_id="entry_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            member_id=MEMBER_A,
            client_id=CLIENT_A,
            sequence=2,
            run_id=RUN_A,
            role=ConversationRole.USER,
            history_scope=HistoryScope.MESSAGES,
            text="previously shared message",
            occurred_at=NOW + timedelta(seconds=2),
        )
    )
    tightened_store = _store(tmp_path, MEMBER_A, CLIENT_A, history_scope=tightened)

    tightened_store.initialize(NOW + timedelta(seconds=3))
    snapshot = tightened_store.read_snapshot(NOW + timedelta(seconds=3))

    assert snapshot.connection_state is ConnectionState.CONNECTED
    assert snapshot.history_scope is tightened
    assert not list((messages.workspace_dir / "conversations").rglob("*.json"))
