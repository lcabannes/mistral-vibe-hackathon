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
    LocalTeamClient,
    PrivacyMode,
    TeamWorkspaceIdentity,
)
from vibe.core.team_workspace.file_store import SharedTeamWorkspaceStore
from vibe.core.team_workspace.service import (
    TeamWorkspaceService,
    build_team_workspace_service,
)

NOW = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
IDENTITY = TeamWorkspaceIdentity(
    workspace_id="ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    project_fingerprint="b" * 64,
    display_name="Shared project",
)


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _service(
    root: Path,
    *,
    member: str,
    client: str,
    privacy: PrivacyMode = PrivacyMode.STATUS,
    history: HistoryScope = HistoryScope.STATUS,
    clock: MutableClock | None = None,
) -> TeamWorkspaceService:
    member_id = f"member_{member * 32}"
    client_id = f"client_{client * 32}"
    store = SharedTeamWorkspaceStore(
        shared_root=root,
        identity=IDENTITY,
        privacy_mode=privacy,
        member_id=member_id,
        client_id=client_id,
        presence_ttl_seconds=30,
        history_scope=history,
    )
    return TeamWorkspaceService(
        identity=IDENTITY,
        privacy_mode=privacy,
        history_scope=history,
        client=LocalTeamClient(
            member_id=member_id,
            member_display_name=f"Member {member.upper()}",
            client_id=client_id,
            branch="feature/team",
        ),
        heartbeat_interval_seconds=60,
        store=store,
        clock=clock or MutableClock(),
    )


@pytest.mark.asyncio
async def test_disabled_service_creates_no_shared_directory(tmp_path: Path) -> None:
    shared_root = tmp_path / "must-not-exist"
    service = build_team_workspace_service(
        enabled=False,
        shared_root=shared_root,
        project_root=tmp_path,
        identity_hint="member@example.com",
    )

    await service.start()
    await service.publish_activity(
        local_run_id="run",
        agent_name="default",
        agent_display_name="Default",
        state=ActivityState.RUNNING,
    )

    assert service.snapshot.connection_state is ConnectionState.DISABLED
    assert not shared_root.exists()


@pytest.mark.asyncio
async def test_two_services_refresh_each_others_activity(tmp_path: Path) -> None:
    first = _service(tmp_path, member="a", client="a")
    second = _service(tmp_path, member="b", client="b")
    try:
        await first.start()
        await second.start()
        await first.publish_activity(
            local_run_id="primary:first",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
            summary=ActivitySummary.USING_TOOL,
        )

        snapshot = await second.refresh()

        assert len(snapshot.members) == 2
        assert len(snapshot.runs) == 1
        assert snapshot.runs[0].state is ActivityState.WORKING
        assert snapshot.runs[0].summary is None
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_summaries_mode_publishes_only_fixed_summary_enum(tmp_path: Path) -> None:
    service = _service(tmp_path, member="a", client="a", privacy=PrivacyMode.SUMMARIES)
    try:
        await service.start()
        snapshot = await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
            summary=ActivitySummary.WAITING_FOR_APPROVAL,
        )

        assert snapshot.runs[0].summary is ActivitySummary.WAITING_FOR_APPROVAL
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_duplicate_intermediate_state_is_coalesced(tmp_path: Path) -> None:
    service = _service(tmp_path, member="a", client="a")
    try:
        await service.start()
        await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        store = service._store
        assert store is not None
        event_paths = list((store.workspace_dir / "events").rglob("*.json"))
        await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )

        assert list((store.workspace_dir / "events").rglob("*.json")) == event_paths
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_bad_shared_root_degrades_without_raising(tmp_path: Path) -> None:
    shared_root = tmp_path / "not-a-directory"
    shared_root.write_text("occupied", encoding="utf-8")
    service = _service(shared_root, member="a", client="a")
    try:
        snapshot = await service.start()
        assert snapshot.connection_state is ConnectionState.DEGRADED
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_listener_receives_immutable_refreshes(tmp_path: Path) -> None:
    service = _service(tmp_path, member="a", client="a")
    snapshots = []
    service.add_listener(snapshots.append)
    try:
        await service.start()
        await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.RUNNING,
        )

        assert snapshots
        assert snapshots[-1].runs[0].state is ActivityState.RUNNING
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_all_shared_filesystem_io_is_offloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core.team_workspace import service as service_module

    calls: list[str] = []

    async def fake_to_thread(function, *args):
        calls.append(function.__name__)
        return function(*args)

    monkeypatch.setattr(service_module.asyncio, "to_thread", fake_to_thread)
    service = _service(tmp_path, member="a", client="a")
    try:
        await service.start()
        await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.RUNNING,
        )
        await service.refresh()
    finally:
        await service.stop()

    assert calls == [
        "initialize",
        "write_presence",
        "read_snapshot",
        "write_event",
        "read_snapshot",
        "read_snapshot",
    ]


@pytest.mark.asyncio
async def test_expired_presence_after_clock_advance(tmp_path: Path) -> None:
    clock = MutableClock()
    service = _service(tmp_path, member="a", client="a", clock=clock)
    try:
        await service.start()
        clock.now += timedelta(seconds=31)
        snapshot = await service.refresh()

        assert snapshot.members[0].presence.value == "offline"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_publishes_redacted_message_history(tmp_path: Path) -> None:
    service = _service(tmp_path, member="a", client="a", history=HistoryScope.MESSAGES)
    try:
        await service.start()
        await service.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        snapshot = await service.publish_conversation(
            local_run_id="primary",
            role=ConversationRole.USER,
            text="Read /Users/ada/private.txt with API_KEY=secret",
        )

        entry = snapshot.runs[0].history[0]
        assert entry.role is ConversationRole.USER
        assert "ada" not in (entry.text or "")
        assert "secret" not in (entry.text or "")
    finally:
        await service.stop()
