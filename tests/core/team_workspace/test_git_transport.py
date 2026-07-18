from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import threading

from git import Repo
import pytest

from vibe.core.config.team_metadata import (
    clear_team_workspace_leave,
    leave_team_workspace,
)
from vibe.core.team_workspace import (
    ActivityState,
    ConnectionState,
    ConversationRole,
    HistoryScope,
    PrivacyMode,
    SyncError,
    resolve_team_repository_url,
)
from vibe.core.team_workspace.git_transport import (
    GitTeamWorkspaceError,
    GitTeamWorkspaceTransport,
)
from vibe.core.team_workspace.service import build_team_workspace_service

pytestmark = pytest.mark.timeout(30)


def _bare_remote(path: Path) -> Path:
    Repo.init(path, bare=True)
    return path


def _transport(remote: Path, checkout: Path) -> GitTeamWorkspaceTransport:
    return GitTeamWorkspaceTransport(
        remote_url=str(remote),
        checkout_dir=checkout,
        branch="vibe-team-demo",
        timeout_seconds=5,
    )


def _write_client_state(transport: GitTeamWorkspaceTransport, name: str) -> None:
    transport.prepare()
    path = transport.materialization_root / "workspace" / "clients" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "presence.json").write_text(f'{{"client":"{name}"}}', encoding="utf-8")


def _pause_publication_guard(
    transport: GitTeamWorkspaceTransport,
    monkeypatch: pytest.MonkeyPatch,
    *,
    call_number: int,
) -> tuple[threading.Event, threading.Event]:
    original_guard = transport._publication_guard
    ready = threading.Event()
    release = threading.Event()
    guard_calls = 0

    @contextmanager
    def barrier_guard():
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == call_number:
            ready.set()
            if not release.wait(timeout=20):
                raise TimeoutError("publication barrier was not released")
        with original_guard():
            yield

    monkeypatch.setattr(transport, "_publication_guard", barrier_guard)
    return ready, release


def test_two_clients_converge_through_real_bare_remote(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    first = _transport(remote, tmp_path / "first")
    second = _transport(remote, tmp_path / "second")
    _write_client_state(first, "first")
    _write_client_state(second, "second")

    first.sync()
    second.sync()
    first.sync()

    assert (
        first.materialization_root
        / "workspace"
        / "clients"
        / "second"
        / "presence.json"
    ).is_file()
    assert (
        second.materialization_root
        / "workspace"
        / "clients"
        / "first"
        / "presence.json"
    ).is_file()
    assert "vibe-team-demo" in Repo(remote).heads


def test_offline_commit_pushes_after_remote_returns(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    unavailable = tmp_path / "team-offline.git"
    first = _transport(remote, tmp_path / "first")
    second = _transport(remote, tmp_path / "second")
    _write_client_state(first, "first")
    first.sync()

    remote.rename(unavailable)
    _write_client_state(first, "offline-update")
    with pytest.raises(GitTeamWorkspaceError):
        first.sync()
    unavailable.rename(remote)

    first.sync()
    _write_client_state(second, "second")
    second.sync()

    assert (
        second.materialization_root
        / "workspace"
        / "clients"
        / "offline-update"
        / "presence.json"
    ).is_file()


def test_origin_sentinel_uses_separate_branch_without_touching_source_checkout(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "source.git")
    source = tmp_path / "source"
    repo = Repo.init(source, initial_branch="main")
    (source / "README.md").write_text("source\n", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial source commit")
    repo.create_remote("origin", str(remote))
    repo.remote("origin").push("main:main")
    source_head = repo.head.commit.hexsha

    resolved = resolve_team_repository_url(source, "origin")
    assert resolved == str(remote)
    assert resolved is not None
    transport = GitTeamWorkspaceTransport(
        remote_url=resolved,
        checkout_dir=tmp_path / "team-cache",
        branch="vibe-team-demo",
    )
    _write_client_state(transport, "demo")
    transport.sync()

    assert repo.active_branch.name == "main"
    assert repo.head.commit.hexsha == source_head
    assert not repo.is_dirty(untracked_files=True)
    assert {head.name for head in Repo(remote).heads} == {"main", "vibe-team-demo"}


@pytest.mark.asyncio
async def test_two_services_converge_activity_through_bare_git_remote(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    first = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="first@example.com",
        privacy_mode=PrivacyMode.SUMMARIES,
        history_scope=HistoryScope.MESSAGES,
    )
    second = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="second@example.com",
        privacy_mode=PrivacyMode.SUMMARIES,
        history_scope=HistoryScope.MESSAGES,
    )
    try:
        await first.start()
        await first.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        await second.start()

        first_snapshot = await first.refresh()
        second_snapshot = await second.refresh()

        assert len(first_snapshot.members) == 2
        assert len(second_snapshot.members) == 2
        assert first_snapshot.runs[0].state is ActivityState.WORKING
        assert second_snapshot.runs[0].state is ActivityState.WORKING
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_differently_named_source_clones_share_one_team_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source_repo = Repo.init(source, initial_branch="main")
    (source / "README.md").write_text("shared source\n", encoding="utf-8")
    source_repo.index.add(["README.md"])
    source_repo.index.commit("Initial source commit")
    source_remote = tmp_path / "project.git"
    Repo.clone_from(source, source_remote, bare=True)
    alice_clone = tmp_path / "alice-checkout"
    bob_clone = tmp_path / "bob-project"
    Repo.clone_from(source_remote, alice_clone)
    Repo.clone_from(source_remote, bob_clone)
    team_remote = _bare_remote(tmp_path / "team.git")

    first = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=alice_clone,
        team_repository_url=str(team_remote),
        cache_root=tmp_path / "cache",
        identity_hint="alice@example.com",
    )
    second = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=bob_clone,
        team_repository_url=str(team_remote),
        cache_root=tmp_path / "cache",
        identity_hint="bob@example.com",
    )
    try:
        assert first.identity == second.identity
        assert first.identity.display_name == "project"
        assert (await first.start()).connection_state.value == "connected"
        await first.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        assert (await second.start()).connection_state.value == "connected"

        first_snapshot = await first.refresh()
        second_snapshot = await second.refresh()

        assert first_snapshot.connection_state.value == "connected"
        assert second_snapshot.connection_state.value == "connected"
        assert len(first_snapshot.members) == 2
        assert len(second_snapshot.members) == 2
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_stale_message_client_cannot_publish_after_remote_policy_tightens(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    messages = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="messages@example.com",
        history_scope=HistoryScope.MESSAGES,
    )
    markers = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="markers@example.com",
        history_scope=HistoryScope.MARKERS,
    )
    try:
        await messages.start()
        await messages.publish_activity(
            local_run_id="messages-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        await messages.publish_conversation(
            local_run_id="messages-run",
            role=ConversationRole.USER,
            text="message that was already pushed",
        )
        assert (await markers.start()).connection_state.value == "connected"
        await markers.publish_activity(
            local_run_id="markers-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        snapshot = await markers.publish_conversation(
            local_run_id="markers-run",
            role=ConversationRole.USER,
            text="future marker only",
        )
        stale_snapshot = await messages.publish_conversation(
            local_run_id="messages-run",
            role=ConversationRole.ASSISTANT,
            text="raw text after remote revocation",
        )
        marker_snapshot = await markers.refresh()

        assert snapshot.connection_state.value == "connected"
        assert snapshot.history_scope is HistoryScope.MARKERS
        assert stale_snapshot.connection_state.value == "degraded"
        assert stale_snapshot.error is not None
        assert stale_snapshot.error.value == "manifest_mismatch"
        assert marker_snapshot.connection_state.value == "connected"
        marker_history = next(run.history for run in snapshot.runs if run.history)
        assert marker_history[0].text is None
        store = markers._store
        assert store is not None
        current_records = list((store.workspace_dir / "conversations").rglob("*.json"))
        assert len(current_records) == 1
        assert "already pushed" not in current_records[0].read_text(encoding="utf-8")
        audit = tmp_path / "remote-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert "raw text after remote revocation" not in remote_text
    finally:
        await messages.stop()
        await markers.stop()


@pytest.mark.asyncio
async def test_concurrent_policy_tightening_wins_sensitive_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    messages = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="messages@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    markers = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="markers@example.com",
        history_scope=HistoryScope.MARKERS,
        heartbeat_interval_seconds=60,
    )
    try:
        await messages.start()
        await messages.publish_activity(
            local_run_id="messages-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        store = messages._store
        transport = messages._transport
        assert store is not None
        assert transport is not None
        write_conversation = store.write_conversation
        loop = asyncio.get_running_loop()

        def write_after_preflight(entry) -> None:
            write_conversation(entry)
            future = asyncio.run_coroutine_threadsafe(markers.start(), loop)
            assert future.result(timeout=20).connection_state.value == "connected"

        monkeypatch.setattr(store, "write_conversation", write_after_preflight)

        stale = await messages.publish_conversation(
            local_run_id="messages-run",
            role=ConversationRole.USER,
            text="concurrent private message",
        )
        marker_snapshot = await markers.refresh()

        assert stale.connection_state.value == "degraded"
        assert stale.error is not None
        assert stale.error.value == "manifest_mismatch"
        assert marker_snapshot.connection_state.value == "connected"
        assert not list((store.workspace_dir / "conversations").rglob("*.json"))
        assert "concurrent private message" not in Repo(transport.checkout_dir).git.log(
            "-p", "--all"
        )
        audit = tmp_path / "concurrent-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert "concurrent private message" not in remote_text
    finally:
        await messages.stop()
        await markers.stop()


@pytest.mark.asyncio
async def test_policy_tightening_retries_after_sensitive_push_wins_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    messages = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="messages@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    markers = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="markers@example.com",
        history_scope=HistoryScope.MARKERS,
        heartbeat_interval_seconds=60,
    )
    try:
        assert (await messages.start()).connection_state.value == "connected"
        await messages.publish_activity(
            local_run_id="messages-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        marker_store = markers._store
        assert marker_store is not None
        initialize = marker_store.initialize
        loop = asyncio.get_running_loop()
        raced_snapshots = []

        def initialize_then_publish(now) -> None:
            initialize(now)
            if raced_snapshots:
                return
            future = asyncio.run_coroutine_threadsafe(
                messages.publish_conversation(
                    local_run_id="messages-run",
                    role=ConversationRole.USER,
                    text="message that wins the first remote push",
                ),
                loop,
            )
            raced_snapshots.append(future.result(timeout=20))

        monkeypatch.setattr(marker_store, "initialize", initialize_then_publish)

        marker_snapshot = await markers.start()

        assert raced_snapshots[0].connection_state.value == "connected"
        assert marker_snapshot.connection_state.value == "connected"
        assert marker_snapshot.history_scope is HistoryScope.MARKERS
        audit = tmp_path / "inverse-race-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert '"history_scope":"markers"' in remote_text
        assert "message that wins the first remote push" not in remote_text
    finally:
        await messages.stop()
        await markers.stop()


@pytest.mark.asyncio
async def test_leave_returning_before_sensitive_push_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="leaving@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    release = threading.Event()
    try:
        assert (await service.start()).connection_state.value == "connected"
        transport = service._transport
        assert transport is not None

        async def preflight_synced() -> bool:
            return True

        monkeypatch.setattr(service, "_sync_transport", preflight_synced)
        ready, release = _pause_publication_guard(transport, monkeypatch, call_number=1)
        run_result = transport._run_result
        leave_returned = threading.Event()
        pushes_after_leave = []

        def track_push(*args: str, cwd: Path):
            if args[0] == "push" and leave_returned.is_set():
                pushes_after_leave.append(args)
            return run_result(*args, cwd=cwd)

        monkeypatch.setattr(transport, "_run_result", track_push)
        publication = asyncio.create_task(
            service.publish_conversation(
                local_run_id="leave-race",
                role=ConversationRole.USER,
                text="must not cross a completed leave",
            )
        )
        assert await asyncio.to_thread(ready.wait, 15)

        assert await asyncio.to_thread(leave_team_workspace, project) is True
        leave_returned.set()
        release.set()
        snapshot = await publication

        assert snapshot.connection_state.value == "disabled"
        assert service.enabled is False
        assert not pushes_after_leave
        audit = tmp_path / "leave-race-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert "must not cross a completed leave" not in remote_text
    finally:
        release.set()
        clear_team_workspace_leave(project)
        await service.stop()


@pytest.mark.asyncio
async def test_leave_waits_for_sensitive_push_holding_publication_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="locked-push@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    allow_push = threading.Event()
    try:
        assert (await service.start()).connection_state.value == "connected"
        transport = service._transport
        assert transport is not None

        async def preflight_synced() -> bool:
            return True

        monkeypatch.setattr(service, "_sync_transport", preflight_synced)
        run_result = transport._run_result
        push_started = threading.Event()
        push_calls = 0

        def pause_sensitive_push(*args: str, cwd: Path):
            nonlocal push_calls
            if args[0] == "push":
                push_calls += 1
                if push_calls == 1:
                    push_started.set()
                    if not allow_push.wait(timeout=20):
                        raise TimeoutError("sensitive push barrier was not released")
            return run_result(*args, cwd=cwd)

        monkeypatch.setattr(transport, "_run_result", pause_sensitive_push)
        publication = asyncio.create_task(
            service.publish_conversation(
                local_run_id="locked-push",
                role=ConversationRole.USER,
                text="publication linearized before leave",
            )
        )
        assert await asyncio.to_thread(push_started.wait, 15)

        leave = asyncio.create_task(asyncio.to_thread(leave_team_workspace, project))
        await asyncio.sleep(0.1)
        assert not leave.done()

        allow_push.set()
        assert (await publication).connection_state.value == "connected"
        assert await leave is True
        disabled = await service.publish_activity(
            local_run_id="locked-push",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
        )
        assert disabled.connection_state.value == "disabled"
    finally:
        allow_push.set()
        clear_team_workspace_leave(project)
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["activity", "heartbeat"])
async def test_leave_returning_before_status_sync_prevents_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint=f"{operation}@example.com",
        heartbeat_interval_seconds=60,
    )
    release = threading.Event()
    try:
        assert (await service.start()).connection_state.value == "connected"
        transport = service._transport
        assert transport is not None
        ready, release = _pause_publication_guard(transport, monkeypatch, call_number=1)
        run_result = transport._run_result
        leave_returned = threading.Event()
        pushes_after_leave = []

        def track_push(*args: str, cwd: Path):
            if args[0] == "push" and leave_returned.is_set():
                pushes_after_leave.append(args)
            return run_result(*args, cwd=cwd)

        monkeypatch.setattr(transport, "_run_result", track_push)
        remote_head = Repo(remote).heads["vibe-team-demo"].commit.hexsha
        if operation == "activity":
            publication = asyncio.create_task(
                service.publish_activity(
                    local_run_id="leave-status-race",
                    agent_name="default",
                    agent_display_name="Default",
                    state=ActivityState.WORKING,
                )
            )
        else:
            publication = asyncio.create_task(service._heartbeat_and_refresh())
        assert await asyncio.to_thread(ready.wait, 15)

        assert await asyncio.to_thread(leave_team_workspace, project) is True
        leave_returned.set()
        release.set()
        await publication

        assert service.snapshot.connection_state.value == "disabled"
        assert service.enabled is False
        assert not pushes_after_leave
        assert Repo(remote).heads["vibe-team-demo"].commit.hexsha == remote_head
    finally:
        release.set()
        clear_team_workspace_leave(project)
        await service.stop()


@pytest.mark.asyncio
async def test_message_publication_fails_closed_while_status_stays_queued(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    offline_remote = tmp_path / "team-offline.git"
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="offline@example.com",
        history_scope=HistoryScope.MESSAGES,
    )
    try:
        assert (await service.start()).connection_state.value == "connected"
        remote.rename(offline_remote)
        queued = await service.publish_activity(
            local_run_id="offline-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
        )
        blocked = await service.publish_conversation(
            local_run_id="offline-run",
            role=ConversationRole.USER,
            text="must never be queued while policy is unverifiable",
        )

        assert queued.connection_state.value == "degraded"
        assert blocked.connection_state.value == "degraded"
        store = service._store
        assert store is not None
        assert not list((store.workspace_dir / "conversations").rglob("*.json"))

        offline_remote.rename(remote)
        recovered = await service.refresh()

        assert recovered.connection_state.value == "connected"
        assert recovered.runs[0].state is ActivityState.ATTENTION
        audit = tmp_path / "offline-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert "must never be queued" not in remote_text
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_publication_guard_oserror_degrades_and_status_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="lock-error@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    try:
        assert (await service.start()).connection_state is ConnectionState.CONNECTED
        transport = service._transport
        store = service._store
        assert transport is not None
        assert store is not None
        publication_guard = transport._publication_guard
        remote_head = Repo(remote).heads["vibe-team-demo"].commit.hexsha

        def unavailable_guard():
            raise OSError("publication lock unavailable")

        monkeypatch.setattr(transport, "_publication_guard", unavailable_guard)

        conversation = await service.publish_conversation(
            local_run_id="lock-error-run",
            role=ConversationRole.USER,
            text="must remain unpublished",
        )
        status = await service.publish_activity(
            local_run_id="lock-error-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
        )
        refreshed = await service.refresh()

        for snapshot in (conversation, status, refreshed):
            assert snapshot.connection_state is ConnectionState.DEGRADED
            assert snapshot.error is SyncError.TRANSPORT_FAILED
        assert service.snapshot is refreshed
        assert not list((store.workspace_dir / "conversations").rglob("*.json"))
        assert Repo(remote).heads["vibe-team-demo"].commit.hexsha == remote_head

        monkeypatch.setattr(transport, "_publication_guard", publication_guard)
        recovered = await service.refresh()

        assert recovered.connection_state is ConnectionState.CONNECTED
        assert recovered.error is None
        assert recovered.runs[0].state is ActivityState.ATTENTION
        assert Repo(remote).heads["vibe-team-demo"].commit.hexsha != remote_head
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_active_service_observes_leave_and_live_rejoin(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="active@example.com",
        history_scope=HistoryScope.MESSAGES,
        heartbeat_interval_seconds=60,
    )
    snapshots = []
    service.add_listener(snapshots.append)
    try:
        assert (await service.start()).connection_state.value == "connected"
        await service.publish_activity(
            local_run_id="active-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        remote_head_before_leave = Repo(remote).heads["vibe-team-demo"].commit.hexsha

        assert leave_team_workspace(project) is True
        blocked_message = await service.publish_conversation(
            local_run_id="active-run",
            role=ConversationRole.USER,
            text="message after local leave",
        )
        blocked_activity = await service.publish_activity(
            local_run_id="active-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
        )
        await service._heartbeat_and_refresh()

        assert blocked_message.connection_state.value == "disabled"
        assert blocked_activity.connection_state.value == "disabled"
        assert service.enabled is False
        assert snapshots[-1].connection_state.value == "disabled"
        assert (
            Repo(remote).heads["vibe-team-demo"].commit.hexsha
            == remote_head_before_leave
        )
        store = service._store
        assert store is not None
        assert not list((store.workspace_dir / "conversations").rglob("*.json"))

        assert clear_team_workspace_leave(project) is True
        rejoined = await service.refresh()
        assert rejoined.connection_state.value == "connected"
        assert service.enabled is True
        published = await service.publish_activity(
            local_run_id="active-run",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.ATTENTION,
        )
        assert published.connection_state.value == "connected"
        assert (
            Repo(remote).heads["vibe-team-demo"].commit.hexsha
            != remote_head_before_leave
        )
        audit = tmp_path / "leave-audit"
        Repo.clone_from(remote, audit, branch="vibe-team-demo")
        remote_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (audit / "state").rglob("*.json")
        )
        assert "message after local leave" not in remote_text
    finally:
        clear_team_workspace_leave(project)
        await service.stop()
