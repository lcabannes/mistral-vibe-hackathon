from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, Unpack

from vibe.core.team_workspace.file_store import (
    SharedTeamWorkspaceStore,
    TeamWorkspaceStoreError,
)
from vibe.core.team_workspace.git_transport import (
    GitTeamWorkspaceError,
    GitTeamWorkspaceTransport,
)
from vibe.core.team_workspace.identity import (
    derive_entry_id,
    derive_event_id,
    derive_run_id,
    discover_current_branch,
    discover_workspace_identity,
    new_client_id,
    resolve_member_identity,
    resolve_team_repository_url,
)
from vibe.core.team_workspace.models import (
    ActivityState,
    ActivitySummary,
    ConnectionState,
    ConversationRole,
    HistoryScope,
    PrivacyMode,
    SyncError,
    TeamActivityEvent,
    TeamConversationEntry,
    TeamMemberPresence,
    TeamWorkspaceIdentity,
    TeamWorkspaceSnapshot,
)

type TeamWorkspaceListener = Callable[[TeamWorkspaceSnapshot], None]
type Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class LocalTeamClient:
    member_id: str
    member_display_name: str
    client_id: str
    branch: str | None


class TeamWorkspaceBuildOptions(TypedDict, total=False):
    member_name: str
    privacy_mode: PrivacyMode
    history_scope: HistoryScope
    history_limit: int
    heartbeat_interval_seconds: float
    presence_ttl_seconds: float
    identity_hint: str
    team_repository_url: str
    team_branch: str
    cache_root: Path | None
    clock: Clock


@dataclass(frozen=True, slots=True)
class _ResolvedBuildOptions:
    member_name: str
    privacy_mode: PrivacyMode
    history_scope: HistoryScope
    history_limit: int
    heartbeat_interval_seconds: float
    presence_ttl_seconds: float
    identity_hint: str
    team_repository_url: str
    team_branch: str
    cache_root: Path | None
    clock: Clock

    @classmethod
    def resolve(cls, options: TeamWorkspaceBuildOptions) -> _ResolvedBuildOptions:
        return cls(
            member_name=options.get("member_name", ""),
            privacy_mode=options.get("privacy_mode", PrivacyMode.STATUS),
            history_scope=options.get("history_scope", HistoryScope.STATUS),
            history_limit=options.get("history_limit", 50),
            heartbeat_interval_seconds=options.get("heartbeat_interval_seconds", 5.0),
            presence_ttl_seconds=options.get("presence_ttl_seconds", 30.0),
            identity_hint=options.get("identity_hint", ""),
            team_repository_url=options.get("team_repository_url", ""),
            team_branch=options.get("team_branch", "vibe-team-demo"),
            cache_root=options.get("cache_root"),
            clock=options.get("clock", _utc_now),
        )


class TeamWorkspaceService:
    def __init__(
        self,
        *,
        identity: TeamWorkspaceIdentity,
        privacy_mode: PrivacyMode,
        history_scope: HistoryScope,
        client: LocalTeamClient,
        heartbeat_interval_seconds: float,
        store: SharedTeamWorkspaceStore | None,
        transport: GitTeamWorkspaceTransport | None = None,
        enabled: bool = True,
        clock: Clock = _utc_now,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.enabled = enabled
        self.identity = identity
        self.privacy_mode = privacy_mode
        self.history_scope = history_scope
        self.member_id = client.member_id
        self.member_display_name = client.member_display_name
        self.client_id = client.client_id
        self.branch = client.branch
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._store = store
        self._transport = transport
        self._clock = clock
        self._listeners: list[TeamWorkspaceListener] = []
        self._io_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._poll_task: asyncio.Task[None] | None = None
        self._started = False
        self._presence_revision = 0
        self._sequence = 0
        self._last_activity: dict[
            str, tuple[str, str, ActivityState, ActivitySummary | None]
        ] = {}
        self._snapshot = self._empty_snapshot(
            ConnectionState.DISCONNECTED if self.enabled else ConnectionState.DISABLED
        )

    @property
    def snapshot(self) -> TeamWorkspaceSnapshot:
        return self._snapshot

    def add_listener(self, listener: TeamWorkspaceListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TeamWorkspaceListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def start(self) -> TeamWorkspaceSnapshot:
        if self._started or not self.enabled:
            return self._snapshot
        if self._store is None:
            self._replace_snapshot(
                self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.TRANSPORT_FAILED
                )
            )
            return self._snapshot
        self._started = True
        self._stop_event.clear()
        await self._heartbeat_and_refresh()
        self._poll_task = asyncio.create_task(
            self._poll(), name=f"team-workspace-{self.client_id}"
        )
        return self._snapshot

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        task = self._poll_task
        self._poll_task = None
        if task is None:
            return
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def refresh(self) -> TeamWorkspaceSnapshot:
        if not self.enabled or self._store is None:
            return self._snapshot
        async with self._io_lock:
            transport_failed = not await self._sync_transport()
            try:
                snapshot = await asyncio.to_thread(
                    self._store.read_snapshot, self._clock()
                )
            except (OSError, TeamWorkspaceStoreError):
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.READ_FAILED
                )
            if transport_failed:
                snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)
            return snapshot

    async def publish_activity(
        self,
        *,
        local_run_id: str,
        agent_name: str,
        agent_display_name: str,
        state: ActivityState,
        summary: ActivitySummary | None = None,
    ) -> TeamWorkspaceSnapshot:
        if not self.enabled or self._store is None:
            return self._snapshot
        if not self._started:
            await self.start()
        safe_summary = summary if self.privacy_mode is PrivacyMode.SUMMARIES else None
        signature = (agent_name, agent_display_name, state, safe_summary)
        run_id = derive_run_id(self.identity.workspace_id, self.member_id, local_run_id)
        if self._last_activity.get(run_id) == signature:
            return self._snapshot

        async with self._io_lock:
            self._sequence += 1
            event = TeamActivityEvent(
                workspace_id=self.identity.workspace_id,
                event_id=derive_event_id(self.client_id, self._sequence, run_id),
                member_id=self.member_id,
                member_display_name=self.member_display_name,
                client_id=self.client_id,
                sequence=self._sequence,
                run_id=run_id,
                agent_name=agent_name,
                agent_display_name=agent_display_name,
                state=state,
                privacy_mode=self.privacy_mode,
                summary=safe_summary,
                occurred_at=self._clock(),
            )
            try:
                await asyncio.to_thread(self._store.write_event, event)
                self._last_activity[run_id] = signature
                transport_failed = not await self._sync_transport()
                snapshot = await asyncio.to_thread(
                    self._store.read_snapshot, self._clock()
                )
            except (OSError, TeamWorkspaceStoreError):
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
            else:
                if transport_failed:
                    snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)
            return snapshot

    async def publish_conversation(
        self, *, local_run_id: str, role: ConversationRole, text: str
    ) -> TeamWorkspaceSnapshot:
        if (
            not self.enabled
            or self._store is None
            or self.history_scope is HistoryScope.STATUS
        ):
            return self._snapshot
        if not self._started:
            await self.start()
        async with self._io_lock:
            self._sequence += 1
            run_id = derive_run_id(
                self.identity.workspace_id, self.member_id, local_run_id
            )
            entry = TeamConversationEntry(
                workspace_id=self.identity.workspace_id,
                entry_id=derive_entry_id(self.client_id, self._sequence, run_id),
                member_id=self.member_id,
                client_id=self.client_id,
                sequence=self._sequence,
                run_id=run_id,
                role=role,
                history_scope=self.history_scope,
                text=text if self.history_scope is HistoryScope.MESSAGES else None,
                occurred_at=self._clock(),
            )
            try:
                await asyncio.to_thread(self._store.write_conversation, entry)
                transport_failed = not await self._sync_transport()
                snapshot = await asyncio.to_thread(
                    self._store.read_snapshot, self._clock()
                )
            except (OSError, TeamWorkspaceStoreError):
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
            else:
                if transport_failed:
                    snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)
            return snapshot

    async def _poll(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.heartbeat_interval_seconds
                )
                return
            except TimeoutError:
                await self._heartbeat_and_refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._replace_snapshot(
                    self._empty_snapshot(
                        ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                    )
                )

    async def _heartbeat_and_refresh(self) -> None:
        if not self.enabled or self._store is None:
            return
        async with self._io_lock:
            self._presence_revision += 1
            now = self._clock()
            presence = TeamMemberPresence(
                workspace_id=self.identity.workspace_id,
                member_id=self.member_id,
                member_display_name=self.member_display_name,
                client_id=self.client_id,
                branch=self.branch,
                revision=self._presence_revision,
                last_seen_at=now,
            )
            try:
                if self._transport is not None:
                    await asyncio.to_thread(self._transport.prepare)
                await asyncio.to_thread(self._store.initialize, now)
                await asyncio.to_thread(self._store.write_presence, presence)
                transport_failed = not await self._sync_transport()
                snapshot = await asyncio.to_thread(self._store.read_snapshot, now)
            except TeamWorkspaceStoreError as error:
                snapshot = self._empty_snapshot(ConnectionState.DEGRADED, error.code)
            except OSError:
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
            else:
                if transport_failed:
                    snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)

    def _replace_snapshot(self, snapshot: TeamWorkspaceSnapshot) -> None:
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        for listener in tuple(self._listeners):
            with suppress(Exception):
                listener(snapshot)

    def _empty_snapshot(
        self, state: ConnectionState, error: SyncError | None = None
    ) -> TeamWorkspaceSnapshot:
        return TeamWorkspaceSnapshot(
            identity=self.identity,
            privacy_mode=self.privacy_mode,
            history_scope=self.history_scope,
            connection_state=state,
            generated_at=self._clock(),
            error=error,
        )

    async def _sync_transport(self) -> bool:
        if self._transport is None:
            return True
        try:
            await asyncio.to_thread(self._transport.sync)
            return True
        except GitTeamWorkspaceError:
            return False

    @staticmethod
    def _with_transport_error(snapshot: TeamWorkspaceSnapshot) -> TeamWorkspaceSnapshot:
        return snapshot.model_copy(
            update={
                "connection_state": ConnectionState.DEGRADED,
                "error": SyncError.TRANSPORT_FAILED,
            }
        )


def build_team_workspace_service(
    *,
    enabled: bool,
    shared_root: Path | None,
    project_root: Path,
    **options: Unpack[TeamWorkspaceBuildOptions],
) -> TeamWorkspaceService:
    settings = _ResolvedBuildOptions.resolve(options)
    identity = discover_workspace_identity(project_root)
    member_id, display_name = resolve_member_identity(
        identity.workspace_id, settings.member_name, settings.identity_hint
    )
    client_id = new_client_id()
    resolved_team_remote = resolve_team_repository_url(
        project_root, settings.team_repository_url
    )
    transport: GitTeamWorkspaceTransport | None = None
    materialization_root = shared_root
    if enabled and settings.team_repository_url:
        if resolved_team_remote is not None:
            from vibe.core.paths import VIBE_HOME

            base = (
                settings.cache_root or shared_root or VIBE_HOME.path / "team-workspaces"
            )
            transport = GitTeamWorkspaceTransport(
                remote_url=resolved_team_remote,
                checkout_dir=base / identity.workspace_id / client_id / "repo",
                branch=settings.team_branch,
            )
            materialization_root = transport.materialization_root
        else:
            materialization_root = None
    configured = enabled and (
        bool(settings.team_repository_url) or shared_root is not None
    )
    store = (
        SharedTeamWorkspaceStore(
            shared_root=materialization_root,
            identity=identity,
            privacy_mode=settings.privacy_mode,
            history_scope=settings.history_scope,
            history_limit=settings.history_limit,
            member_id=member_id,
            client_id=client_id,
            presence_ttl_seconds=settings.presence_ttl_seconds,
        )
        if configured and materialization_root is not None
        else None
    )
    return TeamWorkspaceService(
        identity=identity,
        privacy_mode=settings.privacy_mode,
        history_scope=settings.history_scope,
        client=LocalTeamClient(
            member_id=member_id,
            member_display_name=display_name,
            client_id=client_id,
            branch=discover_current_branch(project_root),
        ),
        heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
        store=store,
        transport=transport,
        enabled=configured,
        clock=settings.clock,
    )
