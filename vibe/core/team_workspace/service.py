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


class _TeamWorkspaceLocallyDisabled(Exception):
    pass


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
    respect_local_leave: bool
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
    respect_local_leave: bool
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
            respect_local_leave=options.get("respect_local_leave", True),
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
        runtime_enabled_check: Callable[[], bool] | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self._configured_enabled = enabled
        self._locally_disabled = False
        self._runtime_enabled_check = runtime_enabled_check
        from vibe.core.config.team_metadata import team_workspace_lock_for_id

        self._publication_guard = lambda: team_workspace_lock_for_id(
            identity.workspace_id
        )
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
    def enabled(self) -> bool:
        return self._configured_enabled and not self._locally_disabled

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
        if self._started or not self._configured_enabled:
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
        if not self._configured_enabled or self._store is None:
            return self._snapshot
        async with self._io_lock:
            if not await self._refresh_runtime_enabled():
                return self._snapshot
            sync_result = await self._sync_transport()
            if sync_result is None:
                return self._snapshot
            transport_failed = not sync_result
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
        store = self._store
        if not self._configured_enabled or store is None:
            return self._snapshot
        if not self._started:
            await self.start()
        safe_summary = summary if self.privacy_mode is PrivacyMode.SUMMARIES else None
        signature = (agent_name, agent_display_name, state, safe_summary)
        run_id = derive_run_id(self.identity.workspace_id, self.member_id, local_run_id)
        async with self._io_lock:
            if not await self._refresh_runtime_enabled():
                return self._snapshot
            if self._last_activity.get(run_id) == signature:
                return self._snapshot
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
            sync_result: bool | None = True
            transport_failed = False
            try:
                if self._transport is None:
                    await asyncio.to_thread(
                        self._write_local_publication, lambda: store.write_event(event)
                    )
                else:
                    await asyncio.to_thread(store.write_event, event)
                self._last_activity[run_id] = signature
                sync_result = await self._sync_transport()
                if sync_result is None:
                    snapshot = self._snapshot
                else:
                    transport_failed = not sync_result
                    snapshot = await asyncio.to_thread(
                        store.read_snapshot, self._clock()
                    )
            except _TeamWorkspaceLocallyDisabled:
                self._mark_locally_disabled()
                snapshot = self._snapshot
            except (OSError, TeamWorkspaceStoreError):
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
            else:
                if sync_result is not None and transport_failed:
                    snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)
            return snapshot

    async def publish_conversation(
        self, *, local_run_id: str, role: ConversationRole, text: str
    ) -> TeamWorkspaceSnapshot:
        store = self._store
        if (
            not self._configured_enabled
            or store is None
            or self.history_scope is HistoryScope.STATUS
        ):
            return self._snapshot
        if not self._started:
            await self.start()
        async with self._io_lock:
            if not await self._refresh_runtime_enabled():
                return self._snapshot
            transport = self._transport
            if transport is not None:
                sync_result = await self._sync_transport()
                if sync_result is None:
                    return self._snapshot
                if not sync_result:
                    snapshot = self._empty_snapshot(
                        ConnectionState.DEGRADED, SyncError.TRANSPORT_FAILED
                    )
                    self._replace_snapshot(snapshot)
                    return snapshot

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
                if transport is None:
                    await asyncio.to_thread(
                        self._write_local_publication,
                        lambda: store.write_conversation(entry),
                    )
                else:
                    await asyncio.to_thread(
                        transport.publish_sensitive,
                        validate_policy=self._validate_conversation_publication,
                        write_sensitive=lambda: store.write_conversation(entry),
                    )
                snapshot = await asyncio.to_thread(store.read_snapshot, self._clock())
            except _TeamWorkspaceLocallyDisabled:
                self._mark_locally_disabled()
                snapshot = self._snapshot
            except TeamWorkspaceStoreError as error:
                snapshot = self._empty_snapshot(ConnectionState.DEGRADED, error.code)
            except GitTeamWorkspaceError:
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.TRANSPORT_FAILED
                )
            except OSError:
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
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
        store = self._store
        if not self._configured_enabled or store is None:
            return
        async with self._io_lock:
            if not await self._refresh_runtime_enabled():
                return
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
            sync_result: bool | None = True
            transport_failed = False
            try:
                if self._transport is None:
                    await asyncio.to_thread(
                        self._write_local_publication,
                        lambda: self._initialize_and_write_presence(
                            store, now, presence
                        ),
                    )
                else:
                    await asyncio.to_thread(self._transport.hydrate)
                    needs_tightening = await asyncio.to_thread(
                        store.needs_history_scope_tightening
                    )
                    if needs_tightening:
                        await asyncio.to_thread(
                            self._transport.publish_policy_tightening,
                            apply_policy=lambda: store.initialize(now),
                            validate_publication=self._validate_runtime_publication,
                        )
                    await asyncio.to_thread(store.initialize, now)
                    await asyncio.to_thread(store.write_presence, presence)
                sync_result = await self._sync_transport()
                if sync_result is None:
                    snapshot = self._snapshot
                else:
                    transport_failed = not sync_result
                    snapshot = await asyncio.to_thread(store.read_snapshot, now)
            except _TeamWorkspaceLocallyDisabled:
                self._mark_locally_disabled()
                snapshot = self._snapshot
            except TeamWorkspaceStoreError as error:
                snapshot = self._empty_snapshot(ConnectionState.DEGRADED, error.code)
            except GitTeamWorkspaceError:
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.TRANSPORT_FAILED
                )
            except OSError:
                snapshot = self._empty_snapshot(
                    ConnectionState.DEGRADED, SyncError.WRITE_FAILED
                )
            else:
                if sync_result is not None and transport_failed:
                    snapshot = self._with_transport_error(snapshot)
            self._replace_snapshot(snapshot)

    async def _refresh_runtime_enabled(self) -> bool:
        runtime_enabled = True
        if self._runtime_enabled_check is not None:
            try:
                runtime_enabled = await asyncio.to_thread(self._runtime_enabled_check)
            except OSError:
                runtime_enabled = False
        if runtime_enabled:
            if self._locally_disabled:
                self._locally_disabled = False
                self._replace_snapshot(
                    self._empty_snapshot(ConnectionState.DISCONNECTED)
                )
            return True
        self._mark_locally_disabled()
        return False

    def _validate_conversation_publication(self) -> None:
        self._validate_runtime_publication()
        store = self._store
        if store is None:
            raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED)
        store.validate_conversation_policy()

    def _validate_runtime_publication(self) -> None:
        if self._runtime_enabled_check is not None:
            try:
                runtime_enabled = self._runtime_enabled_check()
            except OSError as error:
                raise _TeamWorkspaceLocallyDisabled from error
            if not runtime_enabled:
                raise _TeamWorkspaceLocallyDisabled

    def _write_local_publication(self, write: Callable[[], None]) -> None:
        with self._publication_guard():
            self._validate_runtime_publication()
            write()

    @staticmethod
    def _initialize_and_write_presence(
        store: SharedTeamWorkspaceStore, now: datetime, presence: TeamMemberPresence
    ) -> None:
        store.initialize(now)
        store.write_presence(presence)

    def _mark_locally_disabled(self) -> None:
        if self._locally_disabled:
            return
        self._locally_disabled = True
        self._last_activity.clear()
        self._replace_snapshot(self._empty_snapshot(ConnectionState.DISABLED))

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

    async def _sync_transport(self) -> bool | None:
        if self._transport is None:
            return True
        try:
            await asyncio.to_thread(
                self._transport.sync,
                validate_publication=self._validate_runtime_publication,
            )
            return True
        except _TeamWorkspaceLocallyDisabled:
            self._mark_locally_disabled()
            return None
        except (GitTeamWorkspaceError, OSError):
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
    from vibe.core.config.team_metadata import team_workspace_lock_for_id

    publication_guard = lambda: team_workspace_lock_for_id(identity.workspace_id)
    runtime_enabled_check: Callable[[], bool] | None = None
    if settings.respect_local_leave:
        from vibe.core.config.team_metadata import is_team_workspace_left_id

        runtime_enabled_check = lambda: (
            not is_team_workspace_left_id(identity.workspace_id)
        )
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
                publication_guard=publication_guard,
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
        runtime_enabled_check=runtime_enabled_check,
        clock=settings.clock,
    )
