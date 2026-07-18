from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import os
from pathlib import Path
import tempfile

from pydantic import BaseModel, ValidationError

from vibe.core.team_workspace.models import (
    MAX_HISTORY_PER_RUN,
    ActivityState,
    ConnectionState,
    HistoryScope,
    PresenceState,
    PrivacyMode,
    SyncError,
    TeamActivityEvent,
    TeamConversationEntry,
    TeamMemberPresence,
    TeamMemberSnapshot,
    TeamRunSnapshot,
    TeamWorkspaceIdentity,
    TeamWorkspaceManifest,
    TeamWorkspaceSnapshot,
)
from vibe.core.utils.io import read_safe

DEFAULT_MAX_FILE_BYTES = 64 * 1024
DEFAULT_MAX_EVENT_FILES = 2_000
_MANIFEST_CREATED_AT = datetime(1970, 1, 1, tzinfo=UTC)
_HISTORY_SCOPE_RANK = {
    HistoryScope.STATUS: 0,
    HistoryScope.MARKERS: 1,
    HistoryScope.MESSAGES: 2,
}


class TeamWorkspaceStoreError(Exception):
    def __init__(self, code: SyncError) -> None:
        super().__init__(code.value)
        self.code = code


class SharedTeamWorkspaceStore:
    def __init__(
        self,
        *,
        shared_root: Path,
        identity: TeamWorkspaceIdentity,
        privacy_mode: PrivacyMode,
        member_id: str | None = None,
        client_id: str | None = None,
        presence_ttl_seconds: float = 30.0,
        history_scope: HistoryScope = HistoryScope.STATUS,
        history_limit: int = 50,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_event_files: int = DEFAULT_MAX_EVENT_FILES,
    ) -> None:
        if presence_ttl_seconds <= 0:
            raise ValueError("presence_ttl_seconds must be positive")
        if (
            max_file_bytes < 1
            or max_event_files < 1
            or not 1 <= history_limit <= MAX_HISTORY_PER_RUN
        ):
            raise ValueError("file bounds must be positive")
        self.shared_root = shared_root.expanduser()
        self.identity = identity
        self.privacy_mode = privacy_mode
        self.member_id = member_id
        self.client_id = client_id
        self.presence_ttl_seconds = presence_ttl_seconds
        self.history_scope = history_scope
        self.history_limit = history_limit
        self.max_file_bytes = max_file_bytes
        self.max_event_files = max_event_files

    @property
    def workspace_dir(self) -> Path:
        return self.shared_root / self.identity.workspace_id

    @property
    def manifest_path(self) -> Path:
        return self.workspace_dir / "workspace.json"

    def initialize(self, now: datetime) -> None:
        self._ensure_directory(self.shared_root)
        self._ensure_directory(self.workspace_dir)
        manifest = TeamWorkspaceManifest(
            identity=self.identity,
            privacy_mode=self.privacy_mode,
            history_scope=self.history_scope,
            history_limit=self.history_limit,
            created_at=_MANIFEST_CREATED_AT,
        )
        if self.manifest_path.exists():
            existing = self._read_model(
                self.manifest_path, TeamWorkspaceManifest, required=True
            )
            if existing is not None and self._can_tighten_history_scope(existing):
                self._atomic_write(self.manifest_path, manifest)
                self._delete_record_tree(self.workspace_dir / "conversations")
                return
            if existing != manifest:
                raise TeamWorkspaceStoreError(SyncError.MANIFEST_MISMATCH)
            return
        self._atomic_write(self.manifest_path, manifest)

    def needs_history_scope_tightening(self) -> bool:
        if not self.manifest_path.exists():
            return False
        existing = self._read_model(
            self.manifest_path, TeamWorkspaceManifest, required=True
        )
        return existing is not None and self._can_tighten_history_scope(existing)

    def write_presence(self, presence: TeamMemberPresence) -> None:
        self._require_local_owner(presence.member_id, presence.client_id)
        path = (
            self.workspace_dir
            / "members"
            / presence.member_id
            / "clients"
            / presence.client_id
            / "presence.json"
        )
        self._atomic_write(path, presence)

    def write_event(self, event: TeamActivityEvent) -> None:
        self._require_local_owner(event.member_id, event.client_id)
        stored, _ = self._load_event_files()
        start_candidates = [
            item.started_at or item.occurred_at
            for _, item in stored
            if item.run_id == event.run_id
            and item.member_id == event.member_id
            and item.client_id == event.client_id
        ]
        start_candidates.append(event.started_at or event.occurred_at)
        event = event.model_copy(update={"started_at": min(start_candidates)})
        filename = f"{event.sequence:020d}-{event.event_id}.json"
        path = (
            self.workspace_dir / "events" / event.member_id / event.client_id / filename
        )
        if path.exists():
            existing = self._read_model(path, TeamActivityEvent, required=True)
            if existing != event:
                raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED)
            return
        self._atomic_write(path, event)
        self._compact_event_files()

    def write_conversation(self, entry: TeamConversationEntry) -> None:
        self._require_local_owner(entry.member_id, entry.client_id)
        self.validate_conversation_policy()
        filename = f"{entry.sequence:020d}-{entry.entry_id}.json"
        path = (
            self.workspace_dir
            / "conversations"
            / entry.member_id
            / entry.client_id
            / filename
        )
        if path.exists():
            existing = self._read_model(path, TeamConversationEntry, required=True)
            if existing != entry:
                raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED)
            return
        self._atomic_write(path, entry)
        self._compact_conversation_files()

    def validate_conversation_policy(self) -> None:
        manifest = self._read_model(
            self.manifest_path, TeamWorkspaceManifest, required=True
        )
        if (
            manifest is None
            or manifest.identity != self.identity
            or manifest.privacy_mode is not self.privacy_mode
            or manifest.history_scope is not self.history_scope
            or manifest.history_limit != self.history_limit
        ):
            raise TeamWorkspaceStoreError(SyncError.MANIFEST_MISMATCH)

    def read_snapshot(self, now: datetime) -> TeamWorkspaceSnapshot:
        degraded = False
        try:
            manifest = self._read_model(
                self.manifest_path, TeamWorkspaceManifest, required=True
            )
        except TeamWorkspaceStoreError as error:
            return self._error_snapshot(now, error.code)
        if (
            manifest is None
            or manifest.identity != self.identity
            or manifest.privacy_mode is not self.privacy_mode
            or manifest.history_scope is not self.history_scope
            or manifest.history_limit != self.history_limit
        ):
            return self._error_snapshot(now, SyncError.MANIFEST_MISMATCH)

        presences, presence_degraded = self._read_presences()
        events, event_degraded = self._read_events()
        conversations, conversation_degraded = self._read_conversations()
        degraded = presence_degraded or event_degraded or conversation_degraded
        runs = self._project_runs(events, conversations, self.history_limit)
        members = self._project_members(presences, runs, events, now)
        return TeamWorkspaceSnapshot(
            identity=self.identity,
            privacy_mode=self.privacy_mode,
            history_scope=self.history_scope,
            connection_state=(
                ConnectionState.DEGRADED if degraded else ConnectionState.CONNECTED
            ),
            generated_at=now,
            members=members,
            runs=runs,
            error=SyncError.READ_FAILED if degraded else None,
        )

    def _read_presences(self) -> tuple[list[TeamMemberPresence], bool]:
        records: list[TeamMemberPresence] = []
        degraded = False
        members_root = self.workspace_dir / "members"
        for member_dir in self._child_directories(members_root):
            clients_root = member_dir / "clients"
            for client_dir in self._child_directories(clients_root):
                path = client_dir / "presence.json"
                if not path.exists():
                    continue
                try:
                    record = self._read_model(path, TeamMemberPresence, required=True)
                except TeamWorkspaceStoreError:
                    degraded = True
                    continue
                if (
                    record is None
                    or record.workspace_id != self.identity.workspace_id
                    or record.member_id != member_dir.name
                    or record.client_id != client_dir.name
                ):
                    degraded = True
                    continue
                records.append(record)
        return records, degraded

    def _read_conversations(self) -> tuple[list[TeamConversationEntry], bool]:
        if self.history_scope is HistoryScope.STATUS:
            return [], False
        stored, degraded = self._load_conversation_files()
        selected = self._select_bounded_records(stored)
        return [record for _, record in selected], degraded or len(stored) > len(
            selected
        )

    def _read_events(self) -> tuple[list[TeamActivityEvent], bool]:
        stored, degraded = self._load_event_files()
        current = self._current_event_files(stored)
        selected = self._select_bounded_records(current)
        return [record for _, record in selected], degraded or len(current) > len(
            selected
        )

    def _load_event_files(self) -> tuple[list[tuple[Path, TeamActivityEvent]], bool]:
        stored: list[tuple[Path, TeamActivityEvent]] = []
        degraded = False
        root = self.workspace_dir / "events"
        for member_dir in self._child_directories(root):
            for client_dir in self._child_directories(member_dir):
                for path in self._json_files(client_dir):
                    try:
                        record = self._read_model(
                            path, TeamActivityEvent, required=True
                        )
                    except TeamWorkspaceStoreError:
                        degraded = True
                        continue
                    if record is None:
                        degraded = True
                        continue
                    expected = f"{record.sequence:020d}-{record.event_id}.json"
                    if (
                        record.workspace_id != self.identity.workspace_id
                        or record.member_id != member_dir.name
                        or record.client_id != client_dir.name
                        or record.privacy_mode is not self.privacy_mode
                        or path.name != expected
                    ):
                        degraded = True
                        continue
                    stored.append((path, record))
        return stored, degraded

    def _load_conversation_files(
        self,
    ) -> tuple[list[tuple[Path, TeamConversationEntry]], bool]:
        stored: list[tuple[Path, TeamConversationEntry]] = []
        degraded = False
        root = self.workspace_dir / "conversations"
        for member_dir in self._child_directories(root):
            for client_dir in self._child_directories(member_dir):
                for path in self._json_files(client_dir):
                    try:
                        record = self._read_model(
                            path, TeamConversationEntry, required=True
                        )
                    except TeamWorkspaceStoreError:
                        degraded = True
                        continue
                    if record is None:
                        degraded = True
                        continue
                    expected = f"{record.sequence:020d}-{record.entry_id}.json"
                    if (
                        record.workspace_id != self.identity.workspace_id
                        or record.member_id != member_dir.name
                        or record.client_id != client_dir.name
                        or record.history_scope is not self.history_scope
                        or path.name != expected
                    ):
                        degraded = True
                        continue
                    stored.append((path, record))
        return stored, degraded

    @staticmethod
    def _current_event_files(
        stored: list[tuple[Path, TeamActivityEvent]],
    ) -> list[tuple[Path, TeamActivityEvent]]:
        current: dict[str, tuple[Path, TeamActivityEvent]] = {}
        seen_event_ids: set[tuple[str, str]] = set()
        ordered = sorted(
            stored,
            key=lambda item: (
                item[1].occurred_at,
                item[1].client_id,
                item[1].sequence,
                item[1].event_id,
            ),
        )
        for path, event in ordered:
            event_key = (event.client_id, event.event_id)
            if event_key in seen_event_ids:
                continue
            seen_event_ids.add(event_key)
            previous = current.get(event.run_id)
            if previous is not None and previous[1].state.is_terminal:
                if not event.state.is_terminal:
                    continue
            current[event.run_id] = path, event
        return list(current.values())

    def _select_bounded_records[T: TeamActivityEvent | TeamConversationEntry](
        self, stored: list[tuple[Path, T]]
    ) -> list[tuple[Path, T]]:
        by_stream: dict[tuple[str, str], list[tuple[Path, T]]] = defaultdict(list)
        for item in stored:
            record = item[1]
            by_stream[(record.member_id, record.client_id)].append(item)
        streams = list(by_stream.values())
        for records in streams:
            records.sort(key=self._record_recency, reverse=True)
        streams.sort(key=lambda records: self._record_recency(records[0]), reverse=True)

        selected: list[tuple[Path, T]] = []
        position = 0
        while len(selected) < self.max_event_files:
            added = False
            for records in streams:
                if position >= len(records):
                    continue
                selected.append(records[position])
                added = True
                if len(selected) >= self.max_event_files:
                    break
            if not added:
                break
            position += 1
        return selected

    @staticmethod
    def _record_recency(
        item: tuple[Path, TeamActivityEvent | TeamConversationEntry],
    ) -> tuple[datetime, int, str, str]:
        record = item[1]
        record_id = (
            record.event_id
            if isinstance(record, TeamActivityEvent)
            else record.entry_id
        )
        return record.occurred_at, record.sequence, record.client_id, record_id

    def _compact_event_files(self) -> None:
        stored, _ = self._load_event_files()
        retained = self._select_bounded_records(self._current_event_files(stored))
        self._delete_superseded_records(self.workspace_dir / "events", stored, retained)

    def _compact_conversation_files(self) -> None:
        stored, _ = self._load_conversation_files()
        retained = self._select_bounded_records(stored)
        self._delete_superseded_records(
            self.workspace_dir / "conversations", stored, retained
        )

    def _delete_superseded_records[T: TeamActivityEvent | TeamConversationEntry](
        self, root: Path, stored: list[tuple[Path, T]], retained: list[tuple[Path, T]]
    ) -> None:
        keep = {path for path, _ in retained}
        for path, _ in sorted(stored, key=lambda item: str(item[0])):
            if path in keep:
                continue
            self._unlink_record(path)
            self._remove_empty_record_parents(path.parent, root)

    def _delete_record_tree(self, root: Path) -> None:
        for member_dir in self._child_directories(root):
            for client_dir in self._child_directories(member_dir):
                for path in self._json_files(client_dir):
                    self._unlink_record(path)
                self._remove_empty_record_parents(client_dir, root)

    @staticmethod
    def _unlink_record(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED) from error

    @staticmethod
    def _remove_empty_record_parents(parent: Path, root: Path) -> None:
        current = parent
        while current != root:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _can_tighten_history_scope(self, existing: TeamWorkspaceManifest) -> bool:
        return (
            existing.identity == self.identity
            and existing.privacy_mode is self.privacy_mode
            and existing.history_limit == self.history_limit
            and _HISTORY_SCOPE_RANK[self.history_scope]
            < _HISTORY_SCOPE_RANK[existing.history_scope]
        )

    @staticmethod
    def _project_runs(
        events: list[TeamActivityEvent],
        conversations: list[TeamConversationEntry],
        history_limit: int,
    ) -> tuple[TeamRunSnapshot, ...]:
        seen_event_ids: set[tuple[str, str]] = set()
        runs: dict[str, TeamRunSnapshot] = {}
        started: dict[str, datetime] = {}
        ordered = sorted(
            events,
            key=lambda item: (
                item.occurred_at,
                item.client_id,
                item.sequence,
                item.event_id,
            ),
        )
        for event in ordered:
            event_key = (event.client_id, event.event_id)
            if event_key in seen_event_ids:
                continue
            seen_event_ids.add(event_key)
            current = runs.get(event.run_id)
            if (
                current is not None
                and event.client_id == current.client_id
                and event.sequence <= current.sequence
            ):
                continue
            if (
                current is not None
                and current.state.is_terminal
                and not event.state.is_terminal
            ):
                continue
            event_started_at = event.started_at or event.occurred_at
            started[event.run_id] = min(
                started.get(event.run_id, event_started_at), event_started_at
            )
            runs[event.run_id] = TeamRunSnapshot(
                run_id=event.run_id,
                member_id=event.member_id,
                member_display_name=event.member_display_name,
                client_id=event.client_id,
                agent_name=event.agent_name,
                agent_display_name=event.agent_display_name,
                state=event.state,
                summary=event.summary,
                started_at=started[event.run_id],
                updated_at=event.occurred_at,
                sequence=event.sequence,
            )
        history_by_run: dict[str, list[TeamConversationEntry]] = defaultdict(list)
        seen_entries: set[tuple[str, str]] = set()
        for entry in sorted(
            conversations,
            key=lambda item: (item.client_id, item.sequence, item.entry_id),
        ):
            entry_key = (entry.client_id, entry.entry_id)
            if entry_key in seen_entries:
                continue
            seen_entries.add(entry_key)
            history_by_run[entry.run_id].append(entry)
        with_history = (
            run.model_copy(
                update={"history": tuple(history_by_run[run.run_id][-history_limit:])}
            )
            for run in runs.values()
        )
        return tuple(
            sorted(with_history, key=lambda item: item.updated_at, reverse=True)[:100]
        )

    def _project_members(
        self,
        presences: list[TeamMemberPresence],
        runs: tuple[TeamRunSnapshot, ...],
        events: list[TeamActivityEvent],
        now: datetime,
    ) -> tuple[TeamMemberSnapshot, ...]:
        by_member: dict[str, list[TeamMemberPresence]] = defaultdict(list)
        for presence in presences:
            by_member[presence.member_id].append(presence)

        latest_event_by_member: dict[str, TeamActivityEvent] = {}
        for event in events:
            current = latest_event_by_member.get(event.member_id)
            if current is None or event.occurred_at > current.occurred_at:
                latest_event_by_member[event.member_id] = event

        members: list[TeamMemberSnapshot] = []
        all_member_ids = set(by_member) | set(latest_event_by_member)
        for member_id in all_member_ids:
            member_presences = by_member.get(member_id, [])
            latest_presence = max(
                member_presences,
                key=lambda item: (item.last_seen_at, item.revision),
                default=None,
            )
            latest_event = latest_event_by_member.get(member_id)
            if latest_presence is not None:
                display_name = latest_presence.member_display_name
                last_seen = latest_presence.last_seen_at
                branch = latest_presence.branch
            elif latest_event is not None:
                display_name = latest_event.member_display_name
                last_seen = latest_event.occurred_at
                branch = None
            else:  # pragma: no cover - set membership guarantees one source
                continue
            online = any(
                max(0.0, (now - item.last_seen_at).total_seconds())
                <= self.presence_ttl_seconds
                for item in member_presences
            )
            active_run_count = sum(
                run.member_id == member_id
                and not run.state.is_terminal
                and run.state is not ActivityState.IDLE
                for run in runs
            )
            members.append(
                TeamMemberSnapshot(
                    member_id=member_id,
                    display_name=display_name,
                    presence=(
                        PresenceState.ONLINE if online else PresenceState.OFFLINE
                    ),
                    branch=branch,
                    last_seen_at=last_seen,
                    client_count=max(1, len(member_presences)),
                    active_run_count=active_run_count,
                )
            )
        return tuple(
            sorted(
                members,
                key=lambda item: (
                    item.presence is PresenceState.OFFLINE,
                    -item.last_seen_at.timestamp(),
                    item.display_name.casefold(),
                ),
            )
        )

    def _atomic_write(self, path: Path, model: BaseModel) -> None:
        self._ensure_safe_parent(path.parent)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                file.write(model.model_dump_json())
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
        except OSError as error:
            raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED) from error
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _read_model[T: BaseModel](
        self, path: Path, model: type[T], *, required: bool
    ) -> T | None:
        if path.is_symlink() or not path.is_file():
            if required:
                raise TeamWorkspaceStoreError(SyncError.READ_FAILED)
            return None
        try:
            if path.stat().st_size > self.max_file_bytes:
                raise TeamWorkspaceStoreError(SyncError.READ_FAILED)
            return model.model_validate_json(read_safe(path, raise_on_error=True).text)
        except (OSError, ValueError, ValidationError) as error:
            raise TeamWorkspaceStoreError(SyncError.READ_FAILED) from error

    def _require_local_owner(self, member_id: str, client_id: str) -> None:
        if member_id != self.member_id or client_id != self.client_id:
            raise TeamWorkspaceStoreError(SyncError.WRITE_FAILED)

    def _ensure_safe_parent(self, parent: Path) -> None:
        try:
            relative = parent.relative_to(self.workspace_dir)
        except ValueError as error:
            raise TeamWorkspaceStoreError(SyncError.INVALID_ROOT) from error
        current = self.workspace_dir
        self._ensure_directory(current)
        for part in relative.parts:
            current /= part
            self._ensure_directory(current)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise TeamWorkspaceStoreError(SyncError.INVALID_ROOT)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TeamWorkspaceStoreError(SyncError.INVALID_ROOT) from error
        if not path.is_dir():
            raise TeamWorkspaceStoreError(SyncError.INVALID_ROOT)

    @staticmethod
    def _child_directories(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(
                sorted(
                    (
                        child
                        for child in path.iterdir()
                        if child.is_dir() and not child.is_symlink()
                    ),
                    key=lambda child: child.name,
                )
            )
        except OSError:
            return ()

    @staticmethod
    def _json_files(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(
                sorted(
                    (
                        child
                        for child in path.iterdir()
                        if child.suffix == ".json"
                        and child.is_file()
                        and not child.is_symlink()
                    ),
                    key=lambda child: child.name,
                )
            )
        except OSError:
            return ()

    def _error_snapshot(self, now: datetime, error: SyncError) -> TeamWorkspaceSnapshot:
        return TeamWorkspaceSnapshot(
            identity=self.identity,
            privacy_mode=self.privacy_mode,
            history_scope=self.history_scope,
            connection_state=ConnectionState.DEGRADED,
            generated_at=now,
            error=error,
        )
