from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import tomllib

import tomli_w

from vibe.core.paths import VIBE_HOME
from vibe.core.utils.keyring import (
    delete_secret_from_keyring,
    get_secret_from_keyring,
    set_secret_in_keyring,
)

logger = logging.getLogger(__name__)

SECRET_VAULT_INDEX_FILE = "secret_vault.toml"


@dataclass(frozen=True)
class SecretEntry:
    placeholder: str
    rule_name: str
    created_at: str


def _index_path() -> Path:
    return VIBE_HOME.path / SECRET_VAULT_INDEX_FILE


class PersistentSecretStore:
    """Placeholder index on disk, secret values in the OS keychain.

    The index file maps placeholders to rule metadata and holds the global
    counter that keeps placeholders stable across sessions. It never contains
    secret values: those live in the keychain under the placeholder as the
    account name, so `[REDACTED:aws-access-key-id:3]` in any old session log
    can always be resolved back to its value.
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self._index_path = index_path or _index_path()
        self._entries: dict[str, SecretEntry] = {}
        self._counter = 0
        self._load_index()

    @property
    def counter(self) -> int:
        return self._counter

    def entries(self) -> list[SecretEntry]:
        return list(self._entries.values())

    def next_placeholder(self, rule_name: str) -> str:
        self._counter += 1
        return f"[REDACTED:{rule_name}:{self._counter}]"

    def store(self, placeholder: str, rule_name: str, secret: str) -> bool:
        """Persist a secret; returns False when the keychain is unavailable."""
        if not set_secret_in_keyring(placeholder, secret):
            return False
        self._entries[placeholder] = SecretEntry(
            placeholder=placeholder,
            rule_name=rule_name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._save_index()
        return True

    def lookup(self, placeholder: str) -> str | None:
        """Resolve a placeholder to its secret value via the keychain."""
        if placeholder not in self._entries:
            return None
        return get_secret_from_keyring(placeholder)

    def load_all(self) -> dict[str, str]:
        """Resolve every indexed placeholder; skips keychain misses."""
        resolved: dict[str, str] = {}
        for placeholder in self._entries:
            value = get_secret_from_keyring(placeholder)
            if value is not None:
                resolved[placeholder] = value
            else:
                logger.warning(
                    "Secret for %s is indexed but missing from the keychain.",
                    placeholder,
                )
        return resolved

    def delete(self, placeholder: str) -> bool:
        if placeholder not in self._entries:
            return False
        delete_secret_from_keyring(placeholder)
        del self._entries[placeholder]
        self._save_index()
        return True

    def _load_index(self) -> None:
        try:
            raw = self._index_path.read_bytes()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning("Cannot read secret vault index: %s", e)
            return
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            logger.warning("Invalid secret vault index; starting fresh: %s", e)
            return
        self._counter = int(data.get("counter", 0))
        for item in data.get("secrets", []):
            placeholder = item.get("placeholder")
            if not isinstance(placeholder, str):
                continue
            self._entries[placeholder] = SecretEntry(
                placeholder=placeholder,
                rule_name=str(item.get("rule_name", "unknown")),
                created_at=str(item.get("created_at", "")),
            )

    def _save_index(self) -> None:
        data = {
            "counter": self._counter,
            "secrets": [
                {
                    "placeholder": e.placeholder,
                    "rule_name": e.rule_name,
                    "created_at": e.created_at,
                }
                for e in self._entries.values()
            ],
        }
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index_path, "wb") as f:
                tomli_w.dump(data, f)
            self._index_path.chmod(0o600)
        except OSError as e:
            logger.warning("Cannot write secret vault index: %s", e)
