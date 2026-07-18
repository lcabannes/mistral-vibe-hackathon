from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import re
import tomllib

import tomli_w

from vibe.core.paths import VIBE_HOME

logger = logging.getLogger(__name__)

_ENV_SANITIZE_RE = re.compile(r"[^A-Z0-9]+")

# The vault lives inside a protected directory: readable by you and by the
# local model (local_task bypasses the path guard), but blocked from the
# cloud-facing loop like any other protected path. Secret values are stored
# here in plaintext — safe against the network/cloud threat this feature
# targets, so keep the file 0600 and out of backups/git.
SECRET_VAULT_DIR = "vault"
SECRET_VAULT_FILE = "secrets.toml"


@dataclass(frozen=True)
class SecretEntry:
    placeholder: str
    rule_name: str
    created_at: str


def vault_dir() -> Path:
    return VIBE_HOME.path / SECRET_VAULT_DIR


def env_var_name(placeholder: str) -> str:
    """Stable environment-variable name for a placeholder.

    ``[REDACTED:github-token:1]`` -> ``VIBE_SECRET_GITHUB_TOKEN_1``. Shell
    expansion happens locally, so a model writing ``$VIBE_SECRET_GITHUB_TOKEN_1``
    in a command uses the real value without ever seeing it.
    """
    inner = placeholder.strip("[]").removeprefix("REDACTED:")
    return f"VIBE_SECRET_{_ENV_SANITIZE_RE.sub('_', inner.upper()).strip('_')}"


def vault_env_vars(vault_path: Path | None = None) -> dict[str, str]:
    """Every vault secret as an env mapping, freshly loaded from disk."""
    store = PersistentSecretStore(vault_path=vault_path)
    return {
        env_var_name(placeholder): value
        for placeholder, value in store.load_all().items()
    }


def _vault_path() -> Path:
    return vault_dir() / SECRET_VAULT_FILE


class PersistentSecretStore:
    """File-backed secret vault in a cloud-inaccessible directory.

    A single TOML file holds placeholders, metadata, the global counter, and
    the secret values themselves. It sits under a protected path so the cloud
    model can never read it, while the local model (and you) can. TOML keeps
    multiline secrets (e.g. PEM private keys) round-tripping intact.
    """

    def __init__(self, vault_path: Path | None = None) -> None:
        self._path = vault_path or _vault_path()
        self._entries: dict[str, SecretEntry] = {}
        self._values: dict[str, str] = {}
        self._counter = 0
        self._load()

    @property
    def counter(self) -> int:
        return self._counter

    def entries(self) -> list[SecretEntry]:
        return list(self._entries.values())

    def next_placeholder(self, rule_name: str) -> str:
        self._counter += 1
        return f"[REDACTED:{rule_name}:{self._counter}]"

    def store(self, placeholder: str, rule_name: str, secret: str) -> bool:
        self._entries[placeholder] = SecretEntry(
            placeholder=placeholder,
            rule_name=rule_name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._values[placeholder] = secret
        return self._save()

    def lookup(self, placeholder: str) -> str | None:
        return self._values.get(placeholder)

    def load_all(self) -> dict[str, str]:
        return dict(self._values)

    def delete(self, placeholder: str) -> bool:
        if placeholder not in self._entries:
            return False
        del self._entries[placeholder]
        self._values.pop(placeholder, None)
        self._save()
        return True

    def _load(self) -> None:
        try:
            data = tomllib.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            logger.warning("Cannot read secret vault; starting fresh: %s", e)
            return
        self._counter = int(data.get("counter", 0))
        for item in data.get("secrets", []):
            placeholder = item.get("placeholder")
            value = item.get("value")
            if not isinstance(placeholder, str) or not isinstance(value, str):
                continue
            self._entries[placeholder] = SecretEntry(
                placeholder=placeholder,
                rule_name=str(item.get("rule_name", "unknown")),
                created_at=str(item.get("created_at", "")),
            )
            self._values[placeholder] = value

    def _save(self) -> bool:
        data = {
            "counter": self._counter,
            "secrets": [
                {
                    "placeholder": e.placeholder,
                    "rule_name": e.rule_name,
                    "created_at": e.created_at,
                    "value": self._values.get(e.placeholder, ""),
                }
                for e in self._entries.values()
            ],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                tomli_w.dump(data, f)
            self._path.chmod(0o600)
        except OSError as e:
            logger.warning("Cannot write secret vault: %s", e)
            return False
        return True
