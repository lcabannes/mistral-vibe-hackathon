from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.privacy_routing import DEFAULT_RULES, SecretVault
from vibe.core.secret_store import PersistentSecretStore

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


@pytest.fixture
def keyring_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """In-memory keychain so tests exercise the real store logic."""
    storage: dict[str, str] = {}
    monkeypatch.setattr(
        "vibe.core.secret_store.set_secret_in_keyring",
        lambda name, value: storage.__setitem__(name, value) or True,
    )
    monkeypatch.setattr("vibe.core.secret_store.get_secret_from_keyring", storage.get)
    monkeypatch.setattr(
        "vibe.core.secret_store.delete_secret_from_keyring",
        lambda name: storage.pop(name, None),
    )
    return storage


class TestPersistentSecretStore:
    def test_store_and_lookup(self, tmp_path: Path, keyring_stub: dict[str, str]):
        store = PersistentSecretStore(index_path=tmp_path / "vault.toml")
        placeholder = store.next_placeholder("aws-access-key-id")
        assert store.store(placeholder, "aws-access-key-id", AWS_KEY)
        assert store.lookup(placeholder) == AWS_KEY

    def test_counter_survives_reload(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        store = PersistentSecretStore(index_path=index)
        p1 = store.next_placeholder("jwt")
        store.store(p1, "jwt", "secret-one")

        reloaded = PersistentSecretStore(index_path=index)
        p2 = reloaded.next_placeholder("jwt")
        assert p1 != p2
        assert p1.endswith(":1]")
        assert p2.endswith(":2]")

    def test_load_all_resolves_persisted_secrets(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        store = PersistentSecretStore(index_path=index)
        placeholder = store.next_placeholder("aws-access-key-id")
        store.store(placeholder, "aws-access-key-id", AWS_KEY)

        reloaded = PersistentSecretStore(index_path=index)
        assert reloaded.load_all() == {placeholder: AWS_KEY}

    def test_index_never_contains_secret_values(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        store = PersistentSecretStore(index_path=index)
        placeholder = store.next_placeholder("aws-access-key-id")
        store.store(placeholder, "aws-access-key-id", AWS_KEY)
        assert AWS_KEY not in index.read_text()

    def test_delete_removes_from_index_and_keychain(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        store = PersistentSecretStore(index_path=tmp_path / "vault.toml")
        placeholder = store.next_placeholder("jwt")
        store.store(placeholder, "jwt", "secret")
        assert store.delete(placeholder)
        assert store.lookup(placeholder) is None
        assert not keyring_stub

    def test_keychain_unavailable_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VIBE_TEST_DISABLE_KEYRING", "1")
        store = PersistentSecretStore(index_path=tmp_path / "vault.toml")
        placeholder = store.next_placeholder("jwt")
        assert not store.store(placeholder, "jwt", "secret")


class TestVaultPersistence:
    def test_vault_restores_secrets_across_instances(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        vault = SecretVault(store=PersistentSecretStore(index_path=index))
        redacted = vault.redact(f"key: {AWS_KEY}", DEFAULT_RULES)
        assert AWS_KEY not in redacted

        # New session: fresh vault, same store — placeholder must resolve.
        vault2 = SecretVault(store=PersistentSecretStore(index_path=index))
        assert vault2.rehydrate(redacted) == f"key: {AWS_KEY}"

    def test_same_secret_same_placeholder_across_sessions(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        vault = SecretVault(store=PersistentSecretStore(index_path=index))
        first = vault.redact(AWS_KEY, DEFAULT_RULES)

        vault2 = SecretVault(store=PersistentSecretStore(index_path=index))
        second = vault2.redact(AWS_KEY, DEFAULT_RULES)
        assert first == second

    def test_forget_removes_persisted_secret(
        self, tmp_path: Path, keyring_stub: dict[str, str]
    ):
        index = tmp_path / "vault.toml"
        vault = SecretVault(store=PersistentSecretStore(index_path=index))
        redacted = vault.redact(AWS_KEY, DEFAULT_RULES)
        placeholder = vault.placeholders()[0]
        assert vault.forget(placeholder)

        vault2 = SecretVault(store=PersistentSecretStore(index_path=index))
        # Placeholder no longer resolves anywhere.
        assert vault2.rehydrate(redacted) == redacted
