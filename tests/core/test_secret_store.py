from __future__ import annotations

from pathlib import Path
import stat

from vibe.core.privacy_routing import DEFAULT_RULES, SecretVault
from vibe.core.secret_store import PersistentSecretStore

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
PEM_KEY = "-----BEGIN RSA PRIVATE KEY-----\nline1\nline2\n-----END RSA PRIVATE KEY-----"


class TestPersistentSecretStore:
    def test_store_and_lookup(self, tmp_path: Path):
        store = PersistentSecretStore(vault_path=tmp_path / "secrets.toml")
        placeholder = store.next_placeholder("aws-access-key-id")
        assert store.store(placeholder, "aws-access-key-id", AWS_KEY)
        assert store.lookup(placeholder) == AWS_KEY

    def test_counter_survives_reload(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        p1 = store.next_placeholder("jwt")
        store.store(p1, "jwt", "secret-one")

        reloaded = PersistentSecretStore(vault_path=path)
        p2 = reloaded.next_placeholder("jwt")
        assert p1.endswith(":1]")
        assert p2.endswith(":2]")

    def test_values_persist_across_instances(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        placeholder = store.next_placeholder("aws-access-key-id")
        store.store(placeholder, "aws-access-key-id", AWS_KEY)

        reloaded = PersistentSecretStore(vault_path=path)
        assert reloaded.load_all() == {placeholder: AWS_KEY}
        assert reloaded.lookup(placeholder) == AWS_KEY

    def test_multiline_secret_roundtrips(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        placeholder = store.next_placeholder("private-key-block")
        store.store(placeholder, "private-key-block", PEM_KEY)
        assert PersistentSecretStore(vault_path=path).lookup(placeholder) == PEM_KEY

    def test_vault_file_is_owner_only(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        store.store(store.next_placeholder("jwt"), "jwt", "secret")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_delete_removes_entry_and_value(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        placeholder = store.next_placeholder("jwt")
        store.store(placeholder, "jwt", "secret")
        assert store.delete(placeholder)
        assert store.lookup(placeholder) is None
        assert AWS_KEY not in path.read_text()

    def test_delete_unknown_returns_false(self, tmp_path: Path):
        store = PersistentSecretStore(vault_path=tmp_path / "secrets.toml")
        assert not store.delete("[REDACTED:jwt:99]")


class TestVaultPersistence:
    def test_vault_restores_secrets_across_instances(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        vault = SecretVault(store=PersistentSecretStore(vault_path=path))
        redacted = vault.redact(f"key: {AWS_KEY}", DEFAULT_RULES)
        assert AWS_KEY not in redacted

        vault2 = SecretVault(store=PersistentSecretStore(vault_path=path))
        assert vault2.rehydrate(redacted) == f"key: {AWS_KEY}"

    def test_same_secret_same_placeholder_across_sessions(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        vault = SecretVault(store=PersistentSecretStore(vault_path=path))
        first = vault.redact(AWS_KEY, DEFAULT_RULES)

        vault2 = SecretVault(store=PersistentSecretStore(vault_path=path))
        second = vault2.redact(AWS_KEY, DEFAULT_RULES)
        assert first == second

    def test_forget_removes_persisted_secret(self, tmp_path: Path):
        path = tmp_path / "secrets.toml"
        vault = SecretVault(store=PersistentSecretStore(vault_path=path))
        redacted = vault.redact(AWS_KEY, DEFAULT_RULES)
        placeholder = vault.placeholders()[0]
        assert vault.forget(placeholder)

        vault2 = SecretVault(store=PersistentSecretStore(vault_path=path))
        assert vault2.rehydrate(redacted) == redacted


class TestVaultDirProtected:
    def test_vault_dir_is_always_a_protected_path(self):
        from tests.conftest import build_test_vibe_config
        from vibe.core.config import ModelConfig, PrivacyRoutingConfig, ProviderConfig
        from vibe.core.path_guard import is_protected_path, protection_patterns
        from vibe.core.secret_store import vault_dir
        from vibe.core.types import Backend

        config = build_test_vibe_config(
            active_model="cloud",
            models=[
                ModelConfig(name="m", provider="mistral", alias="cloud"),
                ModelConfig(name="l", provider="local", alias="private"),
            ],
            providers=[
                ProviderConfig(
                    name="mistral",
                    api_base="https://api.mistral.ai/v1",
                    api_key_env_var="MISTRAL_API_KEY",
                    backend=Backend.MISTRAL,
                ),
                ProviderConfig(name="local", api_base="http://localhost:8000/v1"),
            ],
            privacy_routing=PrivacyRoutingConfig(
                enabled=True, mode="redact", private_model="private"
            ),
        )
        patterns = protection_patterns(config)
        secret_file = str(vault_dir() / "secrets.toml")
        assert is_protected_path(secret_file, patterns)


class TestEnvVarExposure:
    def test_env_var_name_derivation(self):
        from vibe.core.secret_store import env_var_name

        assert env_var_name("[REDACTED:github-token:1]") == "VIBE_SECRET_GITHUB_TOKEN_1"
        assert (
            env_var_name("[REDACTED:aws-access-key-id:12]")
            == "VIBE_SECRET_AWS_ACCESS_KEY_ID_12"
        )
        assert env_var_name("[REDACTED:custom-0:3]") == "VIBE_SECRET_CUSTOM_0_3"

    def test_vault_env_vars_maps_all_secrets(self, tmp_path: Path):
        from vibe.core.secret_store import vault_env_vars

        path = tmp_path / "secrets.toml"
        store = PersistentSecretStore(vault_path=path)
        p = store.next_placeholder("github-token")
        store.store(p, "github-token", "ghp_realvalue123")

        env = vault_env_vars(vault_path=path)
        assert env == {"VIBE_SECRET_GITHUB_TOKEN_1": "ghp_realvalue123"}
