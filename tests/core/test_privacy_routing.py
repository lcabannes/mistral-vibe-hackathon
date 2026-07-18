from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import (
    ModelConfig,
    PrivacyRoutingConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.privacy_routing import (
    DEFAULT_RULES,
    PrivacyRouter,
    SecretVault,
    find_sensitive_match,
)
from vibe.core.types import Backend, LLMMessage, PrivacyRouteEngagedEvent, Role

AWS_KEY_TEXT = "creds: AKIAIOSFODNN7EXAMPLE"
PRIVATE_KEY_TEXT = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"


def _routing_config(**privacy_kwargs) -> VibeConfig:
    models = [
        ModelConfig(name="devstral-latest", provider="mistral", alias="cloud"),
        ModelConfig(name="devstral-small", provider="local", alias="private"),
    ]
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        ),
        ProviderConfig(name="local", api_base="http://localhost:8000/v1"),
    ]
    privacy_kwargs.setdefault("enabled", True)
    privacy_kwargs.setdefault("mode", "route")
    privacy_kwargs.setdefault("private_model", "private")
    privacy = PrivacyRoutingConfig(**privacy_kwargs)
    return build_test_vibe_config(
        active_model="cloud",
        models=models,
        providers=providers,
        privacy_routing=privacy,
    )


class TestSensitivityRules:
    @pytest.mark.parametrize(
        "text",
        [
            AWS_KEY_TEXT,
            PRIVATE_KEY_TEXT,
            "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            'API_KEY = "supersecretvalue123"',
            "MY_SERVICE_TOKEN=abcdef123456789",
        ],
    )
    def test_detects_sensitive_content(self, text: str):
        assert find_sensitive_match(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "def hello(): return 42",
            "the password field is required",
            "path = '/Users/alice/project'",
            "API_KEY = os.environ['API_KEY']",
        ],
    )
    def test_ignores_ordinary_content(self, text: str):
        assert find_sensitive_match(text) is None


class TestPrivacyRouter:
    def test_disabled_router_never_routes(self):
        config = build_test_vibe_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        router.scan(messages)
        assert not router.is_sensitive

    def test_routes_to_private_model_on_sensitive_content(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        router.scan(messages)
        assert router.is_sensitive
        routed = router.apply(config.get_active_model())
        assert routed.alias == "private"

    def test_routing_is_sticky_across_later_calls(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        router.scan(messages)
        messages.append(LLMMessage(role=Role.user, content="harmless follow-up"))
        router.scan(messages)
        assert router.apply(config.get_active_model()).alias == "private"

    def test_reset_returns_to_default_model(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        router.scan([LLMMessage(role=Role.user, content=AWS_KEY_TEXT)])
        router.reset()
        assert not router.is_sensitive
        assert router.apply(config.get_active_model()).alias == "cloud"

    def test_scans_only_new_messages(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content="clean")]
        router.scan(messages)
        # Mutating an already scanned message is not picked up; only appends are.
        messages[0] = LLMMessage(role=Role.user, content=AWS_KEY_TEXT)
        router.scan(messages)
        assert not router.is_sensitive
        messages.append(LLMMessage(role=Role.user, content=AWS_KEY_TEXT))
        router.scan(messages)
        assert router.is_sensitive

    def test_rescans_after_history_rewrite(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        router.scan([LLMMessage(role=Role.user, content="a" * 3)] * 5)
        shorter = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        router.scan(shorter)
        assert router.is_sensitive

    def test_notice_emitted_once(self):
        config = _routing_config()
        router = PrivacyRouter(lambda: config)
        router.scan([LLMMessage(role=Role.user, content=AWS_KEY_TEXT)])
        notice = router.consume_notice()
        assert notice is not None
        assert notice.model_alias == "private"
        assert router.consume_notice() is None

    def test_custom_pattern_matches(self):
        config = _routing_config(custom_patterns=[r"PROJECT-CODENAME-\d+"])
        router = PrivacyRouter(lambda: config)
        router.scan([LLMMessage(role=Role.user, content="see PROJECT-CODENAME-42")])
        assert router.is_sensitive


class TestPrivacyRoutingConfigValidation:
    def test_enabled_requires_private_model(self):
        with pytest.raises(ValueError, match="private_model"):
            _routing_config(private_model="")

    def test_private_model_must_exist(self):
        models = [
            ModelConfig(name="devstral-latest", provider="mistral", alias="cloud")
        ]
        providers = [
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                backend=Backend.MISTRAL,
            )
        ]
        with pytest.raises(ValueError, match="not in your"):
            build_test_vibe_config(
                active_model="cloud",
                models=models,
                providers=providers,
                privacy_routing=PrivacyRoutingConfig(
                    enabled=True, mode="route", private_model="ghost"
                ),
            )

    def test_invalid_custom_pattern_rejected(self):
        with pytest.raises(ValueError, match="Invalid privacy_routing"):
            PrivacyRoutingConfig(
                enabled=True, private_model="private", custom_patterns=["[unclosed"]
            )


class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_sensitive_prompt_emits_event_and_routes_model(self):
        config = _routing_config()
        backend = FakeBackend([mock_llm_chunk(content="ok")])
        agent = build_test_agent_loop(config=config, backend=backend)

        events = [e async for e in agent.act(f"use this key {AWS_KEY_TEXT}")]

        route_events = [e for e in events if isinstance(e, PrivacyRouteEngagedEvent)]
        assert len(route_events) == 1
        assert route_events[0].model_alias == "private"
        assert agent._privacy_router.is_sensitive

    @pytest.mark.asyncio
    async def test_clean_prompt_does_not_route(self):
        config = _routing_config()
        backend = FakeBackend([mock_llm_chunk(content="ok")])
        agent = build_test_agent_loop(config=config, backend=backend)

        events = [e async for e in agent.act("hello, refactor main please")]

        assert not any(isinstance(e, PrivacyRouteEngagedEvent) for e in events)
        assert not agent._privacy_router.is_sensitive

    @pytest.mark.asyncio
    async def test_clear_history_resets_routing(self):
        config = _routing_config()
        backend = FakeBackend([
            mock_llm_chunk(content="ok"),
            mock_llm_chunk(content="ok"),
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act(f"key: {AWS_KEY_TEXT}")]
        assert agent._privacy_router.is_sensitive

        await agent.clear_history()
        assert not agent._privacy_router.is_sensitive


def _redact_config(**privacy_kwargs) -> VibeConfig:
    """Config with redact mode: secrets are masked, no model switch."""
    models = [ModelConfig(name="devstral-latest", provider="mistral", alias="cloud")]
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        )
    ]
    privacy_kwargs.setdefault("enabled", True)
    privacy_kwargs.setdefault("mode", "redact")
    privacy = PrivacyRoutingConfig(**privacy_kwargs)
    return build_test_vibe_config(
        active_model="cloud",
        models=models,
        providers=providers,
        privacy_routing=privacy,
    )


class TestSecretVault:
    def test_register_returns_stable_placeholder(self):
        vault = SecretVault()
        p1 = vault.register("AKIAIOSFODNN7EXAMPLE", "aws-access-key-id")
        p2 = vault.register("AKIAIOSFODNN7EXAMPLE", "aws-access-key-id")
        assert p1 == p2
        assert "[REDACTED:aws-access-key-id:1]" == p1

    def test_different_secrets_get_different_placeholders(self):
        vault = SecretVault()
        p1 = vault.register("secret1", "rule-a")
        p2 = vault.register("secret2", "rule-b")
        assert p1 != p2

    def test_redact_masks_secrets_in_text(self):
        vault = SecretVault()
        text = f"use {AWS_KEY_TEXT} to connect"
        redacted = vault.redact(text, DEFAULT_RULES)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:aws-access-key-id:1]" in redacted

    def test_rehydrate_restores_secrets(self):
        vault = SecretVault()
        text = "key is AKIAIOSFODNN7EXAMPLE"
        redacted = vault.redact(text, DEFAULT_RULES)
        restored = vault.rehydrate(redacted)
        assert restored == text

    def test_redact_preserves_context_for_named_group_rules(self):
        vault = SecretVault()
        text = 'password = "hunter2hunter2"'
        redacted = vault.redact(text, DEFAULT_RULES)
        # The keyword + assignment syntax is kept; only the value is masked.
        assert "password" in redacted
        assert "hunter2hunter2" not in redacted
        assert "[REDACTED:" in redacted

    def test_redact_idempotent(self):
        vault = SecretVault()
        text = "AKIAIOSFODNN7EXAMPLE"
        r1 = vault.redact(text, DEFAULT_RULES)
        r2 = vault.redact(r1, DEFAULT_RULES)
        assert r1 == r2


class TestRedactModeRouter:
    def test_does_not_switch_models(self):
        config = _redact_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        router.scan(messages)
        # In redact mode, scan doesn't flip sticky state.
        assert not router.is_sensitive
        assert router.apply(config.get_active_model()).alias == "cloud"

    def test_redact_for_wire_masks_secrets(self):
        config = _redact_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        redacted = router.redact_for_wire(messages)
        assert "AKIAIOSFODNN7EXAMPLE" not in (redacted[0].content or "")
        assert "[REDACTED:" in (redacted[0].content or "")

    def test_rehydrate_tool_arguments_restores_secrets(self):
        config = _redact_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        redacted = router.redact_for_wire(messages)
        placeholder = (redacted[0].content or "").split("REDACTED")[1]
        placeholder = "[REDACTED" + placeholder.split("]")[0] + "]"
        restored = router.rehydrate_tool_arguments(
            f'{{"path": ".env", "content": "{placeholder}"}}'
        )
        assert "AKIAIOSFODNN7EXAMPLE" in restored

    def test_disabled_redact_is_passthrough(self):
        config = build_test_vibe_config()
        router = PrivacyRouter(lambda: config)
        messages = [LLMMessage(role=Role.user, content=AWS_KEY_TEXT)]
        out = router.redact_for_wire(messages)
        assert out is messages


class TestAgentLoopRedactIntegration:
    @pytest.mark.asyncio
    async def test_secrets_redacted_before_reaching_backend(self):
        config = _redact_config()
        backend = FakeBackend([mock_llm_chunk(content="ok")])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act(f"use this key {AWS_KEY_TEXT}")]

        sent_messages = backend.requests_messages[0]
        for msg in sent_messages:
            if msg.content:
                assert "AKIAIOSFODNN7EXAMPLE" not in msg.content
        assert not agent._privacy_router.is_sensitive


class TestSecretRedactionNotification:
    @pytest.mark.asyncio
    async def test_redacted_secret_emits_event_before_response(self):
        from vibe.core.types import SecretRedactedEvent

        config = _redact_config()
        backend = FakeBackend([mock_llm_chunk(content="ok")])
        agent = build_test_agent_loop(config=config, backend=backend)

        events = [e async for e in agent.act(f"my key is {AWS_KEY_TEXT}")]

        redacted = [e for e in events if isinstance(e, SecretRedactedEvent)]
        assert len(redacted) == 1
        assert redacted[0].placeholder.startswith("[REDACTED:aws-access-key-id:")

    @pytest.mark.asyncio
    async def test_same_secret_not_announced_twice_in_session(self):
        from vibe.core.types import SecretRedactedEvent

        config = _redact_config()
        backend = FakeBackend([
            mock_llm_chunk(content="ok"),
            mock_llm_chunk(content="ok again"),
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        first = [e async for e in agent.act(f"key: {AWS_KEY_TEXT}")]
        second = [e async for e in agent.act(f"same key: {AWS_KEY_TEXT}")]

        assert sum(isinstance(e, SecretRedactedEvent) for e in first) == 1
        assert sum(isinstance(e, SecretRedactedEvent) for e in second) == 0

    def test_persisted_secret_still_announced_in_new_session(self):
        # The bug: a secret already in the keychain vault from a past session
        # produced no notification because the vault didn't grow.
        vault = SecretVault()
        vault.register("AKIAIOSFODNN7EXAMPLE", "aws-access-key-id")
        vault.consume_unannounced()

        # Same secret redacted again later in the session: no re-announcement.
        vault.redact("key AKIAIOSFODNN7EXAMPLE", DEFAULT_RULES)
        assert vault.consume_unannounced() == []

        # Fresh session vault that already knows the secret (as if loaded from
        # the keychain): first use this session must announce.
        fresh = SecretVault()
        placeholder = fresh.register("AKIAIOSFODNN7EXAMPLE", "aws-access-key-id")
        fresh.consume_unannounced()
        fresh2 = SecretVault()
        fresh2._by_secret["AKIAIOSFODNN7EXAMPLE"] = placeholder
        fresh2._by_placeholder[placeholder] = "AKIAIOSFODNN7EXAMPLE"
        fresh2.redact("key AKIAIOSFODNN7EXAMPLE", DEFAULT_RULES)
        assert fresh2.consume_unannounced() == [placeholder]


class TestFreshRedactionTurn:
    @pytest.mark.asyncio
    async def test_injects_secure_notice_and_suppresses_tool_calls(self):
        from vibe.core.types import FunctionCall, ToolCall

        config = _redact_config()
        # Model tries to call a tool on the redaction turn; it must be dropped.
        tool_chunk = mock_llm_chunk(
            content="Let me check something",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"command": "ls"}'),
                )
            ],
        )
        backend = FakeBackend([[tool_chunk]])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act(f"use {AWS_KEY_TEXT} to deploy")]

        # The injected instruction reached the wire.
        sent = backend.requests_messages[0]
        injected = [
            m for m in sent if m.injected and "confidential value" in (m.content or "")
        ]
        assert len(injected) == 1
        assert "do not call any tools" in injected[0].content
        # The tool call was stripped: no tool-role message ever appeared.
        assert all(m.role != Role.tool for m in agent.messages)

    @pytest.mark.asyncio
    async def test_old_placeholder_in_history_does_not_retrigger(self):
        config = _redact_config()
        backend = FakeBackend([
            mock_llm_chunk(content="noted"),
            mock_llm_chunk(content="sure"),
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act(f"key: {AWS_KEY_TEXT}")]
        # Second turn has no new secret; the notice must not be re-injected.
        [_ async for _ in agent.act("now write a haiku")]

        second_call = backend.requests_messages[1]
        fresh_notices = [
            m
            for m in second_call
            if m.injected and "confidential value" in (m.content or "")
        ]
        # Only the original injection from turn one remains in history.
        assert len(fresh_notices) == 1


class TestSecretEnvGating:
    def test_main_loop_gets_no_secret_env_by_default(self):
        config = _redact_config()
        agent = build_test_agent_loop(config=config, backend=FakeBackend())
        assert agent._resolve_secret_env() == {}

    def test_main_loop_gets_env_when_opted_in(self, monkeypatch):
        config = _redact_config(expose_secrets_as_env=True)
        agent = build_test_agent_loop(config=config, backend=FakeBackend())
        monkeypatch.setattr(
            "vibe.core.agent_loop._loop.vault_env_vars",
            lambda: {"VIBE_SECRET_JWT_1": "value"},
        )
        assert agent._resolve_secret_env() == {"VIBE_SECRET_JWT_1": "value"}

    def test_local_subagent_always_gets_env(self, monkeypatch):
        config = _redact_config()
        agent = build_test_agent_loop(
            config=config, backend=FakeBackend(), bypass_path_guard=True
        )
        monkeypatch.setattr(
            "vibe.core.agent_loop._loop.vault_env_vars",
            lambda: {"VIBE_SECRET_JWT_1": "value"},
        )
        assert agent._resolve_secret_env() == {"VIBE_SECRET_JWT_1": "value"}


class TestSecretEnvReachesBash:
    @pytest.mark.asyncio
    async def test_env_var_expands_in_bash_subprocess(self, monkeypatch, tmp_path):
        from vibe.core.types import FunctionCall, ToolCall

        config = _redact_config(expose_secrets_as_env=True)
        config = config.model_copy(
            update={
                "bypass_tool_permissions": True,
                "tools": {"bash": {"permission": "always"}},
            }
        )
        monkeypatch.setattr(
            "vibe.core.agent_loop._loop.vault_env_vars",
            lambda: {"VIBE_SECRET_JWT_1": "the-real-secret"},
        )
        out_file = tmp_path / "out.txt"
        bash_chunk = mock_llm_chunk(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    index=0,
                    function=FunctionCall(
                        name="bash",
                        arguments=(
                            '{"command": "printf %s \\"$VIBE_SECRET_JWT_1\\" > '
                            + f'{out_file}"}}'
                        ),
                    ),
                )
            ],
        )
        backend = FakeBackend([[bash_chunk], [mock_llm_chunk(content="done")]])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act("write the secret to a file")]

        assert out_file.read_text() == "the-real-secret"
