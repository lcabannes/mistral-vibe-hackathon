from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe.core.config import AnyVibeConfig, ModelConfig
    from vibe.core.secret_store import PersistentSecretStore
    from vibe.core.types import LLMMessage, ToolCall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensitivityRule:
    name: str
    pattern: re.Pattern[str]


def _rule(name: str, pattern: str, flags: int = 0) -> SensitivityRule:
    return SensitivityRule(name=name, pattern=re.compile(pattern, flags))


# Credential-focused defaults: each rule should be specific enough that ordinary
# code and prose never match. Broader detection (emails, names, ...) belongs in
# user-provided custom_patterns where the false-positive tradeoff is theirs.
#
# Rules double as redaction extractors: when a rule defines a named group
# `secret`, only that span is replaced with a placeholder (keeping e.g. the
# variable name for context); otherwise the whole match is the secret.
DEFAULT_RULES: tuple[SensitivityRule, ...] = (
    # Swallow the whole PEM body: redacting only the BEGIN header would leave
    # the key material in the text. Tolerate a missing END marker (truncated
    # paste) by redacting through to the end of the text.
    _rule(
        "private-key-block",
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY-----"
        r"[\s\S]*?"
        r"(?:-----END (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY-----|\Z)",
    ),
    _rule("aws-access-key-id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    _rule("github-token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    _rule("slack-token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    _rule("google-api-key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    _rule("api-secret-key", r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    _rule(
        "jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    _rule(
        "secret-assignment",
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token)\b"
        r"\s*[=:]\s*['\"](?P<secret>[^'\"\s]{8,})['\"]",
    ),
    # Strict dotenv shape (no spaces around '='): code like
    # `API_KEY = os.environ["API_KEY"]` must not match.
    _rule(
        "env-secret-line",
        r"^[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIALS)="
        r"(?P<secret>[^\s'\"$]{8,})$",
        re.MULTILINE,
    ),
)

_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[A-Za-z0-9_-]+:\d+\]")


class SecretVault:
    """Local-only, two-way map between real secrets and stable placeholders.

    Secret values never appear in any outbound request: redaction happens at
    the egress boundary and rehydration at ingress, so the wire only ever
    carries placeholders.

    With a ``PersistentSecretStore`` attached, values are written to the OS
    keychain and placeholders stay globally stable across sessions (the store
    owns the counter); a resumed session's placeholders resolve again. Without
    one, the vault is purely in-memory and dies with the session.
    """

    def __init__(self, store: PersistentSecretStore | None = None) -> None:
        self._by_secret: dict[str, str] = {}
        self._by_placeholder: dict[str, str] = {}
        self._counter = 0
        # Placeholders substituted into an outbound message but not yet
        # surfaced to the user this session. Announcement is keyed on use,
        # not registration: a secret already in the persistent vault from a
        # past session must still be announced the first time it appears in
        # this one.
        self._unannounced: list[str] = []
        self._announced: set[str] = set()
        self._store = store
        if store is not None:
            for placeholder, secret in store.load_all().items():
                self._by_secret[secret] = placeholder
                self._by_placeholder[placeholder] = secret

    def __len__(self) -> int:
        return len(self._by_secret)

    def register(self, secret: str, rule_name: str) -> str:
        if existing := self._by_secret.get(secret):
            self._mark_used(existing)
            return existing
        if self._store is not None:
            placeholder = self._store.next_placeholder(rule_name)
            if not self._store.store(placeholder, rule_name, secret):
                logger.warning(
                    "Keychain unavailable; secret %s is session-scoped only.",
                    placeholder,
                )
        else:
            self._counter += 1
            placeholder = f"[REDACTED:{rule_name}:{self._counter}]"
        self._by_secret[secret] = placeholder
        self._by_placeholder[placeholder] = secret
        self._mark_used(placeholder)
        return placeholder

    def _mark_used(self, placeholder: str) -> None:
        if placeholder not in self._announced and placeholder not in self._unannounced:
            self._unannounced.append(placeholder)

    def forget(self, placeholder: str) -> bool:
        """Drop a secret from memory and, when persistent, from the keychain."""
        secret = self._by_placeholder.pop(placeholder, None)
        if secret is None:
            return False
        self._by_secret.pop(secret, None)
        if self._store is not None:
            self._store.delete(placeholder)
        return True

    def placeholders(self) -> list[str]:
        """Known placeholders, metadata only — values are never exposed."""
        return list(self._by_placeholder)

    def consume_unannounced(self) -> list[str]:
        """Placeholders redacted since the last call, each announced once."""
        placeholders = self._unannounced
        self._announced.update(placeholders)
        self._unannounced = []
        return placeholders

    def redact(self, text: str, rules: Sequence[SensitivityRule]) -> str:
        """Replace every secret in text with its placeholder, registering new ones."""
        for rule in rules:

            def _sub(match: re.Match[str], rule_name: str = rule.name) -> str:
                full = match.group(0)
                secret = (
                    match.group("secret") if "secret" in match.re.groupindex else None
                )
                target = full if secret is None else secret
                # Never re-register an already-redacted span: a broad rule can
                # match a placeholder an earlier rule produced, and nesting
                # placeholders would break rehydration.
                if _PLACEHOLDER_RE.search(target):
                    return full
                if secret is None:
                    return self.register(full, rule_name)
                placeholder = self.register(secret, rule_name)
                start, end = match.span("secret")
                offset = match.start(0)
                return full[: start - offset] + placeholder + full[end - offset :]

            text = rule.pattern.sub(_sub, text)
        return text

    def rehydrate(self, text: str) -> str:
        """Swap placeholders back for their real values (unknown ones untouched)."""
        if not self._by_placeholder:
            return text
        return _PLACEHOLDER_RE.sub(
            lambda m: self._by_placeholder.get(m.group(0), m.group(0)), text
        )


@dataclass(frozen=True)
class RouteNotice:
    """A pending 'routing flipped to the private model' notification."""

    rule_name: str
    model_alias: str


def find_sensitive_match(
    text: str, extra_rules: Sequence[SensitivityRule] = ()
) -> SensitivityRule | None:
    for rule in (*DEFAULT_RULES, *extra_rules):
        if rule.pattern.search(text):
            return rule
    return None


class PrivacyRouter:
    """Sticky sensitivity-based model routing.

    Scans conversation messages incrementally for credential-like content.
    Once anything sensitive lands in the context, every later LLM call is
    routed to the configured private model: content already in the context is
    resent on each call, so the switch must persist until the history is
    cleared, not just for the triggering call.
    """

    def __init__(
        self,
        config_getter: Callable[[], AnyVibeConfig],
        secret_store: PersistentSecretStore | None = None,
    ) -> None:
        self._config = config_getter
        self._scan_index = 0
        self._triggered_rule: SensitivityRule | None = None
        self._pending_notice: RouteNotice | None = None
        self._custom_rules_source: tuple[str, ...] | None = None
        self._custom_rules: tuple[SensitivityRule, ...] = ()
        self._secret_store = secret_store
        self.vault = SecretVault(store=secret_store)

    @property
    def is_sensitive(self) -> bool:
        return self._triggered_rule is not None

    def reset(self) -> None:
        """Forget session sensitivity state when the history is cleared.

        Persistent secrets survive: they are keyed to placeholders that may
        appear in other sessions' logs. A persistent-store-backed vault is
        rebuilt from the keychain; an in-memory vault is dropped entirely.
        """
        self._scan_index = 0
        self._triggered_rule = None
        self._pending_notice = None
        self.vault = SecretVault(store=self._secret_store)

    def consume_notice(self) -> RouteNotice | None:
        notice = self._pending_notice
        self._pending_notice = None
        return notice

    def scan(self, messages: Sequence[LLMMessage]) -> None:
        """Scan messages appended since the last scan; flips sticky state on a match.

        Only meaningful in "route" mode; "redact" mode masks secrets at the wire
        boundary instead of switching models, so no sticky state is needed.
        """
        settings = self._config().privacy_routing
        if (
            not settings.enabled
            or settings.mode != "route"
            or self._triggered_rule is not None
        ):
            return
        if len(messages) < self._scan_index:
            # The history was rewritten (e.g. compaction); rescan from the start.
            self._scan_index = 0
        extra_rules = self._resolve_custom_rules(tuple(settings.custom_patterns))
        for message in messages[self._scan_index :]:
            if not message.content:
                continue
            if rule := find_sensitive_match(message.content, extra_rules):
                self._triggered_rule = rule
                self._pending_notice = RouteNotice(
                    rule_name=rule.name, model_alias=settings.private_model
                )
                logger.info(
                    "Privacy routing engaged (rule=%s); routing to model '%s'.",
                    rule.name,
                    settings.private_model,
                )
                break
        self._scan_index = len(messages)

    def apply(self, model: ModelConfig) -> ModelConfig:
        """Return the model this call must use given the current sticky state."""
        if self._triggered_rule is None:
            return model
        settings = self._config().privacy_routing
        private_model = self._config().models.get(settings.private_model)
        if private_model is None:
            # Config validation prevents this; never fail open silently if it happens.
            raise ValueError(
                f"Privacy routing is engaged but private model "
                f"'{settings.private_model}' is not configured."
            )
        return private_model

    @property
    def redact_mode_active(self) -> bool:
        settings = self._config().privacy_routing
        return settings.enabled and settings.mode == "redact"

    def custom_rules(self) -> tuple[SensitivityRule, ...]:
        """The user's configured custom patterns, compiled and cached."""
        settings = self._config().privacy_routing
        return self._resolve_custom_rules(tuple(settings.custom_patterns))

    def redact_for_wire(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """Mask secrets in every outbound message; the local history keeps real values.

        The full history is re-redacted on every call (not just the new tail):
        earlier messages are resent with each request, and the vault makes
        repeat redaction cheap and placeholder-stable.
        """
        if not self.redact_mode_active:
            return messages
        settings = self._config().privacy_routing
        rules = (
            *DEFAULT_RULES,
            *self._resolve_custom_rules(tuple(settings.custom_patterns)),
        )
        redacted: list[LLMMessage] = []
        for message in messages:
            updates: dict[str, object] = {}
            if message.content:
                masked = self.vault.redact(message.content, rules)
                if masked != message.content:
                    updates["content"] = masked
            if message.tool_calls:
                masked_calls = [
                    self._redact_tool_call(call, rules) for call in message.tool_calls
                ]
                if any(
                    m is not c
                    for m, c in zip(masked_calls, message.tool_calls, strict=True)
                ):
                    updates["tool_calls"] = masked_calls
            redacted.append(message.model_copy(update=updates) if updates else message)
        return redacted

    def rehydrate_tool_arguments(self, arguments: str) -> str:
        """Restore real secrets in tool arguments coming back from the model.

        The model only ever saw placeholders; when it echoes one into a tool
        call (e.g. writing a key to a config file), the real value must be
        substituted locally before the tool runs.
        """
        if not self.redact_mode_active:
            return arguments
        return self.vault.rehydrate(arguments)

    def _redact_tool_call(
        self, call: ToolCall, rules: tuple[SensitivityRule, ...]
    ) -> ToolCall:
        arguments = call.function.arguments
        if not arguments:
            return call
        masked = self.vault.redact(arguments, rules)
        if masked == arguments:
            return call
        return call.model_copy(
            update={"function": call.function.model_copy(update={"arguments": masked})}
        )

    def _resolve_custom_rules(
        self, patterns: tuple[str, ...]
    ) -> tuple[SensitivityRule, ...]:
        if patterns != self._custom_rules_source:
            self._custom_rules = tuple(
                _rule(f"custom-{i}", pattern) for i, pattern in enumerate(patterns)
            )
            self._custom_rules_source = patterns
        return self._custom_rules
