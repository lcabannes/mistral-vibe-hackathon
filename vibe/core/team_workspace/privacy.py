from __future__ import annotations

import re

from vibe.core.pii import scrub_paths

MAX_SHARED_MESSAGE_LENGTH = 500

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)"
    r"\s*[:=]\s*[^\s,;]+"
)
_ENV_RE = re.compile(r"(?m)^[A-Z][A-Z0-9_]{2,}\s*=.*$")
_COMMAND_RE = re.compile(r"(?m)^\s*(?:\$|>|%)\s+.*$")
_SLASH_COMMAND_RE = re.compile(r"(?m)^\s*/[A-Za-z][^\r\n]*$")


def sanitize_shared_message(value: str) -> str:
    text = _FENCED_CODE_RE.sub("[Code omitted]", value)
    text = _PRIVATE_KEY_RE.sub("[Private key omitted]", text)
    text = _BEARER_RE.sub("Bearer [Filtered]", text)
    text = _SECRET_RE.sub(r"\1=[Filtered]", text)
    text = _ENV_RE.sub("[Environment value omitted]", text)
    text = _COMMAND_RE.sub("[Command omitted]", text)
    text = _SLASH_COMMAND_RE.sub("[Command omitted]", text)
    scrubbed = scrub_paths(text)
    if not isinstance(scrubbed, str):  # pragma: no cover - input is always a string
        return "[Filtered]"
    normalized = "\n".join(line.rstrip() for line in scrubbed.strip().splitlines())
    return (normalized or "[Redacted]")[:MAX_SHARED_MESSAGE_LENGTH]
