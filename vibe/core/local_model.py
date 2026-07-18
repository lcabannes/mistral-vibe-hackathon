from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import shlex
import subprocess
from typing import TYPE_CHECKING, Literal

import httpx

if TYPE_CHECKING:
    from vibe.core.config import AnyVibeConfig, ModelConfig, ProviderConfig

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 3.0
_WARMUP_TIMEOUT = 120.0  # first request may load model weights from disk
_POLL_INTERVAL = 0.5

LocalModelStatus = Literal["ready", "started", "unreachable", "not_configured"]


@dataclass(frozen=True)
class LocalModelReport:
    status: LocalModelStatus
    endpoint: str = ""
    detail: str = ""


def _resolve_private_provider(
    config: AnyVibeConfig,
) -> tuple[ModelConfig, ProviderConfig] | None:
    settings = config.privacy_routing
    if not settings.enabled or not settings.private_model:
        return None
    model = config.models.get(settings.private_model)
    if model is None:
        return None
    try:
        provider = config.get_provider_for_model(model)
    except ValueError:
        return None
    return model, provider


async def _endpoint_alive(api_base: str) -> bool:
    url = f"{api_base.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            response = await client.get(url)
    except httpx.HTTPError:
        return False
    # Any non-server-error means something is listening; llama.cpp and Ollama
    # both serve /v1/models, but a 404 from a bare server still proves liveness.
    return not response.is_server_error


async def _warmup(api_base: str, model_name: str) -> None:
    """One-token completion to force the server to load model weights."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=_WARMUP_TIMEOUT) as client:
            await client.post(url, json=payload)
    except httpx.HTTPError as e:
        logger.info("Local model warmup did not complete: %s", e)


def _spawn_server(command: str) -> str | None:
    """Start the local server detached; returns an error string on failure."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"Invalid local_server_command: {e}"
    if not argv:
        return "local_server_command is empty."
    try:
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # survives Vibe exiting
        )
    except OSError as e:
        return f"Could not start local model server: {e}"
    return None


async def ensure_local_model_ready(config: AnyVibeConfig) -> LocalModelReport:
    """Health-check (and optionally start and warm) the private model server.

    Called in the background at session start so the first sensitive
    operation doesn't collide with a dead endpoint or cold weights. Never
    raises: reports status for the UI to surface.
    """
    resolved = _resolve_private_provider(config)
    if resolved is None:
        return LocalModelReport(status="not_configured")
    model, provider = resolved
    settings = config.privacy_routing
    endpoint = provider.api_base

    started = False
    if not await _endpoint_alive(endpoint):
        if not settings.local_server_command:
            return LocalModelReport(
                status="unreachable",
                endpoint=endpoint,
                detail=(
                    f"No server is answering at {endpoint}. Start it manually, "
                    f"or set privacy_routing.local_server_command to have Vibe "
                    f"launch it (make sure the command binds the same port as "
                    f"the provider's api_base)."
                ),
            )
        if error := _spawn_server(settings.local_server_command):
            return LocalModelReport(
                status="unreachable", endpoint=endpoint, detail=error
            )
        started = True
        deadline = (
            asyncio.get_running_loop().time() + settings.local_server_startup_timeout
        )
        while not await _endpoint_alive(endpoint):
            if asyncio.get_running_loop().time() > deadline:
                return LocalModelReport(
                    status="unreachable",
                    endpoint=endpoint,
                    detail=(
                        f"Started `{settings.local_server_command}` but "
                        f"{endpoint} did not come up within "
                        f"{settings.local_server_startup_timeout:.0f}s. Check "
                        f"that the command binds the port in the provider's "
                        f"api_base."
                    ),
                )
            await asyncio.sleep(_POLL_INTERVAL)

    if settings.warmup:
        await _warmup(endpoint, model.name)

    return LocalModelReport(status="started" if started else "ready", endpoint=endpoint)
