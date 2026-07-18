from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.config import (
    ModelConfig,
    PrivacyRoutingConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.local_model import ensure_local_model_ready
from vibe.core.types import Backend

_MODULE = "vibe.core.local_model"


def _privacy_config(**privacy_kwargs) -> VibeConfig:
    models = [
        ModelConfig(name="devstral-latest", provider="mistral", alias="cloud"),
        ModelConfig(name="devstral", provider="local", alias="private"),
    ]
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        ),
        ProviderConfig(name="local", api_base="http://127.0.0.1:8080/v1"),
    ]
    privacy_kwargs.setdefault("enabled", True)
    privacy_kwargs.setdefault("mode", "redact")
    privacy_kwargs.setdefault("private_model", "private")
    privacy_kwargs.setdefault("warmup", False)
    return build_test_vibe_config(
        active_model="cloud",
        models=models,
        providers=providers,
        privacy_routing=PrivacyRoutingConfig(**privacy_kwargs),
    )


class TestEnsureLocalModelReady:
    @pytest.mark.asyncio
    async def test_not_configured_without_privacy_routing(self):
        report = await ensure_local_model_ready(build_test_vibe_config())
        assert report.status == "not_configured"

    @pytest.mark.asyncio
    async def test_ready_when_endpoint_alive(self):
        config = _privacy_config()
        with patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=True)):
            report = await ensure_local_model_ready(config)
        assert report.status == "ready"
        assert report.endpoint == "http://127.0.0.1:8080/v1"

    @pytest.mark.asyncio
    async def test_unreachable_without_server_command(self):
        config = _privacy_config()
        with patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=False)):
            report = await ensure_local_model_ready(config)
        assert report.status == "unreachable"
        assert "local_server_command" in report.detail

    @pytest.mark.asyncio
    async def test_starts_server_and_waits_for_readiness(self):
        config = _privacy_config(local_server_command="ollama serve")
        alive = AsyncMock(side_effect=[False, False, True])
        with (
            patch(f"{_MODULE}._endpoint_alive", alive),
            patch(f"{_MODULE}._spawn_server", return_value=None) as spawn,
        ):
            report = await ensure_local_model_ready(config)
        assert report.status == "started"
        spawn.assert_called_once_with("ollama serve")

    @pytest.mark.asyncio
    async def test_startup_timeout_reports_unreachable(self):
        config = _privacy_config(
            local_server_command="ollama serve", local_server_startup_timeout=0.0
        )
        with (
            patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=False)),
            patch(f"{_MODULE}._spawn_server", return_value=None),
        ):
            report = await ensure_local_model_ready(config)
        assert report.status == "unreachable"
        assert "did not come up" in report.detail

    @pytest.mark.asyncio
    async def test_spawn_failure_reports_unreachable(self):
        config = _privacy_config(local_server_command="nonexistent-binary serve")
        with (
            patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=False)),
            patch(
                f"{_MODULE}._spawn_server",
                return_value="Could not start local model server: not found",
            ),
        ):
            report = await ensure_local_model_ready(config)
        assert report.status == "unreachable"
        assert "Could not start" in report.detail

    @pytest.mark.asyncio
    async def test_warmup_fires_when_enabled(self):
        config = _privacy_config(warmup=True)
        warmup = AsyncMock()
        with (
            patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=True)),
            patch(f"{_MODULE}._warmup", warmup),
        ):
            report = await ensure_local_model_ready(config)
        assert report.status == "ready"
        warmup.assert_awaited_once_with("http://127.0.0.1:8080/v1", "devstral")

    @pytest.mark.asyncio
    async def test_no_warmup_when_disabled(self):
        config = _privacy_config(warmup=False)
        warmup = AsyncMock()
        with (
            patch(f"{_MODULE}._endpoint_alive", AsyncMock(return_value=True)),
            patch(f"{_MODULE}._warmup", warmup),
        ):
            await ensure_local_model_ready(config)
        warmup.assert_not_awaited()
