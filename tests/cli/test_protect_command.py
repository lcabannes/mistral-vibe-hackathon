from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.core.config import (
    ModelConfig,
    PrivacyRoutingConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.types import Backend


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
        ProviderConfig(name="local", api_base="http://127.0.0.1:11434/v1"),
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


@pytest.mark.asyncio
async def test_protect_lists_defaults_and_user_paths() -> None:
    config = _privacy_config(protected_paths=["contracts/**"])
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        handled = await app._handle_command("/protect")
        await pilot.pause()
        messages = app.query(UserCommandMessage)
        contents = [m._content for m in messages]

    assert handled is True
    listing = next(c for c in contents if "Protected paths" in c)
    assert "~/.ssh/**" in listing
    assert "contracts/**" in listing


@pytest.mark.asyncio
async def test_protect_add_persists_pattern() -> None:
    config = _privacy_config()
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        await app._handle_command("/protect payroll/**")
        await pilot.pause()

    assert "payroll/**" in app.agent_loop.config.privacy_routing.protected_paths


@pytest.mark.asyncio
async def test_protect_remove_deletes_pattern() -> None:
    config = _privacy_config(protected_paths=["payroll/**"])
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        await app._handle_command("/protect remove payroll/**")
        await pilot.pause()

    assert "payroll/**" not in app.agent_loop.config.privacy_routing.protected_paths


@pytest.mark.asyncio
async def test_protect_remove_unknown_pattern_errors() -> None:
    config = _privacy_config()
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        await app._handle_command("/protect remove nothere/**")
        await pilot.pause()
        errors = app.query(ErrorMessage)
        assert any("not in your protected paths" in str(e._error) for e in errors)


@pytest.mark.asyncio
async def test_protect_requires_privacy_routing_enabled() -> None:
    config = build_test_vibe_config()
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        await app._handle_command("/protect contracts/**")
        await pilot.pause()
        errors = app.query(ErrorMessage)
        assert any("Privacy routing is disabled" in str(e._error) for e in errors)


@pytest.mark.asyncio
async def test_prominent_stream_renders_full_local_response() -> None:
    from vibe.cli.textual_ui.widgets.messages import LocalModelHeader, LocalModelMessage
    from vibe.core.types import ToolStreamEvent

    config = _privacy_config()
    app = build_test_vibe_app(config=config)

    long_text = (
        "Here is the summary of the file. It contains a discussion about "
        "psychohistory, ethics, and the formation of a guild to oversee it."
    )
    async with app.run_test() as pilot:
        handler = app.event_handler
        # Simulate the local model streaming its answer in chunks.
        for chunk in (long_text[:20], long_text[20:70], long_text[70:]):
            await handler.handle_event(
                ToolStreamEvent(
                    tool_name="local_task",
                    message=chunk,
                    tool_call_id="call-1",
                    prominent=True,
                )
            )
        await handler.finalize_streaming()
        await pilot.pause()

        headers = app.query(LocalModelHeader)
        assert len(list(headers)) == 1
        widgets = list(app.query(LocalModelMessage))
        assert len(widgets) == 1
        # The full text must be retained, not just the first chunk.
        assert widgets[0].get_content() == long_text
