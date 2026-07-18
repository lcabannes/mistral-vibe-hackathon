from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.config import (
    ModelConfig,
    PrivacyRoutingConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.local_task import (
    LocalTask,
    LocalTaskArgs,
    LocalTaskResult,
    LocalTaskToolConfig,
)
from vibe.core.types import AssistantEvent, Backend, LLMMessage, Role, ToolStreamEvent

SECRET_CONTENT = "DB_PASSWORD=hunter2hunter2"


def _privacy_config(**privacy_kwargs) -> VibeConfig:
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
    privacy_kwargs.setdefault("mode", "redact")
    privacy_kwargs.setdefault("private_model", "private")
    return build_test_vibe_config(
        active_model="cloud",
        models=models,
        providers=providers,
        privacy_routing=PrivacyRoutingConfig(**privacy_kwargs),
    )


@pytest.fixture
def local_task_tool() -> LocalTask:
    return LocalTask(config_getter=lambda: LocalTaskToolConfig(), state=BaseToolState())


@pytest.fixture
def ctx() -> InvokeContext:
    return InvokeContext(tool_call_id="test-call-id")


class TestAvailability:
    def test_available_when_privacy_routing_configured(self):
        assert LocalTask.is_available(_privacy_config())

    def test_unavailable_without_privacy_routing(self):
        assert not LocalTask.is_available(build_test_vibe_config())

    def test_unavailable_without_private_model(self):
        config = _privacy_config(private_model="", mode="redact")
        assert not LocalTask.is_available(config)


class TestReturnContract:
    """The privacy boundary: nothing content-derived crosses back."""

    @pytest.mark.asyncio
    async def test_returns_only_status_never_content(
        self, local_task_tool: LocalTask, ctx: InvokeContext
    ):
        mock_messages = [
            LLMMessage(role=Role.system, content="system"),
            LLMMessage(role=Role.user, content="task"),
            LLMMessage(role=Role.assistant, content=SECRET_CONTENT),
        ]

        async def mock_act(task: str):
            yield AssistantEvent(content=f"The env file contains {SECRET_CONTENT}")

        with (
            patch(
                "vibe.core.tools.builtins.local_task.VibeConfig.load",
                return_value=_privacy_config(),
            ),
            patch("vibe.core.tools.builtins.local_task.AgentLoop") as mock_loop_class,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = mock_messages
            mock_loop_class.return_value = mock_loop

            args = LocalTaskArgs(task="check the env file")
            result = await collect_result(local_task_tool.run(args, ctx))

        assert isinstance(result, LocalTaskResult)
        assert result.completed is True
        assert result.turns_used == 1
        # The result model has exactly two fields — no channel for content.
        assert set(LocalTaskResult.model_fields) == {"completed", "turns_used"}

    def test_result_extra_tells_orchestrator_not_to_fabricate(
        self, local_task_tool: LocalTask
    ):
        extra = local_task_tool.get_result_extra(
            LocalTaskResult(completed=True, turns_used=1)
        )
        assert extra is not None
        assert "do not summarize" in extra

    @pytest.mark.asyncio
    async def test_streams_activity_to_ui(
        self, local_task_tool: LocalTask, ctx: InvokeContext
    ):
        async def mock_act(task: str):
            yield AssistantEvent(content="working on it")

        with (
            patch(
                "vibe.core.tools.builtins.local_task.VibeConfig.load",
                return_value=_privacy_config(),
            ),
            patch("vibe.core.tools.builtins.local_task.AgentLoop") as mock_loop_class,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop_class.return_value = mock_loop

            args = LocalTaskArgs(task="do the thing")
            stream_events = []
            result = None
            async for item in local_task_tool.run(args, ctx):
                if isinstance(item, ToolStreamEvent):
                    stream_events.append(item)
                else:
                    result = item

        # The local model's answer is visible to the user as a first-class
        # (prominent) message...
        answer_events = [e for e in stream_events if "working on it" in e.message]
        assert answer_events
        assert all(e.prominent for e in answer_events)
        # ...but the returned result carries none of it.
        assert isinstance(result, LocalTaskResult)

    @pytest.mark.asyncio
    async def test_exception_reports_failure_without_details(
        self, local_task_tool: LocalTask, ctx: InvokeContext
    ):
        async def mock_act(task: str):
            yield AssistantEvent(content="starting")
            raise RuntimeError(f"crashed while reading {SECRET_CONTENT}")

        with (
            patch(
                "vibe.core.tools.builtins.local_task.VibeConfig.load",
                return_value=_privacy_config(),
            ),
            patch("vibe.core.tools.builtins.local_task.AgentLoop") as mock_loop_class,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop_class.return_value = mock_loop

            args = LocalTaskArgs(task="doomed task")
            result = await collect_result(local_task_tool.run(args, ctx))

        assert isinstance(result, LocalTaskResult)
        assert result.completed is False
        # The exception text (which could quote protected content) is not
        # part of the result model at all.
        assert SECRET_CONTENT not in str(result.model_dump())


class TestConfiguration:
    @pytest.mark.asyncio
    async def test_requires_privacy_routing_enabled(
        self, local_task_tool: LocalTask, ctx: InvokeContext
    ):
        with patch(
            "vibe.core.tools.builtins.local_task.VibeConfig.load",
            return_value=build_test_vibe_config(),
        ):
            args = LocalTaskArgs(task="anything")
            with pytest.raises(ToolError, match="privacy_routing"):
                await collect_result(local_task_tool.run(args, ctx))

    @pytest.mark.asyncio
    async def test_subagent_pinned_to_private_model(
        self, local_task_tool: LocalTask, ctx: InvokeContext
    ):
        async def mock_act(task: str):
            yield AssistantEvent(content="ok")

        with (
            patch(
                "vibe.core.tools.builtins.local_task.VibeConfig.load",
                return_value=_privacy_config(),
            ),
            patch("vibe.core.tools.builtins.local_task.AgentLoop") as mock_loop_class,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop_class.return_value = mock_loop

            args = LocalTaskArgs(task="anything")
            await collect_result(local_task_tool.run(args, ctx))

            orchestrator = mock_loop_class.call_args.kwargs["config_orchestrator"]
            subagent_config = orchestrator.config
            assert subagent_config.active_model == "private"
            # Privacy routing is off inside the trust zone (and prevents
            # recursive local_task spawning).
            assert not subagent_config.privacy_routing.enabled
            assert mock_loop_class.call_args.kwargs["bypass_path_guard"] is True
            # Small local models choke on the ~5k-token CLI prompt (slow
            # prefill, instruction echo); the subagent runs lean.
            assert subagent_config.system_prompt_id == "minimal"
            assert not subagent_config.include_prompt_detail
