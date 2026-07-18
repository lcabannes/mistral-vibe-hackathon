from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import BUILTIN_AGENTS, AgentType
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.telemetry.types import LaunchContext, TerminalEmulator
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskResult, TaskToolConfig
from vibe.core.tools.permissions import PermissionContext, RequiredPermission
from vibe.core.types import (
    AgentStats,
    ApprovalResponse,
    AssistantEvent,
    LLMMessage,
    Role,
    SubagentLifecycleEvent,
    SubagentLifecycleState,
    ToolCallEvent,
    ToolStreamEvent,
)


@pytest.fixture
def task_tool() -> Task:
    return Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())


class TestTaskArgs:
    def test_default_agent_is_explore(self) -> None:
        args = TaskArgs(task="do something")
        assert args.agent == "explore"

    def test_custom_values(self) -> None:
        args = TaskArgs(task="do something", agent="explore")
        assert args.task == "do something"
        assert args.agent == "explore"


class TestTaskToolValidation:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config()
        manager = AgentManager(LegacyConfigOrchestrator(config))
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            launch_context=LaunchContext(
                agent_entrypoint="cli",
                agent_version="1.0.0",
                client_name="vibe_cli",
                client_version="1.0.0",
                terminal_emulator=TerminalEmulator.VSCODE,
            ),
        )

    @pytest.mark.asyncio
    async def test_rejects_primary_agent(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="do something", agent="default")

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "agent" in str(exc_info.value).lower()
        assert "subagent" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_agent(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="do something", agent="nonexistent")

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "Unknown agent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_requires_agent_manager_in_context(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="explore")
        ctx = InvokeContext(tool_call_id="test-call-id")  # No agent_manager

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "agent_manager" in str(exc_info.value).lower()

    def test_explore_agent_is_valid_subagent(self) -> None:
        agent = BUILTIN_AGENTS["explore"]
        assert agent.agent_type == AgentType.SUBAGENT


class TestTaskToolResolvePermission:
    def test_explore_allowed_by_default(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="explore")
        result = task_tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_unknown_agent_returns_none(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="custom_agent")
        result = task_tool.resolve_permission(args)
        assert result is None

    def test_denylist_takes_precedence(self) -> None:
        config = TaskToolConfig(allowlist=["explore"], denylist=["explore"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_glob_pattern_in_allowlist(self) -> None:
        config = TaskToolConfig(allowlist=["exp*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_glob_pattern_in_denylist(self) -> None:
        config = TaskToolConfig(denylist=["danger*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="dangerous_agent")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_empty_lists_returns_none(self) -> None:
        config = TaskToolConfig(allowlist=[], denylist=[])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert result is None

    def test_default_config_has_explore_in_allowlist(self) -> None:
        config = TaskToolConfig()
        assert "explore" in config.allowlist


class TestTaskToolExecution:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config()
        manager = AgentManager(LegacyConfigOrchestrator(config))
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            launch_context=LaunchContext(
                agent_entrypoint="cli",
                agent_version="1.0.0",
                client_name="vibe_cli",
                client_version="1.0.0",
                terminal_emulator=TerminalEmulator.VSCODE,
            ),
        )

    @pytest.mark.asyncio
    async def test_child_session_metadata_records_parent_session(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        async def mock_act(task: str):
            yield AssistantEvent(content="done")

        ctx.session_id = "parent-session"
        with patch("vibe.core.tools.builtins.task.AgentLoop") as loop_class:
            child = MagicMock()
            child.act = mock_act
            child.session_id = "child-session"
            child.messages = [LLMMessage(role=Role.assistant, content="done")]
            child.stats = AgentStats()
            child.aclose = AsyncMock()
            loop_class.return_value = child

            await collect_result(task_tool.run(TaskArgs(task="work"), ctx))

        child.session_logger.reset_session.assert_called_once_with(
            "child-session", parent_session_id="parent-session"
        )

    @pytest.mark.asyncio
    async def test_happy_path_returns_subagent_response(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool successfully runs a subagent and returns its response."""
        mock_messages = [
            LLMMessage(role=Role.system, content="system"),
            LLMMessage(role=Role.user, content="task"),
            LLMMessage(role=Role.assistant, content="response 1"),
            LLMMessage(role=Role.assistant, content="response 2"),
        ]

        async def mock_act(task: str):
            yield AssistantEvent(content="Hello from subagent!")
            yield AssistantEvent(content=" More content.")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="explore the codebase", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.response == "Hello from subagent! More content."
            assert result.turns_used == 2  # 2 assistant messages in mock_messages
            assert result.completed is True
            assert (
                mock_agent_loop_class.call_args.kwargs[
                    "launch_context"
                ].terminal_emulator
                is TerminalEmulator.VSCODE
            )

    @pytest.mark.asyncio
    async def test_real_child_callbacks_emit_attention_and_recover_before_terminal(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        stats = AgentStats(session_prompt_tokens=12, session_completion_tokens=4)
        approval_release = asyncio.Event()
        user_input_release = asyncio.Event()
        approval_returned = False
        user_input_returned = False
        approval_calls: list[
            tuple[str, BaseModel, str, list[RequiredPermission] | None]
        ] = []
        approval_results: list[tuple[ApprovalResponse, str | None]] = []
        user_input_result = TaskArgs(task="answered")
        user_input_results: list[BaseModel] = []

        async def approval_callback(
            tool_name: str,
            callback_args: BaseModel,
            tool_call_id: str,
            required_permissions: list[RequiredPermission] | None,
        ) -> tuple[ApprovalResponse, str | None]:
            nonlocal approval_returned
            approval_calls.append((
                tool_name,
                callback_args,
                tool_call_id,
                required_permissions,
            ))
            await approval_release.wait()
            approval_returned = True
            return ApprovalResponse.NO, "not now"

        async def user_input_callback(callback_args: BaseModel) -> BaseModel:
            nonlocal user_input_returned
            await user_input_release.wait()
            user_input_returned = True
            return user_input_result

        ctx.approval_callback = approval_callback
        ctx.user_input_callback = user_input_callback

        async def mock_act(task: str):
            yield ToolCallEvent(
                tool_call_id="child-tool-call",
                tool_name="task",
                tool_class=Task,
                args=TaskArgs(task="inspect nested call"),
            )
            approval_results.append(
                await mock_agent_loop.approval_callback(
                    "write_file", TaskArgs(task="write"), "child-approval", None
                )
            )
            user_input_results.append(
                await mock_agent_loop.user_input_callback(TaskArgs(task="question"))
            )
            yield AssistantEvent(content="stopped", stopped_by_middleware=True)

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = [
                LLMMessage(role=Role.assistant, content="stopped")
            ]
            mock_agent_loop.session_id = "child-session"
            mock_agent_loop.stats = stats
            mock_agent_loop.aclose = AsyncMock()
            mock_agent_loop.set_approval_callback = MagicMock(
                side_effect=lambda callback: setattr(
                    mock_agent_loop, "approval_callback", callback
                )
            )
            mock_agent_loop.set_user_input_callback = MagicMock(
                side_effect=lambda callback: setattr(
                    mock_agent_loop, "user_input_callback", callback
                )
            )
            mock_agent_loop_class.return_value = mock_agent_loop

            events: list[ToolStreamEvent | TaskResult] = []
            async for event in task_tool.run(
                TaskArgs(task="explore the codebase"), ctx
            ):
                events.append(event)
                if not isinstance(event, SubagentLifecycleEvent):
                    continue
                if event.current_activity == "Approval needed for write_file":
                    assert not approval_returned
                    approval_release.set()
                elif event.current_activity == "Waiting for user input":
                    assert not user_input_returned
                    user_input_release.set()

        lifecycle = [
            event for event in events if isinstance(event, SubagentLifecycleEvent)
        ]
        assert [event.state for event in lifecycle] == [
            SubagentLifecycleState.RUNNING,
            SubagentLifecycleState.WORKING,
            SubagentLifecycleState.ATTENTION,
            SubagentLifecycleState.WORKING,
            SubagentLifecycleState.ATTENTION,
            SubagentLifecycleState.WORKING,
            SubagentLifecycleState.CANCELLED,
        ]
        assert {event.tool_call_id for event in lifecycle} == {"test-call-id"}
        assert {event.agent_name for event in lifecycle} == {"explore"}
        assert lifecycle[0].agent_display_name == "Explore"
        assert lifecycle[0].task == "explore the codebase"
        assert lifecycle[0].child_session_id == "child-session"
        assert lifecycle[1].current_activity == (
            "Running explore agent: inspect nested call"
        )
        assert lifecycle[3].current_activity == lifecycle[1].current_activity
        assert lifecycle[5].current_activity == lifecycle[1].current_activity
        assert lifecycle[-1].terminal_usage is not None
        assert lifecycle[-1].terminal_usage.prompt_tokens == 12
        assert lifecycle[-1].terminal_usage.completion_tokens == 4
        assert approval_calls[0][0] == "write_file"
        assert approval_calls[0][2] == "child-approval"
        assert approval_results == [(ApprovalResponse.NO, "not now")]
        assert user_input_results == [user_input_result]

    @pytest.mark.asyncio
    async def test_handles_stopped_by_middleware(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool reports incomplete when stopped by middleware."""
        mock_messages = [
            LLMMessage(role=Role.system, content="system"),
            LLMMessage(role=Role.assistant, content="partial"),
        ]

        async def mock_act(task: str):
            yield AssistantEvent(content="Partial response", stopped_by_middleware=True)

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.completed is False

    @pytest.mark.asyncio
    async def test_handles_subagent_exception(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool gracefully handles exceptions from subagent."""
        mock_messages = [LLMMessage(role=Role.system, content="system")]

        async def mock_act(task: str):
            yield AssistantEvent(content="Starting...")
            raise RuntimeError("Simulated error")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.completed is False
            assert "Simulated error" in result.response

    @pytest.mark.asyncio
    async def test_exception_emits_failed_lifecycle(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        async def mock_act(task: str):
            raise RuntimeError("boom")
            yield

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = []
            mock_agent_loop.session_id = "child-session"
            mock_agent_loop.stats = AgentStats()
            mock_agent_loop.aclose = AsyncMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            states = []
            async for event in task_tool.run(TaskArgs(task="fail"), ctx):
                if isinstance(event, SubagentLifecycleEvent):
                    states.append(event.state)

        assert states == [SubagentLifecycleState.RUNNING, SubagentLifecycleState.FAILED]

    @pytest.mark.asyncio
    async def test_closes_subagent_loop_on_success(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        async def mock_act(task: str):
            yield AssistantEvent(content="done")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = [LLMMessage(role=Role.assistant, content="a")]
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop.aclose = AsyncMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            await collect_result(task_tool.run(args, ctx))

            mock_agent_loop.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_subagent_loop_on_exception(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        async def mock_act(task: str):
            yield AssistantEvent(content="starting")
            raise RuntimeError("boom")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = [LLMMessage(role=Role.assistant, content="a")]
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop.aclose = AsyncMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            await collect_result(task_tool.run(args, ctx))

            mock_agent_loop.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_subagent_loop_on_cancellation(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        callback_started = asyncio.Event()
        callback_cancelled = asyncio.Event()

        async def approval_callback(
            tool_name: str,
            callback_args: BaseModel,
            tool_call_id: str,
            required_permissions: list[RequiredPermission] | None,
        ) -> tuple[ApprovalResponse, str | None]:
            callback_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                callback_cancelled.set()
                raise
            return ApprovalResponse.YES, None

        ctx.approval_callback = approval_callback

        async def mock_act(task: str):
            yield ToolCallEvent(
                tool_call_id="child-tool-call",
                tool_name="task",
                tool_class=Task,
                args=TaskArgs(task="wait for approval"),
            )
            await mock_agent_loop.approval_callback(
                "write_file", TaskArgs(task="write"), "child-approval", None
            )
            yield AssistantEvent(content="unreachable")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = []
            mock_agent_loop.session_id = "child-session"
            mock_agent_loop.stats = AgentStats()
            mock_agent_loop.aclose = AsyncMock()
            mock_agent_loop.set_approval_callback = MagicMock(
                side_effect=lambda callback: setattr(
                    mock_agent_loop, "approval_callback", callback
                )
            )
            mock_agent_loop_class.return_value = mock_agent_loop

            run_task = asyncio.create_task(
                collect_result(task_tool.run(TaskArgs(task="wait"), ctx))
            )
            await callback_started.wait()
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

            assert callback_cancelled.is_set()
            mock_agent_loop.aclose.assert_awaited_once()
