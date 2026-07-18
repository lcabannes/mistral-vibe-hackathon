from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
import fnmatch

from pydantic import BaseModel, Field

from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import AgentType, BuiltinAgentName
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext, RequiredPermission
from vibe.core.tools.ui import (
    ToolCallDisplay,
    ToolResultDisplay,
    ToolUIData,
    ToolUIDataAdapter,
)
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    LLMUsage,
    Role,
    SubagentLifecycleEvent,
    SubagentLifecycleState,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    WaitingForInputEvent,
)


@dataclass
class _TaskExecution:
    response_parts: list[str] = field(default_factory=list)
    completed: bool = True
    terminal_state: SubagentLifecycleState = SubagentLifecycleState.COMPLETED
    active_state: SubagentLifecycleState = SubagentLifecycleState.RUNNING
    current_activity: str | None = None


class TaskArgs(BaseModel):
    task: str = Field(description="The task for the agent to perform")
    agent: str = Field(
        default="explore",
        description="The type of specialized subagent to use for this task",
    )


class TaskResult(BaseModel):
    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int = Field(description="Number of turns the subagent used")
    completed: bool = Field(description="Whether the task completed normally")


class TaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default=[BuiltinAgentName.EXPLORE])


class Task(
    BaseTool[TaskArgs, TaskResult, TaskToolConfig, BaseToolState],
    ToolUIData[TaskArgs, TaskResult],
):
    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, TaskArgs):
            return ToolCallDisplay(summary=f"Running {args.agent} agent: {args.task}")
        return ToolCallDisplay(summary="Running subagent")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TaskResult):
            turn_word = "turn" if result.turns_used == 1 else "turns"
            if not result.completed:
                return ToolResultDisplay(
                    success=False,
                    message=f"Agent interrupted after {result.turns_used} {turn_word}",
                )
            return ToolResultDisplay(
                success=True,
                message=f"Agent completed in {result.turns_used} {turn_word}",
            )
        return ToolResultDisplay(success=True, message="Agent completed")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running subagent"

    def resolve_permission(self, args: TaskArgs) -> PermissionContext | None:
        agent_name = args.agent

        for pattern in self.config.denylist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.NEVER)

        for pattern in self.config.allowlist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.ALWAYS)

        return None

    @staticmethod
    def _terminal_usage(subagent_loop: AgentLoop) -> LLMUsage | None:
        prompt_tokens = subagent_loop.stats.session_prompt_tokens
        completion_tokens = subagent_loop.stats.session_completion_tokens
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return None
        return LLMUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    def _lifecycle_event(
        self,
        *,
        ctx: InvokeContext,
        args: TaskArgs,
        agent_display_name: str,
        child_session_id: str,
        state: SubagentLifecycleState,
        current_activity: str | None = None,
        terminal_usage: LLMUsage | None = None,
    ) -> SubagentLifecycleEvent:
        return SubagentLifecycleEvent(
            tool_name=self.get_name(),
            message=current_activity or state.value,
            tool_call_id=ctx.tool_call_id,
            agent_name=args.agent,
            agent_display_name=agent_display_name,
            task=args.task,
            child_session_id=child_session_id,
            state=state,
            current_activity=current_activity,
            terminal_usage=terminal_usage,
        )

    def _child_progress_event(
        self,
        event: BaseEvent,
        execution: _TaskExecution,
        *,
        ctx: InvokeContext,
        args: TaskArgs,
        agent_display_name: str,
        child_session_id: str,
    ) -> ToolStreamEvent | None:
        progress: ToolStreamEvent | None = None
        if isinstance(event, AssistantEvent) and event.content:
            execution.response_parts.append(event.content)
            if event.stopped_by_middleware:
                execution.completed = False
                execution.terminal_state = SubagentLifecycleState.CANCELLED
        elif isinstance(event, ToolCallEvent):
            adapter = ToolUIDataAdapter(event.tool_class)
            execution.active_state = SubagentLifecycleState.WORKING
            execution.current_activity = adapter.get_call_display(event).summary
            progress = self._lifecycle_event(
                ctx=ctx,
                args=args,
                agent_display_name=agent_display_name,
                child_session_id=child_session_id,
                state=SubagentLifecycleState.WORKING,
                current_activity=execution.current_activity,
            )
        elif isinstance(event, ToolResultEvent):
            if event.skipped:
                execution.completed = False
                execution.terminal_state = SubagentLifecycleState.CANCELLED
            elif event.result and event.tool_class:
                adapter = ToolUIDataAdapter(event.tool_class)
                display = adapter.get_result_display(event)
                progress = ToolStreamEvent(
                    tool_name=self.get_name(),
                    message=f"{event.tool_name}: {display.message}",
                    tool_call_id=ctx.tool_call_id,
                )
        elif isinstance(event, WaitingForInputEvent):
            progress = self._lifecycle_event(
                ctx=ctx,
                args=args,
                agent_display_name=agent_display_name,
                child_session_id=child_session_id,
                state=SubagentLifecycleState.ATTENTION,
                current_activity=event.label or "Waiting for input",
            )
        return progress

    def _set_tracked_callbacks(
        self,
        *,
        subagent_loop: AgentLoop,
        ctx: InvokeContext,
        args: TaskArgs,
        agent_display_name: str,
        child_session_id: str,
        execution: _TaskExecution,
        progress_queue: asyncio.Queue[ToolStreamEvent | None],
    ) -> None:
        if ctx.approval_callback:
            approval_callback = ctx.approval_callback

            async def tracked_approval_callback(
                tool_name: str,
                callback_args: BaseModel,
                tool_call_id: str,
                required_permissions: list[RequiredPermission] | None,
            ) -> tuple[ApprovalResponse, str | None]:
                progress_queue.put_nowait(
                    self._lifecycle_event(
                        ctx=ctx,
                        args=args,
                        agent_display_name=agent_display_name,
                        child_session_id=child_session_id,
                        state=SubagentLifecycleState.ATTENTION,
                        current_activity=f"Approval needed for {tool_name}",
                    )
                )
                result = await approval_callback(
                    tool_name, callback_args, tool_call_id, required_permissions
                )
                progress_queue.put_nowait(
                    self._lifecycle_event(
                        ctx=ctx,
                        args=args,
                        agent_display_name=agent_display_name,
                        child_session_id=child_session_id,
                        state=execution.active_state,
                        current_activity=execution.current_activity,
                    )
                )
                return result

            subagent_loop.set_approval_callback(tracked_approval_callback)

        if ctx.user_input_callback:
            user_input_callback = ctx.user_input_callback

            async def tracked_user_input_callback(
                callback_args: BaseModel,
            ) -> BaseModel:
                progress_queue.put_nowait(
                    self._lifecycle_event(
                        ctx=ctx,
                        args=args,
                        agent_display_name=agent_display_name,
                        child_session_id=child_session_id,
                        state=SubagentLifecycleState.ATTENTION,
                        current_activity="Waiting for user input",
                    )
                )
                result = await user_input_callback(callback_args)
                progress_queue.put_nowait(
                    self._lifecycle_event(
                        ctx=ctx,
                        args=args,
                        agent_display_name=agent_display_name,
                        child_session_id=child_session_id,
                        state=execution.active_state,
                        current_activity=execution.current_activity,
                    )
                )
                return result

            subagent_loop.set_user_input_callback(tracked_user_input_callback)

    async def _consume_child_events(
        self,
        *,
        subagent_loop: AgentLoop,
        task_text: str,
        execution: _TaskExecution,
        progress_queue: asyncio.Queue[ToolStreamEvent | None],
        ctx: InvokeContext,
        args: TaskArgs,
        agent_display_name: str,
        child_session_id: str,
    ) -> None:
        try:
            async with aclosing(subagent_loop.act(task_text)) as events:
                async for event in events:
                    progress = self._child_progress_event(
                        event,
                        execution,
                        ctx=ctx,
                        args=args,
                        agent_display_name=agent_display_name,
                        child_session_id=child_session_id,
                    )
                    if progress is not None:
                        progress_queue.put_nowait(progress)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            execution.completed = False
            execution.terminal_state = SubagentLifecycleState.FAILED
            execution.response_parts.append(f"\n[Subagent error: {e}]")
        finally:
            progress_queue.put_nowait(None)

    async def run(
        self, args: TaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if not ctx or not ctx.agent_manager:
            raise ToolError("Task tool requires agent_manager in context")

        agent_manager = ctx.agent_manager

        producer: asyncio.Task[None] | None = None
        try:
            agent_profile = agent_manager.get_agent(args.agent)
        except ValueError as e:
            raise ToolError(f"Unknown agent: {args.agent}") from e

        if agent_profile.agent_type != AgentType.SUBAGENT:
            raise ToolError(
                f"Agent '{args.agent}' is a {agent_profile.agent_type.value} agent. "
                f"Only subagents can be used with the task tool. "
                f"This is a security constraint to prevent recursive spawning."
            )

        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=args.agent,
            enabled=ctx.session_dir is not None,
        )
        base_config = VibeConfig.load(session_logging=session_logging)
        subagent_loop = AgentLoop(
            config_orchestrator=LegacyConfigOrchestrator(base_config),
            agent_name=args.agent,
            launch_context=ctx.launch_context,
            is_subagent=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store,
            hook_config_result=ctx.hook_config_result,
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id
            subagent_loop.session_logger.reset_session(
                subagent_loop.session_id, parent_session_id=ctx.session_id
            )

        child_session_id = str(subagent_loop.session_id)
        agent_display_name = agent_profile.display_name
        execution = _TaskExecution()
        progress_queue: asyncio.Queue[ToolStreamEvent | None] = asyncio.Queue()
        self._set_tracked_callbacks(
            subagent_loop=subagent_loop,
            ctx=ctx,
            args=args,
            agent_display_name=agent_display_name,
            child_session_id=child_session_id,
            execution=execution,
            progress_queue=progress_queue,
        )

        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
            )

        try:
            yield self._lifecycle_event(
                ctx=ctx,
                args=args,
                agent_display_name=agent_display_name,
                child_session_id=child_session_id,
                state=SubagentLifecycleState.RUNNING,
            )
            producer = asyncio.create_task(
                self._consume_child_events(
                    subagent_loop=subagent_loop,
                    task_text=task_text,
                    execution=execution,
                    progress_queue=progress_queue,
                    ctx=ctx,
                    args=args,
                    agent_display_name=agent_display_name,
                    child_session_id=child_session_id,
                )
            )
            while (progress := await progress_queue.get()) is not None:
                yield progress
            await producer
        finally:
            if producer is not None and not producer.done():
                producer.cancel()
            if producer is not None:
                with suppress(asyncio.CancelledError):
                    await producer
            with suppress(Exception):
                await subagent_loop.aclose()

        turns_used = sum(msg.role == Role.assistant for msg in subagent_loop.messages)
        yield self._lifecycle_event(
            ctx=ctx,
            args=args,
            agent_display_name=agent_display_name,
            child_session_id=child_session_id,
            state=execution.terminal_state,
            terminal_usage=self._terminal_usage(subagent_loop),
        )
        yield TaskResult(
            response="".join(execution.response_parts),
            turns_used=turns_used,
            completed=execution.completed,
        )
