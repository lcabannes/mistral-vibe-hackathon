from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from pathlib import Path
import re

from pydantic import BaseModel, Field

from vibe.core.agent_loop import AgentLoop
from vibe.core.config import AnyVibeConfig, SessionLoggingConfig, VibeConfig
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.prompts import SystemPrompt
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import (
    ToolCallDisplay,
    ToolResultDisplay,
    ToolUIData,
    ToolUIDataAdapter,
)
from vibe.core.types import (
    AssistantEvent,
    Role,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)


class LocalTaskArgs(BaseModel):
    task: str = Field(
        description=(
            "A precise, self-contained brief for the local model: what to do, "
            "which files are involved, and what counts as done. Phrase file "
            "operations as concrete actions, never as requests to report file "
            "contents — the contents stay local."
        )
    )


class LocalTaskResult(BaseModel):
    """Deliberately content-free: this is the privacy boundary.

    Nothing derived from protected file contents may appear here — the local
    subagent's work is streamed to the user's screen only, and the cloud
    model learns nothing beyond completion status.
    """

    completed: bool = Field(description="Whether the task completed normally")
    turns_used: int = Field(description="Number of turns the local agent used")


class LocalTaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class LocalTask(
    BaseTool[LocalTaskArgs, LocalTaskResult, LocalTaskToolConfig, BaseToolState],
    ToolUIData[LocalTaskArgs, LocalTaskResult],
):
    """Run a task against the local/private model with protected-path access.

    The subagent's transcript never enters the parent conversation: its
    activity is streamed to the UI for the user, and only ``completed`` and
    ``turns_used`` return to the cloud-visible context.
    """

    @classmethod
    def is_available(cls, config: AnyVibeConfig | None = None) -> bool:
        if config is None:
            return True
        settings = config.privacy_routing
        return settings.enabled and bool(settings.private_model)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, LocalTaskArgs):
            return ToolCallDisplay(summary=f"Local (private) task: {args.task}")
        return ToolCallDisplay(summary="Running local private task")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, LocalTaskResult):
            turn_word = "turn" if result.turns_used == 1 else "turns"
            if not result.completed:
                return ToolResultDisplay(
                    success=False,
                    message=(
                        f"Local task interrupted after {result.turns_used} "
                        f"{turn_word} (details stayed local)"
                    ),
                )
            return ToolResultDisplay(
                success=True,
                message=(
                    f"Local task completed in {result.turns_used} {turn_word} "
                    f"(details stayed local)"
                ),
            )
        return ToolResultDisplay(success=True, message="Local task completed")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running local private task"

    @staticmethod
    def _build_task_text(task: str) -> str:
        """Inject file contents directly into the prompt for small local models.

        Small models (7B-class) reliably fail to emit tool calls when a system
        prompt is present — they describe what they would do instead of doing it.
        Since the local model's value is reasoning about protected content, not
        navigating tools, we pre-read any file paths mentioned in the brief and
        inject the contents so the model can answer directly without tools.
        """
        prefix = (
            "You are running locally with access to protected files. Answer "
            "the user directly — your response is shown to them as-is.\n\n"
        )
        # Extract absolute paths from the task brief
        path_pattern = re.compile(r"(?:/[\w.~@-]+)+")
        paths = path_pattern.findall(task)
        injected_files: list[str] = []
        for p in dict.fromkeys(paths):
            try:
                file_path = Path(p).expanduser()
                if file_path.is_file():
                    content = file_path.read_text(errors="replace")
                    injected_files.append(
                        f"--- FILE: {p} ---\n{content}\n--- END FILE ---"
                    )
            except (OSError, ValueError):
                continue
        file_context = "\n\n".join(injected_files) + "\n\n" if injected_files else ""
        return f"{prefix}{file_context}Task: {task}"

    def get_result_extra(self, result: LocalTaskResult) -> str | None:
        # Without this, the orchestrator tends to fabricate the deliverable
        # it never saw (e.g. invent a plausible file summary from the name).
        return (
            "The local model's full response was already displayed directly "
            "to the user. You did NOT see the file contents or the response, "
            "so do not summarize, restate, or invent them. Briefly acknowledge "
            "that the task was handled by the local model and ask if the user "
            "needs anything else."
        )

    async def run(
        self, args: LocalTaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LocalTaskResult, None]:
        if not ctx:
            raise ToolError("local_task requires invocation context")

        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix="local_task",
            enabled=ctx.session_dir is not None,
        )
        base_config = VibeConfig.load(session_logging=session_logging)
        settings = base_config.privacy_routing
        if not settings.enabled or not settings.private_model:
            raise ToolError(
                "local_task requires privacy_routing.enabled = true and "
                "privacy_routing.private_model set to a configured model alias."
            )
        if settings.private_model not in base_config.models:
            raise ToolError(
                f"Privacy routing private model '{settings.private_model}' "
                f"is not in your configured models."
            )
        # Pin the whole subagent to the private model, compaction included:
        # every token of this task's context may contain protected content.
        # Privacy routing is off inside — the local model IS the trust zone
        # (no redaction needed against it), and this also makes local_task
        # unavailable in the subagent, preventing recursive spawning.
        #
        # The minimal system prompt (plus trimmed prompt sections) matters for
        # small local models: the full CLI prompt is ~5k tokens, which costs
        # ~10s of prefill per turn on consumer hardware and makes 7B-class
        # models recite the instructions instead of following them.
        private = base_config.models[settings.private_model]
        local_config = base_config.model_copy(
            update={
                "active_model": settings.private_model,
                "compaction_model": private,
                "privacy_routing": settings.model_copy(update={"enabled": False}),
                "system_prompt_id": SystemPrompt.MINIMAL.value,
                "include_project_context": False,
                "include_prompt_detail": False,
                "include_model_info": False,
                "include_commit_signature": False,
            }
        )

        subagent_loop = AgentLoop(
            config_orchestrator=LegacyConfigOrchestrator(local_config),
            launch_context=ctx.launch_context,
            is_subagent=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store,
            hook_config_result=ctx.hook_config_result,
            bypass_path_guard=True,
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id
        if ctx.approval_callback:
            subagent_loop.set_approval_callback(ctx.approval_callback)

        task_text = self._build_task_text(args.task)

        completed = True
        try:
            async with aclosing(subagent_loop.act(task_text)) as events:
                async for event in events:
                    # Stream activity to the user's screen only. Assistant text
                    # and tool results from this loop never reach the parent
                    # context: the yielded stream events are UI-only and the
                    # final LocalTaskResult carries no content.
                    if isinstance(event, AssistantEvent) and event.content:
                        if event.stopped_by_middleware:
                            completed = False
                        yield ToolStreamEvent(
                            tool_name=self.get_name(),
                            message=event.content,
                            tool_call_id=ctx.tool_call_id,
                            prominent=True,
                        )
                    elif isinstance(event, ToolResultEvent):
                        if event.skipped:
                            completed = False
                        elif event.result and event.tool_class:
                            adapter = ToolUIDataAdapter(event.tool_class)
                            display = adapter.get_result_display(event)
                            yield ToolStreamEvent(
                                tool_name=self.get_name(),
                                message=f"{event.tool_name}: {display.message}",
                                tool_call_id=ctx.tool_call_id,
                            )
            turns_used = sum(
                msg.role == Role.assistant for msg in subagent_loop.messages
            )
        except Exception:
            # Even the exception text could carry protected content (paths,
            # file snippets in tool errors); report failure without details.
            completed = False
            turns_used = sum(
                msg.role == Role.assistant for msg in subagent_loop.messages
            )
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()

        yield LocalTaskResult(completed=completed, turns_used=turns_used)
