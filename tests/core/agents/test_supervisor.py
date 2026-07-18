from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.events import (
    MAX_MANAGED_AGENT_RESPONSE_CHARS,
    MAX_MANAGED_AGENT_TASK_CHARS,
    ManagedAgentLifecycleEvent,
    get_managed_agent_callback_context,
)
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import AgentType, ManagedAgentState
from vibe.core.agents.supervisor import AgentSupervisor
from vibe.core.config import SessionLoggingConfig
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.tools.builtins.task import Task
from vibe.core.tools.permissions import PermissionStore, RequiredPermission
from vibe.core.types import (
    ApprovalCallback,
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    UserInputCallback,
)


class FakeSessionLogger:
    def __init__(self) -> None:
        self.parent_session_id: str | None = None

    def reset_session(
        self, session_id: str, *, parent_session_id: str | None = None
    ) -> None:
        self.parent_session_id = parent_session_id


class FakeTelemetryClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeStats:
    steps = 0
    session_prompt_tokens = 0
    session_completion_tokens = 0
    context_tokens = 0
    session_cost = 0.0


class FakeManagedLoop:
    def __init__(
        self,
        act: Callable[[str], AsyncGenerator[BaseEvent, None]] | None = None,
    ) -> None:
        self.session_id = "child-session"
        self.parent_session_id: str | None = None
        self.session_logger = FakeSessionLogger()
        self.telemetry_client = FakeTelemetryClient()
        self.config = build_test_vibe_config()
        self.stats = FakeStats()
        self.prompts: list[str] = []
        self.closed = False
        self.approval_callback: ApprovalCallback | None = None
        self.user_input_callback: UserInputCallback | None = None
        self._act = act

    async def act(self, msg: str, **kwargs: object) -> AsyncGenerator[BaseEvent, None]:
        self.prompts.append(msg)
        if self._act is not None:
            async for event in self._act(msg):
                yield event
            return
        await asyncio.sleep(0)
        yield AssistantEvent(content=f"response: {msg}")

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        self.approval_callback = callback

    def set_user_input_callback(self, callback: UserInputCallback) -> None:
        self.user_input_callback = callback

    async def wait_until_ready(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


class QuestionArgs(BaseModel):
    question: str


async def next_event(
    events: AsyncGenerator[ManagedAgentLifecycleEvent, None],
    state: ManagedAgentState,
    *,
    queued_messages: int | None = None,
) -> ManagedAgentLifecycleEvent:
    async with asyncio.timeout(2):
        async for event in events:
            if event.state is not state:
                continue
            if queued_messages is not None and event.queued_messages != queued_messages:
                continue
            return event
    raise AssertionError(f"Managed agent did not emit {state}")


def make_supervisor(
    *,
    factory=None,
    approval_callback: ApprovalCallback | None = None,
    user_input_callback: UserInputCallback | None = None,
    parent_session_id_getter: Callable[[], str] = lambda: "parent-session",
) -> AgentSupervisor:
    config = build_test_vibe_config()
    return AgentSupervisor(
        base_config_getter=lambda: config,
        agent_manager=AgentManager(LegacyConfigOrchestrator(config)),
        permission_store=PermissionStore(),
        approval_callback_getter=lambda: approval_callback,
        user_input_callback_getter=lambda: user_input_callback,
        parent_session_id_getter=parent_session_id_getter,
        session_dir_getter=lambda: None,
        launch_context=None,
        hook_config_result=None,
        loop_factory=factory,
    )


async def subscribed_events(
    supervisor: AgentSupervisor,
) -> tuple[
    AsyncGenerator[ManagedAgentLifecycleEvent, None],
    asyncio.Task[ManagedAgentLifecycleEvent],
]:
    events = supervisor.subscribe_events()
    first = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    return events, first


@pytest.mark.asyncio
async def test_supervisor_streams_persistent_agent_lifecycle_and_output() -> None:
    loops: list[FakeManagedLoop] = []
    listener_events: list[ManagedAgentLifecycleEvent] = []

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        loop = FakeManagedLoop()
        loops.append(loop)
        return loop

    supervisor = make_supervisor(factory=factory)
    supervisor.set_lifecycle_listener(listener_events.append)
    events, first = await subscribed_events(supervisor)

    started = await supervisor.start("default", "first task", name="Worker One")
    assert (await first).state is ManagedAgentState.STARTING
    idle = await next_event(events, ManagedAgentState.IDLE)
    queued = await supervisor.message(started.agent_id, "follow up")
    assert queued.queued_messages == 1
    await next_event(events, ManagedAgentState.IDLE, queued_messages=0)

    snapshot = supervisor.output(started.agent_id)
    assert loops[0].prompts == ["first task", "follow up"]
    assert snapshot.last_response == "response: follow up"
    assert started.agent_id == "worker-one-1"
    assert started.child_session_id == "child-session"
    assert loops[0].parent_session_id == "parent-session"
    assert loops[0].session_logger.parent_session_id == "parent-session"
    assert idle.model_dump().keys() == {
        "sequence",
        "agent_id",
        "profile",
        "agent_display_name",
        "parent_session_id",
        "child_session_id",
        "state",
        "current_activity",
        "queued_messages",
    }
    assert listener_events
    assert listener_events[-1].sequence >= listener_events[0].sequence

    await supervisor.stop(started.agent_id)
    stopped = await next_event(events, ManagedAgentState.STOPPED)
    assert stopped.agent_id == started.agent_id
    assert loops[0].closed is True
    assert loops[0].telemetry_client.closed is True
    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_supervisor_reports_tool_approval_and_question_attention() -> None:
    approval_entered = asyncio.Event()
    approval_release = asyncio.Event()
    question_entered = asyncio.Event()
    question_release = asyncio.Event()
    callback_contexts = []
    loop: FakeManagedLoop

    async def approval_callback(
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        callback_contexts.append(get_managed_agent_callback_context())
        approval_entered.set()
        await approval_release.wait()
        return ApprovalResponse.YES, None

    async def user_input_callback(args: BaseModel) -> BaseModel:
        callback_contexts.append(get_managed_agent_callback_context())
        question_entered.set()
        await question_release.wait()
        return QuestionArgs(question="answered")

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        yield ToolCallEvent(
            tool_call_id="tool-call",
            tool_name="write_file",
            tool_class=Task,
        )
        assert loop.approval_callback is not None
        await loop.approval_callback("write_file", BaseModel(), "approval", None)
        assert loop.user_input_callback is not None
        await loop.user_input_callback(QuestionArgs(question="continue?"))
        yield AssistantEvent(content="done")

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        nonlocal loop
        loop = FakeManagedLoop(act)
        return loop

    supervisor = make_supervisor(
        factory=factory,
        approval_callback=approval_callback,
        user_input_callback=user_input_callback,
    )
    events, first = await subscribed_events(supervisor)
    started = await supervisor.start("default", "work")
    await first

    await approval_entered.wait()
    assert supervisor.output(started.agent_id).state is ManagedAgentState.ATTENTION
    approval = await next_event(events, ManagedAgentState.ATTENTION)
    assert approval.current_activity == "Approval needed for write_file"
    approval_release.set()
    await question_entered.wait()
    question = await next_event(events, ManagedAgentState.ATTENTION)
    assert question.current_activity == "Waiting for user input"
    question_release.set()
    await next_event(events, ManagedAgentState.IDLE)

    assert [context.agent_id for context in callback_contexts if context] == [
        started.agent_id,
        started.agent_id,
    ]
    assert get_managed_agent_callback_context() is None
    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_supervisor_bounds_prompts_queue_output_and_error() -> None:
    release = asyncio.Event()
    calls = 0

    async def act(prompt: str) -> AsyncGenerator[BaseEvent, None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
            yield AssistantEvent(content="x" * (MAX_MANAGED_AGENT_RESPONSE_CHARS + 25))
            return
        raise RuntimeError("failure " + "y" * 3_000)
        yield

    loop = FakeManagedLoop(act)
    supervisor = make_supervisor(factory=lambda *_args: loop)
    events, first = await subscribed_events(supervisor)
    task = "t" * (MAX_MANAGED_AGENT_TASK_CHARS + 25)
    started = await supervisor.start("default", task)
    await first
    await next_event(events, ManagedAgentState.RUNNING)
    for index in range(20):
        await supervisor.message(started.agent_id, f"queued {index}")
    with pytest.raises(ValueError, match="too many queued messages"):
        await supervisor.message(started.agent_id, "overflow")
    assert len(loop.prompts[0]) == MAX_MANAGED_AGENT_TASK_CHARS

    release.set()
    idle = await next_event(events, ManagedAgentState.IDLE)
    assert len(idle.last_response) == MAX_MANAGED_AGENT_RESPONSE_CHARS
    failed = await next_event(events, ManagedAgentState.FAILED)
    assert len(failed.error or "") == 2_000
    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_session_change_stops_workers_and_clears_old_registry() -> None:
    release = asyncio.Event()
    loop = FakeManagedLoop()

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        await release.wait()
        yield AssistantEvent(content="done")

    loop._act = act
    supervisor = make_supervisor(factory=lambda *_args: loop)
    events, first = await subscribed_events(supervisor)
    started = await supervisor.start("default", "work")
    await first
    await next_event(events, ManagedAgentState.RUNNING)

    await supervisor.stop_for_session_change()

    stopped = await next_event(events, ManagedAgentState.STOPPED)
    assert stopped.agent_id == started.agent_id
    assert supervisor.list() == ()
    assert loop.closed is True
    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_supervisor_runs_real_agent_loop_workflow() -> None:
    config = build_test_vibe_config()

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ):
        child_config = config.model_copy(deep=True)
        child_config.session_logging = logging
        return build_test_agent_loop(
            config=child_config,
            agent_name=profile,
            backend=FakeBackend(mock_llm_chunk(content="real child response")),
            permission_store=PermissionStore(),
        )

    supervisor = make_supervisor(factory=factory)
    events, first = await subscribed_events(supervisor)
    started = await supervisor.start("default", "perform real turn")
    await first
    await next_event(events, ManagedAgentState.IDLE)

    snapshot = supervisor.output(started.agent_id)
    assert snapshot.last_response == "real child response"
    assert snapshot.state is ManagedAgentState.IDLE
    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_supervisor_rejects_nested_orchestrators() -> None:
    supervisor = make_supervisor()

    with pytest.raises(ValueError, match="cannot launch another orchestrator"):
        await supervisor.start("orchestrator", "delegate everything")

    await supervisor.aclose()
