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
from vibe.core.agents.supervisor import MANAGED_AGENT_EVENT_QUEUE_SIZE, AgentSupervisor
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
        self, act: Callable[[str], AsyncGenerator[BaseEvent, None]] | None = None
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

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        loop = FakeManagedLoop()
        loops.append(loop)
        return loop

    supervisor = make_supervisor(factory=factory)
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
        "task",
        "state",
        "current_activity",
        "queued_messages",
        "error",
        "last_response",
        "usage",
    }

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
            tool_call_id="tool-call", tool_name="write_file", tool_class=Task
        )
        assert loop.approval_callback is not None
        await loop.approval_callback(
            "write_file", QuestionArgs(question="approve?"), "approval", None
        )
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

    await asyncio.wait_for(approval_entered.wait(), timeout=2)
    assert supervisor.output(started.agent_id).state is ManagedAgentState.ATTENTION
    approval = await next_event(events, ManagedAgentState.ATTENTION)
    assert approval.current_activity == "Approval needed for write_file"
    approval_release.set()
    await asyncio.wait_for(question_entered.wait(), timeout=2)
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
async def test_overlapping_approvals_restore_base_after_final_resolution() -> None:
    entered = {name: asyncio.Event() for name in ("first", "second")}
    release = {name: asyncio.Event() for name in ("first", "second")}
    callbacks_done = asyncio.Event()
    finish_turn = asyncio.Event()
    callback_tasks: list[asyncio.Task[tuple[ApprovalResponse, str | None]]] = []
    loop: FakeManagedLoop

    async def approval_callback(
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        entered[tool_call_id].set()
        await release[tool_call_id].wait()
        return ApprovalResponse.YES, None

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        yield ToolCallEvent(tool_call_id="batch", tool_name="batch", tool_class=Task)
        assert loop.approval_callback is not None

        async def request_approval(
            tool_name: str, question: str, tool_call_id: str
        ) -> tuple[ApprovalResponse, str | None]:
            assert loop.approval_callback is not None
            return await loop.approval_callback(
                tool_name, QuestionArgs(question=question), tool_call_id, None
            )

        callback_tasks.extend([
            asyncio.create_task(
                request_approval("first_tool", "approve first?", "first")
            ),
            asyncio.create_task(
                request_approval("second_tool", "approve second?", "second")
            ),
        ])
        await asyncio.gather(*callback_tasks)
        callbacks_done.set()
        await finish_turn.wait()
        yield AssistantEvent(content="done")

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        nonlocal loop
        loop = FakeManagedLoop(act)
        return loop

    supervisor = make_supervisor(factory=factory, approval_callback=approval_callback)
    started = await supervisor.start("default", "work")
    await asyncio.wait_for(entered["first"].wait(), timeout=2)
    await asyncio.wait_for(entered["second"].wait(), timeout=2)
    assert supervisor.output(started.agent_id).state is ManagedAgentState.ATTENTION

    release["second"].set()
    await asyncio.wait_for(asyncio.shield(callback_tasks[1]), timeout=2)
    one_pending = supervisor.output(started.agent_id)
    assert one_pending.state is ManagedAgentState.ATTENTION
    assert one_pending.current_activity == "Approval needed for first_tool"

    release["first"].set()
    await asyncio.wait_for(callbacks_done.wait(), timeout=2)
    none_pending = supervisor.output(started.agent_id)
    assert none_pending.state is ManagedAgentState.WORKING
    assert none_pending.current_activity == "Running batch"

    finish_turn.set()
    while supervisor.output(started.agent_id).state is not ManagedAgentState.IDLE:
        await asyncio.sleep(0)
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_overlapping_approvals_report_serialized_active_request() -> None:
    callback_lock = asyncio.Lock()
    callback_called = {name: asyncio.Event() for name in ("first", "second")}
    callback_active = {name: asyncio.Event() for name in ("first", "second")}
    release = {name: asyncio.Event() for name in ("first", "second")}
    both_pending = asyncio.Event()
    callbacks_done = asyncio.Event()
    finish_turn = asyncio.Event()
    active_request: str | None = None
    loop: FakeManagedLoop

    async def approval_callback(
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        nonlocal active_request
        callback_called[tool_call_id].set()
        async with callback_lock:
            active_request = tool_call_id
            callback_active[tool_call_id].set()
            await release[tool_call_id].wait()
            active_request = None
        return ApprovalResponse.YES, None

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        yield ToolCallEvent(tool_call_id="batch", tool_name="batch", tool_class=Task)

        async def request_approval(
            tool_name: str, tool_call_id: str
        ) -> tuple[ApprovalResponse, str | None]:
            assert loop.approval_callback is not None
            return await loop.approval_callback(
                tool_name, QuestionArgs(question="approve?"), tool_call_id, None
            )

        first_task = asyncio.create_task(request_approval("first_tool", "first"))
        await callback_active["first"].wait()
        second_task = asyncio.create_task(request_approval("second_tool", "second"))
        await callback_called["second"].wait()
        both_pending.set()
        await asyncio.gather(first_task, second_task)
        callbacks_done.set()
        await finish_turn.wait()
        yield AssistantEvent(content="done")

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        nonlocal loop
        loop = FakeManagedLoop(act)
        return loop

    supervisor = make_supervisor(factory=factory, approval_callback=approval_callback)
    started = await supervisor.start("default", "work")
    await asyncio.wait_for(both_pending.wait(), timeout=2)

    assert active_request == "first"
    first_active = supervisor.output(started.agent_id)
    assert first_active.state is ManagedAgentState.ATTENTION
    assert first_active.current_activity == "Approval needed for first_tool"

    release["first"].set()
    await asyncio.wait_for(callback_active["second"].wait(), timeout=2)
    assert active_request == "second"
    second_active = supervisor.output(started.agent_id)
    assert second_active.state is ManagedAgentState.ATTENTION
    assert second_active.current_activity == "Approval needed for second_tool"

    release["second"].set()
    await asyncio.wait_for(callbacks_done.wait(), timeout=2)
    restored = supervisor.output(started.agent_id)
    assert restored.state is ManagedAgentState.WORKING
    assert restored.current_activity == "Running batch"

    finish_turn.set()
    while supervisor.output(started.agent_id).state is not ManagedAgentState.IDLE:
        await asyncio.sleep(0)
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_approval_and_question_stay_attention_when_question_is_cancelled() -> (
    None
):
    approval_entered = asyncio.Event()
    approval_release = asyncio.Event()
    question_entered = asyncio.Event()
    callbacks_done = asyncio.Event()
    finish_turn = asyncio.Event()
    approval_task: asyncio.Task[tuple[ApprovalResponse, str | None]] | None = None
    question_task: asyncio.Task[BaseModel] | None = None
    loop: FakeManagedLoop

    async def approval_callback(
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        approval_entered.set()
        await approval_release.wait()
        return ApprovalResponse.YES, None

    async def user_input_callback(args: BaseModel) -> BaseModel:
        question_entered.set()
        await asyncio.Event().wait()
        return args

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        nonlocal approval_task, question_task
        yield ToolCallEvent(tool_call_id="batch", tool_name="batch", tool_class=Task)
        assert loop.approval_callback is not None
        assert loop.user_input_callback is not None

        async def request_approval() -> tuple[ApprovalResponse, str | None]:
            assert loop.approval_callback is not None
            return await loop.approval_callback(
                "write_file", QuestionArgs(question="approve?"), "approval", None
            )

        async def request_user_input() -> BaseModel:
            assert loop.user_input_callback is not None
            return await loop.user_input_callback(QuestionArgs(question="continue?"))

        approval_task = asyncio.create_task(request_approval())
        question_task = asyncio.create_task(request_user_input())
        await asyncio.gather(approval_task, question_task, return_exceptions=True)
        callbacks_done.set()
        await finish_turn.wait()
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
    started = await supervisor.start("default", "work")
    await asyncio.wait_for(approval_entered.wait(), timeout=2)
    await asyncio.wait_for(question_entered.wait(), timeout=2)

    assert question_task is not None
    question_task.cancel()
    await asyncio.gather(question_task, return_exceptions=True)
    one_pending = supervisor.output(started.agent_id)
    assert one_pending.state is ManagedAgentState.ATTENTION
    assert one_pending.current_activity == "Approval needed for write_file"

    approval_release.set()
    await asyncio.wait_for(callbacks_done.wait(), timeout=2)
    none_pending = supervisor.output(started.agent_id)
    assert none_pending.state is ManagedAgentState.WORKING
    assert none_pending.current_activity == "Running batch"

    finish_turn.set()
    while supervisor.output(started.agent_id).state is not ManagedAgentState.IDLE:
        await asyncio.sleep(0)
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_stop_takes_precedence_over_pending_attention() -> None:
    approval_entered = asyncio.Event()
    loop: FakeManagedLoop

    async def approval_callback(
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        approval_entered.set()
        await asyncio.Event().wait()
        return ApprovalResponse.YES, None

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        yield ToolCallEvent(
            tool_call_id="tool", tool_name="write_file", tool_class=Task
        )
        assert loop.approval_callback is not None
        await loop.approval_callback(
            "write_file", QuestionArgs(question="approve?"), "approval", None
        )

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        nonlocal loop
        loop = FakeManagedLoop(act)
        return loop

    supervisor = make_supervisor(factory=factory, approval_callback=approval_callback)
    started = await supervisor.start("default", "work")
    await asyncio.wait_for(approval_entered.wait(), timeout=2)
    assert supervisor.output(started.agent_id).state is ManagedAgentState.ATTENTION

    stopped = await supervisor.stop(started.agent_id)
    await asyncio.sleep(0)

    assert stopped.state is ManagedAgentState.STOPPED
    assert supervisor.output(started.agent_id).state is ManagedAgentState.STOPPED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_supervisor_bounds_prompts_queue_output_and_error() -> None:
    release = asyncio.Event()
    fail_release = asyncio.Event()
    calls = 0

    async def act(prompt: str) -> AsyncGenerator[BaseEvent, None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
            yield AssistantEvent(content="x" * (MAX_MANAGED_AGENT_RESPONSE_CHARS + 25))
            return
        await fail_release.wait()
        raise RuntimeError("failure " + "y" * 3_000)
        yield

    loop = FakeManagedLoop(act)
    supervisor = make_supervisor(factory=lambda *_args: loop)
    events, first = await subscribed_events(supervisor)
    task = "t" * (MAX_MANAGED_AGENT_TASK_CHARS + 25)
    started = await supervisor.start("default", task)
    await first
    await next_event(events, ManagedAgentState.RUNNING)
    assert len(loop.prompts[0]) == MAX_MANAGED_AGENT_TASK_CHARS

    release.set()
    idle = await next_event(events, ManagedAgentState.IDLE)
    assert len(idle.last_response) == MAX_MANAGED_AGENT_RESPONSE_CHARS

    await supervisor.message(started.agent_id, "failure prompt")
    await next_event(events, ManagedAgentState.RUNNING)
    for index in range(20):
        await supervisor.message(started.agent_id, f"queued {index}")
    with pytest.raises(ValueError, match="too many queued messages"):
        await supervisor.message(started.agent_id, "overflow")
    fail_release.set()
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
async def test_slow_subscriber_converges_to_every_agent_after_overflow() -> None:
    release = asyncio.Event()
    loops: list[FakeManagedLoop] = []

    async def act(_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        await release.wait()
        yield AssistantEvent(content="done")

    def factory(
        profile: str, agent_type: AgentType, logging: SessionLoggingConfig
    ) -> FakeManagedLoop:
        loop = FakeManagedLoop(act)
        loop.session_id = f"child-{len(loops) + 1}"
        loops.append(loop)
        return loop

    supervisor = make_supervisor(factory=factory)
    events = supervisor.subscribe_events()
    first_event = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    started = [
        await supervisor.start("default", f"task {index}", name=f"worker-{index}")
        for index in range(8)
    ]
    while any(
        supervisor.output(agent.agent_id).state is not ManagedAgentState.RUNNING
        for agent in started
    ):
        await asyncio.sleep(0)
    await first_event

    await supervisor.stop(started[0].agent_id)
    published_updates = 0
    for agent in started[1:]:
        for index in range(10):
            await supervisor.message(agent.agent_id, f"queued {index}")
            published_updates += 1
    assert published_updates > MANAGED_AGENT_EVENT_QUEUE_SIZE

    latest: dict[str, ManagedAgentLifecycleEvent] = {}
    async with asyncio.timeout(2):
        while len(latest) < len(started):
            event = await anext(events)
            latest[event.agent_id] = event

    assert set(latest) == {agent.agent_id for agent in started}
    for agent_id, event in latest.items():
        snapshot = supervisor.output(agent_id)
        assert event.state is snapshot.state
        assert event.queued_messages == snapshot.queued_messages
        assert event.current_activity == snapshot.current_activity
    assert latest[started[0].agent_id].state is ManagedAgentState.STOPPED

    await supervisor.aclose()
    await events.aclose()


@pytest.mark.asyncio
async def test_supervisor_runs_real_agent_loop_workflow() -> None:
    config = build_test_vibe_config()

    def factory(profile: str, agent_type: AgentType, logging: SessionLoggingConfig):
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
