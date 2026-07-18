from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.failure_ledger import FailureLedger, failure_key
from vibe.core.permission_stats import SUGGESTION_THRESHOLD, PermissionStats
from vibe.core.types import FunctionCall, LLMChunk, RepeatedFailureEvent, ToolCall


class TestFailureLedger:
    def test_counts_consecutive_failures(self):
        ledger = FailureLedger()
        key = failure_key("bash", {"command": "pytest"})
        assert ledger.record_failure(key) == 1
        assert ledger.record_failure(key) == 2
        assert ledger.count(key) == 2

    def test_success_resets_counter(self):
        ledger = FailureLedger()
        key = failure_key("bash", {"command": "pytest"})
        ledger.record_failure(key)
        ledger.record_failure(key)
        ledger.record_success(key)
        assert ledger.count(key) == 0

    def test_different_args_tracked_separately(self):
        ledger = FailureLedger()
        k1 = failure_key("bash", {"command": "pytest"})
        k2 = failure_key("bash", {"command": "ruff check"})
        ledger.record_failure(k1)
        assert ledger.count(k2) == 0


class TestPermissionStats:
    def test_counts_persist_across_instances(self, tmp_path: Path):
        path = tmp_path / "stats.toml"
        stats = PermissionStats(stats_path=path)
        stats.record_approval("bash", "pytest *")
        stats2 = PermissionStats(stats_path=path)
        assert stats2.count("bash", "pytest *") == 1

    def test_suggests_at_threshold_once(self, tmp_path: Path):
        path = tmp_path / "stats.toml"
        stats = PermissionStats(stats_path=path)
        for _ in range(SUGGESTION_THRESHOLD - 1):
            stats.record_approval("bash", "pytest *")
            assert not stats.should_suggest("bash", "pytest *")
        stats.record_approval("bash", "pytest *")
        assert stats.should_suggest("bash", "pytest *")
        stats.mark_suggested("bash", "pytest *")
        assert not stats.should_suggest("bash", "pytest *")

    def test_suggested_state_persists(self, tmp_path: Path):
        path = tmp_path / "stats.toml"
        stats = PermissionStats(stats_path=path)
        for _ in range(SUGGESTION_THRESHOLD):
            stats.record_approval("bash", "pytest *")
        stats.mark_suggested("bash", "pytest *")
        stats2 = PermissionStats(stats_path=path)
        assert not stats2.should_suggest("bash", "pytest *")


def _failing_tool_chunk(command: str) -> LLMChunk:
    return mock_llm_chunk(
        content="",
        tool_calls=[
            ToolCall(
                id="call-1",
                index=0,
                function=FunctionCall(
                    name="bash", arguments=f'{{"command": "{command}"}}'
                ),
            )
        ],
    )


class TestRepeatedFailureWarning:
    @pytest.mark.asyncio
    async def test_warns_after_threshold_consecutive_failures(self):
        from vibe.core.agent_loop._loop import REPEATED_FAILURE_THRESHOLD

        config = build_test_vibe_config(
            bypass_tool_permissions=True, tools={"bash": {"permission": "always"}}
        )
        # The same failing command every turn, then a final plain response.
        chunks = [
            [_failing_tool_chunk("nonexistent-cmd-xyz --flag")]
            for _ in range(REPEATED_FAILURE_THRESHOLD)
        ]
        chunks.append([mock_llm_chunk(content="giving up")])
        backend = FakeBackend(chunks)
        agent = build_test_agent_loop(config=config, backend=backend)

        events = [e async for e in agent.act("run that command")]

        warnings = [e for e in events if isinstance(e, RepeatedFailureEvent)]
        assert len(warnings) == 1
        assert warnings[0].failure_count == REPEATED_FAILURE_THRESHOLD
        assert warnings[0].tool_name == "bash"
