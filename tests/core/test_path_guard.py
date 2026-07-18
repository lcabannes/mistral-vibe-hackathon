from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import (
    ModelConfig,
    PrivacyRoutingConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.path_guard import (
    DEFAULT_PROTECTED_PATHS,
    find_protected_path_in_args,
    is_protected_path,
)
from vibe.core.tools.builtins.bash import BashArgs
from vibe.core.tools.builtins.read_file import ReadFileArgs
from vibe.core.types import Backend, FunctionCall, LLMChunk, Role, ToolCall


def _guarded_config(**privacy_kwargs) -> VibeConfig:
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


class TestIsProtectedPath:
    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.local",
            "config/.env.production",
            "/abs/path/deploy.pem",
            "certs/server.key",
            "~/.ssh/id_rsa",
            "keys/id_rsa.pub",
            "aws_credentials.json",
        ],
    )
    def test_default_patterns_protect(self, path: str):
        assert is_protected_path(path, DEFAULT_PROTECTED_PATHS)

    @pytest.mark.parametrize(
        "path",
        ["main.py", "README.md", "src/environment.ts", "envelope.txt", "keyboard.py"],
    )
    def test_ordinary_paths_unprotected(self, path: str):
        assert not is_protected_path(path, DEFAULT_PROTECTED_PATHS)

    def test_custom_pattern_with_directory(self):
        patterns = (*DEFAULT_PROTECTED_PATHS, "contracts/**")
        assert is_protected_path("contracts/msa.pdf", patterns)
        assert not is_protected_path("src/contracts.py", patterns)


class TestFindProtectedPathInArgs:
    def test_read_file_protected(self):
        args = ReadFileArgs(file_path="/repo/.env")
        found = find_protected_path_in_args("read_file", args, DEFAULT_PROTECTED_PATHS)
        assert found == "/repo/.env"

    def test_read_file_clean(self):
        args = ReadFileArgs(file_path="/repo/main.py")
        assert (
            find_protected_path_in_args("read_file", args, DEFAULT_PROTECTED_PATHS)
            is None
        )

    def test_bash_cat_env_detected(self):
        args = BashArgs(command="cat .env && echo done")
        found = find_protected_path_in_args("bash", args, DEFAULT_PROTECTED_PATHS)
        assert found == ".env"

    def test_bash_clean_command(self):
        args = BashArgs(command="ls -la src/ | head -5")
        assert (
            find_protected_path_in_args("bash", args, DEFAULT_PROTECTED_PATHS) is None
        )

    def test_no_patterns_means_no_guard(self):
        args = ReadFileArgs(file_path="/repo/.env")
        assert find_protected_path_in_args("read_file", args, ()) is None

    def test_unguarded_tool_ignored(self):
        args = ReadFileArgs(file_path="/repo/.env")
        assert (
            find_protected_path_in_args("todo", args, DEFAULT_PROTECTED_PATHS) is None
        )


def _tool_call_chunk(tool_name: str, arguments: str) -> LLMChunk:
    return mock_llm_chunk(
        content="",
        tool_calls=[
            ToolCall(
                id="call-1",
                index=0,
                function=FunctionCall(name=tool_name, arguments=arguments),
            )
        ],
    )


class TestAgentLoopPathGuard:
    @pytest.mark.asyncio
    async def test_read_of_protected_path_denied_with_redirect(self):
        config = _guarded_config()
        backend = FakeBackend([
            [_tool_call_chunk("read_file", '{"file_path": "/repo/.env"}')],
            [mock_llm_chunk(content="understood")],
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act("read the env file")]

        tool_messages = [m for m in agent.messages if m.role == Role.tool]
        assert len(tool_messages) == 1
        assert "protected path" in (tool_messages[0].content or "")
        assert "local_task" in (tool_messages[0].content or "")

    @pytest.mark.asyncio
    async def test_bash_on_protected_path_denied(self):
        config = _guarded_config()
        backend = FakeBackend([
            [_tool_call_chunk("bash", '{"command": "cat ~/.aws/credentials"}')],
            [mock_llm_chunk(content="understood")],
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act("show me aws creds")]

        tool_messages = [m for m in agent.messages if m.role == Role.tool]
        assert len(tool_messages) == 1
        assert "protected path" in (tool_messages[0].content or "")

    @pytest.mark.asyncio
    async def test_guard_disabled_when_privacy_routing_off(self):
        config = build_test_vibe_config()
        backend = FakeBackend([
            [_tool_call_chunk("read_file", '{"file_path": "/nonexistent/.env"}')],
            [mock_llm_chunk(content="ok")],
        ])
        agent = build_test_agent_loop(config=config, backend=backend)

        [_ async for _ in agent.act("read env")]

        tool_messages = [m for m in agent.messages if m.role == Role.tool]
        # The tool ran (and failed on the missing file) instead of being
        # guard-denied: no redirect message.
        assert all("local_task" not in (m.content or "") for m in tool_messages)
