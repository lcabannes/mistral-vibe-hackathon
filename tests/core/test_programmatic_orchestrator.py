from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.programmatic import _prepare_programmatic_config, run_programmatic
from vibe.core.tools.manager import ToolManager


def test_programmatic_rejects_orchestrator_profile_before_startup() -> None:
    with pytest.raises(ValueError, match="not available through the programmatic"):
        run_programmatic(
            build_test_vibe_config(),
            prompt="work",
            agent_name=BuiltinAgentName.ORCHESTRATOR,
        )


def test_programmatic_config_excludes_orchestrator_controls() -> None:
    config = build_test_vibe_config(
        enable_orchestrator_controls=True,
        enable_cli_control=True,
        enable_agent_management=True,
    )

    programmatic_config = _prepare_programmatic_config(config)
    tools = ToolManager(lambda: programmatic_config).available_tools

    assert "control_cli" not in tools
    assert "manage_agents" not in tools
    assert "control_cli" not in config.disabled_tools
    assert "manage_agents" not in config.disabled_tools
