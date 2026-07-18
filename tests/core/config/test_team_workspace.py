from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.config import TeamWorkspaceConfig, VibeConfig, VibeConfigSchema


def test_team_workspace_defaults_are_private_and_disabled() -> None:
    config = TeamWorkspaceConfig()

    assert config.enabled is False
    assert config.shared_root == ""
    assert config.privacy_mode == "status"
    assert config.heartbeat_interval_seconds == 5.0
    assert config.presence_ttl_seconds == 30.0
    assert config.member_name == ""


def test_team_workspace_expands_nonempty_shared_root(tmp_path: Path) -> None:
    shared = tmp_path / "shared"

    config = TeamWorkspaceConfig(shared_root=str(shared), member_name="  Ada  ")

    assert config.shared_root == str(shared.resolve())
    assert config.member_name == "Ada"


@pytest.mark.parametrize("heartbeat", [0, -1])
def test_team_workspace_rejects_nonpositive_heartbeat(heartbeat: float) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        TeamWorkspaceConfig(heartbeat_interval_seconds=heartbeat)


def test_team_workspace_requires_ttl_longer_than_heartbeat() -> None:
    with pytest.raises(ValueError, match="must be greater"):
        TeamWorkspaceConfig(
            heartbeat_interval_seconds=10,
            presence_ttl_seconds=10,
        )


@pytest.mark.parametrize("config_type", [VibeConfig, VibeConfigSchema])
def test_team_workspace_is_available_on_both_config_models(config_type) -> None:
    config = config_type.model_validate({
        "team_workspace": {
            "enabled": True,
            "privacy_mode": "summaries",
            "member_name": "Grace",
        }
    })

    assert config.team_workspace.enabled is True
    assert config.team_workspace.privacy_mode == "summaries"
    assert config.team_workspace.member_name == "Grace"
