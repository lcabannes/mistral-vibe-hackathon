from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def load_agent_room_server() -> ModuleType:
    path = Path(__file__).with_name("agent-room") / "server.py"
    name = "vibe_agent_room_server"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Agent Room server from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
