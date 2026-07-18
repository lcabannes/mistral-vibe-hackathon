# Vibe Agent Room

Interactive hackathon prototype for an agent-control web surface. The loopback
runner launches real Vibe sessions through the supported programmatic
CLI and projects their lifecycle, transcript, token usage, estimated cost, and
session metadata into the room.

From the repository root:

```bash
uv run python web/agent-room/server.py
```

Open <http://localhost:4173/web/agent-room/>.

The room includes state-specific cat motion, agent search and filters, detail
inspection, Chat/History/Status actions, drag-and-drop group assignment, group
creation, and per-group agent launch. The server always binds to `127.0.0.1`,
serves only the room assets, limits concurrent runs, and applies turn, token,
cost, and wall-time bounds. This first bridge exposes read-safe `default` and
`plan` profiles; write-capable profiles require per-run worktree isolation.

Opening the files through a plain static server remains supported as a
view-only demo using `agents.json`. In that mode the launch button is disabled
rather than creating a fake run.
