# Vibe Agent Room

Local web control room for persistent Mistral Vibe agents.

```bash
uv run python web/agent-room/server.py --workdir /path/to/integration-worktree
```

Open <http://127.0.0.1:4173/web/agent-room/>.

Each cat is a real Vibe process with a durable conversation, FIFO prompt queue,
tool approvals, user questions, token/cost/context metrics, and its own Git
worktree and branch. Workers never edit the integration checkout. Stop a worker
after it commits, then use **Validate & merge**; the server first merges and runs
the test suite in a temporary worktree, and only updates the integration branch
after validation succeeds.

The Orchestrator cat controls the same worker registry through Vibe's typed
`manage_agents` port. It can start, inspect, message, and stop agents, while
`control_cli` exposes the allowlisted `/cancel`, `/stop`, and `/merge` room
commands. Browser chat also supports `/help`, `/status`, `/history`, `/queue`,
`/cancel`, `/stop`, and `/retry`; prefix a literal slash with `//`.

The HTTP server binds only to `127.0.0.1`, validates Host and Origin, and serves
an explicit static allowlist. Opening the assets with a plain static server
still provides the view-only `agents.json` demo.
