# Vibe Agent Room

Local web control room for persistent Mistral Vibe agents.

```bash
uv run vibe --server --workdir /path/to/integration-worktree
```

Open <http://127.0.0.1:4173/web/agent-room/>.

The Textual Agent Home and browser are clients of this one backend. They share
run IDs, Vibe sessions, conversations, queues, interactions, metrics, groups,
worktrees, and lifecycle state. A source-checkout CLI discovers the owner via
`~/.vibe/agent-room/server.json`; if no owner is reachable, it starts this
server in the background. Set `VIBE_AGENT_ROOM_URL` to use another loopback
port, or `VIBE_AGENT_ROOM_AUTOSTART=0` to require manual startup. Only one
backend may own a `VIBE_HOME` at a time.

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

In CLI Home, select any retained agent to read the same conversation and
send another message. The controls also cancel a turn, stop a process, approve
or deny a tool call, and answer pending questions. Stopped and failed agents
remain selectable; sending a message resumes the same session and worktree.

The HTTP server binds only to `127.0.0.1`, validates Host and Origin, and serves
an explicit static allowlist. Opening the assets with a plain static server
still provides the view-only `agents.json` demo.
