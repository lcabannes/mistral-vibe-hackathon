Control the interactive Vibe CLI after the current orchestrator turn completes.

- `command`: defer an existing slash command in `value`, including its arguments.
- `switch_agent`: defer a switch to the agent profile named by `value`.
- `navigate_workspace`: defer navigation to `home`, `chat`, `office`, `coworkers`,
  `agents`, `mcp`, or `usage`.

Only actions advertised by the current CLI control adapter are accepted. Deferred
actions run after the current turn so they cannot mutate the UI mid-stream.
