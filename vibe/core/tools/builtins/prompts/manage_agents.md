Control managed agents through the management runtime available to this session.

- `start`: launch an agent with `profile` and a self-contained `task`; optionally set `name`.
- `list`: inspect managed agents and discover available profiles.
- `message`: queue a follow-up with `agent_id` and `message`.
- `output`: read the latest bounded response from `agent_id`.
- `stop`: stop `agent_id`.

Do not launch another orchestrator. Avoid concurrent edits to the same files, and
do not claim completion until the returned snapshot reports it.
