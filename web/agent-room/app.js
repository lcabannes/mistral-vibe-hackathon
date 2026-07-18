const STORAGE_KEY = "vibe-agent-room:v1";
const ACTIVE_STATES = new Set(["requested", "running", "working"]);
const CONTROLLABLE_STATES = new Set([
  "requested",
  "running",
  "working",
  "attention",
  "idle",
  "failed",
]);
const TERMINAL_STATES = new Set(["completed", "cancelled", "stopped"]);
const ATTENTION_STATES = new Set(["attention", "failed"]);
const COORDINATION_GROUP = {
  id: "coordination",
  name: "Coordination",
  color: "#171722",
};
const UNASSIGNED_GROUP = {
  id: "unassigned",
  name: "Unassigned",
  color: "#898991",
};

const STATE_META = {
  idle: { label: "Idle", color: "#73737d" },
  requested: { label: "Queued", color: "#ffaf00" },
  running: { label: "Starting", color: "#ff8205" },
  working: { label: "Working", color: "#fa500f" },
  attention: { label: "Needs input", color: "#e10500" },
  failed: { label: "Failed", color: "#e10500" },
  completed: { label: "Finished", color: "#16835f" },
  cancelled: { label: "Cancelled", color: "#73737d" },
  stopped: { label: "Stopped", color: "#73737d" },
};

const COATS = {
  orange: ["#dc772b", "#f3b66f"],
  mint: ["#79bca2", "#bde3d4"],
  rose: ["#cc777b", "#efb6b7"],
  blue: ["#7188c5", "#b4c1e7"],
  violet: ["#8c78af", "#c8bce0"],
  charcoal: ["#565762", "#a0a1a8"],
  sunny: ["#d6a52b", "#f3d677"],
};

const GROUP_COLORS = ["#ff8205", "#20a779", "#5874d8", "#d25470", "#8c65af"];

const state = {
  agents: [],
  groups: [],
  profiles: [],
  tools: [],
  coordination: null,
  network: null,
  connected: false,
  bridgeSeen: false,
  selectedAgentId: null,
  detailView: "status",
  selectedGroupId: "all",
  statusFilter: "all",
  search: "",
  motionPaused: false,
  positions: new Map(),
  selectedSwatch: GROUP_COLORS[0],
  lastSnapshot: "",
  agentDialogOrigin: null,
  chatDrafts: new Map(),
  chatAttachments: new Map(),
  chatErrors: new Map(),
  composerSelections: new Map(),
  chatScroll: new Map(),
  pendingRequests: new Set(),
};

const elements = {
  shell: document.querySelector(".app-shell"),
  topbar: document.querySelector(".topbar"),
  sidebar: document.querySelector(".sidebar"),
  mainContent: document.querySelector(".main-content"),
  zones: document.querySelector("#zones"),
  groupList: document.querySelector("#group-list"),
  catTemplate: document.querySelector("#cat-template"),
  emptyState: document.querySelector("#empty-state"),
  roomAnnouncer: document.querySelector("#room-announcer"),
  detailPanel: document.querySelector("#detail-panel"),
  detailContent: document.querySelector("#detail-content"),
  closeDetailButton: document.querySelector("#close-detail-button"),
  statusFilters: document.querySelector("#status-filters"),
  agentSearch: document.querySelector("#agent-search"),
  addGroupButton: document.querySelector("#add-group-button"),
  toolbarAddGroupButton: document.querySelector("#toolbar-add-group-button"),
  newAgentButton: document.querySelector("#new-agent-button"),
  groupDialog: document.querySelector("#group-dialog"),
  groupForm: document.querySelector("#group-form"),
  groupName: document.querySelector("#group-name"),
  groupSwatches: document.querySelector("#group-swatches"),
  agentDialog: document.querySelector("#agent-dialog"),
  agentForm: document.querySelector("#agent-form"),
  agentName: document.querySelector("#agent-name"),
  agentGroup: document.querySelector("#agent-group"),
  agentProfile: document.querySelector("#agent-profile"),
  agentTask: document.querySelector("#agent-task"),
  agentAutoApprove: document.querySelector("#agent-auto-approve"),
  agentToolsAll: document.querySelector("#agent-tools-all"),
  agentToolList: document.querySelector("#agent-tool-list"),
  agentSubmit: document.querySelector("#agent-submit"),
  agentDialogNotice: document.querySelector("#agent-dialog-notice"),
  agentDialogError: document.querySelector("#agent-dialog-error"),
  simulateButton: document.querySelector("#simulate-button"),
  feedStatus: document.querySelector("#feed-status"),
  officeClock: document.querySelector("#office-clock"),
  summaryRunning: document.querySelector("#summary-running"),
  summaryAttention: document.querySelector("#summary-attention"),
  summaryPast: document.querySelector("#summary-past"),
  summaryIdle: document.querySelector("#summary-idle"),
  coordinationStrip: document.querySelector("#coordination-strip"),
  workspaceBranch: document.querySelector("#workspace-branch"),
};

function readStoredState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function persistState() {
  const groupAssignments = Object.fromEntries(
    state.agents.map((agent) => [agent.tool_call_id, agent.group_id]),
  );
  const customGroups = state.groups.filter((group) => group.custom);
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ groupAssignments, customGroups }),
  );
}

async function loadRoom() {
  const response = await fetch("agents.json");
  if (!response.ok) throw new Error(`Room configuration returned ${response.status}`);
  const seed = await response.json();
  const stored = readStoredState();
  const customGroups = Array.isArray(stored.customGroups)
    ? stored.customGroups.filter(isValidGroup)
    : [];
  const sourceGroups = seed.groups.filter(isValidGroup);
  const groupIds = new Set([
    COORDINATION_GROUP.id,
    ...sourceGroups.map((group) => group.id),
    ...customGroups.map((group) => group.id),
    UNASSIGNED_GROUP.id,
  ]);

  state.groups = [
    COORDINATION_GROUP,
    ...sourceGroups.filter((group) => group.id !== COORDINATION_GROUP.id),
    ...customGroups.filter((group) => group.id !== COORDINATION_GROUP.id),
    UNASSIGNED_GROUP,
  ];
  try {
    const liveResponse = await fetch("/api/agent-runs", { cache: "no-store" });
    if (!liveResponse.ok) throw new Error("Live bridge unavailable");
    const live = await liveResponse.json();
    state.connected = live.connected === true;
    state.bridgeSeen = state.connected;
    state.profiles = Array.isArray(live.profiles) ? live.profiles : [];
    state.tools = Array.isArray(live.tools) ? live.tools : [];
    state.coordination = live.coordination || null;
    state.network = live.network || null;
    state.agents = normalizeAgents(live.activities, groupIds, stored);
    state.lastSnapshot = JSON.stringify(live.activities);
    updateNetworkStatus();
  } catch {
    state.connected = false;
    state.profiles = [];
    state.agents = normalizeAgents(seed.activities, groupIds, stored);
    setFeedStatus("Demo feed", false);
  }
}

function normalizeAgents(activities, groupIds, stored) {
  if (!Array.isArray(activities)) return [];
  return activities.map((agent) => {
    const storedGroup = stored.groupAssignments?.[agent.tool_call_id];
    const source = agent.source || "demo";
    const normalized = {
      ...agent,
      source,
      group_id:
        source === "demo" && groupIds.has(storedGroup)
          ? storedGroup
          : groupIds.has(agent.group_id)
            ? agent.group_id
            : UNASSIGNED_GROUP.id,
    };
    if (source === "demo") {
      const duration = Math.max(1, agent.updated_at - agent.started_at);
      normalized.updated_at = Date.now() / 1000;
      normalized.started_at = normalized.updated_at - duration;
    }
    return normalized;
  });
}

function setFeedStatus(label, isError) {
  elements.feedStatus.classList.toggle("is-error", isError);
  elements.feedStatus.querySelector("span").textContent = label;
}

function updateNetworkStatus() {
  if (state.network?.authenticated) {
    const label =
      state.network.selected_mode === "direct" ? "Mistral direct" : "Mistral ready";
    setFeedStatus(label, false);
    return;
  }
  if (state.network && !state.network.credential_resolved) {
    setFeedStatus("Sign-in required", true);
    return;
  }
  setFeedStatus(
    state.network ? "Mistral unavailable" : "Live local",
    Boolean(state.network),
  );
}

async function refreshLiveRuns() {
  if (elements.agentDialog.open || elements.groupDialog.open) return;
  try {
    const response = await fetch("/api/agent-runs", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const payload = await response.json();
    state.connected = true;
    state.bridgeSeen = true;
    state.profiles = Array.isArray(payload.profiles) ? payload.profiles : state.profiles;
    state.tools = Array.isArray(payload.tools) ? payload.tools : state.tools;
    state.coordination = payload.coordination || state.coordination;
    state.network = payload.network || state.network;
    updateNetworkStatus();
    const snapshot = JSON.stringify(payload.activities);
    if (snapshot === state.lastSnapshot) return;
    state.lastSnapshot = snapshot;
    const stored = readStoredState();
    const groupIds = new Set(state.groups.map((group) => group.id));
    state.agents = normalizeAgents(payload.activities, groupIds, stored);
    render();
  } catch {
    state.connected = false;
    if (state.bridgeSeen) setFeedStatus("Bridge disconnected", true);
  }
}

function isValidGroup(group) {
  return (
    group &&
    typeof group.id === "string" &&
    typeof group.name === "string" &&
    typeof group.color === "string"
  );
}

function statusMatches(agent) {
  if (state.statusFilter === "live") return agent.runtime_live === true;
  if (state.statusFilter === "attention") return ATTENTION_STATES.has(agent.state);
  if (state.statusFilter === "past") return agent.runtime_live !== true;
  return true;
}

function agentMatches(agent) {
  if (state.selectedGroupId !== "all" && agent.group_id !== state.selectedGroupId) {
    return false;
  }
  if (!statusMatches(agent)) return false;
  if (!state.search) return true;
  const haystack = [
    agent.agent_name,
    agent.agent_display_name,
    agent.task,
    agent.current_activity,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(state.search);
}

function render() {
  captureComposerState();
  const focusKey = document.activeElement?.dataset?.focusKey;
  renderGroupList();
  renderZones();
  renderSummary();
  updateEmptyState();
  if (state.selectedAgentId) renderDetails();
  if (focusKey) {
    requestAnimationFrame(() =>
      document.querySelector(`[data-focus-key="${CSS.escape(focusKey)}"]`)?.focus(),
    );
  }
}

function renderGroupList() {
  elements.groupList.replaceChildren();
  const groups = [
    { id: "all", name: "All groups", color: "#171722" },
    ...state.groups,
  ];

  for (const group of groups) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "group-filter";
    button.classList.toggle("is-active", state.selectedGroupId === group.id);
    button.setAttribute("aria-pressed", String(state.selectedGroupId === group.id));
    button.style.setProperty("--group-color", group.color);
    button.dataset.groupId = group.id;
    button.dataset.focusKey = `group-filter:${group.id}`;

    const dot = document.createElement("span");
    dot.className = "dot";
    const name = document.createElement("span");
    name.textContent = group.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = String(
      group.id === "all"
        ? state.agents.length
        : state.agents.filter((agent) => agent.group_id === group.id).length,
    );
    button.append(dot, name, count);
    button.addEventListener("click", () => {
      state.selectedGroupId = group.id;
      render();
    });
    elements.groupList.append(button);
  }
}

function renderZones() {
  elements.zones.replaceChildren();
  const visibleGroups = state.groups.filter((group) => {
    if (state.selectedGroupId !== "all") return state.selectedGroupId === group.id;
    return (
      group.id !== UNASSIGNED_GROUP.id ||
      state.agents.some((agent) => agent.group_id === UNASSIGNED_GROUP.id)
    );
  });

  for (const group of visibleGroups) {
    const zone = document.createElement("section");
    zone.className = "group-zone";
    zone.dataset.groupId = group.id;
    zone.style.setProperty("--group-color", group.color);

    const heading = document.createElement("div");
    heading.className = "zone-heading";
    const title = document.createElement("strong");
    title.textContent = group.name;
    const count = document.createElement("span");
    const groupAgents = state.agents.filter((agent) => agent.group_id === group.id);
    const matchingAgents = groupAgents.filter(agentMatches);
    count.textContent =
      matchingAgents.length === groupAgents.length
        ? `${groupAgents.length} ${groupAgents.length === 1 ? "agent" : "agents"}`
        : `${matchingAgents.length} of ${groupAgents.length}`;
    const tools = document.createElement("div");
    tools.className = "zone-tools";
    const addAgent = document.createElement("button");
    addAgent.type = "button";
    addAgent.className = "zone-add-agent";
    addAgent.textContent = "+";
    addAgent.title = `Run an agent in ${group.name}`;
    addAgent.setAttribute("aria-label", `Run an agent in ${group.name}`);
    addAgent.addEventListener("click", () => openAgentDialog(group.id, addAgent));
    tools.append(count, addAgent);
    heading.append(title, tools);

    const agentLayer = document.createElement("div");
    agentLayer.className = "zone-agents";
    for (const agent of groupAgents) {
      const agentElement = createAgentElement(agent);
      agentElement.hidden = !agentMatches(agent);
      agentLayer.append(agentElement);
    }
    if (groupAgents.length === 0) {
      const empty = document.createElement("button");
      empty.type = "button";
      empty.className = "zone-empty";
      const emptyTitle = document.createElement("strong");
      emptyTitle.textContent = "No agents";
      const emptyAction = document.createElement("span");
      emptyAction.textContent = "Run an agent";
      empty.append(emptyTitle, emptyAction);
      empty.addEventListener("click", () => openAgentDialog(group.id, empty));
      agentLayer.append(empty);
    }

    zone.append(heading, agentLayer);
    zone.addEventListener("dragover", handleZoneDragOver);
    zone.addEventListener("dragleave", handleZoneDragLeave);
    zone.addEventListener("drop", handleZoneDrop);
    elements.zones.append(zone);
  }

  requestAnimationFrame(placeAgents);
}

function createAgentElement(agent) {
  const element = elements.catTemplate.content.firstElementChild.cloneNode(true);
  const stateMeta = STATE_META[agent.state] || STATE_META.idle;
  const coat = COATS[agent.coat] || COATS.orange;

  element.dataset.agentId = agent.tool_call_id;
  element.draggable = !agent.is_orchestrator;
  element.classList.add(`state-${agent.state}`);
  element.classList.toggle("is-selected", state.selectedAgentId === agent.tool_call_id);
  element.style.setProperty("--coat", coat[0]);
  element.style.setProperty("--coat-light", coat[1]);
  const mainButton = element.querySelector(".agent-main");
  mainButton.dataset.focusKey = `agent-main:${agent.tool_call_id}`;
  mainButton.setAttribute(
    "aria-label",
    `${agent.agent_display_name}, ${stateMeta.label}, ${agent.current_activity || agent.task}`,
  );
  element.querySelector(".activity-bubble").textContent =
    agent.current_activity || agent.task;
  element.querySelector(".agent-label strong").textContent = agent.agent_display_name;
  element.querySelector(".status-label").textContent = stateMeta.label;

  element.querySelector(".agent-actions").setAttribute(
    "aria-label",
    `${agent.agent_display_name} actions`,
  );
  mainButton.addEventListener("click", () => openAgentPanel(agent.tool_call_id, "status"));
  for (const action of element.querySelectorAll("[data-agent-action]")) {
    const view = action.dataset.agentAction;
    const isActive =
      state.selectedAgentId === agent.tool_call_id && state.detailView === view;
    action.classList.toggle("is-active", isActive);
    action.setAttribute("aria-pressed", String(isActive));
    action.dataset.focusKey = `agent-action:${agent.tool_call_id}:${view}`;
    action.addEventListener("click", () => openAgentPanel(agent.tool_call_id, view));
    action.addEventListener("pointerdown", (event) => event.stopPropagation());
    action.draggable = false;
  }
  element.addEventListener("dragstart", handleAgentDragStart);
  element.addEventListener("dragend", handleAgentDragEnd);
  return element;
}

function placeAgents() {
  for (const layer of elements.zones.querySelectorAll(".zone-agents")) {
    const agents = [...layer.querySelectorAll(".agent:not([hidden])")];
    const availableWidth = Math.max(layer.clientWidth, 240);
    const columns = Math.max(1, Math.floor(availableWidth / 126));
    const rows = Math.max(1, Math.ceil(agents.length / columns));
    layer.closest(".group-zone").style.minHeight = `${Math.max(258, 60 + rows * 194)}px`;
    agents.forEach((agentElement, index) => {
      const agentId = agentElement.dataset.agentId;
      const column = index % columns;
      const row = Math.floor(index / columns);
      const slotWidth = availableWidth / columns;
      const x = column * slotWidth + Math.max(16, (slotWidth - 104) / 2);
      const y = 8 + row * 194 + (column % 2) * 8;
      state.positions.set(agentId, { x, y });
      setAgentPosition(agentElement, x, y, 0);
    });
  }
}

function setAgentPosition(element, x, y, duration = 3.2) {
  element.style.setProperty("--x", `${Math.round(x)}px`);
  element.style.setProperty("--y", `${Math.round(y)}px`);
  element.style.setProperty("--move-duration", `${duration}s`);
}

function roamAgents() {
  if (state.motionPaused || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }

  for (const agentElement of elements.zones.querySelectorAll(".agent:not([hidden])")) {
    if (agentElement.matches(":hover") || agentElement.matches(":focus-within")) continue;
    const agent = getAgent(agentElement.dataset.agentId);
    if (!agent?.runtime_live) continue;
    const base = state.positions.get(agent.tool_call_id);
    if (!base) continue;
    const layer = agentElement.parentElement;
    const maxX = Math.max(8, layer.clientWidth - agentElement.offsetWidth - 4);
    const maxY = Math.max(8, layer.clientHeight - agentElement.offsetHeight - 2);
    const x = clamp(base.x + randomBetween(-18, 18), 4, maxX);
    const y = clamp(base.y + randomBetween(-7, 7), 3, maxY);
    setAgentPosition(agentElement, x, y, randomBetween(2.6, 4.4));
  }
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function renderSummary() {
  elements.summaryRunning.textContent = String(
    state.agents.filter((agent) => ACTIVE_STATES.has(agent.state)).length,
  );
  elements.summaryAttention.textContent = String(
    state.agents.filter((agent) => ATTENTION_STATES.has(agent.state)).length,
  );
  elements.summaryPast.textContent = String(
    state.agents.filter((agent) => PAST_STATES.has(agent.state)).length,
  );
  elements.summaryIdle.textContent = String(
    state.agents.filter((agent) => agent.state === "idle").length,
  );
  const coordination = state.coordination;
  if (coordination && elements.coordinationStrip) {
    const memory = Number(coordination.memory_percent || 0).toFixed(1);
    elements.coordinationStrip.textContent = `${coordination.agents} isolated agents · ${coordination.queued_messages} queued · ${memory}% context`;
    if (elements.workspaceBranch) {
      elements.workspaceBranch.textContent = coordination.integration_branch || "Local branch";
    }
  }
}

function updateEmptyState() {
  const hasMatch = state.agents.some(agentMatches);
  elements.emptyState.hidden = hasMatch;
  if (hasMatch) return;
  const title = elements.emptyState.querySelector("strong");
  const message = elements.emptyState.querySelector("span");
  if (state.connected && state.agents.length === 0) {
    title.textContent = "No runs yet";
    message.textContent = "Use + Agent to launch a real Vibe run in a group.";
    return;
  }
  title.textContent = "No agents here";
  message.textContent = "Try another view or search.";
}

function openAgentPanel(agentId, view = "status") {
  state.selectedAgentId = agentId;
  state.detailView = view;
  elements.detailPanel.hidden = false;
  elements.detailPanel.inert = false;
  elements.shell.classList.add("has-detail");
  syncDetailModality();
  for (const element of document.querySelectorAll(".agent")) {
    element.classList.toggle("is-selected", element.dataset.agentId === agentId);
  }
  renderDetails();
  elements.closeDetailButton.focus();
}

function closeDetails({ restoreFocus = true } = {}) {
  const selectedId = state.selectedAgentId;
  state.selectedAgentId = null;
  elements.shell.classList.remove("has-detail");
  elements.detailPanel.hidden = true;
  elements.detailPanel.inert = true;
  syncDetailModality();
  for (const element of document.querySelectorAll(".agent.is-selected")) {
    element.classList.remove("is-selected");
  }
  if (restoreFocus && selectedId) {
    document
      .querySelector(`[data-agent-id="${CSS.escape(selectedId)}"] .agent-main`)
      ?.focus();
  }
}

function syncDetailModality() {
  const isOpen = Boolean(state.selectedAgentId);
  const isModal = isOpen && matchMedia("(max-width: 820px)").matches;
  elements.detailPanel.setAttribute("role", isModal ? "dialog" : "region");
  if (isModal) elements.detailPanel.setAttribute("aria-modal", "true");
  else elements.detailPanel.removeAttribute("aria-modal");
  elements.topbar.inert = isModal;
  elements.sidebar.inert = isModal;
  elements.mainContent.inert = isModal;
}

function renderDetails() {
  captureChatScroll();
  const agent = getAgent(state.selectedAgentId);
  if (!agent) {
    closeDetails({ restoreFocus: false });
    return;
  }

  const stateMeta = STATE_META[agent.state] || STATE_META.idle;
  elements.detailContent.classList.toggle("is-chat", state.detailView === "chat");
  for (const action of document.querySelectorAll(
    `[data-agent-id="${CSS.escape(agent.tool_call_id)}"] [data-agent-action]`,
  )) {
    const isActive = action.dataset.agentAction === state.detailView;
    action.classList.toggle("is-active", isActive);
    action.setAttribute("aria-pressed", String(isActive));
  }
  const coat = COATS[agent.coat] || COATS.orange;
  const initial = agent.agent_display_name.trim().slice(0, 1).toUpperCase();
  const section = document.createDocumentFragment();

  const identity = document.createElement("div");
  identity.className = "detail-identity";
  identity.style.setProperty("--detail-coat", coat[0]);
  identity.style.setProperty("--state-color", stateMeta.color);
  const avatar = document.createElement("div");
  avatar.className = "detail-avatar";
  avatar.textContent = initial;
  const identityCopy = document.createElement("div");
  const title = document.createElement("h2");
  title.id = "detail-title";
  title.textContent = agent.agent_display_name;
  const source = document.createElement("span");
  source.className = "detail-source";
  source.textContent = agent.is_orchestrator
    ? "Control agent"
    : agent.runtime_live
      ? "Isolated worker"
      : "Retained worker";
  title.append(source);
  const detailState = document.createElement("span");
  detailState.className = "detail-state";
  detailState.textContent = stateMeta.label;
  identityCopy.append(title, detailState);
  identity.append(avatar, identityCopy);

  const tabs = document.createElement("div");
  tabs.className = "detail-tabs";
  tabs.setAttribute("aria-label", `${agent.agent_display_name} views`);
  for (const [view, label] of [
    ["chat", "Chat"],
    ["history", "History"],
    ["status", "Status"],
  ]) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.textContent = label;
    tab.dataset.focusKey = `detail-tab:${view}`;
    tab.classList.toggle("is-active", state.detailView === view);
    tab.setAttribute("aria-pressed", String(state.detailView === view));
    tab.addEventListener("click", () => {
      state.detailView = view;
      renderDetails();
      requestAnimationFrame(() =>
        elements.detailContent
          .querySelector(`[data-focus-key="detail-tab:${CSS.escape(view)}"]`)
          ?.focus(),
      );
    });
    tabs.append(tab);
  }

  section.append(identity, tabs);
  if (state.detailView === "chat") section.append(renderChatView(agent));
  else if (state.detailView === "history") section.append(renderHistoryView(agent));
  else section.append(renderStatusView(agent, stateMeta));
  elements.detailContent.replaceChildren(section);
  const composer = elements.detailContent.querySelector(".chat-composer textarea");
  const selection = state.composerSelections.get(agent.tool_call_id);
  if (composer && selection) {
    composer.setSelectionRange(selection[0], selection[1]);
  }
  const transcript = elements.detailContent.querySelector(".transcript");
  if (transcript) {
    const saved = state.chatScroll.get(agent.tool_call_id);
    requestAnimationFrame(() => {
      transcript.scrollTop = saved?.stickToBottom === false
        ? saved.top
        : transcript.scrollHeight;
    });
  }
}

function renderStatusView(agent, stateMeta) {
  const fragment = document.createDocumentFragment();

  const taskSection = document.createElement("section");
  taskSection.className = "detail-section";
  taskSection.style.setProperty("--state-color", stateMeta.color);
  const taskHeading = document.createElement("h3");
  taskHeading.textContent = "Current run";
  const task = document.createElement("p");
  task.className = "detail-task";
  task.textContent = agent.task;
  const activity = document.createElement("p");
  activity.className = "detail-activity";
  activity.textContent = agent.current_activity || "No activity reported";
  taskSection.append(taskHeading, task, activity);

  const metricsSection = document.createElement("section");
  metricsSection.className = "detail-section";
  const metricsHeading = document.createElement("h3");
  metricsHeading.textContent = "Operational status";
  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  const promptTokens = agent.usage?.prompt_tokens;
  const completionTokens = agent.usage?.completion_tokens;
  const totalTokens =
    Number.isFinite(promptTokens) && Number.isFinite(completionTokens)
      ? promptTokens + completionTokens
      : null;
  appendMetric(metrics, formatDuration(runDuration(agent)), "Runtime");
  appendMetric(metrics, agent.model || "Not reported", "Model");
  appendMetric(metrics, agent.turns_used ?? "Not reported", "Turns");
  appendMetric(metrics, formatTokens(promptTokens), "Input tokens");
  appendMetric(metrics, formatTokens(completionTokens), "Output tokens");
  appendMetric(metrics, formatTokens(totalTokens), "Total tokens");
  appendMetric(
    metrics,
    formatMemory(agent.context_tokens, agent.context_limit),
    "Context memory",
  );
  appendMetric(metrics, formatCost(agent.estimated_cost_usd), "Estimated cost");
  appendMetric(metrics, formatTimestamp(agent.updated_at), "Last update");
  metricsSection.append(metricsHeading, metrics);

  const groupSection = document.createElement("section");
  groupSection.className = "detail-section";
  const groupLabel = document.createElement("label");
  groupLabel.className = "group-select";
  groupLabel.textContent = "Group";
  const groupSelect = document.createElement("select");
  groupSelect.dataset.focusKey = "detail-group";
  groupSelect.disabled = Boolean(agent.is_orchestrator);
  groupSelect.setAttribute("aria-label", `Move ${agent.agent_display_name} to group`);
  for (const group of state.groups) {
    const option = document.createElement("option");
    option.value = group.id;
    option.textContent = group.name;
    option.selected = group.id === agent.group_id;
    groupSelect.append(option);
  }
  groupSelect.addEventListener("change", () => moveAgent(agent.tool_call_id, groupSelect.value));
  groupLabel.append(groupSelect);
  const controlActions = document.createElement("div");
  controlActions.className = "detail-actions";
  const sessionId = agent.child_session_id || agent.parent_session_id;
  controlActions.append(
    actionButton("Copy session ID", () => copyText(sessionId)),
  );
  if (agent.runtime_live && ACTIVE_STATES.has(agent.state)) {
    controlActions.append(
      actionButton("Cancel response", () => cancelRun(agent.tool_call_id)),
    );
  }
  if (agent.runtime_live && !agent.is_orchestrator) {
    controlActions.append(
      actionButton("Stop agent", () => stopAgent(agent.tool_call_id), "button-danger"),
    );
  } else if (!agent.runtime_live && agent.merge_status === "ready") {
    controlActions.append(
      actionButton("Validate & merge", () => mergeAgent(agent.tool_call_id)),
    );
  } else if (!agent.runtime_live && !agent.is_orchestrator) {
    controlActions.append(
      actionButton("Continue chat", () => {
        state.detailView = "chat";
        renderDetails();
        requestAnimationFrame(() =>
          elements.detailContent.querySelector(".chat-composer textarea")?.focus(),
        );
      }),
    );
  }
  groupSection.append(groupLabel, controlActions);
  if (agent.source !== "live") {
    const notice = document.createElement("p");
    notice.className = "bridge-notice";
    notice.textContent =
      "This cat is demo data. Start the Agent Room runner to launch and control real Vibe sessions.";
    groupSection.append(notice);
  }

  const dataSection = document.createElement("section");
  dataSection.className = "detail-section";
  const dataHeading = document.createElement("h3");
  dataHeading.textContent = "Run data";
  const dataGrid = document.createElement("dl");
  dataGrid.className = "detail-grid";
  appendDetailData(dataGrid, "Profile", agent.agent_name);
  appendDetailData(dataGrid, "Started", formatTimestamp(agent.started_at));
  appendDetailData(dataGrid, "Parent session", agent.parent_session_id || "Not reported");
  appendDetailData(dataGrid, "Child session", agent.child_session_id || "Not applicable");
  appendDetailData(dataGrid, "Branch", agent.branch || "Not applicable");
  appendDetailData(dataGrid, "Worktree", agent.worktree_path || "Not applicable");
  appendDetailData(
    dataGrid,
    "Worktree state",
    agent.worktree_dirty
      ? `${agent.uncommitted_files || 0} uncommitted file(s)`
      : `${agent.new_commit_count || 0} new commit(s)`,
  );
  appendDetailData(dataGrid, "Merge", formatMergeState(agent));
  dataSection.append(dataHeading, dataGrid);

  fragment.append(taskSection, metricsSection, groupSection, dataSection);
  return fragment;
}

function renderChatView(agent) {
  const section = document.createElement("section");
  section.className = "detail-section chat-view";
  const heading = document.createElement("h3");
  heading.textContent = "Conversation";
  const transcript = document.createElement("div");
  transcript.className = "transcript";
  transcript.dataset.agentTranscript = agent.tool_call_id;
  transcript.addEventListener("scroll", () => {
    const distanceFromBottom =
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
    state.chatScroll.set(agent.tool_call_id, {
      top: transcript.scrollTop,
      stickToBottom: distanceFromBottom < 36,
    });
  });
  const messages = Array.isArray(agent.conversation) ? agent.conversation : [];
  if (messages.length === 0) {
    transcript.append(emptyPanel("No conversation has been reported for this run yet."));
  } else {
    for (const message of messages) {
      const bubble = document.createElement("div");
      bubble.className = "message";
      bubble.classList.toggle("is-user", message.role === "user");
      bubble.classList.toggle("is-system", message.role === "system");
      if (message.id) bubble.dataset.messageId = message.id;
      const role = document.createElement("strong");
      role.textContent =
        message.role === "user"
          ? "You"
          : message.role === "system"
            ? "Room"
            : agent.agent_display_name;
      const content = document.createElement("span");
      content.textContent = String(message.content || "");
      const status = document.createElement("small");
      status.className = "message-status";
      status.textContent = formatMessageStatus(message.status);
      bubble.append(role, content);
      if (Array.isArray(message.attachments) && message.attachments.length > 0) {
        const attachments = document.createElement("span");
        attachments.className = "message-attachments";
        attachments.textContent = message.attachments
          .map((attachment) => attachment.alias || "Image")
          .join(" · ");
        bubble.append(attachments);
      }
      if (message.status && message.status !== "succeeded") bubble.append(status);
      transcript.append(bubble);
    }
  }

  const pendingApprovals = (agent.approvals || []).filter(
    (approval) => approval.status === "pending",
  );
  for (const approval of pendingApprovals) {
    transcript.append(renderApprovalRequest(agent, approval));
  }
  const pendingQuestions = (agent.questions || []).filter(
    (question) => question.status === "pending",
  );
  for (const question of pendingQuestions) {
    transcript.append(renderQuestionRequest(agent, question));
  }

  section.append(heading, transcript);
  const chatError = state.chatErrors.get(agent.tool_call_id);
  if (chatError) {
    const error = document.createElement("p");
    error.className = "chat-error";
    error.setAttribute("role", "alert");
    error.textContent = chatError;
    section.append(error);
  }
  if (agent.source === "live") {
    if (!agent.runtime_live) {
      const notice = document.createElement("p");
      notice.className = "bridge-notice";
      notice.textContent = "Stopped · the next message resumes this conversation.";
      section.append(notice);
    }
    section.append(renderComposer(agent));
  } else {
    const notice = document.createElement("p");
    notice.className = "bridge-notice";
    notice.textContent = "Demo transcript · connect the Agent Room to chat.";
    section.append(notice);
  }
  return section;
}

function renderComposer(agent) {
  const wrapper = document.createElement("div");
  wrapper.className = "composer-wrap";
  const attachedImages = state.chatAttachments.get(agent.tool_call_id) || [];
  const attachmentList = document.createElement("div");
  attachmentList.className = "attachment-list";
  attachmentList.hidden = attachedImages.length === 0;
  for (const [index, attachment] of attachedImages.entries()) {
    const item = document.createElement("span");
    item.textContent = attachment.alias;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = `Remove ${attachment.alias}`;
    remove.setAttribute("aria-label", remove.title);
    remove.addEventListener("click", () => {
      const next = [...attachedImages];
      next.splice(index, 1);
      state.chatAttachments.set(agent.tool_call_id, next);
      renderDetails();
    });
    item.append(remove);
    attachmentList.append(item);
  }
  const commandMenu = document.createElement("div");
  commandMenu.className = "command-menu";
  commandMenu.hidden = true;
  commandMenu.setAttribute("role", "listbox");
  const form = document.createElement("form");
  form.className = "chat-composer";
  const textarea = document.createElement("textarea");
  textarea.rows = 2;
  textarea.maxLength = 8000;
  textarea.placeholder = agent.is_orchestrator
    ? "Message the orchestrator"
    : `Message ${agent.agent_display_name}`;
  textarea.value = state.chatDrafts.get(agent.tool_call_id) || "";
  textarea.dataset.focusKey = `chat-composer:${agent.tool_call_id}`;
  textarea.setAttribute("aria-label", textarea.placeholder);
  const send = document.createElement("button");
  send.type = "submit";
  send.className = "button button-orange composer-send";
  send.textContent = agent.runtime_live ? "Send" : "Resume & send";
  send.disabled =
    !state.connected ||
    state.pendingRequests.has(agent.tool_call_id) ||
    !textarea.value.trim();

  const updateCommandMenu = () => {
    const query = textarea.value.trim().toLowerCase();
    const commands = ["/help", "/status", "/history", "/queue", "/cancel", "/stop", "/retry"];
    const matches = query.startsWith("/") && !query.startsWith("//")
      ? commands.filter((command) => command.startsWith(query.split(" ")[0]))
      : [];
    commandMenu.replaceChildren();
    commandMenu.hidden = matches.length === 0;
    for (const command of matches) {
      const option = document.createElement("button");
      option.type = "button";
      option.setAttribute("role", "option");
      option.textContent = command;
      option.addEventListener("click", () => {
        textarea.value = command;
        state.chatDrafts.set(agent.tool_call_id, command);
        commandMenu.hidden = true;
        send.disabled = !state.connected;
        textarea.focus();
      });
      commandMenu.append(option);
    }
  };
  textarea.addEventListener("input", () => {
    state.chatDrafts.set(agent.tool_call_id, textarea.value);
    send.disabled =
      !state.connected ||
      state.pendingRequests.has(agent.tool_call_id) ||
      !textarea.value.trim();
    updateCommandMenu();
  });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !commandMenu.hidden) {
      commandMenu.hidden = true;
      event.stopPropagation();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendAgentMessage(agent.tool_call_id, textarea.value);
  });
  form.append(textarea, send);
  const queue = document.createElement("div");
  queue.className = "composer-meta";
  queue.textContent = agent.queued_messages
    ? `${agent.queued_messages} queued`
    : agent.state === "failed"
      ? "Ready to retry"
      : state.connected
        ? agent.runtime_live
          ? "Ready"
          : "Ready to resume"
        : "Bridge disconnected";
  const attachInput = document.createElement("input");
  attachInput.type = "file";
  attachInput.accept = "image/png,image/jpeg,image/gif,image/webp";
  attachInput.multiple = true;
  attachInput.hidden = true;
  const attach = document.createElement("button");
  attach.type = "button";
  attach.className = "button composer-attach";
  attach.textContent = "+ Image";
  attach.disabled = !state.connected || attachedImages.length >= 4;
  attach.addEventListener("click", () => attachInput.click());
  attachInput.addEventListener("change", async () => {
    try {
      const remaining = Math.max(0, 4 - attachedImages.length);
      const files = [...attachInput.files].slice(0, remaining);
      const additions = await Promise.all(files.map(imageAttachmentFromFile));
      state.chatAttachments.set(agent.tool_call_id, [
        ...attachedImages,
        ...additions,
      ]);
      state.chatErrors.delete(agent.tool_call_id);
    } catch (error) {
      state.chatErrors.set(agent.tool_call_id, error.message);
    }
    renderDetails();
  });
  queue.prepend(attach, attachInput);
  if (agent.runtime_live && ACTIVE_STATES.has(agent.state)) {
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "button composer-cancel";
    cancel.textContent = "Cancel response";
    cancel.addEventListener("click", () => void cancelRun(agent.tool_call_id));
    queue.append(cancel);
  }
  wrapper.append(attachmentList, commandMenu, form, queue);
  updateCommandMenu();
  return wrapper;
}

async function imageAttachmentFromFile(file) {
  const allowed = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
  if (!allowed.has(file.type)) throw new Error(`${file.name} is not a supported image`);
  if (file.size > 4 * 1024 * 1024) throw new Error(`${file.name} is larger than 4 MB`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  }
  return { alias: file.name, mime_type: file.type, data: btoa(binary) };
}

function renderApprovalRequest(agent, approval) {
  const card = document.createElement("section");
  card.className = "approval-request";
  const title = document.createElement("strong");
  title.textContent = `${approval.tool_name || "Tool"} needs approval`;
  const args = document.createElement("pre");
  args.textContent = JSON.stringify(approval.arguments || {}, null, 2);
  const permissions = document.createElement("p");
  permissions.textContent = (approval.permissions || [])
    .map((permission) => permission.label || permission.pattern)
    .filter(Boolean)
    .join(" · ");
  const actions = document.createElement("div");
  actions.className = "approval-actions";
  const approve = actionButton("Approve once", () =>
    resolveApproval(agent.tool_call_id, approval.id, "approve_once"),
  );
  const deny = actionButton(
    "Deny",
    () => resolveApproval(agent.tool_call_id, approval.id, "deny"),
    "button-danger",
  );
  actions.append(approve, deny);
  card.append(title, args);
  if (permissions.textContent) card.append(permissions);
  card.append(actions);
  return card;
}

function renderQuestionRequest(agent, request) {
  const form = document.createElement("form");
  form.className = "question-request";
  const answers = [];
  for (const [index, question] of (request.questions || []).entries()) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = question.question;
    fieldset.append(legend);
    const select = document.createElement("select");
    select.setAttribute("aria-label", question.question);
    select.required = true;
    for (const optionData of question.options || []) {
      const option = document.createElement("option");
      option.value = optionData.label;
      option.textContent = optionData.label;
      select.append(option);
    }
    const other = document.createElement("option");
    other.value = "__other__";
    other.textContent = "Other";
    if (!question.hide_other) select.append(other);
    const custom = document.createElement("input");
    custom.placeholder = "Your answer";
    custom.hidden = select.value !== "__other__";
    select.addEventListener("change", () => {
      custom.hidden = select.value !== "__other__";
      custom.required = !custom.hidden;
    });
    answers.push({ question, select, custom, index });
    fieldset.append(select, custom);
    form.append(fieldset);
  }
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button button-orange";
  submit.textContent = "Answer";
  form.append(submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = answers.map(({ question, select, custom }) => ({
      question: question.question,
      answer: select.value === "__other__" ? custom.value : select.value,
      is_other: select.value === "__other__",
    }));
    void answerQuestion(agent.tool_call_id, request.id, payload);
  });
  return form;
}

function renderHistoryView(agent) {
  const section = document.createElement("section");
  section.className = "detail-section";
  const heading = document.createElement("h3");
  heading.textContent = "Run history";
  const eventList = document.createElement("ol");
  eventList.className = "event-list";
  const events = Array.isArray(agent.events) ? [...agent.events].reverse() : [];
  if (events.length === 0) {
    section.append(heading, emptyPanel("No lifecycle events have been reported."));
    return section;
  }
  for (const event of events) {
    const item = document.createElement("li");
    if (typeof event === "string") {
      item.textContent = event;
    } else {
      const label = event.label || event.kind || "Activity";
      const timestamp = event.at ? ` · ${formatTimestamp(event.at)}` : "";
      const detail = event.detail ? ` — ${event.detail}` : "";
      item.textContent = `${label}${timestamp}${detail}`;
    }
    eventList.append(item);
  }
  section.append(heading, eventList);
  return section;
}

function appendMetric(grid, value, label) {
  const metric = document.createElement("div");
  metric.className = "metric";
  metric.dataset.metric = label.toLowerCase().replace(/\s+/g, "-");
  const amount = document.createElement("strong");
  amount.textContent = String(value);
  amount.title = String(value);
  const caption = document.createElement("span");
  caption.textContent = label;
  metric.append(amount, caption);
  grid.append(metric);
}

function emptyPanel(message) {
  const empty = document.createElement("div");
  empty.className = "empty-panel";
  empty.textContent = message;
  return empty;
}

function actionButton(label, handler, extraClass = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button ${extraClass}`.trim();
  button.dataset.focusKey = `detail-action:${label.toLowerCase().replace(/\s+/g, "-")}`;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function runDuration(agent) {
  const end = TERMINAL_STATES.has(agent.state) ? agent.updated_at : Date.now() / 1000;
  return end - agent.started_at;
}

function formatTokens(value) {
  return Number.isFinite(value) ? Number(value).toLocaleString() : "Not reported";
}

function formatMemory(used, limit) {
  if (!Number.isFinite(used) || !Number.isFinite(limit) || limit <= 0) {
    return "Not reported";
  }
  return `${Number(used).toLocaleString()} / ${Number(limit).toLocaleString()} (${((used / limit) * 100).toFixed(1)}%)`;
}

function formatMessageStatus(status) {
  return {
    queued: "Queued",
    running: "Running",
    failed: "Failed",
    cancelled: "Cancelled",
  }[status] || "";
}

function formatMergeState(agent) {
  if (agent.merge_status === "merged") return `Merged ${shortId(agent.merge_commit || "")}`;
  if (agent.merge_status === "failed") return agent.merge_error || "Validation failed";
  if (agent.merge_status === "ready") return "Ready to validate";
  if (agent.worktree_dirty) return "Commit changes first";
  if (agent.runtime_live) return "Stop agent before merge";
  return "No committed changes";
}

function formatCost(value) {
  return Number.isFinite(value) ? `$${Number(value).toFixed(4)}` : "Not reported";
}

function formatTimestamp(value) {
  if (!Number.isFinite(value)) return "Not reported";
  return new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value * 1000));
}

async function copyText(value) {
  if (!value) return;
  await navigator.clipboard.writeText(String(value));
}

async function cancelRun(agentId) {
  const response = await fetch(`/api/agent-runs/${encodeURIComponent(agentId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  if (response.ok) await refreshLiveRuns();
}

async function stopAgent(agentId) {
  await postAgentAction(agentId, "stop", {});
}

async function mergeAgent(agentId) {
  await postAgentAction(agentId, "merge", {});
}

async function sendAgentMessage(agentId, rawContent) {
  const content = rawContent.trim();
  if (!content || state.pendingRequests.has(agentId)) return;
  const agent = getAgent(agentId);
  if (!agent || agent.source !== "live" || !state.connected) return;
  const images = state.chatAttachments.get(agentId) || [];
  const clientMessageId = crypto.randomUUID();
  const optimistic = {
    id: `optimistic-${clientMessageId}`,
    client_message_id: clientMessageId,
    role: "user",
    content,
    attachments: images.map(({ alias, mime_type }) => ({ alias, mime_type })),
    status: "queued",
    created_at: Date.now() / 1000,
  };
  agent.conversation = [...(agent.conversation || []), optimistic];
  state.chatDrafts.set(agentId, "");
  state.chatErrors.delete(agentId);
  state.pendingRequests.add(agentId);
  renderDetails();
  try {
    const response = await fetch(
      `/api/agent-runs/${encodeURIComponent(agentId)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content,
          images,
          client_message_id: clientMessageId,
        }),
      },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Send failed (${response.status})`);
    if (state.chatAttachments.get(agentId) === images) {
      state.chatAttachments.delete(agentId);
    }
    state.lastSnapshot = "";
    await refreshLiveRuns();
  } catch (error) {
    optimistic.status = "failed";
    if (!state.chatDrafts.get(agentId)?.trim()) {
      state.chatDrafts.set(agentId, content);
    }
    state.chatErrors.set(agentId, error.message);
  } finally {
    state.pendingRequests.delete(agentId);
    if (state.selectedAgentId === agentId && state.detailView === "chat") {
      renderDetails();
      requestAnimationFrame(() =>
        elements.detailContent
          .querySelector(`[data-focus-key="chat-composer:${CSS.escape(agentId)}"]`)
          ?.focus(),
      );
    }
  }
}

async function resolveApproval(agentId, approvalId, decision) {
  await postAgentAction(agentId, `approvals/${encodeURIComponent(approvalId)}`, {
    decision,
  });
}

async function answerQuestion(agentId, questionId, answers) {
  await postAgentAction(agentId, `questions/${encodeURIComponent(questionId)}`, {
    answers,
  });
}

async function postAgentAction(agentId, action, body) {
  const requestKey = `${agentId}:${action}`;
  if (state.pendingRequests.has(requestKey)) return;
  state.pendingRequests.add(requestKey);
  state.chatErrors.delete(agentId);
  try {
    const response = await fetch(
      `/api/agent-runs/${encodeURIComponent(agentId)}/${action}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Action failed (${response.status})`);
    state.lastSnapshot = "";
    await refreshLiveRuns();
  } catch (error) {
    state.chatErrors.set(agentId, error.message);
    if (state.selectedAgentId === agentId) renderDetails();
  } finally {
    state.pendingRequests.delete(requestKey);
  }
}

function captureComposerState() {
  captureChatScroll();
  const composer = elements.detailContent.querySelector(".chat-composer textarea");
  if (!composer || !state.selectedAgentId) return;
  state.chatDrafts.set(state.selectedAgentId, composer.value);
  state.composerSelections.set(state.selectedAgentId, [
    composer.selectionStart,
    composer.selectionEnd,
  ]);
}

function captureChatScroll() {
  const transcript = elements.detailContent.querySelector(".transcript");
  const agentId = transcript?.dataset.agentTranscript;
  if (!transcript || !agentId) return;
  const distanceFromBottom =
    transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
  state.chatScroll.set(agentId, {
    top: transcript.scrollTop,
    stickToBottom: distanceFromBottom < 36,
  });
}

function appendDetailData(list, label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = String(value);
  detail.title = String(value);
  wrapper.append(term, detail);
  list.append(wrapper);
}

function shortId(value) {
  return value.length > 17 ? `${value.slice(0, 14)}…` : value;
}

function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Math.round(seconds));
  if (safeSeconds < 60) return `${safeSeconds}s`;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}m ${remainder}s`;
}

function getAgent(agentId) {
  return state.agents.find((agent) => agent.tool_call_id === agentId);
}

async function moveAgent(agentId, groupId) {
  const agent = getAgent(agentId);
  if (!agent || !state.groups.some((group) => group.id === groupId)) return;
  const previousGroup = agent.group_id;
  agent.group_id = groupId;
  persistState();
  render();
  if (agent.source !== "live" || !state.connected) return;
  try {
    const response = await fetch(
      `/api/agent-runs/${encodeURIComponent(agentId)}/group`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ group_id: groupId }),
      },
    );
    if (!response.ok) throw new Error();
  } catch {
    agent.group_id = previousGroup;
    persistState();
    render();
  }
}

function handleAgentDragStart(event) {
  const element = event.currentTarget;
  element.classList.add("is-dragging");
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", element.dataset.agentId);
}

function handleAgentDragEnd(event) {
  event.currentTarget.classList.remove("is-dragging");
  for (const zone of elements.zones.querySelectorAll(".is-dragover")) {
    zone.classList.remove("is-dragover");
  }
}

function handleZoneDragOver(event) {
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  event.currentTarget.classList.add("is-dragover");
}

function handleZoneDragLeave(event) {
  if (!event.currentTarget.contains(event.relatedTarget)) {
    event.currentTarget.classList.remove("is-dragover");
  }
}

function handleZoneDrop(event) {
  event.preventDefault();
  const zone = event.currentTarget;
  zone.classList.remove("is-dragover");
  moveAgent(event.dataTransfer.getData("text/plain"), zone.dataset.groupId);
}

function renderSwatches() {
  elements.groupSwatches.replaceChildren();
  for (const color of GROUP_COLORS) {
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = "swatch";
    swatch.classList.toggle("is-selected", color === state.selectedSwatch);
    swatch.setAttribute("aria-pressed", String(color === state.selectedSwatch));
    swatch.style.setProperty("--swatch", color);
    swatch.setAttribute("aria-label", `Select ${color}`);
    swatch.addEventListener("click", () => {
      state.selectedSwatch = color;
      renderSwatches();
    });
    elements.groupSwatches.append(swatch);
  }
}

function createGroup() {
  const name = elements.groupName.value.trim();
  if (!name) return;
  const idRoot = name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "group";
  let id = idRoot;
  let suffix = 2;
  while (id === "all" || state.groups.some((group) => group.id === id)) {
    id = `${idRoot}-${suffix}`;
    suffix += 1;
  }
  state.groups.push({ id, name, color: state.selectedSwatch, custom: true });
  state.selectedGroupId = "all";
  persistState();
  render();
}

function openAgentDialog(groupId, origin) {
  state.agentDialogOrigin = origin || elements.newAgentButton;
  elements.agentForm.reset();
  elements.agentDialogError.hidden = true;
  elements.agentDialogError.textContent = "";
  elements.agentGroup.replaceChildren();
  for (const group of state.groups) {
    const option = document.createElement("option");
    option.value = group.id;
    option.textContent = group.name;
    elements.agentGroup.append(option);
  }
  const preferredGroup =
    groupId && groupId !== "all"
      ? groupId
      : state.selectedGroupId !== "all"
        ? state.selectedGroupId
        : UNASSIGNED_GROUP.id;
  elements.agentGroup.value = preferredGroup;

  if (state.profiles.length > 0) {
    elements.agentProfile.replaceChildren();
    for (const profile of state.profiles) {
      const option = document.createElement("option");
      option.value = profile.name;
      option.textContent = profile.display_name || profile.name;
      elements.agentProfile.append(option);
    }
    const preferredProfile = state.profiles.find(
      (profile) => profile.name === "default",
    );
    if (preferredProfile) elements.agentProfile.value = preferredProfile.name;
  }
  elements.agentAutoApprove.checked = true;
  renderAgentToolPicker();

  elements.agentSubmit.disabled = !state.connected;
  elements.agentSubmit.textContent = state.connected ? "Launch agent" : "Bridge unavailable";
  updateAgentLaunchNotice();
  elements.agentDialog.showModal();
  requestAnimationFrame(() => elements.agentName.focus());
}

function renderAgentToolPicker() {
  elements.agentToolList.replaceChildren();
  for (const tool of state.tools) {
    const label = document.createElement("label");
    label.className = "tool-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = tool.name;
    checkbox.checked = true;
    const name = document.createElement("span");
    name.textContent = tool.display_name || tool.name;
    label.append(checkbox, name);
    elements.agentToolList.append(label);
  }
  syncAgentToolSelection();
}

function syncAgentToolSelection() {
  const tools = [...elements.agentToolList.querySelectorAll('input[type="checkbox"]')];
  const checked = tools.filter((tool) => tool.checked).length;
  elements.agentToolsAll.checked = tools.length > 0 && checked === tools.length;
  elements.agentToolsAll.indeterminate = checked > 0 && checked < tools.length;
  updateAgentLaunchNotice();
}

function selectedAgentTools() {
  return [...elements.agentToolList.querySelectorAll('input[type="checkbox"]:checked')]
    .map((tool) => tool.value);
}

function updateAgentLaunchNotice() {
  if (!state.connected) {
    elements.agentDialogNotice.textContent =
      "This page is using demo data. Start server.py to enable real Vibe runs.";
    return;
  }
  const toolCount = selectedAgentTools().length;
  const approval = elements.agentAutoApprove.checked ? "Auto-approve" : "Ask first";
  elements.agentDialogNotice.textContent =
    `${approval} · ${toolCount} tool${toolCount === 1 ? "" : "s"} · isolated worktree`;
}

async function createAgentRun() {
  elements.agentSubmit.disabled = true;
  elements.agentSubmit.textContent = "Launching…";
  elements.agentDialogError.hidden = true;
  try {
    const response = await fetch("/api/agent-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: elements.agentProfile.value,
        display_name: elements.agentName.value.trim(),
        task: elements.agentTask.value.trim(),
        group_id: elements.agentGroup.value,
        enabled_tools: selectedAgentTools(),
        auto_approve: elements.agentAutoApprove.checked,
        client_message_id: crypto.randomUUID(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Launch failed (${response.status})`);
    elements.agentDialog.close();
    state.lastSnapshot = "";
    await refreshLiveRuns();
    const group = state.groups.find((candidate) => candidate.id === payload.group_id);
    elements.roomAnnouncer.textContent = `${payload.agent_display_name} queued in ${group?.name || "Unassigned"}.`;
    openAgentPanel(payload.tool_call_id, "status");
  } catch (error) {
    elements.agentDialogError.textContent = error.message;
    elements.agentDialogError.hidden = false;
  } finally {
    elements.agentSubmit.disabled = !state.connected;
    elements.agentSubmit.textContent = state.connected ? "Launch agent" : "Bridge unavailable";
  }
}

function refreshRuntimeMetric() {
  if (state.detailView !== "status" || !state.selectedAgentId) return;
  const agent = getAgent(state.selectedAgentId);
  const value = elements.detailContent.querySelector('[data-metric="runtime"] strong');
  if (agent && value) value.textContent = formatDuration(runDuration(agent));
}

function updateClock() {
  elements.officeClock.textContent = new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}

function bindEvents() {
  elements.statusFilters.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    state.statusFilter = button.dataset.filter;
    for (const filter of elements.statusFilters.querySelectorAll("button")) {
      filter.classList.toggle("is-active", filter === button);
      filter.setAttribute("aria-pressed", String(filter === button));
    }
    renderZones();
    updateEmptyState();
  });

  elements.agentSearch.addEventListener("input", () => {
    state.search = elements.agentSearch.value.trim().toLowerCase();
    renderZones();
    updateEmptyState();
  });

  elements.closeDetailButton.addEventListener("click", () => closeDetails());
  document.addEventListener("keydown", (event) => {
    if (
      event.key === "Escape" &&
      state.selectedAgentId &&
      !elements.groupDialog.open &&
      !elements.agentDialog.open
    ) {
      closeDetails();
    }
  });

  const openGroupDialog = () => {
    elements.groupForm.reset();
    state.selectedSwatch = GROUP_COLORS[state.groups.length % GROUP_COLORS.length];
    renderSwatches();
    elements.groupDialog.showModal();
    requestAnimationFrame(() => elements.groupName.focus());
  };
  elements.addGroupButton.addEventListener("click", openGroupDialog);
  elements.toolbarAddGroupButton?.addEventListener("click", openGroupDialog);
  elements.newAgentButton.addEventListener("click", () =>
    openAgentDialog(state.selectedGroupId, elements.newAgentButton),
  );

  elements.groupForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    if (!elements.groupForm.reportValidity()) {
      event.preventDefault();
      return;
    }
    createGroup();
  });

  elements.agentForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    if (!elements.agentForm.reportValidity()) return;
    void createAgentRun();
  });

  elements.agentDialog.addEventListener("close", () => {
    state.agentDialogOrigin?.focus();
    state.agentDialogOrigin = null;
  });
  elements.agentProfile.addEventListener("change", updateAgentLaunchNotice);
  elements.agentAutoApprove.addEventListener("change", updateAgentLaunchNotice);
  elements.agentToolsAll.addEventListener("change", () => {
    for (const tool of elements.agentToolList.querySelectorAll('input[type="checkbox"]')) {
      tool.checked = elements.agentToolsAll.checked;
    }
    syncAgentToolSelection();
  });
  elements.agentToolList.addEventListener("change", syncAgentToolSelection);

  elements.simulateButton.addEventListener("click", () => {
    state.motionPaused = !state.motionPaused;
    document.body.classList.toggle("motion-paused", state.motionPaused);
    elements.simulateButton.textContent = state.motionPaused ? "Resume motion" : "Pause motion";
  });

  window.addEventListener("resize", () => {
    placeAgents();
    syncDetailModality();
  });
}

async function start() {
  bindEvents();
  renderSwatches();
  updateClock();
  try {
    await loadRoom();
    render();
  } catch (error) {
    setFeedStatus("Feed unavailable", true);
    elements.emptyState.hidden = false;
    elements.emptyState.querySelector("strong").textContent = "Agent feed unavailable";
    elements.emptyState.querySelector("span").textContent = error.message;
  }
  setInterval(roamAgents, 4200);
  setInterval(updateClock, 30000);
  setInterval(refreshRuntimeMetric, 1000);
  setInterval(() => void refreshLiveRuns(), 1200);
}

start();
