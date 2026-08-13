const state = {
  stats: null,
  discovery: { enabled: false, models: [] },
  busy: new Set(),
  conversations: new Map(),
  chatModel: null,
  chatAbort: null,
  refreshTimer: null,
};

const colors = ["#c7ff5b", "#65a9ff", "#c77dff", "#ff9f68", "#55d6be", "#f47da5", "#ffd166"];
const $ = (selector) => document.querySelector(selector);

document.addEventListener("DOMContentLoaded", () => {
  wireInterface();
  showNetworkWarning();
  addArgumentRow();
  refreshAll();
  scheduleRefresh();
});

function wireInterface() {
  $("#refresh-button").addEventListener("click", refreshAll);
  $("#new-profile-button").addEventListener("click", openProfileDialog);
  $("#profile-filter").addEventListener("input", renderProfiles);
  $("#add-argument").addEventListener("click", () => addArgumentRow());
  $("#raw-args-toggle").addEventListener("change", toggleRawArguments);
  $("#profile-form").addEventListener("submit", createProfile);
  $("#chat-form").addEventListener("submit", sendChatMessage);
  $("#stop-chat").addEventListener("click", stopChat);
  $("#clear-chat").addEventListener("click", clearChat);
  $("#chat-dialog").addEventListener("cancel", stopChat);
  document.addEventListener("visibilitychange", scheduleRefresh);
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = document.getElementById(button.dataset.closeDialog);
      if (dialog.id === "chat-dialog" && state.chatAbort) stopChat();
      dialog.close();
    });
  });
}

function showNetworkWarning() {
  const host = window.location.hostname.replace(/^\[|\]$/g, "");
  const loopback = host === "localhost" || host === "::1" || host.startsWith("127.");
  $("#network-warning").classList.toggle("hidden", loopback);
}

function scheduleRefresh() {
  window.clearInterval(state.refreshTimer);
  state.refreshTimer = window.setInterval(refreshAll, document.hidden ? 10000 : 2000);
}

async function refreshAll() {
  try {
    state.stats = await api("/control/stats");
    setConnection(true);
    renderOverview();
    renderProfiles();
    $("#last-updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    setConnection(false);
    if (!state.stats) $("#profile-list").innerHTML = '<div class="empty-state">The pool API is unavailable.</div>';
  }
}

function setConnection(online) {
  const element = $("#connection-status");
  element.className = `connection ${online ? "online" : "offline"}`;
  element.lastChild.textContent = online ? " Online" : " Offline";
}

function renderOverview() {
  const stats = state.stats;
  const system = stats.system;
  $("#available-memory").textContent = formatBytes(system.available_bytes);
  $("#available-percent").textContent = `${formatPercent(system.available_bytes, system.total_bytes)} of ${formatBytes(system.total_bytes)}`;
  $("#pool-usage").textContent = formatBytes(stats.pool_usage_bytes);
  $("#pool-budget").textContent = stats.pool_budget_bytes
    ? `${formatPercent(stats.pool_usage_bytes, stats.pool_budget_bytes)} of ${formatBytes(stats.pool_budget_bytes)} budget`
    : "Unlimited budget";
  $("#running-count").textContent = stats.running_models;
  $("#registered-count").textContent = `${stats.registered_models} registered`;
  $("#active-count").textContent = stats.active_requests;
  $("#starting-count").textContent = `${stats.starting_models} starting`;
  renderMemoryBar();
}

function renderMemoryBar() {
  const { system, pool_usage_bytes: poolUsage, pool_budget_bytes: budget, processes } = state.stats;
  const total = Math.max(system.total_bytes, 1);
  const loaded = processes.filter((profile) => profile.actual_memory_bytes && profile.pid);
  const modelContainer = $("#model-segments");
  const legend = $("#memory-legend");
  modelContainer.replaceChildren();
  legend.replaceChildren();

  loaded.forEach((profile) => {
    const color = modelColor(profile.id);
    const segment = document.createElement("div");
    segment.className = "model-segment";
    segment.style.width = `${Math.max(0, profile.actual_memory_bytes / Math.max(poolUsage, 1) * 100)}%`;
    segment.style.background = color;
    segment.title = memoryBreakdown(profile);
    modelContainer.append(segment);
    legend.append(legendItem(color, profile.id, formatBytes(profile.actual_memory_bytes)));
  });
  modelContainer.style.width = `${Math.min(100, poolUsage / total * 100)}%`;

  const outside = Math.max(0, system.outside_pool_resident_bytes);
  const availableCapacity = Math.max(0, total - poolUsage - outside);
  $("#memory-free").style.width = `${availableCapacity / total * 100}%`;
  $("#memory-other").style.width = `${outside / total * 100}%`;
  $("#memory-other").title = `Other system use: ${formatBytes(outside)}`;
  $("#memory-free").title = `Available capacity: ${formatBytes(availableCapacity)}`;
  legend.append(legendItem("#59616d", "Other system use", formatBytes(outside)));
  legend.append(legendItem("#151a20", "Available capacity", formatBytes(availableCapacity)));

  placeMarker("#budget-marker", budget ? budget / total : null);
  placeMarker("#normal-marker", (total - system.normal_headroom_bytes) / total);
  placeMarker("#critical-marker", (total - system.critical_headroom_bytes) / total);
  $("#memory-bar").setAttribute(
    "aria-label",
    `Pool ${formatBytes(poolUsage)}, available capacity ${formatBytes(availableCapacity)}, other system use ${formatBytes(outside)}, system available ${formatBytes(system.available_bytes)}`,
  );
}

function legendItem(color, label, value) {
  const item = document.createElement("span");
  item.className = "legend-item";
  const swatch = document.createElement("i");
  swatch.className = "legend-swatch";
  swatch.style.background = color;
  const text = document.createElement("span");
  const name = document.createElement("b");
  name.textContent = label;
  text.append(name, ` · ${value}`);
  item.append(swatch, text);
  return item;
}

function placeMarker(selector, fraction) {
  const marker = $(selector);
  marker.classList.toggle("hidden", fraction === null);
  if (fraction !== null) marker.style.left = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

function renderProfiles() {
  if (!state.stats) return;
  const filter = $("#profile-filter").value.trim().toLowerCase();
  const order = { running: 0, starting: 1, stopping: 2, failed: 3, registered: 4 };
  const profiles = [...state.stats.processes]
    .filter((profile) => `${profile.id} ${profile.model_path}`.toLowerCase().includes(filter))
    .sort((a, b) => order[a.status] - order[b.status] || a.id.localeCompare(b.id));
  const list = $("#profile-list");
  list.replaceChildren();
  if (!profiles.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = filter ? "No profiles match this filter." : "No profiles registered yet.";
    list.append(empty);
    return;
  }
  profiles.forEach((profile) => list.append(profileCard(profile)));
}

function profileCard(profile) {
  const card = document.createElement("article");
  card.className = "profile-card";
  const activeText = profile.active_requests
    ? `${profile.active_requests} active request${profile.active_requests === 1 ? "" : "s"}`
    : "Idle";
  const memory = profile.actual_memory_bytes
    ? formatBytes(profile.actual_memory_bytes)
    : `${formatBytes(profile.predicted_memory_bytes)} predicted`;
  const lastUsed = profile.last_used_at ? relativeTime(profile.last_used_at) : "Never used";

  const identity = document.createElement("div");
  identity.className = "profile-identity";
  const titleRow = document.createElement("div");
  titleRow.className = "profile-title-row";
  const title = document.createElement("h3");
  title.className = "profile-title";
  title.textContent = profile.id;
  const status = document.createElement("span");
  status.className = `status-pill ${profile.status}`;
  status.textContent = profile.status;
  titleRow.append(title, status);
  const path = document.createElement("p");
  path.className = "profile-path";
  path.title = profile.model_path;
  path.textContent = profile.model_path;
  identity.append(titleRow, path);

  const memoryStat = profileStat("Memory", memory, activeText, profile.active_requests > 0);
  memoryStat.title = memoryBreakdown(profile);
  card.append(identity, memoryStat, profileStat("Eviction", `Priority ${profile.priority}`, lastUsed));

  const actions = document.createElement("div");
  actions.className = "profile-actions";
  const busy = state.busy.has(profile.id) || ["starting", "stopping"].includes(profile.status);
  if (profile.status === "running") {
    actions.append(actionButton("Chat", "quiet", busy, () => openChat(profile.id)));
    actions.append(actionButton("Unload", "quiet", busy, () => profileAction(profile, "unload")));
  } else {
    actions.append(actionButton(profile.status === "failed" ? "Retry" : "Load", "primary", busy, () => profileAction(profile, "load")));
    actions.append(actionButton("Force", "quiet", busy, () => profileAction(profile, "force")));
  }
  actions.append(actionButton("Delete", "danger", busy, () => profileAction(profile, "delete")));
  card.append(actions);

  if (profile.last_error) {
    const error = document.createElement("p");
    error.className = "profile-error";
    error.textContent = profile.last_error;
    card.append(error);
  }
  return card;
}

function profileStat(label, value, subtitle = null, subtitleActive = false) {
  const stat = document.createElement("div");
  stat.className = "profile-stat";
  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value;
  stat.append(name, content);
  if (subtitle) {
    const extra = document.createElement("span");
    if (subtitleActive) extra.className = "activity";
    extra.textContent = subtitle;
    stat.append(extra);
  }
  return stat;
}

function actionButton(label, kind, disabled, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button button-${kind} button-small`;
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", handler);
  return button;
}

async function profileAction(profile, action) {
  const messages = {
    unload: `Unload ${profile.id}? Any active requests will be interrupted.`,
    delete: `Delete profile ${profile.id}? Its process will stop and the registration cannot be recovered.`,
    force: `Force load ${profile.id}? This may evict a profile with active requests, but will not bypass memory limits.`,
  };
  if (messages[action] && !window.confirm(messages[action])) return;
  state.busy.add(profile.id);
  renderProfiles();
  try {
    if (action === "load") await api(`/control/models/${encodeURIComponent(profile.id)}/start`, { method: "POST", body: { force: false } });
    if (action === "force") await api(`/control/models/${encodeURIComponent(profile.id)}/start`, { method: "POST", body: { force: true } });
    if (action === "unload") await api(`/control/models/${encodeURIComponent(profile.id)}/unload`, { method: "POST" });
    if (action === "delete") await api(`/control/models/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
    toast(`${profile.id} ${action === "delete" ? "deleted" : action === "unload" ? "unloaded" : "loaded"}.`);
  } catch (error) {
    if (action === "load" && error.status === 507 && window.confirm(`${error.message}\n\nForce loading may interrupt an active request. Continue?`)) {
      try {
        await api(`/control/models/${encodeURIComponent(profile.id)}/start`, { method: "POST", body: { force: true } });
        toast(`${profile.id} force loaded.`);
      } catch (forcedError) {
        toast(forcedError.message, true);
      }
    } else {
      toast(error.message, true);
    }
  } finally {
    state.busy.delete(profile.id);
    await refreshAll();
  }
}

async function openProfileDialog() {
  const dialog = $("#profile-dialog");
  $("#profile-form-error").classList.add("hidden");
  try {
    state.discovery = await api("/control/model-files");
  } catch (error) {
    state.discovery = { enabled: false, models: [] };
    toast(`Model discovery failed: ${error.message}`, true);
  }
  populateModelOptions();
  dialog.showModal();
  $("#profile-id").focus();
}

function populateModelOptions() {
  const select = $("#model-path");
  select.replaceChildren();
  const choices = new Map();
  state.discovery.models.forEach((model) => choices.set(model.path, `${model.relative_path} · ${formatBytes(model.size_bytes)}`));
  (state.stats?.processes || []).forEach((profile) => {
    if (!choices.has(profile.model_path)) choices.set(profile.model_path, `${profile.model_path.split("/").pop()} · existing profile`);
  });
  if (!choices.size) {
    const option = document.createElement("option");
    option.textContent = "No model files available";
    option.value = "";
    select.append(option);
  } else {
    [...choices].sort((a, b) => a[1].localeCompare(b[1])).forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.append(option);
    });
  }
  const disabled = !state.discovery.enabled;
  $("#discovery-warning").classList.toggle("hidden", !disabled);
  $("#model-path-help").textContent = disabled
    ? "Only files referenced by existing profiles are available"
    : `${state.discovery.models.length} GGUF file${state.discovery.models.length === 1 ? "" : "s"} discovered`;
  $("#create-profile-submit").disabled = !choices.size;
}

function addArgumentRow(argument = "", value = "") {
  const row = document.createElement("div");
  row.className = "argument-row";
  const argumentInput = document.createElement("input");
  argumentInput.className = "argument-name";
  argumentInput.placeholder = "--argument";
  argumentInput.value = argument;
  const valueInput = document.createElement("input");
  valueInput.className = "argument-value";
  valueInput.placeholder = "value (optional)";
  valueInput.value = value;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "icon-button";
  remove.setAttribute("aria-label", "Remove argument");
  remove.textContent = "×";
  remove.addEventListener("click", () => row.remove());
  row.append(argumentInput, valueInput, remove);
  $("#argument-rows").append(row);
}

function toggleRawArguments() {
  const raw = $("#raw-args-toggle").checked;
  if (raw) $("#raw-arguments").value = JSON.stringify(argumentsFromRows());
  $("#argument-editor").classList.toggle("hidden", raw);
  $("#raw-arguments-field").classList.toggle("hidden", !raw);
}

function argumentsFromRows() {
  const result = [];
  document.querySelectorAll(".argument-row").forEach((row) => {
    const argument = row.querySelector(".argument-name").value.trim();
    const value = row.querySelector(".argument-value").value.trim();
    if (argument) result.push(argument);
    if (argument && value) result.push(value);
  });
  return result;
}

async function createProfile(event) {
  event.preventDefault();
  const errorElement = $("#profile-form-error");
  errorElement.classList.add("hidden");
  let args;
  try {
    args = $("#raw-args-toggle").checked ? JSON.parse($("#raw-arguments").value || "[]") : argumentsFromRows();
    if (!Array.isArray(args) || !args.every((item) => typeof item === "string")) throw new Error("Raw arguments must be a JSON array of strings.");
  } catch (error) {
    errorElement.textContent = error.message;
    errorElement.classList.remove("hidden");
    return;
  }
  const estimate = $("#memory-estimate").value;
  const payload = {
    id: $("#profile-id").value,
    model_path: $("#model-path").value,
    args,
    priority: Number($("#profile-priority").value),
    initialize: $("#initialize-profile").checked,
  };
  if (estimate) payload.estimated_memory_bytes = Number(estimate);
  const submit = $("#create-profile-submit");
  submit.disabled = true;
  submit.textContent = payload.initialize ? "Creating and loading…" : "Creating…";
  try {
    await api("/control/models", { method: "POST", body: payload });
    $("#profile-dialog").close();
    $("#profile-form").reset();
    $("#argument-rows").replaceChildren();
    addArgumentRow();
    toggleRawArguments();
    toast(`${payload.id} created.`);
    await refreshAll();
  } catch (error) {
    errorElement.textContent = error.message;
    errorElement.classList.remove("hidden");
  } finally {
    submit.disabled = false;
    submit.textContent = "Create profile";
  }
}

function openChat(modelId) {
  state.chatModel = modelId;
  if (!state.conversations.has(modelId)) state.conversations.set(modelId, []);
  $("#chat-title").textContent = modelId;
  renderChat();
  $("#chat-dialog").showModal();
  $("#chat-input").focus();
}

function renderChat() {
  const container = $("#chat-messages");
  container.replaceChildren();
  const messages = state.conversations.get(state.chatModel) || [];
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "Send a message to test this profile.";
    container.append(empty);
    return;
  }
  messages.forEach((message) => container.append(messageElement(message)));
  container.scrollTop = container.scrollHeight;
}

function messageElement(message) {
  const element = document.createElement("div");
  element.className = `message ${message.role}${message.error ? " error" : ""}`;
  const role = document.createElement("span");
  role.className = "message-role";
  role.textContent = message.role === "user" ? "You" : "Assistant";
  element.append(role);
  if (message.reasoning) {
    const details = document.createElement("details");
    details.className = "reasoning";
    const summary = document.createElement("summary");
    summary.textContent = "Reasoning";
    const content = document.createElement("div");
    content.className = "reasoning-content";
    content.textContent = message.reasoning;
    details.append(summary, content);
    element.append(details);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.content || (message.pending ? "…" : "");
  element.append(content);
  return element;
}

async function sendChatMessage(event) {
  event.preventDefault();
  if (state.chatAbort) return;
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  const conversation = state.conversations.get(state.chatModel);
  conversation.push({ role: "user", content: text });
  const assistant = { role: "assistant", content: "", reasoning: "", pending: true };
  conversation.push(assistant);
  input.value = "";
  renderChat();
  setChatBusy(true);
  state.chatAbort = new AbortController();
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.chatModel,
        messages: conversation.slice(0, -1).map(({ role, content }) => ({ role, content })),
        stream: true,
      }),
      signal: state.chatAbort.signal,
    });
    if (!response.ok) throw await apiError(response);
    await consumeSse(response, (data) => {
      const delta = data.choices?.[0]?.delta || {};
      assistant.content += delta.content || "";
      assistant.reasoning += delta.reasoning_content || "";
      renderChat();
    });
  } catch (error) {
    if (error.name === "AbortError") assistant.content ||= "Generation stopped.";
    else {
      assistant.content ||= error.message;
      assistant.error = true;
    }
  } finally {
    assistant.pending = false;
    state.chatAbort = null;
    setChatBusy(false);
    renderChat();
    refreshAll();
  }
}

async function consumeSse(response, onData) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload && payload !== "[DONE]") onData(JSON.parse(payload));
      }
    }
    if (done) break;
  }
}

function stopChat() {
  state.chatAbort?.abort();
}

function clearChat() {
  if (state.chatAbort) return;
  state.conversations.set(state.chatModel, []);
  renderChat();
}

function setChatBusy(busy) {
  $("#send-chat").disabled = busy;
  $("#chat-input").disabled = busy;
  $("#clear-chat").disabled = busy;
  $("#stop-chat").classList.toggle("hidden", !busy);
}

async function api(path, options = {}) {
  const request = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, request);
  if (!response.ok) throw await apiError(response);
  if (response.status === 204) return null;
  return response.json();
}

async function apiError(response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    message = body.error?.message || body.detail || message;
  } catch (_) {
    // Retain the HTTP status when an upstream response is not JSON.
  }
  const error = new Error(message);
  error.status = response.status;
  return error;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function memoryBreakdown(profile) {
  if (!profile.actual_memory_bytes) return `${profile.id}: ${formatBytes(profile.predicted_memory_bytes)} predicted`;
  const parts = [
    `${profile.id}: ${formatBytes(profile.actual_memory_bytes)} system RAM`,
    `${formatBytes(profile.process_memory_bytes)} process PSS/RSS`,
  ];
  if (profile.gpu_shared_memory_bytes) parts.push(`${formatBytes(profile.gpu_shared_memory_bytes)} GPU shared system memory`);
  if (profile.gpu_dedicated_memory_bytes) parts.push(`${formatBytes(profile.gpu_dedicated_memory_bytes)} dedicated VRAM`);
  return parts.join(" · ");
}

function modelColor(modelId) {
  let hash = 0;
  for (const character of modelId) hash = (hash * 31 + character.codePointAt(0)) >>> 0;
  return colors[hash % colors.length];
}

function formatPercent(value, total) {
  if (!total) return "0%";
  const percent = value / total * 100;
  return `${percent >= 10 ? percent.toFixed(0) : percent.toFixed(1)}%`;
}

function relativeTime(timestamp) {
  const seconds = Math.max(0, Date.now() / 1000 - timestamp);
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function toast(message, error = false) {
  const element = document.createElement("div");
  element.className = `toast${error ? " error" : ""}`;
  element.textContent = message;
  $("#toast-region").append(element);
  window.setTimeout(() => element.remove(), 5000);
}
