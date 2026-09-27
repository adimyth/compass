const state = {
  sessionId: `scope_${crypto.randomUUID().replaceAll("-", "")}`,
  snapshot: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);

function upper(value) {
  return String(value || "—").replaceAll("_", " ").toUpperCase();
}

function latestEvent(snapshot) {
  const events = snapshot?.events || [];
  return events.length ? events[events.length - 1] : null;
}

async function request(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Scope Check request failed.");
  return payload;
}

function setBusy(busy) {
  state.busy = busy;
  $("#step-button").disabled = busy;
  $("#run-button").disabled = busy;
  $("#reset-button").disabled = busy;
  $("#confirm-button").disabled = busy || state.snapshot?.status !== "awaiting_confirmation";
}

function renderProbabilities(event) {
  const target = $("#probabilities");
  target.replaceChildren();
  if (!event?.probabilities) {
    const copy = document.createElement("p");
    copy.textContent = "Run the first check to see the complete distribution.";
    target.append(copy);
    return;
  }
  Object.entries(event.probabilities).sort((a, b) => b[1] - a[1]).forEach(([scope, probability]) => {
    const row = document.createElement("div");
    row.className = `prob-row ${scope} ${scope === event.scope ? "selected" : ""}`;
    const label = document.createElement("span");
    label.textContent = upper(scope);
    const track = document.createElement("div");
    track.className = "prob-track";
    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    track.append(fill);
    const value = document.createElement("b");
    value.textContent = `${Math.round(probability * 100)}%`;
    row.append(label, track, value);
    target.append(row);
  });
}

function renderTrace(snapshot) {
  const events = snapshot.events || [];
  const target = $("#trace-list");
  target.replaceChildren();
  if (!events.length) {
    const item = document.createElement("li");
    item.innerHTML = "<span>00</span><strong>Awaiting first proposal</strong><small>No tool call has reached Compass.</small>";
    target.append(item);
    return;
  }
  events.forEach((event, index) => {
    const item = document.createElement("li");
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const summary = document.createElement("strong");
    summary.textContent = `${event.title} → ${upper(event.scope)}`;
    const outcome = document.createElement("small");
    outcome.textContent = upper(event.disposition);
    item.append(number, summary, outcome);
    target.append(item);
  });
}

function renderDecision(event) {
  const scope = $("#decision-scope");
  const policy = $("#policy-disposition");
  if (!event) {
    scope.textContent = "AWAITING CHECK";
    scope.className = "";
    $("#decision-model").textContent = "—";
    $("#decision-confidence").textContent = "—";
    $("#decision-latency").textContent = "—";
    $("#decision-tokens").textContent = "—";
    $("#decision-match").textContent = "—";
    policy.textContent = "AWAITING DECISION";
    policy.className = "policy-disposition neutral";
    $("#policy-note").textContent = "Compass cannot execute a tool call. The local policy owns the outcome.";
    renderProbabilities(null);
    return;
  }
  scope.textContent = upper(event.scope);
  scope.className = event.scope;
  $("#decision-model").textContent = event.model;
  $("#decision-confidence").textContent = `${Math.round(event.confidence * 100)}%`;
  $("#decision-latency").textContent = `${event.latency_ms.toFixed(0)} ms`;
  $("#decision-tokens").textContent = event.input_tokens.toLocaleString();
  $("#decision-match").textContent = event.matches_trace_expectation ? "MATCHED" : "MISMATCH";
  policy.textContent = upper(event.disposition);
  policy.className = `policy-disposition ${event.disposition}`;
  $("#policy-note").textContent = event.policy_note;
  renderProbabilities(event);
}

function render(snapshot) {
  state.snapshot = snapshot;
  const current = snapshot.case;
  const status = snapshot.status;
  const statusLabel = {
    ready: "SYSTEM READY",
    active: "TRACE ACTIVE",
    awaiting_confirmation: "WAITING FOR USER",
    complete: "TRACE COMPLETE",
  }[status] || upper(status);
  const statusNode = $("#trace-status");
  statusNode.textContent = statusLabel;
  statusNode.className = status;
  $("#case-label").textContent = current ? `TRACE ${String(snapshot.progress.current + 1).padStart(2, "0")} / ${upper(current.title)}` : "TRACE COMPLETE / CALENDAR UPDATE SIMULATED";
  if (current) {
    $("#user-request").textContent = current.user_request;
    $("#tool-result").textContent = current.untrusted_tool_result;
    $("#proposed-action").textContent = JSON.stringify(current.proposed_action, null, 2);
    $("#case-note").textContent = current.note;
  } else {
    $("#user-request").textContent = "The requested hiking event reached the confirmation boundary.";
    $("#tool-result").textContent = "No additional tool result is needed.";
    $("#proposed-action").textContent = "{\n  \"status\": \"trace complete\"\n}";
    $("#case-note").textContent = "The simulator contacted no inbox, drive, calendar, or payment service.";
  }
  $("#step-button").disabled = state.busy || status === "complete" || status === "awaiting_confirmation";
  $("#run-button").disabled = state.busy || status === "complete" || status === "awaiting_confirmation";
  $("#confirm-button").disabled = state.busy || status !== "awaiting_confirmation";
  renderDecision(latestEvent(snapshot));
  renderTrace(snapshot);
}

async function reset() {
  $("#error-message").textContent = "";
  const snapshot = await request("/api/reset", { session_id: state.sessionId });
  render(snapshot);
}

async function step() {
  setBusy(true);
  $("#error-message").textContent = "";
  try {
    const snapshot = await request("/api/step", { session_id: state.sessionId });
    render(snapshot);
  } catch (error) {
    $("#error-message").textContent = error.message;
  } finally {
    setBusy(false);
    render(state.snapshot);
  }
}

async function confirm() {
  setBusy(true);
  $("#error-message").textContent = "";
  try {
    const snapshot = await request("/api/confirm", { session_id: state.sessionId });
    render(snapshot);
  } catch (error) {
    $("#error-message").textContent = error.message;
  } finally {
    setBusy(false);
    render(state.snapshot);
  }
}

async function runTrace() {
  setBusy(true);
  $("#error-message").textContent = "";
  try {
    while (state.snapshot.status !== "complete" && state.snapshot.status !== "awaiting_confirmation") {
      const snapshot = await request("/api/step", { session_id: state.sessionId });
      render(snapshot);
      if (snapshot.status === "complete" || snapshot.status === "awaiting_confirmation") break;
      await new Promise((resolve) => window.setTimeout(resolve, 650));
    }
  } catch (error) {
    $("#error-message").textContent = error.message;
  } finally {
    setBusy(false);
    render(state.snapshot);
  }
}

$("#step-button").addEventListener("click", step);
$("#run-button").addEventListener("click", runTrace);
$("#confirm-button").addEventListener("click", confirm);
$("#reset-button").addEventListener("click", reset);
reset().catch((error) => { $("#error-message").textContent = error.message; });
