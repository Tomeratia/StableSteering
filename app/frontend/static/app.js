const traceLog = document.getElementById("trace-log");
const pageStatus = document.getElementById("page-status");
const progressPanel = document.getElementById("progress-panel");
const progressBar = document.getElementById("progress-bar");
const progressValue = document.getElementById("progress-value");
const progressLabel = document.getElementById("progress-label");
const configYamlEditor = document.getElementById("config-yaml-editor");
const reloadConfigButton = document.getElementById("reload-config-button");

function appendTrace(message) {
  if (!traceLog) return;
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} ${message}`;
  traceLog.prepend(item);
}

function setStatus(message, isError = false) {
  if (!pageStatus) return;
  pageStatus.hidden = !message;
  pageStatus.textContent = message || "";
  pageStatus.classList.toggle("error", Boolean(isError));
}

function setProgress(progress, label = "Working...") {
  if (!progressPanel || !progressBar || !progressValue || !progressLabel) return;
  const clamped = Math.max(0, Math.min(100, Number(progress || 0)));
  progressPanel.hidden = false;
  progressBar.style.width = `${clamped}%`;
  progressValue.textContent = `${clamped}%`;
  progressLabel.textContent = label;
}

function clearProgress() {
  if (!progressPanel || !progressBar || !progressValue || !progressLabel) return;
  progressPanel.hidden = true;
  progressBar.style.width = "0%";
  progressValue.textContent = "0%";
  progressLabel.textContent = "Working...";
}

function traceFrontend(event, details = {}) {
  const payload = {
    event,
    page: window.location.pathname,
    session_id: document.getElementById("next-round-button")?.dataset.sessionId || null,
    round_id: document.getElementById("submit-feedback-button")?.dataset.roundId || null,
    details,
  };
  appendTrace(`${event} ${JSON.stringify(details)}`);
  console.info("[StableSteering trace]", payload);
  try {
    const body = JSON.stringify(payload);
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon("/frontend-events", blob);
      return;
    }
    fetch("/frontend-events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch((error) => console.warn("Failed to persist frontend trace event", error));
  } catch (error) {
    console.warn("Failed to persist frontend trace event", error);
  }
}

async function postJson(url, body) {
  traceFrontend("http.request.started", { url, body_keys: Object.keys(body || {}) });
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    const message = "Could not reach the server. Make sure the app is still running, then try again.";
    traceFrontend("http.request.failed", { url, detail: message, error_name: error?.name || "network_error" });
    throw new Error(message);
  }
  if (!response.ok) {
    const text = await response.text();
    let message = text || `Request failed: ${response.status}`;
    try {
      const parsed = JSON.parse(text);
      message = parsed.message || parsed.detail || message;
    } catch {
      message = text || message;
    }
    traceFrontend("http.request.failed", { url, status: response.status, detail: message });
    throw new Error(message);
  }
  const data = await response.json();
  traceFrontend("http.request.completed", { url, status: response.status });
  return data;
}

async function getJson(url) {
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function pollJob(statusUrl, { onProgress } = {}) {
  for (;;) {
    const response = await fetch(statusUrl, { method: "GET" });
    if (!response.ok) {
      throw new Error(`Failed to poll job status: ${response.status}`);
    }
    const job = await response.json();
    if (onProgress) {
      onProgress(job);
    }
    if (job.state === "succeeded") {
      return job.result;
    }
    if (job.state === "failed") {
      throw new Error(job.error || "Asynchronous job failed.");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 250));
  }
}

async function runNextRoundJob(sessionId, { queuedLabel, runningFallbackLabel } = {}) {
  const job = await postJson(`/sessions/${sessionId}/rounds/next/async`, {});
  setStatus(queuedLabel || "Queueing round generation...");
  setProgress(55, queuedLabel || "Queueing next round");
  await pollJob(job.status_url, {
    onProgress: (snapshot) => {
      setStatus(snapshot.status_message);
      setProgress(snapshot.progress, snapshot.status_message || runningFallbackLabel || "Generating next round");
    },
  });
}

function collectRatings() {
  return Array.from(document.querySelectorAll(".rating-input")).map((input) => ({
    candidateId: input.dataset.candidateId,
    rating: Number(input.value || 0),
  })).filter((entry) => entry.rating > 0);
}

function ratingCaption(value) {
  if (!value) {
    return "No rating selected yet";
  }
  if (value === 1) return "1 star selected";
  return `${value} stars selected`;
}

function applyStarRating(candidateId, value) {
  const numericValue = Number(value || 0);
  const hiddenInput = document.querySelector(`.rating-input[data-candidate-id="${candidateId}"]`);
  if (hiddenInput) {
    hiddenInput.value = String(numericValue);
  }
  const buttons = Array.from(document.querySelectorAll(`.star-button[data-candidate-id="${candidateId}"]`));
  buttons.forEach((button) => {
    const buttonValue = Number(button.dataset.ratingValue || 0);
    const active = numericValue > 0 && buttonValue <= numericValue;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  const caption = document.querySelector(`.star-rating-caption[data-candidate-id="${candidateId}"]`);
  if (caption) {
    caption.textContent = ratingCaption(numericValue);
  }
}

function buildFeedbackPayload(feedbackMode) {
  if (feedbackMode === "pairwise") {
    const winner = document.querySelector(".pairwise-winner-input:checked")?.dataset.candidateId;
    const loser = document.querySelector(".pairwise-loser-input:checked")?.dataset.candidateId;
    if (!winner || !loser) {
      throw new Error("Pairwise feedback requires one explicit winner and one explicit loser.");
    }
    if (winner === loser) {
      throw new Error("Pairwise feedback requires different winner and loser candidates.");
    }
    return {
      feedback_type: "pairwise",
      payload: {
        winner_candidate_id: winner,
        loser_candidate_id: loser,
      },
    };
  }

  if (feedbackMode === "winner_only") {
    const winner = document.querySelector(".winner-only-input:checked")?.dataset.candidateId;
    if (!winner) {
      throw new Error("Winner-only feedback requires choosing one winner.");
    }
    return {
      feedback_type: "winner_only",
      payload: {
        winner_candidate_id: winner,
      },
    };
  }

  if (feedbackMode === "approve_reject") {
    const approvalInputs = Array.from(document.querySelectorAll(".approve-checkbox"));
    const approvals = Object.fromEntries(
      approvalInputs.map((input) => [input.dataset.candidateId, Boolean(input.checked)])
    );
    const approvedEntries = approvalInputs.filter((input) => input.checked);
    if (!approvedEntries.length) {
      throw new Error("Approve/reject feedback requires at least one approved candidate.");
    }
    const winner = document.querySelector(".approve-winner-input:checked")?.dataset.candidateId;
    if (!winner) {
      throw new Error("Approve/reject feedback requires choosing the preferred approved winner.");
    }
    if (!approvals[winner]) {
      throw new Error("The approve/reject winner must also be marked approved.");
    }
    return {
      feedback_type: "approve_reject",
      payload: {
        winner_candidate_id: winner,
        approvals,
      },
    };
  }

  if (feedbackMode === "top_k") {
    const ranks = Array.from(document.querySelectorAll(".rank-input"))
      .map((input) => ({
        candidateId: input.dataset.candidateId,
        rank: Number(input.value || 0),
      }))
      .filter((entry) => entry.rank > 0);
    if (ranks.length < 2) {
      throw new Error("Top-k feedback requires ranking at least two candidates.");
    }
    const uniqueRanks = new Set(ranks.map((entry) => entry.rank));
    if (uniqueRanks.size !== ranks.length) {
      throw new Error("Top-k feedback requires unique ranks.");
    }
    ranks.sort((left, right) => left.rank - right.rank || left.candidateId.localeCompare(right.candidateId));
    const approvals = Object.fromEntries(
      ranks.map((entry) => [entry.candidateId, entry.rank])
    );
    return {
      feedback_type: "top_k",
      payload: {
        ranking: ranks.map((entry) => entry.candidateId),
        ranks: approvals,
      },
    };
  }

  if (feedbackMode === "best_vs_incumbent") {
    const selected = document.querySelector(".bvi-winner-input:checked");
    if (!selected) throw new Error("Choose either the current winner or the new challenger before submitting.");
    const winnerId = selected.dataset.candidateId;
    const isIncumbent = selected.dataset.isIncumbent === "true";
    const incumbentInput = document.querySelector(".bvi-winner-input[data-is-incumbent='true']");
    const challengerInput = document.querySelector(".bvi-winner-input[data-is-incumbent='false']");
    return {
      feedback_type: "best_vs_incumbent",
      payload: {
        winner_candidate_id: winnerId,
        incumbent_candidate_id: incumbentInput?.dataset.candidateId || "",
        challenger_candidate_id: challengerInput?.dataset.candidateId || "",
        kept_incumbent: isIncumbent,
      },
    };
  }

  if (feedbackMode === "critique_rating") {
    const ratingEntries = collectRatings();
    if (!ratingEntries.length) {
      throw new Error("Critique rating feedback requires at least one explicit rating.");
    }
    const ratings = Object.fromEntries(ratingEntries.map((entry) => [entry.candidateId, entry.rating]));
    return {
      feedback_type: "critique_rating",
      payload: {
        ratings,
        critique_tags: collectCritiqueTags(),
      },
    };
  }

  const ratingEntries = collectRatings();
  if (!ratingEntries.length) {
    throw new Error("Scalar rating feedback requires at least one explicit rating.");
  }
  const sorted = [...ratingEntries].sort((left, right) => right.rating - left.rating || left.candidateId.localeCompare(right.candidateId));
  const ratings = Object.fromEntries(ratingEntries.map((entry) => [entry.candidateId, entry.rating]));

  return {
    feedback_type: "scalar_rating",
    payload: { ratings },
  };
}

function collectCritiqueTags() {
  const tags = {};
  Array.from(document.querySelectorAll(".critique-tag-pill[aria-pressed='true']")).forEach((pill) => {
    const candidateId = pill.dataset.candidateId;
    const tag = pill.dataset.tag;
    if (!candidateId || !tag) {
      return;
    }
    if (!tags[candidateId]) {
      tags[candidateId] = [];
    }
    tags[candidateId].push(tag);
  });
  return tags;
}

// Quick-config panel: bidirectional sync between dropdowns and YAML editor.
const QUICK_CONFIG_FIELDS = {
  "qc-feedback-mode": "feedback_mode",
  "qc-updater": "updater",
  "qc-sampler": "sampler",
  "qc-steering-mode": "steering_mode",
};

function yamlSetField(yaml, key, value) {
  const re = new RegExp(`^(${key}:\\s*)(.+)$`, "m");
  return re.test(yaml) ? yaml.replace(re, `$1${value}`) : yaml;
}

function yamlGetField(yaml, key) {
  const m = yaml.match(new RegExp(`^${key}:\\s*(.+)$`, "m"));
  return m ? m[1].trim() : null;
}

function syncDropdownsFromYaml(yaml) {
  for (const [selectId, yamlKey] of Object.entries(QUICK_CONFIG_FIELDS)) {
    const val = yamlGetField(yaml, yamlKey);
    const select = document.getElementById(selectId);
    if (select && val) {
      const match = Array.from(select.options).find((o) => o.value === val);
      if (match) select.value = val;
    }
  }
}

const qcEditor = document.getElementById("config-yaml-editor");
if (qcEditor) {
  syncDropdownsFromYaml(qcEditor.value);
  qcEditor.addEventListener("input", () => syncDropdownsFromYaml(qcEditor.value));

  for (const [selectId, yamlKey] of Object.entries(QUICK_CONFIG_FIELDS)) {
    const select = document.getElementById(selectId);
    if (!select) continue;
    select.addEventListener("change", () => {
      qcEditor.value = yamlSetField(qcEditor.value, yamlKey, select.value);
    });
  }
}

const setupForm = document.getElementById("setup-form");
const setupSubmitButton = document.getElementById("setup-submit-button");
if (setupForm) {
  traceFrontend("page.loaded", { view: "setup" });
  reloadConfigButton?.addEventListener("click", async () => {
    reloadConfigButton.disabled = true;
    setStatus("Reloading default YAML template...");
    traceFrontend("setup.config.reload.clicked");
    try {
      const payload = await getJson("/setup/config-template");
      if (configYamlEditor) {
        configYamlEditor.value = payload.config_yaml || "";
      }
      setStatus("Default YAML template reloaded.");
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      reloadConfigButton.disabled = false;
    }
  });
  setupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    traceFrontend("setup.submit.clicked");
    if (setupSubmitButton) setupSubmitButton.disabled = true;
    if (reloadConfigButton) reloadConfigButton.disabled = true;
    setStatus("Creating experiment and session from YAML configuration...");
    try {
      const form = new FormData(setupForm);
      const payload = await postJson("/setup/session", {
        experiment_name: form.get("experiment_name"),
        description: form.get("description"),
        prompt: form.get("prompt"),
        negative_prompt: form.get("negative_prompt"),
        config_yaml: form.get("config_yaml"),
      });
      traceFrontend("experiment.created", { experiment_id: payload.experiment.id });
      traceFrontend("session.created", { session_id: payload.session.id });
      setStatus("Session created. Opening the interactive view...");
      window.location.href = `/sessions/${payload.session.id}/view`;
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      if (setupSubmitButton) setupSubmitButton.disabled = false;
      if (reloadConfigButton) reloadConfigButton.disabled = false;
    }
  });
}

const nextRoundButton = document.getElementById("next-round-button");
if (nextRoundButton) {
  traceFrontend("page.loaded", { view: "session", session_id: nextRoundButton.dataset.sessionId });
  nextRoundButton.addEventListener("click", async () => {
    if (nextRoundButton.disabled) return;
    const sessionId = nextRoundButton.dataset.sessionId;
    traceFrontend("round.generate.clicked", { session_id: sessionId });
    nextRoundButton.disabled = true;
    setStatus("Queueing round generation...");
    setProgress(5, "Queueing next round");
    try {
      await runNextRoundJob(sessionId, {
        queuedLabel: "Queueing round generation...",
        runningFallbackLabel: "Generating next round",
      });
      setStatus("Round generated. Refreshing session view...");
      setProgress(100, "Round completed");
      window.location.reload();
    } catch (error) {
      setStatus(error.message, true);
      clearProgress();
      nextRoundButton.disabled = false;
    }
  });
}

Array.from(document.querySelectorAll(".star-button")).forEach((button) => {
  button.addEventListener("click", () => {
    if (button.disabled) return;
    const candidateId = button.dataset.candidateId;
    const value = Number(button.dataset.ratingValue || 0);
    applyStarRating(candidateId, value);
    traceFrontend("feedback.rating.selected", { candidate_id: candidateId, rating: value });
  });
});

Array.from(document.querySelectorAll(".critique-tag-pill")).forEach((pill) => {
  pill.addEventListener("click", () => {
    if (pill.disabled) return;
    const selected = pill.getAttribute("aria-pressed") === "true";
    pill.setAttribute("aria-pressed", selected ? "false" : "true");
    pill.classList.toggle("selected", !selected);
    traceFrontend("feedback.critique_tag.toggled", {
      candidate_id: pill.dataset.candidateId,
      tag: pill.dataset.tag,
      selected: !selected,
    });
  });
});

const submitFeedbackButton = document.getElementById("submit-feedback-button");
if (submitFeedbackButton) {
  traceFrontend("round.visible", { round_id: submitFeedbackButton.dataset.roundId });
  submitFeedbackButton.addEventListener("click", async () => {
    if (submitFeedbackButton.disabled) return;
    const feedbackMode = submitFeedbackButton.dataset.feedbackMode || "scalar_rating";
    const sessionId = submitFeedbackButton.dataset.sessionId;
    try {
      const request = buildFeedbackPayload(feedbackMode);
      traceFrontend("feedback.submit.clicked", {
        round_id: submitFeedbackButton.dataset.roundId,
        feedback_mode: feedbackMode,
        interaction_mode: feedbackMode,
      });
      submitFeedbackButton.disabled = true;
      setStatus("Queueing feedback submission...");
      setProgress(5, "Queueing feedback");
      const job = await postJson(`/rounds/${submitFeedbackButton.dataset.roundId}/feedback/async`, {
        ...request,
      });
      await pollJob(job.status_url, {
        onProgress: (snapshot) => {
          setStatus(snapshot.status_message);
          setProgress(snapshot.progress, snapshot.status_message || "Applying feedback");
        },
      });
      traceFrontend("feedback.submit.completed", {
        round_id: submitFeedbackButton.dataset.roundId,
        session_id: sessionId,
      });
      setStatus("Feedback applied. Starting the next round...");
      setProgress(50, "Preparing next round");
      await runNextRoundJob(sessionId, {
        queuedLabel: "Queueing next round after feedback...",
        runningFallbackLabel: "Generating the next round",
      });
      setStatus("Next round ready. Refreshing session view...");
      setProgress(100, "Next round completed");
      window.location.reload();
    } catch (error) {
      setStatus(error.message, true);
      clearProgress();
      submitFeedbackButton.disabled = false;
    }
  });
}
