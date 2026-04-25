const state = {
  username: null,
  profile: null,
  questions: [],
  reports: [],
  posture: { status: "Not analyzed", feedback: "No posture analysis yet" },
  stream: null,
  speechRecognizer: null,
  confidenceChart: null,
  toneChart: null,
  speechTimers: {},
  rlSessionStarted: false,
  practiceThread: null,
  coachSummary: null,
  firebase: {
    auth: null,
    enabled: false,
    serverEnabled: false,
    webConfigured: false,
    webConfig: null,
  }
};

function skillLabel(skill) {
  if (skill === "dsa") return "DSA";
  if (skill === "system_design") return "System Design";
  return "Communication";
}

function fmtSigned(num, decimals = 2) {
  if (typeof num !== "number" || Number.isNaN(num)) return "-";
  const sign = num >= 0 ? "+" : "";
  return `${sign}${num.toFixed(decimals)}`;
}

function clamp01(num) {
  if (typeof num !== "number" || Number.isNaN(num)) return 0;
  return Math.max(0, Math.min(1, num));
}

async function refreshCoachSummary() {
  try {
    const summary = await api("/api/coach/summary");
    state.coachSummary = summary;
    return summary;
  } catch (_e) {
    state.coachSummary = null;
    return null;
  }
}

function applyCoachSettingsToUi(summary) {
  const settings = summary?.settings;
  if (!settings) return;

  const personality = document.getElementById("coachPersonality");
  const adaptive = document.getElementById("adaptivePersonality");
  const fixWeakness = document.getElementById("fixWeaknessMode");

  if (personality) {
    personality.value = settings.coach_personality || "friendly";
    personality.disabled = Boolean(settings.adaptive_personality);
  }
  if (adaptive) adaptive.checked = Boolean(settings.adaptive_personality);
  if (fixWeakness) fixWeakness.checked = settings.training_mode === "fix_weakness";

  const note = document.getElementById("weaknessModeNote");
  if (note) {
    if (settings.training_mode === "fix_weakness") {
      const weakest = summary?.skills?.weakest ? skillLabel(summary.skills.weakest) : "weakest";
      note.textContent = `Mode: Fix Weakness → Only asks ${weakest} until improved.`;
    } else {
      note.textContent = "Mode: Random interview questions.";
    }
  }
}

function renderCoachSummaryToProgress(summary) {
  if (!summary) return;
  const readinessEl = document.getElementById("readinessNumbers");
  const skillsEl = document.getElementById("skillBreakdown");
  const weakestEl = document.getElementById("weakestSkillNote");
  const effEl = document.getElementById("effectivenessNumbers");
  const scorecardEl = document.getElementById("scorecardNumbers");

  const readiness = summary?.readiness?.readiness || {};
  if (readinessEl) {
    readinessEl.innerHTML = `Start: ${typeof readiness.start === "number" ? readiness.start.toFixed(2) : "-"}<br />End: ${typeof readiness.end === "number" ? readiness.end.toFixed(2) : "-"}`;
  }

  const s = summary?.skills?.scores || {};
  if (skillsEl) {
    const dsa = Math.round(clamp01(s.dsa) * 100);
    const sys = Math.round(clamp01(s.system_design) * 100);
    const comm = Math.round(clamp01(s.communication) * 100);
    skillsEl.innerHTML = `DSA → ${dsa}%<br />System Design → ${sys}%<br />Communication → ${comm}%`;
  }

  if (weakestEl) {
    const weakest = summary?.skills?.weakest || "communication";
    const lvl = summary?.skills?.levels?.[weakest] || "";
    weakestEl.textContent = `Weakest skill: ${skillLabel(weakest)}${lvl ? ` (${lvl})` : ""}. Then improve weakest.`;
  }

  const eff = summary?.effectiveness || {};
  if (effEl) {
    effEl.innerHTML = `Hint → ${fmtSigned(eff.hint)} improvement<br />Example → ${fmtSigned(eff.example)} improvement<br />Follow-up → ${fmtSigned(eff.follow_up)} improvement`;
  }

  const before = summary?.scorecard?.before || {};
  const after = summary?.scorecard?.after || {};
  if (scorecardEl) {
    scorecardEl.innerHTML = `Before: ${typeof before.confidence === "number" ? before.confidence.toFixed(2) : "-"} / ${typeof before.clarity === "number" ? before.clarity.toFixed(2) : "-"} / ${typeof before.technical_depth === "number" ? before.technical_depth.toFixed(2) : "-"}<br />After: ${typeof after.confidence === "number" ? after.confidence.toFixed(2) : "-"} / ${typeof after.clarity === "number" ? after.clarity.toFixed(2) : "-"} / ${typeof after.technical_depth === "number" ? after.technical_depth.toFixed(2) : "-"}`;
  }
}

function renderAgentBrain(result) {
  const wrap = document.getElementById("agentBrain");
  const out = document.getElementById("agentBrainText");
  if (!wrap || !out) return;

  const brain = result?.agent_brain;
  if (!brain) {
    wrap.classList.add("hidden");
    out.textContent = "";
    return;
  }

  const score = typeof brain?.state?.score === "number" ? brain.state.score.toFixed(2) : "-";
  const weak = brain?.state?.weak ? skillLabel(brain.state.weak) : "-";
  const action = brain?.action || "-";
  const reason = brain?.reason || "-";
  const example = brain?.example;

  let text = `State:\n- Score: ${score}\n- Weak: ${weak}\n\nAction:\n-> ${action}\n\nReason:\n-> ${reason}`;
  if (example) {
    text += `\n\nExample:\n${String(example).trim()}`;
  }

  out.textContent = text;
  wrap.classList.remove("hidden");
}

function readFirebaseBootstrapConfig() {
  const el = document.getElementById("firebaseBootstrap");
  if (!el) return { enabled: false, web_config: null };
  try {
    const parsed = JSON.parse(el.textContent || "{}");
    return {
      enabled: Boolean(parsed?.enabled),
      web_config: parsed?.web_config && typeof parsed.web_config === "object" ? parsed.web_config : null,
    };
  } catch (_e) {
    return { enabled: false, web_config: null };
  }
}

function initFirebaseEmailLinkAuth() {
  const box = document.getElementById("firebaseEmailLinkBox");
  const boot = readFirebaseBootstrapConfig();
  const cfg = boot.web_config;

  state.firebase.serverEnabled = Boolean(boot.enabled);
  state.firebase.webConfig = cfg;
  state.firebase.webConfigured = Boolean(cfg && cfg.apiKey && cfg.authDomain && cfg.projectId);
  state.firebase.enabled = state.firebase.webConfigured;

  if (!state.firebase.webConfigured || typeof window.firebase === "undefined") {
    if (box) box.classList.add("hidden");
    return;
  }

  try {
    if (!firebase.apps?.length) {
      firebase.initializeApp(cfg);
    }
    state.firebase.auth = firebase.auth();
    if (box) box.classList.remove("hidden");
  } catch (_e) {
    state.firebase.auth = null;
    state.firebase.enabled = false;
    if (box) box.classList.add("hidden");
  }
}

async function sendFirebaseEmailSignInLink() {
  if (!state.firebase.auth) {
    showToast("Firebase email link is not configured.", "error");
    return;
  }

  const email = document.getElementById("emailLinkEmail")?.value?.trim();
  if (!email) {
    showToast("Please enter your email address.", "error");
    return;
  }

  setLoading(true, "Sending sign-in link...");
  try {
    const actionCodeSettings = {
      url: window.location.origin + window.location.pathname,
      handleCodeInApp: true,
    };
    await state.firebase.auth.sendSignInLinkToEmail(email, actionCodeSettings);
    window.localStorage.setItem("firebaseEmailForSignIn", email);
    setLoading(false);
    showToast("Sign-in link sent. Check your email.", "success");
  } catch (e) {
    setLoading(false);
    showToast(e?.message || "Unable to send sign-in link.", "error");
  }
}

async function completeFirebaseEmailLinkSignIn() {
  if (!state.firebase.auth) {
    return;
  }

  const url = window.location.href;
  if (!state.firebase.auth.isSignInWithEmailLink(url)) {
    return;
  }

  const storedEmail = window.localStorage.getItem("firebaseEmailForSignIn") || "";
  const inputEmail = document.getElementById("emailLinkEmail")?.value?.trim() || "";
  const email = storedEmail || inputEmail;
  if (!email) {
    showToast("Enter your email to complete sign-in.", "error");
    return;
  }

  if (!state.firebase.serverEnabled) {
    showToast("Server Firebase is not configured. Set FIREBASE_SERVICE_ACCOUNT_* env vars.", "error");
    return;
  }

  setLoading(true, "Signing you in...");
  try {
    const cred = await state.firebase.auth.signInWithEmailLink(email, url);
    const user = cred?.user;
    const idToken = user ? await user.getIdToken() : "";
    if (!idToken) throw new Error("Unable to get Firebase token");

    await api("/api/auth/firebase/session", { method: "POST", body: { id_token: idToken } });
    window.localStorage.removeItem("firebaseEmailForSignIn");
    try {
      window.history.replaceState({}, document.title, window.location.pathname);
    } catch (_e) {
      // Ignore.
    }

    setLoading(false);
    await loginFlow();
    showToast("Signed in via email link.", "success");
  } catch (e) {
    setLoading(false);
    showToast(e?.message || "Email link sign-in failed.", "error");
  }
}

function resetPracticeThread(reason = "") {
  state.practiceThread = null;

  const thread = document.getElementById("practiceThread");
  const items = document.getElementById("practiceThreadItems");
  const promptText = document.getElementById("practicePromptText");

  if (items) items.innerHTML = "";
  if (thread) thread.classList.add("hidden");

  // Reset prompt to the selected/root question.
  const q = currentQuestion();
  if (promptText) promptText.textContent = q;

  if (reason) showToast(reason);
}

function getActivePracticeQuestion() {
  if (state.practiceThread?.stopped) return currentQuestion();
  if (state.practiceThread?.nextQuestion) return state.practiceThread.nextQuestion;
  return currentQuestion();
}

function ensurePracticeThread(rootQuestion) {
  if (state.practiceThread && !state.practiceThread.stopped) return;
  state.practiceThread = {
    rootQuestion,
    turns: [],
    nextQuestion: null,
    stopped: false
  };
}

function appendPracticeTurn(question, answer) {
  const thread = document.getElementById("practiceThread");
  const items = document.getElementById("practiceThreadItems");
  if (!thread || !items) return;

  thread.classList.remove("hidden");

  const wrapper = document.createElement("div");
  wrapper.className = "practice-thread-item";

  const qLabel = document.createElement("small");
  qLabel.textContent = "Question";
  const qText = document.createElement("strong");
  qText.textContent = question;
  const qBlock = document.createElement("div");
  qBlock.className = "practice-thread-block";
  qBlock.appendChild(qLabel);
  qBlock.appendChild(qText);

  const aLabel = document.createElement("small");
  aLabel.textContent = "Your answer";
  const aText = document.createElement("div");
  aText.className = "practice-thread-answer";
  aText.textContent = answer;
  const aBlock = document.createElement("div");
  aBlock.className = "practice-thread-block";
  aBlock.appendChild(aLabel);
  aBlock.appendChild(aText);

  wrapper.appendChild(qBlock);
  wrapper.appendChild(aBlock);
  items.appendChild(wrapper);
}

function setPracticePrompt(question) {
  const prompt = document.getElementById("practicePrompt");
  const promptText = document.getElementById("practicePromptText");
  if (!prompt || !promptText) return;
  prompt.classList.remove("hidden");
  promptText.textContent = String(question || "").trim();
}

function stopPracticeInterview() {
  if (!state.practiceThread) return;
  state.practiceThread.stopped = true;
  state.practiceThread.nextQuestion = null;
  setPracticePrompt(currentQuestion());
  showToast("Interview stopped. Submit again to start a new flow.");
}

function speakText(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return;
  if (!("speechSynthesis" in window) || typeof window.SpeechSynthesisUtterance !== "function") {
    return;
  }

  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(trimmed);
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  } catch (_e) {
    // Ignore browser speech failures.
  }
}

function isRlModeEnabled() {
  const el = document.getElementById("useRlMode");
  return Boolean(el && el.checked);
}

function setRlStatus(text) {
  const status = document.getElementById("rlStatus");
  if (status) status.textContent = text;
}

function resetRlSummary() {
  const summary = document.getElementById("rlSummary");
  if (summary) summary.classList.add("hidden");
  document.getElementById("rlAttempt").textContent = "-";
  document.getElementById("rlReward").textContent = "-";
  document.getElementById("rlAvgGrade").textContent = "-";
  document.getElementById("rlStrategy").textContent = "-";
  document.getElementById("rlSessionState").textContent = "No active RL session.";
}

function updateRlSummary(result) {
  const summary = document.getElementById("rlSummary");
  if (summary) summary.classList.remove("hidden");

  const attempt = result?.session_progress?.attempt;
  const totalReward = result?.session_progress?.total_reward;
  const avgGrade = result?.session_progress?.avg_grade;
  const strategy = result?.rl_strategy;

  document.getElementById("rlAttempt").textContent = attempt ?? "-";
  document.getElementById("rlReward").textContent =
    typeof totalReward === "number" ? Math.min(Math.max(totalReward, 0), 1).toFixed(2) : "-";
  document.getElementById("rlAvgGrade").textContent =
    typeof avgGrade === "number" ? avgGrade.toFixed(2) : "-";
  document.getElementById("rlStrategy").textContent = strategy || "-";
  document.getElementById("rlSessionState").textContent = result?.episode_done
    ? "Episode completed. Next submit starts a new episode."
    : `Episode running${attempt ? ` (attempt ${attempt})` : ""}.`;
}

function setupPasswordVisibility() {
  document.querySelectorAll(".password-toggle").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = btn.getAttribute("data-toggle-password");
      const input = document.getElementById(targetId);
      if (!input) return;
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
    });
  });
}

function updatePostureUi(status, feedback) {
  const safeStatus = status || "Not analyzed";
  const safeFeedback = feedback || "No posture analysis yet";

  const statusEl = document.getElementById("postureStatus");
  const feedbackEl = document.getElementById("postureFeedback");
  if (statusEl) statusEl.textContent = `Posture: ${safeStatus}`;
  if (feedbackEl) feedbackEl.textContent = `Feedback: ${safeFeedback}`;

  const overlay = document.getElementById("postureOverlay");
  const overlayStatus = document.getElementById("postureOverlayStatus");
  const overlayFeedback = document.getElementById("postureOverlayFeedback");
  if (!overlay || !overlayStatus || !overlayFeedback) return;

  overlayStatus.textContent = `Posture: ${safeStatus}`;
  overlayFeedback.textContent = safeFeedback;

  overlay.classList.remove("posture-overlay-good", "posture-overlay-warning", "posture-overlay-neutral");
  if (safeStatus === "Good") {
    overlay.classList.add("posture-overlay-good");
  } else if (safeStatus === "Needs Improvement" || safeStatus === "Not Visible") {
    overlay.classList.add("posture-overlay-warning");
  } else {
    overlay.classList.add("posture-overlay-neutral");
  }
}

async function ensureRlSession() {
  if (!isRlModeEnabled()) return;
  if (state.rlSessionStarted) return;

  const difficulty = document.getElementById("rlDifficulty")?.value || "medium";
  const fixWeakness = Boolean(document.getElementById("fixWeaknessMode")?.checked);
  await api("/api/rl/new-session", {
    method: "POST",
    body: {
      difficulty,
      training_mode: fixWeakness ? "fix_weakness" : "normal",
      target_skill: "auto",
    }
  });
  state.rlSessionStarted = true;
  setRlStatus(`RL session active (${difficulty})`);
}

function buildRlFeedback(result) {
  const attempt = result?.session_progress?.attempt ?? "-";
  const reward = typeof result?.reward === "number" ? Math.min(Math.max(result.reward, 0), 1).toFixed(2) : "-";
  const strategy = result?.rl_strategy || "none";
  const rlFeedback = result?.rl_feedback || "";
  return `${result.feedback}\n\nRL Strategy: ${strategy}\nRL Feedback: ${rlFeedback}\nAttempt: ${attempt}\nReward: ${reward}`;
}

function formatSecondsToClock(totalSeconds) {
  const mins = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const secs = String(totalSeconds % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

function startSpeechTimer(timerElementId) {
  if (!timerElementId) return;
  const timerEl = document.getElementById(timerElementId);
  if (!timerEl) return;

  stopSpeechTimer(timerElementId);
  const startedAt = Date.now();
  timerEl.textContent = "Recording Time: 00:00";

  state.speechTimers[timerElementId] = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
    timerEl.textContent = `Recording Time: ${formatSecondsToClock(elapsedSeconds)}`;
  }, 1000);
}

function stopSpeechTimer(timerElementId) {
  const timerId = state.speechTimers[timerElementId];
  if (timerId) {
    clearInterval(timerId);
    delete state.speechTimers[timerElementId];
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function apiUpload(path, formData, options = {}) {
  const res = await fetch(path, {
    method: "POST",
    body: formData,
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function setMessage(id, text) {
  document.getElementById(id).textContent = text || "";
}

function showToast(text, type = "success") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = text;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 2800);
}

function setLoading(active, text = "Processing...") {
  const overlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  if (!overlay || !loadingText) return;

  loadingText.textContent = text;
  overlay.classList.toggle("hidden", !active);
}

function animateCounter(elementId, targetValue, decimals = 0, durationMs = 700) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const rawCurrent = Number.parseFloat(el.textContent);
  const startValue = Number.isFinite(rawCurrent) ? rawCurrent : 0;
  const startTime = performance.now();

  const tick = (now) => {
    const progress = Math.min((now - startTime) / durationMs, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = startValue + (targetValue - startValue) * eased;
    el.textContent = value.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
}

function initParallaxBackground() {
  const a = document.querySelector(".bg-shape-a");
  const b = document.querySelector(".bg-shape-b");
  if (!a || !b) return;

  window.addEventListener("mousemove", (event) => {
    const x = (event.clientX / window.innerWidth - 0.5) * 16;
    const y = (event.clientY / window.innerHeight - 0.5) * 16;
    a.style.transform = `translate(${x}px, ${y}px)`;
    b.style.transform = `translate(${-x}px, ${-y}px)`;
  });
}

function initCardTilt() {
  const cards = document.querySelectorAll(".about-card");
  cards.forEach((card) => {
    card.addEventListener("mousemove", (event) => {
      const rect = card.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width;
      const py = (event.clientY - rect.top) / rect.height;
      const rx = (0.5 - py) * 5;
      const ry = (px - 0.5) * 5;
      card.style.transform = `perspective(700px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-2px)`;
    });

    card.addEventListener("mouseleave", () => {
      card.style.transform = "translateY(0)";
    });
  });
}

function switchAuthMode(mode) {
  const loginPanel = document.getElementById("loginPanel");
  const signupPanel = document.getElementById("signupPanel");
  const loginBtn = document.getElementById("showLoginBtn");
  const signupBtn = document.getElementById("showSignupBtn");

  const showLogin = mode === "login";
  loginPanel.classList.toggle("hidden", !showLogin);
  signupPanel.classList.toggle("hidden", showLogin);
  loginBtn.classList.toggle("active", showLogin);
  signupBtn.classList.toggle("active", !showLogin);
  setMessage("authMessage", "");
}

function currentQuestion() {
  const select = document.getElementById("questionSelect");
  const custom = document.getElementById("customQuestion");
  if (select.value === "Add custom question...") {
    return custom.value.trim();
  }
  return select.value;
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
      if (tab.dataset.tab === "progress") loadProgress();
      if (tab.dataset.tab === "reports") loadReports();
      if (tab.dataset.tab === "practice") {
        document.getElementById("analysisResult").textContent = "";
      }
    });
  });
}

function renderAtsResults(result) {
  const resultsEl = document.getElementById("atsResults");
  const groupsEl = document.getElementById("atsGroups");
  const scoreEl = document.getElementById("atsScore");
  const issuesEl = document.getElementById("atsIssues");
  if (!resultsEl || !groupsEl || !scoreEl || !issuesEl) return;

  const score = Number.isFinite(result?.score) ? Math.max(0, Math.min(100, result.score)) : 0;
  const issues = Number.isFinite(result?.issues) ? Math.max(0, result.issues) : 0;
  scoreEl.textContent = String(score);
  issuesEl.textContent = String(issues);

  groupsEl.innerHTML = "";
  const groups = Array.isArray(result?.groups) ? result.groups : [];

  groups.forEach((group, idx) => {
    const details = document.createElement("details");
    details.className = "ats-group";
    details.open = idx === 0;

    const summary = document.createElement("summary");
    summary.className = "ats-group-summary";

    const title = document.createElement("span");
    title.className = "ats-group-title";
    title.textContent = group?.label || "";

    const badge = document.createElement("span");
    badge.className = "ats-badge";
    const percent = Number.isFinite(group?.percent) ? Math.max(0, Math.min(100, group.percent)) : 0;
    badge.textContent = `${percent}%`;

    summary.appendChild(title);
    summary.appendChild(badge);
    details.appendChild(summary);

    const itemsWrap = document.createElement("div");
    itemsWrap.className = "ats-items";

    const items = Array.isArray(group?.items) ? group.items : [];
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "ats-item";

      const icon = document.createElement("span");
      const hasIssue = (item?.issues || 0) > 0 || item?.status === "issue";
      icon.className = `ats-icon ${hasIssue ? "issue" : "ok"}`;
      icon.textContent = hasIssue ? "✕" : "✓";

      const label = document.createElement("span");
      label.className = "ats-item-label";
      label.textContent = item?.label || "";

      const pill = document.createElement("span");
      pill.className = `ats-pill ${hasIssue ? "issue" : "ok"}`;
      const issueCount = Number.isFinite(item?.issues) ? Math.max(0, item.issues) : 0;
      if (!hasIssue) {
        pill.textContent = "No issues";
      } else {
        pill.textContent = `${issueCount || 1} issue${issueCount === 1 ? "" : "s"}`;
      }

      row.appendChild(icon);
      row.appendChild(label);
      row.appendChild(pill);

      const hasHelp = hasIssue && ((item?.what || "").trim() || (item?.how || "").trim());
      if (hasHelp) {
        const help = document.createElement("div");
        help.className = "ats-item-help";
        const what = document.createElement("div");
        what.className = "ats-item-help-line";
        what.textContent = `What's wrong: ${item.what}`;
        const how = document.createElement("div");
        how.className = "ats-item-help-line";
        how.textContent = `How to fix: ${item.how}`;
        help.appendChild(what);
        help.appendChild(how);

        const block = document.createElement("div");
        block.className = "ats-item-block";
        block.appendChild(row);
        block.appendChild(help);
        itemsWrap.appendChild(block);
        return;
      }

      itemsWrap.appendChild(row);
    });

    details.appendChild(itemsWrap);
    groupsEl.appendChild(details);
  });

  resultsEl.classList.remove("hidden");
}

function initAtsChecker() {
  const fileInput = document.getElementById("atsResumeFile");
  const btn = document.getElementById("atsCheckBtn");
  const msg = document.getElementById("atsMessage");
  const resultsEl = document.getElementById("atsResults");

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (msg) msg.textContent = "";
      if (resultsEl) resultsEl.classList.add("hidden");
    });
  }

  if (!btn) return;
  btn.addEventListener("click", async () => {
    try {
      if (msg) msg.textContent = "";
      const file = fileInput?.files?.[0];
      if (!file) {
        showToast("Please upload a resume file.", "error");
        return;
      }

      const fd = new FormData();
      fd.append("resume", file);

      setLoading(true, "Checking ATS score...");
      const result = await apiUpload("/api/ats/check", fd);
      renderAtsResults(result);
      showToast("ATS check complete", "success");
    } catch (e) {
      if (msg) msg.textContent = e.message || "ATS check failed";
      showToast(e.message || "ATS check failed", "error");
    } finally {
      setLoading(false);
    }
  });
}

function togglePracticeMode() {
  const mode = document.getElementById("practiceMode").value;
  document.getElementById("audioPractice").classList.toggle("hidden", mode !== "Record Audio Response");
  document.getElementById("textPractice").classList.toggle("hidden", mode !== "Text Input Response");
  document.getElementById("videoPractice").classList.toggle("hidden", mode !== "Video Practice");
}

function initSpeech(targetElementId, timerElementId) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    alert("Speech recognition is not supported in this browser.");
    return null;
  }

  const recognizer = new SR();
  recognizer.continuous = true;
  recognizer.interimResults = true;
  recognizer.lang = "en-US";

  let finalizedTranscript = "";

  recognizer.onstart = () => {
    startSpeechTimer(timerElementId);
  };

  recognizer.onresult = (event) => {
    let interimTranscript = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const chunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalizedTranscript += `${chunk} `;
      } else {
        interimTranscript += chunk;
      }
    }

    const textBox = document.getElementById(targetElementId);
    textBox.value = `${finalizedTranscript}${interimTranscript}`.trim();
  };

  recognizer.onend = () => {
    stopSpeechTimer(timerElementId);
  };

  recognizer.onerror = () => {
    stopSpeechTimer(timerElementId);
    showToast("Speech recognition error occurred.", "error");
  };

  return recognizer;
}

function safeStartRecognition(recognizer) {
  if (!recognizer) return;
  try {
    recognizer.start();
  } catch (_err) {
    // Ignore duplicate start calls.
  }
}

async function doLogin() {
  try {
    const username = document.getElementById("loginUsername").value.trim();
    const password = document.getElementById("loginPassword").value;
    setLoading(true, "Logging in...");
    await api("/api/login", { method: "POST", body: { username, password } });
    setMessage("authMessage", "Logged in successfully");
    setLoading(false);
    await loginFlow();
  } catch (e) {
    setLoading(false);
    setMessage("authMessage", e.message);
    showToast(e.message, "error");
  }
}

async function doSignup() {
  try {
    const full_name = document.getElementById("signupFullName")?.value?.trim();
    const email = document.getElementById("signupEmail")?.value?.trim();
    const phone = document.getElementById("signupPhone")?.value?.trim();
    const university = document.getElementById("signupUniversity")?.value?.trim();
    const college_year = document.getElementById("signupCollegeYear")?.value?.trim();
    const degree = document.getElementById("signupDegree")?.value?.trim();
    const major = document.getElementById("signupMajor")?.value?.trim();
    const linkedin = document.getElementById("signupLinkedIn")?.value?.trim();
    const about = document.getElementById("signupAbout")?.value?.trim();
    const otp = document.getElementById("signupOtp")?.value?.trim();

    const username = document.getElementById("signupUsername").value.trim();
    const password = document.getElementById("signupPassword").value;
    const confirm_password = document.getElementById("signupConfirmPassword").value;
    setLoading(true, "Creating account...");
    await api("/api/signup", {
      method: "POST",
      body: {
        username,
        password,
        confirm_password,
        full_name,
        email,
        phone,
        university,
        college_year,
        degree,
        major,
        linkedin,
        about,
        otp,
      },
    });
    setLoading(false);
    setMessage("authMessage", "Account created. Please login.");
    switchAuthMode("login");
    showToast("Account created. Please login.", "success");
  } catch (e) {
    setLoading(false);
    setMessage("authMessage", e.message);
    showToast(e.message, "error");
  }
}

async function sendEmailOtp() {
  try {
    const email = document.getElementById("signupEmail")?.value?.trim();
    if (!email) {
      showToast("Please enter your email first.", "error");
      return;
    }
    setLoading(true, "Sending OTP...");
    const result = await api("/api/auth/send-email-otp", { method: "POST", body: { email } });
    setLoading(false);
    setMessage("authMessage", result.message || "OTP sent. Check your inbox.");
    showToast(result.message || "OTP sent", "success");
  } catch (e) {
    setLoading(false);
    setMessage("authMessage", e.message);
    showToast(e.message, "error");
  }
}

async function loginFlow() {
  // Always lock the app view first, then unlock only for authenticated users.
  document.getElementById("authSection").classList.remove("hidden");
  document.getElementById("appSection").classList.add("hidden");
  document.getElementById("logoutBtn").classList.add("hidden");

  // Reset account view while we check auth.
  state.username = null;
  state.profile = null;
  resetAccountProfileForm();

  const me = await api("/api/me");
  if (!me.logged_in) {
    return;
  }

  state.username = me.username;
  state.profile = me.profile && typeof me.profile === "object" ? me.profile : null;
  document.getElementById("authSection").classList.add("hidden");
  document.getElementById("appSection").classList.remove("hidden");
  document.getElementById("logoutBtn").classList.remove("hidden");
  document.getElementById("accountUsername").textContent = `Username: ${state.username}`;
  populateAccountProfileForm(state.profile);

  await loadMeta();
  await loadReports();
  await loadProgress();
  showToast(`Welcome, ${state.username}`);
  if (state.profile && state.profile.gamification) {
    updateGamificationUi(state.profile.gamification);
  }
}

function resetAccountProfileForm() {
  const ids = [
    "accountFullName",
    "accountEmail",
    "accountPhone",
    "accountUniversity",
    "accountCollegeYear",
    "accountDegree",
    "accountMajor",
    "accountLinkedIn",
    "accountAbout",
  ];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
}

function updateGamificationUi(gamification, deltaXp = 0, eventSourceElement = null) {
  if (!gamification) return;
  
  const bar = document.getElementById("gamificationBar");
  if (bar) bar.classList.remove("hidden");
  
  const levelEl = document.getElementById("gamiLevel");
  const xpEl = document.getElementById("gamiXp");
  const streakEl = document.getElementById("gamiStreak");
  const progressFill = document.getElementById("gamiProgressFill");
  const progressText = document.getElementById("gamiProgressText");
  const badgesContainer = document.getElementById("gamiBadgesContainer");
  
  if (levelEl) levelEl.textContent = gamification.level || 1;
  if (xpEl) xpEl.textContent = gamification.total_xp || gamification.xp || 0;
  if (streakEl) streakEl.textContent = gamification.streak || 0;
  
  const levelXp = (gamification.total_xp || gamification.xp || 0) % 100;
  if (progressFill) progressFill.style.width = `${levelXp}%`;
  if (progressText) progressText.textContent = `${levelXp}%`;
  
  if (badgesContainer) {
    const badges = gamification.badges || [];
    if (badges.length > 0) {
      badgesContainer.innerHTML = badges.map(b => `<span class="gami-badge">${b}</span>`).join("");
    } else {
      badgesContainer.innerHTML = `<span class="gami-badge empty">No badges yet</span>`;
    }
  }
  
  if (deltaXp > 0 && eventSourceElement) {
    const floatEl = document.createElement("div");
    floatEl.className = "floating-xp";
    floatEl.textContent = `+${deltaXp} XP!`;
    const rect = eventSourceElement.getBoundingClientRect();
    floatEl.style.left = `${rect.left + rect.width / 2}px`;
    floatEl.style.top = `${rect.top + window.scrollY}px`;
    document.body.appendChild(floatEl);
    setTimeout(() => { floatEl.remove(); }, 1500);
  }
}

function populateAccountProfileForm(profile) {
  if (!profile || typeof profile !== "object") {
    resetAccountProfileForm();
    return;
  }

  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value == null ? "" : String(value);
  };

  setValue("accountFullName", profile.full_name);
  setValue("accountEmail", profile.email);
  setValue("accountPhone", profile.phone);
  setValue("accountUniversity", profile.university);
  setValue("accountCollegeYear", profile.college_year);
  setValue("accountDegree", profile.degree);
  setValue("accountMajor", profile.major);
  setValue("accountLinkedIn", profile.linkedin);
  setValue("accountAbout", profile.about);
}

function readAccountProfileForm() {
  const get = (id) => document.getElementById(id)?.value?.trim() || "";
  return {
    full_name: get("accountFullName"),
    email: get("accountEmail"),
    phone: get("accountPhone"),
    university: get("accountUniversity"),
    college_year: get("accountCollegeYear"),
    degree: get("accountDegree"),
    major: get("accountMajor"),
    linkedin: get("accountLinkedIn"),
    about: get("accountAbout"),
  };
}

async function saveAccountProfile() {
  setLoading(true, "Saving profile...");
  try {
    const payload = readAccountProfileForm();
    const result = await api("/api/profile", { method: "POST", body: payload });
    state.profile = result.profile && typeof result.profile === "object" ? result.profile : payload;
    populateAccountProfileForm(state.profile);
    showToast("Profile updated", "success");
  } catch (e) {
    showToast(e.message || "Unable to save profile", "error");
    throw e;
  } finally {
    setLoading(false);
  }
}

async function loadMeta() {
  const meta = await api("/api/meta");
  state.questions = [...meta.questions, "Add custom question..."];

  const select = document.getElementById("questionSelect");
  select.innerHTML = "";
  state.questions.forEach((q) => {
    const option = document.createElement("option");
    option.value = q;
    option.textContent = q;
    select.appendChild(option);
  });

  // Keep the practice prompt synced with the selected question.
  setPracticePrompt(currentQuestion());
}

async function loadReports() {
  const data = await api("/api/reports");
  state.reports = data.reports || [];

  const reportsList = document.getElementById("reportsList");
  reportsList.innerHTML = "";
  if (!state.reports.length) {
    reportsList.textContent = "No reports yet.";
  } else {
    state.reports.forEach((r, idx) => {
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `<strong>Report ${idx + 1}</strong><br>${r.timestamp}<br><em>${r.question}</em><br>Confidence: ${r.analysis.confidence.confidence_score.toFixed(1)} / 10`;
      reportsList.appendChild(div);
    });
  }

  const sessions = state.reports.length;
  const avgConfidence = sessions
    ? state.reports.reduce((acc, r) => acc + r.analysis.confidence.confidence_score, 0) / sessions
    : 0;
  animateCounter("totalSessions", sessions, 0);
  animateCounter("avgConfidence", avgConfidence, 1);
  document.getElementById("userStatus").textContent = sessions ? "Active" : "New";
  document.getElementById("accountSessions").textContent = `Total Sessions: ${sessions}`;
}

async function loadProgress() {
  const data = await api("/api/progress");
  const timeline = data.timeline || [];
  const labels = timeline.map((_, idx) => `Session ${idx + 1}`);
  const confidence = timeline.map((r) => r.confidence);
  const tone = timeline.map((r) => r.tone);

  const sessions = timeline.length;
  const avgConfidence = sessions
    ? confidence.reduce((a, b) => a + b, 0) / sessions
    : 0;
  const avgTone = sessions ? tone.reduce((a, b) => a + b, 0) / sessions : 0;
  const bestConfidence = sessions ? Math.max(...confidence) : 0;

  document.getElementById("progressSessions").textContent = String(sessions);
  document.getElementById("progressAvgConfidence").textContent = avgConfidence.toFixed(1);
  document.getElementById("progressAvgTone").textContent = avgTone.toFixed(2);
  document.getElementById("progressBestConfidence").textContent = bestConfidence.toFixed(1);

  const insight = document.getElementById("progressInsight");
  if (!sessions) {
    insight.textContent = "No sessions yet. Complete a practice round to unlock charts and trend insights.";
  } else {
    const trend = confidence[sessions - 1] - confidence[0];
    if (trend > 0.3) {
      insight.textContent = `Confidence trend is improving (+${trend.toFixed(1)}). Keep up the practice momentum.`;
    } else if (trend < -0.3) {
      insight.textContent = `Confidence trend dipped (${trend.toFixed(1)}). Revisit recent feedback and practice again.`;
    } else {
      insight.textContent = "Confidence trend is stable. Try varying questions to continue improving.";
    }
  }

  if (state.confidenceChart) state.confidenceChart.destroy();
  if (state.toneChart) state.toneChart.destroy();

  state.confidenceChart = new Chart(document.getElementById("confidenceChart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Confidence",
        data: confidence,
        borderColor: "#6f7dff",
        backgroundColor: "rgba(111, 125, 255, 0.2)",
        fill: true,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#ddd6ff" } } },
      scales: {
        x: { ticks: { color: "#b9b2df" }, grid: { color: "rgba(133, 118, 198, 0.2)" } },
        y: { min: 0, max: 10, ticks: { color: "#b9b2df" }, grid: { color: "rgba(133, 118, 198, 0.2)" } }
      }
    }
  });

  state.toneChart = new Chart(document.getElementById("toneChart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Tone",
        data: tone,
        borderColor: "#ff4fd8",
        backgroundColor: "rgba(255, 79, 216, 0.18)",
        fill: true,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#ddd6ff" } } },
      scales: {
        x: { ticks: { color: "#b9b2df" }, grid: { color: "rgba(133, 118, 198, 0.2)" } },
        y: { min: -1, max: 1, ticks: { color: "#b9b2df" }, grid: { color: "rgba(133, 118, 198, 0.2)" } }
      }
    }
  });

  const summary = await refreshCoachSummary();
  if (summary) {
    applyCoachSettingsToUi(summary);
    renderCoachSummaryToProgress(summary);
  }
}

async function submitTextPractice() {
  const question = getActivePracticeQuestion();
  const answer = document.getElementById("textAnswer").value.trim();
  if (!question || !answer) {
    showToast("Question and answer are required", "error");
    return;
  }

  const rootQuestion = state.practiceThread?.rootQuestion || currentQuestion();
  ensurePracticeThread(rootQuestion);
  if (state.practiceThread.stopped) {
    resetPracticeThread();
    ensurePracticeThread(currentQuestion());
  }
  const turns = [...(state.practiceThread?.turns || []), { question, answer }];

  setLoading(true, "Analyzing text response...");
  let result;
  if (isRlModeEnabled()) {
    await ensureRlSession();
    result = await api("/api/rl/practice/text", {
      method: "POST",
      body: {
        question,
        answer,
        root_question: state.practiceThread.rootQuestion,
        thread_turns: turns,
        task_difficulty: document.getElementById("rlDifficulty")?.value || "medium",
        use_agent_feedback: true
      }
    });
    if (result.episode_done) {
      state.rlSessionStarted = false;
      setRlStatus("RL episode completed. Next submit starts a new episode.");
    } else {
      setRlStatus(`RL active: attempt ${result?.session_progress?.attempt || 1}`);
    }
    updateRlSummary(result);
  } else {
    result = await api("/api/practice/text", {
      method: "POST",
      body: { question, answer, root_question: state.practiceThread.rootQuestion, thread_turns: turns }
    });
  }
  setLoading(false);

  if (result && result.gamification) {
    const btn = document.getElementById("submitTextBtn") || document.body;
    updateGamificationUi(result.gamification, result.gamification.earned_xp, btn);
  }

  state.practiceThread.turns = turns;
  appendPracticeTurn(question, answer);
  state.practiceThread.nextQuestion = result.follow_up_question || null;
  setPracticePrompt(state.practiceThread.nextQuestion || state.practiceThread.rootQuestion);
  speakText(state.practiceThread.nextQuestion);

  document.getElementById("analysisResult").textContent = isRlModeEnabled()
    ? buildRlFeedback(result)
    : result.feedback;

  renderAgentBrain(result);
  document.getElementById("textAnswer").value = "";
  showToast("Text response analyzed", "success");
  await loadReports();
  await loadProgress();
}

async function submitAudioPractice() {
  const question = getActivePracticeQuestion();
  const transcription = document.getElementById("audioTranscript").value.trim();
  if (!question || !transcription) {
    showToast("Question and transcription are required", "error");
    return;
  }

  const rootQuestion = state.practiceThread?.rootQuestion || currentQuestion();
  ensurePracticeThread(rootQuestion);
  if (state.practiceThread.stopped) {
    resetPracticeThread();
    ensurePracticeThread(currentQuestion());
  }
  const turns = [...(state.practiceThread?.turns || []), { question, answer: transcription }];

  setLoading(true, "Analyzing audio response...");
  let result;
  if (isRlModeEnabled()) {
    await ensureRlSession();
    result = await api("/api/rl/practice/text", {
      method: "POST",
      body: {
        question,
        answer: transcription,
        root_question: state.practiceThread.rootQuestion,
        thread_turns: turns,
        task_difficulty: document.getElementById("rlDifficulty")?.value || "medium",
        use_agent_feedback: true
      }
    });
    if (result.episode_done) {
      state.rlSessionStarted = false;
      setRlStatus("RL episode completed. Next submit starts a new episode.");
    } else {
      setRlStatus(`RL active: attempt ${result?.session_progress?.attempt || 1}`);
    }
    updateRlSummary(result);
  } else {
    result = await api("/api/practice/audio", {
      method: "POST",
      body: {
        question,
        transcription,
        root_question: state.practiceThread.rootQuestion,
        thread_turns: turns
      }
    });
  }
  setLoading(false);

  if (result && result.gamification) {
    const btn = document.getElementById("submitAudioBtn") || document.body;
    updateGamificationUi(result.gamification, result.gamification.earned_xp, btn);
  }

  state.practiceThread.turns = turns;
  appendPracticeTurn(question, transcription);
  state.practiceThread.nextQuestion = result.follow_up_question || null;
  setPracticePrompt(state.practiceThread.nextQuestion || state.practiceThread.rootQuestion);
  speakText(state.practiceThread.nextQuestion);

  document.getElementById("analysisResult").textContent = isRlModeEnabled()
    ? buildRlFeedback(result)
    : result.feedback;

  renderAgentBrain(result);
  document.getElementById("audioTranscript").value = "";
  showToast("Audio response analyzed", "success");
  await loadReports();
  await loadProgress();
}

function frameFromVideo(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.8);
}

async function startCamera() {
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  const video = document.getElementById("videoPreview");
  video.srcObject = state.stream;
  updatePostureUi(state.posture.status, state.posture.feedback);
  showToast("Camera started", "success");

  const loop = async () => {
    if (!state.stream) return;
    try {
      const image = frameFromVideo(video);
      const result = await api("/api/posture/frame", {
        method: "POST",
        body: { image }
      });
      state.posture = { status: result.status, feedback: result.feedback };
      updatePostureUi(result.status, result.feedback);
    } catch (_e) {
      // Ignore transient frame errors.
    }
    setTimeout(loop, 2000);
  };

  loop();
}

function stopCamera() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  const video = document.getElementById("videoPreview");
  video.srcObject = null;
  state.posture = { status: "Not analyzed", feedback: "No posture analysis yet" };
  updatePostureUi(state.posture.status, state.posture.feedback);
  showToast("Camera stopped");
}

async function submitVideoPractice() {
  const question = getActivePracticeQuestion();
  const transcription = document.getElementById("videoTranscript").value.trim();
  if (!question || !transcription) {
    showToast("Question and transcription are required", "error");
    return;
  }

  const rootQuestion = state.practiceThread?.rootQuestion || currentQuestion();
  ensurePracticeThread(rootQuestion);
  if (state.practiceThread.stopped) {
    resetPracticeThread();
    ensurePracticeThread(currentQuestion());
  }
  const turns = [...(state.practiceThread?.turns || []), { question, answer: transcription }];

  setLoading(true, "Submitting video practice...");
  let result;
  if (isRlModeEnabled()) {
    await ensureRlSession();
    result = await api("/api/rl/practice/text", {
      method: "POST",
      body: {
        question,
        answer: transcription,
        root_question: state.practiceThread.rootQuestion,
        thread_turns: turns,
        task_difficulty: document.getElementById("rlDifficulty")?.value || "medium",
        use_agent_feedback: true
      }
    });
    if (result.episode_done) {
      state.rlSessionStarted = false;
      setRlStatus("RL episode completed. Next submit starts a new episode.");
    } else {
      setRlStatus(`RL active: attempt ${result?.session_progress?.attempt || 1}`);
    }
    updateRlSummary(result);
  } else {
    result = await api("/api/practice/video", {
      method: "POST",
      body: {
        question,
        transcription,
        posture: state.posture,
        root_question: state.practiceThread.rootQuestion,
        thread_turns: turns
      }
    });
  }
  setLoading(false);

  if (result && result.gamification) {
    const btn = document.getElementById("submitVideoBtn") || document.body;
    updateGamificationUi(result.gamification, result.gamification.earned_xp, btn);
  }

  state.practiceThread.turns = turns;
  appendPracticeTurn(question, transcription);
  state.practiceThread.nextQuestion = result.follow_up_question || null;
  setPracticePrompt(state.practiceThread.nextQuestion || state.practiceThread.rootQuestion);
  speakText(state.practiceThread.nextQuestion);

  document.getElementById("analysisResult").textContent = isRlModeEnabled()
    ? buildRlFeedback(result)
    : result.feedback;

  renderAgentBrain(result);
  document.getElementById("videoTranscript").value = "";
  showToast("Video practice submitted", "success");
  await loadReports();
  await loadProgress();
}

async function generateMockInterview() {
  setMessage("mockMessage", "Generating mock interview video. This can take 1-2 minutes...");
  setLoading(true, "Generating mock interview video...");

  try {
    const result = await api("/api/mock-interview", {
      method: "POST",
      body: {
        role: document.getElementById("mockRole").value.trim(),
        experience: document.getElementById("mockExperience").value,
        interview_type: document.getElementById("mockType").value,
        additional_details: document.getElementById("mockDetails").value.trim()
      }
    });

    document.getElementById("mockTranscript").textContent = result.transcript || "";
    const link = document.getElementById("downloadMockLink");
    const player = document.getElementById("mockVideoPlayer");

    if (result.video_available && result.video_id) {
      setMessage("mockMessage", "Video generated. Use the link below to download.");
      link.href = `/api/mock-interview/${result.video_id}`;
      link.textContent = "Download Generated Video";
      link.classList.remove("hidden");
      player.src = `/api/mock-interview/${result.video_id}`;
      player.classList.remove("hidden");
      showToast("Mock interview video generated", "success");
    } else {
      setMessage(
        "mockMessage",
        result.warning || "Transcript generated, but video is unavailable on this machine."
      );
      link.classList.add("hidden");
      player.removeAttribute("src");
      player.classList.add("hidden");
      showToast("Transcript generated (video unavailable)", "error");
    }

    if (result.warning) {
      showToast(result.warning, "error");
    }
  } catch (e) {
    setMessage("mockMessage", e.message || "Failed to generate mock interview");
    showToast(e.message || "Failed to generate mock interview", "error");
  } finally {
    setLoading(false);
  }
}

function setupEvents() {
  initTabs();
  initAtsChecker();

  document.getElementById("loginBtn").addEventListener("click", doLogin);
  document.getElementById("signupBtn").addEventListener("click", doSignup);

  ["loginUsername", "loginPassword"].forEach((id) => {
    document.getElementById(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        doLogin();
      }
    });
  });

  ["signupUsername", "signupPassword", "signupConfirmPassword"].forEach((id) => {
    document.getElementById(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        doSignup();
      }
    });
  });

  document.getElementById("showLoginBtn").addEventListener("click", () => switchAuthMode("login"));
  document.getElementById("showSignupBtn").addEventListener("click", () => switchAuthMode("signup"));
  document.getElementById("toSignupLink").addEventListener("click", () => switchAuthMode("signup"));
  document.getElementById("toLoginLink").addEventListener("click", () => switchAuthMode("login"));

  const sendOtpBtn = document.getElementById("sendEmailOtpBtn");
  if (sendOtpBtn) {
    sendOtpBtn.addEventListener("click", () => {
      sendEmailOtp().catch(() => undefined);
    });
  }

  const sendEmailLinkBtn = document.getElementById("sendEmailLinkBtn");
  if (sendEmailLinkBtn) {
    sendEmailLinkBtn.addEventListener("click", () => {
      sendFirebaseEmailSignInLink().catch(() => undefined);
    });
  }

  const completeEmailLinkBtn = document.getElementById("completeEmailLinkBtn");
  if (completeEmailLinkBtn) {
    completeEmailLinkBtn.addEventListener("click", () => {
      completeFirebaseEmailLinkSignIn().catch(() => undefined);
    });
  }

  document.getElementById("logoutBtn").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    stopCamera();
    await loginFlow();
    showToast("Logged out");
  });

  const saveAccountBtn = document.getElementById("saveAccountBtn");
  if (saveAccountBtn) {
    saveAccountBtn.addEventListener("click", () => {
      saveAccountProfile().catch(() => undefined);
    });
  }

  document.getElementById("practiceMode").addEventListener("change", togglePracticeMode);
  document.getElementById("practiceMode").addEventListener("change", () => {
    resetPracticeThread();
    document.getElementById("analysisResult").textContent = "";
  });
  document.getElementById("useRlMode").addEventListener("change", async (event) => {
    if (event.target.checked) {
      state.rlSessionStarted = false;
      setRlStatus("RL mode enabled. Session starts on your next submit.");
      resetRlSummary();
      return;
    }
    state.rlSessionStarted = false;
    setRlStatus("RL mode is off.");
    resetRlSummary();
  });

  document.getElementById("rlDifficulty").addEventListener("change", () => {
    state.rlSessionStarted = false;
    if (isRlModeEnabled()) {
      setRlStatus("Difficulty changed. Next submit starts a new RL session.");
      resetRlSummary();
    }
  });

  const personality = document.getElementById("coachPersonality");
  if (personality) {
    personality.addEventListener("change", async () => {
      try {
        const adaptiveCheckbox = document.getElementById("adaptivePersonality");
        let adaptiveVal = false;
        if (adaptiveCheckbox && adaptiveCheckbox.checked) {
          adaptiveCheckbox.checked = false;
        }
        await api("/api/coach/settings", {
          method: "POST",
          body: { 
            coach_personality: personality.value,
            adaptive_personality: false 
          }
        });
        state.rlSessionStarted = false;
        resetRlSummary();
        showToast("Coach personality updated", "success");
        await loadProgress();
      } catch (e) {
        showToast(e.message || "Failed to update", "error");
      }
    });
  }

  const adaptive = document.getElementById("adaptivePersonality");
  if (adaptive) {
    adaptive.addEventListener("change", async () => {
      try {
        if (personality) {
          personality.disabled = adaptive.checked;
        }
        await api("/api/coach/settings", {
          method: "POST",
          body: { adaptive_personality: Boolean(adaptive.checked) }
        });
        state.rlSessionStarted = false;
        resetRlSummary();
        showToast("Adaptive personality updated", "success");
        await loadProgress();
      } catch (e) {
        showToast(e.message || "Failed to update", "error");
      }
    });
  }

  const fixWeakness = document.getElementById("fixWeaknessMode");
  if (fixWeakness) {
    fixWeakness.addEventListener("change", async () => {
      try {
        await api("/api/coach/settings", {
          method: "POST",
          body: { training_mode: fixWeakness.checked ? "fix_weakness" : "normal", target_skill: "auto" }
        });
        state.rlSessionStarted = false;
        resetRlSummary();
        showToast("Training mode updated", "success");
        await loadProgress();
      } catch (e) {
        showToast(e.message || "Failed to update", "error");
      }
    });
  }

  document.getElementById("questionSelect").addEventListener("change", (e) => {
    document.getElementById("customQuestion").classList.toggle("hidden", e.target.value !== "Add custom question...");
    resetPracticeThread("Question changed — starting a new practice flow.");
    document.getElementById("analysisResult").textContent = "";
    setPracticePrompt(currentQuestion());
  });

  const customQuestion = document.getElementById("customQuestion");
  if (customQuestion) {
    customQuestion.addEventListener("input", () => {
      if (!state.practiceThread) setPracticePrompt(currentQuestion());
    });
  }

  const stopBtn = document.getElementById("stopInterviewBtn");
  if (stopBtn) {
    stopBtn.addEventListener("click", stopPracticeInterview);
  }

  document.getElementById("analyzeTextBtn").addEventListener("click", () => {
    submitTextPractice().catch((e) => {
      setLoading(false);
      showToast(e.message, "error");
    });
  });

  document.getElementById("analyzeAudioBtn").addEventListener("click", () => {
    submitAudioPractice().catch((e) => {
      setLoading(false);
      showToast(e.message, "error");
    });
  });

  const audioRecognizer = initSpeech("audioTranscript", "audioSpeechTimer");
  document.getElementById("startSpeechBtn").addEventListener("click", () => safeStartRecognition(audioRecognizer));
  document.getElementById("stopSpeechBtn").addEventListener("click", () => {
    if (audioRecognizer) audioRecognizer.stop();
    stopSpeechTimer("audioSpeechTimer");
  });

  const videoRecognizer = initSpeech("videoTranscript", "videoSpeechTimer");
  document.getElementById("startVideoSpeechBtn").addEventListener("click", () => safeStartRecognition(videoRecognizer));
  document.getElementById("stopVideoSpeechBtn").addEventListener("click", () => {
    if (videoRecognizer) videoRecognizer.stop();
    stopSpeechTimer("videoSpeechTimer");
  });

  document.getElementById("startCameraBtn").addEventListener("click", () => {
    startCamera().catch((e) => showToast(e.message, "error"));
  });

  document.getElementById("stopCameraBtn").addEventListener("click", stopCamera);
  document.getElementById("submitVideoBtn").addEventListener("click", () => {
    submitVideoPractice().catch((e) => {
      setLoading(false);
      showToast(e.message, "error");
    });
  });

  const mockBtn = document.getElementById("generateMockBtn");
  if (mockBtn) {
    mockBtn.addEventListener("click", () => {
      generateMockInterview().catch((e) => {
        setLoading(false);
        setMessage("mockMessage", e.message);
        showToast(e.message, "error");
      });
    });
  }

  document.getElementById("downloadPdfBtn").addEventListener("click", () => {
    const startDate = document.getElementById("startDate").value;
    const endDate = document.getElementById("endDate").value;
    const params = new URLSearchParams();
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);
    window.open(`/api/reports/pdf?${params.toString()}`, "_blank");
    showToast("Downloading PDF report", "success");
  });

  ["startDate", "endDate"].forEach((id) => {
    const dateInput = document.getElementById(id);
    if (!dateInput) return;
    ["focus", "click"].forEach((evt) => {
      dateInput.addEventListener(evt, () => {
        if (typeof dateInput.showPicker === "function") {
          dateInput.showPicker();
        }
      });
    });
  });
}

async function bootstrap() {
  setupEvents();
  setupPasswordVisibility();
  initParallaxBackground();
  initCardTilt();
  resetRlSummary();
  updatePostureUi(state.posture.status, state.posture.feedback);
  switchAuthMode("login");

  initFirebaseEmailLinkAuth();
  // If the user landed via a Firebase email-link sign-in, complete it before checking /api/me.
  await completeFirebaseEmailLinkSignIn();
  await loginFlow();
}

bootstrap().catch(() => {
  setMessage("authMessage", "Unable to initialize app.");
});
