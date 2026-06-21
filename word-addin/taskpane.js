/* =========================================================
   Scholia task-pane — Office.js client
   =========================================================
   Talks ONLY to the local Scholia bridge on 127.0.0.1.
   Loaded from https://127.0.0.1:8765/taskpane.html, so all
   /cite and /discover calls are same-origin (no CORS, no
   mixed-content blocking).

   Hard rules enforced here:
   - NEVER generates, drafts, or rewrites manuscript prose.
   - NEVER sends text anywhere except 127.0.0.1.
   - Display-only: clicking a paper link opens the user's
     Zotero or a DOI URL — nothing is inserted into Word.
   ========================================================= */

"use strict";

const BRIDGE_BASE = window.location.origin; // https://127.0.0.1:8765

// ---- DOM refs ----
let btnGround, btnDiscover;
let statusBanner, idleHint;
let resultsSection, verdictBadge, paperList;
let discoverSection, discoverList;
let bridgeStatus;

Office.onReady(info => {
  // Wire up DOM refs after Office is ready (DOM is guaranteed loaded).
  btnGround      = document.getElementById("btn-ground");
  btnDiscover    = document.getElementById("btn-discover");
  statusBanner   = document.getElementById("status-banner");
  idleHint       = document.getElementById("idle-hint");
  resultsSection = document.getElementById("results");
  verdictBadge   = document.getElementById("verdict-badge");
  paperList      = document.getElementById("paper-list");
  discoverSection= document.getElementById("discover-results");
  discoverList   = document.getElementById("discover-list");
  bridgeStatus   = document.getElementById("bridge-status");

  if (info.host === Office.HostType.Word) {
    btnGround.addEventListener("click", onGroundClick);
    btnDiscover.addEventListener("click", onDiscoverClick);
    checkBridgeHealth();
  } else {
    showBanner("This add-in only works inside Word.", "error");
  }
});

// ---- Health check ----
async function checkBridgeHealth() {
  try {
    const res = await fetch(`${BRIDGE_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    bridgeStatus.textContent = `Engine running · ${data.papers} papers`;
    bridgeStatus.className = "bridge-status ok";
  } catch {
    bridgeStatus.textContent = "Engine offline";
    bridgeStatus.className = "bridge-status error";
  }
}

// ---- Ground action ----
async function onGroundClick() {
  const text = await getSelection();
  if (!text) {
    showBanner("No text selected — select a passage in your document first.", "info");
    return;
  }
  showBanner("Grounding…", "loading");
  setButtonsEnabled(false);
  hideResults();

  try {
    const res = await fetch(`${BRIDGE_BASE}/cite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passage: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    hideBanner();
    renderGroundResults(data);
  } catch (e) {
    const msg = e.message.includes("fetch") || e.message.includes("Failed")
      ? "Cannot reach the Scholia engine. Start the bridge first: scholia serve --serve-addin"
      : `Error: ${e.message}`;
    showBanner(msg, "error");
  } finally {
    setButtonsEnabled(true);
  }
}

// ---- Discover action ----
async function onDiscoverClick() {
  const text = await getSelection();
  if (!text) {
    showBanner("No text selected — select a passage in your document first.", "info");
    return;
  }
  showBanner("Discovering new papers…", "loading");
  setButtonsEnabled(false);
  hideResults();

  try {
    const res = await fetch(`${BRIDGE_BASE}/discover`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passage: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    hideBanner();
    renderDiscoverResults(data);
  } catch (e) {
    const msg = e.message.includes("fetch") || e.message.includes("Failed")
      ? "Cannot reach the Scholia engine. Start the bridge first: scholia serve --serve-addin"
      : `Error: ${e.message}`;
    showBanner(msg, "error");
  } finally {
    setButtonsEnabled(true);
  }
}

// ---- Office.js: read selection ----
async function getSelection() {
  return new Promise((resolve, reject) => {
    Word.run(async ctx => {
      try {
        const sel = ctx.document.getSelection();
        sel.load("text");
        await ctx.sync();
        resolve(sel.text.trim());
      } catch (e) {
        reject(e);
      }
    });
  });
}

// ---- Render: ground results ----
function renderGroundResults(data) {
  idleHint.classList.add("hidden");
  discoverSection.classList.add("hidden");

  // Verdict badge
  const cc = data.claim_check || {};
  if (cc.supported) {
    verdictBadge.textContent = "SUPPORTED by your library";
    verdictBadge.className = "verdict-badge verdict-supported";
  } else {
    verdictBadge.textContent = "Not clearly supported — check your sources";
    verdictBadge.className = "verdict-badge verdict-unsupported";
  }

  // Paper list
  paperList.innerHTML = "";
  const suggestions = data.suggestions || [];
  if (suggestions.length === 0) {
    paperList.innerHTML = "<li class='paper-item'><em>No matching papers found in your library.</em></li>";
  } else {
    suggestions.forEach(s => {
      paperList.appendChild(buildPaperItem(s));
    });
  }

  resultsSection.classList.remove("hidden");
}

// ---- Render: discover results ----
function renderDiscoverResults(data) {
  idleHint.classList.add("hidden");
  resultsSection.classList.add("hidden");

  discoverList.innerHTML = "";
  const candidates = data.candidates || [];
  if (candidates.length === 0) {
    discoverList.innerHTML = "<li class='paper-item'><em>Nothing new found — everything relevant may already be in your library.</em></li>";
  } else {
    candidates.forEach((c, i) => {
      discoverList.appendChild(buildDiscoverItem(c, i + 1));
    });
  }

  discoverSection.classList.remove("hidden");
}

// ---- Build DOM: supporting paper ----
function buildPaperItem(s) {
  const li = document.createElement("li");
  li.className = "paper-item";

  const score = s.score != null ? s.score.toFixed(3) : "";
  const author = s.first_author || "Unknown";
  const year   = s.year || "n.d.";

  li.innerHTML = `
    <div class="paper-rank">
      #${s.rank}
      ${score ? `<span class="paper-score">score ${score}</span>` : ""}
    </div>
    <div class="paper-title">${escHtml(s.title || "(untitled)")}</div>
    <div class="paper-meta">${escHtml(author)} &middot; ${escHtml(String(year))}</div>
    <div class="paper-links">
      ${s.doi ? `<a href="https://doi.org/${escHtml(s.doi)}" target="_blank" rel="noopener">DOI</a>` : ""}
      ${s.zotero_link ? `<a href="${escHtml(s.zotero_link)}" target="_blank">Open in Zotero</a>` : ""}
    </div>`;
  return li;
}

// ---- Build DOM: discovery candidate ----
function buildDiscoverItem(c, rank) {
  const li = document.createElement("li");
  li.className = "paper-item";

  const author = c.authors && c.authors.length > 0
    ? c.authors[0].split(",")[0].trim()
    : "Unknown";
  const year = c.year || "n.d.";
  const snippet = c.snippet || "";

  li.innerHTML = `
    <div class="paper-rank">#${rank} · ${escHtml(c.source || "")}</div>
    <div class="paper-title">${escHtml(c.title || "(untitled)")}</div>
    <div class="paper-meta">${escHtml(author)} &middot; ${escHtml(String(year))}</div>
    ${snippet ? `<div class="paper-meta" style="margin-top:3px;font-style:italic">${escHtml(snippet)}</div>` : ""}
    <div class="paper-links">
      ${c.doi ? `<a href="https://doi.org/${escHtml(c.doi)}" target="_blank" rel="noopener">DOI</a>` : ""}
    </div>`;
  return li;
}

// ---- Helpers ----
function setButtonsEnabled(enabled) {
  btnGround.disabled   = !enabled;
  btnDiscover.disabled = !enabled;
}

function showBanner(msg, type) {
  statusBanner.textContent = msg;
  statusBanner.className = `banner ${type}`;
  statusBanner.classList.remove("hidden");
}

function hideBanner() {
  statusBanner.classList.add("hidden");
}

function hideResults() {
  resultsSection.classList.add("hidden");
  discoverSection.classList.add("hidden");
  idleHint.classList.remove("hidden");
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
