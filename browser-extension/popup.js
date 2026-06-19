/*
 * popup.js — the action popup UI.
 * ============================================================================
 * Buttons:
 *   - "Ground selection": ask the active Word-Online tab for its current
 *     selection/paragraph (via content.js extractText), then ground it through
 *     the background worker -> POST /cite. Renders supporting papers + verdict.
 *   - "Discover": same captured text -> POST /discover. Renders new candidates.
 *
 * All bridge I/O goes through the background worker (chrome.runtime.sendMessage)
 * so there is no CORS in play and the localhost host-permission is honored.
 *
 * Rendering is display-only: titles, authors, year, and CLICKABLE doi.org +
 * zotero:// links. No prose is generated anywhere.
 */

const $ = (id) => document.getElementById(id);
let capturedPassage = '';

function setStatus(text, cls) {
  const el = $('status');
  el.textContent = text;
  el.className = 'status' + (cls ? ' ' + cls : '');
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function send(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (resp) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
      } else {
        resolve(resp || { ok: false, error: 'No response from worker.' });
      }
    });
  });
}

// Ask the active tab's content script for the current selection/paragraph.
function captureFromActiveTab() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || !tabs[0]) {
        resolve({ text: '', method: 'no-tab' });
        return;
      }
      chrome.tabs.sendMessage(tabs[0].id, { type: 'scholia-extract-only' }, (resp) => {
        if (chrome.runtime.lastError || !resp) {
          // Content script not present (not a Word page) or page blocked it.
          resolve({ text: '', method: 'no-content-script' });
        } else {
          resolve(resp);
        }
      });
    });
  });
}

// ---- rendering -------------------------------------------------------------

function paperLinks(p) {
  const links = [];
  if (p.doi) {
    links.push(
      '<a href="https://doi.org/' +
        encodeURIComponent(p.doi) +
        '" target="_blank" rel="noopener">DOI ' +
        escapeHtml(p.doi) +
        '</a>'
    );
  }
  if (p.zotero_link) {
    links.push('<a href="' + escapeHtml(p.zotero_link) + '">Zotero</a>');
  }
  return links.join('');
}

function renderCite(data) {
  $('ranking-signal').textContent = data.ranking_signal || '';
  const cc = data.claim_check || {};
  const verdict = $('verdict');
  verdict.style.display = 'block';
  if (cc.supported) {
    verdict.className = 'verdict supported';
    verdict.textContent =
      'SUPPORTED by your library (top ' +
      (cc.top_score != null ? cc.top_score.toFixed(3) : '?') +
      ' ≥ ' +
      cc.threshold +
      ')';
  } else {
    verdict.className = 'verdict unsupported';
    verdict.textContent =
      'UNSUPPORTED by your library (top ' +
      (cc.top_score != null ? cc.top_score.toFixed(3) : '?') +
      ' < ' +
      cc.threshold +
      ') — consider Discover.';
  }

  const sugg = data.suggestions || [];
  let html = '<div class="section-title">Supporting papers (' + sugg.length + ')</div>';
  if (!sugg.length) {
    html += '<div class="empty">No matching papers in your library.</div>';
  } else {
    for (const p of sugg) {
      html +=
        '<div class="paper">' +
        '<span class="score">' +
        (p.score != null ? p.score.toFixed(3) : '') +
        '</span>' +
        '<div class="title">' +
        escapeHtml(p.title) +
        '</div>' +
        '<div class="meta">' +
        escapeHtml(p.first_author) +
        ' · ' +
        escapeHtml(p.year || 'n.d.') +
        '</div>' +
        '<div class="links">' +
        paperLinks(p) +
        '</div>' +
        '</div>';
    }
  }
  $('results').innerHTML = html;
}

function renderDiscover(data) {
  const cands = data.candidates || [];
  let html =
    '<div class="section-title">Discovered — NOT in your library (' +
    cands.length +
    ')</div>';
  html +=
    '<div class="empty" style="margin-bottom:6px">Suggestions only. Query sent: <i>' +
    escapeHtml(data.query || '') +
    '</i></div>';
  if (!cands.length) {
    html += '<div class="empty">No new candidates found.</div>';
  } else {
    for (const c of cands) {
      const author = (c.authors && c.authors[0]) || 'Unknown';
      const doi = c.doi
        ? '<a href="https://doi.org/' +
          encodeURIComponent(c.doi) +
          '" target="_blank" rel="noopener">DOI ' +
          escapeHtml(c.doi) +
          '</a>'
        : '<span class="empty">no DOI</span>';
      html +=
        '<div class="paper candidate">' +
        '<div class="title">' +
        escapeHtml(c.title) +
        '</div>' +
        '<div class="meta">' +
        escapeHtml(author) +
        ' · ' +
        escapeHtml(c.year || 'n.d.') +
        ' · <span class="src">' +
        escapeHtml(c.source || '') +
        '</span></div>' +
        '<div class="links">' +
        doi +
        '</div>' +
        '</div>';
    }
  }
  $('results').innerHTML = html;
}

// ---- actions ---------------------------------------------------------------

async function refreshCapture() {
  const cap = await captureFromActiveTab();
  capturedPassage = (cap && cap.text) || '';
  const box = $('captured');
  if (capturedPassage) {
    box.style.display = 'block';
    box.textContent =
      '[' + (cap.method || '?') + '] ' + capturedPassage.slice(0, 280) +
      (capturedPassage.length > 280 ? '…' : '');
  } else {
    box.style.display = 'none';
  }
  const ok = !!capturedPassage;
  $('ground-btn').disabled = !ok;
  $('discover-btn').disabled = !ok;
  return ok;
}

async function doGround() {
  if (!(await refreshCapture())) {
    setStatus('No selection captured in the active tab. Select text in Word Online first.', 'bad');
    return;
  }
  setStatus('Grounding ' + capturedPassage.length + ' chars…');
  const res = await send({ type: 'scholia-ground', passage: capturedPassage });
  if (!res.ok) {
    setStatus(res.error, 'bad');
    return;
  }
  setStatus('Grounded ' + capturedPassage.length + ' chars.', 'ok');
  renderCite(res.data);
}

async function doDiscover() {
  if (!(await refreshCapture())) {
    setStatus('No selection captured in the active tab. Select text in Word Online first.', 'bad');
    return;
  }
  setStatus('Discovering for ' + capturedPassage.length + ' chars…');
  const res = await send({ type: 'scholia-discover', passage: capturedPassage });
  if (!res.ok) {
    setStatus(res.error, 'bad');
    return;
  }
  setStatus('Discovery complete.', 'ok');
  renderDiscover(res.data);
}

async function init() {
  $('ground-btn').addEventListener('click', doGround);
  $('discover-btn').addEventListener('click', doDiscover);

  // Health check + initial capture preview.
  const h = await send({ type: 'scholia-health' });
  if (h.ok) {
    setStatus(
      'Bridge OK — ' + h.data.papers + ' papers (' + h.data.embedder + ').',
      'ok'
    );
  } else {
    setStatus('Bridge offline. Start `scholia serve`.', 'bad');
  }
  await refreshCapture();

  // If the floating button already grounded something, show it immediately.
  const last = await send({ type: 'scholia-last' });
  if (last.ok && last.data) {
    renderCite(last.data);
  }
}

document.addEventListener('DOMContentLoaded', init);
