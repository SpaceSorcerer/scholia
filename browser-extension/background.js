/*
 * background.js — MV3 service worker.
 * ============================================================================
 * The ONLY component that talks to the local Scholia bridge. Doing the fetch
 * here (rather than from the content script) means the request originates from
 * the extension's own context with host_permissions for
 * http://127.0.0.1:8765/*, so the page's origin / CORS policy never applies —
 * no preflight, no Access-Control-Allow-Origin needed on the server.
 *
 * Message protocol (from content.js / popup.js):
 *   { type:'scholia-ground',   passage, [k], [rerank] }  -> POST /cite
 *   { type:'scholia-discover', passage, [limit] }        -> POST /discover
 *   { type:'scholia-health' }                            -> GET  /health
 *   { type:'scholia-last' }                              -> last cached cite result
 * Each resolves to { ok:true, data } or { ok:false, error }.
 *
 * PRIVACY: contacts ONLY the localhost bridge. The bridge itself only sends a
 * short keyword query (never the passage) to scholarly APIs during /discover.
 */

const BRIDGE = 'http://127.0.0.1:8765';
const TIMEOUT_MS = 30000; // cross-encoder rerank can be ~2s+/query; be generous.

// In-memory cache of the most recent cite result so a freshly-opened popup can
// show whatever the floating button last grounded. Cleared on worker restart
// (acceptable: it is a convenience, not a source of truth).
let lastCite = null;

function bridgeUnreachableError(err) {
  return (
    'Cannot reach the Scholia bridge at ' +
    BRIDGE +
    '. Start it with `scholia serve` (or `scholia serve --fake-embedder` for a ' +
    'model-free run). Underlying error: ' +
    (err && err.message ? err.message : String(err))
  );
}

async function bridgeFetch(path, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const resp = await fetch(BRIDGE + path, {
      ...options,
      signal: controller.signal
    });
    const text = await resp.text();
    let json;
    try {
      json = text ? JSON.parse(text) : {};
    } catch (_e) {
      return { ok: false, error: 'Bridge returned non-JSON (HTTP ' + resp.status + '): ' + text.slice(0, 200) };
    }
    if (!resp.ok) {
      return { ok: false, error: (json && json.error) || 'Bridge HTTP ' + resp.status };
    }
    return { ok: true, data: json };
  } catch (err) {
    return { ok: false, error: bridgeUnreachableError(err) };
  } finally {
    clearTimeout(timer);
  }
}

async function cite(passage, k, rerank) {
  const body = { passage };
  if (typeof k === 'number') body.k = k;
  if (typeof rerank === 'boolean') body.rerank = rerank;
  const res = await bridgeFetch('/cite', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (res.ok) lastCite = res.data;
  return res;
}

async function discover(passage, limit) {
  const body = { passage };
  if (typeof limit === 'number') body.limit = limit;
  return bridgeFetch('/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

async function health() {
  return bridgeFetch('/health', { method: 'GET' });
}

// --------------------------------------------------------------------------
// Message router. Returning true keeps the channel open for the async reply.
// --------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || !msg.type) {
    sendResponse({ ok: false, error: 'No message type.' });
    return false;
  }

  switch (msg.type) {
    case 'scholia-ground':
      cite(msg.passage, msg.k, msg.rerank).then(sendResponse);
      return true;
    case 'scholia-discover':
      discover(msg.passage, msg.limit).then(sendResponse);
      return true;
    case 'scholia-health':
      health().then(sendResponse);
      return true;
    case 'scholia-last':
      sendResponse({ ok: !!lastCite, data: lastCite, error: lastCite ? null : 'No grounding yet.' });
      return false;
    default:
      sendResponse({ ok: false, error: 'Unknown message type: ' + msg.type });
      return false;
  }
});

// --------------------------------------------------------------------------
// Keyboard command -> relay to the active tab's content script to capture +
// ground (capture must happen in the page, not the worker).
// --------------------------------------------------------------------------
if (chrome.commands && chrome.commands.onCommand) {
  chrome.commands.onCommand.addListener((command) => {
    if (command !== 'ground-selection') return;
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || !tabs[0]) return;
      chrome.tabs.sendMessage(tabs[0].id, { type: 'scholia-ground-command' }, () => {
        // Swallow "no receiving end" when the active tab is not a Word page.
        void chrome.runtime.lastError;
      });
    });
  });
}
