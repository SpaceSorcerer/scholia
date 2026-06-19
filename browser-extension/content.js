/*
 * content.js — runs inside Word Online pages.
 * ============================================================================
 * Responsibilities:
 *   1. extractText(): best-effort capture of what the user is writing.
 *   2. A small floating "Ground (Scholia)" button + a keyboard command.
 *   3. On trigger, send the captured text to the background service worker
 *      (which does the cross-origin fetch to the local bridge — NOT this script,
 *      to avoid page-CORS).
 *
 * PRIVACY: this script only reads text the user authored and hands it to the
 * background worker, which contacts ONLY http://127.0.0.1:8765 (the local
 * bridge). No page content is ever sent anywhere else.
 *
 * The DOM selectors live in selectors.js (loaded before this file) so capture
 * can be fixed without editing this logic.
 */

(function () {
  'use strict';

  // selectors.js sets window.ScholiaSelectors; fall back to a minimal generic
  // set so the script never throws if selectors.js failed to load.
  const SEL = (typeof window !== 'undefined' && window.ScholiaSelectors) || {
    EDITABLE_REGION_SELECTORS: ['[contenteditable="true"]', '[role="textbox"]'],
    PARAGRAPH_SELECTORS: ['p', 'div']
  };

  // --------------------------------------------------------------------------
  // CAPTURE
  // --------------------------------------------------------------------------

  /**
   * Return the first editable-region element on the page, or null.
   * Tries each selector in EDITABLE_REGION_SELECTORS (most specific first).
   * @param {Document|Element} root
   * @returns {Element|null}
   */
  function findEditableRegion(root) {
    const scope = root || document;
    for (const sel of SEL.EDITABLE_REGION_SELECTORS) {
      try {
        const el = scope.querySelector(sel);
        if (el) return el;
      } catch (_e) {
        // invalid selector string — skip, try the next one
      }
    }
    return null;
  }

  /**
   * Walk up from a node to the nearest paragraph/block element using
   * PARAGRAPH_SELECTORS. Returns the node's containing block, or null.
   * @param {Node|null} node
   * @returns {Element|null}
   */
  function nearestParagraph(node) {
    let el = node && node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    while (el && el !== document.body) {
      for (const sel of SEL.PARAGRAPH_SELECTORS) {
        try {
          if (el.matches && el.matches(sel)) return el;
        } catch (_e) {
          // bad selector — ignore
        }
      }
      el = el.parentElement;
    }
    return null;
  }

  function clean(text) {
    return (text || '').replace(/\s+/g, ' ').trim();
  }

  /**
   * Best-effort capture of the text the user is writing in Word Online.
   *
   * Strategy, in priority order:
   *   (a) The current SELECTION — what the user highlighted. Most precise.
   *   (b) The PARAGRAPH/BLOCK the caret sits in (selection collapsed). Uses the
   *       selection's anchorNode, walked up to the nearest paragraph block.
   *   (c) DOCUMENTED FALLBACK: the visible text of the whole editable region,
   *       trimmed to a sane cap, so the button still does *something* useful
   *       even when (a) and (b) come up empty.
   *
   * Returns { text, method } so the UI can tell the user WHERE the text came
   * from (and so a wrong source is diagnosable).
   *
   * @param {Object} [opts]
   * @param {Document|Element} [opts.root=document] - search scope (tests pass a mock).
   * @param {Selection} [opts.selection] - inject a selection (tests/iframes).
   * @param {number} [opts.maxChars=2000] - cap for the whole-region fallback.
   * @returns {{text: string, method: string}}
   */
  function extractText(opts) {
    opts = opts || {};
    const root = opts.root || document;
    const maxChars = opts.maxChars || 2000;
    const selection =
      opts.selection ||
      (typeof window !== 'undefined' && window.getSelection
        ? window.getSelection()
        : null);

    // (a) Active selection text.
    if (selection) {
      const selText = clean(selection.toString());
      if (selText) {
        return { text: selText, method: 'selection' };
      }
    }

    // (b) Surrounding paragraph/block of the caret.
    if (selection && selection.anchorNode) {
      const para = nearestParagraph(selection.anchorNode);
      if (para) {
        const paraText = clean(para.innerText || para.textContent);
        if (paraText) {
          return { text: paraText, method: 'paragraph' };
        }
      }
    }

    // (c) Documented fallback: whole visible editable region (capped).
    const region = findEditableRegion(root);
    if (region) {
      let regionText = clean(region.innerText || region.textContent);
      if (regionText) {
        if (regionText.length > maxChars) {
          regionText = regionText.slice(0, maxChars);
        }
        return { text: regionText, method: 'region-fallback' };
      }
    }

    return { text: '', method: 'none' };
  }

  // Expose for unit tests / playwright injection. (Pure, no side effects.)
  if (typeof window !== 'undefined') {
    window.ScholiaExtract = {
      extractText,
      findEditableRegion,
      nearestParagraph
    };
  }

  // --------------------------------------------------------------------------
  // TRIGGER: floating button + background round-trip
  // --------------------------------------------------------------------------

  // Content scripts in non-extension test pages won't have chrome.runtime.
  const HAS_RUNTIME =
    typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage;

  function flash(msg, isError) {
    let toast = document.getElementById('scholia-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'scholia-toast';
      toast.style.cssText =
        'position:fixed;z-index:2147483647;bottom:64px;right:16px;max-width:320px;' +
        'padding:10px 14px;border-radius:8px;font:13px/1.4 system-ui,sans-serif;' +
        'box-shadow:0 2px 12px rgba(0,0,0,.25);color:#fff;transition:opacity .3s;';
      document.body.appendChild(toast);
    }
    toast.style.background = isError ? '#b00020' : '#1f6feb';
    toast.textContent = 'Scholia: ' + msg;
    toast.style.opacity = '1';
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      toast.style.opacity = '0';
    }, 4000);
  }

  /**
   * Capture text and ask the background worker to ground it. Result is stored
   * by the worker and also pushed to the popup if open; here we just surface a
   * lightweight inline toast so the user gets feedback from the page itself.
   */
  function triggerGround() {
    const { text, method } = extractText();
    if (!text) {
      flash(
        'No text captured. Select a sentence first. ' +
          '(If selection never works here, the DOM selectors need tweaking — see README.)',
        true
      );
      return;
    }
    if (!HAS_RUNTIME) {
      flash('captured ' + text.length + ' chars (' + method + '), but no extension runtime.', true);
      return;
    }
    flash('grounding ' + text.length + ' chars (' + method + ')…', false);
    chrome.runtime.sendMessage(
      { type: 'scholia-ground', passage: text, method },
      (resp) => {
        if (chrome.runtime.lastError) {
          flash('worker error: ' + chrome.runtime.lastError.message, true);
          return;
        }
        if (!resp || !resp.ok) {
          flash((resp && resp.error) || 'bridge unreachable — start `scholia serve`', true);
          return;
        }
        const cc = resp.data && resp.data.claim_check;
        const verdict = cc ? (cc.supported ? 'SUPPORTED' : 'UNSUPPORTED') : '?';
        const n = (resp.data && resp.data.suggestions && resp.data.suggestions.length) || 0;
        flash(verdict + ' — ' + n + ' paper(s). Open the Scholia popup for details.', !cc || !cc.supported);
      }
    );
  }

  function injectButton() {
    if (document.getElementById('scholia-ground-btn')) return;
    if (!document.body) return;
    const btn = document.createElement('button');
    btn.id = 'scholia-ground-btn';
    btn.type = 'button';
    btn.textContent = 'Ground (Scholia)';
    btn.title = 'Ground the current selection against your Scholia library (Ctrl+Shift+G)';
    btn.style.cssText =
      'position:fixed;z-index:2147483647;bottom:16px;right:16px;' +
      'padding:8px 14px;border:none;border-radius:8px;cursor:pointer;' +
      'background:#1f6feb;color:#fff;font:600 13px/1 system-ui,sans-serif;' +
      'box-shadow:0 2px 10px rgba(0,0,0,.3);';
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      triggerGround();
    });
    document.body.appendChild(btn);
  }

  // Keyboard command from manifest "commands" is delivered to the background
  // worker, which relays a message here. Also support a direct in-page hotkey
  // as a redundant path (some Word shortcuts may swallow the command event).
  if (HAS_RUNTIME) {
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if (msg && msg.type === 'scholia-ground-command') {
        triggerGround();
        sendResponse({ ok: true });
      } else if (msg && msg.type === 'scholia-extract-only') {
        // Popup asks the active tab for the current selection without grounding.
        sendResponse(extractText());
      }
      return true;
    });
  }

  // Inject the button once the DOM is ready. Word Online mutates the DOM a lot;
  // re-assert the button on a light interval so it survives re-renders.
  if (typeof document !== 'undefined' && document.body) {
    injectButton();
    setInterval(injectButton, 3000);
  } else if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', injectButton);
  }
})();
