# Scholia — Word Online capture extension (MV3)

A Manifest-V3 browser extension (Chrome / Edge) that gives the **live, in-editor
Scholia experience inside Word Online**: it captures the sentence you are
writing, grounds it against your own validated Zotero library via the local
`scholia serve` bridge, and shows supporting papers + a SUPPORTED/UNSUPPORTED
claim-check + discovery — without leaving the browser, and without ever sending
your prose to the cloud.

> **Status: solid prototype.** The plumbing (capture → background worker → local
> bridge → results UI) is built and validated against a local mock and the live
> bridge. The **one thing that needs your validation against the real,
> authenticated Word Online app is the DOM capture selectors** — see
> [If capture returns nothing](#if-capture-returns-nothing-fix-the-selectors).

---

## What it does

- **Floating "Ground (Scholia)" button** (bottom-right of the Word page) and a
  **keyboard shortcut** (`Ctrl+Shift+G`) capture your current selection (or the
  paragraph the caret is in) and ground it.
- **Popup** (toolbar icon): "Ground selection" + "Discover" buttons; renders
  supporting papers with **clickable DOI and `zotero://` links**, the
  SUPPORTED/UNSUPPORTED verdict, and discovered (not-in-library) candidates.
- **100% local & display-only.** The extension contacts *only* the Scholia
  bridge on `http://127.0.0.1:8765`. No prose is generated; no cloud LLM is
  contacted. (Discovery, run by the bridge, sends only a short keyword query —
  never your draft — to PubMed / Semantic Scholar.)

---

## Architecture (and why it's built this way)

```
Word Online page
  ├─ content.js   capture: selection → paragraph → whole-region (selectors.js)
  │                 │  chrome.runtime.sendMessage({type:'scholia-ground', passage})
  ▼                 ▼
background.js (service worker)  ──fetch──▶  http://127.0.0.1:8765  (scholia serve)
  │   POST /cite, POST /discover, GET /health        the local bridge
  ▲                 │
  └─ results ───────┘
popup.html/js  ── ask active tab for selection, render papers + verdict + links
```

**The fetch to the bridge happens in the background service worker, NOT the
content script — on purpose.** A `fetch` from the Word page itself would be a
cross-origin request that triggers a CORS *preflight* (`OPTIONS`), which the
tiny stdlib bridge does not answer (it returns `501`), so the request fails. The
MV3 service worker, holding `host_permissions` for `http://127.0.0.1:8765/*`,
issues the request from the **extension's own context**, which is not subject to
page CORS — no preflight, no server-side CORS headers needed. This keeps the
localhost bridge minimal and unexposed. (Verified: a page-context fetch is
blocked; the worker pattern is the standard MV3 way around it.)

| File | Role |
|---|---|
| `manifest.json` | MV3 manifest: host permissions (bridge + Word domains), content scripts on Word domains, background worker, action popup, `Ctrl+Shift+G` command. |
| `selectors.js` | **The Word-Online DOM selectors, isolated for easy tweaking.** This is the file you edit if capture misses. |
| `content.js` | `extractText()` (selection → paragraph → region fallback), floating button, keyboard relay, page-side toast. |
| `background.js` | Service worker: the ONLY component that fetches the bridge. Message router + keyboard-command relay. |
| `popup.html` / `popup.js` | Toolbar popup: Ground / Discover buttons, results pane, health status. |
| `mock/word-online-mock.html` | A local mock editor for validating capture logic without the real app. |

---

## Install (load unpacked)

1. Start the bridge in a terminal (see next section).
2. Open **Edge** → `edge://extensions`  (or **Chrome** → `chrome://extensions`).
3. Toggle **Developer mode** (top-right / left sidebar).
4. Click **Load unpacked** and select this folder:
   `E:\Claude\scholia\browser-extension`
5. Pin the **Scholia** icon to the toolbar (optional, convenient).
6. Open a document in **Word Online** (`https://*.officeapps.live.com/...`).
   You should see the blue **Ground (Scholia)** button bottom-right.

> Edge and Chrome both load MV3 unpacked the same way. No build step, no npm —
> it's vanilla JS.

---

## Start the bridge

The extension is a **client** of `scholia serve`. From the repo root
(`E:\Claude\scholia`):

```bash
# Model-free run (no weights download — great for a first smoke test):
scholia index --corpus tests/fixtures/corpus --index-dir .scholia_ext_index --fake-embedder
scholia serve --index-dir .scholia_ext_index --fake-embedder --no-rerank

# Real run against your actual library:
scholia index --corpus "/path/to/your/zotero-mirror"
scholia serve            # loads your real index + models once
```

The popup's status line shows **"Bridge OK — N papers"** when it can reach it,
or **"Bridge offline. Start `scholia serve`."** when it can't.

---

## Use it

- **In the page:** select a sentence in Word Online → click **Ground (Scholia)**
  (or press `Ctrl+Shift+G`). A toast shows the verdict; open the popup for the
  full paper list with links.
- **In the popup:** click **Ground selection** (it pulls your current selection
  from the page) or **Discover**. Click any **DOI** or **Zotero** link to open
  it.

---

## If capture returns nothing (fix the selectors)

**This is the most likely thing you'll need to tweak.** Word Online does *not*
render the document as normal HTML paragraphs — it paints a layered/canvas-ish
surface and keeps the editable text in a hidden contenteditable region whose
class names are obfuscated build artifacts that **Microsoft changes without
notice**. The selectors shipped in `selectors.js` are a best-effort starting
point (verified against the mock); the live app may use different ones.

If the floating button says *"No text captured"* even with text selected, or the
popup shows nothing:

1. **Open the doc in Word Online and select a sentence.**
2. **Open DevTools** (`F12`) → **Console**.
3. **Find the real selection container** — paste and run:
   ```js
   let n = document.getSelection().anchorNode;
   while (n && n.nodeType === 3) n = n.parentElement;   // climb out of text node
   // Walk up and print each ancestor's tag + classes + key attrs:
   for (let el = n; el && el !== document.body; el = el.parentElement) {
     console.log(el.tagName, '| class=', el.className,
                 '| contenteditable=', el.getAttribute('contenteditable'),
                 '| role=', el.getAttribute('role'),
                 '| aria-label=', el.getAttribute('aria-label'));
   }
   ```
4. **Identify two things** from that output:
   - the **editable region** (the big container with `contenteditable="true"`
     or `role="textbox"` or an obvious `...EditingSurface...`/`...ViewPanel...`
     class), and
   - the **paragraph block** (the wrapper around a single line/sentence).
5. **Edit `selectors.js`** and add the real selectors at the **top** of the
   matching list (first match wins):
   ```js
   EDITABLE_REGION_SELECTORS: [
     'div.TheRealClassYouFound',   // ← add yours first
     // ...existing fallbacks below...
   ],
   PARAGRAPH_SELECTORS: [
     'div.TheRealParagraphClass',  // ← add yours first
     // ...existing fallbacks below...
   ],
   ```
6. **Reload the extension** (`edge://extensions` → the Scholia card → reload
   ↻), refresh the Word tab, and try again.

You only ever edit `selectors.js` — `content.js` reads from it and never
hardcodes a class name, so capture logic stays intact.

**Frame note.** Word Online sometimes renders the editor inside an `<iframe>`.
The manifest already injects on `all_frames: true`, so the content script runs in
the frame too; if capture still fails, confirm in DevTools which frame holds the
editable region (the DevTools frame dropdown) and that its URL matches one of the
`content_scripts` match patterns in `manifest.json` (add the frame's host there
if it differs).

---

## Validate the capture logic (without real Word Online)

A local mock (`mock/word-online-mock.html`) mimics the Word-for-web structure
(`div.EditingSurfaceBody[contenteditable]` containing `.Paragraph` blocks) so you
can confirm `extractText()` works end to end. To check it manually:

```bash
# Serve the extension dir so same-origin script injection works, then open
# http://127.0.0.1:8800/mock/word-online-mock.html in a browser:
cd browser-extension && python -m http.server 8800
```

Open the page, open DevTools console, and run:

```js
// load the real extraction code, then test all three tiers
for (const f of ['/selectors.js', '/content.js']) {
  const s = document.createElement('script');
  s.textContent = await (await fetch(f)).text();
  document.head.appendChild(s);
}
const sel = getSelection(); sel.removeAllRanges();
const r = document.createRange(); r.selectNodeContents(document.getElementById('p2')); sel.addRange(r);
console.log(window.ScholiaExtract.extractText());   // → {method:'selection', text:'QKI ...'}
```

This validates the *extraction mechanics*. It does **not** validate the live
Word-Online selectors — only the real app can do that (your job, above).

---

## What you (Brian) must validate against real Word Online

The build is automatable up to a point; these need the authenticated app, which
the agent cannot log into:

1. **The DOM selectors** in `selectors.js` actually match the live editor (the
   single most likely tweak — see the section above).
2. **The floating button is visible** and not hidden/overlapped by Word's own
   chrome; and that it survives Word's frequent DOM re-renders (it re-injects
   every 3 s as a safety net).
3. **The keyboard shortcut** `Ctrl+Shift+G` isn't swallowed by a Word command;
   rebind in `chrome://extensions/shortcuts` if needed.
4. **`all_frames` reaches the editor frame** (Word may iframe the editor; see the
   Frame note).
5. **End-to-end**: select → Ground → the popup shows real papers from *your*
   real index (run `scholia serve` against your real library, not the fake one).

---

## Privacy & integrity

- The extension contacts **only** `http://127.0.0.1:8765` (the local bridge).
  Nothing else.
- It is **display-only**: it renders papers/verdicts the bridge returns; it never
  generates prose and cannot fabricate a citation.
- Your draft never goes to the cloud. Discovery (run by the bridge) sends only a
  short keyword query, never the passage.
