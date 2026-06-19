/*
 * selectors.js — Word Online DOM selectors, ISOLATED for easy tweaking.
 * ============================================================================
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Word Online (Word for the web) does NOT render the document as ordinary HTML
 * paragraphs. The visible page is painted on a layered/canvas-ish surface, and
 * the *real* editable text lives behind it in a hidden contenteditable region
 * that Microsoft uses for IME / accessibility / input. Microsoft changes these
 * class names and the DOM structure WITHOUT NOTICE (they are obfuscated build
 * artifacts, e.g. `EditingSurfaceBody`, `Paragraph`, GUID-like classes).
 *
 * Therefore: the selectors below are a BEST-EFFORT STARTING POINT, verified
 * against a local mock, but they ALMOST CERTAINLY need adjustment against the
 * live, authenticated app. They are deliberately quarantined in this one file
 * so you can fix capture by editing ONLY here — never touching content.js.
 *
 * HOW TO FIND THE RIGHT SELECTORS (see browser-extension/README.md for full steps):
 *   1. Open a doc in Word Online, select a sentence.
 *   2. DevTools (F12) → Console → run:
 *        document.getSelection().anchorNode
 *      then walk up `.parentElement` until you hit the editable block. Note its
 *      class / [contenteditable] / role attributes.
 *   3. Add or reorder the matching strings in EDITABLE_REGION_SELECTORS and
 *        PARAGRAPH_SELECTORS below. First match wins.
 *
 * The extraction LOGIC (content.js) is generic and selector-driven: it asks
 * this file "what counts as the editor?" and "what counts as a paragraph?" and
 * never hardcodes a class name itself.
 */

const ScholiaSelectors = {
  /*
   * Candidate selectors for the editable document region, MOST-SPECIFIC FIRST.
   * extractText() uses these to (a) scope the "surrounding paragraph" fallback
   * and (b) the "whole visible editor text" last-ditch fallback.
   *
   * The list mixes Word-Online-specific class fragments with generic
   * contenteditable/role hooks so capture degrades gracefully if Microsoft
   * renames a class. Add the real one you find at the TOP.
   */
  EDITABLE_REGION_SELECTORS: [
    'div.EditingSurfaceBody',          // observed Word-for-web editing surface
    'div[aria-label="Document"]',      // accessibility label on the canvas host
    'div[contenteditable="true"]',     // generic editable container
    'div[role="textbox"]',             // ARIA textbox role
    'div.WACViewPanel',                // Word App Container view panel (older)
    '[contenteditable="true"]'         // last-ditch generic
  ],

  /*
   * Candidate selectors for a single paragraph / text block inside the editor.
   * Used to grab "the block the caret is in" when there is no active selection.
   * Word-for-web has historically used a `.Paragraph` wrapper; the generic
   * fallbacks cover renamed builds and the local mock.
   */
  PARAGRAPH_SELECTORS: [
    'div.Paragraph',                   // observed Word-for-web paragraph block
    'p.Paragraph',
    '[role="paragraph"]',
    'p',                               // generic + mock
    'div'                              // last-ditch
  ]
};

// Expose for both the content script (which loads this file first) and tests.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ScholiaSelectors; // node / unit-test context
}
if (typeof window !== 'undefined') {
  window.ScholiaSelectors = ScholiaSelectors;
}
