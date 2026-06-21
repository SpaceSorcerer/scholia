/* =========================================================
   Scholia ribbon command shim (commands.js)
   =========================================================
   Loaded invisibly by the Office runtime when the ribbon
   button uses an ExecuteFunction action.
   The "Ground" ribbon button opens the task pane directly
   via ShowTaskpane, so this file is a lightweight shim that
   merely initialises Office.js.  It is included to satisfy
   the Office Add-in manifest requirement for a FunctionFile.
   ========================================================= */

"use strict";

Office.onReady(() => {
  // No ribbon function commands are registered yet — the
  // manifest's ribbon button uses ShowTaskpane, not
  // ExecuteFunction.  This shim is here in case a future
  // function command (e.g. a keyboard-shortcut handler) is
  // added without requiring a manifest update to FunctionFile.
});
