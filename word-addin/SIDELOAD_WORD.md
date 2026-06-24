# Scholia Word Add-in — Sideload Guide

This guide walks you through getting the Scholia pane to appear inside desktop Word.
No app-store account is needed, no admin rights are required, and nothing gets uploaded
anywhere — this is "sideloading" a local add-in via the Windows registry.

**Time:** 3–5 minutes on first run (mostly certificate trust).

---

## Prerequisites

1. **Scholia engine installed** — you must have the Python package installed and an
   index built (`scholia index`). If you have not done this, follow `../START_HERE.md`
   first.

2. **Node.js** — already installed (available as `npx` on your PATH). Used by the
   one-click registration helper.

---

## Step 1 — Start the bridge in add-in mode

Double-click **`Scholia for Word.bat`** on your Desktop (or run
`launchers\Scholia - Start Bridge (Word Addin).bat`). Keep that window open.

What it does:

- Starts the Scholia JSON API on `https://127.0.0.1:8765`.
- On first run, generates a self-signed localhost certificate in
  `C:\Users\<you>\.scholia\` and installs it into the Windows Trusted Root store
  (no admin required — CurrentUser only).
- Serves the task-pane files so Office can load the pane.

You will see output like:

```
Scholia serving on https://127.0.0.1:8765
  Task pane: https://127.0.0.1:8765/taskpane.html
```

**Verify:** Open Edge or Chrome and go to `https://127.0.0.1:8765/health`. You should
see `{"status":"ok",...}` with no security warning. If you see a certificate warning,
the cert is not trusted — run `launchers\Scholia - Trust Certificate (one time).bat`.

---

## Step 2 — Register the add-in in Windows (developer registry method)

This is the **primary desktop-Word path** — it requires no admin, no network share,
and no Upload button. It works by writing one registry entry under your own user
account (`HKCU`) that tells Office "load this manifest for development."

**Double-click `Add Scholia to Word.lnk` on your Desktop.**

(The shortcut runs `launchers\Add Scholia to Word.bat`, which calls
`npx office-addin-dev-settings register` with the manifest path. If Node/npx is
unavailable, the bat falls back to a direct `reg add` command — same result either
way.)

What gets written to the registry:

```
Key:   HKEY_CURRENT_USER\Software\Microsoft\Office\16.0\WEF\Developer
Name:  3301E123-D634-449E-A651-5370FDEBF7F3
Data:  E:\Claude\scholia\word-addin\manifest.xml
```

You only need to do this **once** — or re-run it if Office repairs itself and clears
developer settings (this can happen after a major Office update).

To verify the registration worked, open PowerShell and run:

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Office\16.0\WEF\Developer"
```

You should see the manifest path listed.

---

## Step 3 — Open Word and find the Scholia pane

1. **Fully close Word** (all windows) after registering. Developer add-ins are only
   read at Word startup.

2. Reopen your document.

3. **Where to find Scholia** — look in these places in order:

   - **Home tab → "Scholia" button in the ribbon** (most likely on the far right of
     the Home tab, in a "Scholia" group). Clicking it opens the task pane.
   - **Insert → Add-ins → My Add-ins → Developer Add-ins tab**. Scholia should appear
     there; click Add.
   - If neither shows it, try Insert → Get Add-ins → My Add-ins → Developer Add-ins.

4. The Scholia task pane opens on the right side of your document. The bridge must be
   running (Step 1) for the pane to load content — if the bridge is stopped, the pane
   shows "Engine offline."

---

## Using the add-in

1. Select a passage in your document (one or two sentences).
2. Click **Ground** in the Scholia pane.
3. The pane shows:
   - A **SUPPORTED / Not clearly supported** verdict.
   - A ranked list of supporting papers from your Zotero library, with clickable
     DOI and "Open in Zotero" links.
4. Click **Discover** to find papers not yet in your library that are relevant to the
   selected passage (keywords only — nothing else leaves your machine).

**The add-in never rewrites or inserts text into your document.** Display-only.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pane is blank / grey | Bridge not running. Double-click "Scholia for Word.bat". |
| "Your connection is not private" / cert error | Cert not trusted. Run `Scholia - Trust Certificate (one time).bat`, then restart Word. |
| "Engine offline" in the footer | Bridge stopped. Restart it. |
| "No text selected" | Select text before clicking Ground. |
| Scholia not visible in ribbon or My Add-ins | Re-run `Add Scholia to Word.bat`, then fully close and reopen Word. |
| Word Online: "Upload My Add-in" is greyed out | Org policy blocks custom add-ins. Use the browser extension instead (`../browser-extension/README.md`). |

---

## Re-running after an Office update

If Word updates and the add-in disappears: the bridge registration persists in the
registry, but double-check by running `Add Scholia to Word.bat` again. It is
idempotent — safe to run any time.

---

## Fallback: Trusted Catalog / UNC Share (older method)

If the registry method does not work for some reason (rare; can happen with heavily
managed enterprise Office builds), you can fall back to the Shared Folder Trusted
Catalog approach:

1. Create a local folder, e.g. `C:\Scholia-Addin\`, and copy `manifest.xml` into it.
2. In Word: **File → Options → Trust Center → Trust Center Settings → Trusted Add-in
   Catalogs**. Add `C:\Scholia-Addin\` as a catalog, check "Show in Menu", click OK.
3. Restart Word, then **Insert → Add-ins → My Add-ins → Shared Folder**.

This requires no admin but is more steps and resets on every Word repair.

---

## Security note

The localhost certificate is self-signed and valid for 825 days. It covers only
`127.0.0.1` and `localhost` — it cannot impersonate any real website. It is stored in
`~/.scholia/` and never committed to git. When it expires, delete
`~/.scholia/localhost.crt` and `localhost.key` and run the bridge again to regenerate.
