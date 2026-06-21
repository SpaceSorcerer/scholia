# Scholia Word Add-in — Sideload Guide

This guide walks you through installing the Scholia add-in in Word (desktop or
online) so the task pane appears inside your document. No app-store account is
needed — this is "sideloading" a local add-in.

**Time:** about 10 minutes on first run (mostly certificate trust).

---

## Prerequisites

1. **Scholia engine installed** — you must have the Python package installed and an
   index built (`scholia index`). If you have not done this, follow `../START_HERE.md`
   first.

2. **Node.js** — already installed at `C:\Program Files\nodejs\` (or available as
   `node` on your PATH). Only needed for certificate trust (one-time).

---

## Step 1 — Start the bridge in add-in mode

Open a terminal and run:

```
python -m scholia serve --serve-addin
```

**What this does:**

- Starts the Scholia JSON API on `https://127.0.0.1:8765`.
- On first run it generates a self-signed localhost certificate and saves it to
  `C:\Users\<you>\.scholia\localhost.crt` (and `localhost.key`).
- Serves the task-pane static files from `word-addin\` at the same HTTPS address.

Leave this terminal window open while you use Word.

You will see output like:

```
Generating self-signed localhost certificate (first-run, stored in C:\Users\...) ...
  Certificate: C:\Users\...\.scholia\localhost.crt
  ...
IMPORTANT: Trust the certificate before sideloading the add-in.
  See word-addin/SIDELOAD_WORD.md for the trust step.
Scholia serving on https://127.0.0.1:8765
  Task pane: https://127.0.0.1:8765/taskpane.html
```

---

## Step 2 — Trust the certificate (one time only)

The self-signed certificate must be added to Windows' Trusted Root store so your
browser and Word will not show a security warning when the task pane loads.

**Double-click `Scholia - Trust Certificate (one time)` on your Desktop** (or run
`python -m scholia trust-cert` in a terminal).

Done — no admin required.  This is a one-time step; you never need to repeat it.

> **Fallback (manual):** If the command fails, open a Run dialog (`Win+R`), type
> `certlm.msc`, navigate to **Trusted Root Certification Authorities → Certificates**,
> right-click → **All Tasks → Import…**, and browse to
> `C:\Users\<your-username>\.scholia\localhost.crt`.

**Verify the trust step worked:**

Open Chrome or Edge and navigate to:
```
https://127.0.0.1:8765/health
```
You should see `{"status":"ok","papers":361,...}` with no security warning. If you
still see a warning, the cert is not trusted — repeat Step 2.

---

## Step 3 — Sideload the add-in in Word Desktop

This is the simplest reliable path (Shared Folder Trusted Catalog).

1. Create a folder somewhere on your computer, for example:
   ```
   C:\Scholia-Addin\
   ```

2. Copy `manifest.xml` from this `word-addin\` directory into that folder.

3. Open **Word** (desktop).

4. Go to **File → Options → Trust Center → Trust Center Settings…**

5. Click **Trusted Add-in Catalogs** on the left.

6. In the "Catalog URL" box type:
   ```
   C:\Scholia-Addin\
   ```
   Click **Add catalog**. The path appears in the list.

7. Check the **"Show in Menu"** checkbox next to it.

8. Click **OK**, then **OK** again.

9. **Restart Word** (the catalog is only read at startup).

10. In Word, go to **Insert → Get Add-ins → My Add-ins → Shared Folder**.

11. You should see **Scholia** in the list. Click **Add**.

The Scholia task pane will open on the right side of your document with a
"Ground selection" button.

---

## Step 4 — Sideload the add-in in Word Online

Word Online has a simpler upload path:

1. Open your document in **Word Online** (office.com or SharePoint).

2. Go to **Insert → Add-ins → Upload My Add-in**.
   (If you do not see "Upload My Add-in", your organisation may have disabled
   custom add-ins — contact your IT admin.)

3. Click **Browse**, navigate to this `word-addin\` folder, and select
   `manifest.xml`.

4. Click **Upload**.

The Scholia pane opens immediately in the right panel.

> Note: Word Online must be able to reach `https://127.0.0.1:8765` from your
> browser tab. This works because the bridge is on your local machine and the add-in
> runs in your browser — same machine, same network interface. It will NOT work if
> you open the document on another device.

---

## Using the add-in

1. Select a passage in your document (one or two sentences).
2. Click **Ground selection** in the Scholia pane.
3. The pane shows:
   - A **SUPPORTED / Not clearly supported** verdict.
   - A ranked list of supporting papers from your Zotero library, with clickable
     DOI and "Open in Zotero" links.
4. Click **Discover** to find papers NOT yet in your library that are relevant
   to the selected passage (keyword-only query; nothing else leaves your machine).

**The add-in never rewrites or inserts text into your document.** It is
display-only — a research assistant, not a ghostwriter.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pane is blank / shows a grey screen | The bridge is not running. Run `python -m scholia serve --serve-addin` in a terminal. |
| "Your connection is not private" / cert error in the pane | The localhost certificate is not trusted. Repeat Step 2. Then restart Word. |
| "Engine offline" in the footer | The bridge stopped. Restart it (`python -m scholia serve --serve-addin`). |
| "No text selected" message | Select text in the document before clicking Ground. |
| Scholia not visible in Insert → My Add-ins | Restart Word after adding the Shared Folder Catalog (Step 3, item 9). |
| Word Online: "Upload My Add-in" is greyed out | Your org policy blocks custom add-ins. Use the browser extension instead (see `../browser-extension/README.md`). |

---

## Security note

The localhost certificate is self-signed and valid for 825 days. It is stored in
`~/.scholia/` and never committed to git (listed in `.gitignore`). It covers only
`127.0.0.1` and `localhost` — it cannot be used to impersonate any real website.
Trusting it grants no additional privileges beyond letting your own local software
serve HTTPS to your own browser.

When the certificate expires, delete `~/.scholia/localhost.crt` and `localhost.key`
and run `python -m scholia serve --serve-addin` again to regenerate.
