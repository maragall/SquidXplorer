# scripts

Utility scripts for the SquidMIP / SquidXplorer repo.

## Building the desktop app (macOS)

```sh
conda create -n squidbuild python=3.11 && conda activate squidbuild
pip install ".[gui]" pyinstaller
python scripts/build_app.py --dataset /path/to/an/acquisition
```

`build_app.py` freezes `hcs-viewer.spec`, then **runs the frozen binary** against a real
acquisition (`hcs_viewer_entry.py --selftest`) and prints a JSON verdict. Use `--dataset`; without
it nothing is verified, and an unverified bundle is not evidence. Add `--clean` to delete the
~660 MB of build scratch afterwards.

Measured on an M-series Mac, 2026-08-05: **284 MB**, `arm64`, ~4.5 min. Largest components are
PyQt6 (51 MB), tensorstore (47 MB), scipy (35 MB).

Two things to know before you trust a build:

* **`"ingested": false` does not necessarily mean the app is broken.** The selftest ANDs ingestion
  with a compute check that runs `bgsub` **and `decon`**, and `decon` needs `petakit`, which is an
  optional extra (`pip install ".[gui,decon]"`). With petakit absent the verdict is `false` while
  every other field shows the plate read correctly. Read `compute.error` before concluding
  anything.
* **The selftest is headless and cannot see the renderer.** It exercises the reader and the
  operators, not napari. A bundle can pass `--selftest` and still show
  "napari viewer unavailable" in every view window — that exact bug shipped in the first build
  and is why `napari` is now in the spec's `collect_all` list. **Open a view window before you
  call a build good.**

Signing: see [SIGNING.md](../SIGNING.md). The bundle is ad-hoc signed automatically and Gatekeeper
still rejects it; fixing that needs an Apple Developer ID.

## Git sync watchdog

`git-sync-watchdog.sh` keeps your local checkout in sync with the shared branch
that Julio and Spencer both push to, **without ever auto-resolving a conflict**.

### What it does

Every `WATCH_INTERVAL` seconds (default 60) it:

1. Runs `git fetch` (a network failure is logged and skipped, never fatal).
2. Compares your local branch to its tracked upstream:
   - **Up to date** — does nothing.
   - **Upstream is ahead only** (you have no unpushed commits) — runs
     `git pull --ff-only` and posts a macOS notification
     "pulled N commits from <branch>".
   - **Upstream ahead but your working tree is dirty** — does nothing
     destructive; notifies "remote advanced but working tree dirty —
     commit/stash then pull".
   - **Diverged** (you have local commits the remote doesn't AND the remote has
     commits you don't) — **does NOT merge or rebase**. It posts a macOS
     notification and prints a stderr line naming the branch and both SHAs so a
     human resolves it.

**The promise:** it only ever *fast-forwards*. It never merges, never rebases,
never force-anythings, and never touches your working tree while you have
uncommitted changes. Anything that could conflict is surfaced to you, not
performed.

### Run it manually

```sh
bash scripts/git-sync-watchdog.sh
```

Single iteration (useful for testing / a cron one-shot), then exit:

```sh
WATCH_ONCE=1 bash scripts/git-sync-watchdog.sh
```

Custom interval (e.g. every 30s):

```sh
WATCH_INTERVAL=30 bash scripts/git-sync-watchdog.sh
```

Safe to Ctrl-C at any time.

### Install as a launchd agent (runs at login, kept alive)

```sh
cp scripts/com.cephla.squidxplorer.gitsync.plist ~/Library/LaunchAgents/
# Edit the copied file and replace the two placeholders:
#   __REPO_PATH__    -> the absolute path to this repo
#   __SCRIPT_PATH__  -> __REPO_PATH__/scripts/git-sync-watchdog.sh
launchctl load ~/Library/LaunchAgents/com.cephla.squidxplorer.gitsync.plist
```

Check status / logs:

```sh
launchctl list | grep com.cephla.squidxplorer.gitsync
tail -f /tmp/gitsync-watchdog.err.log
```

### Uninstall

```sh
launchctl unload ~/Library/LaunchAgents/com.cephla.squidxplorer.gitsync.plist
rm ~/Library/LaunchAgents/com.cephla.squidxplorer.gitsync.plist
```
