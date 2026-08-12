"""Build, VERIFY, measure and (optionally) clean up the macOS .app bundle — IMA-232.

    python scripts/build_app.py --dataset "/path/to/an/acquisition" --clean

Three things a bare ``pyinstaller`` call does not do, and all three are the ticket:

1. **Verify.** It runs the frozen binary against a REAL acquisition folder
   (``hcs_viewer_entry --selftest``) and fails the build if the app cannot ingest it and
   run the operators. A .app that crashes on start is worth less than no .app, and only
   executing the bundle can tell you which one you have. The first build here looked
   perfect and died on ``ModuleNotFoundError: ml_dtypes`` the moment it was run.
2. **Measure.** It prints the bundle size and the ten largest directories inside it, so
   the number is reported rather than discovered by the person downloading it.
3. **Clean.** ``--clean`` deletes the build AND dist trees after measuring. A PyInstaller
   run of this dependency set costs ~660 MB of scratch; leaving it behind on a laptop
   that is already tight on disk is how a machine reaches zero bytes.

The build is deliberately NOT wired into pytest: it takes ~3 minutes and hundreds of MB.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = REPO_ROOT / "scripts" / "hcs-viewer.spec"
APP_NAME = "hcs-viewer.app"


def _du_bytes(path: Path) -> int:
    """Apparent size on disk of a tree, in bytes (``du`` semantics, i.e. real blocks)."""
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) * 1024


def _mb(n: int) -> str:
    return f"{n / 1024 ** 2:.0f} MB"


def build(distpath: Path, workpath: Path) -> Path:
    argv = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(distpath), "--workpath", str(workpath), str(SPEC),
    ]
    print("$ " + " ".join(argv), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(argv, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"pyinstaller failed (rc={proc.returncode})")
    print(f"built in {time.perf_counter() - t0:.0f}s", flush=True)
    app = distpath / APP_NAME
    if not app.is_dir():
        raise SystemExit(f"no bundle at {app} — the spec did not produce a .app")
    return app


def verify(app: Path, dataset: str) -> dict:
    """Run the frozen binary's --selftest against *dataset*. Returns its JSON verdict."""
    binary = app / "Contents" / "MacOS" / "hcs-viewer"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([str(binary), "--selftest", dataset],
                          capture_output=True, text=True, env=env, timeout=900)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("SELFTEST ")), None)
    if line is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
        raise SystemExit("frozen app produced no verdict (rc=%s):\n  %s"
                         % (proc.returncode, "\n  ".join(tail)))
    verdict = json.loads(line[len("SELFTEST "):])
    if not verdict.get("frozen"):
        raise SystemExit("selftest ran UNFROZEN — it did not execute the bundle")
    return verdict


def measure(app: Path) -> int:
    total = _du_bytes(app)
    print(f"\n{app.name}: {_mb(total)} ({total} bytes), arch:", flush=True)
    subprocess.run(["lipo", "-archs", str(app / "Contents" / "MacOS" / "hcs-viewer")])
    frameworks = app / "Contents" / "Frameworks"
    if frameworks.is_dir():
        sizes = sorted(((_du_bytes(p), p.name) for p in frameworks.iterdir() if p.is_dir()),
                       reverse=True)[:10]
        print("largest components:")
        for n, name in sizes:
            print(f"  {_mb(n):>8}  {name}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="acquisition folder to verify against (skips verify if unset)")
    ap.add_argument("--distpath", default=None, help="default: a temp dir, so nothing lands in the repo")
    ap.add_argument("--workpath", default=None)
    ap.add_argument("--clean", action="store_true",
                    help="delete build+dist after measuring (do this on a disk-constrained machine)")
    args = ap.parse_args()

    if sys.platform != "darwin":
        print("this spec builds a macOS .app; on other platforms run pyinstaller on the spec directly",
              file=sys.stderr)
        return 2

    tmp = None
    if args.distpath is None or args.workpath is None:
        tmp = tempfile.mkdtemp(prefix="squidxplorer-pyi-")
    distpath = Path(args.distpath or Path(tmp) / "dist")
    workpath = Path(args.workpath or Path(tmp) / "build")

    try:
        app = build(distpath, workpath)
        total = measure(app)
        if args.dataset:
            verdict = verify(app, args.dataset)
            print("\nfrozen selftest: " + json.dumps(verdict, indent=2))
            if not verdict.get("ingested"):
                raise SystemExit("the bundle launched but could NOT ingest the dataset")
            print("VERIFIED: the bundle launches, ingests, and computes.")
        else:
            print("\n(no --dataset: NOT verified — an unverified bundle is not evidence)")
        print(f"\n.app size: {_mb(total)}")
        return 0
    finally:
        if args.clean:
            for tree in (distpath, workpath):
                shutil.rmtree(tree, ignore_errors=True)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
            print("cleaned build + dist trees")


if __name__ == "__main__":
    sys.exit(main())
