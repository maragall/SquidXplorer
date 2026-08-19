"""What one run WAS: the provenance record written beside every save.

Colin's cheapest win from the operator-model alignment doc: a save that cannot say what run
produced it is unreproducible by construction. ``RunSpec`` captures the run's identity (operator,
kwargs, scope, the reader's ``source_id``) and its environment (squidxplorer version, the git sha
when running from a checkout, key dependency versions, a timestamp); ``write_runspec`` lands it
as ``runspec.json`` at the output root of every save, and only there — never on a preview, never
into the SOURCE acquisition, and a write failure is a logged warning, never the run's.

Input hashing is explicitly OUT: whether a multi-hundred-GB plate's provenance should hash the
acquisition's manifest files or the TIFF bytes too is Julio's open question in the alignment doc
(`AI-docs/SquidXplorer/to-do/2026-08-19-operator-model-alignment-ian-colin.md`), not a default
this module gets to pick.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

#: The file a save leaves at its output root.
RUNSPEC_NAME = "runspec.json"

#: Dependencies whose versions can change a run's pixels; tilefusion rides only when installed.
_KEY_DEPENDENCIES = ("numpy", "tifffile", "zarr", "tilefusion")


def _package_version() -> str:
    """The installed distribution's version; the package's own ``__version__`` as fallback."""
    try:
        from importlib.metadata import version

        return version("squidxplorer")
    except Exception:
        import squidxplorer

        return squidxplorer.__version__


def _git_sha() -> Optional[str]:
    """HEAD's sha when the package runs from a checkout; None otherwise (a wheel has no sha)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    sha = (out.stdout or "").strip()
    return sha if out.returncode == 0 and sha else None


def _dependency_versions() -> dict:
    """``{name: version}`` for the key dependencies; an absent optional is honestly absent."""
    from importlib.metadata import version

    found = {}
    for name in _KEY_DEPENDENCIES:
        try:
            found[name] = version(name)
        except Exception:
            continue
    return found


@dataclass(frozen=True)
class RunSpec:
    """One run's identity and environment, frozen at capture time."""

    operator: str
    #: The one kwargs dict BOTH dispatch arms ran with (the copy arm's ``copy=True`` included).
    operator_kwargs: Optional[dict]
    #: The run's scope — ``{region: [fov, ...]}``, a region list, or None (the whole plate).
    regions: Optional[object]
    #: The per-region crop, ``None``, or the ``N_FOVS_LOOP_DEFAULT`` sentinel (serialized by name).
    n_fovs: object
    #: The reader's declared identity (``contract.reader``); None for an in-memory reader.
    source_id: Optional[str]
    squidxplorer_version: str
    #: None on an installed wheel, and the file says so.
    git_sha: Optional[str]
    dependencies: dict
    timestamp: str

    @classmethod
    def capture(cls, reader, *, operator: str, operator_kwargs: Optional[dict] = None,
                regions=None, n_fovs=None) -> "RunSpec":
        """The spec of the run about to happen, environment measured now."""
        src = getattr(reader, "source_id", None)
        return cls(
            operator=str(operator),
            operator_kwargs=dict(operator_kwargs) if operator_kwargs else None,
            regions=regions,
            n_fovs=n_fovs,
            source_id=str(src) if src else None,
            squidxplorer_version=_package_version(),
            git_sha=_git_sha(),
            dependencies=_dependency_versions(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self) -> str:
        """Deterministic for one instance: sorted keys, non-JSON values by their repr."""
        return json.dumps(asdict(self), sort_keys=True, indent=2, default=str)


def write_runspec(spec: RunSpec, out_root, *, result: Optional[dict] = None) -> Optional[Path]:
    """Write ``runspec.json`` at *out_root*; a failure is a WARNING and None, never the run's.

    *result* is the writer manifest's outcome facts — the alignment doc asks the record to say
    what the run CAME TO, not only what was asked — read here, never re-derived.
    """
    if spec.source_id and Path(str(out_root)).resolve() == Path(spec.source_id).resolve():
        _log.warning("runspec: refusing to write %s into the source acquisition %s.",
                     RUNSPEC_NAME, out_root)
        return None
    path = Path(out_root) / RUNSPEC_NAME
    record = json.loads(spec.to_json())
    if result is not None:
        record["result"] = {k: result.get(k)
                            for k in ("n_fields_written", "complete", "stopped")}
    try:
        path.write_text(json.dumps(record, sort_keys=True, indent=2, default=str))
    except Exception as exc:
        _log.warning("runspec: could not write %s (%s); the save itself is unaffected.",
                     path, exc)
        return None
    return path
