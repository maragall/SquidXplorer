"""The local-vs-http benchmark harness, with no Odon and no real acquisition needed."""

from __future__ import annotations

import json
import urllib.request

import pytest

from squidxplorer import _odon_bench as ob


@pytest.fixture
def plate(tmp_path):
    """A minimal plate-shaped tree: plate.ome.zarr/A/1/0/{zarr.json, level 0 chunks}."""
    field = tmp_path / "plate.ome.zarr" / "A" / "1" / "0"
    (field / "0" / "c" / "0" / "0").mkdir(parents=True)
    (field / "zarr.json").write_text("{}")
    (field / "0" / "zarr.json").write_text("{}")
    for i in range(4):
        (field / "0" / "c" / "0" / "0" / str(i)).write_bytes(bytes(1024 * (i + 1)))
    return tmp_path / "plate.ome.zarr"


def test_chunk_files_skips_metadata(plate):
    field = plate / "A" / "1" / "0"
    files = ob.chunk_files(field, level="0")
    assert len(files) == 4
    assert all(f.name != "zarr.json" for f in files)


def test_chunk_files_respects_the_limit(plate):
    assert len(ob.chunk_files(plate / "A" / "1" / "0", level="0", limit=2)) == 2


def test_serve_directory_serves_and_then_stops(plate):
    with ob.serve_directory(plate.parent) as base_url:
        url = f"{base_url}/plate.ome.zarr/A/1/0/zarr.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200
            assert resp.read() == b"{}"
    with pytest.raises(Exception):
        urllib.request.urlopen(url, timeout=2)


def test_benchmark_transport_reads_the_same_bytes_both_ways(plate):
    field = plate / "A" / "1" / "0"
    with ob.serve_directory(plate.parent) as base_url:
        rows = ob.benchmark_transport(field, base_url, plate, level="0", limit=4, workers=2)
    assert [r.label.split()[0] for r in rows] == ["local", "local", "http", "http"]
    assert all(r.errors == 0 for r in rows)
    assert len({r.bytes for r in rows}) == 1
    assert all(r.n == 4 for r in rows)


def test_transport_result_rates_are_finite_and_empty_is_nan():
    r = ob.TransportResult(label="x", n=2, bytes=2 * 1024 ** 2, wall_ms=1000.0,
                           per_item_ms=(1.0, 3.0))
    assert r.mb_per_s == pytest.approx(2.0)
    assert r.median_ms == pytest.approx(2.0)
    assert r.p95_ms == pytest.approx(3.0)
    import math
    assert math.isnan(ob.TransportResult(label="x").median_ms)


def test_benchmark_odon_skips_cleanly_with_a_reason(plate, monkeypatch):
    def _missing():
        raise FileNotFoundError("odon not found. Looked at $ODON_BIN, PATH, ...")

    monkeypatch.setattr("squidxplorer._odon.find_odon", _missing)
    result = ob.benchmark_odon(plate / "A" / "1" / "0")
    assert result.available is False
    assert "odon not found" in result.reason
    assert result.local_ms == ()
    assert result.local_ok is False


def test_odon_remote_probe_skips_cleanly(monkeypatch):
    monkeypatch.setattr("squidxplorer._odon.find_odon",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("nope")))
    assert ob.odon_remote_probe("http://127.0.0.1:1/x")["available"] is False


def test_run_states_what_did_not_run_when_odon_is_absent(plate, monkeypatch):
    monkeypatch.setattr("squidxplorer._odon.find_odon",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("nope")))
    report = ob.run(plate, limit=4, workers=2, repeats=1)
    assert report.transport, "the transport half must still run without odon"
    joined = " ".join(report.not_measured)
    assert "framerate" in joined
    assert "no odon binary was found" in joined
    text = ob.format_report(report)
    assert "NOT MEASURED" in text and "NOT RUN" in text


def test_format_report_never_invents_an_odon_number(plate, monkeypatch):
    monkeypatch.setattr("squidxplorer._odon.find_odon",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("nope")))
    text = ob.format_report(ob.run(plate, limit=4, workers=2, repeats=1))
    odon_section = text.split("ODON:")[1].split("NOT MEASURED")[0]
    assert "first paint" not in odon_section
    assert "ms" not in odon_section


def test_run_reports_a_refusal_verbatim(plate, monkeypatch):
    monkeypatch.setattr(ob, "_run_odon", lambda binary, target: (
        (0.01, "OK: loaded tile level 4 path '4'", 0) if not str(target).startswith("http")
        else (0.01, 'Error: failed to canonicalize dataset root: "%s"' % target, 1)))
    monkeypatch.setattr("squidxplorer._odon.find_odon", lambda: "/fake/odon")
    report = ob.run(plate, limit=4, workers=2, repeats=2)
    assert report.odon.local_ok is True
    assert report.odon.remote_ok is False
    text = ob.format_report(report)
    assert "REFUSED" in text
    assert "failed to canonicalize dataset root" in text
    assert "cannot open an http-served store" in " ".join(report.not_measured)


def test_format_report_separates_cold_from_warm(plate, monkeypatch):
    calls = {"n": 0}

    def fake(binary, target):
        calls["n"] += 1
        return ((0.2 if calls["n"] == 1 else 0.01), "OK: loaded tile level 4", 0)

    monkeypatch.setattr(ob, "_run_odon", fake)
    monkeypatch.setattr("squidxplorer._odon.find_odon", lambda: "/fake/odon")
    text = ob.format_report(ob.run(plate, limit=4, workers=2, repeats=4))
    assert "cold (run 1)" in text and "warm (median)" in text
    assert "OK: loaded tile" in text


def test_write_json_round_trips(plate, tmp_path, monkeypatch):
    monkeypatch.setattr("squidxplorer._odon.find_odon",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("nope")))
    report = ob.run(plate, limit=4, workers=2, repeats=1)
    payload = json.loads(ob.write_json(report, tmp_path / "r.json").read_text())
    assert payload["odon"]["available"] is False
    assert len(payload["transport"]) == 4
    assert payload["not_measured"]
