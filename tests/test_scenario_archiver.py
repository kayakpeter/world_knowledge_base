import json
from datetime import datetime, timezone
from processing.scenario_archiver import is_expired, archive_expired_scenarios

NOW = datetime(2026, 5, 14, tzinfo=timezone.utc)

def _write(path, expires_at):
    path.write_text(json.dumps({
        "scenario_id": path.stem, "trigger_event": "test",
        "expires_at": expires_at, "status": "complete",
    }))

def test_is_expired_true_for_past():
    assert is_expired({"expires_at": "2026-04-01T00:00:00+00:00"}, NOW) is True

def test_is_expired_false_for_future():
    assert is_expired({"expires_at": "2026-06-01T00:00:00+00:00"}, NOW) is False

def test_is_expired_false_when_missing():
    assert is_expired({"scenario_id": "x"}, NOW) is False

def test_archive_moves_only_expired(tmp_path):
    active = tmp_path / "active"; archive = tmp_path / "archive"
    active.mkdir()
    _write(active / "old.json", "2026-04-01T00:00:00+00:00")
    _write(active / "fresh.json", "2026-06-01T00:00:00+00:00")
    moved = archive_expired_scenarios(active_dir=active, archive_dir=archive, now=NOW)
    assert moved == ["old.json"]
    assert (archive / "old.json").exists()
    assert (active / "fresh.json").exists()
    assert not (active / "old.json").exists()

def test_archive_skips_unparseable(tmp_path):
    active = tmp_path / "active"; archive = tmp_path / "archive"
    active.mkdir()
    (active / "bad.json").write_text("{not json")
    moved = archive_expired_scenarios(active_dir=active, archive_dir=archive, now=NOW)
    assert moved == []
    assert (active / "bad.json").exists()

def test_archive_creates_archive_dir(tmp_path):
    active = tmp_path / "active"; active.mkdir()
    archive = tmp_path / "archive"
    _write(active / "old.json", "2026-04-01T00:00:00+00:00")
    archive_expired_scenarios(active_dir=active, archive_dir=archive, now=NOW)
    assert archive.is_dir()
