import tempfile
from pathlib import Path
from datetime import date
from processing.kb_health_monitor import KBHealthMonitor, HealthStatus


def test_healthy_when_files_present():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        today = date.today().isoformat().replace("-", "")
        (base / "raw").mkdir()
        (base / "raw" / f"observations_{today}_120000.parquet").touch()
        monitor = KBHealthMonitor(data_root=base, today=today)
        status = monitor.check()
        assert status.observations_fresh is True


def test_stale_when_no_files():
    with tempfile.TemporaryDirectory() as td:
        monitor = KBHealthMonitor(data_root=Path(td))
        status = monitor.check()
        assert status.observations_fresh is False
        assert not status.is_healthy
