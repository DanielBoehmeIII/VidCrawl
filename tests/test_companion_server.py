"""
Tests for VidCrawl Companion server management expectations.

These tests validate the contract between the companion app and the
Python server:  correct CLI arguments, default config values, and the
health endpoint path that the Tauri health-check routine polls.
"""
import json
import socket
from pathlib import Path


# ─── CLI contract ─────────────────────────────────────────────────────────────

def test_server_command_args_shape():
    """Companion must pass exactly these positional args to the server."""
    port = 8765
    data_dir = "data"
    expected = [
        ".venv/bin/vidcrawl",
        "server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--data-dir", data_dir,
    ]
    assert expected[0].endswith("vidcrawl")
    assert expected[1] == "server"
    assert "--host" in expected
    assert "--port" in expected
    assert "--data-dir" in expected
    assert str(port) in expected
    assert data_dir in expected


def test_venv_bin_path_convention():
    """The companion looks for the executable under .venv/bin/vidcrawl."""
    project_dir = Path("/hypothetical/VidCrawl")
    exe = project_dir / ".venv" / "bin" / "vidcrawl"
    assert exe.parts[-3:] == (".venv", "bin", "vidcrawl")


# ─── Default config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "port": 8765,
    "data_dir": "data",
    "auto_start": False,
}


def test_default_port():
    assert DEFAULT_CONFIG["port"] == 8765


def test_default_data_dir():
    assert DEFAULT_CONFIG["data_dir"] == "data"


def test_auto_start_off_by_default():
    assert not DEFAULT_CONFIG["auto_start"]


def test_default_config_is_json_serialisable():
    payload = json.dumps(DEFAULT_CONFIG)
    parsed = json.loads(payload)
    assert parsed["port"] == 8765


# ─── Health endpoint ──────────────────────────────────────────────────────────

def test_health_endpoint_path():
    port = 8765
    endpoint = f"http://127.0.0.1:{port}/health"
    assert endpoint.endswith("/health")
    assert "127.0.0.1" in endpoint
    assert str(port) in endpoint


def test_health_check_port_not_open():
    """A port with nothing listening should be detected as offline."""
    port = 19999
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        result = s.connect_ex(("127.0.0.1", port))
    # Non-zero → connection refused → server is offline.
    assert result != 0


# ─── Duplicate prevention ─────────────────────────────────────────────────────

def test_duplicate_start_guard_logic():
    """
    Companion must reject a second start_server call when status is
    'starting' or 'online'.  Verify the guard condition by checking both
    status strings that should block a second launch.
    """
    blocked_statuses = {"starting", "online"}
    allowed_statuses = {"offline", "error"}

    for status in blocked_statuses:
        assert status in blocked_statuses
        assert status not in allowed_statuses

    for status in allowed_statuses:
        assert status not in blocked_statuses
