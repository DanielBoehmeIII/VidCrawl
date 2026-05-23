# VidCrawl Companion App

VidCrawl Companion is a lightweight desktop app built with [Tauri 2](https://tauri.app) that starts and stops the VidCrawl local API server and shows its status at a glance. ClipBounce connects to this local server instead of a remote service, keeping your video data private and eliminating per-request API costs.

---

## Why a local companion app?

| Concern       | Cloud service                   | Local companion             |
|---------------|---------------------------------|-----------------------------|
| Privacy       | Video metadata sent off-device  | All data stays on your Mac/Linux/Windows machine |
| Cost          | Per-request API fees            | Free — your CPU, your disk  |
| Latency       | Network round-trip on every call | Sub-millisecond localhost   |
| Offline use   | Requires internet               | Works completely offline    |

---

## Architecture

```
ClipBounce (browser extension / Electron app)
    │  HTTP  POST /clipbounce/capture
    ▼
VidCrawl API  http://127.0.0.1:8765
    │  manages
    ▼
VidCrawl Python CLI / SQLite database
```

The Companion app is a thin Tauri shell around the existing Python server. It does **not** bundle Python — you run the server from your development checkout (`.venv/bin/vidcrawl server …`).

---

## Dev setup

### Prerequisites

| Tool | Install |
|------|---------|
| Rust + Cargo | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js ≥18 | via [nvm](https://github.com/nvm-sh/nvm) or system package manager |
| Tauri CLI v2 | `cargo install tauri-cli` (already available once Rust is installed) |
| Python ≥3.11 + venv | already required by VidCrawl |
| **Linux only** | `sudo apt-get install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev` |
| **macOS only** | Xcode Command Line Tools: `xcode-select --install` |

### 1 — Install VidCrawl Python server

```bash
# from the repo root
python -m venv .venv
.venv/bin/pip install -e ".[server]"
# verify
.venv/bin/vidcrawl --version
```

### 2 — Run the companion in dev mode

```bash
cd companion
cargo tauri dev
```

This compiles the Rust layer (~60 s first time, then incremental) and opens a native window. Hot-reload is **not** enabled for the static HTML/JS frontend; edit files and restart `cargo tauri dev` to see changes.

### 3 — Configure the companion

On first launch, open the **Configuration** section:

| Field | Description | Default |
|-------|-------------|---------|
| Port | Port the Python server binds to | `8765` |
| Data Directory | Path to VidCrawl data dir, relative to Project Directory | `data` |
| Project Directory | Absolute path to the VidCrawl repo checkout | `$PWD` at launch |
| Auto-start | Start the server automatically when the companion opens | off |

Click **Save Config** — settings are persisted to the OS config directory:

- **Linux**: `~/.config/com.vidcrawl.companion/config.json`
- **macOS**: `~/Library/Application Support/com.vidcrawl.companion/config.json`
- **Windows**: `%APPDATA%\com.vidcrawl.companion\config.json`

### 4 — Start the server

Click **Start Server**. The companion runs:

```
<project_dir>/.venv/bin/vidcrawl server \
    --host 127.0.0.1 \
    --port <port> \
    --data-dir <data_dir>
```

The status indicator turns yellow (Starting) while the process boots, then green (Online) once `GET /health` returns 200. Server stdout/stderr streams live into the **Server Output** panel.

### 5 — Stop the server

Click **Stop Server**. The companion sends SIGKILL to the child process and waits for it to exit. Closing the companion window also kills the server automatically.

---

## How ClipBounce connects

ClipBounce sends requests to `http://127.0.0.1:8765`. No CORS header is needed for same-machine loopback. Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Server liveness (polled by companion every 2 s) |
| `POST` | `/clipbounce/capture` | Ingest a URL from the browser extension |
| `GET`  | `/search?q=…` | Full-text search |
| `GET`  | `/videos` | List all indexed videos |
| `GET`  | `/stats` | DB counts |

Example (from the browser extension or ClipBounce):

```js
const res = await fetch("http://127.0.0.1:8765/clipbounce/capture", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }),
});
const job = await res.json();
console.log(job.job_id, job.status); // "queued"
```

Poll `GET /jobs/<job_id>` (or re-POST the same URL — the server deduplicates) until `status === "ready"`.

---

## Running the Rust tests

The pure logic lives in `companion/core` (no WebKit/GTK dependency) and runs anywhere:

```bash
cd companion/core
cargo test
```

Expected output:

```
running 6 tests
test tests::default_config_values ... ok
test tests::status_serialization_roundtrip ... ok
test tests::server_state_default_is_offline ... ok
test tests::health_check_refuses_on_closed_port ... ok
test tests::endpoint_format_matches_expected ... ok
test tests::config_json_roundtrip ... ok
```

## Running the Python tests

```bash
# from repo root
.venv/bin/pytest tests/test_companion_server.py -v
```

---

## Production bundle (future milestone)

> Bundling Python is **not** part of this milestone. For a future distributable:
>
> 1. Add a sidecar binary (frozen with PyInstaller or Nuitka) to `src-tauri/binaries/`.
> 2. Reference it in `tauri.conf.json` under `bundle.externalBin`.
> 3. Update `start_server` in `lib.rs` to resolve the sidecar path via `tauri::process::current_binary()`.

For the dev companion, the user is expected to have Python + a virtualenv in the repo directory.

---

## Project layout

```
companion/
├── package.json          # npm scripts (cargo tauri dev / build)
├── src/
│   ├── index.html        # app UI (vanilla HTML)
│   ├── styles.css        # dark-theme styles
│   └── main.js           # Tauri JS API bridge
└── src-tauri/
    ├── Cargo.toml
    ├── build.rs
    ├── tauri.conf.json
    ├── capabilities/
    │   └── default.json  # Tauri 2 permission model
    └── src/
        ├── main.rs       # binary entry point
        └── lib.rs        # commands, state, tests
```
