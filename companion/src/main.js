// Tauri 2.x global API (requires withGlobalTauri: true in tauri.conf.json).
const { invoke } = window.__TAURI__.core;
const { listen }  = window.__TAURI__.event;

let config = null;
let pollTimer = null;
let dashTimer = null;

// Cached data for the dashboard sections.
let allVideos = [];
let libStatusFilter = 'all';

// ─── Initialisation ──────────────────────────────────────────────────────────

async function init() {
  try {
    config = await invoke("load_config");
    applyConfigToForm(config);
  } catch (err) {
    appendLog(`[Error loading config] ${err}`);
    config = { port: 8765, data_dir: "data", project_dir: "", auto_start: false };
  }

  // Stream server stdout/stderr lines as they arrive.
  await listen("server-log", (event) => {
    const line = event.payload;
    appendLog(line, line.startsWith("[ERR]"));
  });

  startPolling();

  if (config.auto_start && config.project_dir) {
    await startServer();
  }
}

// ─── Status polling ──────────────────────────────────────────────────────────

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshStatus, 2000);
  refreshStatus();
}

async function refreshStatus() {
  try {
    const resp = await invoke("get_status");
    applyStatus(resp.status, resp.error, resp.endpoint, resp.port);

    if (resp.status === "starting") {
      const port = config ? config.port : resp.port;
      try {
        const healthy = await invoke("check_health", { port });
        if (healthy) {
          applyStatus("online", null, resp.endpoint, port);
        }
      } catch (_) { /* ignore */ }
    }
  } catch (err) {
    console.error("Status poll failed:", err);
  }
}

// ─── UI helpers ──────────────────────────────────────────────────────────────

function applyStatus(status, error, endpoint, port) {
  const dot      = document.getElementById("status-dot");
  const text     = document.getElementById("status-text");
  const ep       = document.getElementById("endpoint");
  const btnStart = document.getElementById("btn-start");
  const btnStop  = document.getElementById("btn-stop");

  dot.className = `dot ${status}`;
  text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  if (error) text.textContent += ` — ${error}`;

  ep.textContent = endpoint || `http://127.0.0.1:${port}`;

  const running = status === "online" || status === "starting";
  btnStart.disabled = running;
  btnStop.disabled  = !running;

  const apiSections = document.querySelectorAll(".api-section");
  apiSections.forEach(el => {
    el.style.display = status === "online" ? "" : "none";
  });

  if (status === "online") {
    onServerOnline();
  } else {
    onServerOffline();
  }
}

let _wasOnline = false;

function onServerOnline() {
  if (!_wasOnline) {
    _wasOnline = true;
    refreshDashboard();
    loadLibrary();
    loadQueue();
    if (dashTimer) clearInterval(dashTimer);
    dashTimer = setInterval(() => {
      refreshDashboard();
      loadQueue();
    }, 10000);
  }
}

function onServerOffline() {
  _wasOnline = false;
  if (dashTimer) { clearInterval(dashTimer); dashTimer = null; }
}

function applyConfigToForm(cfg) {
  document.getElementById("cfg-port").value        = cfg.port;
  document.getElementById("cfg-data-dir").value    = cfg.data_dir;
  document.getElementById("cfg-project-dir").value = cfg.project_dir;
  document.getElementById("cfg-auto-start").checked = cfg.auto_start;
}

function appendLog(line, isErr = false) {
  const panel = document.getElementById("log-panel");
  const div = document.createElement("div");
  div.className = isErr ? "log-line err" : "log-line";
  const ts = new Date().toLocaleTimeString([], { hour12: false });
  div.textContent = `[${ts}] ${line}`;
  panel.appendChild(div);
  while (panel.children.length > 500) {
    panel.removeChild(panel.firstChild);
  }
  panel.scrollTop = panel.scrollHeight;
}

function clearLogs() {
  document.getElementById("log-panel").innerHTML = "";
}

// ─── Button handlers ─────────────────────────────────────────────────────────

async function startServer() {
  const port       = parseInt(document.getElementById("cfg-port").value, 10);
  const dataDir    = document.getElementById("cfg-data-dir").value.trim();
  const projectDir = document.getElementById("cfg-project-dir").value.trim();

  if (!projectDir) {
    appendLog("[Error] Project Directory is required — set it in Configuration.", true);
    return;
  }

  appendLog(`Starting VidCrawl server on port ${port}…`);
  try {
    await invoke("start_server", { port, dataDir, projectDir });
    applyStatus("starting", null, `http://127.0.0.1:${port}`, port);
  } catch (err) {
    appendLog(`[Error] ${err}`, true);
    applyStatus("error", String(err), `http://127.0.0.1:${port}`, port);
  }
}

async function stopServer() {
  const port = config ? config.port : 8765;
  appendLog("Stopping server…");
  try {
    await invoke("stop_server");
    applyStatus("offline", null, `http://127.0.0.1:${port}`, port);
    appendLog("Server stopped.");
    allVideos = [];
  } catch (err) {
    appendLog(`[Error] ${err}`, true);
  }
}

async function openDataFolder() {
  const dataDir    = document.getElementById("cfg-data-dir").value.trim();
  const projectDir = document.getElementById("cfg-project-dir").value.trim();
  try {
    await invoke("open_data_folder", { projectDir, dataDir });
  } catch (err) {
    appendLog(`[Error] ${err}`, true);
  }
}

async function saveConfig() {
  const port       = parseInt(document.getElementById("cfg-port").value, 10);
  const dataDir    = document.getElementById("cfg-data-dir").value.trim();
  const projectDir = document.getElementById("cfg-project-dir").value.trim();
  const autoStart  = document.getElementById("cfg-auto-start").checked;

  config = { port, data_dir: dataDir, project_dir: projectDir, auto_start: autoStart };

  try {
    await invoke("save_config", { config });
    const msg = document.getElementById("save-msg");
    msg.textContent = "Saved!";
    setTimeout(() => { msg.textContent = ""; }, 2000);
  } catch (err) {
    appendLog(`[Error saving config] ${err}`, true);
  }
}

// ─── API helpers ─────────────────────────────────────────────────────────────

function apiBase() {
  const port = config ? config.port : 8765;
  return `http://127.0.0.1:${port}`;
}

async function apiFetch(path, timeoutMs = 8000) {
  const resp = await fetch(`${apiBase()}${path}`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtDuration(sec) {
  if (!sec && sec !== 0) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function fmtNum(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString();
}

// ─── Dashboard ───────────────────────────────────────────────────────────────

async function refreshDashboard() {
  try {
    const data = await apiFetch('/stats', 5000);
    setText('stat-videos',    fmtNum(data.videos));
    setText('stat-moments',   fmtNum(data.moments));
    setText('stat-ideas',     fmtNum(data.ideas));
    setText('stat-keyframes', fmtNum(data.keyframes));
    setText('stat-fts',       fmtNum(data.fts_rows));
    setText('stat-jobs',      fmtNum(data.jobs));
  } catch { /* silently ignore if server briefly unreachable */ }
}

// ─── Video Library ───────────────────────────────────────────────────────────

async function loadLibrary() {
  try {
    allVideos = await apiFetch('/videos', 10000);
    filterLibrary();
  } catch (err) {
    document.getElementById('library-list').innerHTML =
      `<div class="list-empty">Could not load videos: ${escHtml(err.message)}</div>`;
  }
}

function setLibTab(btn, filter) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  libStatusFilter = filter;
  filterLibrary();
}

function filterLibrary() {
  const q = (document.getElementById('lib-filter')?.value || '').toLowerCase();
  let filtered = libStatusFilter === 'all'
    ? allVideos
    : allVideos.filter(v => {
        if (libStatusFilter === 'ingesting') return v.status === 'ingesting' || v.status === 'pending';
        return v.status === libStatusFilter;
      });
  if (q) filtered = filtered.filter(v => (v.title || '').toLowerCase().includes(q));

  const countEl = document.getElementById('library-count');
  if (countEl) countEl.textContent = filtered.length;

  const listEl = document.getElementById('library-list');
  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="list-empty">No videos found.</div>';
    return;
  }

  listEl.innerHTML = filtered.map(v => `
    <div class="lib-row">
      <div class="lib-row-main">
        <span class="status-dot-sm status-dot-${v.status}"></span>
        <span class="lib-title" title="${escHtml(v.url || '')}">
          ${escHtml(v.title || v.video_id || '—')}
        </span>
      </div>
      <div class="lib-row-meta">
        <span class="status-badge status-${v.status}">${escHtml(v.status)}</span>
        <span class="lib-dur">${fmtDuration(v.duration_sec)}</span>
        ${v.video_id ? `<code class="vid-id">${escHtml(v.video_id.slice(0, 11))}</code>` : ''}
      </div>
      ${v.error_message ? `<div class="lib-err">${escHtml(v.error_message)}</div>` : ''}
    </div>
  `).join('');
}

// ─── Job Queue ───────────────────────────────────────────────────────────────

async function loadQueue() {
  const active = allVideos.length > 0
    ? allVideos.filter(v => v.status === 'pending' || v.status === 'ingesting')
    : [];

  const countEl = document.getElementById('queue-active-count');
  if (countEl) {
    countEl.textContent = active.length > 0 ? active.length : '';
    countEl.style.display = active.length > 0 ? 'inline-flex' : 'none';
  }

  const listEl = document.getElementById('queue-list');
  if (!listEl) return;

  if (active.length === 0) {
    // Show recent non-active videos as history.
    const recent = [...allVideos]
      .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
      .slice(0, 5);

    if (recent.length === 0) {
      listEl.innerHTML = '<div class="list-empty">No ingestion jobs yet.</div>';
      return;
    }
    listEl.innerHTML = `
      <div class="queue-empty-msg">No active jobs. Recent:</div>
      ${recent.map(v => renderQueueRow(v)).join('')}
    `;
    return;
  }

  listEl.innerHTML = active.map(v => renderQueueRow(v)).join('');
}

function renderQueueRow(v) {
  const isActive = v.status === 'ingesting' || v.status === 'pending';
  return `
    <div class="queue-row ${isActive ? 'queue-row-active' : ''}">
      <div class="queue-row-top">
        ${isActive ? '<span class="spinner"></span>' : `<span class="status-dot-sm status-dot-${v.status}"></span>`}
        <span class="queue-title">${escHtml(v.title || v.video_id || '—')}</span>
        <span class="status-badge status-${v.status}">${escHtml(v.status)}</span>
      </div>
      ${v.url ? `<div class="queue-url">${escHtml(v.url)}</div>` : ''}
      ${v.error_message ? `<div class="queue-err">${escHtml(v.error_message)}</div>` : ''}
    </div>
  `;
}

// ─── Search / Retrieval Tester ────────────────────────────────────────────────

async function runSearch() {
  const q       = (document.getElementById('search-q')?.value || '').trim();
  const videoId = (document.getElementById('search-video-id')?.value || '').trim();
  if (!q) return;

  const statusEl  = document.getElementById('search-status');
  const resultsEl = document.getElementById('search-results');
  statusEl.textContent  = 'Searching…';
  resultsEl.innerHTML   = '';

  try {
    let path = `/search?q=${encodeURIComponent(q)}&limit=8`;
    if (videoId) path += `&video_id=${encodeURIComponent(videoId)}`;
    const results = await apiFetch(path, 10000);
    statusEl.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;

    if (results.length === 0) {
      resultsEl.innerHTML = '<div class="list-empty">No moments matched.</div>';
      return;
    }

    resultsEl.innerHTML = results.map(r => `
      <div class="search-result">
        <div class="sr-header">
          <span class="sr-ts">${escHtml(r.timestamp_label || `${r.start_sec?.toFixed(0)}s`)}</span>
          <span class="sr-score">${(r.score ?? 0).toFixed(2)}</span>
          <span class="sr-video">${escHtml(r.video_title || r.video_id || '')}</span>
        </div>
        ${r.transcript_snippet
          ? `<div class="sr-snippet">"${escHtml(r.transcript_snippet)}"</div>` : ''}
        ${r.idea_summary
          ? `<div class="sr-idea">${escHtml(r.idea_summary)}</div>` : ''}
        ${r.match_reasons?.length
          ? `<div class="sr-reasons">${r.match_reasons.map(m => `<span class="reason-tag">${escHtml(m)}</span>`).join('')}</div>`
          : ''}
      </div>
    `).join('');
  } catch (err) {
    statusEl.textContent = '';
    resultsEl.innerHTML = `<div class="list-empty err-text">Error: ${escHtml(err.message)}</div>`;
  }
}

// ─── Boot ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", init);
