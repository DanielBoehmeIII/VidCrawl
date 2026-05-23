use std::process::Child;
use std::sync::Mutex;

use tauri::Manager;

pub use vidcrawl_companion_core::{
    do_health_check, status_blocks_start, CompanionConfig, ServerStatus,
};

// ─── Runtime state (Tauri managed) ───────────────────────────────────────────

pub struct ServerState {
    child: Option<Child>,
    pub status: ServerStatus,
    pub error: Option<String>,
    pub port: u16,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            child: None,
            status: ServerStatus::Offline,
            error: None,
            port: 8765,
        }
    }
}

// ─── Response type ────────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct StatusResponse {
    pub status: ServerStatus,
    pub error: Option<String>,
    pub endpoint: String,
    pub port: u16,
}

// ─── Tauri commands (isolated module to avoid #[macro_export] name collisions) ─

mod commands {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::sync::Mutex;
    use std::thread;
    use tauri::{AppHandle, Emitter, Manager, State};
    use vidcrawl_companion_core::{do_health_check, status_blocks_start};

    use crate::{CompanionConfig, ServerState, ServerStatus, StatusResponse};

    fn config_path(app: &AppHandle) -> std::path::PathBuf {
        app.path()
            .app_config_dir()
            .expect("cannot resolve app config dir")
            .join("config.json")
    }

    #[tauri::command]
    pub fn get_status(state: State<'_, Mutex<ServerState>>) -> StatusResponse {
        let mut s = state.lock().unwrap();
        if let Some(ref mut child) = s.child {
            if let Ok(Some(exit_status)) = child.try_wait() {
                let code = exit_status.code().unwrap_or(-1);
                s.status = ServerStatus::Error;
                s.error = Some(format!("Server exited unexpectedly (code {})", code));
                s.child = None;
            }
        }
        let port = s.port;
        StatusResponse {
            status: s.status.clone(),
            error: s.error.clone(),
            endpoint: format!("http://127.0.0.1:{}", port),
            port,
        }
    }

    #[tauri::command]
    pub fn start_server(
        app: AppHandle,
        state: State<'_, Mutex<ServerState>>,
        port: u16,
        data_dir: String,
        project_dir: String,
    ) -> Result<(), String> {
        {
            let s = state.lock().unwrap();
            if status_blocks_start(&s.status) {
                return Err("Server is already running".to_string());
            }
        }

        let venv_bin = std::path::Path::new(&project_dir)
            .join(".venv")
            .join("bin")
            .join("vidcrawl");

        if !venv_bin.exists() {
            return Err(format!(
                "vidcrawl not found at {}. Run: pip install -e .[server]",
                venv_bin.display()
            ));
        }

        let mut child = Command::new(&venv_bin)
            .args([
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--data-dir",
                &data_dir,
            ])
            .current_dir(&project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn server: {}", e))?;

        let stdout = child.stdout.take().expect("stdout not captured");
        let stderr = child.stderr.take().expect("stderr not captured");

        let app1 = app.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().flatten() {
                let _ = app1.emit("server-log", &line);
            }
        });

        let app2 = app.clone();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                let _ = app2.emit("server-log", format!("[ERR] {}", line));
            }
        });

        {
            let mut s = state.lock().unwrap();
            s.child = Some(child);
            s.status = ServerStatus::Starting;
            s.error = None;
            s.port = port;
        }

        Ok(())
    }

    #[tauri::command]
    pub fn stop_server(state: State<'_, Mutex<ServerState>>) -> Result<(), String> {
        let mut s = state.lock().unwrap();
        if let Some(mut child) = s.child.take() {
            child
                .kill()
                .map_err(|e| format!("Failed to stop server: {}", e))?;
            let _ = child.wait();
        }
        s.status = ServerStatus::Offline;
        s.error = None;
        Ok(())
    }

    #[tauri::command]
    pub fn check_health(state: State<'_, Mutex<ServerState>>, port: u16) -> bool {
        let healthy = do_health_check(port);
        if healthy {
            let mut s = state.lock().unwrap();
            if s.status == ServerStatus::Starting {
                s.status = ServerStatus::Online;
            }
        }
        healthy
    }

    #[tauri::command]
    pub fn open_data_folder(project_dir: String, data_dir: String) -> Result<(), String> {
        let base = if project_dir.is_empty() {
            std::env::current_dir().map_err(|e| e.to_string())?
        } else {
            std::path::PathBuf::from(&project_dir)
        };
        let full = base.join(&data_dir);
        if !full.exists() {
            return Err(format!("Data directory not found: {}", full.display()));
        }

        #[cfg(target_os = "linux")]
        Command::new("xdg-open")
            .arg(full.as_os_str())
            .spawn()
            .map_err(|e| format!("xdg-open failed: {}", e))?;

        #[cfg(target_os = "macos")]
        Command::new("open")
            .arg(full.as_os_str())
            .spawn()
            .map_err(|e| format!("open failed: {}", e))?;

        #[cfg(target_os = "windows")]
        Command::new("explorer")
            .arg(full.as_os_str())
            .spawn()
            .map_err(|e| format!("explorer failed: {}", e))?;

        Ok(())
    }

    #[tauri::command]
    pub fn load_config(app: AppHandle) -> Result<CompanionConfig, String> {
        let path = config_path(&app);
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("Cannot read config: {}", e))?;
            serde_json::from_str(&text).map_err(|e| format!("Cannot parse config: {}", e))
        } else {
            Ok(CompanionConfig::default())
        }
    }

    #[tauri::command]
    pub fn save_config(app: AppHandle, config: CompanionConfig) -> Result<(), String> {
        let path = config_path(&app);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create config dir: {}", e))?;
        }
        let text = serde_json::to_string_pretty(&config)
            .map_err(|e| format!("Cannot serialize config: {}", e))?;
        std::fs::write(&path, text).map_err(|e| format!("Cannot write config: {}", e))
    }
}

// ─── App entry point ──────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(ServerState::default()))
        .invoke_handler(tauri::generate_handler![
            commands::get_status,
            commands::start_server,
            commands::stop_server,
            commands::check_health,
            commands::open_data_folder,
            commands::load_config,
            commands::save_config,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<Mutex<ServerState>>();
                let mut s = state.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(mut child) = s.child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                s.status = ServerStatus::Offline;
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
