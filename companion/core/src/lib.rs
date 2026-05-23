//! Pure logic for VidCrawl Companion — no Tauri / GTK dependency.
//! Imported by the Tauri crate; can be tested in CI without system GUI libs.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─── Status ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServerStatus {
    Offline,
    Starting,
    Online,
    Error,
}

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionConfig {
    pub port: u16,
    pub data_dir: String,
    pub project_dir: String,
    pub auto_start: bool,
}

impl Default for CompanionConfig {
    fn default() -> Self {
        let project_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default();
        Self {
            port: 8765,
            data_dir: "data".to_string(),
            project_dir,
            auto_start: false,
        }
    }
}

// ─── Health check (stdlib only) ───────────────────────────────────────────────

/// Returns true when `GET /health` on 127.0.0.1:{port} responds with HTTP 200.
pub fn do_health_check(port: u16) -> bool {
    let addr = format!("127.0.0.1:{}", port);
    let Ok(addr_parsed) = addr.parse::<std::net::SocketAddr>() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr_parsed, Duration::from_secs(1)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let req = "GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut resp = String::new();
    let _ = stream.read_to_string(&mut resp);
    resp.starts_with("HTTP/1.0 200") || resp.starts_with("HTTP/1.1 200")
}

/// Duplicate-start guard: returns true when the current status blocks a new start.
pub fn status_blocks_start(status: &ServerStatus) -> bool {
    matches!(status, ServerStatus::Starting | ServerStatus::Online)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_port() {
        let cfg = CompanionConfig {
            port: 8765,
            data_dir: "data".to_string(),
            project_dir: String::new(),
            auto_start: false,
        };
        assert_eq!(cfg.port, 8765);
        assert_eq!(cfg.data_dir, "data");
        assert!(!cfg.auto_start);
    }

    #[test]
    fn status_serialization_roundtrip() {
        for (variant, json) in [
            (ServerStatus::Offline,  "\"offline\""),
            (ServerStatus::Starting, "\"starting\""),
            (ServerStatus::Online,   "\"online\""),
            (ServerStatus::Error,    "\"error\""),
        ] {
            let serialized = serde_json::to_string(&variant).unwrap();
            assert_eq!(serialized, json);
            let parsed: ServerStatus = serde_json::from_str(json).unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn duplicate_start_guard() {
        assert!(status_blocks_start(&ServerStatus::Starting));
        assert!(status_blocks_start(&ServerStatus::Online));
        assert!(!status_blocks_start(&ServerStatus::Offline));
        assert!(!status_blocks_start(&ServerStatus::Error));
    }

    #[test]
    fn health_check_on_closed_port_returns_false() {
        assert!(!do_health_check(19999));
    }

    #[test]
    fn endpoint_format() {
        let port = 8765u16;
        assert_eq!(format!("http://127.0.0.1:{}", port), "http://127.0.0.1:8765");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = CompanionConfig {
            port: 9000,
            data_dir: "mydata".to_string(),
            project_dir: "/repos/VidCrawl".to_string(),
            auto_start: true,
        };
        let json  = serde_json::to_string(&cfg).unwrap();
        let back: CompanionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.port, 9000);
        assert_eq!(back.data_dir, "mydata");
        assert!(back.auto_start);
    }
}
