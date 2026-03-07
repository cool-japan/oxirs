//! File-based audit log backend with rotation support.
//!
//! Writes audit records in JSONL format, rotating by daily schedule or max size.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{Local, Utc};
// serde::Serialize unused - removed
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::Mutex;

use super::AuditRecord;

/// Errors specific to the file audit backend.
#[derive(Debug, thiserror::Error)]
pub enum FileBackendError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Log directory does not exist: {0}")]
    DirectoryNotFound(PathBuf),
}

/// Result type for file backend operations.
pub type FileBackendResult<T> = Result<T, FileBackendError>;

/// Configuration for file-based audit logging.
#[derive(Debug, Clone)]
pub struct FileBackendConfig {
    /// Directory where audit log files are stored.
    pub log_dir: PathBuf,
    /// Base filename (without extension). Date will be appended when rotating.
    pub base_filename: String,
    /// Maximum file size in bytes before rotation (default: 100 MB).
    pub max_file_size_bytes: u64,
    /// Whether to rotate daily regardless of file size.
    pub rotate_daily: bool,
    /// Number of rotated log files to retain (0 = unlimited).
    pub retain_files: usize,
}

impl Default for FileBackendConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("/var/log/oxirs"),
            base_filename: "audit".to_string(),
            max_file_size_bytes: 100 * 1024 * 1024, // 100 MB
            rotate_daily: true,
            retain_files: 30,
        }
    }
}

impl FileBackendConfig {
    /// Create a new config with the given log directory.
    pub fn new(log_dir: impl Into<PathBuf>) -> Self {
        Self {
            log_dir: log_dir.into(),
            ..Default::default()
        }
    }
}

/// Internal state protected by a mutex.
struct FileState {
    writer: BufWriter<File>,
    current_path: PathBuf,
    current_date: String,
    bytes_written: u64,
}

impl FileState {
    async fn new(path: PathBuf) -> FileBackendResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        let bytes_written = file.metadata().await?.len();
        let today = Local::now().format("%Y-%m-%d").to_string();
        Ok(Self {
            writer: BufWriter::new(file),
            current_path: path,
            current_date: today,
            bytes_written,
        })
    }
}

/// File-based audit log backend.
///
/// Writes `AuditRecord` values as newline-delimited JSON (JSONL).
/// Supports daily rotation and size-based rotation.
pub struct FileAuditBackend {
    config: FileBackendConfig,
    state: Arc<Mutex<Option<FileState>>>,
}

impl FileAuditBackend {
    /// Create a new backend with the provided configuration.
    pub async fn new(config: FileBackendConfig) -> FileBackendResult<Self> {
        if !config.log_dir.exists() {
            tokio::fs::create_dir_all(&config.log_dir).await?;
        }
        let backend = Self {
            config,
            state: Arc::new(Mutex::new(None)),
        };
        // Eagerly open the current log file.
        backend.ensure_open().await?;
        Ok(backend)
    }

    /// Derive the current log file path (may include today's date when rotating daily).
    fn current_log_path(&self, date: &str) -> PathBuf {
        if self.config.rotate_daily {
            self.config
                .log_dir
                .join(format!("{}-{}.jsonl", self.config.base_filename, date))
        } else {
            self.config
                .log_dir
                .join(format!("{}.jsonl", self.config.base_filename))
        }
    }

    /// Ensure the log file is open and current (rotate if needed).
    async fn ensure_open(&self) -> FileBackendResult<()> {
        let today = Local::now().format("%Y-%m-%d").to_string();
        let mut guard = self.state.lock().await;

        let needs_init = guard.is_none();
        let needs_date_rotate = guard
            .as_ref()
            .map(|s| self.config.rotate_daily && s.current_date != today)
            .unwrap_or(false);
        let needs_size_rotate = guard
            .as_ref()
            .map(|s| s.bytes_written >= self.config.max_file_size_bytes)
            .unwrap_or(false);

        if needs_init || needs_date_rotate || needs_size_rotate {
            if let Some(ref mut s) = *guard {
                s.writer.flush().await?;
            }
            let path = if needs_size_rotate && !needs_date_rotate {
                // Append a timestamp to avoid clobbering.
                let ts = Utc::now().timestamp_millis();
                self.config.log_dir.join(format!(
                    "{}-{}-{}.jsonl",
                    self.config.base_filename, today, ts
                ))
            } else {
                self.current_log_path(&today)
            };
            let state = FileState::new(path).await?;
            if self.config.retain_files > 0 {
                self.prune_old_files().await;
            }
            *guard = Some(state);
        }
        Ok(())
    }

    /// Remove old rotated log files beyond the retain limit.
    async fn prune_old_files(&self) {
        let dir = &self.config.log_dir;
        let prefix = format!("{}-", self.config.base_filename);
        let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
            return;
        };
        let mut files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && name_str.ends_with(".jsonl") {
                if let Ok(meta) = entry.metadata().await {
                    if let Ok(modified) = meta.modified() {
                        files.push((entry.path(), modified));
                    }
                }
            }
        }
        if files.len() <= self.config.retain_files {
            return;
        }
        files.sort_by_key(|(_, t)| *t);
        let to_remove = files.len() - self.config.retain_files;
        for (path, _) in files.into_iter().take(to_remove) {
            let _ = tokio::fs::remove_file(path).await;
        }
    }

    /// Write a single audit record to the current log file.
    pub async fn write(&self, record: &AuditRecord) -> FileBackendResult<()> {
        self.ensure_open().await?;
        let mut guard = self.state.lock().await;
        let state = guard
            .as_mut()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "log file not open"))?;

        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        let bytes = line.as_bytes();
        state.writer.write_all(bytes).await?;
        state.bytes_written += bytes.len() as u64;
        Ok(())
    }

    /// Flush the internal buffer to the underlying file.
    pub async fn flush(&self) -> FileBackendResult<()> {
        let mut guard = self.state.lock().await;
        if let Some(ref mut s) = *guard {
            s.writer.flush().await?;
        }
        Ok(())
    }

    /// Return the path of the currently active log file.
    pub async fn current_path(&self) -> Option<PathBuf> {
        let guard = self.state.lock().await;
        guard.as_ref().map(|s| s.current_path.clone())
    }

    /// Return the number of bytes written to the current log file.
    pub async fn bytes_written(&self) -> u64 {
        let guard = self.state.lock().await;
        guard.as_ref().map(|s| s.bytes_written).unwrap_or(0)
    }
}

/// Minimal syslog-style audit backend that writes to stderr in a structured format.
///
/// In production environments, pair this with a syslog forwarder.
pub struct SyslogAuditBackend {
    facility: String,
    app_name: String,
}

impl SyslogAuditBackend {
    /// Create a new syslog backend.
    pub fn new(facility: impl Into<String>, app_name: impl Into<String>) -> Self {
        Self {
            facility: facility.into(),
            app_name: app_name.into(),
        }
    }

    /// Write a record via the syslog-compatible format to stderr.
    pub async fn write(&self, record: &AuditRecord) -> Result<(), serde_json::Error> {
        let json = serde_json::to_string(record)?;
        let msg = format!(
            "<{facility}> {app} audit: {json}",
            facility = self.facility,
            app = self.app_name,
            json = json,
        );
        // In a real deployment, write to the syslog socket.
        // For now, use tracing to emit as a structured log event.
        tracing::info!(target: "audit", "{}", msg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEvent, AuditRecord};
    use std::net::IpAddr;

    fn sample_record() -> AuditRecord {
        AuditRecord {
            timestamp: Utc::now(),
            user_id: "test-user".to_string(),
            client_ip: "127.0.0.1".parse::<IpAddr>().ok(),
            event_type: AuditEvent::QueryExecuted,
            resource: "/sparql".to_string(),
            query_text: Some("SELECT * WHERE { ?s ?p ?o }".to_string()),
            duration_ms: 42,
            success: true,
            details: None,
        }
    }

    #[tokio::test]
    async fn test_file_backend_creates_log_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_audit_test_{}",
            Utc::now().timestamp_millis()
        ));
        let config = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(config)
            .await
            .expect("should create backend");
        assert!(tmp.exists());
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_file_backend_writes_jsonl() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_audit_write_{}",
            Utc::now().timestamp_millis()
        ));
        let config = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(config).await.expect("backend");
        let rec = sample_record();
        backend.write(&rec).await.expect("write");
        backend.flush().await.expect("flush");

        let path = backend.current_path().await.expect("path");
        let contents = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(contents.contains("QueryExecuted"));
        assert!(contents.contains("test-user"));
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_file_backend_tracks_bytes() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_audit_bytes_{}",
            Utc::now().timestamp_millis()
        ));
        let config = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(config).await.expect("backend");
        let rec = sample_record();
        backend.write(&rec).await.expect("write");
        backend.flush().await.expect("flush");
        assert!(backend.bytes_written().await > 0);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_file_backend_multiple_records() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_audit_multi_{}",
            Utc::now().timestamp_millis()
        ));
        let config = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(config).await.expect("backend");
        for _ in 0..10 {
            backend.write(&sample_record()).await.expect("write");
        }
        backend.flush().await.expect("flush");

        let path = backend.current_path().await.expect("path");
        let contents = tokio::fs::read_to_string(&path).await.expect("read");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 10);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_syslog_backend_creates() {
        let backend = SyslogAuditBackend::new("LOG_LOCAL0", "oxirs-fuseki");
        let rec = sample_record();
        backend.write(&rec).await.expect("syslog write");
    }

    #[tokio::test]
    async fn test_file_backend_size_based_rotation() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_audit_size_{}",
            Utc::now().timestamp_millis()
        ));
        let config = FileBackendConfig {
            log_dir: tmp.clone(),
            base_filename: "audit".to_string(),
            max_file_size_bytes: 1, // Trigger immediately.
            rotate_daily: false,
            retain_files: 10,
        };
        let backend = FileAuditBackend::new(config).await.expect("backend");
        backend.write(&sample_record()).await.expect("write 1");
        backend.flush().await.expect("flush 1");
        // Next write should open a new file.
        backend.write(&sample_record()).await.expect("write 2");
        backend.flush().await.expect("flush 2");
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }
}
