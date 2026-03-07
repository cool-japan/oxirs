//! # Serve Command
//!
//! Server lifecycle management command for the Oxirs CLI.
//!
//! This module provides configuration and dry-run semantics for starting an
//! OxiRS SPARQL endpoint without establishing actual network sockets.  It is
//! designed so that all logic can be unit-tested without spawning real servers.
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::serve_command::{ServeCommand, ServeConfig, ServeOutcome};
//!
//! let config = ServeConfig::new()
//!     .with_host("0.0.0.0")
//!     .with_port(8080)
//!     .with_cors(true);
//!
//! let cmd = ServeCommand::new();
//! let outcome = cmd.dry_run(&config);
//! assert_eq!(outcome, ServeOutcome::Started { bind_addr: "0.0.0.0:8080".to_string() });
//! ```

// ─── LogLevel ─────────────────────────────────────────────────────────────────

/// Logging verbosity level for the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Only critical errors.
    Error,
    /// Warnings and errors.
    Warn,
    /// Informational messages (default).
    Info,
    /// Verbose debug output.
    Debug,
    /// Full trace output.
    Trace,
}

impl LogLevel {
    /// Returns the canonical string label for this level.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

// ─── ServeConfig ─────────────────────────────────────────────────────────────

/// Configuration options for an OxiRS server instance.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Bind host address (e.g. `"127.0.0.1"` or `"0.0.0.0"`).
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
    /// Optional path to the RDF dataset directory or file.
    pub dataset_path: Option<String>,
    /// Optional path to a TOML configuration file.
    pub config_file: Option<String>,
    /// Maximum number of simultaneous client connections.
    pub max_connections: usize,
    /// Maximum query execution time in milliseconds.
    pub query_timeout_ms: u64,
    /// Whether to enable Cross-Origin Resource Sharing headers.
    pub enable_cors: bool,
    /// Whether to require HTTP authentication.
    pub enable_auth: bool,
    /// Logging verbosity.
    pub log_level: LogLevel,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3030,
            dataset_path: None,
            config_file: None,
            max_connections: 100,
            query_timeout_ms: 30_000,
            enable_cors: false,
            enable_auth: false,
            log_level: LogLevel::Info,
        }
    }
}

impl ServeConfig {
    /// Creates a [`ServeConfig`] with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the bind host (builder style).
    pub fn with_host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    /// Sets the TCP port (builder style).
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Sets the dataset path (builder style).
    pub fn with_dataset(mut self, path: &str) -> Self {
        self.dataset_path = Some(path.to_string());
        self
    }

    /// Sets the configuration file path (builder style).
    pub fn with_config_file(mut self, path: &str) -> Self {
        self.config_file = Some(path.to_string());
        self
    }

    /// Sets the maximum number of connections (builder style).
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Sets the query timeout in milliseconds (builder style).
    pub fn with_query_timeout(mut self, timeout_ms: u64) -> Self {
        self.query_timeout_ms = timeout_ms;
        self
    }

    /// Enables or disables CORS headers (builder style).
    pub fn with_cors(mut self, enabled: bool) -> Self {
        self.enable_cors = enabled;
        self
    }

    /// Enables or disables HTTP authentication (builder style).
    pub fn with_auth(mut self, enabled: bool) -> Self {
        self.enable_auth = enabled;
        self
    }

    /// Sets the log level (builder style).
    pub fn with_log_level(mut self, level: LogLevel) -> Self {
        self.log_level = level;
        self
    }

    /// Validates the configuration, returning `Err` with a human-readable
    /// message when any constraint is violated.
    ///
    /// Validation rules:
    /// - `port` must be > 0
    /// - `max_connections` must be > 0
    /// - `query_timeout_ms` must be > 0
    /// - `host` must not be empty
    pub fn validate(&self) -> Result<(), String> {
        if self.host.is_empty() {
            return Err("host must not be empty".to_string());
        }
        if self.port == 0 {
            return Err("port must be greater than 0".to_string());
        }
        if self.max_connections == 0 {
            return Err("max_connections must be greater than 0".to_string());
        }
        if self.query_timeout_ms == 0 {
            return Err("query_timeout_ms must be greater than 0".to_string());
        }
        Ok(())
    }

    /// Renders the configuration as a human-readable TOML-like summary string.
    ///
    /// The summary always includes the host, port, max_connections,
    /// query_timeout_ms, log level, and the boolean flags.
    pub fn summary(&self) -> String {
        let mut parts = vec![
            format!("host = \"{}\"", self.host),
            format!("port = {}", self.port),
            format!("max_connections = {}", self.max_connections),
            format!("query_timeout_ms = {}", self.query_timeout_ms),
            format!("log_level = \"{}\"", self.log_level.as_str()),
            format!("enable_cors = {}", self.enable_cors),
            format!("enable_auth = {}", self.enable_auth),
        ];
        if let Some(ref dp) = self.dataset_path {
            parts.push(format!("dataset_path = \"{}\"", dp));
        }
        if let Some(ref cf) = self.config_file {
            parts.push(format!("config_file = \"{}\"", cf));
        }
        parts.join("\n")
    }
}

// ─── ServeOutcome ─────────────────────────────────────────────────────────────

/// The outcome of a [`ServeCommand::dry_run`] invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeOutcome {
    /// Server configuration was valid; the bind address is returned.
    Started {
        /// The `"host:port"` string the server would listen on.
        bind_addr: String,
    },
    /// Configuration is invalid; the error message is returned.
    ConfigError(String),
    /// Another server instance is already running on this address.
    AlreadyRunning,
}

// ─── ServeCommand ─────────────────────────────────────────────────────────────

/// Handler for the `serve` CLI subcommand.
pub struct ServeCommand;

impl ServeCommand {
    /// Creates a new [`ServeCommand`].
    pub fn new() -> Self {
        Self
    }

    /// Validates `config` and simulates a server start without binding any
    /// network sockets.
    ///
    /// Returns [`ServeOutcome::Started`] on success or
    /// [`ServeOutcome::ConfigError`] when validation fails.
    pub fn dry_run(&self, config: &ServeConfig) -> ServeOutcome {
        match config.validate() {
            Ok(()) => ServeOutcome::Started {
                bind_addr: Self::bind_addr(config),
            },
            Err(msg) => ServeOutcome::ConfigError(msg),
        }
    }

    /// Formats the `"host:port"` bind address string from `config`.
    pub fn bind_addr(config: &ServeConfig) -> String {
        format!("{}:{}", config.host, config.port)
    }
}

impl Default for ServeCommand {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_host() {
        assert_eq!(ServeConfig::default().host, "127.0.0.1");
    }

    #[test]
    fn test_default_port() {
        assert_eq!(ServeConfig::default().port, 3030);
    }

    #[test]
    fn test_default_max_connections() {
        assert_eq!(ServeConfig::default().max_connections, 100);
    }

    #[test]
    fn test_default_query_timeout() {
        assert_eq!(ServeConfig::default().query_timeout_ms, 30_000);
    }

    #[test]
    fn test_default_cors_disabled() {
        assert!(!ServeConfig::default().enable_cors);
    }

    #[test]
    fn test_default_auth_disabled() {
        assert!(!ServeConfig::default().enable_auth);
    }

    #[test]
    fn test_default_log_level_info() {
        assert_eq!(ServeConfig::default().log_level, LogLevel::Info);
    }

    #[test]
    fn test_default_dataset_path_none() {
        assert!(ServeConfig::default().dataset_path.is_none());
    }

    #[test]
    fn test_default_config_file_none() {
        assert!(ServeConfig::default().config_file.is_none());
    }

    // ── Builder methods ───────────────────────────────────────────────────────

    #[test]
    fn test_with_host() {
        let cfg = ServeConfig::new().with_host("0.0.0.0");
        assert_eq!(cfg.host, "0.0.0.0");
    }

    #[test]
    fn test_with_port() {
        let cfg = ServeConfig::new().with_port(8080);
        assert_eq!(cfg.port, 8080);
    }

    #[test]
    fn test_with_dataset() {
        let cfg = ServeConfig::new().with_dataset("/data/my.ttl");
        assert_eq!(cfg.dataset_path, Some("/data/my.ttl".to_string()));
    }

    #[test]
    fn test_with_config_file() {
        let cfg = ServeConfig::new().with_config_file("/etc/oxirs.toml");
        assert_eq!(cfg.config_file, Some("/etc/oxirs.toml".to_string()));
    }

    #[test]
    fn test_with_cors_true() {
        let cfg = ServeConfig::new().with_cors(true);
        assert!(cfg.enable_cors);
    }

    #[test]
    fn test_with_cors_false() {
        let cfg = ServeConfig::new().with_cors(false);
        assert!(!cfg.enable_cors);
    }

    #[test]
    fn test_with_auth_true() {
        let cfg = ServeConfig::new().with_auth(true);
        assert!(cfg.enable_auth);
    }

    #[test]
    fn test_with_auth_false() {
        let cfg = ServeConfig::new().with_auth(false);
        assert!(!cfg.enable_auth);
    }

    #[test]
    fn test_with_log_level_debug() {
        let cfg = ServeConfig::new().with_log_level(LogLevel::Debug);
        assert_eq!(cfg.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_with_log_level_trace() {
        let cfg = ServeConfig::new().with_log_level(LogLevel::Trace);
        assert_eq!(cfg.log_level, LogLevel::Trace);
    }

    #[test]
    fn test_with_max_connections() {
        let cfg = ServeConfig::new().with_max_connections(500);
        assert_eq!(cfg.max_connections, 500);
    }

    #[test]
    fn test_with_query_timeout() {
        let cfg = ServeConfig::new().with_query_timeout(60_000);
        assert_eq!(cfg.query_timeout_ms, 60_000);
    }

    #[test]
    fn test_builder_chain_all() {
        let cfg = ServeConfig::new()
            .with_host("10.0.0.1")
            .with_port(9090)
            .with_dataset("/data")
            .with_config_file("/cfg.toml")
            .with_max_connections(200)
            .with_query_timeout(5_000)
            .with_cors(true)
            .with_auth(true)
            .with_log_level(LogLevel::Warn);
        assert_eq!(cfg.host, "10.0.0.1");
        assert_eq!(cfg.port, 9090);
        assert_eq!(cfg.dataset_path, Some("/data".to_string()));
        assert_eq!(cfg.config_file, Some("/cfg.toml".to_string()));
        assert_eq!(cfg.max_connections, 200);
        assert_eq!(cfg.query_timeout_ms, 5_000);
        assert!(cfg.enable_cors);
        assert!(cfg.enable_auth);
        assert_eq!(cfg.log_level, LogLevel::Warn);
    }

    // ── validate ──────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_default_ok() {
        assert!(ServeConfig::new().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_port_error() {
        let cfg = ServeConfig::new().with_port(0);
        assert!(cfg.validate().is_err());
        let msg = cfg.validate().unwrap_err();
        assert!(msg.contains("port"));
    }

    #[test]
    fn test_validate_zero_max_connections_error() {
        let cfg = ServeConfig::new().with_max_connections(0);
        assert!(cfg.validate().is_err());
        let msg = cfg.validate().unwrap_err();
        assert!(msg.contains("max_connections"));
    }

    #[test]
    fn test_validate_zero_timeout_error() {
        let cfg = ServeConfig::new().with_query_timeout(0);
        assert!(cfg.validate().is_err());
        let msg = cfg.validate().unwrap_err();
        assert!(msg.contains("query_timeout_ms"));
    }

    #[test]
    fn test_validate_empty_host_error() {
        let cfg = ServeConfig::new().with_host("");
        assert!(cfg.validate().is_err());
        let msg = cfg.validate().unwrap_err();
        assert!(msg.contains("host"));
    }

    #[test]
    fn test_validate_max_port() {
        let cfg = ServeConfig::new().with_port(65535);
        assert!(cfg.validate().is_ok());
    }

    // ── summary ───────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_contains_host() {
        let cfg = ServeConfig::new().with_host("192.168.1.1");
        assert!(cfg.summary().contains("192.168.1.1"));
    }

    #[test]
    fn test_summary_contains_port() {
        let cfg = ServeConfig::new().with_port(4040);
        assert!(cfg.summary().contains("4040"));
    }

    #[test]
    fn test_summary_contains_default_host_and_port() {
        let cfg = ServeConfig::new();
        let s = cfg.summary();
        assert!(s.contains("127.0.0.1"));
        assert!(s.contains("3030"));
    }

    #[test]
    fn test_summary_contains_log_level() {
        let cfg = ServeConfig::new().with_log_level(LogLevel::Debug);
        assert!(cfg.summary().contains("debug"));
    }

    #[test]
    fn test_summary_contains_cors() {
        let cfg = ServeConfig::new().with_cors(true);
        assert!(cfg.summary().contains("enable_cors = true"));
    }

    #[test]
    fn test_summary_contains_dataset_when_set() {
        let cfg = ServeConfig::new().with_dataset("/my/data");
        assert!(cfg.summary().contains("/my/data"));
    }

    #[test]
    fn test_summary_no_dataset_when_not_set() {
        let cfg = ServeConfig::new();
        assert!(!cfg.summary().contains("dataset_path"));
    }

    // ── dry_run ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_started_default_config() {
        let cmd = ServeCommand::new();
        let cfg = ServeConfig::new();
        let outcome = cmd.dry_run(&cfg);
        assert_eq!(
            outcome,
            ServeOutcome::Started {
                bind_addr: "127.0.0.1:3030".to_string()
            }
        );
    }

    #[test]
    fn test_dry_run_started_custom_host_port() {
        let cmd = ServeCommand::new();
        let cfg = ServeConfig::new().with_host("0.0.0.0").with_port(8080);
        let outcome = cmd.dry_run(&cfg);
        assert_eq!(
            outcome,
            ServeOutcome::Started {
                bind_addr: "0.0.0.0:8080".to_string()
            }
        );
    }

    #[test]
    fn test_dry_run_config_error_invalid_port() {
        let cmd = ServeCommand::new();
        let cfg = ServeConfig::new().with_port(0);
        let outcome = cmd.dry_run(&cfg);
        assert!(matches!(outcome, ServeOutcome::ConfigError(_)));
    }

    #[test]
    fn test_dry_run_config_error_message() {
        let cmd = ServeCommand::new();
        let cfg = ServeConfig::new().with_port(0);
        if let ServeOutcome::ConfigError(msg) = cmd.dry_run(&cfg) {
            assert!(msg.contains("port"));
        } else {
            panic!("expected ConfigError");
        }
    }

    #[test]
    fn test_dry_run_config_error_zero_connections() {
        let cmd = ServeCommand::new();
        let cfg = ServeConfig::new().with_max_connections(0);
        assert!(matches!(cmd.dry_run(&cfg), ServeOutcome::ConfigError(_)));
    }

    // ── bind_addr ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bind_addr_default() {
        let cfg = ServeConfig::new();
        assert_eq!(ServeCommand::bind_addr(&cfg), "127.0.0.1:3030");
    }

    #[test]
    fn test_bind_addr_custom() {
        let cfg = ServeConfig::new().with_host("::1").with_port(9999);
        assert_eq!(ServeCommand::bind_addr(&cfg), "::1:9999");
    }

    #[test]
    fn test_bind_addr_zero_zero_zero_zero() {
        let cfg = ServeConfig::new().with_host("0.0.0.0").with_port(3030);
        assert_eq!(ServeCommand::bind_addr(&cfg), "0.0.0.0:3030");
    }

    // ── LogLevel ─────────────────────────────────────────────────────────────

    #[test]
    fn test_log_level_error_as_str() {
        assert_eq!(LogLevel::Error.as_str(), "error");
    }

    #[test]
    fn test_log_level_warn_as_str() {
        assert_eq!(LogLevel::Warn.as_str(), "warn");
    }

    #[test]
    fn test_log_level_info_as_str() {
        assert_eq!(LogLevel::Info.as_str(), "info");
    }

    #[test]
    fn test_log_level_debug_as_str() {
        assert_eq!(LogLevel::Debug.as_str(), "debug");
    }

    #[test]
    fn test_log_level_trace_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "trace");
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn test_log_level_equality() {
        assert_eq!(LogLevel::Info, LogLevel::Info);
        assert_ne!(LogLevel::Info, LogLevel::Debug);
    }

    // ── ServeCommand default ──────────────────────────────────────────────────

    #[test]
    fn test_serve_command_default() {
        let _cmd = ServeCommand;
    }

    // ── ServeOutcome equality ─────────────────────────────────────────────────

    #[test]
    fn test_serve_outcome_started_equality() {
        let a = ServeOutcome::Started {
            bind_addr: "127.0.0.1:3030".to_string(),
        };
        let b = ServeOutcome::Started {
            bind_addr: "127.0.0.1:3030".to_string(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_serve_outcome_config_error_equality() {
        let a = ServeOutcome::ConfigError("bad port".to_string());
        let b = ServeOutcome::ConfigError("bad port".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn test_serve_outcome_already_running() {
        let o = ServeOutcome::AlreadyRunning;
        assert_eq!(o, ServeOutcome::AlreadyRunning);
    }

    #[test]
    fn test_serve_outcome_inequality() {
        let a = ServeOutcome::Started {
            bind_addr: "127.0.0.1:3030".to_string(),
        };
        let b = ServeOutcome::AlreadyRunning;
        assert_ne!(a, b);
    }
}
