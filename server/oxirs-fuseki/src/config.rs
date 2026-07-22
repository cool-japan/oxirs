//! Advanced server configuration management with validation and hot-reload.
//!
//! This module is split into domain sub-modules:
//! - [`config_server`]: network/HTTP/TLS/dataset server setup types
//! - [`config_security`]: auth, identity, certs, SAML, OAuth, JWT, LDAP, MFA, ReBAC
//! - [`config_runtime`]: rate limiting, CORS, sessions, metrics, observability, caching, logging

#[path = "config_runtime.rs"]
pub mod config_runtime;
#[path = "config_security.rs"]
pub mod config_security;
#[path = "config_server.rs"]
pub mod config_server;

pub use config_runtime::*;
pub use config_security::*;
pub use config_server::*;

use crate::error::{FusekiError, FusekiResult};
use figment::{
    providers::{Env, Format, Toml, Yaml},
    Figment,
};
#[cfg(feature = "hot-reload")]
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
#[cfg(feature = "hot-reload")]
use std::sync::mpsc;
use std::time::Duration;
#[cfg(feature = "hot-reload")]
use tokio::sync::watch;
use tracing::{info, warn};
use validator::Validate;

/// Main server configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerConfig {
    #[validate(nested)]
    pub server: ServerSettings,
    #[validate(nested)]
    pub datasets: HashMap<String, DatasetConfig>,

    #[validate(nested)]
    pub security: SecurityConfig,

    #[validate(nested)]
    pub monitoring: MonitoringConfig,

    #[validate(nested)]
    pub performance: PerformanceConfig,

    #[validate(nested)]
    pub logging: LoggingConfig,

    /// Federation configuration for distributed query execution
    /// Note: Currently not serializable, uses defaults at runtime
    #[serde(skip)]
    pub federation: Option<crate::federation::FederationConfig>,

    /// Streaming configuration for event processing
    /// Note: Currently not serializable, uses defaults at runtime
    #[serde(skip)]
    pub streaming: Option<crate::streaming::StreamingConfig>,

    /// HTTP protocol configuration (HTTP/2, HTTP/3)
    #[validate(nested)]
    pub http_protocol: HttpProtocolSettings,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            server: ServerSettings {
                port: 3030,
                host: "localhost".to_string(),
                admin_ui: true,
                cors: true,
                max_connections: 1000,
                // Coarse whole-request deadline for the axum `TimeoutLayer`. It
                // must sit ABOVE the per-query execution budget so the budget is
                // what normally fires (a precise 408 carrying the query's own
                // message) and this layer only guarantees the connection is not
                // held forever. The invariant enforced at startup
                // (`Runtime::build_router`) is:
                //   request_timeout_secs > max_query_time_secs + QUERY_TIMEOUT_GRACE_SECS
                // With the shipped defaults 300 (max_query_time) + 5 (grace) that
                // is 305, so 310 leaves the budget a clear margin to fire first.
                // The previous default of 30 inverted the relationship: the
                // TimeoutLayer preempted the budget at 30 s with a generic 408
                // while the detached blocking task kept running up to 300 s.
                request_timeout_secs: 310,
                graceful_shutdown_timeout_secs: 30,
                tls: None,
                backup_directory: None,
                static_asset_dir: None,
                config_file: None,
            },
            datasets: HashMap::new(),
            security: SecurityConfig {
                auth_required: false,
                users: HashMap::new(),
                jwt: None,
                oauth: None,
                ldap: None,
                rate_limiting: None,
                cors: CorsConfig {
                    enabled: true,
                    allow_origins: vec!["*".to_string()],
                    allow_methods: vec![
                        "GET".to_string(),
                        "POST".to_string(),
                        "PUT".to_string(),
                        "DELETE".to_string(),
                    ],
                    allow_headers: vec!["*".to_string()],
                    expose_headers: vec![],
                    allow_credentials: false,
                    max_age_secs: 3600,
                },
                session: SessionConfig {
                    secret: uuid::Uuid::new_v4().to_string(),
                    timeout_secs: 3600,
                    secure: false,
                    http_only: true,
                    same_site: SameSitePolicy::Lax,
                },
                authentication: AuthenticationConfig { enabled: false },
                api_keys: None,
                certificate: None,
                saml: None,
                rebac: None,
                mfa: None,
            },
            monitoring: MonitoringConfig {
                metrics: MetricsConfig {
                    enabled: true,
                    endpoint: "/metrics".to_string(),
                    port: None,
                    namespace: "oxirs_fuseki".to_string(),
                    collect_system_metrics: true,
                    histogram_buckets: vec![
                        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                    ],
                },
                health_checks: HealthCheckConfig {
                    enabled: true,
                    interval_secs: 30,
                    timeout_secs: 5,
                    checks: vec!["store".to_string(), "memory".to_string()],
                },
                tracing: TracingConfig {
                    enabled: false,
                    endpoint: None,
                    service_name: "oxirs-fuseki".to_string(),
                    sample_rate: 0.1,
                    output: TracingOutput::Stdout,
                },
                prometheus: None,
            },
            performance: PerformanceConfig {
                caching: CacheConfig {
                    enabled: true,
                    max_size: 1000,
                    ttl_secs: 300,
                    query_cache_enabled: true,
                    result_cache_enabled: true,
                    plan_cache_enabled: true,
                },
                connection_pool: ConnectionPoolConfig {
                    min_connections: 1,
                    max_connections: 10,
                    connection_timeout_secs: 30,
                    idle_timeout_secs: 600,
                    max_lifetime_secs: 3600,
                },
                query_optimization: QueryOptimizationConfig {
                    enabled: true,
                    max_query_time_secs: 300,
                    max_result_size: 1_000_000,
                    parallel_execution: true,
                    thread_pool_size: get_cpu_count(),
                },
                rate_limiting: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: LogFormat::Text,
                output: LogOutput::Stdout,
                file_config: None,
            },
            federation: None,
            streaming: None,
            http_protocol: HttpProtocolSettings::default(),
        }
    }
}

impl ServerConfig {
    /// Load configuration using Figment (supports TOML, YAML, env vars)
    pub fn load() -> FusekiResult<Self> {
        let config: Self = Figment::new()
            .merge(Toml::file("oxirs-fuseki.toml"))
            .merge(Yaml::file("oxirs-fuseki.yaml"))
            .merge(Yaml::file("oxirs-fuseki.yml"))
            .merge(Env::prefixed("OXIRS_FUSEKI_"))
            .extract()
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to load configuration: {e}"))
            })?;

        // Validate the configuration
        config.validate().map_err(|e| {
            FusekiError::validation(format!("Configuration validation failed: {e}"))
        })?;
        config.validate_auth_reachable()?;

        Ok(config)
    }

    /// Load configuration from a specific file
    pub fn from_file<P: AsRef<Path>>(path: P) -> FusekiResult<Self> {
        let path = path.as_ref();
        let config: Self = match path.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => {
                let figment = Figment::new()
                    .merge(Toml::file(path))
                    .merge(Env::prefixed("OXIRS_FUSEKI_"));
                figment.extract()
            }
            Some("yaml") | Some("yml") => {
                let figment = Figment::new()
                    .merge(Yaml::file(path))
                    .merge(Env::prefixed("OXIRS_FUSEKI_"));
                figment.extract()
            }
            _ => {
                return Err(FusekiError::configuration(format!(
                    "Unsupported configuration file format: {path:?}"
                )));
            }
        }
        .map_err(|e| {
            FusekiError::configuration(format!("Failed to load configuration from {path:?}: {e}"))
        })?;

        // Validate the configuration
        config.validate().map_err(|e| {
            FusekiError::validation(format!("Configuration validation failed: {e}"))
        })?;
        config.validate_auth_reachable()?;

        info!("Configuration loaded from {:?}", path);
        Ok(config)
    }

    /// Fail loud when `security.auth_required` is set but no authentication
    /// backend is actually reachable — i.e. no static users, OAuth, LDAP,
    /// enabled SAML, enabled API keys, or enabled client-certificate auth is
    /// configured. Such a configuration would start successfully and then
    /// reject every single request forever (the auth-enforcing middleware
    /// layers are only ever wired in when `auth_required` is true), which is
    /// indistinguishable from a broken deployment. Per the fail-loud
    /// contract this must refuse to start rather than silently deploy a
    /// server nobody can use.
    pub fn validate_auth_reachable(&self) -> FusekiResult<()> {
        if !self.security.auth_required {
            return Ok(());
        }
        let has_static_users = !self.security.users.is_empty();
        let has_oauth = self.security.oauth.is_some();
        let has_ldap = self.security.ldap.is_some();
        let has_saml = self.security.saml.as_ref().is_some_and(|saml| saml.enabled);
        let has_api_keys = self
            .security
            .api_keys
            .as_ref()
            .is_some_and(|keys| keys.enabled);
        let has_certificate_auth = self
            .security
            .certificate
            .as_ref()
            .is_some_and(|cert| cert.enabled);

        if !(has_static_users
            || has_oauth
            || has_ldap
            || has_saml
            || has_api_keys
            || has_certificate_auth)
        {
            return Err(FusekiError::configuration(
                "security.auth_required is true but no authentication backend is configured \
                 (no static users, OAuth, LDAP, enabled SAML, enabled API keys, or enabled \
                 client-certificate auth); every request would be permanently rejected. \
                 Configure at least one authentication method, or set auth_required = false",
            ));
        }
        Ok(())
    }

    /// Save configuration to YAML file
    pub fn save_yaml<P: AsRef<Path>>(&self, path: P) -> FusekiResult<()> {
        let content = serde_yaml::to_string(self).map_err(|e| {
            FusekiError::configuration(format!("Failed to serialize configuration to YAML: {e}"))
        })?;

        std::fs::write(&path, content).map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to write configuration to {:?}: {}",
                path.as_ref(),
                e
            ))
        })?;

        info!("Configuration saved to {:?}", path.as_ref());
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> FusekiResult<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            FusekiError::configuration(format!("Failed to serialize configuration to TOML: {e}"))
        })?;

        std::fs::write(&path, content).map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to write configuration to {:?}: {}",
                path.as_ref(),
                e
            ))
        })?;

        info!("Configuration saved to {:?}", path.as_ref());
        Ok(())
    }

    /// Get the socket address for the server
    pub fn socket_addr(&self) -> FusekiResult<SocketAddr> {
        use std::net::ToSocketAddrs;

        let addr = format!("{}:{}", self.server.host, self.server.port);

        // Try to resolve the hostname to socket addresses
        let socket_addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .map_err(|e| {
                FusekiError::configuration(format!("Invalid host:port combination '{addr}': {e}"))
            })?
            .collect();

        // Return the first resolved address
        socket_addrs.into_iter().next().ok_or_else(|| {
            FusekiError::configuration(format!("No valid socket address found for '{addr}'"))
        })
    }

    /// Get request timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.server.request_timeout_secs)
    }

    /// Get graceful shutdown timeout as Duration
    pub fn graceful_shutdown_timeout(&self) -> Duration {
        Duration::from_secs(self.server.graceful_shutdown_timeout_secs)
    }

    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.server.tls.is_some()
    }

    /// Check if authentication is required
    pub fn requires_auth(&self) -> bool {
        self.security.auth_required
    }

    /// Check if metrics are enabled
    pub fn metrics_enabled(&self) -> bool {
        self.monitoring.metrics.enabled
    }

    /// Check if tracing is enabled
    pub fn tracing_enabled(&self) -> bool {
        self.monitoring.tracing.enabled
    }

    /// Validate configuration and return detailed errors
    pub fn validate_detailed(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check port availability (basic check)
        if self.server.port < 1024 && !is_privileged_user() {
            errors.push(format!(
                "Port {} requires elevated privileges. Consider using port >= 1024",
                self.server.port
            ));
        }

        // Check TLS configuration
        if let Some(ref tls) = self.server.tls {
            if !tls.cert_path.exists() {
                errors.push(format!(
                    "TLS certificate file not found: {:?}",
                    tls.cert_path
                ));
            }
            if !tls.key_path.exists() {
                errors.push(format!("TLS key file not found: {:?}", tls.key_path));
            }
        }

        // Check dataset locations
        for (name, dataset) in &self.datasets {
            if dataset.location.is_empty() {
                errors.push(format!("Dataset '{name}' has empty location"));
            }

            // Check if SHACL shape files exist
            for shape_file in &dataset.shacl_shapes {
                if !shape_file.exists() {
                    errors.push(format!(
                        "SHACL shape file not found for dataset '{name}': {shape_file:?}"
                    ));
                }
            }
        }

        // Check JWT configuration
        if let Some(ref jwt) = self.security.jwt {
            if jwt.secret.len() < 32 {
                errors.push("JWT secret must be at least 32 characters long".to_string());
            }
        }

        // Check logging configuration
        if let Some(ref file_config) = self.logging.file_config {
            if let Some(parent) = file_config.path.parent() {
                if !parent.exists() {
                    errors.push(format!("Log file directory does not exist: {parent:?}"));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Configuration watcher for hot-reloading
#[cfg(feature = "hot-reload")]
pub struct ConfigWatcher {
    _watcher: RecommendedWatcher,
    receiver: tokio::sync::watch::Receiver<ServerConfig>,
}

#[cfg(feature = "hot-reload")]
impl ConfigWatcher {
    /// Create a new configuration watcher
    pub fn new<P: AsRef<Path>>(
        config_path: P,
    ) -> FusekiResult<(Self, tokio::sync::watch::Receiver<ServerConfig>)> {
        let config_path = config_path.as_ref().to_path_buf();
        let initial_config = ServerConfig::from_file(&config_path)?;

        let (tx, rx) = tokio::sync::watch::channel(initial_config);
        let (file_tx, file_rx) = mpsc::channel();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if let Err(e) = file_tx.send(event) {
                        warn!("Failed to send file watch event: {}", e);
                    }
                }
                Err(e) => warn!("File watch error: {}", e),
            }
        })
        .map_err(|e| FusekiError::configuration(format!("Failed to create file watcher: {}", e)))?;

        watcher
            .watch(&config_path, RecursiveMode::NonRecursive)
            .map_err(|e| {
                FusekiError::configuration(format!(
                    "Failed to watch config file {:?}: {}",
                    config_path, e
                ))
            })?;

        // Spawn background task to handle file events
        let config_path_clone = config_path.clone();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            while let Ok(event) = file_rx.recv() {
                if event.kind.is_modify() {
                    // Debounce rapid file changes
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    match ServerConfig::from_file(&config_path_clone) {
                        Ok(new_config) => {
                            if let Err(e) = tx_clone.send(new_config) {
                                warn!("Failed to send updated config: {}", e);
                            } else {
                                info!("Configuration reloaded from {:?}", config_path_clone);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to reload configuration: {}", e);
                        }
                    }
                }
            }
        });

        let config_watcher = ConfigWatcher {
            _watcher: watcher,
            receiver: rx.clone(),
        };

        Ok((config_watcher, rx))
    }

    /// Get the current configuration
    pub fn current_config(&self) -> ServerConfig {
        self.receiver.borrow().clone()
    }
}

/// Check if the current user has elevated privileges
fn is_privileged_user() -> bool {
    #[cfg(unix)]
    {
        // Use std::env to check USER environment variable as a safe alternative
        std::env::var("USER")
            .map(|user| user == "root")
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        // On Windows, assume non-privileged for simplicity
        // In a real implementation, you'd check for administrator privileges
        false
    }
}

/// Get the number of CPU cores available
fn get_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) // Fallback to 4 cores
}

/// Custom validation function for PathBuf
pub(crate) fn validate_path_pub(path: &Path) -> Result<(), validator::ValidationError> {
    if path.as_os_str().is_empty() {
        return Err(validator::ValidationError::new("path_empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.server.port, 3030);
        assert_eq!(config.server.host, "localhost");
        assert!(config.server.admin_ui);
        assert!(config.server.cors);
        assert!(!config.security.auth_required);
        assert!(config.datasets.is_empty());
        assert!(config.security.users.is_empty());
        assert!(config.monitoring.metrics.enabled);
        assert!(config.performance.caching.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ServerConfig::default();

        // Valid configuration should pass
        assert!(config.validate().is_ok());

        // Invalid port should fail
        config.server.port = 0;
        assert!(config.validate().is_err());

        // Reset to valid
        config.server.port = 3030;

        // Empty host should fail
        config.server.host = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn regression_auth_required_with_no_backend_fails_loud() {
        // auth_required = true with the default empty security config (no
        // static users, OAuth, LDAP, SAML, API keys, or certificate auth)
        // must be rejected at startup rather than silently deployed as a
        // server that rejects every request forever.
        let mut config = ServerConfig::default();
        config.security.auth_required = true;
        assert!(
            config.validate_auth_reachable().is_err(),
            "auth_required=true with no reachable auth backend must fail loud"
        );
    }

    #[test]
    fn regression_auth_required_with_static_users_passes() {
        let mut config = ServerConfig::default();
        config.security.auth_required = true;
        config.security.users.insert(
            "admin".to_string(),
            UserConfig {
                password_hash: "$argon2id$dummy".to_string(),
                roles: vec!["admin".to_string()],
                permissions: vec![],
                enabled: true,
                email: None,
                full_name: None,
                last_login: None,
                failed_login_attempts: 0,
                locked_until: None,
            },
        );
        assert!(config.validate_auth_reachable().is_ok());
    }

    #[test]
    fn regression_auth_not_required_skips_backend_check() {
        // The default (auth_required = false) must never be rejected by
        // this check regardless of how empty the security config is.
        let config = ServerConfig::default();
        assert!(config.validate_auth_reachable().is_ok());
    }

    #[test]
    fn test_socket_addr() {
        let config = ServerConfig::default();
        let addr = config.socket_addr().unwrap();
        assert_eq!(addr.port(), 3030);
    }

    #[test]
    fn test_timeouts() {
        let config = ServerConfig::default();
        // request_timeout_secs must exceed max_query_time_secs (300) + grace (5)
        // so the per-query budget fires before the coarse HTTP TimeoutLayer.
        assert_eq!(config.request_timeout().as_secs(), 310);
        assert_eq!(config.graceful_shutdown_timeout().as_secs(), 30);
    }

    #[test]
    fn test_tls_config() {
        let mut config = ServerConfig::default();
        assert!(!config.is_tls_enabled());

        config.server.tls = Some(TlsConfig {
            cert_path: "/path/to/cert.pem".into(),
            key_path: "/path/to/key.pem".into(),
            require_client_cert: false,
            ca_cert_path: None,
        });
        assert!(config.is_tls_enabled());
    }

    #[test]
    fn test_jwt_config_validation() {
        let mut jwt_config = JwtConfig {
            secret: "short".to_string(),
            expiration_secs: 3600,
            issuer: "oxirs-fuseki".to_string(),
            audience: "oxirs-users".to_string(),
        };

        // Short secret should fail validation
        assert!(jwt_config.validate().is_err());

        // Long enough secret should pass
        jwt_config.secret = "a".repeat(32);
        assert!(jwt_config.validate().is_ok());
    }

    #[test]
    fn test_rate_limit_config() {
        let rate_limit = RateLimitConfig {
            requests_per_minute: 100,
            burst_size: 10,
            per_ip: true,
            per_user: false,
            whitelist: vec!["127.0.0.1".to_string()],
        };

        assert!(rate_limit.validate().is_ok());
    }

    #[test]
    fn test_service_types() {
        let service = ServiceConfig {
            name: "query".to_string(),
            service_type: ServiceType::SparqlQuery,
            endpoint: "sparql".to_string(),
            auth_required: false,
            rate_limit: None,
        };

        assert!(service.validate().is_ok());
    }

    #[test]
    fn test_monitoring_config() {
        let monitoring = MonitoringConfig {
            metrics: MetricsConfig {
                enabled: true,
                endpoint: "/metrics".to_string(),
                port: Some(9090),
                namespace: "test".to_string(),
                collect_system_metrics: true,
                histogram_buckets: vec![0.1, 1.0, 10.0],
            },
            health_checks: HealthCheckConfig {
                enabled: true,
                interval_secs: 30,
                timeout_secs: 5,
                checks: vec!["store".to_string()],
            },
            tracing: TracingConfig {
                enabled: false,
                endpoint: None,
                service_name: "test".to_string(),
                sample_rate: 0.1,
                output: TracingOutput::Stdout,
            },
            prometheus: None,
        };

        assert!(monitoring.validate().is_ok());
    }

    #[test]
    fn test_performance_config() {
        let performance = PerformanceConfig {
            caching: CacheConfig {
                enabled: true,
                max_size: 1000,
                ttl_secs: 300,
                query_cache_enabled: true,
                result_cache_enabled: true,
                plan_cache_enabled: true,
            },
            connection_pool: ConnectionPoolConfig {
                min_connections: 1,
                max_connections: 10,
                connection_timeout_secs: 30,
                idle_timeout_secs: 600,
                max_lifetime_secs: 3600,
            },
            query_optimization: QueryOptimizationConfig {
                enabled: true,
                max_query_time_secs: 300,
                max_result_size: 1_000_000,
                parallel_execution: true,
                thread_pool_size: 4,
            },
            rate_limiting: None,
        };

        assert!(performance.validate().is_ok());
    }

    #[test]
    fn test_logging_config() {
        let logging = LoggingConfig {
            level: "info".to_string(),
            format: LogFormat::Json,
            output: LogOutput::Stdout,
            file_config: None,
        };

        assert!(logging.validate().is_ok());
    }

    #[test]
    fn test_user_config_extended() {
        let user = UserConfig {
            password_hash: "$argon2id$v=19$m=65536,t=3,p=4$...".to_string(),
            roles: vec!["admin".to_string(), "user".to_string()],
            permissions: vec![],
            enabled: true,
            email: Some("admin@example.com".to_string()),
            full_name: Some("Administrator".to_string()),
            last_login: None,
            failed_login_attempts: 0,
            locked_until: None,
        };

        assert!(user.validate().is_ok());
        assert_eq!(user.roles.len(), 2);
        assert!(user.enabled);
        assert_eq!(user.failed_login_attempts, 0);
    }

    #[test]
    fn test_cors_config() {
        let cors = CorsConfig {
            enabled: true,
            allow_origins: vec!["http://localhost:3000".to_string()],
            allow_methods: vec!["GET".to_string(), "POST".to_string()],
            allow_headers: vec!["Content-Type".to_string()],
            expose_headers: vec![],
            allow_credentials: true,
            max_age_secs: 3600,
        };

        assert!(cors.validate().is_ok());
    }

    #[test]
    fn test_session_config() {
        let session = SessionConfig {
            secret: "a".repeat(32),
            timeout_secs: 3600,
            secure: true,
            http_only: true,
            same_site: SameSitePolicy::Strict,
        };

        assert!(session.validate().is_ok());
    }

    #[test]
    fn test_save_and_load_yaml() {
        let config = ServerConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("yaml");

        // Save configuration
        config.save_yaml(&temp_path).unwrap();

        // Load configuration
        let loaded_config = ServerConfig::from_file(&temp_path).unwrap();

        assert_eq!(config.server.port, loaded_config.server.port);
        assert_eq!(config.server.host, loaded_config.server.host);
    }

    #[test]
    fn test_save_and_load_toml() {
        let config = ServerConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("toml");

        // Save configuration
        config.save_toml(&temp_path).unwrap();

        // Load configuration
        let loaded_config = ServerConfig::from_file(&temp_path).unwrap();

        assert_eq!(config.server.port, loaded_config.server.port);
        assert_eq!(config.server.host, loaded_config.server.host);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_detailed_validation() {
        let mut config = ServerConfig::default();

        // Should pass validation for default config
        assert!(config.validate_detailed().is_ok());

        // Add invalid dataset
        let dataset = DatasetConfig {
            name: "test".to_string(),
            location: String::new(), // Empty location should fail
            read_only: false,
            text_index: None,
            shacl_shapes: vec![],
            services: vec![],
            access_control: None,
            backup: None,
        };

        config.datasets.insert("test".to_string(), dataset);

        let errors = config.validate_detailed().unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("empty location")));
    }
}
