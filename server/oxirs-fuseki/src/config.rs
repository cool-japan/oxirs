//! Advanced server configuration management with validation and hot-reload

#[cfg(feature = "saml")]
use crate::auth::saml::SamlConfig;
use crate::error::{FusekiError, FusekiResult};
use figment::{
    providers::{Env, Format, Toml, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{info, warn};
use validator::{Validate, ValidationError};

#[cfg(feature = "hot-reload")]
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
#[cfg(feature = "hot-reload")]
use std::sync::mpsc;
#[cfg(feature = "hot-reload")]
use tokio::sync::watch;

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
}

/// Server-level settings with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerSettings {
    #[validate(range(min = 1, max = 65535))]
    pub port: u16,

    #[validate(length(min = 1))]
    pub host: String,

    pub admin_ui: bool,
    pub cors: bool,

    #[validate(range(min = 1))]
    pub max_connections: usize,

    #[validate(range(min = 1))]
    pub request_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub graceful_shutdown_timeout_secs: u64,

    pub tls: Option<TlsConfig>,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TlsConfig {
    #[validate(custom(function = "validate_path"))]
    pub cert_path: PathBuf,

    #[validate(custom(function = "validate_path"))]
    pub key_path: PathBuf,

    pub require_client_cert: bool,

    pub ca_cert_path: Option<PathBuf>,
}

/// Dataset configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DatasetConfig {
    #[validate(length(min = 1))]
    pub name: String,

    #[validate(length(min = 1))]
    pub location: String,

    pub read_only: bool,

    #[validate(nested)]
    pub text_index: Option<TextIndexConfig>,

    pub shacl_shapes: Vec<PathBuf>,

    #[validate(nested)]
    pub services: Vec<ServiceConfig>,

    #[validate(nested)]
    pub access_control: Option<AccessControlConfig>,

    pub backup: Option<BackupConfig>,
}

/// Service configuration for datasets
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServiceConfig {
    #[validate(length(min = 1))]
    pub name: String,

    pub service_type: ServiceType,

    #[validate(length(min = 1))]
    pub endpoint: String,

    pub auth_required: bool,

    pub rate_limit: Option<RateLimitConfig>,
}

/// Types of services supported
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceType {
    SparqlQuery,
    SparqlUpdate,
    GraphStore,
    GraphQL,
    Rest,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AccessControlConfig {
    pub read_roles: Vec<String>,
    pub write_roles: Vec<String>,
    pub admin_roles: Vec<String>,
    pub public_read: bool,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BackupConfig {
    pub enabled: bool,

    pub directory: PathBuf,

    #[validate(range(min = 1))]
    pub interval_hours: u64,

    #[validate(range(min = 1))]
    pub retain_count: usize,
}

/// Text indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TextIndexConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub analyzer: String,

    #[validate(range(min = 1))]
    pub max_results: usize,

    pub stemming: bool,

    pub stop_words: Vec<String>,
}

/// Security configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SecurityConfig {
    pub auth_required: bool,

    #[validate(nested)]
    pub users: HashMap<String, UserConfig>,

    #[validate(nested)]
    pub jwt: Option<JwtConfig>,

    #[validate(nested)]
    pub oauth: Option<OAuthConfig>,

    #[validate(nested)]
    pub ldap: Option<LdapConfig>,

    #[validate(nested)]
    pub rate_limiting: Option<RateLimitConfig>,

    pub cors: CorsConfig,

    pub session: SessionConfig,

    #[validate(nested)]
    pub authentication: AuthenticationConfig,

    #[validate(nested)]
    pub api_keys: Option<ApiKeyConfig>,

    #[validate(nested)]
    pub certificate: Option<CertificateConfig>,

    #[validate(nested)]
    pub saml: Option<SamlConfig>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AuthenticationConfig {
    pub enabled: bool,
}

/// API Key configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyConfig {
    /// Enable API key authentication
    pub enabled: bool,

    /// Default key expiration in days
    #[validate(range(min = 1, max = 3650))] // 1 day to 10 years
    pub default_expiration_days: u32,

    /// Maximum number of keys per user
    #[validate(range(min = 1, max = 100))]
    pub max_keys_per_user: u32,

    /// Default rate limiting for API keys
    pub default_rate_limit: Option<ApiKeyRateLimit>,

    /// Enable usage analytics
    pub usage_analytics: bool,

    /// Storage backend configuration
    pub storage: ApiKeyStorageConfig,
}

/// API key rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyRateLimit {
    #[validate(range(min = 1))]
    pub requests_per_minute: u32,

    #[validate(range(min = 1))]
    pub requests_per_hour: u32,

    #[validate(range(min = 1))]
    pub requests_per_day: u32,

    #[validate(range(min = 1))]
    pub burst_limit: u32,
}

/// API key storage configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyStorageConfig {
    /// Storage backend type
    pub backend: ApiKeyStorageBackend,

    /// Connection string or file path
    #[validate(length(min = 1))]
    pub connection: String,

    /// Encryption key for sensitive data
    #[validate(length(min = 32))]
    pub encryption_key: Option<String>,
}

/// API key storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyStorageBackend {
    Memory,
    File,
    Sqlite,
    Postgres,
    Redis,
}

/// Certificate authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CertificateConfig {
    /// Enable certificate authentication
    pub enabled: bool,

    /// Require client certificates for all connections
    pub require_client_cert: bool,

    /// Trust store configuration - paths to trusted CA certificates
    pub trust_store: Vec<PathBuf>,

    /// Certificate Revocation List (CRL) URLs or file paths
    pub crl_sources: Vec<String>,

    /// Enable CRL checking
    pub check_crl: bool,

    /// Enable OCSP checking
    pub check_ocsp: bool,

    /// Allow self-signed certificates (for development only)
    pub allow_self_signed: bool,

    /// Certificate to user mapping rules
    pub user_mapping: CertificateUserMapping,

    /// Maximum certificate chain length
    #[validate(range(min = 1, max = 10))]
    pub max_chain_length: u8,

    /// Certificate validation strictness
    pub validation_level: CertificateValidationLevel,

    /// Trusted issuer DN patterns for certificate validation
    /// Certificates from these issuers will be trusted without requiring CA certificates in trust store
    pub trusted_issuers: Option<Vec<String>>,
}

/// Certificate user mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CertificateUserMapping {
    /// How to extract username from certificate
    pub username_source: CertificateUsernameSource,

    /// Subject DN to username mapping rules
    pub dn_mapping_rules: Vec<DnMappingRule>,

    /// Default roles for certificate users
    pub default_roles: Vec<String>,

    /// OU to role mapping
    pub ou_role_mapping: HashMap<String, Vec<String>>,
}

/// Source for extracting username from certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateUsernameSource {
    /// Use Common Name (CN) from subject DN
    CommonName,
    /// Use entire subject DN
    SubjectDn,
    /// Use email from Subject Alternative Name
    EmailSan,
    /// Use custom regex pattern
    CustomPattern(String),
}

/// Subject DN to username mapping rule
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DnMappingRule {
    /// Regex pattern to match against subject DN
    #[validate(length(min = 1))]
    pub pattern: String,

    /// Replacement string (supports capture groups)
    #[validate(length(min = 1))]
    pub replacement: String,

    /// Roles to assign to users matching this pattern
    pub roles: Vec<String>,
}

/// Certificate validation strictness levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateValidationLevel {
    /// Strict validation - all checks must pass
    Strict,
    /// Moderate validation - allow some minor issues
    Moderate,
    /// Permissive validation - for development/testing
    Permissive,
}

/// SAML authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlConfig {
    /// Enable SAML authentication
    pub enabled: bool,

    /// Service Provider (SP) entity ID
    #[validate(length(min = 1))]
    pub sp_entity_id: String,

    /// SP X.509 certificate for signing
    pub sp_cert_path: Option<PathBuf>,

    /// SP private key for signing
    pub sp_key_path: Option<PathBuf>,

    /// Identity Provider (IdP) configuration
    pub idp: SamlIdpConfig,

    /// Assertion consumer service URL
    #[validate(length(min = 1))]
    pub acs_url: String,

    /// Single logout service URL
    pub slo_url: Option<String>,

    /// SAML attribute mappings
    pub attribute_mappings: SamlAttributeMappings,

    /// Session timeout in seconds
    #[validate(range(min = 300, max = 86400))]
    pub session_timeout_secs: u64,
}

/// SAML Identity Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlIdpConfig {
    /// IdP entity ID
    #[validate(length(min = 1))]
    pub entity_id: String,

    /// IdP SSO URL
    #[validate(length(min = 1))]
    pub sso_url: String,

    /// IdP SLO URL
    pub slo_url: Option<String>,

    /// IdP X.509 certificate for verification
    #[validate(custom(function = "validate_path"))]
    pub cert_path: PathBuf,

    /// IdP metadata URL (alternative to manual configuration)
    pub metadata_url: Option<String>,
}

/// SAML attribute mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlAttributeMappings {
    /// SAML attribute name for username
    #[validate(length(min = 1))]
    pub username_attribute: String,

    /// SAML attribute name for email
    pub email_attribute: Option<String>,

    /// SAML attribute name for full name
    pub name_attribute: Option<String>,

    /// SAML attribute name for groups/roles
    pub groups_attribute: Option<String>,

    /// Group to role mapping
    pub group_role_mapping: HashMap<String, Vec<String>>,
}

/// JWT configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct JwtConfig {
    #[validate(length(min = 32))]
    pub secret: String,

    #[validate(range(min = 300, max = 86400))] // 5 min to 24 hours
    pub expiration_secs: u64,

    #[validate(length(min = 1))]
    pub issuer: String,

    #[validate(length(min = 1))]
    pub audience: String,
}

/// OAuth configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OAuthConfig {
    #[validate(length(min = 1))]
    pub provider: String,

    #[validate(length(min = 1))]
    pub client_id: String,

    #[validate(length(min = 1))]
    pub client_secret: String,

    #[validate(url)]
    pub auth_url: String,

    #[validate(url)]
    pub token_url: String,

    #[validate(url)]
    pub user_info_url: String,

    pub scopes: Vec<String>,
}

/// LDAP configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LdapConfig {
    #[validate(url)]
    pub server: String,

    #[validate(length(min = 1))]
    pub bind_dn: String,

    #[validate(length(min = 1))]
    pub bind_password: String,

    #[validate(length(min = 1))]
    pub user_base_dn: String,

    #[validate(length(min = 1))]
    pub user_filter: String,

    #[validate(length(min = 1))]
    pub group_base_dn: String,

    #[validate(length(min = 1))]
    pub group_filter: String,

    pub use_tls: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RateLimitConfig {
    #[validate(range(min = 1))]
    pub requests_per_minute: u32,

    #[validate(range(min = 1))]
    pub burst_size: u32,

    pub per_ip: bool,

    pub per_user: bool,

    pub whitelist: Vec<String>,
}

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CorsConfig {
    pub enabled: bool,
    pub allow_origins: Vec<String>,
    pub allow_methods: Vec<String>,
    pub allow_headers: Vec<String>,
    pub expose_headers: Vec<String>,
    pub allow_credentials: bool,

    #[validate(range(min = 0, max = 86400))]
    pub max_age_secs: u64,
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SessionConfig {
    #[validate(length(min = 32))]
    pub secret: String,

    #[validate(range(min = 300, max = 86400))] // 5 min to 24 hours
    pub timeout_secs: u64,

    pub secure: bool,

    pub http_only: bool,

    pub same_site: SameSitePolicy,
}

/// SameSite cookie policy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SameSitePolicy {
    Strict,
    Lax,
    None,
}

/// User configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct UserConfig {
    #[validate(length(min = 1))]
    pub password_hash: String,

    pub roles: Vec<String>,

    pub permissions: Vec<crate::auth::types::Permission>,

    pub enabled: bool,

    pub email: Option<String>,

    pub full_name: Option<String>,

    pub last_login: Option<chrono::DateTime<chrono::Utc>>,

    pub failed_login_attempts: u32,

    pub locked_until: Option<chrono::DateTime<chrono::Utc>>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MonitoringConfig {
    pub metrics: MetricsConfig,
    pub health_checks: HealthCheckConfig,
    pub tracing: TracingConfig,
    pub prometheus: Option<PrometheusConfig>,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricsConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub endpoint: String,

    #[validate(range(min = 1, max = 65535))]
    pub port: Option<u16>,

    pub namespace: String,

    pub collect_system_metrics: bool,

    pub histogram_buckets: Vec<f64>,
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PrometheusConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub endpoint: String,

    #[validate(range(min = 1, max = 65535))]
    pub port: Option<u16>,

    pub namespace: String,

    pub job_name: String,

    pub instance: String,

    pub scrape_interval_secs: u64,

    pub timeout_secs: u64,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HealthCheckConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub interval_secs: u64,

    #[validate(range(min = 1))]
    pub timeout_secs: u64,

    pub checks: Vec<String>,
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TracingConfig {
    pub enabled: bool,

    pub endpoint: Option<String>,

    pub service_name: String,

    pub sample_rate: f64,

    pub output: TracingOutput,
}

/// Tracing output options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracingOutput {
    Stdout,
    Stderr,
    File,
    Jaeger,
    Otlp,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PerformanceConfig {
    pub caching: CacheConfig,
    pub connection_pool: ConnectionPoolConfig,
    pub query_optimization: QueryOptimizationConfig,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CacheConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub max_size: usize,

    #[validate(range(min = 1))]
    pub ttl_secs: u64,

    pub query_cache_enabled: bool,

    pub result_cache_enabled: bool,

    pub plan_cache_enabled: bool,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConnectionPoolConfig {
    #[validate(range(min = 1))]
    pub min_connections: usize,

    #[validate(range(min = 1))]
    pub max_connections: usize,

    #[validate(range(min = 1))]
    pub connection_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub idle_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub max_lifetime_secs: u64,
}

/// Query optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct QueryOptimizationConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub max_query_time_secs: u64,

    #[validate(range(min = 1))]
    pub max_result_size: usize,

    pub parallel_execution: bool,

    #[validate(range(min = 1))]
    pub thread_pool_size: usize,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoggingConfig {
    #[validate(length(min = 1))]
    pub level: String,

    pub format: LogFormat,

    pub output: LogOutput,

    pub file_config: Option<FileLogConfig>,
}

/// Log format options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
    Compact,
}

/// Log output options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogOutput {
    Stdout,
    Stderr,
    File,
    Both,
}

/// File logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct FileLogConfig {
    pub path: PathBuf,

    #[validate(range(min = 1))]
    pub max_size_mb: u64,

    #[validate(range(min = 1))]
    pub max_files: usize,

    pub compress: bool,
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
                request_timeout_secs: 30,
                graceful_shutdown_timeout_secs: 30,
                tls: None,
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
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: LogFormat::Text,
                output: LogOutput::Stdout,
                file_config: None,
            },
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
                FusekiError::configuration(format!("Failed to load configuration: {}", e))
            })?;

        // Validate the configuration
        config.validate().map_err(|e| {
            FusekiError::validation(format!("Configuration validation failed: {}", e))
        })?;

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
                    "Unsupported configuration file format: {:?}",
                    path
                )));
            }
        }
        .map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to load configuration from {:?}: {}",
                path, e
            ))
        })?;

        // Validate the configuration
        config.validate().map_err(|e| {
            FusekiError::validation(format!("Configuration validation failed: {}", e))
        })?;

        info!("Configuration loaded from {:?}", path);
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn save_yaml<P: AsRef<Path>>(&self, path: P) -> FusekiResult<()> {
        let content = serde_yaml::to_string(self).map_err(|e| {
            FusekiError::configuration(format!("Failed to serialize configuration to YAML: {}", e))
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
            FusekiError::configuration(format!("Failed to serialize configuration to TOML: {}", e))
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
                FusekiError::configuration(format!(
                    "Invalid host:port combination '{}': {}",
                    addr, e
                ))
            })?
            .collect();

        // Return the first resolved address
        socket_addrs.into_iter().next().ok_or_else(|| {
            FusekiError::configuration(format!("No valid socket address found for '{}'", addr))
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
                errors.push(format!("Dataset '{}' has empty location", name));
            }

            // Check if SHACL shape files exist
            for shape_file in &dataset.shacl_shapes {
                if !shape_file.exists() {
                    errors.push(format!(
                        "SHACL shape file not found for dataset '{}': {:?}",
                        name, shape_file
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
                    errors.push(format!("Log file directory does not exist: {:?}", parent));
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

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics: MetricsConfig::default(),
            health_checks: HealthCheckConfig::default(),
            tracing: TracingConfig::default(),
            prometheus: None,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "/metrics".to_string(),
            port: None,
            namespace: "oxirs".to_string(),
            collect_system_metrics: true,
            histogram_buckets: vec![0.001, 0.01, 0.1, 1.0, 10.0],
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 30,
            timeout_secs: 5,
            checks: vec!["database".to_string(), "memory".to_string()],
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            service_name: "oxirs-fuseki".to_string(),
            sample_rate: 1.0,
            output: TracingOutput::Stdout,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth_required: false,
            users: HashMap::new(),
            jwt: None,
            oauth: None,
            ldap: None,
            rate_limiting: None,
            cors: CorsConfig::default(),
            session: SessionConfig::default(),
            authentication: AuthenticationConfig::default(),
            api_keys: None,
            certificate: None,
            saml: None,
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_origins: vec!["*".to_string()],
            allow_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allow_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            expose_headers: vec![],
            allow_credentials: false,
            max_age_secs: 3600,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            secret: "default-session-secret-change-in-production".to_string(),
            timeout_secs: 3600,
            secure: false,
            http_only: true,
            same_site: SameSitePolicy::Lax,
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
        unsafe { libc::geteuid() == 0 }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
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
    fn test_socket_addr() {
        let config = ServerConfig::default();
        let addr = config.socket_addr().unwrap();
        assert_eq!(addr.port(), 3030);
    }

    #[test]
    fn test_timeouts() {
        let config = ServerConfig::default();
        assert_eq!(config.request_timeout().as_secs(), 30);
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
        let mut temp_file = NamedTempFile::new().unwrap();
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

/// Custom validation function for PathBuf
fn validate_path(path: &PathBuf) -> Result<(), ValidationError> {
    if path.as_os_str().is_empty() {
        return Err(ValidationError::new("path_empty"));
    }
    Ok(())
}
