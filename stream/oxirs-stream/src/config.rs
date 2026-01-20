//! # Advanced Configuration Management
//!
//! This module provides comprehensive configuration management for oxirs-stream with:
//! - Dynamic configuration updates without restart
//! - Environment-based configuration loading
//! - Secret management integration
//! - SSL/TLS certificate management
//! - Authentication configuration
//! - Performance tuning profiles

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tracing::info;

use crate::{CompressionType, StreamBackendType, StreamConfig, StreamPerformanceConfig};

/// Configuration source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigSource {
    /// Configuration from file
    File { path: PathBuf },
    /// Configuration from environment variables
    Environment { prefix: String },
    /// Configuration from remote source (e.g., etcd, consul)
    Remote { url: String, key: String },
    /// Configuration from memory (for testing)
    Memory { data: HashMap<String, String> },
}

/// Configuration manager with dynamic reload support
pub struct ConfigManager {
    /// Current configuration
    current_config: Arc<RwLock<StreamConfig>>,
    /// Configuration sources in priority order
    sources: Vec<ConfigSource>,
    /// Configuration change notifier
    change_notifier: broadcast::Sender<ConfigChangeEvent>,
    /// Secret manager
    secret_manager: Arc<SecretManager>,
    /// Environment detector
    environment: Environment,
    /// Performance profiles
    performance_profiles: HashMap<String, PerformanceProfile>,
    /// SSL/TLS manager
    tls_manager: Arc<TlsManager>,
}

/// Configuration change event
#[derive(Debug, Clone)]
pub enum ConfigChangeEvent {
    /// Configuration reloaded
    Reloaded {
        old_config: Box<StreamConfig>,
        new_config: Box<StreamConfig>,
    },
    /// Configuration validation failed
    ValidationFailed { reason: String },
    /// Secret rotated
    SecretRotated { secret_name: String },
    /// TLS certificate updated
    TlsCertificateUpdated { cert_type: String },
}

/// Environment detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

impl Environment {
    /// Detect environment from various sources
    pub fn detect() -> Self {
        if let Ok(env) = std::env::var("OXIRS_ENV") {
            match env.to_lowercase().as_str() {
                "dev" | "development" => Environment::Development,
                "test" | "testing" => Environment::Testing,
                "staging" | "stage" => Environment::Staging,
                "prod" | "production" => Environment::Production,
                _ => Environment::Development,
            }
        } else if let Ok(env) = std::env::var("RUST_ENV") {
            match env.to_lowercase().as_str() {
                "production" => Environment::Production,
                _ => Environment::Development,
            }
        } else {
            Environment::Development
        }
    }

    /// Get environment-specific defaults
    pub fn get_defaults(&self) -> ConfigDefaults {
        match self {
            Environment::Development => ConfigDefaults {
                log_level: "debug".to_string(),
                enable_debug_endpoints: true,
                connection_timeout_secs: 60,
                max_connections: 10,
                enable_compression: false,
                enable_metrics: true,
                enable_profiling: true,
            },
            Environment::Testing => ConfigDefaults {
                log_level: "info".to_string(),
                enable_debug_endpoints: true,
                connection_timeout_secs: 30,
                max_connections: 5,
                enable_compression: false,
                enable_metrics: true,
                enable_profiling: false,
            },
            Environment::Staging => ConfigDefaults {
                log_level: "info".to_string(),
                enable_debug_endpoints: false,
                connection_timeout_secs: 30,
                max_connections: 50,
                enable_compression: true,
                enable_metrics: true,
                enable_profiling: false,
            },
            Environment::Production => ConfigDefaults {
                log_level: "warn".to_string(),
                enable_debug_endpoints: false,
                connection_timeout_secs: 30,
                max_connections: 100,
                enable_compression: true,
                enable_metrics: true,
                enable_profiling: false,
            },
        }
    }
}

/// Environment-specific configuration defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDefaults {
    pub log_level: String,
    pub enable_debug_endpoints: bool,
    pub connection_timeout_secs: u64,
    pub max_connections: usize,
    pub enable_compression: bool,
    pub enable_metrics: bool,
    pub enable_profiling: bool,
}

/// Secret manager for handling sensitive configuration
pub struct SecretManager {
    /// Secret store backend
    backend: SecretBackend,
    /// Cached secrets with expiration
    cache: Arc<RwLock<HashMap<String, CachedSecret>>>,
    /// Secret rotation interval
    rotation_interval: Duration,
}

/// Secret backend types
#[derive(Debug, Clone)]
pub enum SecretBackend {
    /// Environment variables
    Environment { prefix: String },
    /// File-based secrets
    File { directory: PathBuf },
    /// HashiCorp Vault
    Vault {
        url: String,
        token: String,
        mount_path: String,
    },
    /// AWS Secrets Manager
    AwsSecretsManager { region: String },
    /// Memory-based (for testing)
    Memory {
        secrets: Arc<RwLock<HashMap<String, String>>>,
    },
}

/// Cached secret with metadata
#[derive(Debug, Clone)]
struct CachedSecret {
    value: String,
    cached_at: std::time::Instant,
    expires_at: Option<std::time::Instant>,
    version: u64,
}

impl SecretManager {
    /// Create a new secret manager
    pub fn new(backend: SecretBackend, rotation_interval: Duration) -> Self {
        Self {
            backend,
            cache: Arc::new(RwLock::new(HashMap::new())),
            rotation_interval,
        }
    }

    /// Get a secret value
    pub async fn get_secret(&self, name: &str) -> Result<String> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(name) {
                if cached
                    .expires_at
                    .map_or(true, |exp| exp > std::time::Instant::now())
                {
                    return Ok(cached.value.clone());
                }
            }
        }

        // Fetch from backend
        let value = match &self.backend {
            SecretBackend::Environment { prefix } => {
                let key = format!("{prefix}_{}", name.to_uppercase());
                std::env::var(key).map_err(|_| anyhow!("Secret {name} not found in environment"))
            }
            SecretBackend::File { directory } => {
                let path = directory.join(name);
                fs::read_to_string(&path)
                    .map_err(|e| anyhow!("Failed to read secret from {path:?}: {e}"))
                    .map(|s| s.trim().to_string())
            }
            SecretBackend::Memory { secrets } => {
                let secrets = secrets.read().await;
                secrets
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("Secret {name} not found"))
            }
            _ => {
                // Vault and AWS implementations would go here
                return Err(anyhow!("Secret backend not implemented"));
            }
        }?;

        // Cache the secret
        let cached = CachedSecret {
            value: value.clone(),
            cached_at: std::time::Instant::now(),
            expires_at: Some(std::time::Instant::now() + self.rotation_interval),
            version: 1,
        };

        self.cache.write().await.insert(name.to_string(), cached);
        Ok(value)
    }

    /// Set a secret (for testing)
    pub async fn set_secret(&self, name: &str, value: &str) -> Result<()> {
        match &self.backend {
            SecretBackend::Memory { secrets } => {
                secrets
                    .write()
                    .await
                    .insert(name.to_string(), value.to_string());
                self.cache.write().await.remove(name);
                Ok(())
            }
            _ => Err(anyhow!("Set secret only supported for memory backend")),
        }
    }

    /// Rotate all secrets
    pub async fn rotate_secrets(&self) -> Result<()> {
        self.cache.write().await.clear();
        info!("Rotated all cached secrets");
        Ok(())
    }
}

/// TLS/SSL certificate manager
pub struct TlsManager {
    /// Certificate store
    certs: Arc<RwLock<HashMap<String, TlsCertificate>>>,
    /// Certificate paths
    cert_paths: HashMap<String, CertPaths>,
    /// Auto-reload enabled
    auto_reload: bool,
}

/// TLS certificate with metadata
#[derive(Debug, Clone)]
pub struct TlsCertificate {
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
    pub ca_pem: Option<Vec<u8>>,
    pub loaded_at: std::time::Instant,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Certificate file paths
#[derive(Debug, Clone)]
struct CertPaths {
    cert_path: PathBuf,
    key_path: PathBuf,
    ca_path: Option<PathBuf>,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(auto_reload: bool) -> Self {
        Self {
            certs: Arc::new(RwLock::new(HashMap::new())),
            cert_paths: HashMap::new(),
            auto_reload,
        }
    }

    /// Parse certificate expiration from PEM data
    fn parse_certificate_expiration(cert_pem: &[u8]) -> Option<chrono::DateTime<chrono::Utc>> {
        // Simple PEM certificate expiration parsing
        // In a production system, we'd use proper X.509 parsing libraries
        let cert_str = String::from_utf8_lossy(cert_pem);

        // For now, we'll implement a basic parser that looks for common patterns
        // This is a placeholder implementation that could be enhanced with proper X.509 parsing
        if cert_str.contains("-----BEGIN CERTIFICATE-----") {
            // Set a default expiration of 1 year from now for valid certificates
            // In practice, this should parse the actual certificate validity period
            Some(chrono::Utc::now() + chrono::Duration::days(365))
        } else {
            None
        }
    }

    /// Load a certificate
    pub async fn load_certificate(
        &self,
        name: &str,
        cert_path: &Path,
        key_path: &Path,
        ca_path: Option<&Path>,
    ) -> Result<()> {
        let cert_pem = fs::read(cert_path)
            .map_err(|e| anyhow!("Failed to read certificate {}: {}", cert_path.display(), e))?;

        let key_pem = fs::read(key_path)
            .map_err(|e| anyhow!("Failed to read key {}: {}", key_path.display(), e))?;

        let ca_pem = if let Some(ca) = ca_path {
            Some(fs::read(ca).map_err(|e| anyhow!("Failed to read CA {}: {}", ca.display(), e))?)
        } else {
            None
        };

        let expires_at = Self::parse_certificate_expiration(&cert_pem);

        let cert = TlsCertificate {
            cert_pem,
            key_pem,
            ca_pem,
            loaded_at: std::time::Instant::now(),
            expires_at,
        };

        self.certs.write().await.insert(name.to_string(), cert);

        info!("Loaded TLS certificate: {}", name);
        Ok(())
    }

    /// Get a certificate
    pub async fn get_certificate(&self, name: &str) -> Result<TlsCertificate> {
        self.certs
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Certificate {} not found", name))
    }

    /// Check certificate expiration
    pub async fn check_expiration(&self) -> Vec<(String, chrono::DateTime<chrono::Utc>)> {
        let certs = self.certs.read().await;
        let mut expiring = Vec::new();

        for (name, cert) in certs.iter() {
            if let Some(expires) = cert.expires_at {
                let days_until = (expires - chrono::Utc::now()).num_days();
                if days_until < 30 {
                    expiring.push((name.clone(), expires));
                }
            }
        }

        expiring
    }
}

/// Performance tuning profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub name: String,
    pub description: String,
    pub settings: StreamPerformanceConfig,
    pub recommended_for: Vec<String>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub async fn new(sources: Vec<ConfigSource>) -> Result<Self> {
        let environment = Environment::detect();
        let (tx, _) = broadcast::channel(100);

        // Initialize secret manager
        let secret_backend = SecretBackend::Environment {
            prefix: "OXIRS_SECRET".to_string(),
        };
        let secret_manager = Arc::new(SecretManager::new(
            secret_backend,
            Duration::from_secs(3600),
        ));

        // Initialize TLS manager
        let tls_manager = Arc::new(TlsManager::new(true));

        // Load initial configuration
        let initial_config = StreamConfig::default();

        let mut manager = Self {
            current_config: Arc::new(RwLock::new(initial_config)),
            sources,
            change_notifier: tx,
            secret_manager,
            environment,
            performance_profiles: Self::create_default_profiles(),
            tls_manager,
        };

        // Load configuration from sources
        manager.reload().await?;

        Ok(manager)
    }

    /// Create default performance profiles
    fn create_default_profiles() -> HashMap<String, PerformanceProfile> {
        let mut profiles = HashMap::new();

        // Low latency profile
        profiles.insert(
            "low-latency".to_string(),
            PerformanceProfile {
                name: "low-latency".to_string(),
                description: "Optimized for minimal latency".to_string(),
                settings: StreamPerformanceConfig {
                    enable_batching: false,
                    enable_pipelining: true,
                    buffer_size: 4096,
                    prefetch_count: 10,
                    enable_zero_copy: true,
                    enable_simd: true,
                    parallel_processing: true,
                    worker_threads: Some(4),
                },
                recommended_for: vec!["real-time".to_string(), "trading".to_string()],
            },
        );

        // High throughput profile
        profiles.insert(
            "high-throughput".to_string(),
            PerformanceProfile {
                name: "high-throughput".to_string(),
                description: "Optimized for maximum throughput".to_string(),
                settings: StreamPerformanceConfig {
                    enable_batching: true,
                    enable_pipelining: true,
                    buffer_size: 65536,
                    prefetch_count: 1000,
                    enable_zero_copy: true,
                    enable_simd: true,
                    parallel_processing: true,
                    worker_threads: None, // Use all available
                },
                recommended_for: vec!["batch-processing".to_string(), "etl".to_string()],
            },
        );

        // Balanced profile
        profiles.insert(
            "balanced".to_string(),
            PerformanceProfile {
                name: "balanced".to_string(),
                description: "Balanced between latency and throughput".to_string(),
                settings: StreamPerformanceConfig::default(),
                recommended_for: vec!["general".to_string(), "web-services".to_string()],
            },
        );

        // Resource-constrained profile
        profiles.insert(
            "resource-constrained".to_string(),
            PerformanceProfile {
                name: "resource-constrained".to_string(),
                description: "Optimized for limited resources".to_string(),
                settings: StreamPerformanceConfig {
                    enable_batching: true,
                    enable_pipelining: false,
                    buffer_size: 2048,
                    prefetch_count: 10,
                    enable_zero_copy: false,
                    enable_simd: false,
                    parallel_processing: false,
                    worker_threads: Some(2),
                },
                recommended_for: vec!["edge".to_string(), "iot".to_string()],
            },
        );

        profiles
    }

    /// Reload configuration from all sources
    pub async fn reload(&mut self) -> Result<()> {
        let old_config = self.current_config.read().await.clone();
        let mut new_config = old_config.clone();

        // Apply environment defaults
        let defaults = self.environment.get_defaults();
        new_config.max_connections = defaults.max_connections;
        new_config.connection_timeout = Duration::from_secs(defaults.connection_timeout_secs);
        new_config.enable_compression = defaults.enable_compression;
        new_config.monitoring.enable_metrics = defaults.enable_metrics;
        new_config.monitoring.enable_profiling = defaults.enable_profiling;

        // Load from each source in order
        for source in &self.sources {
            match source {
                ConfigSource::File { path } => {
                    if let Ok(content) = fs::read_to_string(path) {
                        if let Ok(file_config) = toml::from_str::<StreamConfig>(&content) {
                            new_config = self.merge_configs(new_config, file_config);
                        }
                    }
                }
                ConfigSource::Environment { prefix } => {
                    new_config = self.load_from_env(new_config, prefix).await?;
                }
                ConfigSource::Memory { data } => {
                    new_config = self.apply_overrides(new_config, data.clone());
                }
                _ => {
                    // Remote source would be implemented here
                }
            }
        }

        // Apply secrets
        new_config = self.apply_secrets(new_config).await?;

        // Validate configuration
        self.validate_config(&new_config)?;

        // Update current configuration
        *self.current_config.write().await = new_config.clone();

        // Notify listeners
        let _ = self.change_notifier.send(ConfigChangeEvent::Reloaded {
            old_config: Box::new(old_config),
            new_config: Box::new(new_config),
        });

        info!("Configuration reloaded successfully");
        Ok(())
    }

    /// Load configuration from environment variables
    async fn load_from_env(&self, mut config: StreamConfig, prefix: &str) -> Result<StreamConfig> {
        // Backend selection
        if let Ok(backend) = std::env::var(format!("{prefix}_BACKEND")) {
            config.backend = match backend.as_str() {
                "kafka" => {
                    #[cfg(feature = "kafka")]
                    {
                        let brokers: Vec<String> = std::env::var(format!("{prefix}_KAFKA_BROKERS"))
                            .unwrap_or_else(|_| "localhost:9092".to_string())
                            .split(',')
                            .map(|s| s.to_string())
                            .collect();
                        StreamBackendType::Kafka {
                            brokers,
                            security_protocol: std::env::var(format!("{}_KAFKA_SECURITY", prefix))
                                .ok(),
                            sasl_config: None,
                        }
                    }
                    #[cfg(not(feature = "kafka"))]
                    {
                        let _ = std::env::var(format!("{prefix}_KAFKA_BROKERS"));
                        StreamBackendType::Memory {
                            max_size: Some(10000),
                            persistence: false,
                        }
                    }
                }
                "memory" => StreamBackendType::Memory {
                    max_size: Some(10000),
                    persistence: false,
                },
                _ => config.backend,
            };
        }

        // Connection settings
        if let Ok(max_conn) = std::env::var(format!("{prefix}_MAX_CONNECTIONS")) {
            if let Ok(val) = max_conn.parse() {
                config.max_connections = val;
            }
        }

        // Compression
        if let Ok(compression) = std::env::var(format!("{prefix}_COMPRESSION")) {
            config.compression_type = match compression.as_str() {
                "gzip" => CompressionType::Gzip,
                "snappy" => CompressionType::Snappy,
                "lz4" => CompressionType::Lz4,
                "zstd" => CompressionType::Zstd,
                _ => CompressionType::None,
            };
            config.enable_compression = compression != "none";
        }

        Ok(config)
    }

    /// Apply secrets to configuration
    async fn apply_secrets(&self, mut config: StreamConfig) -> Result<StreamConfig> {
        // Apply SASL password if using Kafka
        #[cfg(feature = "kafka")]
        if let StreamBackendType::Kafka {
            brokers,
            security_protocol,
            sasl_config: _,
        } = &config.backend
        {
            if security_protocol.as_deref() == Some("SASL_SSL") {
                if let Ok(username) = self.secret_manager.get_secret("kafka_username").await {
                    if let Ok(password) = self.secret_manager.get_secret("kafka_password").await {
                        #[cfg(feature = "kafka")]
                        {
                            config.backend = StreamBackendType::Kafka {
                                brokers: brokers.clone(),
                                security_protocol: security_protocol.clone(),
                                sasl_config: Some(crate::SaslConfig {
                                    mechanism: crate::SaslMechanism::ScramSha256,
                                    username,
                                    password,
                                }),
                            };
                        }
                    }
                }
            }
        }

        // Apply TLS certificates
        if config.security.enable_tls {
            if let Ok(_cert) = self.tls_manager.get_certificate("client").await {
                // Certificate paths would be set here
                config.security.client_cert_path = Some("/tmp/client.crt".to_string());
                config.security.client_key_path = Some("/tmp/client.key".to_string());
            }
        }

        Ok(config)
    }

    /// Merge two configurations
    fn merge_configs(&self, _base: StreamConfig, override_config: StreamConfig) -> StreamConfig {
        // This would implement a proper merge strategy
        // For now, just return the override
        override_config
    }

    /// Apply key-value overrides
    fn apply_overrides(
        &self,
        mut config: StreamConfig,
        overrides: HashMap<String, String>,
    ) -> StreamConfig {
        for (key, value) in overrides {
            match key.as_str() {
                "topic" => config.topic = value,
                "batch_size" => {
                    if let Ok(size) = value.parse() {
                        config.batch_size = size;
                    }
                }
                "max_connections" => {
                    if let Ok(max) = value.parse() {
                        config.max_connections = max;
                    }
                }
                _ => {}
            }
        }
        config
    }

    /// Validate configuration
    fn validate_config(&self, config: &StreamConfig) -> Result<()> {
        // Validate connection limits
        if config.max_connections == 0 {
            return Err(anyhow!("max_connections must be greater than 0"));
        }

        // Validate batch size
        if config.batch_size == 0 {
            return Err(anyhow!("batch_size must be greater than 0"));
        }

        // Validate topic name
        if config.topic.is_empty() {
            return Err(anyhow!("topic name cannot be empty"));
        }

        // Backend-specific validation
        match &config.backend {
            #[cfg(feature = "kafka")]
            StreamBackendType::Kafka { brokers, .. } => {
                if brokers.is_empty() {
                    return Err(anyhow!("Kafka brokers list cannot be empty"));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> StreamConfig {
        self.current_config.read().await.clone()
    }

    /// Subscribe to configuration changes
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.change_notifier.subscribe()
    }

    /// Apply a performance profile
    pub async fn apply_performance_profile(&mut self, profile_name: &str) -> Result<()> {
        let profile = self
            .performance_profiles
            .get(profile_name)
            .ok_or_else(|| anyhow!("Performance profile {} not found", profile_name))?
            .clone();

        let mut config = self.current_config.write().await;
        config.performance = profile.settings;

        info!("Applied performance profile: {}", profile_name);
        Ok(())
    }

    /// Get available performance profiles
    pub fn get_performance_profiles(&self) -> Vec<&PerformanceProfile> {
        self.performance_profiles.values().collect()
    }

    /// Update a specific configuration value
    pub async fn update_value(&mut self, key: &str, value: String) -> Result<()> {
        let mut overrides = HashMap::new();
        overrides.insert(key.to_string(), value);

        let current = self.current_config.read().await.clone();
        let updated = self.apply_overrides(current, overrides);

        self.validate_config(&updated)?;
        *self.current_config.write().await = updated;

        info!("Updated configuration key: {}", key);
        Ok(())
    }

    /// Get secret manager
    pub fn secret_manager(&self) -> &Arc<SecretManager> {
        &self.secret_manager
    }

    /// Get TLS manager
    pub fn tls_manager(&self) -> &Arc<TlsManager> {
        &self.tls_manager
    }
}

/// Configuration builder for easy setup
pub struct ConfigBuilder {
    sources: Vec<ConfigSource>,
    environment: Option<Environment>,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            environment: None,
        }
    }

    /// Add a file source
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.sources.push(ConfigSource::File { path: path.into() });
        self
    }

    /// Add environment variable source
    pub fn with_env(mut self, prefix: impl Into<String>) -> Self {
        self.sources.push(ConfigSource::Environment {
            prefix: prefix.into(),
        });
        self
    }

    /// Add memory source for overrides
    pub fn with_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        self.sources.push(ConfigSource::Memory { data: overrides });
        self
    }

    /// Set environment explicitly
    pub fn with_environment(mut self, env: Environment) -> Self {
        self.environment = Some(env);
        self
    }

    /// Build the configuration manager
    pub async fn build(self) -> Result<ConfigManager> {
        ConfigManager::new(self.sources).await
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_environment_detection() {
        let env = Environment::detect();
        let defaults = env.get_defaults();
        assert!(defaults.max_connections > 0);
    }

    #[tokio::test]
    async fn test_config_builder() {
        let mut overrides = HashMap::new();
        overrides.insert("topic".to_string(), "test-topic".to_string());

        let manager = ConfigBuilder::new()
            .with_env("OXIRS")
            .with_overrides(overrides)
            .build()
            .await
            .unwrap();

        let config = manager.get_config().await;
        assert_eq!(config.topic, "test-topic");
    }

    #[tokio::test]
    async fn test_secret_manager() {
        let backend = SecretBackend::Memory {
            secrets: Arc::new(RwLock::new(HashMap::new())),
        };
        let manager = SecretManager::new(backend, Duration::from_secs(60));

        // Set and get secret
        manager.set_secret("test_key", "test_value").await.unwrap();
        let value = manager.get_secret("test_key").await.unwrap();
        assert_eq!(value, "test_value");

        // Test cache
        let value2 = manager.get_secret("test_key").await.unwrap();
        assert_eq!(value2, "test_value");
    }

    #[tokio::test]
    async fn test_performance_profiles() {
        let manager = ConfigBuilder::new().build().await.unwrap();
        let profiles = manager.get_performance_profiles();

        assert!(profiles.len() >= 4);
        assert!(profiles.iter().any(|p| p.name == "low-latency"));
        assert!(profiles.iter().any(|p| p.name == "high-throughput"));
    }

    #[tokio::test]
    async fn test_config_validation() {
        let manager = ConfigBuilder::new().build().await.unwrap();

        // Test invalid configuration
        let invalid_config = StreamConfig {
            max_connections: 0,
            ..Default::default()
        };

        assert!(manager.validate_config(&invalid_config).is_err());
    }
}
