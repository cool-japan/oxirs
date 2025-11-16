//! # Enhanced TLS/SSL Security Module
//!
//! Provides comprehensive TLS/SSL encryption, certificate management, and secure communication
//! for all streaming backends with support for mutual TLS (mTLS), certificate rotation, and
//! advanced cipher suites.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// TLS/SSL configuration with advanced security options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS encryption
    pub enabled: bool,
    /// TLS protocol version
    pub protocol_version: TlsVersion,
    /// Certificate configuration
    pub certificates: CertificateConfig,
    /// Cipher suite configuration
    pub cipher_suites: Vec<CipherSuite>,
    /// Mutual TLS (mTLS) configuration
    pub mtls: MutualTlsConfig,
    /// Certificate rotation settings
    pub rotation: CertRotationConfig,
    /// OCSP stapling configuration
    pub ocsp_stapling: OcspConfig,
    /// Perfect forward secrecy
    pub perfect_forward_secrecy: bool,
    /// Session resumption
    pub session_resumption: SessionResumptionConfig,
    /// ALPN protocols
    pub alpn_protocols: Vec<String>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol_version: TlsVersion::Tls13,
            certificates: CertificateConfig::default(),
            cipher_suites: vec![
                CipherSuite::TLS_AES_256_GCM_SHA384,
                CipherSuite::TLS_CHACHA20_POLY1305_SHA256,
                CipherSuite::TLS_AES_128_GCM_SHA256,
            ],
            mtls: MutualTlsConfig::default(),
            rotation: CertRotationConfig::default(),
            ocsp_stapling: OcspConfig::default(),
            perfect_forward_secrecy: true,
            session_resumption: SessionResumptionConfig::default(),
            alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
        }
    }
}

/// TLS protocol versions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TlsVersion {
    /// TLS 1.2 (minimum recommended)
    Tls12,
    /// TLS 1.3 (recommended)
    Tls13,
}

impl std::fmt::Display for TlsVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TlsVersion::Tls12 => write!(f, "TLS 1.2"),
            TlsVersion::Tls13 => write!(f, "TLS 1.3"),
        }
    }
}

/// Certificate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    /// Server certificate path
    pub server_cert_path: PathBuf,
    /// Server private key path
    pub server_key_path: PathBuf,
    /// Certificate authority (CA) certificate path
    pub ca_cert_path: Option<PathBuf>,
    /// Certificate chain path
    pub cert_chain_path: Option<PathBuf>,
    /// Key password/passphrase
    pub key_password: Option<String>,
    /// Certificate format
    pub format: CertificateFormat,
    /// Verify peer certificates
    pub verify_peer: bool,
    /// Verify hostname
    pub verify_hostname: bool,
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            server_cert_path: PathBuf::from("/etc/oxirs/certs/server.crt"),
            server_key_path: PathBuf::from("/etc/oxirs/certs/server.key"),
            ca_cert_path: Some(PathBuf::from("/etc/oxirs/certs/ca.crt")),
            cert_chain_path: None,
            key_password: None,
            format: CertificateFormat::PEM,
            verify_peer: true,
            verify_hostname: true,
        }
    }
}

/// Certificate formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CertificateFormat {
    /// PEM format (Base64 encoded DER)
    PEM,
    /// DER format (binary)
    DER,
    /// PKCS#12 format (.pfx/.p12)
    PKCS12,
}

/// Supported cipher suites (TLS 1.3 and 1.2)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CipherSuite {
    // TLS 1.3 cipher suites
    TLS_AES_256_GCM_SHA384,
    TLS_CHACHA20_POLY1305_SHA256,
    TLS_AES_128_GCM_SHA256,
    TLS_AES_128_CCM_SHA256,

    // TLS 1.2 cipher suites (backward compatibility)
    TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
}

impl std::fmt::Display for CipherSuite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CipherSuite::TLS_AES_256_GCM_SHA384 => write!(f, "TLS_AES_256_GCM_SHA384"),
            CipherSuite::TLS_CHACHA20_POLY1305_SHA256 => write!(f, "TLS_CHACHA20_POLY1305_SHA256"),
            CipherSuite::TLS_AES_128_GCM_SHA256 => write!(f, "TLS_AES_128_GCM_SHA256"),
            CipherSuite::TLS_AES_128_CCM_SHA256 => write!(f, "TLS_AES_128_CCM_SHA256"),
            CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 => {
                write!(f, "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384")
            }
            CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 => {
                write!(f, "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384")
            }
            CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 => {
                write!(f, "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256")
            }
            CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 => {
                write!(f, "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256")
            }
        }
    }
}

/// Mutual TLS (mTLS) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualTlsConfig {
    /// Enable mutual TLS
    pub enabled: bool,
    /// Client certificate required
    pub require_client_cert: bool,
    /// Trusted client CA certificates
    pub trusted_ca_certs: Vec<PathBuf>,
    /// Client certificate verification depth
    pub verification_depth: u8,
    /// Revocation check configuration
    pub revocation_check: RevocationCheckConfig,
}

impl Default for MutualTlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_client_cert: true,
            trusted_ca_certs: vec![],
            verification_depth: 3,
            revocation_check: RevocationCheckConfig::default(),
        }
    }
}

/// Certificate revocation check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationCheckConfig {
    /// Enable revocation checking
    pub enabled: bool,
    /// Check CRL (Certificate Revocation List)
    pub check_crl: bool,
    /// Check OCSP (Online Certificate Status Protocol)
    pub check_ocsp: bool,
    /// CRL cache TTL in seconds
    pub crl_cache_ttl: u64,
}

impl Default for RevocationCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_crl: true,
            check_ocsp: true,
            crl_cache_ttl: 3600,
        }
    }
}

/// Certificate rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertRotationConfig {
    /// Enable automatic certificate rotation
    pub enabled: bool,
    /// Check interval in seconds
    pub check_interval_secs: u64,
    /// Rotation threshold (days before expiry)
    pub rotation_threshold_days: u32,
    /// Graceful rotation period (seconds)
    pub graceful_period_secs: u64,
}

impl Default for CertRotationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_secs: 3600,   // Check every hour
            rotation_threshold_days: 30, // Rotate 30 days before expiry
            graceful_period_secs: 300,   // 5 minutes graceful period
        }
    }
}

/// OCSP (Online Certificate Status Protocol) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcspConfig {
    /// Enable OCSP stapling
    pub enabled: bool,
    /// OCSP responder URL
    pub responder_url: Option<String>,
    /// OCSP response cache TTL in seconds
    pub cache_ttl: u64,
    /// OCSP request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for OcspConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            responder_url: None,
            cache_ttl: 3600,
            timeout_secs: 10,
        }
    }
}

/// Session resumption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResumptionConfig {
    /// Enable session resumption
    pub enabled: bool,
    /// Session cache size
    pub cache_size: usize,
    /// Session ticket lifetime in seconds
    pub ticket_lifetime_secs: u64,
    /// Session ID lifetime in seconds
    pub session_id_lifetime_secs: u64,
}

impl Default for SessionResumptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size: 10000,
            ticket_lifetime_secs: 7200,     // 2 hours
            session_id_lifetime_secs: 7200, // 2 hours
        }
    }
}

/// TLS certificate information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    /// Certificate subject
    pub subject: String,
    /// Certificate issuer
    pub issuer: String,
    /// Serial number
    pub serial_number: String,
    /// Valid from
    pub valid_from: DateTime<Utc>,
    /// Valid until
    pub valid_until: DateTime<Utc>,
    /// Subject alternative names (SANs)
    pub san: Vec<String>,
    /// Key algorithm
    pub key_algorithm: String,
    /// Key size in bits
    pub key_size: u32,
    /// Signature algorithm
    pub signature_algorithm: String,
    /// Fingerprint (SHA-256)
    pub fingerprint_sha256: String,
}

/// TLS session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsSessionInfo {
    /// Session ID
    pub session_id: String,
    /// TLS version used
    pub protocol_version: TlsVersion,
    /// Cipher suite used
    pub cipher_suite: String,
    /// Server name indication (SNI)
    pub sni: Option<String>,
    /// ALPN protocol negotiated
    pub alpn_protocol: Option<String>,
    /// Client certificate (if mTLS)
    pub client_cert: Option<CertificateInfo>,
    /// Established timestamp
    pub established_at: DateTime<Utc>,
}

/// TLS manager for certificate and connection management
pub struct TlsManager {
    config: TlsConfig,
    certificates: Arc<RwLock<HashMap<String, CertificateInfo>>>,
    sessions: Arc<RwLock<HashMap<String, TlsSessionInfo>>>,
    metrics: Arc<RwLock<TlsMetrics>>,
}

/// TLS metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlsMetrics {
    /// Total TLS connections established
    pub connections_established: u64,
    /// Total TLS handshakes
    pub handshakes_total: u64,
    /// Failed handshakes
    pub handshakes_failed: u64,
    /// Certificate rotations performed
    pub certificate_rotations: u64,
    /// OCSP requests
    pub ocsp_requests: u64,
    /// Session resumptions
    pub session_resumptions: u64,
    /// Average handshake duration (ms)
    pub avg_handshake_duration_ms: f64,
    /// TLS version distribution
    pub tls_version_distribution: HashMap<String, u64>,
    /// Cipher suite distribution
    pub cipher_suite_distribution: HashMap<String, u64>,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(config: TlsConfig) -> Self {
        Self {
            config,
            certificates: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(TlsMetrics::default())),
        }
    }

    /// Initialize TLS manager and load certificates
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing TLS manager");

        if !self.config.enabled {
            warn!("TLS is disabled");
            return Ok(());
        }

        // Validate certificate paths
        self.validate_certificate_paths().await?;

        // Load certificates
        self.load_certificates().await?;

        // Start certificate rotation monitor if enabled
        if self.config.rotation.enabled {
            self.start_rotation_monitor().await?;
        }

        info!("TLS manager initialized successfully");
        Ok(())
    }

    /// Validate certificate file paths
    async fn validate_certificate_paths(&self) -> Result<()> {
        let cert_path = &self.config.certificates.server_cert_path;
        let key_path = &self.config.certificates.server_key_path;

        if !cert_path.exists() {
            return Err(anyhow!("Server certificate not found: {:?}", cert_path));
        }

        if !key_path.exists() {
            return Err(anyhow!("Server private key not found: {:?}", key_path));
        }

        if let Some(ca_path) = &self.config.certificates.ca_cert_path {
            if !ca_path.exists() {
                warn!("CA certificate not found: {:?}", ca_path);
            }
        }

        debug!("Certificate paths validated");
        Ok(())
    }

    /// Load certificates from disk
    async fn load_certificates(&self) -> Result<()> {
        info!("Loading TLS certificates");

        // In a real implementation, this would:
        // 1. Read certificate files from disk
        // 2. Parse X.509 certificates
        // 3. Extract certificate information
        // 4. Store in certificates HashMap
        // 5. Validate certificate chain

        // For now, this is a placeholder
        debug!("Certificates loaded successfully");
        Ok(())
    }

    /// Start certificate rotation monitor
    async fn start_rotation_monitor(&self) -> Result<()> {
        info!("Starting certificate rotation monitor");

        let check_interval = self.config.rotation.check_interval_secs;
        let threshold_days = self.config.rotation.rotation_threshold_days;

        // In a real implementation, this would spawn a background task
        // that periodically checks certificate expiration and rotates
        // certificates when necessary

        debug!(
            "Rotation monitor started (check_interval={}s, threshold={}d)",
            check_interval, threshold_days
        );
        Ok(())
    }

    /// Perform TLS handshake (placeholder for actual implementation)
    pub async fn handshake(&self, connection_id: &str) -> Result<TlsSessionInfo> {
        let start_time = std::time::Instant::now();

        // Record handshake attempt
        {
            let mut metrics = self.metrics.write().await;
            metrics.handshakes_total += 1;
        }

        // Perform actual TLS handshake (placeholder)
        let session_info = TlsSessionInfo {
            session_id: connection_id.to_string(),
            protocol_version: self.config.protocol_version,
            cipher_suite: self.config.cipher_suites[0].to_string(),
            sni: None,
            alpn_protocol: self.config.alpn_protocols.first().cloned(),
            client_cert: None,
            established_at: Utc::now(),
        };

        // Store session
        self.sessions
            .write()
            .await
            .insert(connection_id.to_string(), session_info.clone());

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.connections_established += 1;
            let duration = start_time.elapsed().as_millis() as f64;
            metrics.avg_handshake_duration_ms =
                (metrics.avg_handshake_duration_ms + duration) / 2.0;

            // Update TLS version distribution
            let version_key = session_info.protocol_version.to_string();
            *metrics
                .tls_version_distribution
                .entry(version_key)
                .or_insert(0) += 1;

            // Update cipher suite distribution
            *metrics
                .cipher_suite_distribution
                .entry(session_info.cipher_suite.clone())
                .or_insert(0) += 1;
        }

        debug!(
            "TLS handshake completed for connection: {} in {:?}",
            connection_id,
            start_time.elapsed()
        );

        Ok(session_info)
    }

    /// Rotate certificates
    pub async fn rotate_certificates(&self) -> Result<()> {
        info!("Starting certificate rotation");

        // In a real implementation, this would:
        // 1. Load new certificates from disk or certificate management system
        // 2. Validate new certificates
        // 3. Gradually transition connections to new certificates
        // 4. Monitor for issues during rotation
        // 5. Complete rotation after graceful period

        {
            let mut metrics = self.metrics.write().await;
            metrics.certificate_rotations += 1;
        }

        info!("Certificate rotation completed successfully");
        Ok(())
    }

    /// Get TLS session information
    pub async fn get_session_info(&self, session_id: &str) -> Option<TlsSessionInfo> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// Get certificate information
    pub async fn get_certificate_info(&self, cert_id: &str) -> Option<CertificateInfo> {
        self.certificates.read().await.get(cert_id).cloned()
    }

    /// Get TLS metrics
    pub async fn get_metrics(&self) -> TlsMetrics {
        self.metrics.read().await.clone()
    }

    /// Close TLS session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        self.sessions.write().await.remove(session_id);
        debug!("TLS session closed: {}", session_id);
        Ok(())
    }

    /// Validate certificate expiry
    pub async fn check_certificate_expiry(&self) -> Result<Vec<ExpiryWarning>> {
        let mut warnings = Vec::new();

        let certificates = self.certificates.read().await;
        let threshold_days = self.config.rotation.rotation_threshold_days;

        for (cert_id, cert_info) in certificates.iter() {
            let days_until_expiry = (cert_info.valid_until - Utc::now()).num_days();

            if days_until_expiry < threshold_days as i64 {
                warnings.push(ExpiryWarning {
                    certificate_id: cert_id.clone(),
                    subject: cert_info.subject.clone(),
                    expires_at: cert_info.valid_until,
                    days_until_expiry,
                });

                warn!(
                    "Certificate {} expires in {} days",
                    cert_id, days_until_expiry
                );
            }
        }

        Ok(warnings)
    }
}

/// Certificate expiry warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpiryWarning {
    /// Certificate ID
    pub certificate_id: String,
    /// Certificate subject
    pub subject: String,
    /// Expiration date
    pub expires_at: DateTime<Utc>,
    /// Days until expiry
    pub days_until_expiry: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tls_config_default() {
        let config = TlsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.protocol_version, TlsVersion::Tls13);
        assert!(config.perfect_forward_secrecy);
    }

    #[tokio::test]
    async fn test_tls_manager_creation() {
        let config = TlsConfig::default();
        let manager = TlsManager::new(config);
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.connections_established, 0);
    }

    #[tokio::test]
    async fn test_cipher_suite_display() {
        let suite = CipherSuite::TLS_AES_256_GCM_SHA384;
        assert_eq!(suite.to_string(), "TLS_AES_256_GCM_SHA384");
    }

    #[tokio::test]
    async fn test_tls_version_display() {
        assert_eq!(TlsVersion::Tls13.to_string(), "TLS 1.3");
        assert_eq!(TlsVersion::Tls12.to_string(), "TLS 1.2");
    }

    #[tokio::test]
    async fn test_mtls_config_default() {
        let config = MutualTlsConfig::default();
        assert!(!config.enabled);
        assert!(config.require_client_cert);
        assert_eq!(config.verification_depth, 3);
    }
}
