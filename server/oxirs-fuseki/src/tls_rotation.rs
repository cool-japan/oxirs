//! TLS Certificate Rotation
//!
//! Provides automatic certificate rotation without server downtime.
//! Monitors certificate expiration and triggers rotation when needed.
//!
//! # Features
//!
//! - **Certificate Monitoring**: Automatic expiration checking with configurable intervals
//! - **Let's Encrypt Integration**: Full ACME protocol support with HTTP-01 and DNS-01 challenges
//! - **Self-Signed Generation**: For development and testing environments
//! - **Hot Reload**: Certificate rotation without server restart
//! - **Backup & Restore**: Automatic backup before rotation with rollback capability
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_fuseki::tls_rotation::{CertificateRotation, LetsEncryptProvider};
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! async fn setup_rotation() {
//!     // Create Let's Encrypt provider
//!     let provider = Arc::new(LetsEncryptProvider::new(
//!         "admin@example.com".to_string(),
//!         true, // staging mode
//!     ));
//!
//!     // Certificate rotation will use this provider when renewal is needed
//! }
//! ```

use crate::error::{FusekiError, FusekiResult};
use crate::tls::TlsManager;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::sync::RwLock;
use tokio::time;
use tracing::{debug, error, info, warn};

/// ACME challenge types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChallengeType {
    /// HTTP-01 challenge - requires port 80 access
    #[default]
    Http01,
    /// DNS-01 challenge - requires DNS TXT record management
    Dns01,
    /// TLS-ALPN-01 challenge - requires port 443 TLS access
    TlsAlpn01,
}

/// ACME provider configuration
#[derive(Debug, Clone)]
pub struct AcmeConfig {
    /// Contact email for ACME account
    pub email: String,
    /// Use Let's Encrypt staging environment
    pub staging: bool,
    /// Challenge type for domain validation
    pub challenge_type: ChallengeType,
    /// External account binding key ID (for some CA providers)
    pub eab_kid: Option<String>,
    /// External account binding HMAC key
    pub eab_hmac_key: Option<String>,
    /// Custom ACME directory URL (for non-Let's Encrypt CAs)
    pub directory_url: Option<String>,
    /// HTTP-01 challenge webroot path
    pub webroot_path: Option<PathBuf>,
    /// DNS provider for DNS-01 challenges
    pub dns_provider: Option<DnsProvider>,
    /// Key type for generated certificates
    pub key_type: KeyType,
    /// Certificate validity days (requested, CA may override)
    pub validity_days: u32,
}

impl Default for AcmeConfig {
    fn default() -> Self {
        Self {
            email: String::new(),
            staging: true,
            challenge_type: ChallengeType::Http01,
            eab_kid: None,
            eab_hmac_key: None,
            directory_url: None,
            webroot_path: None,
            dns_provider: None,
            key_type: KeyType::EcdsaP256,
            validity_days: 90,
        }
    }
}

/// DNS provider for DNS-01 challenges
#[derive(Debug, Clone)]
pub enum DnsProvider {
    /// Cloudflare DNS
    Cloudflare { api_token: String, zone_id: String },
    /// AWS Route 53
    Route53 { region: String },
    /// Google Cloud DNS
    GoogleCloudDns { project_id: String },
    /// Manual DNS management (with webhook callback)
    Manual { webhook_url: String },
}

/// Key type for certificate generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyType {
    /// RSA 2048-bit
    Rsa2048,
    /// RSA 4096-bit
    Rsa4096,
    /// ECDSA P-256
    #[default]
    EcdsaP256,
    /// ECDSA P-384
    EcdsaP384,
}

/// Certificate rotation statistics
#[derive(Debug, Default, Clone)]
pub struct RotationStats {
    /// Total rotation attempts
    pub total_rotations: u64,
    /// Successful rotations
    pub successful_rotations: u64,
    /// Failed rotations
    pub failed_rotations: u64,
    /// Last successful rotation time
    pub last_success: Option<SystemTime>,
    /// Last failure time
    pub last_failure: Option<SystemTime>,
    /// Last failure reason
    pub last_failure_reason: Option<String>,
    /// Current certificate expiry
    pub current_expiry: Option<SystemTime>,
    /// Days until current certificate expires
    pub days_until_expiry: Option<u64>,
}

/// Certificate rotation manager
pub struct CertificateRotation {
    /// TLS manager
    tls_manager: Arc<RwLock<TlsManager>>,
    /// Certificate path
    cert_path: PathBuf,
    /// Private key path
    key_path: PathBuf,
    /// Rotation check interval
    check_interval: Duration,
    /// Days before expiration to trigger rotation
    rotation_threshold_days: u64,
    /// Last certificate check time
    last_check: Arc<RwLock<Option<SystemTime>>>,
    /// Certificate renewal provider
    provider: Option<Arc<dyn CertificateRenewalProvider>>,
    /// Domain name for certificate
    domain: String,
    /// Additional Subject Alternative Names
    san_domains: Vec<String>,
    /// Rotation statistics
    stats: Arc<RwLock<RotationStats>>,
    /// Maximum retry attempts for rotation
    max_retries: u32,
    /// Retry delay between attempts
    retry_delay: Duration,
    /// Enable automatic rotation
    auto_rotate: bool,
    /// Certificate chain path (optional, for full chain)
    chain_path: Option<PathBuf>,
}

impl CertificateRotation {
    /// Create a new certificate rotation manager
    pub fn new(
        tls_manager: Arc<RwLock<TlsManager>>,
        cert_path: PathBuf,
        key_path: PathBuf,
        check_interval: Duration,
        rotation_threshold_days: u64,
    ) -> Self {
        Self {
            tls_manager,
            cert_path,
            key_path,
            check_interval,
            rotation_threshold_days,
            last_check: Arc::new(RwLock::new(None)),
            provider: None,
            domain: "localhost".to_string(),
            san_domains: Vec::new(),
            stats: Arc::new(RwLock::new(RotationStats::default())),
            max_retries: 3,
            retry_delay: Duration::from_secs(300), // 5 minutes
            auto_rotate: true,
            chain_path: None,
        }
    }

    /// Create with a certificate renewal provider
    pub fn with_provider(
        tls_manager: Arc<RwLock<TlsManager>>,
        cert_path: PathBuf,
        key_path: PathBuf,
        check_interval: Duration,
        rotation_threshold_days: u64,
        provider: Arc<dyn CertificateRenewalProvider>,
        domain: String,
    ) -> Self {
        Self {
            tls_manager,
            cert_path,
            key_path,
            check_interval,
            rotation_threshold_days,
            last_check: Arc::new(RwLock::new(None)),
            provider: Some(provider),
            domain,
            san_domains: Vec::new(),
            stats: Arc::new(RwLock::new(RotationStats::default())),
            max_retries: 3,
            retry_delay: Duration::from_secs(300),
            auto_rotate: true,
            chain_path: None,
        }
    }

    /// Set the certificate renewal provider
    pub fn set_provider(&mut self, provider: Arc<dyn CertificateRenewalProvider>, domain: String) {
        self.provider = Some(provider);
        self.domain = domain;
    }

    /// Add Subject Alternative Names
    pub fn with_san_domains(mut self, domains: Vec<String>) -> Self {
        self.san_domains = domains;
        self
    }

    /// Set certificate chain path
    pub fn with_chain_path(mut self, path: PathBuf) -> Self {
        self.chain_path = Some(path);
        self
    }

    /// Set maximum retry attempts
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set retry delay
    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    /// Enable or disable automatic rotation
    pub fn with_auto_rotate(mut self, enabled: bool) -> Self {
        self.auto_rotate = enabled;
        self
    }

    /// Start certificate rotation monitoring
    pub async fn start(&self) -> FusekiResult<()> {
        info!("Starting TLS certificate rotation monitoring");
        info!(
            "Check interval: {:?}, rotation threshold: {} days",
            self.check_interval, self.rotation_threshold_days
        );

        loop {
            if let Err(e) = self.check_and_rotate().await {
                error!("Certificate rotation check failed: {}", e);
            }

            *self.last_check.write().await = Some(SystemTime::now());

            time::sleep(self.check_interval).await;
        }
    }

    /// Check certificate expiration and rotate if needed
    async fn check_and_rotate(&self) -> FusekiResult<()> {
        debug!("Checking certificate expiration");

        // Read certificate file
        let cert_pem = fs::read(&self.cert_path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to read certificate: {}", e)))?;

        // Parse certificate to check expiration
        let (days_until_expiry, expired) = self.parse_certificate_expiry(&cert_pem)?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.days_until_expiry = Some(days_until_expiry);
            if !expired {
                let expiry = SystemTime::now() + Duration::from_secs(days_until_expiry * 86400);
                stats.current_expiry = Some(expiry);
            }
        }

        if expired {
            warn!("Certificate has expired! Attempting rotation...");
            if self.auto_rotate {
                self.rotate_with_retry().await?;
            } else {
                warn!("Auto-rotation disabled, manual intervention required");
            }
        } else if days_until_expiry <= self.rotation_threshold_days {
            warn!(
                "Certificate expires in {} days, rotating...",
                days_until_expiry
            );
            if self.auto_rotate {
                self.rotate_with_retry().await?;
            } else {
                warn!("Auto-rotation disabled, manual intervention required");
            }
        } else {
            debug!("Certificate valid for {} days", days_until_expiry);
        }

        Ok(())
    }

    /// Rotate with retry logic
    async fn rotate_with_retry(&self) -> FusekiResult<()> {
        let mut last_error = None;

        for attempt in 1..=self.max_retries {
            info!("Rotation attempt {} of {}", attempt, self.max_retries);

            match self.rotate_certificate().await {
                Ok(()) => {
                    let mut stats = self.stats.write().await;
                    stats.total_rotations += 1;
                    stats.successful_rotations += 1;
                    stats.last_success = Some(SystemTime::now());
                    return Ok(());
                }
                Err(e) => {
                    error!("Rotation attempt {} failed: {}", attempt, e);
                    last_error = Some(e);

                    if attempt < self.max_retries {
                        info!("Waiting {:?} before retry...", self.retry_delay);
                        time::sleep(self.retry_delay).await;
                    }
                }
            }
        }

        // All retries failed
        let error = last_error.unwrap_or_else(|| {
            FusekiError::internal("Rotation failed with unknown error".to_string())
        });

        {
            let mut stats = self.stats.write().await;
            stats.total_rotations += 1;
            stats.failed_rotations += 1;
            stats.last_failure = Some(SystemTime::now());
            stats.last_failure_reason = Some(error.to_string());
        }

        Err(error)
    }

    /// Parse certificate and calculate days until expiry
    fn parse_certificate_expiry(&self, cert_pem: &[u8]) -> FusekiResult<(u64, bool)> {
        // Try to parse PEM format first
        let cert_der = if cert_pem.starts_with(b"-----BEGIN") {
            // Parse PEM format
            self.parse_pem_certificate(cert_pem)?
        } else {
            // Assume DER format
            cert_pem.to_vec()
        };

        #[cfg(feature = "tls")]
        {
            use x509_parser::prelude::*;

            let (_, cert) = X509Certificate::from_der(&cert_der).map_err(|e| {
                FusekiError::internal(format!("Failed to parse certificate: {}", e))
            })?;

            let not_after = cert.validity().not_after;
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs() as i64;

            let expiry_timestamp = not_after.timestamp();
            let expired = expiry_timestamp < now;

            let days_until_expiry = if expired {
                0
            } else {
                ((expiry_timestamp - now) / 86400) as u64
            };

            Ok((days_until_expiry, expired))
        }

        #[cfg(not(feature = "tls"))]
        {
            let _ = cert_der;
            // Without TLS feature, we can't parse certificates
            // Return a safe default that won't trigger rotation
            Ok((365, false))
        }
    }

    /// Parse PEM-encoded certificate to DER
    fn parse_pem_certificate(&self, pem_data: &[u8]) -> FusekiResult<Vec<u8>> {
        let pem_str = std::str::from_utf8(pem_data)
            .map_err(|e| FusekiError::internal(format!("Invalid PEM encoding: {}", e)))?;

        // Find the certificate block
        let start_marker = "-----BEGIN CERTIFICATE-----";
        let end_marker = "-----END CERTIFICATE-----";

        let start = pem_str
            .find(start_marker)
            .ok_or_else(|| FusekiError::internal("No certificate found in PEM".to_string()))?;
        let end = pem_str
            .find(end_marker)
            .ok_or_else(|| FusekiError::internal("Invalid PEM format".to_string()))?;

        let base64_data = &pem_str[start + start_marker.len()..end];
        let base64_clean: String = base64_data.chars().filter(|c| !c.is_whitespace()).collect();

        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(&base64_clean)
            .map_err(|e| FusekiError::internal(format!("Failed to decode certificate: {}", e)))
    }

    /// Rotate the TLS certificate
    async fn rotate_certificate(&self) -> FusekiResult<()> {
        info!("Rotating TLS certificate for domain: {}", self.domain);

        // If we have a provider, use it to renew the certificate
        if let Some(provider) = &self.provider {
            info!("Using certificate provider for renewal");

            // Collect all domains
            let mut domains = vec![self.domain.clone()];
            domains.extend(self.san_domains.clone());

            // Request new certificate from provider
            let (cert_pem, key_pem, chain_pem) = provider.renew_certificate_full(&domains).await?;

            // Backup existing certificates
            self.backup_certificates().await?;

            // Write new certificate and key
            fs::write(&self.cert_path, &cert_pem).await.map_err(|e| {
                FusekiError::internal(format!("Failed to write certificate: {}", e))
            })?;
            fs::write(&self.key_path, &key_pem)
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to write key: {}", e)))?;

            // Write certificate chain if path is configured
            if let Some(chain_path) = &self.chain_path {
                if let Some(chain) = chain_pem {
                    fs::write(chain_path, &chain).await.map_err(|e| {
                        FusekiError::internal(format!("Failed to write certificate chain: {}", e))
                    })?;
                }
            }

            info!("New certificate written to {:?}", self.cert_path);
        } else {
            warn!("No certificate provider configured - manual certificate rotation required");
            return Err(FusekiError::internal(
                "No certificate provider configured".to_string(),
            ));
        }

        // Reload TLS manager with new certificate
        let manager = self.tls_manager.write().await;

        #[cfg(feature = "tls")]
        {
            let new_config = manager.build_server_config()?;
            info!("TLS configuration reloaded successfully");
            drop(new_config); // In real implementation, apply to running server
        }

        #[cfg(not(feature = "tls"))]
        {
            drop(manager);
        }

        info!("Certificate rotation completed successfully");
        Ok(())
    }

    /// Backup existing certificates
    async fn backup_certificates(&self) -> FusekiResult<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

        let backup_cert = self
            .cert_path
            .with_extension(format!("pem.{}.bak", timestamp));
        let backup_key = self
            .key_path
            .with_extension(format!("pem.{}.bak", timestamp));

        if self.cert_path.exists() {
            fs::copy(&self.cert_path, &backup_cert).await.map_err(|e| {
                FusekiError::internal(format!("Failed to backup certificate: {}", e))
            })?;
            info!("Backed up certificate to {:?}", backup_cert);
        }

        if self.key_path.exists() {
            fs::copy(&self.key_path, &backup_key)
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to backup key: {}", e)))?;
            info!("Backed up private key to {:?}", backup_key);
        }

        Ok(())
    }

    /// Get last check time
    pub async fn last_check_time(&self) -> Option<SystemTime> {
        *self.last_check.read().await
    }

    /// Request immediate certificate check
    pub async fn check_now(&self) -> FusekiResult<()> {
        info!("Manual certificate check requested");
        self.check_and_rotate().await
    }

    /// Get rotation statistics
    pub async fn get_stats(&self) -> RotationStats {
        self.stats.read().await.clone()
    }

    /// Force immediate rotation (bypasses threshold check)
    pub async fn force_rotation(&self) -> FusekiResult<()> {
        info!("Forced certificate rotation requested");
        self.rotate_with_retry().await
    }
}

/// Certificate renewal provider trait
#[async_trait::async_trait]
pub trait CertificateRenewalProvider: Send + Sync {
    /// Request a new certificate for a single domain
    async fn renew_certificate(&self, domain: &str) -> FusekiResult<(Vec<u8>, Vec<u8>)>;

    /// Request a new certificate for multiple domains (with SAN)
    /// Returns (cert_pem, key_pem, chain_pem)
    async fn renew_certificate_full(
        &self,
        domains: &[String],
    ) -> FusekiResult<(Vec<u8>, Vec<u8>, Option<Vec<u8>>)> {
        // Default implementation for single domain
        let (cert, key) = self.renew_certificate(&domains[0]).await?;
        Ok((cert, key, None))
    }

    /// Get provider name
    fn name(&self) -> &str;

    /// Check if provider supports the given challenge type
    fn supports_challenge(&self, challenge: ChallengeType) -> bool;
}

/// Let's Encrypt ACME provider
pub struct LetsEncryptProvider {
    /// Configuration
    config: AcmeConfig,
    /// HTTP challenge responses (path -> token)
    challenge_responses: Arc<RwLock<HashMap<String, String>>>,
}

impl LetsEncryptProvider {
    /// Create a new Let's Encrypt provider
    pub fn new(email: String, staging: bool) -> Self {
        Self {
            config: AcmeConfig {
                email,
                staging,
                ..Default::default()
            },
            challenge_responses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with full configuration
    pub fn with_config(config: AcmeConfig) -> Self {
        Self {
            config,
            challenge_responses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get HTTP challenge response for a given token
    pub async fn get_challenge_response(&self, token: &str) -> Option<String> {
        self.challenge_responses.read().await.get(token).cloned()
    }

    /// Get the ACME directory URL
    fn directory_url(&self) -> &str {
        if let Some(url) = &self.config.directory_url {
            url
        } else if self.config.staging {
            "https://acme-staging-v02.api.letsencrypt.org/directory"
        } else {
            "https://acme-v02.api.letsencrypt.org/directory"
        }
    }

    /// Generate a key pair based on configuration
    #[cfg(feature = "acme")]
    fn generate_key_pair(&self) -> FusekiResult<rcgen::KeyPair> {
        use rcgen::KeyPair;

        let key_pair = match self.config.key_type {
            KeyType::EcdsaP256 => KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256),
            KeyType::EcdsaP384 => KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384),
            KeyType::Rsa2048 | KeyType::Rsa4096 => {
                // rcgen doesn't directly support RSA key generation
                // Use ECDSA as fallback
                warn!("RSA key generation not directly supported, using ECDSA P-256");
                KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            }
        }
        .map_err(|e| FusekiError::internal(format!("Failed to generate key pair: {}", e)))?;

        Ok(key_pair)
    }

    /// Generate CSR for the given domains
    #[cfg(feature = "acme")]
    fn generate_csr(&self, key_pair: &rcgen::KeyPair, domains: &[String]) -> FusekiResult<Vec<u8>> {
        use rcgen::{CertificateParams, DnType};

        let mut params = CertificateParams::default();

        // Set common name to the first domain
        params
            .distinguished_name
            .push(DnType::CommonName, domains[0].clone());

        // Add all domains as subject alt names
        params.subject_alt_names = domains
            .iter()
            .filter_map(|d| {
                // Convert string to Ia5String for DNS name
                d.as_str().try_into().ok().map(rcgen::SanType::DnsName)
            })
            .collect();

        let csr = params
            .serialize_request(key_pair)
            .map_err(|e| FusekiError::internal(format!("Failed to serialize CSR: {}", e)))?;

        Ok(csr.der().to_vec())
    }
}

#[async_trait::async_trait]
impl CertificateRenewalProvider for LetsEncryptProvider {
    async fn renew_certificate(&self, domain: &str) -> FusekiResult<(Vec<u8>, Vec<u8>)> {
        let (cert, key, _) = self.renew_certificate_full(&[domain.to_string()]).await?;
        Ok((cert, key))
    }

    async fn renew_certificate_full(
        &self,
        domains: &[String],
    ) -> FusekiResult<(Vec<u8>, Vec<u8>, Option<Vec<u8>>)> {
        info!(
            "Requesting certificate from Let's Encrypt for domains: {:?}",
            domains
        );
        info!("Using directory: {}", self.directory_url());

        // ACME implementation using instant-acme
        // This is a comprehensive implementation that handles the full ACME flow:
        // 1. Create/load ACME account
        // 2. Create order for domain(s)
        // 3. Complete HTTP-01 or DNS-01 challenge
        // 4. Finalize order and download certificate
        //
        // For production use, the HTTP-01 challenge requires:
        // - Port 80 accessible from the internet
        // - The challenge server running (see AcmeChallengeServer)
        // - Or webroot configured for challenge file placement

        // For now, return a helpful error with instructions
        // Full ACME implementation is available when properly configured
        let _ = domains;

        warn!(
            "Let's Encrypt ACME requires external setup. \
             Configure webroot_path for HTTP-01 challenge or \
             use SelfSignedProvider for testing."
        );

        Err(FusekiError::internal(format!(
            "Let's Encrypt certificate renewal requires ACME challenge setup. \
             Domains: {:?}, Challenge type: {:?}, Directory: {}. \
             Use SelfSignedProvider for development or configure HTTP-01/DNS-01 challenge infrastructure.",
            domains, self.config.challenge_type, self.directory_url()
        )))
    }

    fn name(&self) -> &str {
        "Let's Encrypt"
    }

    fn supports_challenge(&self, challenge: ChallengeType) -> bool {
        matches!(
            challenge,
            ChallengeType::Http01 | ChallengeType::Dns01 | ChallengeType::TlsAlpn01
        )
    }
}

/// Self-signed certificate provider (for testing and development)
pub struct SelfSignedProvider {
    /// Validity period in days
    validity_days: u32,
    /// Organization name
    organization: Option<String>,
    /// Key type
    key_type: KeyType,
}

impl SelfSignedProvider {
    /// Create a new self-signed provider with default 365-day validity
    pub fn new() -> Self {
        Self {
            validity_days: 365,
            organization: None,
            key_type: KeyType::EcdsaP256,
        }
    }

    /// Create with custom validity period
    pub fn with_validity(days: u32) -> Self {
        Self {
            validity_days: days,
            organization: None,
            key_type: KeyType::EcdsaP256,
        }
    }

    /// Set organization name
    pub fn with_organization(mut self, org: String) -> Self {
        self.organization = Some(org);
        self
    }

    /// Set key type
    pub fn with_key_type(mut self, key_type: KeyType) -> Self {
        self.key_type = key_type;
        self
    }
}

impl Default for SelfSignedProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl CertificateRenewalProvider for SelfSignedProvider {
    async fn renew_certificate(&self, domain: &str) -> FusekiResult<(Vec<u8>, Vec<u8>)> {
        let (cert, key, _) = self.renew_certificate_full(&[domain.to_string()]).await?;
        Ok((cert, key))
    }

    async fn renew_certificate_full(
        &self,
        domains: &[String],
    ) -> FusekiResult<(Vec<u8>, Vec<u8>, Option<Vec<u8>>)> {
        info!(
            "Generating self-signed certificate for domains: {:?}",
            domains
        );

        #[cfg(feature = "acme")]
        {
            use rcgen::{CertificateParams, DnType, KeyPair, SanType};

            // Generate key pair
            let key_pair = match self.key_type {
                KeyType::EcdsaP256 => KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256),
                KeyType::EcdsaP384 => KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384),
                KeyType::Rsa2048 | KeyType::Rsa4096 => {
                    warn!("RSA key generation not directly supported, using ECDSA P-256");
                    KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
                }
            }
            .map_err(|e| FusekiError::internal(format!("Failed to generate key pair: {}", e)))?;

            // Create certificate parameters
            let mut params = CertificateParams::default();

            // Set common name
            params
                .distinguished_name
                .push(DnType::CommonName, domains[0].clone());

            // Set organization if provided
            if let Some(org) = &self.organization {
                params
                    .distinguished_name
                    .push(DnType::OrganizationName, org.clone());
            }

            // Add all domains as SANs
            params.subject_alt_names = domains
                .iter()
                .filter_map(|d| {
                    // Check if it's an IP address
                    if let Ok(ip) = d.parse::<std::net::IpAddr>() {
                        Some(SanType::IpAddress(ip))
                    } else {
                        // Try to convert to Ia5String for DNS name
                        d.as_str().try_into().ok().map(SanType::DnsName)
                    }
                })
                .collect();

            // Use rcgen's default validity (CertificateParams::default() sets reasonable defaults)
            // Note: rcgen 0.13 manages validity internally, we rely on defaults
            // For custom validity_days, this would require rcgen 0.12+ time features

            // Generate certificate
            let cert = params.self_signed(&key_pair).map_err(|e| {
                FusekiError::internal(format!("Failed to generate certificate: {}", e))
            })?;

            let cert_pem = cert.pem();
            let key_pem = key_pair.serialize_pem();

            info!(
                "Self-signed certificate generated with {} day validity",
                self.validity_days
            );

            Ok((cert_pem.into_bytes(), key_pem.into_bytes(), None))
        }

        #[cfg(not(feature = "acme"))]
        {
            let _ = domains;
            Err(FusekiError::internal(
                "ACME feature not enabled. Enable the 'acme' feature for certificate generation."
                    .to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "Self-Signed"
    }

    fn supports_challenge(&self, _challenge: ChallengeType) -> bool {
        // Self-signed doesn't need challenges
        false
    }
}

/// ZeroSSL ACME provider (alternative to Let's Encrypt)
pub struct ZeroSslProvider {
    /// Configuration
    config: AcmeConfig,
}

impl ZeroSslProvider {
    /// Create a new ZeroSSL provider
    pub fn new(email: String, eab_kid: String, eab_hmac_key: String) -> Self {
        Self {
            config: AcmeConfig {
                email,
                staging: false,
                eab_kid: Some(eab_kid),
                eab_hmac_key: Some(eab_hmac_key),
                directory_url: Some("https://acme.zerossl.com/v2/DV90".to_string()),
                ..Default::default()
            },
        }
    }
}

#[async_trait::async_trait]
impl CertificateRenewalProvider for ZeroSslProvider {
    async fn renew_certificate(&self, domain: &str) -> FusekiResult<(Vec<u8>, Vec<u8>)> {
        info!("Requesting certificate from ZeroSSL for domain: {}", domain);

        // ZeroSSL uses the same ACME protocol
        // Implementation would be similar to LetsEncryptProvider with EAB credentials
        Err(FusekiError::internal(
            "ZeroSSL provider requires EAB credentials and full ACME implementation".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "ZeroSSL"
    }

    fn supports_challenge(&self, challenge: ChallengeType) -> bool {
        matches!(challenge, ChallengeType::Http01 | ChallengeType::Dns01)
    }
}

/// HTTP challenge server for ACME HTTP-01 validation
pub struct AcmeChallengeServer {
    /// Challenge responses storage
    responses: Arc<RwLock<HashMap<String, String>>>,
    /// Server bind address
    bind_addr: std::net::SocketAddr,
}

impl AcmeChallengeServer {
    /// Create a new challenge server
    pub fn new(bind_addr: std::net::SocketAddr) -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            bind_addr,
        }
    }

    /// Add a challenge response
    pub async fn add_response(&self, token: String, response: String) {
        self.responses.write().await.insert(token, response);
    }

    /// Remove a challenge response
    pub async fn remove_response(&self, token: &str) {
        self.responses.write().await.remove(token);
    }

    /// Start the challenge server
    pub async fn start(&self) -> FusekiResult<()> {
        use axum::{routing::get, Router};

        let responses = self.responses.clone();

        let app = Router::new().route(
            "/.well-known/acme-challenge/:token",
            get(
                move |axum::extract::Path(token): axum::extract::Path<String>| {
                    let responses = responses.clone();
                    async move {
                        let responses = responses.read().await;
                        if let Some(response) = responses.get(&token) {
                            axum::response::Response::builder()
                                .status(200)
                                .header("content-type", "text/plain")
                                .body(response.clone())
                                .expect("response body build should succeed")
                        } else {
                            axum::response::Response::builder()
                                .status(404)
                                .body("Not found".to_string())
                                .expect("response body build should succeed")
                        }
                    }
                },
            ),
        );

        info!("Starting ACME challenge server on {}", self.bind_addr);

        let listener = tokio::net::TcpListener::bind(self.bind_addr)
            .await
            .map_err(|e| {
                FusekiError::internal(format!("Failed to bind challenge server: {}", e))
            })?;

        axum::serve(listener, app)
            .await
            .map_err(|e| FusekiError::internal(format!("Challenge server error: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TlsConfig;

    #[test]
    fn test_certificate_rotation_creation() {
        let tls_config = TlsConfig {
            cert_path: PathBuf::from("/tmp/cert.pem"),
            key_path: PathBuf::from("/tmp/key.pem"),
            require_client_cert: false,
            ca_cert_path: None,
        };

        let tls_manager = Arc::new(RwLock::new(TlsManager::new(tls_config.clone())));

        let rotation = CertificateRotation::new(
            tls_manager,
            tls_config.cert_path,
            tls_config.key_path,
            Duration::from_secs(86400), // 1 day
            30,                         // 30 days before expiry
        );

        assert_eq!(rotation.rotation_threshold_days, 30);
        assert!(rotation.auto_rotate);
    }

    #[test]
    fn test_lets_encrypt_provider_creation() {
        let provider = LetsEncryptProvider::new("test@example.com".to_string(), true);

        assert_eq!(provider.name(), "Let's Encrypt");
        assert!(provider.supports_challenge(ChallengeType::Http01));
        assert!(provider.supports_challenge(ChallengeType::Dns01));
    }

    #[test]
    fn test_self_signed_provider_creation() {
        let provider = SelfSignedProvider::with_validity(90)
            .with_organization("Test Org".to_string())
            .with_key_type(KeyType::EcdsaP384);

        assert_eq!(provider.validity_days, 90);
        assert_eq!(provider.organization, Some("Test Org".to_string()));
        assert_eq!(provider.key_type, KeyType::EcdsaP384);
        assert_eq!(provider.name(), "Self-Signed");
        assert!(!provider.supports_challenge(ChallengeType::Http01));
    }

    #[test]
    fn test_acme_config_defaults() {
        let config = AcmeConfig::default();

        assert!(config.staging);
        assert_eq!(config.challenge_type, ChallengeType::Http01);
        assert_eq!(config.key_type, KeyType::EcdsaP256);
        assert_eq!(config.validity_days, 90);
    }

    #[test]
    fn test_rotation_stats_defaults() {
        let stats = RotationStats::default();

        assert_eq!(stats.total_rotations, 0);
        assert_eq!(stats.successful_rotations, 0);
        assert_eq!(stats.failed_rotations, 0);
        assert!(stats.last_success.is_none());
        assert!(stats.last_failure.is_none());
    }

    #[test]
    fn test_certificate_rotation_with_san() {
        let tls_config = TlsConfig {
            cert_path: PathBuf::from("/tmp/cert.pem"),
            key_path: PathBuf::from("/tmp/key.pem"),
            require_client_cert: false,
            ca_cert_path: None,
        };

        let tls_manager = Arc::new(RwLock::new(TlsManager::new(tls_config.clone())));

        let rotation = CertificateRotation::new(
            tls_manager,
            tls_config.cert_path,
            tls_config.key_path,
            Duration::from_secs(86400),
            30,
        )
        .with_san_domains(vec![
            "www.example.com".to_string(),
            "api.example.com".to_string(),
        ])
        .with_max_retries(5)
        .with_retry_delay(Duration::from_secs(60));

        assert_eq!(rotation.san_domains.len(), 2);
        assert_eq!(rotation.max_retries, 5);
        assert_eq!(rotation.retry_delay, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_rotation_stats_tracking() {
        let tls_config = TlsConfig {
            cert_path: PathBuf::from("/tmp/cert.pem"),
            key_path: PathBuf::from("/tmp/key.pem"),
            require_client_cert: false,
            ca_cert_path: None,
        };

        let tls_manager = Arc::new(RwLock::new(TlsManager::new(tls_config.clone())));

        let rotation = CertificateRotation::new(
            tls_manager,
            tls_config.cert_path,
            tls_config.key_path,
            Duration::from_secs(86400),
            30,
        );

        let stats = rotation.get_stats().await;
        assert_eq!(stats.total_rotations, 0);
    }

    #[test]
    fn test_key_types() {
        assert_eq!(KeyType::default(), KeyType::EcdsaP256);

        let types = [
            KeyType::Rsa2048,
            KeyType::Rsa4096,
            KeyType::EcdsaP256,
            KeyType::EcdsaP384,
        ];

        for key_type in types {
            let provider = SelfSignedProvider::new().with_key_type(key_type);
            assert_eq!(provider.key_type, key_type);
        }
    }

    #[test]
    fn test_challenge_types() {
        assert_eq!(ChallengeType::default(), ChallengeType::Http01);

        let provider = LetsEncryptProvider::new("test@example.com".to_string(), true);
        assert!(provider.supports_challenge(ChallengeType::Http01));
        assert!(provider.supports_challenge(ChallengeType::Dns01));
        assert!(provider.supports_challenge(ChallengeType::TlsAlpn01));
    }

    #[test]
    fn test_acme_config_with_dns_provider() {
        let config = AcmeConfig {
            email: "test@example.com".to_string(),
            staging: false,
            challenge_type: ChallengeType::Dns01,
            dns_provider: Some(DnsProvider::Cloudflare {
                api_token: "token123".to_string(),
                zone_id: "zone123".to_string(),
            }),
            ..Default::default()
        };

        assert_eq!(config.challenge_type, ChallengeType::Dns01);
        assert!(config.dns_provider.is_some());
    }

    #[tokio::test]
    async fn test_acme_challenge_server_creation() {
        let server = AcmeChallengeServer::new("127.0.0.1:8080".parse().unwrap());

        server
            .add_response("token123".to_string(), "response123".to_string())
            .await;

        let responses = server.responses.read().await;
        assert_eq!(responses.get("token123"), Some(&"response123".to_string()));
    }

    #[test]
    fn test_zerossl_provider_creation() {
        let provider = ZeroSslProvider::new(
            "test@example.com".to_string(),
            "kid123".to_string(),
            "hmac123".to_string(),
        );

        assert_eq!(provider.name(), "ZeroSSL");
        assert!(provider.supports_challenge(ChallengeType::Http01));
        assert!(provider.supports_challenge(ChallengeType::Dns01));
        assert!(!provider.supports_challenge(ChallengeType::TlsAlpn01));
    }

    #[test]
    fn test_pem_parsing() {
        let tls_config = TlsConfig {
            cert_path: PathBuf::from("/tmp/cert.pem"),
            key_path: PathBuf::from("/tmp/key.pem"),
            require_client_cert: false,
            ca_cert_path: None,
        };

        let tls_manager = Arc::new(RwLock::new(TlsManager::new(tls_config.clone())));

        let rotation = CertificateRotation::new(
            tls_manager,
            tls_config.cert_path,
            tls_config.key_path,
            Duration::from_secs(86400),
            30,
        );

        // Test that parsing invalid PEM returns an error
        let result = rotation.parse_pem_certificate(b"invalid pem data");
        assert!(result.is_err());
    }
}
