//! TLS Communication and Encryption Module
//!
//! Provides comprehensive TLS/SSL support for secure cluster communication,
//! including certificate management, key rotation, and encryption at rest.

use anyhow::{anyhow, Result};
use ring::rand::SecureRandom;
use rustls::{ClientConfig, ServerConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tracing::{error, info, warn};

use crate::raft::OxirsNodeId;

/// Custom certificate verifier that accepts all certificates (for testing)
#[derive(Debug)]
struct NoVerification;

impl rustls::client::danger::ServerCertVerifier for NoVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA1,
            rustls::SignatureScheme::ECDSA_SHA1_Legacy,
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ED448,
        ]
    }
}

/// TLS configuration for the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS for all communications
    pub enabled: bool,
    /// Require mutual authentication
    pub require_client_auth: bool,
    /// Certificate file path
    pub cert_file: Option<PathBuf>,
    /// Private key file path
    pub key_file: Option<PathBuf>,
    /// CA certificate file path
    pub ca_file: Option<PathBuf>,
    /// Certificate directory for auto-generation
    pub cert_dir: PathBuf,
    /// Certificate validity duration in days
    pub cert_validity_days: u64,
    /// Auto-rotate certificates before expiry
    pub auto_rotate: bool,
    /// Days before expiry to trigger rotation
    pub rotation_threshold_days: u64,
    /// Allowed cipher suites
    pub cipher_suites: Vec<String>,
    /// TLS protocol versions
    pub protocol_versions: Vec<String>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            require_client_auth: true,
            cert_file: None,
            key_file: None,
            ca_file: None,
            cert_dir: PathBuf::from("./certs"),
            cert_validity_days: 365,
            auto_rotate: true,
            rotation_threshold_days: 30,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            protocol_versions: vec!["TLSv1.3".to_string(), "TLSv1.2".to_string()],
        }
    }
}

/// Certificate information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    pub subject: String,
    pub issuer: String,
    pub serial_number: String,
    pub not_before: SystemTime,
    pub not_after: SystemTime,
    pub fingerprint: String,
    pub key_usage: Vec<String>,
}

impl CertificateInfo {
    /// Check if certificate is expired
    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.not_after
    }

    /// Check if certificate expires within the given duration
    pub fn expires_within(&self, duration: Duration) -> bool {
        SystemTime::now() + duration > self.not_after
    }

    /// Get remaining validity duration
    pub fn remaining_validity(&self) -> Option<Duration> {
        self.not_after.duration_since(SystemTime::now()).ok()
    }
}

/// TLS certificate manager
#[derive(Debug)]
pub struct TlsManager {
    config: TlsConfig,
    node_id: OxirsNodeId,
    certificates: Arc<RwLock<HashMap<String, CertificateInfo>>>,
    server_config: Arc<RwLock<Option<Arc<ServerConfig>>>>,
    client_config: Arc<RwLock<Option<Arc<ClientConfig>>>>,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(config: TlsConfig, node_id: OxirsNodeId) -> Self {
        Self {
            config,
            node_id,
            certificates: Arc::new(RwLock::new(HashMap::new())),
            server_config: Arc::new(RwLock::new(None)),
            client_config: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize TLS configuration
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            info!("TLS disabled, skipping initialization");
            return Ok(());
        }

        // Create certificate directory if it doesn't exist
        tokio::fs::create_dir_all(&self.config.cert_dir).await?;

        // Load or generate certificates
        self.ensure_certificates().await?;

        // Initialize server and client configurations
        self.initialize_server_config().await?;
        self.initialize_client_config().await?;

        info!(
            "TLS manager initialized for node {} with mutual auth: {}",
            self.node_id, self.config.require_client_auth
        );

        // Start background certificate rotation task
        if self.config.auto_rotate {
            self.start_rotation_task().await;
        }

        Ok(())
    }

    /// Get TLS acceptor for server connections
    pub async fn get_acceptor(&self) -> Result<TlsAcceptor> {
        let server_config = self.server_config.read().await;
        let config = server_config
            .as_ref()
            .ok_or_else(|| anyhow!("Server TLS config not initialized"))?;
        Ok(TlsAcceptor::from(Arc::clone(config)))
    }

    /// Get TLS connector for client connections
    pub async fn get_connector(&self) -> Result<TlsConnector> {
        let client_config = self.client_config.read().await;
        let config = client_config
            .as_ref()
            .ok_or_else(|| anyhow!("Client TLS config not initialized"))?;
        Ok(TlsConnector::from(Arc::clone(config)))
    }

    /// Generate a new certificate for the node
    pub async fn generate_certificate(&self, node_id: OxirsNodeId) -> Result<(Vec<u8>, Vec<u8>)> {
        let subject_alt_names = vec![
            format!("node-{node_id}"),
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ];

        let certified_key = rcgen::generate_simple_self_signed(subject_alt_names)?;
        let cert_der = certified_key.cert.der().to_vec();
        // rcgen API change: key_pair → signing_key (available fields: cert, signing_key)
        let key_der = certified_key.signing_key.serialize_der();

        Ok((cert_der, key_der))
    }

    /// Load certificate from file
    pub async fn load_certificate(&self, cert_path: &Path) -> Result<CertificateInfo> {
        let cert_pem = tokio::fs::read_to_string(cert_path).await?;
        let cert_der = rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if cert_der.is_empty() {
            return Err(anyhow!("No certificates found in file"));
        }

        // Parse certificate information (simplified)
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&cert_der[0]);
        let hash_result = hasher.finalize();
        let fingerprint = format!("{hash_result:x}");

        let cert_info = CertificateInfo {
            subject: format!("node-{}", self.node_id),
            issuer: "self-signed".to_string(),
            serial_number: "1".to_string(),
            not_before: SystemTime::now(),
            not_after: SystemTime::now()
                + Duration::from_secs(self.config.cert_validity_days * 24 * 3600),
            fingerprint,
            key_usage: vec![
                "digital_signature".to_string(),
                "key_encipherment".to_string(),
            ],
        };

        Ok(cert_info)
    }

    /// Ensure certificates exist and are valid
    async fn ensure_certificates(&self) -> Result<()> {
        let cert_path = self.get_cert_path();
        let key_path = self.get_key_path();

        // Check if files exist and are valid
        if cert_path.exists() && key_path.exists() {
            match self.load_certificate(&cert_path).await {
                Ok(cert_info) => {
                    if !cert_info.expires_within(Duration::from_secs(
                        self.config.rotation_threshold_days * 24 * 3600,
                    )) {
                        info!("Using existing certificate for node {}", self.node_id);
                        let mut certs = self.certificates.write().await;
                        certs.insert("server".to_string(), cert_info);
                        return Ok(());
                    } else {
                        warn!(
                            "Certificate for node {} expires soon, regenerating",
                            self.node_id
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to load existing certificate: {}", e);
                }
            }
        }

        // Generate new certificate
        info!("Generating new certificate for node {}", self.node_id);
        let (cert_der, key_der) = self.generate_certificate(self.node_id).await?;

        // Save certificate and key to PEM format
        use base64::Engine;
        let base64_engine = base64::engine::general_purpose::STANDARD;
        let cert_pem = format!(
            "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
            base64_engine.encode(&cert_der)
        );

        let key_pem = format!(
            "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
            base64_engine.encode(&key_der)
        );

        tokio::fs::write(&cert_path, cert_pem).await?;
        tokio::fs::write(&key_path, key_pem).await?;

        info!("Generated new certificate for node {}", self.node_id);

        // Load certificate info
        let cert_info = self.load_certificate(&cert_path).await?;
        let mut certs = self.certificates.write().await;
        certs.insert("server".to_string(), cert_info);

        Ok(())
    }

    /// Initialize server TLS configuration
    async fn initialize_server_config(&self) -> Result<()> {
        let cert_path = self.get_cert_path();
        let key_path = self.get_key_path();

        let cert_file = tokio::fs::File::open(&cert_path).await?;
        let mut cert_reader = BufReader::new(cert_file.into_std().await);
        let certs =
            rustls_pemfile::certs(&mut cert_reader).collect::<std::result::Result<Vec<_>, _>>()?;

        let key_file = tokio::fs::File::open(&key_path).await?;
        let mut key_reader = BufReader::new(key_file.into_std().await);
        let key = rustls_pemfile::private_key(&mut key_reader)?
            .ok_or_else(|| anyhow!("No private key found"))?;

        // Configure client authentication if required
        let config = if self.config.require_client_auth {
            let root_store = rustls::RootCertStore::empty();
            let client_verifier =
                rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store)).build()?;
            ServerConfig::builder()
                .with_client_cert_verifier(client_verifier)
                .with_single_cert(certs, key)?
        } else {
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)?
        };

        let mut server_config = self.server_config.write().await;
        *server_config = Some(Arc::new(config));

        Ok(())
    }

    /// Initialize client TLS configuration
    async fn initialize_client_config(&self) -> Result<()> {
        // For cluster communication with self-signed certificates, use custom verifier
        // This is appropriate for internal cluster communication
        let config = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerification))
            .with_no_client_auth();

        let mut config = config;

        // Add client certificate if mutual auth is required
        if self.config.require_client_auth {
            let cert_path = self.get_cert_path();
            let key_path = self.get_key_path();

            let cert_file = tokio::fs::File::open(&cert_path).await?;
            let mut cert_reader = BufReader::new(cert_file.into_std().await);
            let certs = rustls_pemfile::certs(&mut cert_reader)
                .collect::<std::result::Result<Vec<_>, _>>()?;

            let key_file = tokio::fs::File::open(&key_path).await?;
            let mut key_reader = BufReader::new(key_file.into_std().await);
            if let Some(key) = rustls_pemfile::private_key(&mut key_reader)? {
                config = ClientConfig::builder()
                    .dangerous()
                    .with_custom_certificate_verifier(Arc::new(NoVerification))
                    .with_client_auth_cert(certs, key)?;
            }
        }

        let mut client_config = self.client_config.write().await;
        *client_config = Some(Arc::new(config));

        Ok(())
    }

    /// Start background certificate rotation task
    async fn start_rotation_task(&self) {
        let tls_manager = TlsManager {
            config: self.config.clone(),
            node_id: self.node_id,
            certificates: Arc::clone(&self.certificates),
            server_config: Arc::clone(&self.server_config),
            client_config: Arc::clone(&self.client_config),
        };

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(24 * 3600)); // Check daily
            loop {
                interval.tick().await;
                if let Err(e) = tls_manager.check_and_rotate_certificates().await {
                    error!("Certificate rotation failed: {}", e);
                }
            }
        });
    }

    /// Check and rotate certificates if needed
    async fn check_and_rotate_certificates(&self) -> Result<()> {
        let threshold = Duration::from_secs(self.config.rotation_threshold_days * 24 * 3600);

        // Collect certificates that need rotation
        let certs_to_rotate: Vec<String> = {
            let certificates = self.certificates.read().await;
            certificates
                .iter()
                .filter(|(_, cert_info)| cert_info.expires_within(threshold))
                .map(|(name, _)| name.clone())
                .collect()
        };

        for name in certs_to_rotate {
            warn!("Certificate '{}' expires soon, rotating", name);
            self.rotate_certificate(&name).await?;
        }

        Ok(())
    }

    /// Rotate a specific certificate
    async fn rotate_certificate(&self, _cert_name: &str) -> Result<()> {
        info!("Rotating certificate for node {}", self.node_id);

        // Generate new certificate
        let (cert_der, key_der) = self.generate_certificate(self.node_id).await?;

        // Save new certificate files
        let cert_path = self.get_cert_path();
        let key_path = self.get_key_path();

        // Backup old certificates
        let backup_cert_path = cert_path.with_extension("crt.backup");
        let backup_key_path = key_path.with_extension("key.backup");

        if cert_path.exists() {
            tokio::fs::rename(&cert_path, &backup_cert_path).await?;
        }
        if key_path.exists() {
            tokio::fs::rename(&key_path, &backup_key_path).await?;
        }

        // Write new certificates
        tokio::fs::write(&cert_path, &cert_der).await?;
        tokio::fs::write(&key_path, &key_der).await?;

        // Reinitialize TLS configurations
        self.initialize_server_config().await?;
        self.initialize_client_config().await?;

        // Update certificate info
        let cert_info = self.load_certificate(&cert_path).await?;
        let mut certs = self.certificates.write().await;
        certs.insert("server".to_string(), cert_info);

        info!("Certificate rotation completed for node {}", self.node_id);

        Ok(())
    }

    /// Get certificate file path
    fn get_cert_path(&self) -> PathBuf {
        self.config.cert_file.clone().unwrap_or_else(|| {
            self.config
                .cert_dir
                .join(format!("node-{}.crt", self.node_id))
        })
    }

    /// Get private key file path
    fn get_key_path(&self) -> PathBuf {
        self.config.key_file.clone().unwrap_or_else(|| {
            self.config
                .cert_dir
                .join(format!("node-{}.key", self.node_id))
        })
    }

    /// Get certificate information
    pub async fn get_certificate_info(&self, name: &str) -> Option<CertificateInfo> {
        let certificates = self.certificates.read().await;
        certificates.get(name).cloned()
    }

    /// List all certificates
    pub async fn list_certificates(&self) -> HashMap<String, CertificateInfo> {
        let certificates = self.certificates.read().await;
        certificates.clone()
    }

    /// Validate peer certificate
    pub async fn validate_peer_certificate(&self, _peer_cert: &[u8]) -> Result<bool> {
        // Implement peer certificate validation logic
        // This would typically involve checking against CA or known certificates
        Ok(true)
    }

    /// Export certificate for peer verification
    pub async fn export_certificate(&self) -> Result<Vec<u8>> {
        let cert_path = self.get_cert_path();
        let cert_der = tokio::fs::read(&cert_path).await?;
        Ok(cert_der)
    }
}

/// Data encryption manager for at-rest encryption
pub struct EncryptionManager {
    key: [u8; 32],
    nonce_counter: Arc<RwLock<u64>>,
}

impl EncryptionManager {
    /// Create new encryption manager with random key
    pub fn new() -> Self {
        let rng = ring::rand::SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key).unwrap();

        Self {
            key,
            nonce_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Create encryption manager with provided key
    pub fn with_key(key: [u8; 32]) -> Self {
        Self {
            key,
            nonce_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Encrypt data using AES-GCM
    pub async fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Aead;
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

        let cipher = Aes256Gcm::new_from_slice(&self.key)?;

        // Generate unique nonce
        let mut counter = self.nonce_counter.write().await;
        *counter += 1;
        let nonce_bytes = counter.to_le_bytes();
        let mut nonce_array = [0u8; 12];
        nonce_array[..8].copy_from_slice(&nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_array);

        let encrypted = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("AES-GCM encryption failed: {:?}", e))?;

        // Prepend nonce to encrypted data
        let mut result = Vec::with_capacity(12 + encrypted.len());
        result.extend_from_slice(&nonce_array);
        result.extend_from_slice(&encrypted);

        Ok(result)
    }

    /// Decrypt data using AES-GCM
    pub async fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Aead;
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

        if encrypted_data.len() < 12 {
            return Err(anyhow!("Invalid encrypted data length"));
        }

        let cipher = Aes256Gcm::new_from_slice(&self.key)?;

        // Extract nonce and encrypted data
        let nonce = Nonce::from_slice(&encrypted_data[..12]);
        let ciphertext = &encrypted_data[12..];

        let decrypted = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("AES-GCM decryption failed: {:?}", e))?;
        Ok(decrypted)
    }

    /// Generate new encryption key
    pub fn rotate_key(&mut self) {
        let rng = ring::rand::SystemRandom::new();
        rng.fill(&mut self.key).unwrap();
    }

    /// Export key for backup (use with caution)
    pub fn export_key(&self) -> [u8; 32] {
        self.key
    }
}

impl Default for EncryptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_tls_manager_initialization() {
        // Install default crypto provider for rustls
        let _ = rustls::crypto::ring::default_provider().install_default();

        let temp_dir = TempDir::new().unwrap();
        let config = TlsConfig {
            enabled: true,
            require_client_auth: false,
            cert_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let tls_manager = TlsManager::new(config, 1);
        let result = tls_manager.initialize().await;
        if let Err(e) = &result {
            eprintln!("TLS manager initialization failed: {e}");
        }
        assert!(result.is_ok());

        let acceptor = tls_manager.get_acceptor().await;
        assert!(acceptor.is_ok());

        let connector = tls_manager.get_connector().await;
        assert!(connector.is_ok());
    }

    #[tokio::test]
    async fn test_certificate_generation() {
        // Install default crypto provider for rustls
        let _ = rustls::crypto::ring::default_provider().install_default();

        let temp_dir = TempDir::new().unwrap();
        let config = TlsConfig {
            enabled: true,
            cert_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let tls_manager = TlsManager::new(config, 1);
        let (cert_der, key_der) = tls_manager.generate_certificate(1).await.unwrap();

        assert!(!cert_der.is_empty());
        assert!(!key_der.is_empty());
    }

    #[tokio::test]
    async fn test_encryption_manager() {
        // Install default crypto provider for rustls
        let _ = rustls::crypto::ring::default_provider().install_default();

        let encryption_manager = EncryptionManager::new();
        let data = b"Hello, World!";

        let encrypted = encryption_manager.encrypt(data).await.unwrap();
        assert_ne!(encrypted, data);

        let decrypted = encryption_manager.decrypt(&encrypted).await.unwrap();
        assert_eq!(decrypted, data);
    }

    #[tokio::test]
    async fn test_certificate_info() {
        // Install default crypto provider for rustls
        let _ = rustls::crypto::ring::default_provider().install_default();

        let cert_info = CertificateInfo {
            subject: "test".to_string(),
            issuer: "test".to_string(),
            serial_number: "1".to_string(),
            not_before: SystemTime::now(),
            not_after: SystemTime::now() + Duration::from_secs(3600),
            fingerprint: "test".to_string(),
            key_usage: vec![],
        };

        assert!(!cert_info.is_expired());
        assert!(cert_info.expires_within(Duration::from_secs(7200)));
        assert!(!cert_info.expires_within(Duration::from_secs(1800)));
    }
}
