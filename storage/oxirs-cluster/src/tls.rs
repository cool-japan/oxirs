//! TLS Communication and Encryption Module
//!
//! Provides comprehensive TLS/SSL support for secure cluster communication,
//! including certificate management, key rotation, and encryption at rest.

use anyhow::{anyhow, Result};
use rustls::client::danger::ServerCertVerifier;
use rustls::client::WebPkiServerVerifier;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig, SupportedProtocolVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::io::AsyncWriteExt;
use tokio::sync::{OnceCell, RwLock};
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tracing::{error, info, warn};

use crate::raft::OxirsNodeId;

/// Generated certificates start this long before their creation, so peers
/// whose clocks run slightly behind still accept them.
const CLOCK_SKEW_ALLOWANCE: Duration = Duration::from_secs(3600);

/// Upper bound for `cert_validity_days` (100 years).
const MAX_CERT_VALIDITY_DAYS: u64 = 36_500;

const SECONDS_PER_DAY: u64 = 24 * 3600;

/// TLS configuration for the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Run all cluster communication over mutually authenticated TLS.
    ///
    /// Off by default: TLS needs a trust bundle (`ca_file`) that cannot be
    /// defaulted. When off, cluster traffic is neither encrypted nor
    /// authenticated, and starting a network manager logs a warning. When on,
    /// a manager that cannot initialize TLS refuses to start or send rather
    /// than fall back to plaintext.
    pub enabled: bool,
    /// Require mutual authentication
    pub require_client_auth: bool,
    /// Certificate file path
    pub cert_file: Option<PathBuf>,
    /// Private key file path
    pub key_file: Option<PathBuf>,
    /// Trust anchors for authenticating peers: a PEM bundle holding the
    /// cluster CA certificate, the self-signed certificates of every node
    /// (pinning), or both. Required when TLS is enabled; peers whose
    /// certificates do not chain to one of these are rejected. Renewing an
    /// auto-generated certificate keeps its key and subject, so pinned peers
    /// keep trusting it without any change to their bundles.
    pub ca_file: Option<PathBuf>,
    /// Certificate directory for auto-generation
    pub cert_dir: PathBuf,
    /// Validity of generated and renewed certificates, in days (1 to 36 500).
    /// Must exceed `rotation_threshold_days`.
    pub cert_validity_days: u64,
    /// Renew the auto-generated certificate (same key, new validity) at
    /// startup and in a daily background check once it is within
    /// `rotation_threshold_days` of expiry. Operator-provided and CA-issued
    /// certificates are never replaced; they only produce warnings.
    pub auto_rotate: bool,
    /// Days before expiry to trigger rotation
    pub rotation_threshold_days: u64,
    /// Cipher suites to offer and accept, in order of preference. IANA names
    /// (`TLS_AES_128_GCM_SHA256`) and rustls names (`TLS13_AES_128_GCM_SHA256`)
    /// are both accepted; unknown or unsupported names are an error.
    pub cipher_suites: Vec<String>,
    /// TLS protocol versions to allow: `TLSv1.3` and/or `TLSv1.2`. Every
    /// version needs at least one matching cipher suite, and every cipher
    /// suite must belong to an allowed version.
    pub protocol_versions: Vec<String>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
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
            protocol_versions: vec!["TLSv1.3".to_string()],
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

/// What the certificate lifecycle needs to know about a certificate beyond
/// [`CertificateInfo`].
struct CertificateDetails {
    info: CertificateInfo,
    /// Issuer and subject are the same name, as for this node's own
    /// self-signed certificates.
    self_issued: bool,
    subject_key_id: Option<Vec<u8>>,
}

/// Parse the fields of a DER certificate.
fn certificate_details(cert_der: &[u8]) -> Result<CertificateDetails> {
    use x509_parser::prelude::{FromDer, ParsedExtension, X509Certificate};

    let (_, cert) = X509Certificate::from_der(cert_der)
        .map_err(|e| anyhow!("cannot parse certificate: {e}"))?;
    let system_time = |seconds: i64| {
        let magnitude = Duration::from_secs(seconds.unsigned_abs());
        if seconds >= 0 {
            SystemTime::UNIX_EPOCH.checked_add(magnitude)
        } else {
            SystemTime::UNIX_EPOCH.checked_sub(magnitude)
        }
        .ok_or_else(|| anyhow!("certificate date {seconds} is out of range"))
    };

    let mut key_usage = Vec::new();
    if let Some(usage) = cert
        .key_usage()
        .map_err(|e| anyhow!("invalid key usage extension: {e}"))?
    {
        let usage = usage.value;
        let flags = [
            (usage.digital_signature(), "digital_signature"),
            (usage.non_repudiation(), "non_repudiation"),
            (usage.key_encipherment(), "key_encipherment"),
            (usage.data_encipherment(), "data_encipherment"),
            (usage.key_agreement(), "key_agreement"),
            (usage.key_cert_sign(), "key_cert_sign"),
            (usage.crl_sign(), "crl_sign"),
            (usage.encipher_only(), "encipher_only"),
            (usage.decipher_only(), "decipher_only"),
        ];
        key_usage.extend(
            flags
                .iter()
                .filter(|(set, _)| *set)
                .map(|(_, n)| n.to_string()),
        );
    }
    if let Some(usage) = cert
        .extended_key_usage()
        .map_err(|e| anyhow!("invalid extended key usage extension: {e}"))?
    {
        let usage = usage.value;
        let flags = [
            (usage.server_auth, "server_auth"),
            (usage.client_auth, "client_auth"),
            (usage.code_signing, "code_signing"),
            (usage.email_protection, "email_protection"),
            (usage.time_stamping, "time_stamping"),
            (usage.ocsp_signing, "ocsp_signing"),
        ];
        key_usage.extend(
            flags
                .iter()
                .filter(|(set, _)| *set)
                .map(|(_, n)| n.to_string()),
        );
    }

    let subject_key_id =
        cert.extensions()
            .iter()
            .find_map(|extension| match extension.parsed_extension() {
                ParsedExtension::SubjectKeyIdentifier(id) => Some(id.0.to_vec()),
                _ => None,
            });

    Ok(CertificateDetails {
        self_issued: cert.issuer().as_raw() == cert.subject().as_raw(),
        subject_key_id,
        info: CertificateInfo {
            subject: cert.subject().to_string(),
            issuer: cert.issuer().to_string(),
            serial_number: cert.raw_serial_as_string(),
            not_before: system_time(cert.validity().not_before.timestamp())?,
            not_after: system_time(cert.validity().not_after.timestamp())?,
            fingerprint: hex::encode(Sha256::digest(cert_der)),
            key_usage,
        },
    })
}

/// Parse the [`CertificateInfo`] of a DER certificate.
fn certificate_info(cert_der: &[u8]) -> Result<CertificateInfo> {
    Ok(certificate_details(cert_der)?.info)
}

/// Encode DER bytes as a PEM block with 64-column lines (RFC 7468).
fn pem_encode(label: &str, der: &[u8]) -> String {
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(der);
    let mut pem = format!("-----BEGIN {label}-----\n");
    for line in encoded.as_bytes().chunks(64) {
        // Base64 output is ASCII, so the lossy conversion never alters it.
        pem.push_str(&String::from_utf8_lossy(line));
        pem.push('\n');
    }
    pem.push_str(&format!("-----END {label}-----\n"));
    pem
}

/// The first certificate in a PEM file.
async fn read_first_certificate(path: &Path) -> Result<CertificateDer<'static>> {
    let pem = tokio::fs::read(path)
        .await
        .map_err(|e| anyhow!("failed to read certificate {}: {e}", path.display()))?;
    let mut reader = pem.as_slice();
    let first = rustls_pemfile::certs(&mut reader).next();
    let cert = first
        .ok_or_else(|| anyhow!("no certificate found in {}", path.display()))?
        .map_err(|e| anyhow!("cannot parse certificate {}: {e}", path.display()))?;
    Ok(cert)
}

/// Replace `path` with `contents` atomically: write a fresh temporary file
/// next to it, flush it to disk, then rename it into place. With `private`
/// set the file is created readable by its owner only (mode 0600 on Unix),
/// so a private key is never visible to other users, not even briefly.
async fn write_file_atomically(path: &Path, contents: &[u8], private: bool) -> Result<()> {
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("{} is not a file path", path.display()))?
        .to_string_lossy()
        .into_owned();
    let suffix = oxicrypto_rand::random_nonce::<8>()
        .map_err(|e| anyhow!("OS entropy source failed: {e}"))?;
    let temp_path = path.with_file_name(format!(".{file_name}.{}.tmp", hex::encode(suffix)));

    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        if private {
            options.mode(0o600);
        }
    }
    #[cfg(not(unix))]
    let _ = private;

    let written = async {
        let mut file = options.open(&temp_path).await?;
        file.write_all(contents).await?;
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temp_path, path).await
    }
    .await;
    if written.is_err() {
        let _ = tokio::fs::remove_file(&temp_path).await;
    }
    written.map_err(|e| anyhow!("failed to write {}: {e}", path.display()))
}

/// The permission bits of `path` if group or others can access it (Unix);
/// always `None` elsewhere.
fn exposed_key_mode(path: &Path) -> Result<Option<u32>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .map_err(|e| anyhow!("cannot inspect {}: {e}", path.display()))?
            .permissions()
            .mode()
            & 0o777;
        Ok((mode & 0o077 != 0).then_some(mode))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(None)
    }
}

/// Restrict `path` to its owner (mode 0600) on Unix; a no-op elsewhere.
fn restrict_to_owner(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| anyhow!("cannot restrict permissions of {}: {e}", path.display()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// `TLS13_AES_128_GCM_SHA256` (rustls) and `TLS_AES_128_GCM_SHA256` (IANA)
/// name the same suite.
fn canonical_suite_name(name: &str) -> String {
    let upper = name.trim().to_ascii_uppercase();
    match upper.strip_prefix("TLS13_") {
        Some(rest) => format!("TLS_{rest}"),
        None => upper,
    }
}

/// A random positive certificate serial number (RFC 5280 §4.1.2.2).
fn random_serial() -> Result<u64> {
    let bytes = oxicrypto_rand::random_nonce::<8>()
        .map_err(|e| anyhow!("OS entropy source failed: {e}"))?;
    Ok(u64::from_le_bytes(bytes) | 1)
}

/// The template every certificate of node `node_id` follows: issued to
/// `node-{id}` (plus `localhost` and `127.0.0.1`) with both the server- and
/// client-auth extended key usages, so the same identity serves both ends of
/// a mutual-TLS connection.
fn identity_template(node_id: OxirsNodeId) -> oxitls_rcgen::CertificateParamsBuilder {
    let node_san = format!("node-{node_id}");
    oxitls_rcgen::CertificateParamsBuilder::new()
        .with_common_name(node_san.clone())
        .with_dns_names(&[node_san.as_str(), "localhost"])
        .with_ip_addresses(&["127.0.0.1"])
        .with_digital_signature()
        .with_server_auth()
        .with_client_auth()
}

/// TLS certificate manager
#[derive(Debug)]
pub struct TlsManager {
    config: TlsConfig,
    node_id: OxirsNodeId,
    certificates: Arc<RwLock<HashMap<String, CertificateInfo>>>,
    server_config: Arc<RwLock<Option<Arc<ServerConfig>>>>,
    client_config: Arc<RwLock<Option<Arc<ClientConfig>>>>,
    /// Set once [`TlsManager::initialize`] has succeeded; a failed attempt
    /// leaves it unset so that the next call retries.
    initialized: Arc<OnceCell<()>>,
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
            initialized: Arc::new(OnceCell::new()),
        }
    }

    /// The node this manager holds the identity of; its certificate is issued
    /// to `node-{id}`.
    pub fn node_id(&self) -> OxirsNodeId {
        self.node_id
    }

    /// Initialize TLS configuration
    ///
    /// Validates the cipher suite and protocol settings, prepares this node's
    /// identity (generating it on first start, renewing it when it nears
    /// expiry), then loads the trust anchors from [`TlsConfig::ca_file`] and
    /// builds the server and client configurations. Peers are always
    /// authenticated against those anchors: servers by certificate chain and
    /// `node-{id}` name, clients by chain when `require_client_auth` is set.
    ///
    /// Initialization happens once; later calls return immediately, and a
    /// failed attempt is retried by the next call.
    ///
    /// # Errors
    ///
    /// Fails when the cipher suite or protocol settings cannot take effect,
    /// when `ca_file` is not set or holds no usable certificate, when only
    /// one of the certificate and key exists or either cannot be read, or
    /// when the certificate has expired and cannot be renewed.
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            info!("TLS disabled, skipping initialization");
            return Ok(());
        }
        self.initialized
            .get_or_try_init(|| self.initialize_once())
            .await?;
        Ok(())
    }

    async fn initialize_once(&self) -> Result<()> {
        // Settings that cannot take effect are reported before anything is
        // written to disk.
        let provider = self.tls_provider()?;
        self.protocol_versions(&provider)?;

        // Install the Pure Rust crypto provider as the process default before any
        // rustls `ClientConfig`/`ServerConfig` is built (idempotent — a prior
        // install by another component is fine).
        let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

        // Create certificate directory if it doesn't exist
        tokio::fs::create_dir_all(&self.config.cert_dir).await?;

        // Load or generate certificates. This runs before the trust anchors
        // are loaded so that a first start without `ca_file` still leaves the
        // node certificate on disk for building the cluster bundle.
        self.ensure_certificates().await?;

        // Initialize server and client configurations
        let roots = self.load_trust_anchors().await?;
        self.initialize_server_config(&roots).await?;
        self.initialize_client_config(&roots).await?;

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

    /// The crypto provider restricted to `cipher_suites`, in configured order.
    fn tls_provider(&self) -> Result<Arc<CryptoProvider>> {
        let base = oxitls::pure_provider();
        if self.config.cipher_suites.is_empty() {
            return Err(anyhow!(
                "cipher_suites is empty, so no TLS connection could be negotiated"
            ));
        }
        let mut cipher_suites = Vec::with_capacity(self.config.cipher_suites.len());
        for name in &self.config.cipher_suites {
            let wanted = canonical_suite_name(name);
            let suite = base
                .cipher_suites
                .iter()
                .find(|suite| {
                    suite.suite().as_str().map(canonical_suite_name) == Some(wanted.clone())
                })
                .ok_or_else(|| {
                    let supported: Vec<String> = base
                        .cipher_suites
                        .iter()
                        .filter_map(|suite| suite.suite().as_str().map(canonical_suite_name))
                        .collect();
                    anyhow!(
                        "cipher suite `{name}` is unknown or not supported (supported: {})",
                        supported.join(", ")
                    )
                })?;
            if !cipher_suites
                .iter()
                .any(|chosen: &rustls::SupportedCipherSuite| chosen.suite() == suite.suite())
            {
                cipher_suites.push(*suite);
            }
        }
        Ok(Arc::new(CryptoProvider {
            cipher_suites,
            ..(*base).clone()
        }))
    }

    /// The protocol versions in `protocol_versions`, checked against the
    /// cipher suites of `provider` so that every setting takes effect.
    fn protocol_versions(
        &self,
        provider: &CryptoProvider,
    ) -> Result<Vec<&'static SupportedProtocolVersion>> {
        if self.config.protocol_versions.is_empty() {
            return Err(anyhow!(
                "protocol_versions is empty, so no TLS connection could be negotiated"
            ));
        }
        let mut versions: Vec<&'static SupportedProtocolVersion> = Vec::new();
        for name in &self.config.protocol_versions {
            let version = match name.trim().to_ascii_lowercase().as_str() {
                "tlsv1.3" | "tls1.3" | "1.3" => &rustls::version::TLS13,
                "tlsv1.2" | "tls1.2" | "1.2" => &rustls::version::TLS12,
                _ => {
                    return Err(anyhow!(
                        "protocol version `{name}` is not supported (supported: TLSv1.3, TLSv1.2)"
                    ))
                }
            };
            if !versions.contains(&version) {
                versions.push(version);
            }
        }
        for version in &versions {
            if !provider
                .cipher_suites
                .iter()
                .any(|suite| suite.version() == *version)
            {
                return Err(anyhow!(
                    "protocol version {:?} is enabled but none of cipher_suites belongs to it",
                    version.version
                ));
            }
        }
        for suite in &provider.cipher_suites {
            if !versions.contains(&suite.version()) {
                return Err(anyhow!(
                    "cipher suite {:?} belongs to {:?}, which protocol_versions does not enable",
                    suite.suite(),
                    suite.version().version
                ));
            }
        }
        Ok(versions)
    }

    /// The validity window of a certificate issued now: from
    /// [`CLOCK_SKEW_ALLOWANCE`] ago until `cert_validity_days` from now.
    fn validity_window(&self) -> Result<(time::OffsetDateTime, time::OffsetDateTime)> {
        let days = self.config.cert_validity_days;
        if !(1..=MAX_CERT_VALIDITY_DAYS).contains(&days) {
            return Err(anyhow!(
                "cert_validity_days must be between 1 and {MAX_CERT_VALIDITY_DAYS}, not {days}"
            ));
        }
        let now = time::OffsetDateTime::now_utc();
        let validity = Duration::from_secs(days * SECONDS_PER_DAY);
        let not_after = now
            .checked_add(time::Duration::try_from(validity)?)
            .ok_or_else(|| anyhow!("certificate expiry is out of range"))?;
        let not_before = now
            .checked_sub(time::Duration::try_from(CLOCK_SKEW_ALLOWANCE)?)
            .ok_or_else(|| anyhow!("certificate start date is out of range"))?;
        Ok((not_before, not_after))
    }

    /// Generate a new self-signed certificate for the node.
    ///
    /// The certificate follows the node identity template (`node-{id}`,
    /// `localhost` and `127.0.0.1`, server- and client-auth usages) and is
    /// valid for `cert_validity_days`. Peers trust it by including it in
    /// their `ca_file` bundle. Returns the DER certificate and the PKCS#8
    /// DER private key.
    pub async fn generate_certificate(&self, node_id: OxirsNodeId) -> Result<(Vec<u8>, Vec<u8>)> {
        let node_san = format!("node-{node_id}");
        let key_id = oxicrypto_rand::random_nonce::<20>()
            .map_err(|e| anyhow!("OS entropy source failed: {e}"))?;
        let (not_before, not_after) = self.validity_window()?;

        // Pure Rust generation via oxitls-rcgen (Ed25519: fast, no `ring`).
        let params = identity_template(node_id)
            .with_serial_number(random_serial()?)
            .with_subject_key_id(key_id.to_vec())
            .with_validity(not_before, not_after)
            .build()
            .map_err(|e| anyhow!("invalid certificate parameters for {node_san}: {e}"))?;
        let key = oxitls_rcgen::OxiEd25519Key::generate()
            .map_err(|e| anyhow!("failed to generate key for {node_san}: {e}"))?;
        let cert = params
            .self_signed(&key)
            .map_err(|e| anyhow!("failed to self-sign certificate for {node_san}: {e}"))?;

        Ok((cert.der().to_vec(), key.pkcs8_der().to_vec()))
    }

    /// Load certificate from file
    ///
    /// Every field comes from the first certificate in the PEM file.
    pub async fn load_certificate(&self, cert_path: &Path) -> Result<CertificateInfo> {
        let cert_der = read_first_certificate(cert_path).await?;
        certificate_info(&cert_der)
            .map_err(|e| anyhow!("cannot read certificate {}: {e}", cert_path.display()))
    }

    /// Whether the node identity comes from operator-provided `cert_file` /
    /// `key_file`. Such files are never generated or replaced automatically:
    /// a self-signed replacement would not be trusted by any peer.
    fn identity_is_operator_provided(&self) -> bool {
        self.config.cert_file.is_some() || self.config.key_file.is_some()
    }

    fn rotation_threshold(&self) -> Duration {
        Duration::from_secs(
            self.config
                .rotation_threshold_days
                .saturating_mul(SECONDS_PER_DAY),
        )
    }

    /// Ensure this node's certificate and key exist and are usable.
    ///
    /// Operator-provided files must exist and are never written. An
    /// auto-generated identity is created on first start (when neither file
    /// exists), its key is restricted to the owner, and a certificate near
    /// expiry is renewed with the same key when `auto_rotate` is set.
    async fn ensure_certificates(&self) -> Result<()> {
        let cert_path = self.get_cert_path();
        let key_path = self.get_key_path();

        if self.identity_is_operator_provided() {
            if !cert_path.exists() || !key_path.exists() {
                return Err(anyhow!(
                    "configured certificate {} or key {} does not exist",
                    cert_path.display(),
                    key_path.display()
                ));
            }
            if let Some(mode) = exposed_key_mode(&key_path)? {
                warn!(
                    "Private key {} is accessible to other users (mode {mode:o}); restrict it to its owner",
                    key_path.display()
                );
            }
            let cert_info = self.load_certificate(&cert_path).await?;
            self.check_unrenewable_expiry(
                &cert_info,
                "operator-provided certificates are not replaced automatically",
            )?;
            self.certificates
                .write()
                .await
                .insert("server".to_string(), cert_info);
            return Ok(());
        }

        if self.config.rotation_threshold_days >= self.config.cert_validity_days {
            return Err(anyhow!(
                "rotation_threshold_days ({}) must be smaller than cert_validity_days ({}), \
                 or every certificate would need renewing as soon as it is issued",
                self.config.rotation_threshold_days,
                self.config.cert_validity_days
            ));
        }

        match (cert_path.exists(), key_path.exists()) {
            (false, false) => {
                info!("Generating new certificate for node {}", self.node_id);
                let (cert_der, key_der) = self.generate_certificate(self.node_id).await?;
                write_file_atomically(
                    &key_path,
                    pem_encode("PRIVATE KEY", &key_der).as_bytes(),
                    true,
                )
                .await?;
                write_file_atomically(
                    &cert_path,
                    pem_encode("CERTIFICATE", &cert_der).as_bytes(),
                    false,
                )
                .await?;
                info!("Generated new certificate for node {}", self.node_id);
            }
            (true, true) => {
                if let Some(mode) = exposed_key_mode(&key_path)? {
                    restrict_to_owner(&key_path)?;
                    warn!(
                        "Private key {} was accessible to other users (mode {mode:o}); restricted it to mode 600",
                        key_path.display()
                    );
                }
                let cert_info = self.load_certificate(&cert_path).await?;
                if cert_info.expires_within(self.rotation_threshold()) {
                    if self.config.auto_rotate {
                        if let Err(e) = self.renew_identity_certificate().await {
                            self.check_unrenewable_expiry(
                                &cert_info,
                                &format!("renewal failed: {e}"),
                            )?;
                        }
                    } else {
                        self.check_unrenewable_expiry(&cert_info, "auto_rotate is off")?;
                    }
                }
            }
            (true, false) => {
                return Err(anyhow!(
                    "certificate {} has no private key at {}; restore the key, or remove the \
                     certificate to create a new identity (peers must then trust the new certificate)",
                    cert_path.display(),
                    key_path.display()
                ));
            }
            (false, true) => {
                // The key is the identity: a certificate issued for it from the
                // same template is trusted by peers that pinned the lost one.
                // This also completes a first start interrupted between
                // writing the key and the certificate.
                warn!(
                    "Certificate {} is missing; issuing a new one for the existing key {}",
                    cert_path.display(),
                    key_path.display()
                );
                if let Some(mode) = exposed_key_mode(&key_path)? {
                    restrict_to_owner(&key_path)?;
                    warn!(
                        "Private key {} was accessible to other users (mode {mode:o}); restricted it to mode 600",
                        key_path.display()
                    );
                }
                let cert = self.certify_identity_key(None).await?;
                write_file_atomically(
                    &cert_path,
                    pem_encode("CERTIFICATE", &cert).as_bytes(),
                    false,
                )
                .await?;
            }
        }

        let cert_info = self.load_certificate(&cert_path).await?;
        self.certificates
            .write()
            .await
            .insert("server".to_string(), cert_info);
        Ok(())
    }

    /// Handle a certificate near expiry that will not be renewed: warn while
    /// it is still valid, fail once peers would reject it.
    fn check_unrenewable_expiry(&self, cert_info: &CertificateInfo, reason: &str) -> Result<()> {
        let cert_path = self.get_cert_path();
        if cert_info.is_expired() {
            return Err(anyhow!(
                "certificate {} for node {} expired at {:?} and was not renewed ({reason}); \
                 peers reject expired certificates, so replace it",
                cert_path.display(),
                self.node_id,
                cert_info.not_after
            ));
        }
        if cert_info.expires_within(self.rotation_threshold()) {
            warn!(
                "Certificate {} for node {} expires at {:?} and will not be renewed ({reason}); replace it before then",
                cert_path.display(),
                self.node_id,
                cert_info.not_after
            );
        }
        Ok(())
    }

    /// Load the node's Ed25519 private key for re-signing its certificate.
    async fn load_identity_key(&self) -> Result<oxitls_rcgen::OxiEd25519Key> {
        use ed25519_dalek::pkcs8::DecodePrivateKey;

        let key_path = self.get_key_path();
        let pem = tokio::fs::read(&key_path)
            .await
            .map_err(|e| anyhow!("failed to read private key {}: {e}", key_path.display()))?;
        let key = rustls_pemfile::private_key(&mut pem.as_slice())?
            .ok_or_else(|| anyhow!("no private key found in {}", key_path.display()))?;
        let PrivateKeyDer::Pkcs8(pkcs8) = key else {
            return Err(anyhow!(
                "private key {} is not PKCS#8, so its certificate cannot be renewed automatically",
                key_path.display()
            ));
        };
        let signing_key = ed25519_dalek::SigningKey::from_pkcs8_der(pkcs8.secret_pkcs8_der())
            .map_err(|e| {
                anyhow!(
                    "private key {} is not an Ed25519 key, so its certificate cannot be renewed \
                     automatically: {e}",
                    key_path.display()
                )
            })?;
        oxitls_rcgen::OxiEd25519Key::from_seed(signing_key.to_bytes())
            .map_err(|e| anyhow!("cannot use private key {}: {e}", key_path.display()))
    }

    /// Check that peers trusting `current` (pinned as their only anchor) also
    /// accept `renewed`, as a server named `node-{id}` and as a client. Trust
    /// anchors carry no validity period, so this holds even after `current`
    /// itself has expired.
    fn verify_continuity(
        &self,
        current: &CertificateDer<'static>,
        renewed: &CertificateDer<'static>,
    ) -> Result<()> {
        let mut roots = RootCertStore::empty();
        roots.add(current.clone())?;
        let roots = Arc::new(roots);
        let provider = oxitls::pure_provider();
        let now = UnixTime::now();
        let server_name = ServerName::try_from(format!("node-{}", self.node_id))?;

        WebPkiServerVerifier::builder_with_provider(Arc::clone(&roots), Arc::clone(&provider))
            .build()?
            .verify_server_cert(renewed, &[], &server_name, &[], now)
            .map_err(|e| {
                anyhow!("peers pinning the current certificate would reject the renewed one as a server: {e}")
            })?;
        WebPkiClientVerifier::builder_with_provider(roots, provider)
            .build()?
            .verify_client_cert(renewed, &[], now)
            .map_err(|e| {
                anyhow!("peers pinning the current certificate would reject the renewed one as a client: {e}")
            })?;
        Ok(())
    }

    /// Issue a self-signed certificate from the identity template for the
    /// existing key, valid for `cert_validity_days`. `subject_key_id` keeps the
    /// key identifier of an earlier certificate for the same key.
    async fn certify_identity_key(
        &self,
        subject_key_id: Option<Vec<u8>>,
    ) -> Result<CertificateDer<'static>> {
        let key = self.load_identity_key().await?;
        let key_id = match subject_key_id {
            Some(key_id) => key_id,
            None => oxicrypto_rand::random_nonce::<20>()
                .map_err(|e| anyhow!("OS entropy source failed: {e}"))?
                .to_vec(),
        };
        let (not_before, not_after) = self.validity_window()?;
        let cert = identity_template(self.node_id)
            .with_serial_number(random_serial()?)
            .with_subject_key_id(key_id)
            .with_validity(not_before, not_after)
            .build()
            .map_err(|e| anyhow!("invalid certificate parameters: {e}"))?
            .self_signed(&key)
            .map_err(|e| anyhow!("failed to sign certificate for node {}: {e}", self.node_id))?;
        Ok(CertificateDer::from(cert.der().to_vec()))
    }

    /// Re-issue this node's self-signed certificate with a fresh validity
    /// period, keeping its key, subject and key identifier, and write it in
    /// place. The key file is left untouched.
    ///
    /// # Errors
    ///
    /// Refuses operator-provided and CA-issued certificates (only their issuer
    /// can renew them) and keys other than Ed25519, and refuses to install a
    /// certificate that peers pinning the current one would not accept.
    async fn renew_identity_certificate(&self) -> Result<()> {
        let cert_path = self.get_cert_path();
        if self.identity_is_operator_provided() {
            return Err(anyhow!(
                "refusing to replace operator-provided certificate {}",
                cert_path.display()
            ));
        }
        let current = read_first_certificate(&cert_path).await?;
        let details = certificate_details(&current)?;
        if !details.self_issued {
            return Err(anyhow!(
                "certificate {} was issued by {}, not by this node, so only that issuer can renew it",
                cert_path.display(),
                details.info.issuer
            ));
        }
        let renewed = self.certify_identity_key(details.subject_key_id).await?;
        self.verify_continuity(&current, &renewed)?;
        write_file_atomically(
            &cert_path,
            pem_encode("CERTIFICATE", &renewed).as_bytes(),
            false,
        )
        .await?;

        let cert_info = certificate_info(&renewed)?;
        info!(
            "Renewed certificate {} for node {} until {:?}",
            cert_path.display(),
            self.node_id,
            cert_info.not_after
        );
        self.certificates
            .write()
            .await
            .insert("server".to_string(), cert_info);
        Ok(())
    }

    /// Load the node certificate chain and private key from their PEM files.
    async fn load_node_identity(
        &self,
    ) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
        let cert_path = self.get_cert_path();
        let key_path = self.get_key_path();

        let cert_file = tokio::fs::File::open(&cert_path).await?;
        let mut cert_reader = BufReader::new(cert_file.into_std().await);
        let certs =
            rustls_pemfile::certs(&mut cert_reader).collect::<std::result::Result<Vec<_>, _>>()?;
        if certs.is_empty() {
            return Err(anyhow!("no certificate found in {}", cert_path.display()));
        }

        let key_file = tokio::fs::File::open(&key_path).await?;
        let mut key_reader = BufReader::new(key_file.into_std().await);
        let key = rustls_pemfile::private_key(&mut key_reader)?
            .ok_or_else(|| anyhow!("no private key found in {}", key_path.display()))?;

        Ok((certs, key))
    }

    /// Load the trust anchors that peer certificates are verified against.
    ///
    /// # Errors
    ///
    /// Fails when `ca_file` is not set, cannot be read, or holds no usable
    /// certificate. There is no fallback: without anchors peers cannot be
    /// authenticated.
    async fn load_trust_anchors(&self) -> Result<Arc<RootCertStore>> {
        let ca_file = self.config.ca_file.as_ref().ok_or_else(|| {
            anyhow!(
                "TLS is enabled but `ca_file` is not set, so peers cannot be authenticated; \
                 point it at a PEM bundle holding the cluster CA or every node's certificate \
                 (this node's certificate is {})",
                self.get_cert_path().display()
            )
        })?;
        let pem = tokio::fs::read(ca_file)
            .await
            .map_err(|e| anyhow!("failed to read ca_file {}: {e}", ca_file.display()))?;
        let certs = rustls_pemfile::certs(&mut pem.as_slice())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("failed to parse ca_file {}: {e}", ca_file.display()))?;
        if certs.is_empty() {
            return Err(anyhow!(
                "ca_file {} contains no certificates",
                ca_file.display()
            ));
        }

        let mut roots = RootCertStore::empty();
        for cert in certs {
            roots.add(cert).map_err(|e| {
                anyhow!(
                    "ca_file {} holds an unusable certificate: {e}",
                    ca_file.display()
                )
            })?;
        }
        Ok(Arc::new(roots))
    }

    /// Initialize server TLS configuration
    async fn initialize_server_config(&self, roots: &Arc<RootCertStore>) -> Result<()> {
        let (certs, key) = self.load_node_identity().await?;
        let provider = self.tls_provider()?;
        let versions = self.protocol_versions(&provider)?;
        let builder = ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(&versions)?;

        // Configure client authentication if required
        let config = if self.config.require_client_auth {
            let client_verifier =
                WebPkiClientVerifier::builder_with_provider(Arc::clone(roots), provider).build()?;
            builder
                .with_client_cert_verifier(client_verifier)
                .with_single_cert(certs, key)?
        } else {
            builder.with_no_client_auth().with_single_cert(certs, key)?
        };

        let mut server_config = self.server_config.write().await;
        *server_config = Some(Arc::new(config));

        Ok(())
    }

    /// Initialize client TLS configuration
    async fn initialize_client_config(&self, roots: &Arc<RootCertStore>) -> Result<()> {
        let provider = self.tls_provider()?;
        let versions = self.protocol_versions(&provider)?;
        let server_verifier =
            WebPkiServerVerifier::builder_with_provider(Arc::clone(roots), Arc::clone(&provider))
                .build()?;
        let builder = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&versions)?
            .with_webpki_verifier(server_verifier);

        // Present the node certificate when peers require mutual authentication
        let config = if self.config.require_client_auth {
            let (certs, key) = self.load_node_identity().await?;
            builder.with_client_auth_cert(certs, key)?
        } else {
            builder.with_no_client_auth()
        };

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
            initialized: Arc::clone(&self.initialized),
        };

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(SECONDS_PER_DAY)); // Check daily
            loop {
                interval.tick().await;
                if let Err(e) = tls_manager.check_and_rotate_certificates().await {
                    error!("Certificate rotation failed: {}", e);
                }
            }
        });
    }

    /// Renew this node's certificate when it nears expiry.
    ///
    /// Operator-provided certificates are only reported; an auto-generated
    /// one is renewed with the same key (see [`TlsManager::renew_certificate`]).
    async fn check_and_rotate_certificates(&self) -> Result<()> {
        let Some(cert_info) = self.get_certificate_info("server").await else {
            return Ok(());
        };
        if !cert_info.expires_within(self.rotation_threshold()) {
            return Ok(());
        }
        if self.identity_is_operator_provided() {
            warn!(
                "Certificate {} for node {} expires at {:?}; replace it (operator-provided certificates are not regenerated)",
                self.get_cert_path().display(),
                self.node_id,
                cert_info.not_after
            );
            return Ok(());
        }
        warn!(
            "Certificate for node {} expires at {:?}, renewing",
            self.node_id, cert_info.not_after
        );
        self.renew_certificate().await
    }

    /// Renew this node's auto-generated certificate and reload the TLS
    /// configurations to present it.
    ///
    /// The renewed certificate keeps the key and subject, so peers that pin
    /// the current certificate keep trusting the node; it is installed only
    /// after checking exactly that.
    ///
    /// # Errors
    ///
    /// Refuses operator-provided and CA-issued certificates, which only the
    /// operator or the issuer can replace, and keys other than Ed25519.
    async fn renew_certificate(&self) -> Result<()> {
        self.renew_identity_certificate().await?;
        let roots = self.load_trust_anchors().await?;
        self.initialize_server_config(&roots).await?;
        self.initialize_client_config(&roots).await?;
        info!("Certificate renewal completed for node {}", self.node_id);
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

    /// Check a peer certificate against the configured trust anchors.
    ///
    /// Accepts PEM, as returned by [`TlsManager::export_certificate`] (any
    /// further certificates are treated as intermediates), or a single DER
    /// certificate. The chain, validity period and client-auth extended key
    /// usage are verified exactly as for a peer connecting under mutual TLS.
    /// Returns `Ok(false)` when the certificate is not trusted.
    ///
    /// # Errors
    ///
    /// Fails when no trust anchors are configured or `peer_cert` holds no
    /// certificate.
    pub async fn validate_peer_certificate(&self, peer_cert: &[u8]) -> Result<bool> {
        let roots = self.load_trust_anchors().await?;
        // DER certificates start with a SEQUENCE tag; anything else is read as PEM.
        let chain: Vec<CertificateDer<'static>> = if peer_cert.first() == Some(&0x30) {
            vec![CertificateDer::from(peer_cert.to_vec())]
        } else {
            rustls_pemfile::certs(&mut &peer_cert[..])
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let (end_entity, intermediates) = chain
            .split_first()
            .ok_or_else(|| anyhow!("no certificate found in peer_cert"))?;

        let verifier =
            WebPkiClientVerifier::builder_with_provider(roots, oxitls::pure_provider()).build()?;
        match verifier.verify_client_cert(end_entity, intermediates, UnixTime::now()) {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("Peer certificate rejected: {e}");
                Ok(false)
            }
        }
    }

    /// Export certificate for peer verification
    pub async fn export_certificate(&self) -> Result<Vec<u8>> {
        let cert_path = self.get_cert_path();
        let cert_der = tokio::fs::read(&cert_path).await?;
        Ok(cert_der)
    }
}

/// AES-GCM with random 96-bit nonces is limited to 2^32 messages per key
/// (NIST SP 800-38D §8.3), keeping the nonce-collision probability below 2^-32.
const MAX_MESSAGES_PER_KEY: u64 = 1 << 32;

/// Data encryption manager for at-rest encryption
pub struct EncryptionManager {
    key: [u8; 32],
    /// Messages this instance has encrypted under the current key, checked
    /// against [`MAX_MESSAGES_PER_KEY`]. Nonces are random, so instances that
    /// share a key can never collide by construction.
    messages_under_key: AtomicU64,
}

/// Generate a cryptographically random 32-byte AES-256 key.
///
/// Uses the OxiCrypto CSPRNG ([`oxicrypto_rand::random_nonce`]). There is
/// deliberately no fallback: if the OS entropy source fails, key generation
/// fails instead of producing a key an attacker could reconstruct.
fn random_key_32() -> Result<[u8; 32]> {
    oxicrypto_rand::random_nonce::<32>().map_err(|e| anyhow!("OS entropy source failed: {e}"))
}

impl EncryptionManager {
    /// Create new encryption manager with random key
    ///
    /// # Errors
    ///
    /// Returns an error if the OS entropy source fails.
    pub fn new() -> Result<Self> {
        Ok(Self {
            key: random_key_32()?,
            messages_under_key: AtomicU64::new(0),
        })
    }

    /// Create encryption manager with provided key
    pub fn with_key(key: [u8; 32]) -> Self {
        Self {
            key,
            messages_under_key: AtomicU64::new(0),
        }
    }

    /// Encrypt data using AES-256-GCM.
    ///
    /// Every message gets a fresh random 96-bit nonce, so managers sharing a
    /// key (for example one restored from [`EncryptionManager::export_key`]
    /// via [`EncryptionManager::with_key`]) never reuse a nonce. The output is
    /// `nonce (12 bytes) ‖ ciphertext ‖ tag (16 bytes)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS entropy source fails, or once this instance
    /// has encrypted 2^32 messages under the current key; call
    /// [`EncryptionManager::rotate_key`] to continue.
    pub async fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Aead;
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

        let cipher = Aes256Gcm::new_from_slice(&self.key)?;

        self.messages_under_key
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                (used < MAX_MESSAGES_PER_KEY).then_some(used + 1)
            })
            .map_err(|_| {
                anyhow!(
                    "AES-GCM limit of {MAX_MESSAGES_PER_KEY} messages per key reached; rotate the key"
                )
            })?;

        let nonce_array = oxicrypto_rand::random_nonce::<12>()
            .map_err(|e| anyhow!("OS entropy source failed: {e}"))?;
        let nonce = <&Nonce<_>>::from(&nonce_array);

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
        let nonce = <&Nonce<_>>::try_from(&encrypted_data[..12])
            .map_err(|_| anyhow!("Invalid nonce length"))?;
        let ciphertext = &encrypted_data[12..];

        let decrypted = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("AES-GCM decryption failed: {:?}", e))?;
        Ok(decrypted)
    }

    /// Generate new encryption key
    ///
    /// # Errors
    ///
    /// Returns an error if the OS entropy source fails; the current key is
    /// kept in that case.
    pub fn rotate_key(&mut self) -> Result<()> {
        self.key = random_key_32()?;
        *self.messages_under_key.get_mut() = 0;
        Ok(())
    }

    /// Export key for backup (use with caution)
    pub fn export_key(&self) -> [u8; 32] {
        self.key
    }
}

/// Helpers shared by the TLS and network tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn install_provider() {
        let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());
    }

    /// Generate node `node_id`'s certificate and key into `dir` and return a
    /// config that uses them and trusts the PEM bundle at `ca_file`, together
    /// with the node's certificate PEM (for building bundles).
    pub(crate) async fn node_config(
        dir: &Path,
        node_id: OxirsNodeId,
        ca_file: &Path,
        require_client_auth: bool,
    ) -> (TlsConfig, String) {
        let generator = TlsManager::new(TlsConfig::default(), node_id);
        let (cert_der, key_der) = generator.generate_certificate(node_id).await.unwrap();
        let cert_file = dir.join(format!("node-{node_id}.crt"));
        let key_file = dir.join(format!("node-{node_id}.key"));
        let cert_pem = pem_encode("CERTIFICATE", &cert_der);
        std::fs::write(&cert_file, &cert_pem).unwrap();
        std::fs::write(&key_file, pem_encode("PRIVATE KEY", &key_der)).unwrap();
        let config = TlsConfig {
            enabled: true,
            require_client_auth,
            cert_file: Some(cert_file),
            key_file: Some(key_file),
            ca_file: Some(ca_file.to_path_buf()),
            cert_dir: dir.to_path_buf(),
            auto_rotate: false,
            ..Default::default()
        };
        (config, cert_pem)
    }
}

#[cfg(test)]
mod tests;
