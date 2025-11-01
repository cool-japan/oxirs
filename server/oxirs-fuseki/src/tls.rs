//! TLS/SSL support for secure HTTPS connections
//!
//! This module provides comprehensive TLS configuration including:
//! - Certificate and private key loading from PEM files
//! - Client certificate authentication (mTLS)
//! - Certificate chain validation
//! - TLS version and cipher suite configuration
//! - OCSP stapling support (future)

use crate::{
    config::TlsConfig,
    error::{FusekiError, FusekiResult},
};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[cfg(feature = "tls")]
use rustls::{
    pki_types::{CertificateDer, PrivateKeyDer},
    ServerConfig,
};
#[cfg(feature = "tls")]
use rustls_pemfile::{certs, pkcs8_private_keys, rsa_private_keys};

/// TLS certificate manager
pub struct TlsManager {
    config: TlsConfig,
}

impl TlsManager {
    /// Create a new TLS manager from configuration
    pub fn new(config: TlsConfig) -> Self {
        TlsManager { config }
    }

    /// Build a rustls ServerConfig from the TLS configuration
    #[cfg(feature = "tls")]
    pub fn build_server_config(&self) -> FusekiResult<Arc<ServerConfig>> {
        info!("Loading TLS certificates...");

        // Load server certificate chain
        let certs = self.load_certificates(&self.config.cert_path)?;
        info!("Loaded {} certificate(s)", certs.len());

        // Load private key
        let private_key = self.load_private_key(&self.config.key_path)?;
        info!("Loaded private key");

        // Configure client certificate authentication if required
        let mut config = if self.config.require_client_cert {
            info!("Client certificate authentication enabled");
            self.build_config_with_client_auth(certs, private_key)?
        } else {
            // Build base server config without client authentication
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, private_key)
                .map_err(|e| {
                    FusekiError::configuration(format!("Failed to build TLS config: {}", e))
                })?
        };

        // Enable ALPN for HTTP/1.1 and HTTP/2
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        Ok(Arc::new(config))
    }

    /// Build server config when TLS feature is disabled
    #[cfg(not(feature = "tls"))]
    pub fn build_server_config(&self) -> FusekiResult<Arc<()>> {
        Err(FusekiError::configuration(
            "TLS support not enabled. Rebuild with --features tls".to_string(),
        ))
    }

    /// Load certificates from PEM file
    #[cfg(feature = "tls")]
    fn load_certificates(&self, path: &Path) -> FusekiResult<Vec<CertificateDer<'static>>> {
        let cert_file = File::open(path).map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to open certificate file {}: {}",
                path.display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(cert_file);

        let certs: Vec<CertificateDer<'static>> = certs(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to parse certificates: {}", e))
            })?;

        if certs.is_empty() {
            return Err(FusekiError::configuration(format!(
                "No certificates found in {}",
                path.display()
            )));
        }

        Ok(certs)
    }

    /// Load private key from PEM file
    #[cfg(feature = "tls")]
    fn load_private_key(&self, path: &Path) -> FusekiResult<PrivateKeyDer<'static>> {
        let key_file = File::open(path).map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to open private key file {}: {}",
                path.display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(key_file);

        // Try PKCS8 format first
        let mut keys: Vec<_> = pkcs8_private_keys(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to parse PKCS8 private key: {}", e))
            })?;

        if let Some(key) = keys.drain(..).next() {
            debug!("Loaded PKCS8 private key");
            return Ok(PrivateKeyDer::Pkcs8(key));
        }

        // Reset reader and try RSA format
        let key_file = File::open(path).map_err(|e| {
            FusekiError::configuration(format!(
                "Failed to reopen private key file {}: {}",
                path.display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(key_file);

        let mut keys: Vec<_> = rsa_private_keys(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to parse RSA private key: {}", e))
            })?;

        if let Some(key) = keys.drain(..).next() {
            debug!("Loaded RSA private key");
            return Ok(PrivateKeyDer::Pkcs1(key));
        }

        Err(FusekiError::configuration(format!(
            "No valid private key found in {}",
            path.display()
        )))
    }

    /// Build ServerConfig with client certificate authentication (mTLS)
    #[cfg(feature = "tls")]
    fn build_config_with_client_auth(
        &self,
        certs: Vec<CertificateDer<'static>>,
        private_key: PrivateKeyDer<'static>,
    ) -> FusekiResult<ServerConfig> {
        if let Some(ca_cert_path) = &self.config.ca_cert_path {
            info!("Loading CA certificate for client authentication");

            let ca_certs = self.load_certificates(ca_cert_path)?;

            // Create client certificate verifier
            let mut root_store = rustls::RootCertStore::empty();
            for cert in ca_certs {
                root_store.add(cert).map_err(|e| {
                    FusekiError::configuration(format!("Failed to add CA certificate: {}", e))
                })?;
            }

            let client_verifier =
                rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store))
                    .build()
                    .map_err(|e| {
                        FusekiError::configuration(format!(
                            "Failed to build client certificate verifier: {}",
                            e
                        ))
                    })?;

            // Build config with client auth
            let config = ServerConfig::builder()
                .with_client_cert_verifier(client_verifier)
                .with_single_cert(certs, private_key)
                .map_err(|e| {
                    FusekiError::configuration(format!(
                        "Failed to build TLS config with client auth: {}",
                        e
                    ))
                })?;

            Ok(config)
        } else {
            warn!("Client certificate authentication required but no CA certificate path provided");
            Err(FusekiError::configuration(
                "Client certificate authentication requires ca_cert_path".to_string(),
            ))
        }
    }

    /// Validate TLS configuration without loading certificates
    pub fn validate(&self) -> FusekiResult<()> {
        // Check if certificate file exists
        if !self.config.cert_path.exists() {
            return Err(FusekiError::configuration(format!(
                "Certificate file not found: {}",
                self.config.cert_path.display()
            )));
        }

        // Check if private key file exists
        if !self.config.key_path.exists() {
            return Err(FusekiError::configuration(format!(
                "Private key file not found: {}",
                self.config.key_path.display()
            )));
        }

        // Check CA certificate if client auth is required
        if self.config.require_client_cert {
            if let Some(ca_path) = &self.config.ca_cert_path {
                if !ca_path.exists() {
                    return Err(FusekiError::configuration(format!(
                        "CA certificate file not found: {}",
                        ca_path.display()
                    )));
                }
            } else {
                return Err(FusekiError::configuration(
                    "Client certificate authentication requires ca_cert_path".to_string(),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_tls_config_validation() {
        // Use paths that don't exist
        let config = TlsConfig {
            cert_path: PathBuf::from("/nonexistent/cert.pem"),
            key_path: PathBuf::from("/nonexistent/key.pem"),
            require_client_cert: false,
            ca_cert_path: None,
        };

        let manager = TlsManager::new(config);
        assert!(manager.validate().is_err());
    }

    #[test]
    fn test_mtls_config_validation() {
        // Use temporary files that actually exist for this test
        let temp_dir = std::env::temp_dir();
        let cert_path = temp_dir.join("test_cert.pem");
        let key_path = temp_dir.join("test_key.pem");

        // Create empty files for testing
        std::fs::write(&cert_path, "").ok();
        std::fs::write(&key_path, "").ok();

        let config = TlsConfig {
            cert_path,
            key_path,
            require_client_cert: true,
            ca_cert_path: None, // Missing CA cert
        };

        let manager = TlsManager::new(config);
        let result = manager.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires ca_cert_path"));
    }
}
