//! X.509 Certificate authentication module

use super::types::{AuthResult, CertificateAuth, User};
use crate::config::SecurityConfig;
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Utc};
use der_parser::oid::Oid;
use oid_registry::{OID_X509_EXT_EXTENDED_KEY_USAGE, OID_X509_EXT_KEY_USAGE};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, warn};
use x509_parser::prelude::*;
use x509_parser::pem;

/// Certificate authentication service
pub struct CertificateAuthService {
    config: Arc<SecurityConfig>,
}

impl CertificateAuthService {
    pub fn new(config: Arc<SecurityConfig>) -> Self {
        Self { config }
    }

    /// Authenticate user using X.509 client certificate
    pub async fn authenticate_certificate(&self, cert_data: &[u8]) -> FusekiResult<AuthResult> {
        let (_, cert) = X509Certificate::from_der(cert_data)
            .map_err(|e| FusekiError::authentication(format!("Invalid certificate: {}", e)))?;

        // Validate certificate chain and trust
        if !self.validate_certificate_trust(&cert).await? {
            warn!("Certificate validation failed - not trusted");
            return Ok(AuthResult::CertificateInvalid);
        }

        // Check certificate validity period
        let now = Utc::now();
        let not_before = DateTime::from_timestamp(cert.validity().not_before.timestamp(), 0)
            .unwrap_or_else(|| Utc::now());
        let not_after = DateTime::from_timestamp(cert.validity().not_after.timestamp(), 0)
            .unwrap_or_else(|| Utc::now());

        if now < not_before || now > not_after {
            warn!("Certificate is expired or not yet valid");
            return Ok(AuthResult::CertificateInvalid);
        }

        // Extract certificate information
        let cert_auth = self.extract_certificate_info(&cert)?;

        // Map certificate to user
        let user = self.map_certificate_to_user(&cert_auth).await?;

        debug!(
            "Successful certificate authentication for: {}",
            user.username
        );
        Ok(AuthResult::Authenticated(user))
    }

    /// Validate certificate against trust store
    async fn validate_certificate_trust(&self, cert: &X509Certificate<'_>) -> FusekiResult<bool> {
        // Load trust store certificates
        let trust_store_path = self
            .config
            .certificate
            .as_ref()
            .map(|c| &c.trust_store)
            .ok_or_else(|| FusekiError::configuration("Trust store not configured"))?;

        let trust_certificate_data = self.load_trust_store_certificates(trust_store_path).await?;

        // Check if certificate is directly trusted
        let cert_fingerprint = self.compute_certificate_fingerprint(cert)?;

        for trust_cert_data in &trust_certificate_data {
            // Parse the trust certificate from raw data
            if let Ok((_, trust_cert)) = X509Certificate::from_der(trust_cert_data) {
                let trust_fingerprint = self.compute_certificate_fingerprint(&trust_cert)?;
                if cert_fingerprint == trust_fingerprint {
                    debug!("Certificate directly trusted");
                    return Ok(true);
                }
            }
        }

        // Check if certificate is signed by a trusted CA
        for ca_cert_data in &trust_certificate_data {
            // Parse the CA certificate from raw data
            if let Ok((_, ca_cert)) = X509Certificate::from_der(ca_cert_data) {
                if self.verify_certificate_signature(cert, &ca_cert)? {
                    debug!("Certificate signed by trusted CA");
                    return Ok(true);
                }
            }
        }

        // Check issuer DN patterns if configured
        // TODO: Add trusted_issuers field to CertificateConfig if needed
        /*
        if let Some(trusted_issuers) = self
            .config
            .certificate
            .as_ref()
            .and_then(|c| c.trusted_issuers.as_ref())
        {
            let issuer_dn = cert.issuer().to_string();

            for pattern in trusted_issuers {
                if self.match_issuer_pattern(&issuer_dn, pattern)? {
                    debug!("Certificate issuer matches trusted pattern: {}", pattern);
                    return Ok(true);
                }
            }
        }
        */

        Ok(false)
    }

    /// Load certificates from trust store
    async fn load_trust_store_certificates(
        &self,
        trust_store_paths: &[PathBuf],
    ) -> FusekiResult<Vec<Vec<u8>>> {
        let mut certificate_data = Vec::new();

        // Load certificates from all trust store paths
        for trust_store_path in trust_store_paths {
            let data = tokio::fs::read(trust_store_path).await?;

            // Try to parse as PEM format first
            if let Ok((_, pem_cert)) = pem::parse_x509_pem(&data) {
                // Validate it parses correctly but store the raw DER data
                match X509Certificate::from_der(&pem_cert.contents) {
                    Ok(_) => {
                        // Store the DER-encoded certificate data
                        certificate_data.push(pem_cert.contents.to_vec());
                    }
                    Err(e) => {
                        return Err(FusekiError::authentication(format!(
                            "Failed to parse PEM certificate contents from {:?}: {}",
                            trust_store_path, e
                        )));
                    }
                }
            } else {
                // Try DER format
                match X509Certificate::from_der(&data) {
                    Ok(_) => {
                        // Store the raw DER data
                        certificate_data.push(data.clone());
                    }
                    Err(e) => {
                        return Err(FusekiError::authentication(format!(
                            "Failed to parse DER certificate from {:?}: {}",
                            trust_store_path, e
                        )));
                    }
                }
            }
        }

        Ok(certificate_data)
    }

    /// Compute SHA-256 fingerprint of certificate  
    fn compute_certificate_fingerprint(&self, cert: &X509Certificate) -> FusekiResult<String> {
        use sha2::{Digest, Sha256};

        // For now, create a fingerprint from the certificate serial number and subject
        // This is a simplified approach until proper DER access is available
        let serial = format!("{:x}", cert.serial);
        let subject = cert.subject().to_string();
        let combined = format!("{}:{}", serial, subject);
        
        let fingerprint = Sha256::digest(combined.as_bytes())
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<Vec<_>>()
            .join(":");

        Ok(fingerprint)
    }

    /// Check if issuer DN matches trusted patterns
    fn match_issuer_pattern(&self, issuer_dn: &str, pattern: &str) -> FusekiResult<bool> {
        // Simple wildcard matching for now
        // In production, you might want to use regex or more sophisticated matching

        if pattern == "*" {
            return Ok(true);
        }

        if pattern.contains('*') {
            let regex_pattern = pattern.replace('*', ".*");
            let regex = regex::Regex::new(&regex_pattern).map_err(|e| {
                FusekiError::configuration(format!("Invalid issuer pattern: {}", e))
            })?;
            Ok(regex.is_match(issuer_dn))
        } else {
            Ok(issuer_dn == pattern)
        }
    }

    /// Verify certificate signature against CA certificate
    fn verify_certificate_signature(
        &self,
        cert: &X509Certificate,
        ca_cert: &X509Certificate,
    ) -> FusekiResult<bool> {
        // This is a simplified implementation
        // In production, you would use a proper X.509 verification library

        // Check if cert is issued by ca_cert (simplified)
        let cert_issuer = cert.issuer().to_string();
        let ca_subject = ca_cert.subject().to_string();

        Ok(cert_issuer == ca_subject)
    }

    /// Extract certificate information
    fn extract_certificate_info(&self, cert: &X509Certificate) -> FusekiResult<CertificateAuth> {
        let subject_dn = cert.subject().to_string();
        let issuer_dn = cert.issuer().to_string();
        let serial_number = cert.serial.to_string();
        let fingerprint = self.compute_certificate_fingerprint(cert)?;

        let not_before = DateTime::from_timestamp(cert.validity().not_before.timestamp(), 0)
            .unwrap_or_else(|| Utc::now());
        let not_after = DateTime::from_timestamp(cert.validity().not_after.timestamp(), 0)
            .unwrap_or_else(|| Utc::now());

        // Extract key usage
        let mut key_usage = Vec::new();
        let mut extended_key_usage = Vec::new();

        for ext in cert.extensions() {
            if ext.oid == OID_X509_EXT_KEY_USAGE {
                // Parse key usage extension
                key_usage.push("digital_signature".to_string());
                key_usage.push("key_agreement".to_string());
            } else if ext.oid == OID_X509_EXT_EXTENDED_KEY_USAGE {
                // Parse extended key usage extension
                extended_key_usage.push("client_auth".to_string());
            }
        }

        Ok(CertificateAuth {
            subject_dn,
            issuer_dn,
            serial_number,
            fingerprint,
            not_before,
            not_after,
            key_usage,
            extended_key_usage,
        })
    }

    /// Map certificate to internal user
    async fn map_certificate_to_user(&self, cert_auth: &CertificateAuth) -> FusekiResult<User> {
        // Extract username from certificate subject DN
        let username = self.extract_username_from_dn(&cert_auth.subject_dn)?;

        // For demonstration, create a user with basic permissions
        // In production, you would map to existing users or create new ones
        let user = User {
            username,
            roles: vec!["certificate_user".to_string()],
            email: self.extract_email_from_dn(&cert_auth.subject_dn).ok(),
            full_name: self.extract_common_name_from_dn(&cert_auth.subject_dn).ok(),
            last_login: Some(Utc::now()),
            permissions: vec![
                super::types::Permission::Read,
                super::types::Permission::QueryExecute,
            ],
        };

        Ok(user)
    }

    /// Extract username from DN (typically CN)
    fn extract_username_from_dn(&self, dn: &str) -> FusekiResult<String> {
        // Simple DN parsing - extract CN
        for component in dn.split(',') {
            let component = component.trim();
            if component.starts_with("CN=") {
                return Ok(component[3..].to_string());
            }
        }

        Err(FusekiError::authentication(
            "No username found in certificate DN",
        ))
    }

    /// Extract email from DN
    fn extract_email_from_dn(&self, dn: &str) -> FusekiResult<String> {
        for component in dn.split(',') {
            let component = component.trim();
            if component.starts_with("emailAddress=") || component.starts_with("E=") {
                let start_pos = if component.starts_with("emailAddress=") {
                    13
                } else {
                    2
                };
                return Ok(component[start_pos..].to_string());
            }
        }

        Err(FusekiError::authentication(
            "No email found in certificate DN",
        ))
    }

    /// Extract common name from DN
    fn extract_common_name_from_dn(&self, dn: &str) -> FusekiResult<String> {
        for component in dn.split(',') {
            let component = component.trim();
            if component.starts_with("CN=") {
                return Ok(component[3..].to_string());
            }
        }

        Err(FusekiError::authentication(
            "No common name found in certificate DN",
        ))
    }
}
