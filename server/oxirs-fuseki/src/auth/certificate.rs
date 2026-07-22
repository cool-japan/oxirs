//! X.509 Certificate authentication module

use super::types::{AuthResult, CertificateAuth, User};
use crate::config::SecurityConfig;
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Utc};
use oid_registry::{OID_X509_EXT_EXTENDED_KEY_USAGE, OID_X509_EXT_KEY_USAGE};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, warn};
use x509_parser::pem;
use x509_parser::prelude::*;

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
            .map_err(|e| FusekiError::authentication(format!("Invalid certificate: {e}")))?;

        // Validate certificate chain and trust
        if !self.validate_certificate_trust(&cert).await? {
            warn!("Certificate validation failed - not trusted");
            return Ok(AuthResult::CertificateInvalid);
        }

        // Check certificate validity period
        let now = Utc::now();
        let not_before = DateTime::from_timestamp(cert.validity().not_before.timestamp(), 0)
            .unwrap_or_else(Utc::now);
        let not_after = DateTime::from_timestamp(cert.validity().not_after.timestamp(), 0)
            .unwrap_or_else(Utc::now);

        if now < not_before || now > not_after {
            warn!("Certificate is expired or not yet valid");
            return Ok(AuthResult::CertificateInvalid);
        }

        // Revocation checking (CRL/OCSP), gated on the configured flags. A cert
        // that is revoked — or that cannot be revocation-checked when checking is
        // required — must NOT authenticate (fail closed). See `check_revocation`.
        if !self.check_revocation(&cert).await? {
            warn!("Certificate rejected by revocation check");
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

    /// Validate that the configured revocation policy is one this build can
    /// actually honour, failing loud otherwise (so `check_ocsp` / `check_crl`
    /// never give a false sense of security). Cert-independent, so it is unit
    /// testable without a certificate fixture.
    fn revocation_config_check(&self) -> FusekiResult<()> {
        let Some(cert_cfg) = self.config.certificate.as_ref() else {
            return Ok(());
        };
        if cert_cfg.check_ocsp {
            return Err(FusekiError::configuration(
                "certificate.check_ocsp is enabled but OCSP revocation checking is not implemented; \
                 disable check_ocsp or use CRL checking. Failing closed to avoid a false sense of \
                 revocation security.",
            ));
        }
        if cert_cfg.check_crl {
            if cert_cfg.crl_sources.is_empty() {
                return Err(FusekiError::configuration(
                    "certificate.check_crl is enabled but no certificate.crl_sources are \
                     configured; cannot verify revocation. Failing closed.",
                ));
            }
            for source in &cert_cfg.crl_sources {
                if source.starts_with("http://") || source.starts_with("https://") {
                    return Err(FusekiError::configuration(format!(
                        "certificate.crl_sources entry '{source}' is a URL, but network CRL \
                         fetching is not implemented; use a local CRL file path. Failing closed."
                    )));
                }
            }
        }
        Ok(())
    }

    /// Perform configured revocation checking for `cert`.
    ///
    /// Honours the previously-ignored `check_crl` / `check_ocsp` flags and fails
    /// **closed**:
    /// - `check_ocsp = true` → OCSP requires a network responder this build does
    ///   not implement, so rather than silently skipping revocation (a false sense
    ///   of security) it returns an error and authentication fails.
    /// - `check_crl = true` → each `crl_sources` entry is loaded and parsed. A
    ///   local file path is read and checked; a `http(s)://` URL cannot be fetched
    ///   (no network CRL fetch implemented) so it fails closed. If the cert's
    ///   serial appears in any CRL, the certificate is revoked (`Ok(false)`).
    /// - both flags off → no revocation checking is required, returns `Ok(true)`.
    async fn check_revocation(&self, cert: &X509Certificate<'_>) -> FusekiResult<bool> {
        let Some(cert_cfg) = self.config.certificate.as_ref() else {
            return Ok(true);
        };

        // Config-level fail-loud policy (OCSP unimplemented, CRL misconfig).
        self.revocation_config_check()?;

        if !cert_cfg.check_crl {
            return Ok(true);
        }

        let cert_serial = cert.raw_serial().to_vec();
        for source in &cert_cfg.crl_sources {
            let data = tokio::fs::read(source).await.map_err(|e| {
                FusekiError::configuration(format!("failed to read CRL source '{source}': {e}"))
            })?;
            // Accept PEM- or DER-encoded CRLs.
            let der: Vec<u8> = if let Ok((_, pem)) = pem::parse_x509_pem(&data) {
                pem.contents.to_vec()
            } else {
                data
            };
            let (_, crl) = CertificateRevocationList::from_der(&der).map_err(|e| {
                FusekiError::configuration(format!("failed to parse CRL '{source}': {e}"))
            })?;
            for revoked in crl.iter_revoked_certificates() {
                if revoked.raw_serial() == cert_serial.as_slice() {
                    warn!("Certificate serial is present in CRL '{source}' (revoked)");
                    return Ok(false);
                }
            }
        }

        Ok(true)
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
                            "Failed to parse PEM certificate contents from {trust_store_path:?}: {e}"
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
                            "Failed to parse DER certificate from {trust_store_path:?}: {e}"
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
        let combined = format!("{serial}:{subject}");

        let fingerprint = Sha256::digest(combined.as_bytes())
            .iter()
            .map(|b| format!("{b:02X}"))
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
            let regex = regex::Regex::new(&regex_pattern)
                .map_err(|e| FusekiError::configuration(format!("Invalid issuer pattern: {e}")))?;
            Ok(regex.is_match(issuer_dn))
        } else {
            Ok(issuer_dn == pattern)
        }
    }

    /// Verify that `cert` was cryptographically signed by `ca_cert`.
    ///
    /// This performs a real public-key signature verification of the leaf's
    /// `TBSCertificate` against the CA's public key (see
    /// [`super::cert_verify`]). It is NOT a DN-string comparison: an attacker
    /// cannot pass this check merely by copying a trusted CA's Subject DN into
    /// the Issuer field of a self-signed certificate, because the signature
    /// would not verify against the CA's key.
    fn verify_certificate_signature(
        &self,
        cert: &X509Certificate,
        ca_cert: &X509Certificate,
    ) -> FusekiResult<bool> {
        super::cert_verify::verify_certificate_signed_by(cert, ca_cert)
    }

    /// Extract certificate information
    fn extract_certificate_info(&self, cert: &X509Certificate) -> FusekiResult<CertificateAuth> {
        let subject_dn = cert.subject().to_string();
        let issuer_dn = cert.issuer().to_string();
        let serial_number = cert.serial.to_string();
        let fingerprint = self.compute_certificate_fingerprint(cert)?;

        let not_before = DateTime::from_timestamp(cert.validity().not_before.timestamp(), 0)
            .unwrap_or_else(Utc::now);
        let not_after = DateTime::from_timestamp(cert.validity().not_after.timestamp(), 0)
            .unwrap_or_else(Utc::now);

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
            if let Some(stripped) = component.strip_prefix("CN=") {
                return Ok(stripped.to_string());
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
            if let Some(stripped) = component.strip_prefix("CN=") {
                return Ok(stripped.to_string());
            }
        }

        Err(FusekiError::authentication(
            "No common name found in certificate DN",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn cert_config(check_crl: bool, check_ocsp: bool, sources: Vec<String>) -> SecurityConfig {
        use crate::config::config_security::{
            CertificateConfig, CertificateUserMapping, CertificateUsernameSource,
            CertificateValidationLevel,
        };
        SecurityConfig {
            certificate: Some(CertificateConfig {
                enabled: true,
                require_client_cert: true,
                trust_store: vec![],
                crl_sources: sources,
                check_crl,
                check_ocsp,
                allow_self_signed: false,
                user_mapping: CertificateUserMapping {
                    username_source: CertificateUsernameSource::CommonName,
                    dn_mapping_rules: vec![],
                    default_roles: vec![],
                    ou_role_mapping: std::collections::HashMap::new(),
                },
                max_chain_length: 5,
                validation_level: CertificateValidationLevel::Strict,
                trusted_issuers: None,
            }),
            ..SecurityConfig::default()
        }
    }

    #[test]
    fn regression_revocation_config_fails_loud() {
        // OCSP enabled → fail loud (unimplemented).
        let svc = CertificateAuthService::new(Arc::new(cert_config(false, true, vec![])));
        assert!(svc.revocation_config_check().is_err());

        // CRL enabled with no sources → fail loud.
        let svc = CertificateAuthService::new(Arc::new(cert_config(true, false, vec![])));
        assert!(svc.revocation_config_check().is_err());

        // CRL enabled with a URL source → fail loud (no network fetch).
        let svc = CertificateAuthService::new(Arc::new(cert_config(
            true,
            false,
            vec!["https://example.org/crl.pem".to_string()],
        )));
        assert!(svc.revocation_config_check().is_err());

        // CRL enabled with a local file path → config accepted.
        let svc = CertificateAuthService::new(Arc::new(cert_config(
            true,
            false,
            vec!["/etc/oxirs/revoked.crl".to_string()],
        )));
        assert!(svc.revocation_config_check().is_ok());

        // No revocation checking → accepted.
        let svc = CertificateAuthService::new(Arc::new(cert_config(false, false, vec![])));
        assert!(svc.revocation_config_check().is_ok());
    }

    #[test]
    fn test_match_issuer_pattern_exact() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let issuer_dn = "CN=Test CA,O=Test Corp,C=US";
        let pattern = "CN=Test CA,O=Test Corp,C=US";

        let result = auth_service
            .match_issuer_pattern(issuer_dn, pattern)
            .unwrap();
        assert!(result);
    }

    #[test]
    fn test_match_issuer_pattern_wildcard() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let issuer_dn = "CN=Test CA,O=Test Corp,C=US";
        let pattern = "CN=Test CA,O=*,C=US";

        let result = auth_service
            .match_issuer_pattern(issuer_dn, pattern)
            .unwrap();
        assert!(result);
    }

    #[test]
    fn test_match_issuer_pattern_wildcard_all() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let issuer_dn = "CN=Any CA,O=Any Corp,C=US";
        let pattern = "*";

        let result = auth_service
            .match_issuer_pattern(issuer_dn, pattern)
            .unwrap();
        assert!(result);
    }

    #[test]
    fn test_match_issuer_pattern_no_match() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let issuer_dn = "CN=Test CA,O=Test Corp,C=US";
        let pattern = "CN=Different CA,O=Test Corp,C=US";

        let result = auth_service
            .match_issuer_pattern(issuer_dn, pattern)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_extract_username_from_dn() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let dn = "CN=john.doe,O=Test Corp,C=US";
        let username = auth_service.extract_username_from_dn(dn).unwrap();

        assert_eq!(username, "john.doe");
    }

    #[test]
    fn test_extract_email_from_dn() {
        let config = Arc::new(SecurityConfig::default());
        let auth_service = CertificateAuthService::new(config);

        let dn = "CN=John Doe,emailAddress=john.doe@example.com,O=Test Corp,C=US";
        let email = auth_service.extract_email_from_dn(dn).unwrap();

        assert_eq!(email, "john.doe@example.com");
    }
}
