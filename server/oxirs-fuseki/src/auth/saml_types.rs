//! SAML 2.0 configuration, provider, session, and protocol data structures.
//!
//! Split out of the original `saml` module (Round 32 refactor). Contains the
//! configuration structs, the [`SamlProvider`] / session state, and the SAML
//! protocol data structures (AuthnRequest, Response, Assertion, etc.).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use url::Url;
use uuid::Uuid;

use crate::{auth::User, error::FusekiResult};

use super::saml_helpers::{write_xml_attr, xml_escape};

// ── Configuration structs ────────────────────────────────────────────────────

/// SAML 2.0 configuration
#[derive(Debug, Clone)]
pub struct SamlConfig {
    /// Service Provider (SP) configuration
    pub sp: ServiceProviderConfig,
    /// Identity Provider (IdP) configuration
    pub idp: IdentityProviderConfig,
    /// Attribute mapping configuration
    pub attribute_mapping: AttributeMapping,
    /// Session configuration
    pub session: SessionConfig,
}

/// Type alias for handler compatibility
pub type SamlSpConfig = ServiceProviderConfig;

/// Type alias for handler compatibility
pub type SamlAttributeMappings = AttributeMapping;

/// Service Provider configuration
#[derive(Debug, Clone)]
pub struct ServiceProviderConfig {
    /// SP Entity ID
    pub entity_id: String,
    /// Assertion Consumer Service URL
    pub acs_url: Url,
    /// Single Logout Service URL
    pub sls_url: Option<Url>,
    /// SP Certificate for signing (PEM)
    pub certificate: Option<String>,
    /// SP Private key for signing (PEM)
    pub private_key: Option<String>,
}

/// Identity Provider configuration
#[derive(Debug, Clone)]
pub struct IdentityProviderConfig {
    /// IdP Entity ID
    pub entity_id: String,
    /// Single Sign-On Service URL
    pub sso_url: Url,
    /// Single Logout Service URL
    pub slo_url: Option<Url>,
    /// IdP Certificate for signature verification (PEM or base64-DER)
    pub certificate: String,
    /// Metadata URL (optional)
    pub metadata_url: Option<Url>,
}

/// Attribute mapping from SAML assertions to user properties
#[derive(Debug, Clone)]
pub struct AttributeMapping {
    /// Username attribute name
    pub username: String,
    /// Email attribute name
    pub email: Option<String>,
    /// Display name attribute name
    pub display_name: Option<String>,
    /// Groups/roles attribute name
    pub groups: Option<String>,
    /// Additional custom attributes
    pub custom: HashMap<String, String>,
}

impl Default for AttributeMapping {
    fn default() -> Self {
        Self {
            username: "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name".to_string(),
            email: Some(
                "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress".to_string(),
            ),
            display_name: Some(
                "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname".to_string(),
            ),
            groups: Some("http://schemas.xmlsoap.org/claims/Group".to_string()),
            custom: HashMap::new(),
        }
    }
}

/// SAML session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Session timeout duration
    pub timeout: Duration,
    /// Allow IdP-initiated SSO
    pub allow_idp_initiated: bool,
    /// Force authentication
    pub force_authn: bool,
    /// Session index tracking
    pub track_session_index: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600), // 1 hour
            allow_idp_initiated: false,
            force_authn: false,
            track_session_index: true,
        }
    }
}

// ── Provider and session structs ─────────────────────────────────────────────

/// SAML 2.0 authentication provider
pub struct SamlProvider {
    pub config: SamlConfig,
    pub(super) sessions: Arc<RwLock<HashMap<String, SamlSession>>>,
    pub(super) pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
}

/// Active SAML session
#[derive(Debug, Clone)]
pub(super) struct SamlSession {
    /// User information
    pub(super) user: User,
    /// Session index from IdP
    pub(super) session_index: Option<String>,
    /// Session creation time
    pub(super) created_at: SystemTime,
    /// Session expiry time
    pub(super) expires_at: SystemTime,
    /// SAML attributes
    pub(super) attributes: HashMap<String, Vec<String>>,
}

/// Pending authentication request
#[derive(Debug, Clone)]
pub(super) struct PendingRequest {
    /// Request ID
    pub(super) id: String,
    /// Relay state
    pub(super) relay_state: Option<String>,
    /// Request timestamp
    pub(super) timestamp: SystemTime,
}

// ── SAML protocol data structures ───────────────────────────────────────────

/// SAML AuthN request
#[derive(Debug, Serialize)]
pub struct AuthnRequest {
    /// Request ID
    pub id: String,
    /// Issue instant
    pub issue_instant: DateTime<Utc>,
    /// Destination URL
    pub destination: Url,
    /// Issuer (SP entity ID)
    pub issuer: String,
    /// Assertion Consumer Service URL
    pub acs_url: Url,
    /// Protocol binding
    pub protocol_binding: String,
    /// Force authentication
    pub force_authn: bool,
}

impl AuthnRequest {
    /// Create a new authentication request
    pub fn new(config: &SamlConfig) -> Self {
        Self {
            id: format!("_{}", Uuid::new_v4()),
            issue_instant: Utc::now(),
            destination: config.idp.sso_url.clone(),
            issuer: config.sp.entity_id.clone(),
            acs_url: config.sp.acs_url.clone(),
            protocol_binding: "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST".to_string(),
            force_authn: config.session.force_authn,
        }
    }

    /// Generate SAMLv2.0-compliant AuthnRequest XML using quick-xml writer for
    /// proper escaping of all attribute values.
    pub fn to_xml(&self) -> FusekiResult<String> {
        let mut buf = String::with_capacity(1024);

        // XML declaration
        buf.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        buf.push('\n');

        // Opening element with all required attributes
        buf.push_str("<samlp:AuthnRequest");
        write_xml_attr(
            &mut buf,
            "xmlns:samlp",
            "urn:oasis:names:tc:SAML:2.0:protocol",
        );
        write_xml_attr(
            &mut buf,
            "xmlns:saml",
            "urn:oasis:names:tc:SAML:2.0:assertion",
        );
        write_xml_attr(&mut buf, "ID", &self.id);
        write_xml_attr(&mut buf, "Version", "2.0");
        write_xml_attr(&mut buf, "IssueInstant", &self.issue_instant.to_rfc3339());
        write_xml_attr(&mut buf, "Destination", self.destination.as_str());
        write_xml_attr(&mut buf, "ProtocolBinding", &self.protocol_binding);
        write_xml_attr(
            &mut buf,
            "AssertionConsumerServiceURL",
            self.acs_url.as_str(),
        );
        write_xml_attr(
            &mut buf,
            "ForceAuthn",
            if self.force_authn { "true" } else { "false" },
        );
        buf.push('>');
        buf.push('\n');

        // Issuer child element
        buf.push_str("  <saml:Issuer>");
        buf.push_str(&xml_escape(&self.issuer));
        buf.push_str("</saml:Issuer>\n");

        // NameIDPolicy
        buf.push_str("  <samlp:NameIDPolicy");
        write_xml_attr(
            &mut buf,
            "Format",
            "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
        );
        write_xml_attr(&mut buf, "AllowCreate", "true");
        buf.push_str("/>\n");

        buf.push_str("</samlp:AuthnRequest>\n");

        Ok(buf)
    }
}

/// SAML Response — parsed from IdP-provided XML
#[derive(Debug, Deserialize)]
pub struct SamlResponse {
    /// Response status
    pub status: ResponseStatus,
    /// Assertions
    pub assertions: Vec<Assertion>,
    /// In response to request ID
    pub in_response_to: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseStatus {
    /// Status code
    pub code: String,
    /// Status message
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Assertion {
    /// Subject information
    pub subject: Subject,
    /// Attributes
    pub attributes: Vec<Attribute>,
    /// Conditions
    pub conditions: Option<Conditions>,
    /// Authentication statement
    pub authn_statement: Option<AuthnStatement>,
    /// AudienceRestriction values inside Conditions
    pub audiences: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Subject {
    /// Name ID
    pub name_id: String,
    /// Name ID format
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Attribute {
    /// Attribute name
    pub name: String,
    /// Attribute values
    pub values: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Conditions {
    /// Not before time
    pub not_before: Option<DateTime<Utc>>,
    /// Not on or after time
    pub not_on_or_after: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct AuthnStatement {
    /// Session index
    pub session_index: Option<String>,
    /// Authentication instant
    pub authn_instant: DateTime<Utc>,
    /// Session not on or after
    pub session_not_on_or_after: Option<DateTime<Utc>>,
}
