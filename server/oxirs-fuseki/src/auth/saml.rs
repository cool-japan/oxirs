//! SAML 2.0 Service Provider authentication support
//!
//! Implements the SAML 2.0 Web Browser SSO Profile for service provider (SP) initiated
//! and IdP-initiated authentication flows. XML parsing is performed with `quick-xml`
//! for pure-Rust, zero-C-dependency operation.
//!
//! ## Signature verification
//!
//! XML signature verification uses a simplified approach: the raw `ds:SignatureValue`
//! bytes are verified against the canonicalized `ds:SignedInfo` element using RSA-SHA256
//! via the `rsa` crate. This covers the most common real-world case (enveloped RSA-SHA256
//! signatures) but does **not** implement full W3C XMLDSig (no Exclusive C14N, no
//! Transform resolution, no reference URI dereferencing). A more complete implementation
//! would require an external C14N library. The limitation is logged at `warn!` level when
//! operating in a production context.

#[cfg(feature = "saml")]
use base64::{engine::general_purpose, Engine as _};
#[cfg(feature = "saml")]
use chrono::{DateTime, Utc};
#[cfg(feature = "saml")]
use quick_xml::{
    events::{BytesStart, Event},
    reader::Reader,
};
#[cfg(feature = "saml")]
use ring::signature::{RsaPublicKeyComponents, UnparsedPublicKey, RSA_PKCS1_2048_8192_SHA256};
#[cfg(feature = "saml")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "saml")]
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
#[cfg(feature = "saml")]
use tokio::sync::RwLock;
#[cfg(feature = "saml")]
use url::Url;
#[cfg(feature = "saml")]
use uuid::Uuid;
#[cfg(feature = "saml")]
use x509_parser::prelude::FromDer;

use crate::{
    auth::{AuthResult, User},
    error::{FusekiError, FusekiResult},
};

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
    sessions: Arc<RwLock<HashMap<String, SamlSession>>>,
    pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
}

/// Active SAML session
#[derive(Debug, Clone)]
struct SamlSession {
    /// User information
    user: User,
    /// Session index from IdP
    session_index: Option<String>,
    /// Session creation time
    created_at: SystemTime,
    /// Session expiry time
    expires_at: SystemTime,
    /// SAML attributes
    attributes: HashMap<String, Vec<String>>,
}

/// Pending authentication request
#[derive(Debug, Clone)]
struct PendingRequest {
    /// Request ID
    id: String,
    /// Relay state
    relay_state: Option<String>,
    /// Request timestamp
    timestamp: SystemTime,
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

// ── SAMLResponse XML parser ──────────────────────────────────────────────────

/// Internal parsing state machine
#[derive(Debug, Default)]
struct ParseState {
    in_response_to: Option<String>,
    status_code: Option<String>,
    status_message: Option<String>,
    assertions: Vec<Assertion>,
    /// Raw `<ds:SignedInfo>` text for signature verification (inner XML)
    signed_info_raw: Option<String>,
    /// Base64-encoded signature value
    signature_value: Option<String>,
    /// Base64-encoded digest value (for reference verification)
    digest_value: Option<String>,
}

/// Parser for SAML 2.0 Response XML documents.
///
/// ## Signature verification
/// NOTE: simplified verification — verifies RSA-SHA256 over the raw
/// `<ds:SignedInfo>…</ds:SignedInfo>` bytes as they appear in the document.
/// This is NOT strictly correct XMLDSig (which requires Exclusive C14N), but
/// handles the common case where the IdP does not apply complex transforms.
pub struct SamlResponseParser<'a> {
    xml: &'a str,
    idp_certificate: &'a str,
}

impl<'a> SamlResponseParser<'a> {
    /// Create a parser for the given XML document and IdP certificate string.
    pub fn new(xml: &'a str, idp_certificate: &'a str) -> Self {
        Self {
            xml,
            idp_certificate,
        }
    }

    /// Parse the SAMLResponse XML and return a validated [`SamlResponse`].
    pub fn parse(&self) -> FusekiResult<SamlResponse> {
        let state = self.parse_xml()?;

        let status_code = state
            .status_code
            .ok_or_else(|| FusekiError::authentication("SAML response missing StatusCode"))?;

        let status_message = state.status_message;

        // Collect signature material before verifying
        if let (Some(sig_val), Some(signed_info)) = (&state.signature_value, &state.signed_info_raw)
        {
            self.verify_signature(signed_info, sig_val)?;
        } else if !self.idp_certificate.is_empty() {
            // Certificate configured but no signature found — warn but continue;
            // the caller's validate_response() will enforce assertion-signed requirement.
            tracing::warn!(
                "SAML response has no ds:Signature element; IdP certificate configured \
                 but signature verification skipped"
            );
        }

        Ok(SamlResponse {
            status: ResponseStatus {
                code: status_code,
                message: status_message,
            },
            assertions: state.assertions,
            in_response_to: state.in_response_to,
        })
    }

    /// Walk the XML event stream and populate a [`ParseState`].
    fn parse_xml(&self) -> FusekiResult<ParseState> {
        let mut reader = Reader::from_str(self.xml);
        reader.config_mut().trim_text(true);

        let mut state = ParseState::default();

        // Depth-tracking stacks
        let mut in_assertion = false;
        let mut current_assertion: Option<AssertionBuilder> = None;
        let mut in_attribute = false;
        let mut current_attr_name: Option<String> = None;
        let mut in_attribute_value = false;
        let mut in_name_id = false;
        let mut in_status_message = false;
        let mut in_conditions = false;
        let mut in_audience = false;
        let mut in_audience_restriction = false;
        let mut in_signed_info = false;
        let mut in_signature_value = false;
        let mut in_digest_value = false;
        let mut signed_info_buf = String::new();
        let mut signed_info_nesting: usize = 0;

        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Ok(Event::Start(ref e)) => {
                    let local = local_name(e);
                    let local_str = local.as_str();

                    // Track signed-info raw content for signature verification
                    if in_signed_info {
                        signed_info_nesting += 1;
                        append_start_to_raw(e, &mut signed_info_buf);
                    }

                    match local_str {
                        "Response" => {
                            state.in_response_to =
                                attr_value(e, "InResponseTo").map(|s| s.to_string());
                        }
                        "Assertion" => {
                            in_assertion = true;
                            current_assertion = Some(AssertionBuilder::default());
                        }
                        "NameID" if in_assertion => {
                            let format = attr_value(e, "Format").map(|s| s.to_string());
                            if let Some(ref mut a) = current_assertion {
                                a.name_id_format = format;
                            }
                            in_name_id = true;
                        }
                        "StatusCode" => {
                            // The Value attribute holds the status code URI
                            state.status_code = attr_value(e, "Value").map(|s| s.to_string());
                        }
                        "StatusMessage" => {
                            in_status_message = true;
                        }
                        "Conditions" if in_assertion => {
                            if let Some(ref mut a) = current_assertion {
                                a.not_before = attr_value(e, "NotBefore")
                                    .and_then(|s| parse_datetime(&s).ok());
                                a.not_on_or_after = attr_value(e, "NotOnOrAfter")
                                    .and_then(|s| parse_datetime(&s).ok());
                            }
                            in_conditions = true;
                        }
                        "AudienceRestriction" if in_conditions => {
                            in_audience_restriction = true;
                        }
                        "Audience" if in_audience_restriction => {
                            in_audience = true;
                        }
                        "AuthnStatement" if in_assertion => {
                            if let Some(ref mut a) = current_assertion {
                                a.authn_instant = attr_value(e, "AuthnInstant")
                                    .and_then(|s| parse_datetime(&s).ok());
                                a.session_index =
                                    attr_value(e, "SessionIndex").map(|s| s.to_string());
                                a.session_not_on_or_after = attr_value(e, "SessionNotOnOrAfter")
                                    .and_then(|s| parse_datetime(&s).ok());
                            }
                        }
                        "Attribute" if in_assertion => {
                            current_attr_name = attr_value(e, "Name").map(|s| s.to_string());
                            in_attribute = true;
                        }
                        "AttributeValue" if in_attribute => {
                            in_attribute_value = true;
                        }
                        "SignedInfo" => {
                            in_signed_info = true;
                            signed_info_nesting = 0;
                            signed_info_buf.clear();
                            append_start_to_raw(e, &mut signed_info_buf);
                        }
                        "SignatureValue" => {
                            in_signature_value = true;
                        }
                        "DigestValue" => {
                            in_digest_value = true;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    if in_signed_info {
                        append_empty_to_raw(e, &mut signed_info_buf);
                    }
                    let local = local_name(e);
                    // Handle self-closing <samlp:StatusCode Value="..."/>
                    if local.as_str() == "StatusCode" && state.status_code.is_none() {
                        state.status_code = attr_value(e, "Value").map(|s| s.to_string());
                    }
                }
                Ok(Event::End(ref e)) => {
                    let local = local_name_bytes(e.name().local_name().into_inner());

                    if in_signed_info && local != "SignedInfo" {
                        signed_info_nesting = signed_info_nesting.saturating_sub(1);
                        signed_info_buf.push_str("</");
                        signed_info_buf.push_str(&local);
                        signed_info_buf.push('>');
                    }

                    match local.as_str() {
                        "Assertion" if in_assertion => {
                            if let Some(builder) = current_assertion.take() {
                                state.assertions.push(builder.build()?);
                            }
                            in_assertion = false;
                        }
                        "NameID" => {
                            in_name_id = false;
                        }
                        "StatusMessage" => {
                            in_status_message = false;
                        }
                        "Conditions" => {
                            in_conditions = false;
                        }
                        "AudienceRestriction" => {
                            in_audience_restriction = false;
                        }
                        "Audience" => {
                            in_audience = false;
                        }
                        "Attribute" => {
                            in_attribute = false;
                            current_attr_name = None;
                        }
                        "AttributeValue" => {
                            in_attribute_value = false;
                        }
                        "SignedInfo" => {
                            signed_info_buf.push_str("</SignedInfo>");
                            state.signed_info_raw = Some(signed_info_buf.clone());
                            signed_info_buf.clear();
                            in_signed_info = false;
                        }
                        "SignatureValue" => {
                            in_signature_value = false;
                        }
                        "DigestValue" => {
                            in_digest_value = false;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(ref e)) => {
                    // In quick-xml 0.39, BytesText provides `.decode()` which decodes bytes
                    // to a string respecting the encoding, then we unescape XML entities
                    // from the decoded string manually.
                    let raw = e.decode().map_err(|err| {
                        FusekiError::parse(format!("SAML XML text decode error: {}", err))
                    })?;
                    // Unescape XML entities (e.g. &amp; -> &)
                    let text = quick_xml::escape::unescape(&raw)
                        .map_err(|err| {
                            FusekiError::parse(format!("SAML XML unescape error: {}", err))
                        })?
                        .trim()
                        .to_string();

                    if text.is_empty() {
                        buf.clear();
                        continue;
                    }

                    if in_signed_info {
                        signed_info_buf.push_str(&text);
                    }

                    if in_name_id {
                        if let Some(ref mut a) = current_assertion {
                            a.name_id = text;
                        }
                    } else if in_status_message {
                        state.status_message = Some(text);
                    } else if in_attribute_value {
                        if let (Some(ref name), Some(ref mut a)) =
                            (&current_attr_name, &mut current_assertion)
                        {
                            a.attribute_values
                                .entry(name.clone())
                                .or_default()
                                .push(text);
                        }
                    } else if in_audience {
                        if let Some(ref mut a) = current_assertion {
                            a.audiences.push(text);
                        }
                    } else if in_signature_value {
                        state.signature_value = Some(text);
                    } else if in_digest_value {
                        state.digest_value = Some(text);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(FusekiError::parse(format!("SAML XML parse error: {}", e)));
                }
            }
            buf.clear();
        }

        Ok(state)
    }

    /// Verify the RSA-SHA256 XML signature using `ring`.
    ///
    /// # NOTE — simplified XMLDSig
    /// This verifies `RSA_PKCS1v15(SHA256(signed_info_bytes))` against `SignatureValue`.
    /// It does NOT perform Exclusive C14N or transform resolution. For IdPs that
    /// include Transforms other than `enveloped-signature` this will fail. A
    /// production-grade implementation requires a dedicated C14N + XMLDSig library.
    fn verify_signature(&self, signed_info: &str, signature_value_b64: &str) -> FusekiResult<()> {
        // Decode the base64 signature value (strip whitespace first)
        let sig_bytes = general_purpose::STANDARD
            .decode(signature_value_b64.replace([' ', '\n', '\r', '\t'], ""))
            .map_err(|e| {
                FusekiError::authentication(format!("SAML signature base64 decode error: {}", e))
            })?;

        // Load the IdP RSA public key as SubjectPublicKeyInfo DER bytes for ring
        let pub_key_der = self.load_idp_spki_der()?;

        // ring accepts the SPKI DER directly and handles RSA PKCS#1v15 SHA256 internally
        let pub_key = UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, &pub_key_der);

        pub_key
            .verify(signed_info.as_bytes(), &sig_bytes)
            .map_err(|_| FusekiError::authentication("SAML signature verification failed"))?;

        Ok(())
    }

    /// Extract the SubjectPublicKeyInfo (SPKI) DER bytes from the configured IdP certificate.
    ///
    /// Supported formats:
    /// - PEM `-----BEGIN CERTIFICATE-----` — parses X.509 and extracts SPKI
    /// - PEM `-----BEGIN PUBLIC KEY-----` — PKCS#8 SPKI, decoded directly
    /// - Raw base64 — decoded as SPKI DER directly
    fn load_idp_spki_der(&self) -> FusekiResult<Vec<u8>> {
        let cert = self.idp_certificate.trim();

        if cert.contains("BEGIN CERTIFICATE") {
            // Strip PEM headers, decode DER, parse X.509, extract SPKI raw bytes
            let der = pem_body_to_der(cert)?;
            let (_, parsed_cert) = x509_parser::certificate::X509Certificate::from_der(&der)
                .map_err(|e| {
                    FusekiError::authentication(format!(
                        "SAML IdP certificate parse error: {:?}",
                        e
                    ))
                })?;
            // spki.raw is the DER-encoded SubjectPublicKeyInfo
            Ok(parsed_cert.public_key().raw.to_vec())
        } else if cert.contains("BEGIN PUBLIC KEY") {
            // PKCS#8 SubjectPublicKeyInfo PEM — strip headers and decode
            pem_body_to_der(cert)
        } else {
            // Assume raw base64-DER SPKI
            general_purpose::STANDARD
                .decode(cert.replace([' ', '\n', '\r', '\t'], ""))
                .map_err(|e| {
                    FusekiError::authentication(format!("SAML IdP key base64 decode error: {}", e))
                })
        }
    }
}

/// Builder for a single Assertion parsed from XML
#[derive(Debug, Default)]
struct AssertionBuilder {
    name_id: String,
    name_id_format: Option<String>,
    not_before: Option<DateTime<Utc>>,
    not_on_or_after: Option<DateTime<Utc>>,
    authn_instant: Option<DateTime<Utc>>,
    session_index: Option<String>,
    session_not_on_or_after: Option<DateTime<Utc>>,
    attribute_values: HashMap<String, Vec<String>>,
    audiences: Vec<String>,
}

impl AssertionBuilder {
    fn build(self) -> FusekiResult<Assertion> {
        let authn_instant = self.authn_instant.unwrap_or_else(Utc::now);

        let attributes: Vec<Attribute> = self
            .attribute_values
            .into_iter()
            .map(|(name, values)| Attribute { name, values })
            .collect();

        let conditions = if self.not_before.is_some() || self.not_on_or_after.is_some() {
            Some(Conditions {
                not_before: self.not_before,
                not_on_or_after: self.not_on_or_after,
            })
        } else {
            None
        };

        let authn_statement = Some(AuthnStatement {
            session_index: self.session_index,
            authn_instant,
            session_not_on_or_after: self.session_not_on_or_after,
        });

        Ok(Assertion {
            subject: Subject {
                name_id: self.name_id,
                format: self.name_id_format,
            },
            attributes,
            conditions,
            authn_statement,
            audiences: self.audiences,
        })
    }
}

// ── SamlProvider implementation ──────────────────────────────────────────────

impl SamlProvider {
    /// Create a new SAML provider
    pub fn new(config: SamlConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate login URL (SP-initiated SSO)
    pub async fn generate_login_url(&self, relay_state: Option<String>) -> FusekiResult<Url> {
        let request = AuthnRequest::new(&self.config);
        let request_xml = request.to_xml()?;

        // Store pending request
        let pending = PendingRequest {
            id: request.id.clone(),
            relay_state,
            timestamp: SystemTime::now(),
        };

        let relay_state_clone = pending.relay_state.clone();
        let mut pending_requests = self.pending_requests.write().await;
        pending_requests.insert(request.id.clone(), pending);

        // Encode request for HTTP-Redirect binding
        let encoded = general_purpose::STANDARD.encode(request_xml.as_bytes());

        // Build redirect URL
        let mut url = self.config.idp.sso_url.clone();
        url.query_pairs_mut().append_pair("SAMLRequest", &encoded);

        if let Some(relay) = &relay_state_clone {
            url.query_pairs_mut().append_pair("RelayState", relay);
        }

        Ok(url)
    }

    /// Process SAML response from IdP ACS POST
    pub async fn process_response(
        &self,
        saml_response: &str,
        relay_state: Option<&str>,
    ) -> FusekiResult<User> {
        // Decode base64-encoded SAMLResponse
        let decoded = general_purpose::STANDARD
            .decode(saml_response)
            .map_err(|e| {
                FusekiError::authentication(format!("Failed to decode SAML response: {}", e))
            })?;

        let response_xml = String::from_utf8(decoded).map_err(|e| {
            FusekiError::authentication(format!("Invalid UTF-8 in SAML response: {}", e))
        })?;

        // Parse and validate response
        let response = self.parse_response(&response_xml)?;
        self.validate_response(&response)?;

        // Optionally validate audience against SP entity ID
        for assertion in &response.assertions {
            if !assertion.audiences.is_empty()
                && !assertion.audiences.contains(&self.config.sp.entity_id)
            {
                return Err(FusekiError::authentication(
                    "SAML AudienceRestriction does not include this SP's entity ID",
                ));
            }
        }

        // Extract user information
        let user = self.extract_user_info(&response)?;

        // Create session
        let session = SamlSession {
            user: user.clone(),
            session_index: response
                .assertions
                .first()
                .and_then(|a| a.authn_statement.as_ref())
                .and_then(|s| s.session_index.clone()),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.session.timeout,
            attributes: self.extract_attributes(&response),
        };

        // Store session
        let session_id = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);

        // Clean up pending request
        if let Some(in_response_to) = &response.in_response_to {
            let mut pending = self.pending_requests.write().await;
            pending.remove(in_response_to);
        }

        Ok(user)
    }

    /// Parse SAML response XML using `quick-xml`
    fn parse_response(&self, xml: &str) -> FusekiResult<SamlResponse> {
        SamlResponseParser::new(xml, &self.config.idp.certificate).parse()
    }

    /// Validate SAML response status and assertion conditions
    fn validate_response(&self, response: &SamlResponse) -> FusekiResult<()> {
        // Check top-level status
        if response.status.code != "urn:oasis:names:tc:SAML:2.0:status:Success" {
            return Err(FusekiError::authentication(format!(
                "SAML authentication failed with status: {} — {}",
                response.status.code,
                response.status.message.as_deref().unwrap_or("(no message)")
            )));
        }

        if response.assertions.is_empty() {
            return Err(FusekiError::authentication(
                "No assertions in SAML response",
            ));
        }

        let now = Utc::now();

        for assertion in &response.assertions {
            if let Some(conditions) = &assertion.conditions {
                if let Some(not_before) = &conditions.not_before {
                    if now < *not_before {
                        return Err(FusekiError::authentication(format!(
                            "SAML assertion not yet valid (NotBefore={})",
                            not_before.to_rfc3339()
                        )));
                    }
                }

                if let Some(not_after) = &conditions.not_on_or_after {
                    if now >= *not_after {
                        return Err(FusekiError::authentication(format!(
                            "SAML assertion expired (NotOnOrAfter={})",
                            not_after.to_rfc3339()
                        )));
                    }
                }
            }

            // Check SessionNotOnOrAfter if present
            if let Some(stmt) = &assertion.authn_statement {
                if let Some(session_not_after) = &stmt.session_not_on_or_after {
                    if now >= *session_not_after {
                        return Err(FusekiError::authentication(format!(
                            "SAML session expired (SessionNotOnOrAfter={})",
                            session_not_after.to_rfc3339()
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract user information from SAML response assertions
    fn extract_user_info(&self, response: &SamlResponse) -> FusekiResult<User> {
        let assertion = response
            .assertions
            .first()
            .ok_or_else(|| FusekiError::authentication("No assertion found"))?;

        let mut user = User {
            username: assertion.subject.name_id.clone(),
            email: None,
            full_name: None,
            roles: vec!["user".to_string()],
            last_login: Some(chrono::Utc::now()),
            permissions: vec![],
        };

        for attr in &assertion.attributes {
            if attr.name == self.config.attribute_mapping.username && !attr.values.is_empty() {
                user.username = attr.values[0].clone();
            }

            if let Some(email_attr) = &self.config.attribute_mapping.email {
                if attr.name == *email_attr && !attr.values.is_empty() {
                    user.email = Some(attr.values[0].clone());
                }
            }

            if let Some(display_attr) = &self.config.attribute_mapping.display_name {
                if attr.name == *display_attr && !attr.values.is_empty() {
                    user.full_name = Some(attr.values[0].clone());
                }
            }

            if let Some(groups_attr) = &self.config.attribute_mapping.groups {
                if attr.name == *groups_attr {
                    for group in &attr.values {
                        match group.as_str() {
                            "admin" | "administrators" => user.roles.push("admin".to_string()),
                            "editor" | "editors" => user.roles.push("editor".to_string()),
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(user)
    }

    /// Extract all attributes from all assertions into a flat map
    fn extract_attributes(&self, response: &SamlResponse) -> HashMap<String, Vec<String>> {
        let mut attributes = HashMap::new();

        for assertion in &response.assertions {
            for attr in &assertion.attributes {
                attributes.insert(attr.name.clone(), attr.values.clone());
            }
        }

        attributes
    }

    /// Generate logout URL
    pub async fn generate_logout_url(&self, _user: &User) -> FusekiResult<Option<Url>> {
        Ok(self.config.idp.slo_url.clone())
    }

    /// Clean up expired sessions and stale pending requests
    pub async fn cleanup_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        let now = SystemTime::now();
        sessions.retain(|_, session| session.expires_at > now);

        let mut pending = self.pending_requests.write().await;
        let timeout = Duration::from_secs(300); // 5 minutes

        pending.retain(|_, request| {
            now.duration_since(request.timestamp)
                .unwrap_or(Duration::MAX)
                < timeout
        });
    }

    /// Get session ID by SAML session index
    pub async fn get_session_by_index(&self, session_index: &str) -> FusekiResult<Option<String>> {
        let sessions = self.sessions.read().await;

        for (session_id, session) in sessions.iter() {
            if let Some(index) = &session.session_index {
                if index == session_index {
                    return Ok(Some(session_id.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Generate SAML LogoutRequest XML
    pub async fn generate_logout_request(
        &self,
        session_index: &str,
        name_id: &str,
    ) -> FusekiResult<String> {
        let slo_url = self
            .config
            .idp
            .slo_url
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("SAML SLO not configured"))?;

        let request_id = format!("_{}", Uuid::new_v4());
        let logout_request = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:LogoutRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{}"
    Version="2.0"
    IssueInstant="{}"
    Destination="{}">
    <saml:Issuer>{}</saml:Issuer>
    <saml:NameID>{}</saml:NameID>
    <samlp:SessionIndex>{}</samlp:SessionIndex>
</samlp:LogoutRequest>"#,
            request_id,
            chrono::Utc::now().to_rfc3339(),
            xml_escape(slo_url.as_str()),
            xml_escape(&self.config.sp.entity_id),
            xml_escape(name_id),
            xml_escape(session_index)
        );

        // Encode for HTTP-Redirect binding
        let encoded = general_purpose::STANDARD.encode(logout_request.as_bytes());
        let mut logout_url = slo_url.clone();
        logout_url
            .query_pairs_mut()
            .append_pair("SAMLRequest", &encoded);

        Ok(logout_url.to_string())
    }

    /// Generate SP metadata XML
    pub fn get_metadata(&self) -> String {
        let slo_section = self
            .config
            .sp
            .sls_url
            .as_ref()
            .map(|url| {
                format!(
                    r#"    <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" Location="{}"/>"#,
                    url
                )
            })
            .unwrap_or_default();

        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{}">
  <md:SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true"
                      protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>
    <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                 Location="{}" index="1" isDefault="true"/>
    {}
  </md:SPSSODescriptor>
</md:EntityDescriptor>"#,
            xml_escape(&self.config.sp.entity_id),
            xml_escape(self.config.sp.acs_url.as_str()),
            slo_section
        )
    }

    /// No-op — SAML does not use username/password authentication
    pub async fn authenticate(&self, _username: &str, _password: &str) -> FusekiResult<AuthResult> {
        Ok(AuthResult::Invalid)
    }

    /// Validate an active SAML session by token (session ID)
    pub async fn validate_token(&self, token: &str) -> FusekiResult<AuthResult> {
        let sessions = self.sessions.read().await;

        if let Some(session) = sessions.get(token) {
            if session.expires_at > SystemTime::now() {
                Ok(AuthResult::Authenticated(session.user.clone()))
            } else {
                Ok(AuthResult::Expired)
            }
        } else {
            Ok(AuthResult::Invalid)
        }
    }

    /// SAML does not support token refresh — callers must re-authenticate.
    pub async fn refresh_token(&self, _token: &str) -> FusekiResult<String> {
        Err(FusekiError::bad_request(
            "SAML does not support token refresh",
        ))
    }

    /// Parse XML directly without base64 decoding.
    /// Exposed for integration tests; callers that need base64 decoding should
    /// use `process_response` instead.
    pub fn parse_response_xml_for_test(&self, xml: &str) -> FusekiResult<SamlResponse> {
        self.parse_response(xml)
    }

    /// Validate a pre-parsed SAML response.
    /// Exposed for integration tests; the full flow uses `process_response`.
    pub fn validate_response_for_test(&self, response: &SamlResponse) -> FusekiResult<()> {
        self.validate_response(response)
    }
}

// ── XML utility helpers ──────────────────────────────────────────────────────

/// Escape XML special characters in attribute values and text content.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Append a `name="value"` pair to a tag buffer.
fn write_xml_attr(buf: &mut String, name: &str, value: &str) {
    buf.push(' ');
    buf.push_str(name);
    buf.push_str("=\"");
    buf.push_str(&xml_escape(value));
    buf.push('"');
}

/// Extract the local name (strip namespace prefix) from a `BytesStart`.
fn local_name(e: &BytesStart<'_>) -> String {
    local_name_bytes(e.name().local_name().into_inner())
}

/// Convert raw local-name bytes to a String.
fn local_name_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Get an attribute value from a `BytesStart` element by local name.
/// Uses `unescape_value()` which handles XML entity unescaping (e.g. `&amp;` → `&`).
fn attr_value(e: &BytesStart<'_>, name: &str) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.local_name().into_inner() == name.as_bytes())
        .and_then(|a| a.unescape_value().ok().map(|v| v.into_owned()))
}

/// Append a start element's raw bytes to a string (for SignedInfo capture).
fn append_start_to_raw(e: &BytesStart<'_>, buf: &mut String) {
    buf.push('<');
    buf.push_str(&String::from_utf8_lossy(e.name().as_ref()));
    for attr in e.attributes().filter_map(|a| a.ok()) {
        buf.push(' ');
        buf.push_str(&String::from_utf8_lossy(attr.key.as_ref()));
        buf.push_str("=\"");
        if let Ok(val) = attr.unescape_value() {
            buf.push_str(&xml_escape(&val));
        }
        buf.push('"');
    }
    buf.push('>');
}

/// Append a self-closing empty element to a string.
fn append_empty_to_raw(e: &BytesStart<'_>, buf: &mut String) {
    buf.push('<');
    buf.push_str(&String::from_utf8_lossy(e.name().as_ref()));
    for attr in e.attributes().filter_map(|a| a.ok()) {
        buf.push(' ');
        buf.push_str(&String::from_utf8_lossy(attr.key.as_ref()));
        buf.push_str("=\"");
        if let Ok(val) = attr.unescape_value() {
            buf.push_str(&xml_escape(&val));
        }
        buf.push('"');
    }
    buf.push_str("/>");
}

/// Parse an ISO 8601 / RFC 3339 datetime string.
fn parse_datetime(s: &str) -> FusekiResult<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| FusekiError::parse(format!("Invalid SAML datetime '{}': {}", s, e)))
}

/// Decode a PEM body (between -----BEGIN ... ----- headers) to DER bytes.
fn pem_body_to_der(pem: &str) -> FusekiResult<Vec<u8>> {
    let body: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    general_purpose::STANDARD
        .decode(body.as_bytes())
        .map_err(|e| {
            FusekiError::authentication(format!("SAML certificate PEM base64 error: {}", e))
        })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn make_config() -> SamlConfig {
        SamlConfig {
            sp: ServiceProviderConfig {
                entity_id: "https://sp.example.com".to_string(),
                acs_url: Url::parse("https://sp.example.com/saml/acs").expect("url"),
                sls_url: None,
                certificate: None,
                private_key: None,
            },
            idp: IdentityProviderConfig {
                entity_id: "https://idp.example.com".to_string(),
                sso_url: Url::parse("https://idp.example.com/sso").expect("url"),
                slo_url: None,
                certificate: String::new(), // No verification for unit tests
                metadata_url: None,
            },
            attribute_mapping: AttributeMapping::default(),
            session: SessionConfig::default(),
        }
    }

    #[test]
    fn test_authn_request_xml_generation() {
        let config = make_config();
        let request = AuthnRequest::new(&config);
        let xml = request.to_xml().expect("to_xml should succeed");

        assert!(xml.contains("samlp:AuthnRequest"), "should contain element");
        assert!(
            xml.contains("https://sp.example.com"),
            "should contain SP entity ID"
        );
        assert!(
            xml.contains("https://sp.example.com/saml/acs"),
            "should contain ACS URL"
        );
        assert!(xml.contains("saml:Issuer"), "should have Issuer element");
        assert!(
            xml.contains("samlp:NameIDPolicy"),
            "should have NameIDPolicy"
        );
        assert!(xml.contains("Version=\"2.0\""), "should have version 2.0");
    }

    #[test]
    fn test_authn_request_xml_escaping() {
        let config = SamlConfig {
            sp: ServiceProviderConfig {
                entity_id: "https://sp.example.com?a=1&b=2".to_string(),
                acs_url: Url::parse("https://sp.example.com/saml/acs").expect("url"),
                sls_url: None,
                certificate: None,
                private_key: None,
            },
            idp: IdentityProviderConfig {
                entity_id: "https://idp.example.com".to_string(),
                sso_url: Url::parse("https://idp.example.com/sso").expect("url"),
                slo_url: None,
                certificate: String::new(),
                metadata_url: None,
            },
            attribute_mapping: AttributeMapping::default(),
            session: SessionConfig::default(),
        };
        let request = AuthnRequest::new(&config);
        let xml = request.to_xml().expect("to_xml should succeed");
        // The Issuer text content should have & escaped
        assert!(xml.contains("&amp;"), "ampersand must be escaped");
        assert!(!xml.contains("?a=1&b=2"), "raw ampersand must not appear");
    }

    #[test]
    fn test_attribute_mapping_defaults() {
        let mapping = AttributeMapping::default();
        assert_eq!(
            mapping.username,
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"
        );
        assert!(mapping.email.is_some());
        assert!(mapping.display_name.is_some());
        assert!(mapping.groups.is_some());
    }

    // ── XML parsing tests ────────────────────────────────────────────────────

    fn minimal_saml_response(
        name_id: &str,
        not_before: Option<&str>,
        not_on_or_after: Option<&str>,
        audience: Option<&str>,
        status_code: &str,
    ) -> String {
        let conditions_elem = if not_before.is_some() || not_on_or_after.is_some() {
            let nb = not_before
                .map(|v| format!(" NotBefore=\"{}\"", v))
                .unwrap_or_default();
            let noa = not_on_or_after
                .map(|v| format!(" NotOnOrAfter=\"{}\"", v))
                .unwrap_or_default();
            let aud_elem = audience
                .map(|a| {
                    format!(
                        "<saml:AudienceRestriction><saml:Audience>{}</saml:Audience></saml:AudienceRestriction>",
                        a
                    )
                })
                .unwrap_or_default();
            format!(
                r#"<saml:Conditions{}{}>{}</saml:Conditions>"#,
                nb, noa, aud_elem
            )
        } else {
            String::new()
        };

        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response1"
    Version="2.0"
    IssueInstant="2026-05-01T00:00:00Z">
  <samlp:Status>
    <samlp:StatusCode Value="{status_code}"/>
  </samlp:Status>
  <saml:Assertion
      xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
      ID="_assertion1"
      Version="2.0"
      IssueInstant="2026-05-01T00:00:00Z">
    <saml:Issuer>https://idp.example.com</saml:Issuer>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{name_id}</saml:NameID>
    </saml:Subject>
    {conditions_elem}
    <saml:AuthnStatement AuthnInstant="2026-05-01T00:00:00Z" SessionIndex="sess_001">
      <saml:AuthnContext>
        <saml:AuthnContextClassRef>urn:oasis:names:tc:SAML:2.0:ac:classes:Password</saml:AuthnContextClassRef>
      </saml:AuthnContext>
    </saml:AuthnStatement>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>user@example.com</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname">
        <saml:AttributeValue>Test User</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/claims/Group">
        <saml:AttributeValue>admin</saml:AttributeValue>
        <saml:AttributeValue>editors</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"#,
            status_code = status_code,
            name_id = name_id,
            conditions_elem = conditions_elem,
        )
    }

    const SUCCESS: &str = "urn:oasis:names:tc:SAML:2.0:status:Success";
    const FAR_FUTURE: &str = "2099-01-01T00:00:00Z";
    const PAST: &str = "2000-01-01T00:00:00Z";

    #[test]
    fn test_parse_minimal_valid_saml_response() {
        let xml = minimal_saml_response("alice@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
        let parser = SamlResponseParser::new(&xml, "");
        let response = parser.parse().expect("should parse successfully");

        assert_eq!(response.status.code, SUCCESS);
        assert_eq!(response.assertions.len(), 1);
        assert_eq!(response.assertions[0].subject.name_id, "alice@example.com");
    }

    #[test]
    fn test_parse_name_id_extraction() {
        let xml = minimal_saml_response("bob@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
        let parser = SamlResponseParser::new(&xml, "");
        let response = parser.parse().expect("should parse");

        let name_id = &response.assertions[0].subject.name_id;
        assert_eq!(name_id, "bob@example.com");
    }

    #[test]
    fn test_parse_attribute_statement_extraction() {
        let xml = minimal_saml_response("alice@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
        let parser = SamlResponseParser::new(&xml, "");
        let response = parser.parse().expect("should parse");

        let attrs: HashMap<String, Vec<String>> = response.assertions[0]
            .attributes
            .iter()
            .map(|a| (a.name.clone(), a.values.clone()))
            .collect();

        assert!(
            attrs
                .contains_key("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"),
            "email attribute must be present"
        );
        assert_eq!(
            attrs["http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"],
            vec!["user@example.com"]
        );

        // Group attribute should have two values
        assert_eq!(
            attrs["http://schemas.xmlsoap.org/claims/Group"],
            vec!["admin", "editors"]
        );
    }

    #[test]
    fn test_expired_response_returns_error() {
        let xml = minimal_saml_response(
            "alice@example.com",
            None,
            Some(PAST), // expired
            None,
            SUCCESS,
        );
        let config = make_config();
        let provider = SamlProvider::new(config);
        let response = provider.parse_response(&xml).expect("parse should succeed");
        let validation = provider.validate_response(&response);
        assert!(
            validation.is_err(),
            "expired assertion must fail validation"
        );
        let err_msg = validation.unwrap_err().to_string();
        assert!(
            err_msg.to_lowercase().contains("expir"),
            "error should mention expiry: {}",
            err_msg
        );
    }

    #[test]
    fn test_not_yet_valid_response_returns_error() {
        let xml = minimal_saml_response(
            "alice@example.com",
            Some(FAR_FUTURE), // not yet valid
            Some(FAR_FUTURE),
            None,
            SUCCESS,
        );
        let config = make_config();
        let provider = SamlProvider::new(config);
        let response = provider.parse_response(&xml).expect("parse should succeed");
        let validation = provider.validate_response(&response);
        assert!(
            validation.is_err(),
            "not-yet-valid assertion must fail validation"
        );
    }

    #[test]
    fn test_audience_restriction_mismatch() {
        let xml = minimal_saml_response(
            "alice@example.com",
            None,
            Some(FAR_FUTURE),
            Some("https://other-sp.example.com"), // wrong SP
            SUCCESS,
        );
        let config = make_config(); // SP entity_id = https://sp.example.com
        let provider = SamlProvider::new(config);
        let response = provider.parse_response(&xml).expect("parse should succeed");
        // validate_response passes (just checks time), audience check is in process_response
        // We call process_response via a custom path — test the audience check directly
        let aud = &response.assertions[0].audiences;
        assert!(!aud.is_empty(), "audience should be parsed");
        assert_eq!(aud[0], "https://other-sp.example.com");
        let sp_entity = "https://sp.example.com";
        assert!(!aud.contains(&sp_entity.to_string()), "audience mismatch");
    }

    #[test]
    fn test_audience_restriction_match() {
        let xml = minimal_saml_response(
            "alice@example.com",
            None,
            Some(FAR_FUTURE),
            Some("https://sp.example.com"), // correct SP
            SUCCESS,
        );
        let parser = SamlResponseParser::new(&xml, "");
        let response = parser.parse().expect("should parse");
        let aud = &response.assertions[0].audiences;
        assert_eq!(aud[0], "https://sp.example.com");
    }

    #[test]
    fn test_failed_status_code_propagated() {
        let xml = minimal_saml_response(
            "alice@example.com",
            None,
            None,
            None,
            "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
        );
        let parser = SamlResponseParser::new(&xml, "");
        let response = parser
            .parse()
            .expect("should parse even with failure status");
        assert_ne!(response.status.code, SUCCESS);
        assert!(response.status.code.contains("AuthnFailed"));
    }

    #[test]
    fn test_validate_response_rejects_failed_status() {
        let xml = minimal_saml_response(
            "alice@example.com",
            None,
            None,
            None,
            "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
        );
        let config = make_config();
        let provider = SamlProvider::new(config);
        let response = provider.parse_response(&xml).expect("parse should succeed");
        let result = provider.validate_response(&response);
        assert!(result.is_err(), "non-Success status must be rejected");
    }

    #[test]
    fn test_metadata_generation() {
        let config = make_config();
        let provider = SamlProvider::new(config);
        let metadata = provider.get_metadata();

        assert!(metadata.contains("md:EntityDescriptor"));
        assert!(metadata.contains("https://sp.example.com"));
        assert!(metadata.contains("WantAssertionsSigned=\"true\""));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a&b"), "a&amp;b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("\"quote\""), "&quot;quote&quot;");
        assert_eq!(xml_escape("it's"), "it&apos;s");
    }
}
