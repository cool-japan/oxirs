//! SAML 2.0 Service Provider helper for Enterprise SSO.
//!
//! Provides AuthnRequest URL generation (HTTP-Redirect binding) and
//! basic SAMLResponse parsing (structure validation — XML signature
//! verification is deferred to a future layer that has access to the
//! IdP public key).
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_chat::sso::saml_sp::SamlSpHelper;
//! use oxirs_chat::sso::oidc::{SsoConfig, SsoProviderType};
//!
//! let config = SsoConfig {
//!     provider_type: SsoProviderType::Saml,
//!     issuer_url: "https://idp.example.com".to_string(),
//!     client_id: "sp-entity-id".to_string(),
//!     redirect_uri: "https://sp.example.com/auth/saml/acs".to_string(),
//!     scopes: vec![],
//! };
//! let helper = SamlSpHelper::new(config);
//! let url = helper.authn_request_url("relay-state-xyz").expect("build URL");
//! assert!(url.contains("SAMLRequest"));
//! ```

use base64::Engine as _;
use chrono::Utc;
use quick_xml::escape::unescape;
use quick_xml::events::Event;
use quick_xml::Reader;
use uuid::Uuid;

use super::oidc::{SsoConfig, SsoError, SsoUserInfo};

// ── SamlSpHelper ───────────────────────────────────────────────────────────

/// SAML 2.0 Service Provider helper.
///
/// Generates SAML `AuthnRequest` URLs for the HTTP-Redirect binding and
/// parses base64-encoded `SAMLResponse` messages.
pub struct SamlSpHelper {
    config: SsoConfig,
}

impl SamlSpHelper {
    /// Create a new helper with the given provider configuration.
    pub fn new(config: SsoConfig) -> Self {
        Self { config }
    }

    /// Generate a SAML `AuthnRequest` URL using the HTTP-Redirect binding.
    ///
    /// The returned URL is suitable for redirecting the user's browser to the
    /// IdP's Single Sign-On service.  The `SAMLRequest` parameter contains a
    /// deflate-compressed, base64-encoded `AuthnRequest` XML document.
    ///
    /// Note: Actual DEFLATE compression is omitted here for the pure-Rust
    /// zero-extra-dep implementation; the `SAMLRequest` value is plain base64
    /// of the XML (acceptable for many IdPs in test/dev mode, and structurally
    /// valid for the purpose of this integration layer).
    pub fn authn_request_url(&self, relay_state: &str) -> Result<String, SsoError> {
        let request_id = format!("_{}", uuid_simple());
        let issue_instant = utc_now_iso8601();

        let xml = format!(
            r#"<samlp:AuthnRequest
 xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
 xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
 ID="{id}"
 Version="2.0"
 IssueInstant="{instant}"
 Destination="{idp_sso}"
 AssertionConsumerServiceURL="{acs}">
  <saml:Issuer>{sp_entity_id}</saml:Issuer>
  <samlp:NameIDPolicy
   Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
   AllowCreate="true"/>
</samlp:AuthnRequest>"#,
            id = request_id,
            instant = issue_instant,
            idp_sso = self.config.issuer_url,
            acs = self.config.redirect_uri,
            sp_entity_id = self.config.client_id
        );

        // Encode as standard Base64 (HTTP-POST binding compatible;
        // HTTP-Redirect binding would require DEFLATE first).
        let encoded = base64::engine::general_purpose::STANDARD.encode(xml.as_bytes());
        let relay_encoded = percent_encode(relay_state);

        Ok(format!(
            "{}?SAMLRequest={}&RelayState={}",
            self.config.issuer_url,
            percent_encode(&encoded),
            relay_encoded
        ))
    }

    /// Parse a base64-encoded `SAMLResponse` and extract user identity.
    ///
    /// Performs structural validation only:
    /// 1. Base64 decoding.
    /// 2. UTF-8 decoding.
    /// 3. XML parsing for `<saml:NameID>` and `<saml:Attribute>` elements.
    ///
    /// XML signature verification is **not** performed here.
    pub fn parse_response(&self, saml_response_b64: &str) -> Result<SsoUserInfo, SsoError> {
        // Decode base64
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(saml_response_b64.trim())
            .map_err(|e| SsoError::Base64Error(e.to_string()))?;

        let xml_str = std::str::from_utf8(&decoded_bytes).map_err(|e| {
            SsoError::MalformedToken(format!("SAMLResponse is not valid UTF-8: {}", e))
        })?;

        parse_saml_xml(xml_str)
    }
}

// ── XML parsing ────────────────────────────────────────────────────────────

/// Parse a SAML response XML document and extract user identity information.
fn parse_saml_xml(xml: &str) -> Result<SsoUserInfo, SsoError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut name_id: Option<String> = None;
    let mut email: Option<String> = None;
    let mut display_name: Option<String> = None;
    let mut groups: Vec<String> = Vec::new();
    let mut raw_claims: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();

    // State machine for attribute value collection
    let mut current_attr_name: Option<String> = None;
    let mut in_name_id = false;
    let mut in_attr_value = false;
    let mut attr_values: Vec<String> = Vec::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e) | Event::Empty(ref e)) => {
                let local = local_name(e.name().as_ref());
                match local.as_str() {
                    "NameID" => {
                        in_name_id = true;
                    }
                    "Attribute" => {
                        // Flush previous attribute
                        flush_attr(
                            &mut current_attr_name,
                            &mut attr_values,
                            &mut email,
                            &mut display_name,
                            &mut groups,
                            &mut raw_claims,
                        );
                        // Read the Name attribute
                        let attr_name = e
                            .attributes()
                            .filter_map(|a| a.ok())
                            .find(|a| {
                                let k = local_name(a.key.as_ref());
                                k == "Name"
                            })
                            .and_then(|a| {
                                std::str::from_utf8(a.value.as_ref())
                                    .map(|s| s.to_string())
                                    .ok()
                            });
                        current_attr_name = attr_name;
                        attr_values.clear();
                    }
                    "AttributeValue" => {
                        in_attr_value = true;
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                match local.as_str() {
                    "NameID" => {
                        in_name_id = false;
                    }
                    "AttributeValue" => {
                        in_attr_value = false;
                    }
                    "Attribute" => {
                        // Flush
                        flush_attr(
                            &mut current_attr_name,
                            &mut attr_values,
                            &mut email,
                            &mut display_name,
                            &mut groups,
                            &mut raw_claims,
                        );
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                let raw = std::str::from_utf8(e)
                    .map_err(|err| SsoError::MalformedToken(format!("XML UTF-8 error: {}", err)))?;
                let text = unescape(raw)
                    .map_err(|err| {
                        SsoError::MalformedToken(format!("XML unescape error: {}", err))
                    })?
                    .trim()
                    .to_string();
                if in_name_id && !text.is_empty() {
                    name_id = Some(text);
                } else if in_attr_value && !text.is_empty() {
                    attr_values.push(text);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(SsoError::MalformedToken(format!("XML parse error: {}", e)));
            }
            _ => {}
        }
        buf.clear();
    }

    // Final flush
    flush_attr(
        &mut current_attr_name,
        &mut attr_values,
        &mut email,
        &mut display_name,
        &mut groups,
        &mut raw_claims,
    );

    let subject = name_id.ok_or_else(|| {
        SsoError::MalformedToken("SAMLResponse does not contain a NameID element".to_string())
    })?;

    Ok(SsoUserInfo {
        subject,
        email,
        name: display_name,
        groups,
        raw_claims,
    })
}

/// Flush the current attribute into the appropriate field.
fn flush_attr(
    current_attr_name: &mut Option<String>,
    attr_values: &mut Vec<String>,
    email: &mut Option<String>,
    display_name: &mut Option<String>,
    groups: &mut Vec<String>,
    raw_claims: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    if let Some(name) = current_attr_name.take() {
        if attr_values.is_empty() {
            return;
        }
        // Well-known SAML attribute URNs / friendly names
        let lower = name.to_lowercase();
        if lower.contains("emailaddress") || lower.contains("email") {
            if let Some(v) = attr_values.first() {
                *email = Some(v.clone());
            }
        } else if lower.contains("displayname") || lower.contains("name") {
            if let Some(v) = attr_values.first() {
                *display_name = Some(v.clone());
            }
        } else if lower.contains("group") || lower.contains("role") {
            groups.extend_from_slice(attr_values);
        }

        // Always store in raw_claims
        let json_values: Vec<serde_json::Value> = attr_values
            .iter()
            .map(|v| serde_json::Value::String(v.clone()))
            .collect();
        raw_claims.insert(name, serde_json::Value::Array(json_values));
        attr_values.clear();
    }
}

/// Extract the local name (strip namespace prefix) from a byte slice.
fn local_name(name: &[u8]) -> String {
    let s = std::str::from_utf8(name).unwrap_or("");
    match s.rfind(':') {
        Some(pos) => s[pos + 1..].to_string(),
        None => s.to_string(),
    }
}

// ── Minimal helpers ────────────────────────────────────────────────────────

/// Generate a unique hex string for use as a SAML request ID.
fn uuid_simple() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Return an ISO-8601 UTC timestamp string suitable for SAML IssueInstant.
fn utc_now_iso8601() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Minimal percent-encoding for query-string values.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            b => {
                use std::fmt::Write as _;
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sso::oidc::{SsoConfig, SsoProviderType};

    fn make_saml_config() -> SsoConfig {
        SsoConfig {
            provider_type: SsoProviderType::Saml,
            issuer_url: "https://idp.example.com/sso".to_string(),
            client_id: "https://sp.example.com".to_string(),
            redirect_uri: "https://sp.example.com/auth/saml/acs".to_string(),
            scopes: vec![],
        }
    }

    #[test]
    fn test_saml_sp_authn_request_url() {
        let helper = SamlSpHelper::new(make_saml_config());
        let url = helper
            .authn_request_url("my-relay-state")
            .expect("authn request URL");
        assert!(url.contains("SAMLRequest="), "missing SAMLRequest param");
        assert!(url.contains("RelayState="), "missing RelayState param");
        assert!(
            url.starts_with("https://idp.example.com/sso"),
            "wrong base URL"
        );
    }

    #[test]
    fn test_saml_parse_response_malformed() {
        let helper = SamlSpHelper::new(make_saml_config());
        // Not valid base64
        let err = helper
            .parse_response("!!!not-base64!!!")
            .expect_err("should fail");
        assert!(
            matches!(err, SsoError::Base64Error(_)),
            "expected Base64Error, got: {}",
            err
        );
    }

    #[test]
    fn test_saml_parse_response_minimal_valid() {
        let helper = SamlSpHelper::new(make_saml_config());

        // Minimal SAML response XML with a NameID
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">
  <saml:Assertion>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">
        alice@example.com
      </saml:NameID>
    </saml:Subject>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>alice@example.com</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/displayname">
        <saml:AttributeValue>Alice Doe</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.microsoft.com/ws/2008/06/identity/claims/groups">
        <saml:AttributeValue>engineers</saml:AttributeValue>
        <saml:AttributeValue>rdf-users</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"#;

        let b64 = base64::engine::general_purpose::STANDARD.encode(xml.as_bytes());
        let user_info = helper
            .parse_response(&b64)
            .expect("parse valid SAML response");

        assert!(!user_info.subject.is_empty(), "subject must be set");
        assert!(
            user_info.subject.contains("alice@example.com"),
            "subject should be the NameID value"
        );
    }
}
