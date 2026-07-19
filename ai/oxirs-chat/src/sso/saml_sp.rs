//! SAML 2.0 Service Provider helper for Enterprise SSO.
//!
//! Provides AuthnRequest URL generation (HTTP-Redirect binding) and
//! `SAMLResponse` parsing **with XML signature verification** against the
//! configured IdP certificate.
//!
//! ## Security model
//!
//! [`SamlSpHelper::parse_response`] verifies the enveloped RSA-SHA256 XML
//! signature over the `<ds:SignedInfo>` element using the IdP's public key
//! (from the configured X.509 certificate) via the Pure-Rust `rsa` crate,
//! **before** any `NameID`/attribute is extracted and trusted.
//!
//! It **fails closed**:
//! * A response carrying a signature is rejected unless an IdP certificate is
//!   configured *and* the signature verifies.
//! * An **unsigned** response is rejected unless the helper was explicitly
//!   constructed to allow it ([`SamlSpHelper::new_allow_unsigned`]). The default
//!   ([`SamlSpHelper::new`] / [`SamlSpHelper::with_certificate`]) rejects
//!   unsigned responses.
//!
//! ### NOTE — simplified XMLDSig
//! Like the Fuseki SAML SP, this verifies `RSASSA-PKCS1-v1_5(SHA256(signed_info))`
//! over the `<ds:SignedInfo>` bytes as reconstructed from the parse stream. It
//! does **not** implement full W3C Exclusive C14N / transform resolution, so an
//! IdP applying complex transforms may fail verification. This is a deliberate,
//! documented limitation shared with the server-side SAML implementation.
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
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use rsa::pkcs1v15::{Signature as RsaSignature, VerifyingKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier;
use rsa::RsaPublicKey;
use uuid::Uuid;

use super::oidc::{SsoConfig, SsoError, SsoUserInfo};

/// Signature material collected while parsing a `SAMLResponse`.
#[derive(Debug, Default)]
struct SignatureMaterial {
    /// Reconstructed raw `<ds:SignedInfo>…</ds:SignedInfo>` text.
    signed_info_raw: Option<String>,
    /// Base64-encoded `<ds:SignatureValue>` content.
    signature_value: Option<String>,
}

impl SignatureMaterial {
    /// True if the response actually carried a signature.
    fn is_signed(&self) -> bool {
        self.signed_info_raw.is_some() && self.signature_value.is_some()
    }
}

// ── SamlSpHelper ───────────────────────────────────────────────────────────

/// SAML 2.0 Service Provider helper.
///
/// Generates SAML `AuthnRequest` URLs for the HTTP-Redirect binding and
/// parses base64-encoded `SAMLResponse` messages with signature verification.
pub struct SamlSpHelper {
    config: SsoConfig,
    /// IdP X.509 certificate / public key (PEM or raw base64 DER) used to
    /// verify response signatures. `None` disables verification of present
    /// signatures (and, unless `allow_unsigned`, makes signed responses fail).
    idp_certificate: Option<String>,
    /// When true, responses with no `<ds:Signature>` are accepted (dev/test).
    /// Defaults to false — unsigned responses are rejected.
    allow_unsigned: bool,
}

impl SamlSpHelper {
    /// Create a new helper with the given provider configuration.
    ///
    /// No IdP certificate is attached and unsigned responses are rejected, so
    /// [`Self::parse_response`] fails closed until [`Self::with_certificate`]
    /// supplies the IdP's signing certificate.
    pub fn new(config: SsoConfig) -> Self {
        Self {
            config,
            idp_certificate: None,
            allow_unsigned: false,
        }
    }

    /// Create a helper that verifies response signatures against `idp_certificate`
    /// (PEM `-----BEGIN CERTIFICATE-----`, PEM `-----BEGIN PUBLIC KEY-----`, or
    /// raw base64-DER SubjectPublicKeyInfo). Unsigned responses are rejected.
    pub fn with_certificate(config: SsoConfig, idp_certificate: impl Into<String>) -> Self {
        Self {
            config,
            idp_certificate: Some(idp_certificate.into()),
            allow_unsigned: false,
        }
    }

    /// Create a helper that accepts **unsigned** responses.
    ///
    /// # Warning
    /// This disables the SAML SP's primary security control and must only be
    /// used in development/test against a trusted local IdP. Never enable it in
    /// production — an attacker can forge arbitrary identities.
    pub fn new_allow_unsigned(config: SsoConfig) -> Self {
        Self {
            config,
            idp_certificate: None,
            allow_unsigned: true,
        }
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

    /// Parse a base64-encoded `SAMLResponse`, **verify its XML signature**, and
    /// extract user identity.
    ///
    /// Steps:
    /// 1. Base64 + UTF-8 decoding.
    /// 2. XML parsing for `<saml:NameID>`, `<saml:Attribute>`, and the
    ///    `<ds:SignedInfo>` / `<ds:SignatureValue>` signature material.
    /// 3. **Signature-policy enforcement** (see the module-level security model):
    ///    a signed response is RSA-SHA256-verified against the configured IdP
    ///    certificate; an unsigned response is rejected unless the helper was
    ///    built with [`Self::new_allow_unsigned`].
    /// 4. Identity extraction — only reached once the policy is satisfied.
    pub fn parse_response(&self, saml_response_b64: &str) -> Result<SsoUserInfo, SsoError> {
        // Decode base64
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(saml_response_b64.trim())
            .map_err(|e| SsoError::Base64Error(e.to_string()))?;

        let xml_str = std::str::from_utf8(&decoded_bytes).map_err(|e| {
            SsoError::MalformedToken(format!("SAMLResponse is not valid UTF-8: {}", e))
        })?;

        let (user_info, sig) = parse_saml_xml(xml_str)?;

        // Enforce signature policy BEFORE trusting the extracted identity.
        self.enforce_signature_policy(&sig)?;

        Ok(user_info)
    }

    /// Enforce the configured signature policy against the collected material.
    fn enforce_signature_policy(&self, sig: &SignatureMaterial) -> Result<(), SsoError> {
        if sig.is_signed() {
            // A signature is present: it MUST verify against a configured cert.
            let cert = self
                .idp_certificate
                .as_deref()
                .ok_or(SsoError::SignatureVerificationUnavailable)?;
            let signed_info = sig
                .signed_info_raw
                .as_deref()
                .ok_or(SsoError::SignatureInvalid)?;
            let signature_value = sig
                .signature_value
                .as_deref()
                .ok_or(SsoError::SignatureInvalid)?;
            verify_saml_signature(cert, signed_info, signature_value)
        } else if self.allow_unsigned {
            // Explicitly opted into accepting unsigned responses (dev/test).
            tracing::warn!(
                "SAML response has no ds:Signature; accepting because allow_unsigned is set \
                 (INSECURE — do not use in production)"
            );
            Ok(())
        } else {
            // Fail closed: an unsigned response is not trusted.
            Err(SsoError::UnsignedAssertionRejected)
        }
    }
}

/// Verify an enveloped RSA-SHA256 XML signature.
///
/// Verifies `RSASSA-PKCS1-v1_5(SHA256(signed_info))` against `signature_value_b64`
/// using the RSA public key extracted from `idp_certificate`. See the module
/// docs for the C14N caveat.
fn verify_saml_signature(
    idp_certificate: &str,
    signed_info: &str,
    signature_value_b64: &str,
) -> Result<(), SsoError> {
    let sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(signature_value_b64.replace([' ', '\n', '\r', '\t'], ""))
        .map_err(|e| SsoError::Base64Error(format!("SAML signature: {e}")))?;

    let spki_der = load_idp_spki_der(idp_certificate)?;
    let public_key = RsaPublicKey::from_public_key_der(&spki_der)
        .map_err(|e| SsoError::InvalidKey(format!("SAML IdP RSA public key parse error: {e}")))?;

    let verifying_key = VerifyingKey::<rsa::sha2::Sha256>::new(public_key);
    let signature =
        RsaSignature::try_from(sig_bytes.as_slice()).map_err(|_| SsoError::SignatureInvalid)?;

    verifying_key
        .verify(signed_info.as_bytes(), &signature)
        .map_err(|_| SsoError::SignatureInvalid)
}

/// Extract the SubjectPublicKeyInfo (SPKI) DER bytes from a configured IdP
/// certificate / key. Supported inputs:
/// - PEM `-----BEGIN CERTIFICATE-----` — X.509 cert, SPKI extracted via DER walk
/// - PEM `-----BEGIN PUBLIC KEY-----` — PKCS#8 SPKI, decoded directly
/// - Raw base64 — decoded as SPKI DER directly
fn load_idp_spki_der(cert: &str) -> Result<Vec<u8>, SsoError> {
    let cert = cert.trim();
    if cert.contains("BEGIN CERTIFICATE") {
        let der = pem_body_to_der(cert)?;
        extract_spki_from_x509_der(&der)
    } else if cert.contains("BEGIN PUBLIC KEY") {
        pem_body_to_der(cert)
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(cert.replace([' ', '\n', '\r', '\t'], ""))
            .map_err(|e| SsoError::Base64Error(format!("SAML IdP key: {e}")))
    }
}

/// Decode a PEM body (the base64 between `-----BEGIN…-----` / `-----END…-----`).
fn pem_body_to_der(pem: &str) -> Result<Vec<u8>, SsoError> {
    let body: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(body.as_bytes())
        .map_err(|e| SsoError::Base64Error(format!("SAML certificate PEM: {e}")))
}

/// Read one DER TLV starting at `pos`. Returns `(tag, content_start, content_len,
/// next_pos)` where the full TLV occupies `data[pos..next_pos]`.
///
/// Only the definite short/long length forms and single-byte (low-tag-number)
/// tags used by X.509 structures are handled.
fn read_der_tlv(data: &[u8], pos: usize) -> Result<(u8, usize, usize, usize), SsoError> {
    let malformed = || SsoError::InvalidKey("malformed DER in IdP certificate".to_string());
    let tag = *data.get(pos).ok_or_else(malformed)?;
    let len_byte = *data.get(pos + 1).ok_or_else(malformed)? as usize;
    let (content_start, content_len) = if len_byte & 0x80 == 0 {
        // Short form.
        (pos + 2, len_byte)
    } else {
        // Long form: low 7 bits give the number of length octets.
        let num_octets = len_byte & 0x7f;
        if num_octets == 0 || num_octets > 4 {
            return Err(malformed());
        }
        let mut len = 0usize;
        for i in 0..num_octets {
            let b = *data.get(pos + 2 + i).ok_or_else(malformed)? as usize;
            len = (len << 8) | b;
        }
        (pos + 2 + num_octets, len)
    };
    let next = content_start
        .checked_add(content_len)
        .ok_or_else(malformed)?;
    if next > data.len() {
        return Err(malformed());
    }
    Ok((tag, content_start, content_len, next))
}

/// Extract the DER-encoded SubjectPublicKeyInfo from an X.509 certificate DER.
///
/// `Certificate ::= SEQUENCE { tbsCertificate SEQUENCE {...}, ... }` and inside
/// `tbsCertificate` the fields are, in order: `[0] version` (optional),
/// `serialNumber`, `signature`, `issuer`, `validity`, `subject`,
/// `subjectPublicKeyInfo`. We skip to and return the SPKI TLV.
fn extract_spki_from_x509_der(der: &[u8]) -> Result<Vec<u8>, SsoError> {
    let bad = || SsoError::InvalidKey("cannot locate SPKI in IdP certificate".to_string());
    // Certificate ::= SEQUENCE
    let (tag, cert_cs, cert_cl, _) = read_der_tlv(der, 0)?;
    if tag != 0x30 {
        return Err(bad());
    }
    // tbsCertificate ::= SEQUENCE (first element of Certificate)
    let (t2, tbs_cs, tbs_cl, _) = read_der_tlv(der, cert_cs)?;
    if t2 != 0x30 || tbs_cs + tbs_cl > cert_cs + cert_cl {
        return Err(bad());
    }
    let mut pos = tbs_cs;
    let tbs_end = tbs_cs + tbs_cl;
    // Optional [0] EXPLICIT version.
    let (vt, _, _, vnext) = read_der_tlv(der, pos)?;
    if vt == 0xA0 {
        pos = vnext;
    }
    // Skip serialNumber, signature, issuer, validity, subject (5 elements).
    for _ in 0..5 {
        let (_, _, _, next) = read_der_tlv(der, pos)?;
        if next > tbs_end {
            return Err(bad());
        }
        pos = next;
    }
    // subjectPublicKeyInfo ::= SEQUENCE
    let (spki_tag, _, _, spki_next) = read_der_tlv(der, pos)?;
    if spki_tag != 0x30 {
        return Err(bad());
    }
    Ok(der[pos..spki_next].to_vec())
}

// ── XML parsing ────────────────────────────────────────────────────────────

/// Parse a SAML response XML document, extracting user identity information and
/// the `<ds:SignedInfo>` / `<ds:SignatureValue>` signature material.
///
/// The raw SignedInfo is reconstructed from the parse stream (mirroring the
/// server-side SAML SP) so it can be verified against the IdP certificate.
fn parse_saml_xml(xml: &str) -> Result<(SsoUserInfo, SignatureMaterial), SsoError> {
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

    // Signature-capture state.
    let mut sig = SignatureMaterial::default();
    let mut in_signed_info = false;
    let mut signed_info_buf = String::new();
    let mut in_signature_value = false;

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let local = local_name(e.name().as_ref());
                // SignedInfo reconstruction (must run before the semantic match).
                if local == "SignedInfo" {
                    in_signed_info = true;
                    signed_info_buf.clear();
                    append_start_to_raw(e, &mut signed_info_buf);
                } else if in_signed_info {
                    append_start_to_raw(e, &mut signed_info_buf);
                }
                match local.as_str() {
                    "NameID" => {
                        in_name_id = true;
                    }
                    "Attribute" => {
                        flush_attr(
                            &mut current_attr_name,
                            &mut attr_values,
                            &mut email,
                            &mut display_name,
                            &mut groups,
                            &mut raw_claims,
                        );
                        current_attr_name = read_attr_name(e);
                        attr_values.clear();
                    }
                    "AttributeValue" => {
                        in_attr_value = true;
                    }
                    "SignatureValue" => {
                        in_signature_value = true;
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let local = local_name(e.name().as_ref());
                if in_signed_info {
                    append_empty_to_raw(e, &mut signed_info_buf);
                }
                if local == "Attribute" {
                    flush_attr(
                        &mut current_attr_name,
                        &mut attr_values,
                        &mut email,
                        &mut display_name,
                        &mut groups,
                        &mut raw_claims,
                    );
                    current_attr_name = read_attr_name(e);
                    attr_values.clear();
                }
            }
            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                // SignedInfo reconstruction for closing tags.
                if in_signed_info && local != "SignedInfo" {
                    signed_info_buf.push_str("</");
                    signed_info_buf.push_str(&local);
                    signed_info_buf.push('>');
                }
                match local.as_str() {
                    "NameID" => {
                        in_name_id = false;
                    }
                    "AttributeValue" => {
                        in_attr_value = false;
                    }
                    "Attribute" => {
                        flush_attr(
                            &mut current_attr_name,
                            &mut attr_values,
                            &mut email,
                            &mut display_name,
                            &mut groups,
                            &mut raw_claims,
                        );
                    }
                    "SignatureValue" => {
                        in_signature_value = false;
                    }
                    "SignedInfo" => {
                        signed_info_buf.push_str("</SignedInfo>");
                        sig.signed_info_raw = Some(signed_info_buf.clone());
                        signed_info_buf.clear();
                        in_signed_info = false;
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
                if in_signed_info && !text.is_empty() {
                    signed_info_buf.push_str(&text);
                }
                if in_signature_value && !text.is_empty() {
                    // Concatenate (base64 may be wrapped across lines).
                    match sig.signature_value {
                        Some(ref mut v) => v.push_str(&text),
                        None => sig.signature_value = Some(text.clone()),
                    }
                } else if in_name_id && !text.is_empty() {
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

    Ok((
        SsoUserInfo {
            subject,
            email,
            name: display_name,
            groups,
            raw_claims,
        },
        sig,
    ))
}

/// Read the `Name` attribute (local name) from an element.
fn read_attr_name(e: &BytesStart<'_>) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| local_name(a.key.as_ref()) == "Name")
        .and_then(|a| {
            std::str::from_utf8(a.value.as_ref())
                .map(|s| s.to_string())
                .ok()
        })
}

/// Append a start element's raw bytes to a buffer (for SignedInfo capture).
fn append_start_to_raw(e: &BytesStart<'_>, buf: &mut String) {
    buf.push('<');
    buf.push_str(&String::from_utf8_lossy(e.name().as_ref()));
    for attr in e.attributes().filter_map(|a| a.ok()) {
        buf.push(' ');
        buf.push_str(&String::from_utf8_lossy(attr.key.as_ref()));
        buf.push_str("=\"");
        buf.push_str(&String::from_utf8_lossy(attr.value.as_ref()));
        buf.push('"');
    }
    buf.push('>');
}

/// Append a self-closing empty element to a buffer (for SignedInfo capture).
fn append_empty_to_raw(e: &BytesStart<'_>, buf: &mut String) {
    buf.push('<');
    buf.push_str(&String::from_utf8_lossy(e.name().as_ref()));
    for attr in e.attributes().filter_map(|a| a.ok()) {
        buf.push(' ');
        buf.push_str(&String::from_utf8_lossy(attr.key.as_ref()));
        buf.push_str("=\"");
        buf.push_str(&String::from_utf8_lossy(attr.value.as_ref()));
        buf.push('"');
    }
    buf.push_str("/>");
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

    // ── Test key material (throwaway; never a production key) ───────────
    const TEST_RSA_PKCS8_PEM: &str = concat!(
        "-----BEGIN PRIVATE KEY-----\n",
        "MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDq/z2c1qAgjqzJ\n",
        "yG4PCBptlY32sLnYcqOg9H6Q4Ql84xTM9YuP5ZOt1iXOsM8dAYToWOQzLoUIro1g\n",
        "2Rm2DH7QDbIt4PEaYop3AFhPHp1ZFjP6hRzTRINNBDxOSm0uz7H32YRTYokV6Ibn\n",
        "w6SdDPmJqFRgXf9qxBk+d9ljyLiIUQvOzQ+YOPHSos/k4HIPSo3f9U6Pwf+S3p71\n",
        "zV7nqgoRX5whJ8pMQqpX5ZMW9+cAa2zTPd+i8aZxmZHd9gySUqUoumKVm488ysSS\n",
        "nPwjbaA11dgAsDX9zziyZD/cKYFPF0DgLsL9wYcE87Qq52AYh2/zhG/g+FoxOje4\n",
        "QqWw25DHAgMBAAECggEABM/hNRr4AHKrex5NkqU51VCgrZKE27fNPfiDtvfEt/f2\n",
        "bxQAHZw33/FoqMjaFN/5FsDrO1kShFD+uCL58c5jsmL1aRcYGNA3waQSKtyXoEFi\n",
        "Ixkis/jNL4CMs5W2kqTSIh8kJIj6AabXTFunPUgMvBLkV2zVVBxb3/mYTADKNpBY\n",
        "QAEvsu/nToWWg49TgiixpA1k9RIYAQHcI8ZAugjnqiFnicyTthevWQ2cvBwqt4UB\n",
        "lwAASAf2P4qBeogkam+TFvtrnYi6rGskV+4rSRgkrbx5LAZsTVkMDC/eLmbScvyH\n",
        "t1TIpBt5PBw+YJ3hwkpPo+5fYJLWfvEj4rISYu38oQKBgQD11zPoKBlq638QGMxX\n",
        "SWB5o8XBuIw8N+K7GHYZWSUfoWlqy5kXE6A0YWBCdS8x3v84LJKokpZt5un/eACb\n",
        "q7q+o0RLblKLKasurNTCFrvQPtt1ftkidfunOJCbl7Nd5UBTckK80gG2PT5zfZKL\n",
        "B2M+EUV41AUF/tRfVyIQGR9h6QKBgQD0tVI/c0BKiqQfuKbJOduelaU/TamKTIMc\n",
        "ZjtTPJAJhkR3r5L9fWxkRbGcm4H39NZwKsC3oUmeHvQNDMFfn0A9yILyuhrP+BIW\n",
        "Q+t552ohQGRGLfujyfQc4HjFBthogU6XO3dIxFXg4iHkkQynQG+/w1ch+1wK/KxE\n",
        "wy+jhJZ/LwKBgQCbwoz1s6pfDvRDi6K0Tx5cE4Kxea8IXFRAPIBfERcvUkKLUpId\n",
        "h+bCKUwm7z5Gt8Y2ni8RtUawPVTW8v5Xo1e/f4w+yphr6au29/QZQPQgPiMn74W9\n",
        "isk2KuWcX2JaxGycMlHMdrZ085rE67PUeIrNgX3lz1ebc9i0y20ei/xROQKBgQC0\n",
        "FP/bC9CjSpXvdi7fZQG3Gb9K77c1vIq8Covb/HSvXazjO0T74SI0RImpi1NBC2AH\n",
        "mZ7LRBluEK9fLyTbXtGi5f1f7Q8wPwnocsFGq8ORhtaEQvCtn0BTQ+n8bMYzWf1h\n",
        "E/T7iuj8Hs38a7YZGzVhtLpZmqYou7t2uwFC3571JwKBgQDHnEyfkhdrs1GLEnZk\n",
        "LwENaWqvJoGzwWxv4cxA+oeU/olO81WL7zLS7IvvnYa3HXlMEMQZqeyUfBmbnD3a\n",
        "fnj8Y97nHrPh0yreJ7leg7GeY+Vw2QziOUGTeGbXgmqHrE6Amm9I7/Plfgjp0iRn\n",
        "qGQyLF8/TU5I4e0EAxaonr/FSA==\n",
        "-----END PRIVATE KEY-----\n",
    );

    /// A self-signed X.509 certificate wrapping the public half of
    /// [`TEST_RSA_PKCS8_PEM`]. Exercises the DER SPKI-extraction path.
    const TEST_IDP_CERT_PEM: &str = concat!(
        "-----BEGIN CERTIFICATE-----\n",
        "MIIDEzCCAfugAwIBAgIUK6kvDfsavgmpiKZzxr2E2xMsc+4wDQYJKoZIhvcNAQEL\n",
        "BQAwGTEXMBUGA1UEAwwOb3hpcnMtdGVzdC1pZHAwHhcNMjYwNzE2MTg0MjQ5WhcN\n",
        "MzYwNzEzMTg0MjQ5WjAZMRcwFQYDVQQDDA5veGlycy10ZXN0LWlkcDCCASIwDQYJ\n",
        "KoZIhvcNAQEBBQADggEPADCCAQoCggEBAOr/PZzWoCCOrMnIbg8IGm2Vjfawudhy\n",
        "o6D0fpDhCXzjFMz1i4/lk63WJc6wzx0BhOhY5DMuhQiujWDZGbYMftANsi3g8Rpi\n",
        "incAWE8enVkWM/qFHNNEg00EPE5KbS7PsffZhFNiiRXohufDpJ0M+YmoVGBd/2rE\n",
        "GT532WPIuIhRC87ND5g48dKiz+Tgcg9Kjd/1To/B/5LenvXNXueqChFfnCEnykxC\n",
        "qlflkxb35wBrbNM936LxpnGZkd32DJJSpSi6YpWbjzzKxJKc/CNtoDXV2ACwNf3P\n",
        "OLJkP9wpgU8XQOAuwv3BhwTztCrnYBiHb/OEb+D4WjE6N7hCpbDbkMcCAwEAAaNT\n",
        "MFEwHQYDVR0OBBYEFC7gZrbLPNzsT1eGt/WqmrOIuYyFMB8GA1UdIwQYMBaAFC7g\n",
        "ZrbLPNzsT1eGt/WqmrOIuYyFMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEL\n",
        "BQADggEBAFYqrFxAsm15TzjoJlYMUZ3ceegStOG0oYB9RD+78KqQuf6yRRd3K1Xa\n",
        "Fp87iKSLX7qFdE7z+vS65r4JlZr0zT9JWZU5zyVeJbm+SHHSBCfZq7KakzASbAZv\n",
        "JStJIQjNUIYUv+UjMOlZheszKlByx2cUfT3p3d+Sj9zysqJN5Zc95Mi1NygQjPmQ\n",
        "+/sh2+0V2dme9CPQESRgnPTvdKmXav2O5SGtJ4CX7/8lpMLciIXXn9i1JEGLtWAy\n",
        "DzDGpQr2C39GXkmfYJA8RrgrT7APro5coWvdCo0krLoBmL1fyAb1mKOL23upHZhu\n",
        "AJus/O1WcfxGFoBXsgOK+HfIUMJ5wgk=\n",
        "-----END CERTIFICATE-----\n",
    );

    const UNSIGNED_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
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
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"#;

    fn rsa_private_key() -> rsa::RsaPrivateKey {
        use rsa::pkcs8::DecodePrivateKey;
        rsa::RsaPrivateKey::from_pkcs8_pem(TEST_RSA_PKCS8_PEM).expect("parse test RSA key")
    }

    /// Sign the reconstructed SignedInfo bytes with the test key (base64).
    fn sign_signed_info(signed_info: &str) -> String {
        use rsa::pkcs1v15::SigningKey;
        use rsa::signature::{SignatureEncoding, Signer};
        let key = SigningKey::<rsa::sha2::Sha256>::new(rsa_private_key());
        let sig = key.try_sign(signed_info.as_bytes()).expect("sign");
        base64::engine::general_purpose::STANDARD.encode(sig.to_bytes())
    }

    /// Build a signed SAMLResponse with the given SignatureValue.
    fn signed_response_xml(signature_value: &str) -> String {
        format!(
            r##"<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
  <ds:Signature>
    <ds:SignedInfo>
      <ds:CanonicalizationMethod Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>
      <ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/>
      <ds:Reference URI="#assertion-1">
        <ds:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>
        <ds:DigestValue>PLACEHOLDER_DIGEST</ds:DigestValue>
      </ds:Reference>
    </ds:SignedInfo>
    <ds:SignatureValue>{signature_value}</ds:SignatureValue>
  </ds:Signature>
  <saml:Assertion ID="assertion-1">
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">alice@example.com</saml:NameID>
    </saml:Subject>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>alice@example.com</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"##
        )
    }

    /// Build a correctly-signed SAMLResponse (base64) whose signature verifies
    /// against [`TEST_IDP_CERT_PEM`].
    fn correctly_signed_b64() -> String {
        let placeholder = signed_response_xml("PENDING");
        let (_, sig) = parse_saml_xml(&placeholder).expect("parse placeholder");
        let signed_info = sig.signed_info_raw.expect("signed_info captured");
        let sigval = sign_signed_info(&signed_info);
        let signed_xml = signed_response_xml(&sigval);
        base64::engine::general_purpose::STANDARD.encode(signed_xml.as_bytes())
    }

    /// An unsigned response is REJECTED by default (fail closed).
    #[test]
    fn test_saml_unsigned_rejected_by_default() {
        let helper = SamlSpHelper::with_certificate(make_saml_config(), TEST_IDP_CERT_PEM);
        let b64 = base64::engine::general_purpose::STANDARD.encode(UNSIGNED_XML.as_bytes());
        let err = helper
            .parse_response(&b64)
            .expect_err("unsigned response must be rejected");
        assert!(
            matches!(err, SsoError::UnsignedAssertionRejected),
            "expected UnsignedAssertionRejected, got: {err}"
        );
    }

    /// Unsigned responses are only accepted when explicitly opted in.
    #[test]
    fn test_saml_unsigned_allowed_when_opted_in() {
        let helper = SamlSpHelper::new_allow_unsigned(make_saml_config());
        let b64 = base64::engine::general_purpose::STANDARD.encode(UNSIGNED_XML.as_bytes());
        let user_info = helper
            .parse_response(&b64)
            .expect("allow_unsigned accepts unsigned response");
        assert!(user_info.subject.contains("alice@example.com"));
    }

    /// A correctly-signed response verifies and yields the identity.
    #[test]
    fn test_saml_valid_signature_accepted() {
        let helper = SamlSpHelper::with_certificate(make_saml_config(), TEST_IDP_CERT_PEM);
        let b64 = correctly_signed_b64();
        let user_info = helper
            .parse_response(&b64)
            .expect("correctly-signed response must be accepted");
        assert!(user_info.subject.contains("alice@example.com"));
        assert_eq!(user_info.email.as_deref(), Some("alice@example.com"));
    }

    /// Adversarial: tampering with the signed content after signing is detected.
    #[test]
    fn test_saml_tampered_signed_info_rejected() {
        let placeholder = signed_response_xml("PENDING");
        let (_, sig) = parse_saml_xml(&placeholder).expect("parse");
        let signed_info = sig.signed_info_raw.expect("signed_info");
        let sigval = sign_signed_info(&signed_info);
        // Tamper the DigestValue AFTER signing -> reconstruction changes.
        let tampered_xml =
            signed_response_xml(&sigval).replace("PLACEHOLDER_DIGEST", "EVIL_DIGEST_VALUE");
        let b64 = base64::engine::general_purpose::STANDARD.encode(tampered_xml.as_bytes());
        let helper = SamlSpHelper::with_certificate(make_saml_config(), TEST_IDP_CERT_PEM);
        let err = helper
            .parse_response(&b64)
            .expect_err("tampered response must be rejected");
        assert!(
            matches!(err, SsoError::SignatureInvalid),
            "expected SignatureInvalid, got: {err}"
        );
    }

    /// Adversarial: a garbage/forged SignatureValue is rejected.
    #[test]
    fn test_saml_forged_signature_value_rejected() {
        let forged = base64::engine::general_purpose::STANDARD.encode([0u8; 256]);
        let xml = signed_response_xml(&forged);
        let b64 = base64::engine::general_purpose::STANDARD.encode(xml.as_bytes());
        let helper = SamlSpHelper::with_certificate(make_saml_config(), TEST_IDP_CERT_PEM);
        let err = helper
            .parse_response(&b64)
            .expect_err("forged signature must be rejected");
        assert!(matches!(err, SsoError::SignatureInvalid), "got: {err}");
    }

    /// A signed response with NO configured certificate fails closed.
    #[test]
    fn test_saml_signed_but_no_cert_configured_rejected() {
        let helper = SamlSpHelper::new(make_saml_config()); // no cert
        let b64 = correctly_signed_b64();
        let err = helper
            .parse_response(&b64)
            .expect_err("cannot verify without a cert");
        assert!(
            matches!(err, SsoError::SignatureVerificationUnavailable),
            "expected SignatureVerificationUnavailable, got: {err}"
        );
    }

    /// The DER SPKI extractor produces a key the `rsa` crate can load.
    #[test]
    fn test_extract_spki_from_x509_cert() {
        let spki = load_idp_spki_der(TEST_IDP_CERT_PEM).expect("extract SPKI");
        RsaPublicKey::from_public_key_der(&spki).expect("SPKI parses as RSA public key");
    }
}
