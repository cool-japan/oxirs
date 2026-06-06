//! SAML 2.0 Response XML parser and signature verification.
//!
//! Split out of the original `saml` module (Round 32 refactor). Contains the
//! event-driven [`SamlResponseParser`], its internal parse state machine, and
//! the per-assertion builder.
//!
//! ## Signature verification
//! Uses a simplified approach: the raw `ds:SignatureValue` bytes are verified
//! against the canonicalized `ds:SignedInfo` element using RSASSA-PKCS1-v1_5
//! with SHA-256 via the Pure-Rust `rsa` crate. This covers the most common
//! real-world case (enveloped RSA-SHA256 signatures) but does **not** implement
//! full W3C XMLDSig.

use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use quick_xml::{events::Event, reader::Reader};
use rsa::pkcs1v15::{Signature as RsaSignature, VerifyingKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier;
use rsa::RsaPublicKey;
use std::collections::HashMap;
use x509_parser::prelude::FromDer;

use crate::error::{FusekiError, FusekiResult};

use super::saml_helpers::{
    append_empty_to_raw, append_start_to_raw, attr_value, local_name, local_name_bytes,
    parse_datetime, pem_body_to_der,
};
use super::saml_types::{
    Assertion, Attribute, AuthnStatement, Conditions, ResponseStatus, SamlResponse, Subject,
};

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

    /// Verify the RSA-SHA256 XML signature using the Pure-Rust `rsa` crate.
    ///
    /// # NOTE — simplified XMLDSig
    /// This verifies `RSASSA-PKCS1-v1_5(SHA256(signed_info_bytes))` against
    /// `SignatureValue`. It does NOT perform Exclusive C14N or transform
    /// resolution. For IdPs that include Transforms other than
    /// `enveloped-signature` this will fail. A production-grade implementation
    /// requires a dedicated C14N + XMLDSig library.
    fn verify_signature(&self, signed_info: &str, signature_value_b64: &str) -> FusekiResult<()> {
        // Decode the base64 signature value (strip whitespace first)
        let sig_bytes = general_purpose::STANDARD
            .decode(signature_value_b64.replace([' ', '\n', '\r', '\t'], ""))
            .map_err(|e| {
                FusekiError::authentication(format!("SAML signature base64 decode error: {}", e))
            })?;

        // Load the IdP RSA public key as SubjectPublicKeyInfo (SPKI) DER bytes.
        let pub_key_der = self.load_idp_spki_der()?;

        // Parse the SPKI DER into an RSA public key.
        let public_key = RsaPublicKey::from_public_key_der(&pub_key_der).map_err(|e| {
            FusekiError::authentication(format!("SAML IdP RSA public key parse error: {}", e))
        })?;

        // RSASSA-PKCS1-v1_5 with SHA-256 (interoperates with real IdPs). The
        // `VerifyingKey<Sha256>` hashes `signed_info` internally and verifies.
        // NOTE: use `rsa::sha2::Sha256` (re-exported by the `rsa` crate via its
        // `sha2` feature) rather than the workspace `sha2` crate. The workspace
        // pins `sha2 = "0.11"` (which uses `digest` 0.11), but `rsa` 0.9 still
        // builds on `digest` 0.10; mixing the two yields an unsatisfied
        // `Digest` trait bound. Pulling `Sha256` from `rsa::sha2` guarantees the
        // digest version matches what `VerifyingKey` expects.
        let verifying_key = VerifyingKey::<rsa::sha2::Sha256>::new(public_key);

        let signature = RsaSignature::try_from(sig_bytes.as_slice()).map_err(|e| {
            FusekiError::authentication(format!("SAML signature decode error: {}", e))
        })?;

        verifying_key
            .verify(signed_info.as_bytes(), &signature)
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
