//! XML utility helpers for SAML 2.0 processing.
//!
//! Split out of the original `saml` module (Round 32 refactor). Contains the
//! pure XML escaping, attribute extraction, raw-element capture, datetime
//! parsing, and PEM/DER decoding helpers shared by the parser and provider.

use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use quick_xml::events::BytesStart;

use crate::error::{FusekiError, FusekiResult};

/// Escape XML special characters in attribute values and text content.
pub(super) fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Append a `name="value"` pair to a tag buffer.
pub(super) fn write_xml_attr(buf: &mut String, name: &str, value: &str) {
    buf.push(' ');
    buf.push_str(name);
    buf.push_str("=\"");
    buf.push_str(&xml_escape(value));
    buf.push('"');
}

/// Extract the local name (strip namespace prefix) from a `BytesStart`.
pub(super) fn local_name(e: &BytesStart<'_>) -> String {
    local_name_bytes(e.name().local_name().into_inner())
}

/// Convert raw local-name bytes to a String.
pub(super) fn local_name_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Get an attribute value from a `BytesStart` element by local name.
/// Uses `unescape_value()` which handles XML entity unescaping (e.g. `&amp;` → `&`).
pub(super) fn attr_value(e: &BytesStart<'_>, name: &str) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.local_name().into_inner() == name.as_bytes())
        .and_then(|a| a.unescape_value().ok().map(|v| v.into_owned()))
}

/// Append a start element's raw bytes to a string (for SignedInfo capture).
pub(super) fn append_start_to_raw(e: &BytesStart<'_>, buf: &mut String) {
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
pub(super) fn append_empty_to_raw(e: &BytesStart<'_>, buf: &mut String) {
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
pub(super) fn parse_datetime(s: &str) -> FusekiResult<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| FusekiError::parse(format!("Invalid SAML datetime '{}': {}", s, e)))
}

/// Decode a PEM body (between -----BEGIN ... ----- headers) to DER bytes.
pub(super) fn pem_body_to_der(pem: &str) -> FusekiResult<Vec<u8>> {
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
