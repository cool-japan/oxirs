//! Common helpers shared by all cloud backend modules.
//!
//! Holds the HMAC-SHA256 primitives, hex/SHA-256 wrappers, URL helpers, and
//! the naive XML extractors used by the S3 and Azure list parsers.

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

pub(crate) type HmacSha256 = Hmac<Sha256>;

/// Build a `reqwest` client using the Pure Rust rustls TLS stack.
///
/// `reqwest` is configured with the `rustls-no-provider` feature, so a process-
/// default rustls `CryptoProvider` must be installed before constructing any
/// client — without one, `build()` panics. oxirs-core installs the Pure Rust
/// provider via a pre-`main` ctor, but oxirs-samm references no other oxirs-core
/// symbol, so the linker would otherwise drop that dependency (and its ctor)
/// from the test binaries. Calling `ensure_crypto_provider()` here is idempotent
/// and also forces oxirs-core to be linked into every binary and test harness
/// that builds a cloud backend.
pub(crate) fn build_tls_client() -> Result<reqwest::Client, String> {
    oxirs_core::ensure_crypto_provider();
    reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .map_err(|e| format!("Failed to build reqwest client: {e}"))
}

/// Sign bytes with HMAC-SHA256, returning the raw digest.
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take a key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Hex-encode bytes.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// SHA-256 hash of bytes returned as lowercase hex string.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&Sha256::digest(data))
}

/// Extract the `host[:port]` component from a URL string.
pub(crate) fn extract_host(url: &str) -> Result<String, String> {
    let after_scheme = url
        .find("://")
        .map(|i| &url[i + 3..])
        .ok_or_else(|| format!("URL has no scheme: {url}"))?;

    let host = after_scheme
        .split('/')
        .next()
        .ok_or_else(|| format!("Cannot extract host from URL: {url}"))?;
    Ok(host.to_string())
}

/// Percent-encode a string for use in URL query parameters.
pub(crate) fn url_encode(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c]
            }
            other => {
                let mut buf = [0u8; 4];
                let bytes = other.encode_utf8(&mut buf);
                bytes
                    .as_bytes()
                    .iter()
                    .flat_map(|b| format!("%{:02X}", b).chars().collect::<Vec<_>>())
                    .collect()
            }
        })
        .collect()
}

/// Naive XML parser that extracts `<Key>...</Key>` values from an S3 ListObjectsV2 response.
pub(crate) fn parse_s3_list_xml(xml: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut remaining = xml;
    while let Some(start) = remaining.find("<Key>") {
        let rest = &remaining[start + 5..];
        if let Some(end) = rest.find("</Key>") {
            keys.push(rest[..end].to_string());
            remaining = &rest[end + 6..];
        } else {
            break;
        }
    }
    keys
}

/// Naive XML parser that extracts `<Name>...</Name>` values from an Azure ListBlobs response.
pub(crate) fn parse_azure_list_xml(xml: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut remaining = xml;
    while let Some(start) = remaining.find("<Name>") {
        let rest = &remaining[start + 6..];
        if let Some(end) = rest.find("</Name>") {
            names.push(rest[..end].to_string());
            remaining = &rest[end + 7..];
        } else {
            break;
        }
    }
    names
}
