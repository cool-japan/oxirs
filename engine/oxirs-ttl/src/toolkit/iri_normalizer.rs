//! IRI Normalization per RFC 3987 Section 5.3
//!
//! This module provides IRI normalization to convert IRIs into a canonical form
//! for efficient comparison and storage. Normalization includes:
//!
//! - Case normalization (scheme and host to lowercase)
//! - Percent-encoding normalization (decode unreserved characters)
//! - Path normalization (remove unnecessary dot-segments)
//! - Default port removal (http:80, https:443, etc.)
//! - Empty path to "/" for hierarchical URIs
//!
//! # RFC 3987 Compliance
//!
//! Implements the normalization algorithm from RFC 3987 Section 5.3 with
//! additional enhancements from RFC 3986.
//!
//! # Example
//!
//! ```
//! use oxirs_ttl::toolkit::iri_normalizer::{normalize_iri, NormalizedIri};
//!
//! // Case normalization
//! let iri = normalize_iri("HTTP://EXAMPLE.ORG/Path").unwrap();
//! assert_eq!(iri.as_str(), "http://example.org/Path");
//!
//! // Percent-encoding normalization
//! let iri = normalize_iri("http://example.org/%7Euser").unwrap();
//! assert_eq!(iri.as_str(), "http://example.org/~user");
//!
//! // Default port removal
//! let iri = normalize_iri("http://example.org:80/path").unwrap();
//! assert_eq!(iri.as_str(), "http://example.org/path");
//!
//! // Path normalization
//! let iri = normalize_iri("http://example.org/a/./b/../c").unwrap();
//! assert_eq!(iri.as_str(), "http://example.org/a/c");
//! ```

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

/// A normalized IRI in canonical form
///
/// This type represents an IRI that has been normalized according to RFC 3987.
/// Normalized IRIs can be efficiently compared for equivalence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalizedIri {
    /// The normalized IRI string
    iri: String,
}

impl NormalizedIri {
    /// Create a normalized IRI from a string
    ///
    /// This constructor assumes the IRI is already normalized.
    /// Use `normalize_iri()` to normalize an arbitrary IRI.
    pub fn new_unchecked(iri: String) -> Self {
        Self { iri }
    }

    /// Get the normalized IRI as a string slice
    pub fn as_str(&self) -> &str {
        &self.iri
    }

    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.iri
    }

    /// Check if two IRIs are equivalent (same as PartialEq but explicit)
    pub fn is_equivalent(&self, other: &Self) -> bool {
        self.iri == other.iri
    }
}

impl fmt::Display for NormalizedIri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.iri)
    }
}

impl AsRef<str> for NormalizedIri {
    fn as_ref(&self) -> &str {
        &self.iri
    }
}

/// Error types for IRI normalization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationError {
    /// Invalid IRI format
    InvalidFormat(String),
    /// Invalid percent encoding
    InvalidPercentEncoding(String),
    /// Missing scheme
    MissingScheme,
    /// Invalid scheme
    InvalidScheme(String),
}

impl fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "Invalid IRI format: {}", msg),
            Self::InvalidPercentEncoding(seq) => {
                write!(f, "Invalid percent encoding: {}", seq)
            }
            Self::MissingScheme => write!(f, "IRI must have a scheme"),
            Self::InvalidScheme(s) => write!(f, "Invalid scheme: {}", s),
        }
    }
}

impl std::error::Error for NormalizationError {}

/// Result type for normalization operations
pub type NormalizationResult<T> = Result<T, NormalizationError>;

/// Normalize an IRI to canonical form
///
/// This function applies all normalization steps defined in RFC 3987:
/// - Case normalization (scheme and host to lowercase)
/// - Percent-encoding normalization
/// - Path normalization
/// - Default port removal
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::iri_normalizer::normalize_iri;
///
/// let iri = normalize_iri("HTTP://EXAMPLE.ORG:80/A/./B/../C").unwrap();
/// assert_eq!(iri.as_str(), "http://example.org/A/C");
/// ```
pub fn normalize_iri(iri: &str) -> NormalizationResult<NormalizedIri> {
    if iri.is_empty() {
        return Err(NormalizationError::InvalidFormat(
            "IRI cannot be empty".to_string(),
        ));
    }

    // Parse components
    let components = parse_iri_components(iri)?;

    // Apply normalization
    let normalized = normalize_components(&components)?;

    Ok(NormalizedIri::new_unchecked(normalized))
}

/// IRI components for normalization
#[derive(Debug, Clone)]
struct IriComponents {
    scheme: String,
    authority: Option<Authority>,
    path: String,
    query: Option<String>,
    fragment: Option<String>,
}

/// Authority component (userinfo@host:port)
#[derive(Debug, Clone)]
struct Authority {
    userinfo: Option<String>,
    host: String,
    port: Option<u16>,
}

/// Parse IRI into components
fn parse_iri_components(iri: &str) -> NormalizationResult<IriComponents> {
    // Extract scheme
    let colon_pos = iri.find(':').ok_or(NormalizationError::MissingScheme)?;
    let scheme = iri[..colon_pos].to_string();

    if scheme.is_empty() || !is_valid_scheme(&scheme) {
        return Err(NormalizationError::InvalidScheme(scheme));
    }

    let rest = &iri[colon_pos + 1..];

    // Check for authority (starts with //)
    let (authority, path_query_fragment) = if let Some(after_slashes) = rest.strip_prefix("//") {
        let auth_end = after_slashes
            .find(['/', '?', '#'])
            .unwrap_or(after_slashes.len());
        let authority_str = &after_slashes[..auth_end];
        let authority = parse_authority(authority_str)?;
        (Some(authority), &after_slashes[auth_end..])
    } else {
        (None, rest)
    };

    // Parse path, query, and fragment
    let (path, query_fragment) = if let Some(q_pos) = path_query_fragment.find('?') {
        (
            path_query_fragment[..q_pos].to_string(),
            &path_query_fragment[q_pos + 1..],
        )
    } else if let Some(f_pos) = path_query_fragment.find('#') {
        (
            path_query_fragment[..f_pos].to_string(),
            &path_query_fragment[f_pos..],
        )
    } else {
        (path_query_fragment.to_string(), "")
    };

    let (query, fragment) = if !query_fragment.is_empty() {
        if let Some(f_pos) = query_fragment.find('#') {
            (
                Some(query_fragment[..f_pos].to_string()),
                Some(query_fragment[f_pos + 1..].to_string()),
            )
        } else {
            (Some(query_fragment.to_string()), None)
        }
    } else {
        (None, None)
    };

    Ok(IriComponents {
        scheme,
        authority,
        path,
        query,
        fragment,
    })
}

/// Parse authority component
fn parse_authority(authority: &str) -> NormalizationResult<Authority> {
    if authority.is_empty() {
        return Ok(Authority {
            userinfo: None,
            host: String::new(),
            port: None,
        });
    }

    // Split userinfo@host:port
    let (userinfo, host_port) = if let Some(at_pos) = authority.rfind('@') {
        (
            Some(authority[..at_pos].to_string()),
            &authority[at_pos + 1..],
        )
    } else {
        (None, authority)
    };

    // Parse host and port
    let (host, port) = parse_host_port(host_port)?;

    Ok(Authority {
        userinfo,
        host,
        port,
    })
}

/// Parse host:port
fn parse_host_port(host_port: &str) -> NormalizationResult<(String, Option<u16>)> {
    // IPv6 address
    if let Some(bracket_start) = host_port.find('[') {
        let bracket_end = host_port.find(']').ok_or_else(|| {
            NormalizationError::InvalidFormat("Unclosed IPv6 bracket".to_string())
        })?;
        let host = host_port[bracket_start..=bracket_end].to_string();
        let rest = &host_port[bracket_end + 1..];
        let port = if let Some(port_str) = rest.strip_prefix(':') {
            Some(port_str.parse::<u16>().map_err(|_| {
                NormalizationError::InvalidFormat(format!("Invalid port: {}", port_str))
            })?)
        } else if rest.is_empty() {
            None
        } else {
            return Err(NormalizationError::InvalidFormat(
                "Invalid characters after IPv6 address".to_string(),
            ));
        };
        return Ok((host, port));
    }

    // Regular host:port
    if let Some(colon_pos) = host_port.rfind(':') {
        let potential_port = &host_port[colon_pos + 1..];
        // Only treat as port if it's all digits
        if potential_port.chars().all(|c| c.is_ascii_digit()) {
            let port = potential_port.parse::<u16>().map_err(|_| {
                NormalizationError::InvalidFormat(format!("Invalid port: {}", potential_port))
            })?;
            Ok((host_port[..colon_pos].to_string(), Some(port)))
        } else {
            Ok((host_port.to_string(), None))
        }
    } else {
        Ok((host_port.to_string(), None))
    }
}

/// Normalize IRI components
fn normalize_components(components: &IriComponents) -> NormalizationResult<String> {
    // 1. Case normalization: scheme to lowercase
    let scheme = components.scheme.to_lowercase();

    // 2. Authority normalization
    let authority_str = if let Some(ref auth) = components.authority {
        let mut parts = Vec::new();

        // Userinfo (if present)
        if let Some(ref userinfo) = auth.userinfo {
            let normalized_userinfo = normalize_percent_encoding(userinfo)?;
            parts.push(format!("{}@", normalized_userinfo));
        }

        // Host: lowercase for reg-name (not IPv6)
        let normalized_host = if auth.host.starts_with('[') {
            // IPv6: keep as-is (case-insensitive hex is normalized to lowercase)
            auth.host.to_lowercase()
        } else {
            // Reg-name: lowercase + percent-encoding normalization
            normalize_percent_encoding(&auth.host.to_lowercase())?
        };
        parts.push(normalized_host);

        // Port: remove default ports
        if let Some(port) = auth.port {
            if !is_default_port(&scheme, port) {
                parts.push(format!(":{}", port));
            }
        }

        format!("//{}", parts.concat())
    } else {
        String::new()
    };

    // 3. Path normalization
    let normalized_path = normalize_path(&components.path, components.authority.is_some())?;

    // 4. Query normalization
    let query_str = if let Some(ref query) = components.query {
        format!("?{}", normalize_percent_encoding(query)?)
    } else {
        String::new()
    };

    // 5. Fragment normalization
    let fragment_str = if let Some(ref fragment) = components.fragment {
        format!("#{}", normalize_percent_encoding(fragment)?)
    } else {
        String::new()
    };

    // Reconstruct normalized IRI
    Ok(format!(
        "{}:{}{}{}{}",
        scheme, authority_str, normalized_path, query_str, fragment_str
    ))
}

/// Normalize percent-encoding (decode unreserved characters)
///
/// RFC 3986: Unreserved characters are A-Z, a-z, 0-9, -, ., _, ~
/// These should not be percent-encoded.
fn normalize_percent_encoding(s: &str) -> NormalizationResult<String> {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '%' {
            // Read next two characters
            let hex1 = chars
                .next()
                .ok_or_else(|| NormalizationError::InvalidPercentEncoding(format!("%{}", s)))?;
            let hex2 = chars.next().ok_or_else(|| {
                NormalizationError::InvalidPercentEncoding(format!("%{}{}", hex1, s))
            })?;

            let hex_str = format!("{}{}", hex1, hex2);
            let byte = u8::from_str_radix(&hex_str, 16)
                .map_err(|_| NormalizationError::InvalidPercentEncoding(format!("%{}", hex_str)))?;

            // Check if it's an unreserved character
            let decoded = byte as char;
            if is_unreserved(decoded) {
                // Decode unreserved characters
                result.push(decoded);
            } else {
                // Keep percent-encoded (normalize to uppercase)
                result.push_str(&format!("%{}", hex_str.to_uppercase()));
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Check if a character is unreserved (RFC 3986)
fn is_unreserved(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '-' || ch == '.' || ch == '_' || ch == '~'
}

/// Normalize path component
fn normalize_path(path: &str, has_authority: bool) -> NormalizationResult<String> {
    // Empty path for hierarchical URIs should become "/"
    if path.is_empty() && has_authority {
        return Ok("/".to_string());
    }

    // Remove dot segments
    let normalized = remove_dot_segments(path);

    // Percent-encoding normalization
    normalize_percent_encoding(&normalized)
}

/// Remove dot segments from path (RFC 3986 Section 5.2.4)
fn remove_dot_segments(path: &str) -> String {
    let mut output = Vec::new();
    let segments: Vec<&str> = path.split('/').collect();
    let has_trailing_slash = path.ends_with('/') && path.len() > 1;

    for (i, segment) in segments.iter().enumerate() {
        match *segment {
            "" => {
                // Skip empty segments except at the beginning
                if i == 0 {
                    // Keep leading slash
                }
            }
            "." => {
                // Skip current directory
            }
            ".." => {
                // Go up one level (pop last segment)
                output.pop();
            }
            _ => {
                // Regular segment
                output.push(*segment);
            }
        }
    }

    // Reconstruct path
    if path.starts_with('/') {
        if output.is_empty() {
            "/".to_string()
        } else {
            let base_path = format!("/{}", output.join("/"));
            if has_trailing_slash {
                format!("{}/", base_path)
            } else {
                base_path
            }
        }
    } else if output.is_empty() {
        String::new()
    } else {
        let base_path = output.join("/");
        if has_trailing_slash {
            format!("{}/", base_path)
        } else {
            base_path
        }
    }
}

/// Check if port is the default for the scheme
fn is_default_port(scheme: &str, port: u16) -> bool {
    get_default_port(scheme) == Some(port)
}

/// Get default port for a scheme
fn get_default_port(scheme: &str) -> Option<u16> {
    DEFAULT_PORTS.get(scheme).copied()
}

// Default ports for common schemes
lazy_static::lazy_static! {
    static ref DEFAULT_PORTS: HashMap<&'static str, u16> = {
        let mut m = HashMap::new();
        m.insert("http", 80);
        m.insert("https", 443);
        m.insert("ftp", 21);
        m.insert("ftps", 990);
        m.insert("ssh", 22);
        m.insert("telnet", 23);
        m.insert("smtp", 25);
        m.insert("pop3", 110);
        m.insert("imap", 143);
        m.insert("ldap", 389);
        m.insert("ldaps", 636);
        m.insert("ws", 80);
        m.insert("wss", 443);
        m
    };
}

/// Check if a string is a valid scheme
fn is_valid_scheme(scheme: &str) -> bool {
    if scheme.is_empty() {
        return false;
    }
    let mut chars = scheme.chars();

    // First character must be ASCII letter
    let first = chars.next().expect("iterator should have next element");
    if !first.is_ascii_alphabetic() {
        return false;
    }

    // Rest can be ASCII letter, digit, +, -, .
    chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

/// Compare two IRIs for equivalence
///
/// This function normalizes both IRIs and compares them.
/// Use this for IRI comparison instead of string equality.
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::iri_normalizer::iris_equivalent;
///
/// assert!(iris_equivalent(
///     "HTTP://EXAMPLE.ORG/path",
///     "http://example.org/path"
/// ).unwrap());
///
/// assert!(iris_equivalent(
///     "http://example.org:80/path",
///     "http://example.org/path"
/// ).unwrap());
///
/// assert!(!iris_equivalent(
///     "http://example.org/path1",
///     "http://example.org/path2"
/// ).unwrap());
/// ```
pub fn iris_equivalent(iri1: &str, iri2: &str) -> NormalizationResult<bool> {
    let normalized1 = normalize_iri(iri1)?;
    let normalized2 = normalize_iri(iri2)?;
    Ok(normalized1.is_equivalent(&normalized2))
}

/// Normalize an IRI and return as a Cow (avoids allocation if already normalized)
///
/// This is more efficient than `normalize_iri()` when the IRI is likely already normalized.
pub fn normalize_iri_cow(iri: &str) -> NormalizationResult<Cow<'_, str>> {
    let normalized = normalize_iri(iri)?;
    if normalized.as_str() == iri {
        Ok(Cow::Borrowed(iri))
    } else {
        Ok(Cow::Owned(normalized.into_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_normalization() {
        let iri = normalize_iri("HTTP://EXAMPLE.ORG/Path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/Path");
    }

    #[test]
    fn test_percent_encoding_normalization() {
        // Decode unreserved characters
        let iri = normalize_iri("http://example.org/%7Euser").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/~user");

        let iri = normalize_iri("http://example.org/%41%42%43").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/ABC");

        // Keep reserved characters encoded (but uppercase)
        let iri = normalize_iri("http://example.org/path%20with%20spaces").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/path%20with%20spaces");
    }

    #[test]
    fn test_default_port_removal() {
        let iri = normalize_iri("http://example.org:80/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/path");

        let iri = normalize_iri("https://example.org:443/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "https://example.org/path");

        // Non-default port should be kept
        let iri = normalize_iri("http://example.org:8080/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org:8080/path");
    }

    #[test]
    fn test_path_normalization() {
        let iri = normalize_iri("http://example.org/a/./b/../c").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/a/c");

        let iri = normalize_iri("http://example.org/./a/b").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/a/b");

        let iri = normalize_iri("http://example.org/a/b/..").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/a");
    }

    #[test]
    fn test_empty_path_normalization() {
        let iri = normalize_iri("http://example.org").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/");
    }

    #[test]
    fn test_query_and_fragment() {
        let iri = normalize_iri("http://example.org/path?query=value#fragment").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/path?query=value#fragment");

        // Percent-encoding in query and fragment
        let iri = normalize_iri("http://example.org/path?q=%41#%42").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://example.org/path?q=A#B");
    }

    #[test]
    fn test_ipv6_address() {
        let iri = normalize_iri("http://[2001:db8::1]/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://[2001:db8::1]/path");

        let iri = normalize_iri("http://[2001:DB8::1]:8080/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://[2001:db8::1]:8080/path");
    }

    #[test]
    fn test_userinfo() {
        let iri = normalize_iri("http://user:pass@example.org/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://user:pass@example.org/path");

        let iri = normalize_iri("http://%41%42%43@example.org/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "http://ABC@example.org/path");
    }

    #[test]
    fn test_iris_equivalent() {
        assert!(
            iris_equivalent("HTTP://EXAMPLE.ORG/path", "http://example.org/path")
                .expect("valid IRI")
        );

        assert!(
            iris_equivalent("http://example.org:80/path", "http://example.org/path")
                .expect("valid IRI")
        );

        assert!(
            iris_equivalent("http://example.org/a/./b/../c", "http://example.org/a/c")
                .expect("valid IRI")
        );

        assert!(
            !iris_equivalent("http://example.org/path1", "http://example.org/path2")
                .expect("valid IRI")
        );
    }

    #[test]
    fn test_complex_normalization() {
        let iri = normalize_iri("HTTP://USER@EXAMPLE.ORG:80/A/./B/../C/%7Euser?Q=%41#%42")
            .expect("valid IRI");
        assert_eq!(iri.as_str(), "http://USER@example.org/A/C/~user?Q=A#B");
    }

    #[test]
    fn test_non_http_schemes() {
        let iri = normalize_iri("ftp://example.org:21/path").expect("valid IRI");
        assert_eq!(iri.as_str(), "ftp://example.org/path");

        let iri = normalize_iri("urn:isbn:0451450523").expect("valid IRI");
        assert_eq!(iri.as_str(), "urn:isbn:0451450523");
    }

    #[test]
    fn test_invalid_iri() {
        assert!(normalize_iri("").is_err());
        assert!(normalize_iri("not an iri").is_err());
        assert!(normalize_iri("http://example.org/%ZZ").is_err());
    }

    #[test]
    fn test_normalized_iri_methods() {
        let iri1 = normalize_iri("http://example.org/path").expect("valid IRI");
        let iri2 = normalize_iri("HTTP://EXAMPLE.ORG/path").expect("valid IRI");

        assert_eq!(iri1.as_str(), "http://example.org/path");
        assert!(iri1.is_equivalent(&iri2));
        assert_eq!(iri1, iri2);

        let cloned = iri1.clone();
        assert_eq!(iri1, cloned);
    }

    #[test]
    fn test_normalize_iri_cow() {
        // Already normalized - should return Borrowed
        let iri = "http://example.org/path";
        let result = normalize_iri_cow(iri).expect("valid IRI");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, iri);

        // Not normalized - should return Owned
        let iri = "HTTP://EXAMPLE.ORG/path";
        let result = normalize_iri_cow(iri).expect("valid IRI");
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result, "http://example.org/path");
    }

    #[test]
    fn test_urn_normalization() {
        // URN scheme normalization
        let iri = normalize_iri("URN:ISBN:0451450523").expect("valid IRI");
        assert_eq!(iri.as_str(), "urn:ISBN:0451450523");
    }

    #[test]
    fn test_trailing_slash() {
        let iri1 = normalize_iri("http://example.org/path/").expect("valid IRI");
        let iri2 = normalize_iri("http://example.org/path").expect("valid IRI");

        // These should NOT be equivalent (trailing slash matters)
        assert_ne!(iri1, iri2);
        assert_eq!(iri1.as_str(), "http://example.org/path/");
        assert_eq!(iri2.as_str(), "http://example.org/path");
    }
}
