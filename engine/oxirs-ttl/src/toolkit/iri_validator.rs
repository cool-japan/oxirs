//! RFC 3987 IRI validation
//!
//! This module provides validation for Internationalized Resource Identifiers (IRIs)
//! according to RFC 3987. IRIs extend URIs to support Unicode characters.
//!
//! # RFC 3987 Compliance
//!
//! An IRI consists of:
//! - scheme (required)
//! - authority (optional) - userinfo@host:port
//! - path
//! - query (optional)
//! - fragment (optional)
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::toolkit::iri_validator::{validate_iri, IriValidationError};
//!
//! assert!(validate_iri("http://example.org/path").is_ok());
//! assert!(validate_iri("urn:isbn:0451450523").is_ok());
//! assert!(validate_iri("http://example.org/path#fragment").is_ok());
//! assert!(validate_iri("http://example.org/日本語").is_ok());
//!
//! // Invalid IRIs
//! assert!(validate_iri("not an iri").is_err());
//! assert!(validate_iri("http:// invalid").is_err());
//! ```

use std::fmt;

/// Error type for IRI validation failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IriValidationError {
    /// IRI is empty
    Empty,
    /// Missing scheme (e.g., "http:", "urn:")
    MissingScheme,
    /// Invalid scheme format
    InvalidScheme(String),
    /// Invalid authority component
    InvalidAuthority(String),
    /// Invalid host
    InvalidHost(String),
    /// Invalid port
    InvalidPort(String),
    /// Invalid path
    InvalidPath(String),
    /// Invalid query
    InvalidQuery(String),
    /// Invalid fragment
    InvalidFragment(String),
    /// Invalid character in IRI
    InvalidCharacter(char, usize),
    /// Invalid percent encoding
    InvalidPercentEncoding(String),
}

impl fmt::Display for IriValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "IRI cannot be empty"),
            Self::MissingScheme => write!(f, "IRI must have a scheme"),
            Self::InvalidScheme(s) => write!(f, "Invalid scheme: {}", s),
            Self::InvalidAuthority(s) => write!(f, "Invalid authority: {}", s),
            Self::InvalidHost(s) => write!(f, "Invalid host: {}", s),
            Self::InvalidPort(s) => write!(f, "Invalid port: {}", s),
            Self::InvalidPath(s) => write!(f, "Invalid path: {}", s),
            Self::InvalidQuery(s) => write!(f, "Invalid query: {}", s),
            Self::InvalidFragment(s) => write!(f, "Invalid fragment: {}", s),
            Self::InvalidCharacter(c, pos) => {
                write!(f, "Invalid character '{}' at position {}", c, pos)
            }
            Self::InvalidPercentEncoding(s) => write!(f, "Invalid percent encoding: {}", s),
        }
    }
}

impl std::error::Error for IriValidationError {}

/// Result type for IRI validation
pub type IriValidationResult<T> = Result<T, IriValidationError>;

/// Validates an IRI according to RFC 3987
///
/// Returns Ok(()) if the IRI is valid, or an error describing the validation failure.
pub fn validate_iri(iri: &str) -> IriValidationResult<()> {
    if iri.is_empty() {
        return Err(IriValidationError::Empty);
    }

    // Parse the IRI components
    let (scheme, rest) = parse_scheme(iri)?;
    validate_scheme(&scheme)?;

    // Check for authority (starts with //)
    let (authority, path_query_fragment) = if let Some(after_slashes) = rest.strip_prefix("//") {
        let auth_end = after_slashes
            .find(['/', '?', '#'])
            .unwrap_or(after_slashes.len());
        let authority = &after_slashes[..auth_end];
        validate_authority(authority)?;
        (Some(authority), &after_slashes[auth_end..])
    } else {
        (None, rest)
    };

    // Parse path, query, and fragment
    let (path, query_fragment) = if let Some(q_pos) = path_query_fragment.find('?') {
        (&path_query_fragment[..q_pos], &path_query_fragment[q_pos..])
    } else if let Some(f_pos) = path_query_fragment.find('#') {
        (&path_query_fragment[..f_pos], &path_query_fragment[f_pos..])
    } else {
        (path_query_fragment, "")
    };

    // Validate path
    validate_path(path, authority.is_some())?;

    // Parse and validate query and fragment
    if !query_fragment.is_empty() {
        if let Some(after_question) = query_fragment.strip_prefix('?') {
            let (query, fragment) = if let Some(f_pos) = after_question.find('#') {
                (&after_question[..f_pos], &after_question[f_pos..])
            } else {
                (after_question, "")
            };
            validate_query(query)?;
            if let Some(frag) = fragment.strip_prefix('#') {
                validate_fragment(frag)?;
            }
        } else if let Some(frag) = query_fragment.strip_prefix('#') {
            validate_fragment(frag)?;
        }
    }

    Ok(())
}

/// Validates an IRI reference (can be relative)
pub fn validate_iri_reference(iri_ref: &str) -> IriValidationResult<()> {
    if iri_ref.is_empty() {
        return Ok(()); // Empty reference is valid
    }

    // Check if it has a scheme
    if let Some(colon_pos) = iri_ref.find(':') {
        // Check if the part before : is a valid scheme
        let potential_scheme = &iri_ref[..colon_pos];
        if is_valid_scheme_chars(potential_scheme) {
            return validate_iri(iri_ref);
        }
    }

    // It's a relative reference - validate path, query, fragment
    let (path, query_fragment) = if let Some(after_slashes) = iri_ref.strip_prefix("//") {
        // Network-path reference
        let auth_end = after_slashes
            .find(['/', '?', '#'])
            .unwrap_or(after_slashes.len());
        let authority = &after_slashes[..auth_end];
        validate_authority(authority)?;
        (&after_slashes[auth_end..], "")
    } else if let Some(q_pos) = iri_ref.find('?') {
        (&iri_ref[..q_pos], &iri_ref[q_pos..])
    } else if let Some(f_pos) = iri_ref.find('#') {
        (&iri_ref[..f_pos], &iri_ref[f_pos..])
    } else {
        (iri_ref, "")
    };

    // Validate relative path
    if !path.is_empty() {
        validate_path(path, false)?;
    }

    // Validate query and fragment
    if !query_fragment.is_empty() {
        if let Some(after_question) = query_fragment.strip_prefix('?') {
            let (query, fragment) = if let Some(f_pos) = after_question.find('#') {
                (&after_question[..f_pos], &after_question[f_pos..])
            } else {
                (after_question, "")
            };
            validate_query(query)?;
            if let Some(frag) = fragment.strip_prefix('#') {
                validate_fragment(frag)?;
            }
        } else if let Some(frag) = query_fragment.strip_prefix('#') {
            validate_fragment(frag)?;
        }
    }

    Ok(())
}

/// Parse the scheme from an IRI
fn parse_scheme(iri: &str) -> IriValidationResult<(String, &str)> {
    let colon_pos = iri.find(':').ok_or(IriValidationError::MissingScheme)?;

    let scheme = &iri[..colon_pos];
    let rest = &iri[colon_pos + 1..];

    Ok((scheme.to_string(), rest))
}

/// Check if a string contains only valid scheme characters
fn is_valid_scheme_chars(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();

    // First character must be ASCII letter
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() {
        return false;
    }

    // Rest can be ASCII letter, digit, +, -, .
    chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

/// Validate the scheme component
fn validate_scheme(scheme: &str) -> IriValidationResult<()> {
    if !is_valid_scheme_chars(scheme) {
        return Err(IriValidationError::InvalidScheme(scheme.to_string()));
    }
    Ok(())
}

/// Validate the authority component
fn validate_authority(authority: &str) -> IriValidationResult<()> {
    if authority.is_empty() {
        return Ok(());
    }

    // Split userinfo@host:port
    let (userinfo, host_port) = if let Some(at_pos) = authority.rfind('@') {
        let userinfo = &authority[..at_pos];
        validate_userinfo(userinfo)?;
        (Some(userinfo), &authority[at_pos + 1..])
    } else {
        (None, authority)
    };

    // Split host:port
    let (host, port) = if let Some(_bracket_start) = host_port.find('[') {
        // IPv6 address
        let bracket_end = host_port
            .find(']')
            .ok_or_else(|| IriValidationError::InvalidHost("Unclosed IPv6 bracket".to_string()))?;
        let host = &host_port[..bracket_end + 1];
        let rest = &host_port[bracket_end + 1..];
        let port = if let Some(port_str) = rest.strip_prefix(':') {
            Some(port_str)
        } else if rest.is_empty() {
            None
        } else {
            return Err(IriValidationError::InvalidHost(host_port.to_string()));
        };
        (host, port)
    } else if let Some(colon_pos) = host_port.rfind(':') {
        let potential_port = &host_port[colon_pos + 1..];
        // Only treat as port if it's all digits
        if potential_port.chars().all(|c| c.is_ascii_digit()) {
            (&host_port[..colon_pos], Some(potential_port))
        } else {
            (host_port, None)
        }
    } else {
        (host_port, None)
    };

    validate_host(host)?;

    if let Some(port) = port {
        validate_port(port)?;
    }

    let _ = userinfo; // Suppress unused warning

    Ok(())
}

/// Validate userinfo component
fn validate_userinfo(userinfo: &str) -> IriValidationResult<()> {
    for (i, c) in userinfo.chars().enumerate() {
        if !is_valid_userinfo_char(c) {
            return Err(IriValidationError::InvalidCharacter(c, i));
        }
    }
    Ok(())
}

/// Validate host component
fn validate_host(host: &str) -> IriValidationResult<()> {
    if host.is_empty() {
        return Ok(());
    }

    // IPv6 address
    if host.starts_with('[') && host.ends_with(']') {
        return validate_ipv6(&host[1..host.len() - 1]);
    }

    // Reg-name or IPv4
    for (i, c) in host.chars().enumerate() {
        if !is_valid_host_char(c) {
            return Err(IriValidationError::InvalidCharacter(c, i));
        }
    }

    Ok(())
}

/// Validate IPv6 address
fn validate_ipv6(addr: &str) -> IriValidationResult<()> {
    // Basic IPv6 validation
    let parts: Vec<&str> = addr.split(':').collect();
    if parts.len() > 8 {
        return Err(IriValidationError::InvalidHost(format!(
            "IPv6 address has too many parts: {}",
            addr
        )));
    }

    for part in parts {
        if !part.is_empty() && !part.chars().all(|c| c.is_ascii_hexdigit()) {
            // Allow for IPv4-mapped addresses
            if !part.contains('.') {
                return Err(IriValidationError::InvalidHost(format!(
                    "Invalid IPv6 part: {}",
                    part
                )));
            }
        }
    }

    Ok(())
}

/// Validate port component
fn validate_port(port: &str) -> IriValidationResult<()> {
    if port.is_empty() {
        return Ok(());
    }

    if !port.chars().all(|c| c.is_ascii_digit()) {
        return Err(IriValidationError::InvalidPort(port.to_string()));
    }

    Ok(())
}

/// Validate path component
fn validate_path(path: &str, has_authority: bool) -> IriValidationResult<()> {
    if path.is_empty() {
        return Ok(());
    }

    // Path validation depends on whether authority is present
    if has_authority && !path.is_empty() && !path.starts_with('/') {
        return Err(IriValidationError::InvalidPath(
            "Path must start with / when authority is present".to_string(),
        ));
    }

    // Check each character
    let mut chars = path.chars().enumerate().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '%' {
            // Validate percent encoding
            let hex1 = chars.next();
            let hex2 = chars.next();
            match (hex1, hex2) {
                (Some((_, h1)), Some((_, h2)))
                    if h1.is_ascii_hexdigit() && h2.is_ascii_hexdigit() => {}
                _ => {
                    return Err(IriValidationError::InvalidPercentEncoding(format!(
                        "Invalid percent encoding at position {}",
                        i
                    )));
                }
            }
        } else if !is_valid_path_char(c) {
            return Err(IriValidationError::InvalidCharacter(c, i));
        }
    }

    Ok(())
}

/// Validate query component
fn validate_query(query: &str) -> IriValidationResult<()> {
    for (i, c) in query.chars().enumerate() {
        if c == '%' {
            // Already validated in the main loop, skip
            continue;
        }
        if !is_valid_query_char(c) {
            return Err(IriValidationError::InvalidCharacter(c, i));
        }
    }
    Ok(())
}

/// Validate fragment component
fn validate_fragment(fragment: &str) -> IriValidationResult<()> {
    for (i, c) in fragment.chars().enumerate() {
        if c == '%' {
            // Already validated in the main loop, skip
            continue;
        }
        if !is_valid_fragment_char(c) {
            return Err(IriValidationError::InvalidCharacter(c, i));
        }
    }
    Ok(())
}

/// Check if character is valid for userinfo
fn is_valid_userinfo_char(c: char) -> bool {
    is_unreserved(c) || is_sub_delim(c) || c == ':' || c == '%'
}

/// Check if character is valid for host (reg-name)
fn is_valid_host_char(c: char) -> bool {
    is_unreserved(c) || is_sub_delim(c) || c == '%'
}

/// Check if character is valid for path
fn is_valid_path_char(c: char) -> bool {
    is_pchar(c) || c == '/'
}

/// Check if character is valid for query
fn is_valid_query_char(c: char) -> bool {
    is_pchar(c) || is_iprivate(c) || c == '/' || c == '?'
}

/// Check if character is valid for fragment
fn is_valid_fragment_char(c: char) -> bool {
    is_pchar(c) || c == '/' || c == '?'
}

/// RFC 3987 pchar
fn is_pchar(c: char) -> bool {
    is_unreserved(c) || is_sub_delim(c) || c == ':' || c == '@' || c == '%'
}

/// RFC 3986 unreserved characters plus RFC 3987 ucschar
fn is_unreserved(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' || c == '~' || is_ucschar(c)
}

/// RFC 3987 ucschar (Unicode characters allowed in IRIs)
fn is_ucschar(c: char) -> bool {
    let cp = c as u32;
    (0xA0..=0xD7FF).contains(&cp)
        || (0xF900..=0xFDCF).contains(&cp)
        || (0xFDF0..=0xFFEF).contains(&cp)
        || (0x10000..=0x1FFFD).contains(&cp)
        || (0x20000..=0x2FFFD).contains(&cp)
        || (0x30000..=0x3FFFD).contains(&cp)
        || (0x40000..=0x4FFFD).contains(&cp)
        || (0x50000..=0x5FFFD).contains(&cp)
        || (0x60000..=0x6FFFD).contains(&cp)
        || (0x70000..=0x7FFFD).contains(&cp)
        || (0x80000..=0x8FFFD).contains(&cp)
        || (0x90000..=0x9FFFD).contains(&cp)
        || (0xA0000..=0xAFFFD).contains(&cp)
        || (0xB0000..=0xBFFFD).contains(&cp)
        || (0xC0000..=0xCFFFD).contains(&cp)
        || (0xD0000..=0xDFFFD).contains(&cp)
        || (0xE1000..=0xEFFFD).contains(&cp)
}

/// RFC 3987 iprivate (private use characters allowed in query)
fn is_iprivate(c: char) -> bool {
    let cp = c as u32;
    (0xE000..=0xF8FF).contains(&cp)
        || (0xF0000..=0xFFFFD).contains(&cp)
        || (0x100000..=0x10FFFD).contains(&cp)
}

/// RFC 3986 sub-delims
fn is_sub_delim(c: char) -> bool {
    matches!(
        c,
        '!' | '$' | '&' | '\'' | '(' | ')' | '*' | '+' | ',' | ';' | '='
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_iris() {
        assert!(validate_iri("http://example.org").is_ok());
        assert!(validate_iri("http://example.org/").is_ok());
        assert!(validate_iri("http://example.org/path").is_ok());
        assert!(validate_iri("http://example.org/path?query").is_ok());
        assert!(validate_iri("http://example.org/path#fragment").is_ok());
        assert!(validate_iri("http://example.org/path?query#fragment").is_ok());
        assert!(validate_iri("https://example.org:8080/path").is_ok());
        assert!(validate_iri("ftp://user:pass@example.org/file").is_ok());
        assert!(validate_iri("urn:isbn:0451450523").is_ok());
        assert!(validate_iri("mailto:user@example.org").is_ok());
        assert!(validate_iri("file:///path/to/file").is_ok());
    }

    #[test]
    fn test_unicode_iris() {
        assert!(validate_iri("http://example.org/日本語").is_ok());
        assert!(validate_iri("http://example.org/中文").is_ok());
        assert!(validate_iri("http://example.org/العربية").is_ok());
        assert!(validate_iri("http://example.org/한국어").is_ok());
        assert!(validate_iri("http://example.org/path?q=日本語").is_ok());
    }

    #[test]
    fn test_ipv6_iris() {
        assert!(validate_iri("http://[::1]/").is_ok());
        assert!(validate_iri("http://[2001:db8::1]:8080/").is_ok());
        assert!(validate_iri("http://[::ffff:192.168.1.1]/").is_ok());
    }

    #[test]
    fn test_invalid_iris() {
        assert!(validate_iri("").is_err());
        assert!(validate_iri("not-an-iri").is_err());
        assert!(validate_iri("://missing-scheme").is_err());
        assert!(validate_iri("1http://invalid-scheme").is_err());
        assert!(validate_iri("http:// invalid").is_err());
        assert!(validate_iri("http://example.org/path with spaces").is_err());
    }

    #[test]
    fn test_iri_references() {
        assert!(validate_iri_reference("").is_ok());
        assert!(validate_iri_reference("/relative/path").is_ok());
        assert!(validate_iri_reference("relative/path").is_ok());
        assert!(validate_iri_reference("?query").is_ok());
        assert!(validate_iri_reference("#fragment").is_ok());
        assert!(validate_iri_reference("//authority/path").is_ok());
    }

    #[test]
    fn test_percent_encoding() {
        assert!(validate_iri("http://example.org/path%20with%20spaces").is_ok());
        assert!(validate_iri("http://example.org/path%2F").is_ok());
        assert!(validate_iri("http://example.org/path%").is_err());
        assert!(validate_iri("http://example.org/path%G0").is_err());
    }

    #[test]
    fn test_complex_iris() {
        assert!(validate_iri(
            "http://user:password@example.org:8080/path/to/resource?key=value&foo=bar#section"
        )
        .is_ok());
        assert!(validate_iri("http://example.org/~user/file.html").is_ok());
        assert!(validate_iri("http://example.org/a/b/c/d/e/f/g/h/i/j/k").is_ok());
    }

    #[test]
    fn test_rdf_common_schemes() {
        // Common RDF schemes
        assert!(validate_iri("http://www.w3.org/2001/XMLSchema#string").is_ok());
        assert!(validate_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").is_ok());
        assert!(validate_iri("http://www.w3.org/2000/01/rdf-schema#label").is_ok());
        assert!(validate_iri("http://purl.org/dc/terms/title").is_ok());
        assert!(validate_iri("http://xmlns.com/foaf/0.1/name").is_ok());
    }

    #[test]
    fn test_error_messages() {
        let err = validate_iri("").unwrap_err();
        assert_eq!(err.to_string(), "IRI cannot be empty");

        let err = validate_iri("no-scheme").unwrap_err();
        assert!(err.to_string().contains("scheme"));
    }
}
