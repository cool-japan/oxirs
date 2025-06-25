//! IRI and Named Node implementations for RDF

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use regex::Regex;
use lazy_static::lazy_static;
use crate::model::{RdfTerm, SubjectTerm, PredicateTerm, ObjectTerm, GraphNameTerm};
use crate::OxirsError;

lazy_static! {
    /// RFC 3987 IRI validation regex (simplified but comprehensive)
    /// Based on the ABNF grammar from RFC 3987
    static ref IRI_REGEX: Regex = Regex::new(
        r"^([a-zA-Z][a-zA-Z0-9+.-]*):([/?#]([^\s<>{}|\\^`]*))?$"
    ).expect("IRI regex compilation failed");
    
    /// More relaxed IRI regex for relative IRIs
    static ref RELATIVE_IRI_REGEX: Regex = Regex::new(
        r"^[^\s<>{}|\\^`]*$"
    ).expect("Relative IRI regex compilation failed");
    
    /// Regex for detecting percent-encoded sequences
    static ref PERCENT_ENCODED_REGEX: Regex = Regex::new(
        r"%[0-9A-Fa-f]{2}"
    ).expect("Percent encoding regex compilation failed");
    
    /// Regex for scheme validation (more strict)
    static ref SCHEME_REGEX: Regex = Regex::new(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*$"
    ).expect("Scheme regex compilation failed");
}

/// Validates an IRI according to RFC 3987
fn validate_iri(iri: &str) -> Result<(), OxirsError> {
    if iri.is_empty() {
        return Err(OxirsError::Parse("IRI cannot be empty".to_string()));
    }
    
    // Check for forbidden characters first
    if iri.chars().any(|c| {
        matches!(c, 
            '\u{0000}'..='\u{001F}' |  // Control characters
            '\u{007F}'..='\u{009F}' |  // More control characters
            '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`' | ' '  // Reserved characters
        )
    }) {
        return Err(OxirsError::Parse(
            "IRI contains forbidden characters".to_string()
        ));
    }
    
    // Check if it's an absolute IRI
    if iri.contains(':') {
        let parts: Vec<&str> = iri.splitn(2, ':').collect();
        if parts.len() == 2 {
            let scheme = parts[0];
            let hier_part = parts[1];
            
            // Validate scheme
            if !SCHEME_REGEX.is_match(scheme) {
                return Err(OxirsError::Parse(format!(
                    "Invalid IRI scheme: '{}'. Must start with letter and contain only letters, digits, +, -, .",
                    scheme
                )));
            }
            
            // Basic validation of hierarchical part
            if !validate_hier_part(hier_part) {
                return Err(OxirsError::Parse(
                    "Invalid IRI hierarchical part".to_string()
                ));
            }
        }
    } else if !RELATIVE_IRI_REGEX.is_match(iri) {
        return Err(OxirsError::Parse(
            "Invalid relative IRI format".to_string()
        ));
    }
    
    // Validate percent encoding
    validate_percent_encoding(iri)?;
    
    Ok(())
}

/// Validates the hierarchical part of an IRI
fn validate_hier_part(hier_part: &str) -> bool {
    // Very basic validation - in a full implementation this would be much more comprehensive
    // For now, just check for obviously invalid patterns
    
    if hier_part.contains("//") {
        // Authority-based IRI
        let rest = &hier_part[2..];
        // Should contain valid authority followed by path
        !rest.is_empty()
    } else {
        // Path-only IRI
        true
    }
}

/// Validates percent encoding in an IRI
fn validate_percent_encoding(iri: &str) -> Result<(), OxirsError> {
    let mut chars = iri.chars().peekable();
    
    while let Some(c) = chars.next() {
        if c == '%' {
            // Check that we have exactly two hex digits following
            let hex1 = chars.next().ok_or_else(|| {
                OxirsError::Parse("Incomplete percent encoding - missing first hex digit".to_string())
            })?;
            
            let hex2 = chars.next().ok_or_else(|| {
                OxirsError::Parse("Incomplete percent encoding - missing second hex digit".to_string())
            })?;
            
            if !hex1.is_ascii_hexdigit() || !hex2.is_ascii_hexdigit() {
                return Err(OxirsError::Parse(format!(
                    "Invalid percent encoding - non-hex digits: %{}{}", 
                    hex1, hex2
                )));
            }
        }
    }
    
    Ok(())
}

/// Normalizes an IRI according to RFC 3987 recommendations
pub fn normalize_iri(iri: &str) -> String {
    let mut normalized = iri.to_string();
    
    // 1. Convert scheme and host to lowercase
    if let Some(colon_pos) = normalized.find(':') {
        let (scheme, rest) = normalized.split_at(colon_pos);
        let scheme_lower = scheme.to_lowercase();
        
        // Also lowercase the host if it's an authority-based IRI
        if rest.starts_with("://") {
            if let Some(slash_pos) = rest[3..].find('/') {
                let authority = &rest[3..3 + slash_pos];
                let path = &rest[3 + slash_pos..];
                normalized = format!("{}://{}{}", scheme_lower, authority.to_lowercase(), path);
            } else {
                // No path, just authority
                normalized = format!("{}://{}", scheme_lower, rest[3..].to_lowercase());
            }
        } else {
            normalized = format!("{}{}", scheme_lower, rest);
        }
    }
    
    // 2. Normalize percent encoding - convert to uppercase
    normalized = PERCENT_ENCODED_REGEX.replace_all(&normalized, |caps: &regex::Captures| {
        caps[0].to_uppercase()
    }).to_string();
    
    // 3. Remove default port numbers (basic implementation)
    normalized = normalized
        .replace(":80/", "/")  // HTTP default port
        .replace(":443/", "/") // HTTPS default port
        .replace(":21/", "/"); // FTP default port
    
    // 4. Normalize path (remove unnecessary dots, etc.)
    // This is a simplified version - full normalization is complex
    normalized = normalized.replace("/./", "/");
    while normalized.contains("/../") {
        if let Some(pos) = normalized.find("/../") {
            // Find the previous slash
            if let Some(prev_slash) = normalized[..pos].rfind('/') {
                let before = &normalized[..prev_slash];
                let after = &normalized[pos + 4..]; // Skip the leading slash too
                normalized = format!("{}/{}", before, after);
            } else {
                break; // Can't normalize further
            }
        }
    }
    
    normalized
}

/// An RDF Named Node (IRI)
/// 
/// Represents an IRI (Internationalized Resource Identifier) reference.
/// This is one of the core RDF term types, compliant with RFC 3987.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct NamedNode {
    iri: String,
}

impl NamedNode {
    /// Creates a new named node from an IRI string with RFC 3987 validation
    /// 
    /// # Arguments
    /// * `iri` - The IRI string
    /// 
    /// # Errors
    /// Returns an error if the IRI is invalid according to RFC 3987
    pub fn new(iri: impl Into<String>) -> Result<Self, OxirsError> {
        let iri = iri.into();
        validate_iri(&iri)?;
        Ok(NamedNode { iri })
    }
    
    /// Creates a new named node from an IRI string with normalization
    /// 
    /// This applies RFC 3987 normalization rules to the IRI before validation.
    pub fn new_normalized(iri: impl Into<String>) -> Result<Self, OxirsError> {
        let iri = normalize_iri(&iri.into());
        validate_iri(&iri)?;
        Ok(NamedNode { iri })
    }
    
    /// Creates a new named node from an IRI string without validation
    /// 
    /// # Safety
    /// The caller must ensure the IRI is valid
    pub fn new_unchecked(iri: impl Into<String>) -> Self {
        NamedNode { iri: iri.into() }
    }
    
    /// Returns the IRI as a string slice
    pub fn as_str(&self) -> &str {
        &self.iri
    }
    
    /// Returns the IRI as a string
    pub fn into_string(self) -> String {
        self.iri
    }
    
    /// Returns a reference to this NamedNode as a NamedNodeRef
    pub fn as_ref(&self) -> NamedNodeRef<'_> {
        NamedNodeRef::new_unchecked(&self.iri)
    }
}

impl FromStr for NamedNode {
    type Err = OxirsError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl fmt::Display for NamedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.iri)
    }
}

impl RdfTerm for NamedNode {
    fn as_str(&self) -> &str {
        &self.iri
    }
    
    fn is_named_node(&self) -> bool {
        true
    }
}

impl SubjectTerm for NamedNode {}
impl PredicateTerm for NamedNode {}
impl ObjectTerm for NamedNode {}
impl GraphNameTerm for NamedNode {}


/// A borrowed named node
/// 
/// This is an optimized version for temporary references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamedNodeRef<'a> {
    iri: &'a str,
}

impl<'a> NamedNodeRef<'a> {
    /// Creates a new named node reference
    pub fn new(iri: &'a str) -> Result<Self, OxirsError> {
        if iri.is_empty() {
            return Err(OxirsError::Parse("IRI cannot be empty".to_string()));
        }
        
        Ok(NamedNodeRef { iri })
    }
    
    /// Creates a new named node reference without validation
    pub fn new_unchecked(iri: &'a str) -> Self {
        NamedNodeRef { iri }
    }
    
    /// Returns the IRI as a string slice
    pub fn as_str(&self) -> &str {
        self.iri
    }
    
    /// Converts to an owned NamedNode
    pub fn to_owned(&self) -> NamedNode {
        NamedNode::new_unchecked(self.iri.to_string())
    }
}

impl<'a> fmt::Display for NamedNodeRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.iri)
    }
}

impl<'a> RdfTerm for NamedNodeRef<'a> {
    fn as_str(&self) -> &str {
        self.iri
    }
    
    fn is_named_node(&self) -> bool {
        true
    }
}

impl<'a> From<NamedNodeRef<'a>> for NamedNode {
    fn from(node_ref: NamedNodeRef<'a>) -> Self {
        node_ref.to_owned()
    }
}

impl<'a> From<&'a NamedNode> for NamedNodeRef<'a> {
    fn from(node: &'a NamedNode) -> Self {
        NamedNodeRef::new_unchecked(node.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_named_node_creation() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        assert_eq!(node.as_str(), "http://example.org/test");
        assert!(node.is_named_node());
    }
    
    #[test]
    fn test_named_node_display() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        assert_eq!(format!("{}", node), "<http://example.org/test>");
    }
    
    #[test]
    fn test_empty_iri_error() {
        assert!(NamedNode::new("").is_err());
    }
    
    #[test]
    fn test_named_node_ref() {
        let node_ref = NamedNodeRef::new("http://example.org/test").unwrap();
        assert_eq!(node_ref.as_str(), "http://example.org/test");
        
        let owned = node_ref.to_owned();
        assert_eq!(owned.as_str(), "http://example.org/test");
    }
    
    #[test]
    fn test_normalize_iri() {
        // Test scheme lowercasing
        assert_eq!(normalize_iri("HTTP://EXAMPLE.ORG/"), "http://example.org/");
        assert_eq!(normalize_iri("HTTPS://Test.COM/"), "https://test.com/");
        
        // Test percent encoding normalization
        assert_eq!(normalize_iri("http://example.org/%2f"), "http://example.org/%2F");
        assert_eq!(normalize_iri("http://example.org/%2F"), "http://example.org/%2F");
        
        // Test path normalization
        assert_eq!(normalize_iri("http://example.org/./path"), "http://example.org/path");
        assert_eq!(normalize_iri("http://example.org/a/../b"), "http://example.org/b");
        
        // Test default port removal
        assert_eq!(normalize_iri("http://example.org:80/path"), "http://example.org/path");
        assert_eq!(normalize_iri("https://example.org:443/path"), "https://example.org/path");
    }
    
    #[test]
    fn test_named_node_normalized() {
        let node = NamedNode::new_normalized("HTTP://EXAMPLE.ORG/").unwrap();
        assert_eq!(node.as_str(), "http://example.org/");
    }
}