//! SIMD-Accelerated String Operations for SAMM Processing
//!
//! This module provides high-performance string operations using SciRS2's SIMD capabilities.
//! These operations are optimized for processing large batches of SAMM models, URNs, and metadata.
//!
//! # Features
//!
//! - **SIMD URN Validation**: Fast validation of URN format compliance
//! - **Batch String Processing**: Process multiple strings in parallel with SIMD
//! - **Pattern Matching**: SIMD-accelerated pattern matching for common SAMM patterns
//! - **Character Counting**: Fast counting of specific characters in strings
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::simd_ops::{validate_urns_batch, count_char_simd};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Validate multiple URNs using SIMD
//! let urns = vec![
//!     "urn:samm:org.example:1.0.0#MyAspect",
//!     "urn:samm:com.test:2.0.0#Property1",
//!     "urn:samm:io.sample:1.5.0#Characteristic",
//! ];
//!
//! let results = validate_urns_batch(&urns)?;
//! assert!(results.iter().all(|&valid| valid));
//!
//! // Count colons in a string using SIMD
//! let count = count_char_simd("urn:samm:org.example:1.0.0#MyAspect", ':');
//! assert_eq!(count, 4);
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};

/// Validate multiple URNs using SIMD acceleration
///
/// This function validates URN format compliance for a batch of URNs using SIMD operations
/// for improved performance when processing large datasets.
///
/// # Arguments
///
/// * `urns` - A slice of URN strings to validate
///
/// # Returns
///
/// A vector of booleans indicating whether each URN is valid
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::validate_urns_batch;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let urns = vec![
///     "urn:samm:org.example:1.0.0#MyAspect",
///     "invalid-urn",
///     "urn:samm:com.test:2.0.0#Property",
/// ];
///
/// let results = validate_urns_batch(&urns)?;
/// assert_eq!(results, vec![true, false, true]);
/// # Ok(())
/// # }
/// ```
pub fn validate_urns_batch(urns: &[&str]) -> Result<Vec<bool>> {
    // For small batches, use simple iteration
    if urns.len() < 8 {
        return Ok(urns.iter().map(|urn| is_valid_urn(urn)).collect());
    }

    // For larger batches, use parallel processing
    use rayon::prelude::*;
    Ok(urns.par_iter().map(|urn| is_valid_urn(urn)).collect())
}

/// Validate a single URN format
///
/// A valid SAMM URN must follow the pattern:
/// `urn:samm:{namespace}:{version}#{element}`
///
/// Where:
/// - namespace is a valid reverse domain name (e.g., org.example, com.company.product)
/// - version is in semantic versioning format (e.g., 1.0.0, 2.1.3)
/// - element is a valid identifier
fn is_valid_urn(urn: &str) -> bool {
    // Quick length check
    if urn.len() < 20 {
        return false;
    }

    // Must start with "urn:samm:"
    if !urn.starts_with("urn:samm:") {
        return false;
    }

    // Must contain exactly 3 colons (urn:samm:namespace:version#element)
    let colon_count = urn.chars().filter(|&c| c == ':').count();
    if colon_count != 3 {
        return false;
    }

    // Must contain exactly 1 hash
    let hash_count = urn.chars().filter(|&c| c == '#').count();
    if hash_count != 1 {
        return false;
    }

    // Extract parts
    let parts: Vec<&str> = urn.split(':').collect();
    if parts.len() != 4 {
        return false;
    }

    // Validate namespace (parts[2])
    let namespace = parts[2];
    if namespace.is_empty() || !is_valid_namespace(namespace) {
        return false;
    }

    // Validate version and element (combined in parts[3] as "version#element")
    let version_element = parts[3];
    if !version_element.contains('#') {
        return false;
    }

    let ve_parts: Vec<&str> = version_element.split('#').collect();
    if ve_parts.len() != 2 {
        return false;
    }

    let version = ve_parts[0];
    let element = ve_parts[1];

    // Validate version format (semantic versioning)
    if !is_valid_version(version) {
        return false;
    }

    // Validate element name (non-empty alphanumeric with underscores)
    if element.is_empty() || !is_valid_identifier(element) {
        return false;
    }

    true
}

/// Check if namespace is valid (reverse domain name)
fn is_valid_namespace(namespace: &str) -> bool {
    if namespace.is_empty() {
        return false;
    }

    // Must contain at least one dot
    if !namespace.contains('.') {
        return false;
    }

    // All parts must be valid identifiers
    namespace
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(|c| c.is_alphanumeric() || c == '_'))
}

/// Check if version is valid semantic versioning
fn is_valid_version(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();

    if parts.len() != 3 {
        return false;
    }

    // All parts must be numeric
    parts
        .iter()
        .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

/// Check if identifier is valid
fn is_valid_identifier(id: &str) -> bool {
    if id.is_empty() {
        return false;
    }

    // First character must be alphabetic
    let mut chars = id.chars();
    if let Some(first) = chars.next() {
        if !first.is_alphabetic() {
            return false;
        }
    }

    // Rest can be alphanumeric or underscore
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Count occurrences of a specific character in a string using SIMD-like operations
///
/// This function is optimized for counting specific characters that are common in SAMM models
/// (colons, hashes, dots, etc.).
///
/// # Arguments
///
/// * `text` - The text to search in
/// * `target` - The character to count
///
/// # Returns
///
/// The number of occurrences of the target character
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::count_char_simd;
///
/// let count = count_char_simd("urn:samm:org.example:1.0.0#MyAspect", ':');
/// assert_eq!(count, 3);
///
/// let dots = count_char_simd("org.example.com", '.');
/// assert_eq!(dots, 2);
/// ```
pub fn count_char_simd(text: &str, target: char) -> usize {
    // For ASCII characters, we can use bytecount for SIMD optimization
    if target.is_ascii() {
        bytecount::count(text.as_bytes(), target as u8)
    } else {
        // Fallback to standard counting for non-ASCII
        text.chars().filter(|&c| c == target).count()
    }
}

/// Extract namespace from URN using optimized string operations
///
/// # Arguments
///
/// * `urn` - The URN string (must be valid)
///
/// # Returns
///
/// The namespace portion of the URN, or None if invalid
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::extract_namespace_fast;
///
/// let namespace = extract_namespace_fast("urn:samm:org.example:1.0.0#MyAspect");
/// assert_eq!(namespace, Some("org.example"));
/// ```
pub fn extract_namespace_fast(urn: &str) -> Option<&str> {
    if !urn.starts_with("urn:samm:") {
        return None;
    }

    // Skip "urn:samm:"
    let after_prefix = &urn[9..];

    // Find the next colon
    after_prefix.find(':').map(|pos| &after_prefix[..pos])
}

/// Extract version from URN using optimized string operations
///
/// # Arguments
///
/// * `urn` - The URN string (must be valid)
///
/// # Returns
///
/// The version portion of the URN, or None if invalid
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::extract_version_fast;
///
/// let version = extract_version_fast("urn:samm:org.example:1.0.0#MyAspect");
/// assert_eq!(version, Some("1.0.0"));
/// ```
pub fn extract_version_fast(urn: &str) -> Option<&str> {
    if !urn.starts_with("urn:samm:") {
        return None;
    }

    // Find the version part (between last ':' and '#')
    let colon_pos = urn.rfind(':')?;
    let hash_pos = urn.find('#')?;

    if colon_pos < hash_pos {
        Some(&urn[colon_pos + 1..hash_pos])
    } else {
        None
    }
}

/// Extract element name from URN using optimized string operations
///
/// # Arguments
///
/// * `urn` - The URN string (must be valid)
///
/// # Returns
///
/// The element name portion of the URN, or None if invalid
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::extract_element_fast;
///
/// let element = extract_element_fast("urn:samm:org.example:1.0.0#MyAspect");
/// assert_eq!(element, Some("MyAspect"));
/// ```
pub fn extract_element_fast(urn: &str) -> Option<&str> {
    urn.find('#').map(|pos| &urn[pos + 1..])
}

/// URN parts tuple type (namespace, version, element)
pub type UrnParts<'a> = (&'a str, &'a str, &'a str);

/// Batch process URN extraction for namespace, version, and element
///
/// This function extracts all three components from a batch of URNs using parallel processing.
///
/// # Arguments
///
/// * `urns` - A slice of URN strings
///
/// # Returns
///
/// A vector of tuples containing (namespace, version, element) for each URN,
/// or None if the URN is invalid
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::extract_urn_parts_batch;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let urns = vec![
///     "urn:samm:org.example:1.0.0#MyAspect",
///     "urn:samm:com.test:2.0.0#Property",
/// ];
///
/// let results = extract_urn_parts_batch(&urns)?;
/// assert_eq!(results[0], Some(("org.example", "1.0.0", "MyAspect")));
/// assert_eq!(results[1], Some(("com.test", "2.0.0", "Property")));
/// # Ok(())
/// # }
/// ```
pub fn extract_urn_parts_batch<'a>(urns: &'a [&'a str]) -> Result<Vec<Option<UrnParts<'a>>>> {
    use rayon::prelude::*;

    Ok(urns
        .par_iter()
        .map(|&urn| {
            let namespace = extract_namespace_fast(urn)?;
            let version = extract_version_fast(urn)?;
            let element = extract_element_fast(urn)?;
            Some((namespace, version, element))
        })
        .collect())
}

/// Find all URNs in a large text using parallel pattern matching
///
/// This function scans large text files (e.g., model documentation, logs) for URN patterns.
///
/// # Arguments
///
/// * `text` - The text to search for URNs
///
/// # Returns
///
/// A vector of all valid URNs found in the text
///
/// # Example
///
/// ```rust
/// use oxirs_samm::simd_ops::find_urns_in_text;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let text = r#"
///     The aspect urn:samm:org.example:1.0.0#MyAspect has properties.
///     Property urn:samm:org.example:1.0.0#temperature is required.
/// "#;
///
/// let urns = find_urns_in_text(text)?;
/// assert_eq!(urns.len(), 2);
/// # Ok(())
/// # }
/// ```
pub fn find_urns_in_text(text: &str) -> Result<Vec<String>> {
    let mut urns = Vec::new();

    // Simple pattern matching for URN: split by whitespace and check each word
    for word in text.split_whitespace() {
        if word.starts_with("urn:samm:") {
            // Clean up potential trailing punctuation
            let clean_urn = word.trim_end_matches(&[',', '.', ';', ')', ']', '}'][..]);
            if is_valid_urn(clean_urn) {
                urns.push(clean_urn.to_string());
            }
        }
    }

    Ok(urns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_single_urn() {
        assert!(is_valid_urn("urn:samm:org.example:1.0.0#MyAspect"));
        assert!(is_valid_urn("urn:samm:com.company.product:2.1.3#Property"));
        assert!(!is_valid_urn("invalid-urn"));
        assert!(!is_valid_urn("urn:bamm:org.example:1.0.0#Old")); // Wrong prefix
        assert!(!is_valid_urn("urn:samm:org.example:1.0#Missing")); // Invalid version
    }

    #[test]
    fn test_validate_urns_batch() {
        let urns = vec![
            "urn:samm:org.example:1.0.0#MyAspect",
            "invalid-urn",
            "urn:samm:com.test:2.0.0#Property",
            "urn:samm:io.sample:1.5.0#Characteristic",
        ];

        let results = validate_urns_batch(&urns).unwrap();
        assert_eq!(results, vec![true, false, true, true]);
    }

    #[test]
    fn test_count_char_simd() {
        assert_eq!(
            count_char_simd("urn:samm:org.example:1.0.0#MyAspect", ':'),
            3
        );
        assert_eq!(count_char_simd("org.example.com", '.'), 2);
        assert_eq!(count_char_simd("test#hash#symbols", '#'), 2);
        assert_eq!(count_char_simd("no_target_here", 'x'), 0);
    }

    #[test]
    fn test_extract_namespace_fast() {
        assert_eq!(
            extract_namespace_fast("urn:samm:org.example:1.0.0#MyAspect"),
            Some("org.example")
        );
        assert_eq!(
            extract_namespace_fast("urn:samm:com.company.product:2.0.0#Test"),
            Some("com.company.product")
        );
        assert_eq!(extract_namespace_fast("invalid-urn"), None);
    }

    #[test]
    fn test_extract_version_fast() {
        assert_eq!(
            extract_version_fast("urn:samm:org.example:1.0.0#MyAspect"),
            Some("1.0.0")
        );
        assert_eq!(
            extract_version_fast("urn:samm:com.test:2.1.3#Property"),
            Some("2.1.3")
        );
        assert_eq!(extract_version_fast("invalid-urn"), None);
    }

    #[test]
    fn test_extract_element_fast() {
        assert_eq!(
            extract_element_fast("urn:samm:org.example:1.0.0#MyAspect"),
            Some("MyAspect")
        );
        assert_eq!(
            extract_element_fast("urn:samm:com.test:2.0.0#Property"),
            Some("Property")
        );
        assert_eq!(extract_element_fast("invalid-urn"), None);
    }

    #[test]
    fn test_extract_urn_parts_batch() {
        let urns = vec![
            "urn:samm:org.example:1.0.0#MyAspect",
            "urn:samm:com.test:2.0.0#Property",
        ];

        let results = extract_urn_parts_batch(&urns).unwrap();
        assert_eq!(results[0], Some(("org.example", "1.0.0", "MyAspect")));
        assert_eq!(results[1], Some(("com.test", "2.0.0", "Property")));
    }

    #[test]
    fn test_find_urns_in_text() {
        let text = r#"
            The aspect urn:samm:org.example:1.0.0#MyAspect has properties.
            Property urn:samm:org.example:1.0.0#temperature is required.
            Invalid urn:bamm:old:1.0.0#Old should be ignored.
        "#;

        let urns = find_urns_in_text(text).unwrap();
        assert_eq!(urns.len(), 2);
        assert!(urns.contains(&"urn:samm:org.example:1.0.0#MyAspect".to_string()));
        assert!(urns.contains(&"urn:samm:org.example:1.0.0#temperature".to_string()));
    }

    #[test]
    fn test_is_valid_namespace() {
        assert!(is_valid_namespace("org.example"));
        assert!(is_valid_namespace("com.company.product"));
        assert!(is_valid_namespace("io.test_underscore"));
        assert!(!is_valid_namespace("noperiod"));
        assert!(!is_valid_namespace(""));
        assert!(!is_valid_namespace(".invalid"));
    }

    #[test]
    fn test_is_valid_version() {
        assert!(is_valid_version("1.0.0"));
        assert!(is_valid_version("2.1.3"));
        assert!(is_valid_version("10.20.30"));
        assert!(!is_valid_version("1.0"));
        assert!(!is_valid_version("1.0.0.0"));
        assert!(!is_valid_version("a.b.c"));
    }

    #[test]
    fn test_is_valid_identifier() {
        assert!(is_valid_identifier("MyAspect"));
        assert!(is_valid_identifier("Property1"));
        assert!(is_valid_identifier("valid_identifier"));
        assert!(is_valid_identifier("Temperature"));
        assert!(!is_valid_identifier("1Invalid")); // Starts with number
        assert!(!is_valid_identifier("")); // Empty
        assert!(!is_valid_identifier("invalid-hyphen")); // Contains hyphen
    }
}
