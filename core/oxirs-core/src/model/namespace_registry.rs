//! RDF Namespace Registry for CURIE prefix management.
//!
//! Provides bidirectional mapping between short prefixes and full IRI namespaces,
//! enabling CURIE expansion (`ex:Person` → `http://example.org/Person`) and
//! IRI compression (`http://example.org/Person` → `ex:Person`).
//!
//! ## Example
//!
//! ```rust
//! use oxirs_core::model::namespace_registry::{NamespaceRegistry, CurieError};
//!
//! let mut reg = NamespaceRegistry::well_known();
//! reg.register("ex", "http://example.org/").expect("namespace registration should succeed");
//!
//! let expanded = reg.expand("ex:Person").expect("IRI expansion should succeed");
//! assert_eq!(expanded, "http://example.org/Person");
//!
//! let compressed = reg.compress("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("IRI compression should succeed");
//! assert_eq!(compressed, "rdf:type");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during CURIE operations.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum CurieError {
    /// The CURIE string does not contain a colon separator.
    #[error("CURIE is missing colon separator: '{0}'")]
    MissingColon(String),

    /// The prefix in the CURIE is not registered.
    #[error("Unknown prefix '{0}' in CURIE")]
    UnknownPrefix(String),

    /// The CURIE string is otherwise malformed.
    #[error("Invalid CURIE '{0}'")]
    InvalidCurie(String),

    /// Attempt to register a prefix that already exists.
    #[error("Duplicate prefix '{0}' — already registered")]
    DuplicatePrefix(String),

    /// The IRI supplied for registration is empty.
    #[error("IRI cannot be empty for prefix '{0}'")]
    EmptyIri(String),

    /// The prefix supplied for registration is empty.
    #[error("Prefix cannot be empty")]
    EmptyPrefix,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core struct
// ─────────────────────────────────────────────────────────────────────────────

/// Bidirectional registry mapping RDF prefixes ↔ IRI namespaces.
///
/// Supports CURIE expansion and compression. The registry preserves insertion
/// order via a separate key-order list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceRegistry {
    /// prefix → IRI namespace.
    prefix_to_iri: HashMap<String, String>,
    /// IRI namespace → prefix (for fast reverse lookup).
    iri_to_prefix: HashMap<String, String>,
    /// Ordered list of prefix strings (insertion order).
    prefix_order: Vec<String>,
}

impl Default for NamespaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NamespaceRegistry {
    // ── Construction ────────────────────────────────────────────────────────

    /// Creates an empty registry with no prefixes.
    pub fn new() -> Self {
        Self {
            prefix_to_iri: HashMap::new(),
            iri_to_prefix: HashMap::new(),
            prefix_order: Vec::new(),
        }
    }

    /// Creates a registry pre-loaded with the most common well-known prefixes.
    ///
    /// Pre-loaded prefixes:
    /// - `rdf`      → `http://www.w3.org/1999/02/22-rdf-syntax-ns#`
    /// - `rdfs`     → `http://www.w3.org/2000/01/rdf-schema#`
    /// - `owl`      → `http://www.w3.org/2002/07/owl#`
    /// - `xsd`      → `http://www.w3.org/2001/XMLSchema#`
    /// - `dc`       → `http://purl.org/dc/elements/1.1/`
    /// - `dcterms`  → `http://purl.org/dc/terms/`
    /// - `skos`     → `http://www.w3.org/2004/02/skos/core#`
    /// - `foaf`     → `http://xmlns.com/foaf/0.1/`
    pub fn well_known() -> Self {
        let mut reg = Self::new();
        let well_known_pairs = [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
            ("owl", "http://www.w3.org/2002/07/owl#"),
            ("xsd", "http://www.w3.org/2001/XMLSchema#"),
            ("dc", "http://purl.org/dc/elements/1.1/"),
            ("dcterms", "http://purl.org/dc/terms/"),
            ("skos", "http://www.w3.org/2004/02/skos/core#"),
            ("foaf", "http://xmlns.com/foaf/0.1/"),
        ];
        for (prefix, iri) in &well_known_pairs {
            // These are hard-coded valid values; use force_register to bypass
            // duplicate-detection during construction.
            reg.force_register(prefix, iri);
        }
        reg
    }

    // ── Registration ────────────────────────────────────────────────────────

    /// Register a prefix → IRI namespace mapping.
    ///
    /// Returns `Err(CurieError::DuplicatePrefix)` if the prefix is already
    /// registered. Use [`overwrite`](Self::overwrite) to replace an existing
    /// binding.
    pub fn register(&mut self, prefix: &str, iri: &str) -> Result<(), CurieError> {
        if prefix.is_empty() {
            return Err(CurieError::EmptyPrefix);
        }
        if iri.is_empty() {
            return Err(CurieError::EmptyIri(prefix.to_owned()));
        }
        if self.prefix_to_iri.contains_key(prefix) {
            return Err(CurieError::DuplicatePrefix(prefix.to_owned()));
        }
        self.force_register(prefix, iri);
        Ok(())
    }

    /// Register or overwrite a prefix → IRI namespace mapping.
    ///
    /// Unlike [`register`](Self::register), this method silently replaces any
    /// existing binding for `prefix`.
    pub fn overwrite(&mut self, prefix: &str, iri: &str) -> Result<(), CurieError> {
        if prefix.is_empty() {
            return Err(CurieError::EmptyPrefix);
        }
        if iri.is_empty() {
            return Err(CurieError::EmptyIri(prefix.to_owned()));
        }
        // Remove old reverse mapping if prefix was previously registered.
        if let Some(old_iri) = self.prefix_to_iri.get(prefix) {
            let old_iri_owned = old_iri.clone();
            self.iri_to_prefix.remove(&old_iri_owned);
        }
        self.force_register(prefix, iri);
        Ok(())
    }

    /// Internal unconditional insert (used by `well_known` and `overwrite`).
    fn force_register(&mut self, prefix: &str, iri: &str) {
        if !self.prefix_to_iri.contains_key(prefix) {
            self.prefix_order.push(prefix.to_owned());
        }
        self.prefix_to_iri.insert(prefix.to_owned(), iri.to_owned());
        self.iri_to_prefix.insert(iri.to_owned(), prefix.to_owned());
    }

    /// Remove a prefix from the registry.
    ///
    /// Returns `true` if the prefix was present and removed, `false` otherwise.
    pub fn remove(&mut self, prefix: &str) -> bool {
        if let Some(iri) = self.prefix_to_iri.remove(prefix) {
            self.iri_to_prefix.remove(&iri);
            self.prefix_order.retain(|p| p != prefix);
            true
        } else {
            false
        }
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    /// Expand a CURIE (`prefix:local`) to a full IRI.
    ///
    /// Returns `None` if the prefix is not registered or the string has no
    /// colon separator.  For richer error information use
    /// [`expand_checked`](Self::expand_checked).
    pub fn expand(&self, curie: &str) -> Option<String> {
        self.expand_checked(curie).ok()
    }

    /// Expand a CURIE with detailed error information.
    pub fn expand_checked(&self, curie: &str) -> Result<String, CurieError> {
        let colon_pos = curie
            .find(':')
            .ok_or_else(|| CurieError::MissingColon(curie.to_owned()))?;

        // Guard against protocol-like strings ("http://…")
        if colon_pos == 0 {
            return Err(CurieError::InvalidCurie(curie.to_owned()));
        }

        let prefix = &curie[..colon_pos];
        let local = &curie[colon_pos + 1..];

        let iri_base = self
            .prefix_to_iri
            .get(prefix)
            .ok_or_else(|| CurieError::UnknownPrefix(prefix.to_owned()))?;

        Ok(format!("{iri_base}{local}"))
    }

    /// Compress a full IRI to a CURIE using the longest matching prefix.
    ///
    /// Returns `None` if no registered prefix is a prefix of `iri`.
    pub fn compress(&self, iri: &str) -> Option<String> {
        let mut best_prefix: Option<&str> = None;
        let mut best_iri_prefix: Option<&str> = None;
        let mut best_len = 0usize;

        for (iri_prefix, prefix) in &self.iri_to_prefix {
            if iri.starts_with(iri_prefix.as_str()) && iri_prefix.len() > best_len {
                best_len = iri_prefix.len();
                best_prefix = Some(prefix.as_str());
                best_iri_prefix = Some(iri_prefix.as_str());
            }
        }

        let prefix = best_prefix?;
        let iri_base = best_iri_prefix?;
        let local = &iri[iri_base.len()..];
        Some(format!("{prefix}:{local}"))
    }

    /// Look up the IRI for a given prefix.
    pub fn get_iri(&self, prefix: &str) -> Option<&str> {
        self.prefix_to_iri.get(prefix).map(String::as_str)
    }

    /// Look up the prefix for an exact IRI namespace (not a substring match).
    pub fn get_prefix(&self, iri: &str) -> Option<&str> {
        self.iri_to_prefix.get(iri).map(String::as_str)
    }

    /// Returns `true` if the given prefix is registered.
    pub fn contains_prefix(&self, prefix: &str) -> bool {
        self.prefix_to_iri.contains_key(prefix)
    }

    /// Returns `true` if the given IRI namespace is registered.
    pub fn contains_iri(&self, iri: &str) -> bool {
        self.iri_to_prefix.contains_key(iri)
    }

    // ── Iteration & stats ───────────────────────────────────────────────────

    /// Iterate over `(prefix, iri)` pairs in insertion order.
    pub fn iter_prefixes(&self) -> impl Iterator<Item = (&str, &str)> {
        self.prefix_order.iter().filter_map(move |prefix| {
            self.prefix_to_iri
                .get(prefix.as_str())
                .map(|iri| (prefix.as_str(), iri.as_str()))
        })
    }

    /// Number of registered prefixes.
    pub fn len(&self) -> usize {
        self.prefix_order.len()
    }

    /// Returns `true` if no prefixes are registered.
    pub fn is_empty(&self) -> bool {
        self.prefix_order.is_empty()
    }

    /// Return all registered prefixes in insertion order.
    pub fn prefixes(&self) -> Vec<&str> {
        self.prefix_order.iter().map(String::as_str).collect()
    }

    /// Return all registered IRI namespaces in insertion order.
    pub fn iris(&self) -> Vec<&str> {
        self.prefix_order
            .iter()
            .filter_map(|p| self.prefix_to_iri.get(p.as_str()).map(String::as_str))
            .collect()
    }

    /// Merge all entries from `other` into `self`.
    ///
    /// Entries in `self` take precedence — existing prefixes are **not**
    /// overwritten.
    pub fn merge(&mut self, other: &NamespaceRegistry) {
        for (prefix, iri) in other.iter_prefixes() {
            if !self.contains_prefix(prefix) {
                self.force_register(prefix, iri);
            }
        }
    }

    /// Merge all entries from `other`, overwriting existing prefixes in `self`.
    pub fn merge_overwrite(&mut self, other: &NamespaceRegistry) {
        for (prefix, iri) in other.iter_prefixes() {
            // Remove old reverse entry if present
            if let Some(old_iri) = self.prefix_to_iri.get(prefix) {
                let old_iri_clone: String = old_iri.clone();
                self.iri_to_prefix.remove(&old_iri_clone);
            }
            self.force_register(prefix, iri);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Display
// ─────────────────────────────────────────────────────────────────────────────

impl fmt::Display for NamespaceRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for prefix in &self.prefix_order {
            if let Some(iri) = self.prefix_to_iri.get(prefix.as_str()) {
                writeln!(f, "@prefix {prefix}: <{iri}> .")?;
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let reg = NamespaceRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_well_known_has_eight_prefixes() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(reg.len(), 8);
    }

    #[test]
    fn test_well_known_rdf_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.get_iri("rdf"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    #[test]
    fn test_well_known_rdfs_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.get_iri("rdfs"),
            Some("http://www.w3.org/2000/01/rdf-schema#")
        );
    }

    #[test]
    fn test_well_known_owl_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(reg.get_iri("owl"), Some("http://www.w3.org/2002/07/owl#"));
    }

    #[test]
    fn test_well_known_xsd_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.get_iri("xsd"),
            Some("http://www.w3.org/2001/XMLSchema#")
        );
    }

    #[test]
    fn test_well_known_foaf_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(reg.get_iri("foaf"), Some("http://xmlns.com/foaf/0.1/"));
    }

    #[test]
    fn test_well_known_skos_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.get_iri("skos"),
            Some("http://www.w3.org/2004/02/skos/core#")
        );
    }

    #[test]
    fn test_well_known_dc_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(reg.get_iri("dc"), Some("http://purl.org/dc/elements/1.1/"));
    }

    #[test]
    fn test_well_known_dcterms_prefix() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(reg.get_iri("dcterms"), Some("http://purl.org/dc/terms/"));
    }

    // ── Registration ────────────────────────────────────────────────────────

    #[test]
    fn test_register_new_prefix() {
        let mut reg = NamespaceRegistry::new();
        assert!(reg.register("ex", "http://example.org/").is_ok());
        assert_eq!(reg.len(), 1);
        assert!(reg.contains_prefix("ex"));
    }

    #[test]
    fn test_register_duplicate_prefix_fails() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        let err = reg.register("ex", "http://other.org/").unwrap_err();
        assert!(matches!(err, CurieError::DuplicatePrefix(_)));
    }

    #[test]
    fn test_register_empty_prefix_fails() {
        let mut reg = NamespaceRegistry::new();
        let err = reg.register("", "http://example.org/").unwrap_err();
        assert!(matches!(err, CurieError::EmptyPrefix));
    }

    #[test]
    fn test_register_empty_iri_fails() {
        let mut reg = NamespaceRegistry::new();
        let err = reg.register("ex", "").unwrap_err();
        assert!(matches!(err, CurieError::EmptyIri(_)));
    }

    #[test]
    fn test_overwrite_existing_prefix() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert!(reg.overwrite("ex", "http://new.example.org/").is_ok());
        assert_eq!(reg.get_iri("ex"), Some("http://new.example.org/"));
        // Old IRI should no longer be in reverse map
        assert!(!reg.contains_iri("http://example.org/"));
    }

    #[test]
    fn test_remove_existing_prefix() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert!(reg.remove("ex"));
        assert!(!reg.contains_prefix("ex"));
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_prefix_returns_false() {
        let mut reg = NamespaceRegistry::new();
        assert!(!reg.remove("nonexistent"));
    }

    // ── Expand ──────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_simple_curie() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert_eq!(
            reg.expand("ex:Person"),
            Some("http://example.org/Person".to_string())
        );
    }

    #[test]
    fn test_expand_rdf_type() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.expand("rdf:type"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
        );
    }

    #[test]
    fn test_expand_empty_local_name() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert_eq!(reg.expand("ex:"), Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_expand_unknown_prefix_returns_none() {
        let reg = NamespaceRegistry::new();
        assert!(reg.expand("ex:Person").is_none());
    }

    #[test]
    fn test_expand_missing_colon_returns_none() {
        let reg = NamespaceRegistry::new();
        assert!(reg.expand("nocolon").is_none());
    }

    #[test]
    fn test_expand_checked_missing_colon_error() {
        let reg = NamespaceRegistry::new();
        let err = reg.expand_checked("nocolon").unwrap_err();
        assert!(matches!(err, CurieError::MissingColon(_)));
    }

    #[test]
    fn test_expand_checked_unknown_prefix_error() {
        let reg = NamespaceRegistry::new();
        let err = reg.expand_checked("ex:Foo").unwrap_err();
        assert!(matches!(err, CurieError::UnknownPrefix(_)));
    }

    #[test]
    fn test_expand_checked_leading_colon_invalid() {
        let reg = NamespaceRegistry::new();
        let err = reg.expand_checked(":local").unwrap_err();
        assert!(matches!(err, CurieError::InvalidCurie(_)));
    }

    // ── Compress ────────────────────────────────────────────────────────────

    #[test]
    fn test_compress_rdf_type() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.compress("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Some("rdf:type".to_string())
        );
    }

    #[test]
    fn test_compress_rdfs_label() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.compress("http://www.w3.org/2000/01/rdf-schema#label"),
            Some("rdfs:label".to_string())
        );
    }

    #[test]
    fn test_compress_owl_class() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.compress("http://www.w3.org/2002/07/owl#Class"),
            Some("owl:Class".to_string())
        );
    }

    #[test]
    fn test_compress_unknown_iri_returns_none() {
        let reg = NamespaceRegistry::new();
        assert!(reg.compress("http://totally-unknown.example/Foo").is_none());
    }

    #[test]
    fn test_compress_longest_match() {
        let mut reg = NamespaceRegistry::new();
        reg.register("short", "http://example.org/")
            .expect("namespace registration should succeed");
        reg.register("long", "http://example.org/sub/")
            .expect("namespace registration should succeed");
        // "long" should win because it is a longer match
        assert_eq!(
            reg.compress("http://example.org/sub/Thing"),
            Some("long:Thing".to_string())
        );
    }

    #[test]
    fn test_compress_and_expand_roundtrip() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        let original = "http://example.org/MyClass";
        let curie = reg
            .compress(original)
            .expect("IRI compression should succeed");
        let expanded = reg.expand(&curie).expect("IRI expansion should succeed");
        assert_eq!(expanded, original);
    }

    // ── Iteration ───────────────────────────────────────────────────────────

    #[test]
    fn test_iter_prefixes() {
        let mut reg = NamespaceRegistry::new();
        reg.register("a", "http://a.org/")
            .expect("namespace registration should succeed");
        reg.register("b", "http://b.org/")
            .expect("namespace registration should succeed");
        let pairs: Vec<_> = reg.iter_prefixes().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("a", "http://a.org/"));
        assert_eq!(pairs[1], ("b", "http://b.org/"));
    }

    #[test]
    fn test_prefixes_method() {
        let mut reg = NamespaceRegistry::new();
        reg.register("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            .expect("operation should succeed");
        let p = reg.prefixes();
        assert_eq!(p, vec!["rdf"]);
    }

    #[test]
    fn test_iris_method() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        let iris = reg.iris();
        assert_eq!(iris, vec!["http://example.org/"]);
    }

    // ── Merge ────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_no_overwrite() {
        let mut reg1 = NamespaceRegistry::new();
        reg1.register("ex", "http://example.org/")
            .expect("registration should succeed");

        let mut reg2 = NamespaceRegistry::new();
        reg2.register("ex", "http://other.org/")
            .expect("registration should succeed");
        reg2.register("foo", "http://foo.org/")
            .expect("registration should succeed");

        reg1.merge(&reg2);

        // "ex" should NOT be overwritten
        assert_eq!(reg1.get_iri("ex"), Some("http://example.org/"));
        // "foo" should be added
        assert_eq!(reg1.get_iri("foo"), Some("http://foo.org/"));
        assert_eq!(reg1.len(), 2);
    }

    #[test]
    fn test_merge_overwrite() {
        let mut reg1 = NamespaceRegistry::new();
        reg1.register("ex", "http://example.org/")
            .expect("registration should succeed");

        let mut reg2 = NamespaceRegistry::new();
        reg2.register("ex", "http://new.org/")
            .expect("registration should succeed");

        reg1.merge_overwrite(&reg2);
        assert_eq!(reg1.get_iri("ex"), Some("http://new.org/"));
    }

    // ── Display ──────────────────────────────────────────────────────────────

    #[test]
    fn test_display_format() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        let output = format!("{reg}");
        assert!(output.contains("@prefix ex: <http://example.org/> ."));
    }

    // ── Serialization ────────────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let mut reg = NamespaceRegistry::well_known();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");

        let json = serde_json::to_string(&reg).expect("construction should succeed");
        let restored: NamespaceRegistry =
            serde_json::from_str(&json).expect("construction should succeed");

        assert_eq!(restored.len(), reg.len());
        assert_eq!(restored.get_iri("ex"), Some("http://example.org/"));
        assert_eq!(
            restored.get_iri("rdf"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_contains_prefix_true() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert!(reg.contains_prefix("ex"));
        assert!(!reg.contains_prefix("other"));
    }

    #[test]
    fn test_contains_iri_true() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert!(reg.contains_iri("http://example.org/"));
        assert!(!reg.contains_iri("http://other.org/"));
    }

    #[test]
    fn test_get_prefix_by_iri() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/")
            .expect("namespace registration should succeed");
        assert_eq!(reg.get_prefix("http://example.org/"), Some("ex"));
    }

    #[test]
    fn test_default_is_empty() {
        let reg = NamespaceRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_expand_xsd_string() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.expand("xsd:string"),
            Some("http://www.w3.org/2001/XMLSchema#string".to_string())
        );
    }

    #[test]
    fn test_expand_foaf_name() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.expand("foaf:name"),
            Some("http://xmlns.com/foaf/0.1/name".to_string())
        );
    }

    #[test]
    fn test_compress_xsd_integer() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.compress("http://www.w3.org/2001/XMLSchema#integer"),
            Some("xsd:integer".to_string())
        );
    }

    #[test]
    fn test_compress_foaf_agent() {
        let reg = NamespaceRegistry::well_known();
        assert_eq!(
            reg.compress("http://xmlns.com/foaf/0.1/Agent"),
            Some("foaf:Agent".to_string())
        );
    }

    #[test]
    fn test_multiple_registers_increment_len() {
        let mut reg = NamespaceRegistry::new();
        for i in 0..10 {
            reg.register(&format!("p{i}"), &format!("http://p{i}.org/"))
                .expect("operation should succeed");
        }
        assert_eq!(reg.len(), 10);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_remove_then_register_same_prefix() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://old.org/")
            .expect("namespace registration should succeed");
        reg.remove("ex");
        assert!(reg.register("ex", "http://new.org/").is_ok());
        assert_eq!(reg.get_iri("ex"), Some("http://new.org/"));
    }
}
