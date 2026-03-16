//! Namespace/prefix management for Turtle and SPARQL serialization.
//!
//! Provides bidirectional mapping between prefixes (e.g. `rdf:`) and namespace IRIs
//! (e.g. `http://www.w3.org/1999/02/22-rdf-syntax-ns#`), with helpers to
//! abbreviate/expand CURIEs and generate `@prefix` / `PREFIX` declarations.

use std::collections::HashMap;

// ─────────────────────────────────────────────────
// NamespaceError
// ─────────────────────────────────────────────────

/// Errors that can occur when manipulating namespace mappings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceError {
    /// The given prefix is already registered.
    DuplicatePrefix(String),
    /// The given prefix string is malformed (e.g. contains `:`).
    InvalidPrefix(String),
    /// The given namespace IRI is empty or otherwise invalid.
    InvalidNamespace(String),
}

impl std::fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NamespaceError::DuplicatePrefix(p) => write!(f, "Duplicate prefix: {p}"),
            NamespaceError::InvalidPrefix(p) => write!(f, "Invalid prefix: {p}"),
            NamespaceError::InvalidNamespace(n) => write!(f, "Invalid namespace: {n}"),
        }
    }
}

// ─────────────────────────────────────────────────
// PrefixMapping (utility struct)
// ─────────────────────────────────────────────────

/// A single prefix → namespace mapping entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMapping {
    /// The short prefix name (without trailing `:`).
    pub prefix: String,
    /// The full namespace IRI.
    pub namespace: String,
}

// ─────────────────────────────────────────────────
// NamespaceMapper
// ─────────────────────────────────────────────────

/// Bidirectional registry of namespace prefixes and their IRIs.
#[derive(Debug, Clone, Default)]
pub struct NamespaceMapper {
    /// prefix → namespace
    mappings: HashMap<String, String>,
    /// namespace → prefix (reverse lookup)
    reverse: HashMap<String, String>,
}

impl NamespaceMapper {
    /// Create an empty mapper.
    pub fn new() -> Self {
        NamespaceMapper {
            mappings: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Create a mapper pre-loaded with the most common RDF standard prefixes.
    pub fn with_defaults() -> Self {
        let mut m = Self::new();
        // These are guaranteed to succeed; if they fail it is a programming error
        let _ = m.add("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let _ = m.add("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        let _ = m.add("owl", "http://www.w3.org/2002/07/owl#");
        let _ = m.add("xsd", "http://www.w3.org/2001/XMLSchema#");
        let _ = m.add("dc", "http://purl.org/dc/elements/1.1/");
        m
    }

    /// Register a prefix mapping.
    ///
    /// Returns `Err(DuplicatePrefix)` if the prefix is already registered.
    /// Returns `Err(InvalidPrefix)` if the prefix string is empty or contains `:`
    ///   (only the bare name should be supplied; the trailing `:` is added automatically
    ///    during serialization).
    /// Returns `Err(InvalidNamespace)` if the namespace is empty.
    pub fn add(
        &mut self,
        prefix: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Result<(), NamespaceError> {
        let prefix = prefix.into();
        let namespace = namespace.into();

        if prefix.contains(':') {
            return Err(NamespaceError::InvalidPrefix(prefix));
        }
        if namespace.is_empty() {
            return Err(NamespaceError::InvalidNamespace(namespace));
        }
        if self.mappings.contains_key(&prefix) {
            return Err(NamespaceError::DuplicatePrefix(prefix));
        }

        self.reverse.insert(namespace.clone(), prefix.clone());
        self.mappings.insert(prefix, namespace);
        Ok(())
    }

    /// Remove a prefix mapping.  Returns `true` if the mapping existed.
    pub fn remove(&mut self, prefix: &str) -> bool {
        if let Some(ns) = self.mappings.remove(prefix) {
            self.reverse.remove(&ns);
            true
        } else {
            false
        }
    }

    /// Look up the namespace for a given prefix.
    pub fn get_namespace(&self, prefix: &str) -> Option<&str> {
        self.mappings.get(prefix).map(String::as_str)
    }

    /// Reverse-lookup: find a prefix for a given namespace IRI.
    pub fn get_prefix(&self, namespace: &str) -> Option<&str> {
        self.reverse.get(namespace).map(String::as_str)
    }

    /// Number of registered mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Return `true` when no mappings are registered.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Return all registered prefix names.
    pub fn prefix_names(&self) -> Vec<&str> {
        self.mappings.keys().map(String::as_str).collect()
    }

    /// Return all (prefix, namespace) pairs.
    pub fn all_mappings(&self) -> Vec<(&str, &str)> {
        self.mappings
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    /// Try to abbreviate a full IRI to a CURIE (prefix:local).
    ///
    /// Iterates all registered namespaces and returns the first match, choosing
    /// the longest matching namespace for determinism.
    pub fn abbreviate(&self, iri: &str) -> Option<String> {
        // Find the longest matching namespace
        let best = self
            .mappings
            .iter()
            .filter(|(_, ns)| iri.starts_with(ns.as_str()))
            .max_by_key(|(_, ns)| ns.len());

        best.map(|(prefix, ns)| {
            let local = &iri[ns.len()..];
            format!("{prefix}:{local}")
        })
    }

    /// Expand a CURIE (prefix:local) to a full IRI.
    ///
    /// Returns `None` if the prefix is not registered.
    pub fn expand(&self, curie: &str) -> Option<String> {
        let colon = curie.find(':')?;
        let prefix = &curie[..colon];
        let local = &curie[colon + 1..];
        let ns = self.mappings.get(prefix)?;
        Some(format!("{ns}{local}"))
    }

    /// Produce Turtle `@prefix` declarations for all registered mappings.
    pub fn to_turtle_declarations(&self) -> String {
        let mut entries: Vec<(&str, &str)> = self
            .mappings
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        entries.sort_by_key(|(p, _)| *p);
        entries
            .iter()
            .map(|(p, ns)| format!("@prefix {p}: <{ns}> .\n"))
            .collect()
    }

    /// Produce SPARQL `PREFIX` declarations for all registered mappings.
    pub fn to_sparql_declarations(&self) -> String {
        let mut entries: Vec<(&str, &str)> = self
            .mappings
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        entries.sort_by_key(|(p, _)| *p);
        entries
            .iter()
            .map(|(p, ns)| format!("PREFIX {p}: <{ns}>\n"))
            .collect()
    }

    /// Merge all mappings from `other` into this mapper.
    ///
    /// Silently skips any prefix that is already registered (no error).
    pub fn merge(&mut self, other: &NamespaceMapper) {
        for (prefix, namespace) in &other.mappings {
            if !self.mappings.contains_key(prefix.as_str()) {
                let _ = self.add(prefix.clone(), namespace.clone());
            }
        }
    }
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default prefixes ──────────────────────────────────────

    #[test]
    fn test_defaults_contains_rdf() {
        let m = NamespaceMapper::with_defaults();
        assert_eq!(
            m.get_namespace("rdf"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    #[test]
    fn test_defaults_contains_rdfs() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.get_namespace("rdfs").is_some());
    }

    #[test]
    fn test_defaults_contains_owl() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.get_namespace("owl").is_some());
    }

    #[test]
    fn test_defaults_contains_xsd() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.get_namespace("xsd").is_some());
    }

    #[test]
    fn test_defaults_contains_dc() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.get_namespace("dc").is_some());
    }

    #[test]
    fn test_defaults_len_at_least_five() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.len() >= 5);
    }

    // ── Add / Remove ──────────────────────────────────────────

    #[test]
    fn test_add_lookup() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        assert_eq!(m.get_namespace("ex"), Some("http://example.org/"));
    }

    #[test]
    fn test_add_duplicate_error() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        let err = m.add("ex", "http://other.org/").unwrap_err();
        assert_eq!(err, NamespaceError::DuplicatePrefix("ex".into()));
    }

    #[test]
    fn test_add_invalid_prefix_with_colon() {
        let mut m = NamespaceMapper::new();
        let err = m.add("ex:", "http://example.org/").unwrap_err();
        assert_eq!(err, NamespaceError::InvalidPrefix("ex:".into()));
    }

    #[test]
    fn test_add_empty_namespace_error() {
        let mut m = NamespaceMapper::new();
        let err = m.add("ex", "").unwrap_err();
        assert_eq!(err, NamespaceError::InvalidNamespace("".into()));
    }

    #[test]
    fn test_remove_existing() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        assert!(m.remove("ex"));
        assert!(m.get_namespace("ex").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut m = NamespaceMapper::new();
        assert!(!m.remove("nonexistent"));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut m = NamespaceMapper::new();
        assert!(m.is_empty());
        m.add("a", "http://a.org/").expect("should succeed");
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
    }

    // ── Abbreviate IRI ────────────────────────────────────────

    #[test]
    fn test_abbreviate_rdf_type() {
        let m = NamespaceMapper::with_defaults();
        let curie = m.abbreviate("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert_eq!(curie, Some("rdf:type".into()));
    }

    #[test]
    fn test_abbreviate_rdfs_label() {
        let m = NamespaceMapper::with_defaults();
        let curie = m.abbreviate("http://www.w3.org/2000/01/rdf-schema#label");
        assert_eq!(curie, Some("rdfs:label".into()));
    }

    #[test]
    fn test_abbreviate_unknown_iri_returns_none() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.abbreviate("http://unknown.example.org/term").is_none());
    }

    #[test]
    fn test_abbreviate_empty_local() {
        let mut m = NamespaceMapper::new();
        m.add("ns", "http://ns.example.org/")
            .expect("should succeed");
        let curie = m.abbreviate("http://ns.example.org/");
        assert_eq!(curie, Some("ns:".into()));
    }

    // ── Expand CURIE ──────────────────────────────────────────

    #[test]
    fn test_expand_rdf_type() {
        let m = NamespaceMapper::with_defaults();
        let iri = m.expand("rdf:type");
        assert_eq!(
            iri,
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into())
        );
    }

    #[test]
    fn test_expand_unknown_prefix_returns_none() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.expand("unknown:term").is_none());
    }

    #[test]
    fn test_expand_no_colon_returns_none() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.expand("nodot").is_none());
    }

    #[test]
    fn test_expand_xsd_string() {
        let m = NamespaceMapper::with_defaults();
        let iri = m.expand("xsd:string");
        assert_eq!(iri, Some("http://www.w3.org/2001/XMLSchema#string".into()));
    }

    // ── Turtle declarations ───────────────────────────────────

    #[test]
    fn test_turtle_declarations_format() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        let decls = m.to_turtle_declarations();
        assert!(decls.contains("@prefix ex: <http://example.org/> ."));
    }

    #[test]
    fn test_turtle_declarations_all_defaults() {
        let m = NamespaceMapper::with_defaults();
        let decls = m.to_turtle_declarations();
        assert!(decls.contains("@prefix rdf:"));
        assert!(decls.contains("@prefix rdfs:"));
        assert!(decls.contains("@prefix owl:"));
        assert!(decls.contains("@prefix xsd:"));
        assert!(decls.contains("@prefix dc:"));
    }

    // ── SPARQL declarations ───────────────────────────────────

    #[test]
    fn test_sparql_declarations_format() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        let decls = m.to_sparql_declarations();
        assert!(decls.contains("PREFIX ex: <http://example.org/>"));
    }

    #[test]
    fn test_sparql_declarations_all_defaults() {
        let m = NamespaceMapper::with_defaults();
        let decls = m.to_sparql_declarations();
        assert!(decls.contains("PREFIX rdf:"));
        assert!(decls.contains("PREFIX rdfs:"));
    }

    // ── Merge two mappers ─────────────────────────────────────

    #[test]
    fn test_merge_adds_missing_prefixes() {
        let mut m1 = NamespaceMapper::new();
        m1.add("ex", "http://example.org/").expect("should succeed");

        let mut m2 = NamespaceMapper::new();
        m2.add("schema", "https://schema.org/")
            .expect("should succeed");

        m1.merge(&m2);
        assert!(m1.get_namespace("ex").is_some());
        assert!(m1.get_namespace("schema").is_some());
    }

    #[test]
    fn test_merge_skips_duplicates() {
        let mut m1 = NamespaceMapper::new();
        m1.add("ex", "http://example.org/").expect("should succeed");

        let mut m2 = NamespaceMapper::new();
        m2.add("ex", "http://other.org/").expect("should succeed");

        // Should not overwrite
        m1.merge(&m2);
        assert_eq!(m1.get_namespace("ex"), Some("http://example.org/"));
    }

    // ── Get prefix (reverse lookup) ───────────────────────────

    #[test]
    fn test_get_prefix_reverse() {
        let m = NamespaceMapper::with_defaults();
        let prefix = m.get_prefix("http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        assert_eq!(prefix, Some("rdf"));
    }

    #[test]
    fn test_get_prefix_unknown_returns_none() {
        let m = NamespaceMapper::with_defaults();
        assert!(m.get_prefix("http://totally-unknown.example/").is_none());
    }

    // ── all_mappings & prefix_names ────────────────────────────

    #[test]
    fn test_all_mappings_count() {
        let m = NamespaceMapper::with_defaults();
        let all = m.all_mappings();
        assert!(all.len() >= 5);
    }

    #[test]
    fn test_prefix_names_contains_defaults() {
        let m = NamespaceMapper::with_defaults();
        let names = m.prefix_names();
        assert!(names.contains(&"rdf"));
        assert!(names.contains(&"owl"));
    }

    // ── Default impl ──────────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let m = NamespaceMapper::default();
        assert!(m.is_empty());
    }

    // ── PrefixMapping struct ──────────────────────────────────

    #[test]
    fn test_prefix_mapping_struct() {
        let pm = PrefixMapping {
            prefix: "ex".into(),
            namespace: "http://example.org/".into(),
        };
        assert_eq!(pm.prefix, "ex");
        assert_eq!(pm.namespace, "http://example.org/");
    }

    // ── NamespaceError Display ────────────────────────────────

    #[test]
    fn test_error_display_duplicate() {
        let e = NamespaceError::DuplicatePrefix("rdf".into());
        assert!(e.to_string().contains("rdf"));
    }

    #[test]
    fn test_error_display_invalid_prefix() {
        let e = NamespaceError::InvalidPrefix("bad:".into());
        assert!(e.to_string().contains("bad:"));
    }

    #[test]
    fn test_error_display_invalid_namespace() {
        let e = NamespaceError::InvalidNamespace("".into());
        assert!(e.to_string().contains("Invalid namespace"));
    }

    // ── Round-trip abbreviate/expand ─────────────────────────

    #[test]
    fn test_round_trip_abbreviate_expand() {
        let m = NamespaceMapper::with_defaults();
        let iri = "http://www.w3.org/2002/07/owl#Class";
        let curie = m.abbreviate(iri).expect("should succeed");
        let expanded = m.expand(&curie).expect("should succeed");
        assert_eq!(expanded, iri);
    }

    // ── Remove after add ──────────────────────────────────────

    #[test]
    fn test_remove_clears_reverse_index() {
        let mut m = NamespaceMapper::new();
        m.add("ex", "http://example.org/").expect("should succeed");
        m.remove("ex");
        // Reverse lookup must also be gone
        assert!(m.get_prefix("http://example.org/").is_none());
    }

    // ── Multiple additions ────────────────────────────────────

    #[test]
    fn test_add_many_prefixes() {
        let mut m = NamespaceMapper::new();
        for i in 0..10_u32 {
            m.add(format!("ns{i}"), format!("http://ns{i}.example.org/"))
                .expect("should succeed");
        }
        assert_eq!(m.len(), 10);
    }
}
