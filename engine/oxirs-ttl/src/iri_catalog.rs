/// IRI prefix catalog with CURIE expansion and compression.
///
/// A self-contained registry mapping prefixes to IRI namespaces, with
/// CURIE expansion (`prefix:local` → full IRI), IRI abbreviation
/// (full IRI → `prefix:local`), relative IRI resolution, and standard
/// well-known prefix bootstrapping.
///
/// This module provides the `IriCatalog` type, which is separate from the
/// lower-level [`crate::prefix_resolver::PrefixResolver`] in that it is
/// oriented toward catalog-level management (batch operations, default prefixes,
/// SPARQL-style `PREFIX` syntax).
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from IRI catalog operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogError {
    /// The prefix label is not registered in this catalog.
    UnknownPrefix(String),
    /// The CURIE string is malformed (e.g. missing colon separator).
    InvalidCurie(String),
    /// The IRI string is syntactically invalid.
    InvalidIri(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::UnknownPrefix(p) => write!(f, "Unknown prefix '{p}'"),
            CatalogError::InvalidCurie(c) => write!(f, "Invalid CURIE '{c}'"),
            CatalogError::InvalidIri(i) => write!(f, "Invalid IRI '{i}'"),
        }
    }
}

impl std::error::Error for CatalogError {}

// ---------------------------------------------------------------------------
// IriCatalog
// ---------------------------------------------------------------------------

/// An IRI prefix catalog: maps short labels to namespace IRIs and supports
/// round-trip CURIE ↔ full-IRI conversion.
#[derive(Debug, Clone)]
pub struct IriCatalog {
    /// prefix label → namespace IRI
    prefixes: HashMap<String, String>,
    /// Optional base IRI for resolving relative references.
    base: Option<String>,
}

impl IriCatalog {
    /// Create an empty catalog (no predefined prefixes).
    pub fn new() -> Self {
        IriCatalog {
            prefixes: HashMap::new(),
            base: None,
        }
    }

    /// Create a catalog pre-loaded with the given base IRI.
    pub fn with_base(base: impl Into<String>) -> Self {
        IriCatalog {
            prefixes: HashMap::new(),
            base: Some(base.into()),
        }
    }

    /// Create a catalog pre-populated with the most common RDF/OWL prefixes.
    pub fn with_common_prefixes() -> Self {
        let mut catalog = IriCatalog::new();
        catalog.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        catalog.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        catalog.add_prefix("owl", "http://www.w3.org/2002/07/owl#");
        catalog.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        catalog.add_prefix("skos", "http://www.w3.org/2004/02/skos/core#");
        catalog.add_prefix("dcterms", "http://purl.org/dc/terms/");
        catalog.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        catalog.add_prefix("schema", "https://schema.org/");
        catalog
    }

    /// Register a prefix → namespace mapping.  An existing mapping for the
    /// same prefix label is overwritten silently.
    pub fn add_prefix(&mut self, prefix: impl Into<String>, iri: impl Into<String>) {
        self.prefixes.insert(prefix.into(), iri.into());
    }

    /// Set or replace the base IRI.
    pub fn set_base(&mut self, base: impl Into<String>) {
        self.base = Some(base.into());
    }

    /// Expand a CURIE (`prefix:local`) into its full IRI.
    ///
    /// Returns `Err(CatalogError::InvalidCurie)` when the colon separator is
    /// absent, and `Err(CatalogError::UnknownPrefix)` when the prefix is not
    /// registered.
    pub fn resolve_curie(&self, curie: &str) -> Result<String, CatalogError> {
        let colon = curie
            .find(':')
            .ok_or_else(|| CatalogError::InvalidCurie(curie.to_string()))?;
        let prefix = &curie[..colon];
        let local = &curie[colon + 1..];
        let ns = self
            .prefixes
            .get(prefix)
            .ok_or_else(|| CatalogError::UnknownPrefix(prefix.to_string()))?;
        Ok(format!("{ns}{local}"))
    }

    /// Resolve a relative IRI string against the base IRI.
    ///
    /// Returns `Err(CatalogError::InvalidIri)` when no base IRI is configured.
    pub fn resolve_relative(&self, iri: &str) -> Result<String, CatalogError> {
        match &self.base {
            None => Err(CatalogError::InvalidIri(format!(
                "No base IRI set; cannot resolve '{iri}'"
            ))),
            Some(base) => {
                if iri.is_empty() {
                    return Ok(base.clone());
                }
                if iri.starts_with('#') {
                    return Ok(format!("{base}{iri}"));
                }
                // Resolve against the directory part of the base
                let base_dir = base.rfind('/').map_or(base.as_str(), |idx| &base[..=idx]);
                Ok(format!("{base_dir}{iri}"))
            }
        }
    }

    /// Attempt to abbreviate a full IRI to a `prefix:local` CURIE.
    ///
    /// Chooses the registered namespace that is the longest match (most
    /// specific).  Returns `None` if no prefix covers the IRI.
    pub fn to_curie(&self, iri: &str) -> Option<String> {
        let mut best: Option<(&str, &str)> = None;
        for (prefix, ns) in &self.prefixes {
            if iri.starts_with(ns.as_str()) {
                let is_longer = best.map_or(true, |(_, b_ns)| ns.len() > b_ns.len());
                if is_longer {
                    best = Some((prefix.as_str(), ns.as_str()));
                }
            }
        }
        best.map(|(prefix, ns)| {
            let local = &iri[ns.len()..];
            format!("{prefix}:{local}")
        })
    }

    /// Return `true` when `s` looks like an absolute IRI (has a scheme).
    pub fn is_absolute_iri(s: &str) -> bool {
        // An absolute IRI must contain a colon before any slash
        if let Some(colon_pos) = s.find(':') {
            let scheme = &s[..colon_pos];
            // Scheme must be non-empty and contain only ASCII alphanumerics or + - .
            !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        } else {
            false
        }
    }

    /// Split a CURIE string into its `(prefix, local)` parts.
    /// Returns `None` if the string contains no colon.
    pub fn split_curie(curie: &str) -> Option<(&str, &str)> {
        curie
            .find(':')
            .map(|colon| (&curie[..colon], &curie[colon + 1..]))
    }

    /// Number of registered prefixes.
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }

    /// Return all registered `(prefix, namespace)` pairs in an unspecified order.
    pub fn all_prefixes(&self) -> Vec<(&str, &str)> {
        self.prefixes
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }
}

impl Default for IriCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- new / default ---
    #[test]
    fn test_new_empty() {
        let c = IriCatalog::new();
        assert_eq!(c.prefix_count(), 0);
    }

    #[test]
    fn test_default_empty() {
        let c = IriCatalog::default();
        assert_eq!(c.prefix_count(), 0);
    }

    // --- with_base ---
    #[test]
    fn test_with_base_sets_base() {
        let c = IriCatalog::with_base("http://base.org/");
        assert!(c.base.is_some());
    }

    // --- with_common_prefixes ---
    #[test]
    fn test_common_prefixes_rdf() {
        let c = IriCatalog::with_common_prefixes();
        assert!(c
            .resolve_curie("rdf:type")
            .expect("should succeed")
            .contains("rdf-syntax-ns#type"));
    }

    #[test]
    fn test_common_prefixes_xsd() {
        let c = IriCatalog::with_common_prefixes();
        assert_eq!(
            c.resolve_curie("xsd:string").expect("should succeed"),
            "http://www.w3.org/2001/XMLSchema#string"
        );
    }

    #[test]
    fn test_common_prefixes_owl() {
        let c = IriCatalog::with_common_prefixes();
        assert!(c
            .resolve_curie("owl:Class")
            .expect("should succeed")
            .contains("owl#Class"));
    }

    // --- add_prefix ---
    #[test]
    fn test_add_prefix_increments_count() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://example.org/");
        assert_eq!(c.prefix_count(), 1);
    }

    #[test]
    fn test_add_prefix_overwrites() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://first.org/");
        c.add_prefix("ex", "http://second.org/");
        assert_eq!(c.prefix_count(), 1);
        assert_eq!(
            c.resolve_curie("ex:Foo").expect("should succeed"),
            "http://second.org/Foo"
        );
    }

    // --- resolve_curie ---
    #[test]
    fn test_resolve_curie_basic() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://example.org/");
        assert_eq!(
            c.resolve_curie("ex:Person").expect("should succeed"),
            "http://example.org/Person"
        );
    }

    #[test]
    fn test_resolve_curie_empty_local() {
        let mut c = IriCatalog::new();
        c.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        assert_eq!(
            c.resolve_curie("rdf:").expect("should succeed"),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        );
    }

    #[test]
    fn test_resolve_curie_unknown_prefix_error() {
        let c = IriCatalog::new();
        assert!(matches!(
            c.resolve_curie("unknown:Foo"),
            Err(CatalogError::UnknownPrefix(_))
        ));
    }

    #[test]
    fn test_resolve_curie_no_colon_error() {
        let c = IriCatalog::new();
        assert!(matches!(
            c.resolve_curie("nocolon"),
            Err(CatalogError::InvalidCurie(_))
        ));
    }

    // --- resolve_relative ---
    #[test]
    fn test_resolve_relative_success() {
        let c = IriCatalog::with_base("http://base.org/dir/");
        assert_eq!(
            c.resolve_relative("file.ttl").expect("should succeed"),
            "http://base.org/dir/file.ttl"
        );
    }

    #[test]
    fn test_resolve_relative_empty_string_returns_base() {
        let c = IriCatalog::with_base("http://base.org/doc");
        assert_eq!(
            c.resolve_relative("").expect("should succeed"),
            "http://base.org/doc"
        );
    }

    #[test]
    fn test_resolve_relative_fragment() {
        let c = IriCatalog::with_base("http://base.org/ont");
        assert_eq!(
            c.resolve_relative("#Class").expect("should succeed"),
            "http://base.org/ont#Class"
        );
    }

    #[test]
    fn test_resolve_relative_no_base_error() {
        let c = IriCatalog::new();
        assert!(matches!(
            c.resolve_relative("foo"),
            Err(CatalogError::InvalidIri(_))
        ));
    }

    // --- set_base ---
    #[test]
    fn test_set_base_replaces() {
        let mut c = IriCatalog::new();
        c.set_base("http://first.org/");
        c.set_base("http://second.org/");
        let r = c.resolve_relative("x").expect("should succeed");
        assert!(r.contains("second.org"));
    }

    // --- to_curie ---
    #[test]
    fn test_to_curie_basic() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://example.org/");
        assert_eq!(
            c.to_curie("http://example.org/Person"),
            Some("ex:Person".to_string())
        );
    }

    #[test]
    fn test_to_curie_no_match_none() {
        let c = IriCatalog::new();
        assert!(c.to_curie("http://example.org/Foo").is_none());
    }

    #[test]
    fn test_to_curie_longest_prefix_wins() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://example.org/");
        c.add_prefix("exv", "http://example.org/vocab/");
        assert_eq!(
            c.to_curie("http://example.org/vocab/Thing"),
            Some("exv:Thing".to_string())
        );
    }

    // --- is_absolute_iri ---
    #[test]
    fn test_is_absolute_http() {
        assert!(IriCatalog::is_absolute_iri("http://example.org/"));
    }

    #[test]
    fn test_is_absolute_https() {
        assert!(IriCatalog::is_absolute_iri("https://secure.org/path"));
    }

    #[test]
    fn test_is_absolute_urn() {
        assert!(IriCatalog::is_absolute_iri("urn:isbn:0451450523"));
    }

    #[test]
    fn test_is_absolute_relative_path() {
        assert!(!IriCatalog::is_absolute_iri("relative/path"));
    }

    #[test]
    fn test_is_absolute_no_colon() {
        assert!(!IriCatalog::is_absolute_iri("nocohelon"));
    }

    // --- split_curie ---
    #[test]
    fn test_split_curie_basic() {
        assert_eq!(IriCatalog::split_curie("rdf:type"), Some(("rdf", "type")));
    }

    #[test]
    fn test_split_curie_empty_local() {
        assert_eq!(IriCatalog::split_curie("rdf:"), Some(("rdf", "")));
    }

    #[test]
    fn test_split_curie_no_colon_none() {
        assert_eq!(IriCatalog::split_curie("nocolon"), None);
    }

    #[test]
    fn test_split_curie_empty_prefix() {
        assert_eq!(IriCatalog::split_curie(":local"), Some(("", "local")));
    }

    // --- prefix_count ---
    #[test]
    fn test_prefix_count_increases() {
        let mut c = IriCatalog::new();
        assert_eq!(c.prefix_count(), 0);
        c.add_prefix("a", "http://a/");
        c.add_prefix("b", "http://b/");
        assert_eq!(c.prefix_count(), 2);
    }

    // --- all_prefixes ---
    #[test]
    fn test_all_prefixes_contains_added() {
        let mut c = IriCatalog::new();
        c.add_prefix("ex", "http://example.org/");
        let all: Vec<_> = c.all_prefixes();
        assert!(all.iter().any(|(p, _)| *p == "ex"));
    }

    #[test]
    fn test_all_prefixes_count_matches_prefix_count() {
        let c = IriCatalog::with_common_prefixes();
        assert_eq!(c.all_prefixes().len(), c.prefix_count());
    }

    // --- error display ---
    #[test]
    fn test_error_display_unknown_prefix() {
        let e = CatalogError::UnknownPrefix("xyz".to_string());
        assert!(format!("{e}").contains("xyz"));
    }

    #[test]
    fn test_error_display_invalid_curie() {
        let e = CatalogError::InvalidCurie("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn test_error_display_invalid_iri() {
        let e = CatalogError::InvalidIri("..".to_string());
        assert!(format!("{e}").contains(".."));
    }
}
