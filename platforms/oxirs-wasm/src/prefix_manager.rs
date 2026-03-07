//! Namespace/prefix management for RDF serializers.
//!
//! Provides a `PrefixManager` that maps between prefix strings and IRI
//! namespaces, supports abbreviation (full IRI → prefixed name) and
//! expansion (prefixed name → full IRI), and can generate Turtle and SPARQL
//! PREFIX declarations.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Well-known constants
// ---------------------------------------------------------------------------

pub const RDF_PREFIX: &str = "rdf";
pub const RDF_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

pub const RDFS_PREFIX: &str = "rdfs";
pub const RDFS_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#";

pub const OWL_PREFIX: &str = "owl";
pub const OWL_IRI: &str = "http://www.w3.org/2002/07/owl#";

pub const XSD_PREFIX: &str = "xsd";
pub const XSD_IRI: &str = "http://www.w3.org/2001/XMLSchema#";

pub const SCHEMA_PREFIX: &str = "schema";
pub const SCHEMA_IRI: &str = "https://schema.org/";

pub const SKOS_PREFIX: &str = "skos";
pub const SKOS_IRI: &str = "http://www.w3.org/2004/02/skos/core#";

// ---------------------------------------------------------------------------
// Namespace
// ---------------------------------------------------------------------------

/// A single prefix ↔ IRI pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Namespace {
    pub prefix: String,
    pub iri: String,
}

// ---------------------------------------------------------------------------
// AbbrevResult
// ---------------------------------------------------------------------------

/// Result of attempting to abbreviate a full IRI.
#[derive(Debug, Clone, PartialEq)]
pub enum AbbrevResult {
    /// Successfully abbreviated to `prefix:local`.
    PrefixedName { prefix: String, local: String },
    /// No registered namespace matches; the full IRI is returned as-is.
    FullIri(String),
    /// The input is not a valid IRI.
    Invalid,
}

impl AbbrevResult {
    /// Render as a string: `"prefix:local"`, `"<iri>"`, or `"<invalid>"`.
    pub fn to_str(&self) -> String {
        match self {
            AbbrevResult::PrefixedName { prefix, local } => format!("{}:{}", prefix, local),
            AbbrevResult::FullIri(iri) => format!("<{}>", iri),
            AbbrevResult::Invalid => "<invalid>".to_string(),
        }
    }

    /// Returns `true` if a prefix abbreviation was found.
    pub fn is_abbreviated(&self) -> bool {
        matches!(self, AbbrevResult::PrefixedName { .. })
    }
}

// ---------------------------------------------------------------------------
// PrefixError
// ---------------------------------------------------------------------------

/// Errors produced by `PrefixManager` operations.
#[derive(Debug)]
pub enum PrefixError {
    /// The prefix string is already registered.
    DuplicatePrefix(String),
    /// The namespace IRI is already registered under a different prefix.
    DuplicateNamespace(String),
    /// The supplied IRI is syntactically invalid.
    InvalidIri(String),
    /// The supplied prefix string is syntactically invalid.
    InvalidPrefix(String),
    /// No namespace is registered for the given prefix.
    PrefixNotFound(String),
}

impl std::fmt::Display for PrefixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefixError::DuplicatePrefix(p) => write!(f, "Duplicate prefix: {}", p),
            PrefixError::DuplicateNamespace(i) => write!(f, "Duplicate namespace IRI: {}", i),
            PrefixError::InvalidIri(i) => write!(f, "Invalid IRI: {}", i),
            PrefixError::InvalidPrefix(p) => write!(f, "Invalid prefix: {}", p),
            PrefixError::PrefixNotFound(p) => write!(f, "Prefix not found: {}", p),
        }
    }
}

impl std::error::Error for PrefixError {}

// ---------------------------------------------------------------------------
// PrefixManager
// ---------------------------------------------------------------------------

/// Registry mapping between RDF namespace prefixes and IRI bases.
pub struct PrefixManager {
    /// Insertion-ordered list of namespaces.
    namespaces: Vec<Namespace>,
    /// prefix → IRI
    prefix_map: HashMap<String, String>,
    /// IRI → prefix (for abbreviation)
    iri_map: HashMap<String, String>,
}

impl PrefixManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty `PrefixManager`.
    pub fn new() -> Self {
        Self {
            namespaces: Vec::new(),
            prefix_map: HashMap::new(),
            iri_map: HashMap::new(),
        }
    }

    /// Create a `PrefixManager` pre-loaded with common RDF prefixes.
    pub fn with_defaults() -> Self {
        let mut pm = Self::new();
        // Safe: all constants are valid
        pm.add_or_update(RDF_PREFIX, RDF_IRI);
        pm.add_or_update(RDFS_PREFIX, RDFS_IRI);
        pm.add_or_update(OWL_PREFIX, OWL_IRI);
        pm.add_or_update(XSD_PREFIX, XSD_IRI);
        pm.add_or_update(SCHEMA_PREFIX, SCHEMA_IRI);
        pm.add_or_update(SKOS_PREFIX, SKOS_IRI);
        pm
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Add a new prefix–IRI mapping.
    ///
    /// # Errors
    /// - `InvalidPrefix` if `prefix` is empty or contains spaces.
    /// - `InvalidIri` if `iri` is empty or does not end with `#` or `/`.
    /// - `DuplicatePrefix` if the prefix is already registered.
    pub fn add(
        &mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<(), PrefixError> {
        let prefix = prefix.into();
        let iri = iri.into();

        Self::validate_prefix(&prefix)?;
        Self::validate_iri(&iri)?;

        if self.prefix_map.contains_key(&prefix) {
            return Err(PrefixError::DuplicatePrefix(prefix));
        }

        self.insert_internal(prefix, iri);
        Ok(())
    }

    /// Add or overwrite a prefix–IRI mapping (no error on duplicate).
    pub fn add_or_update(&mut self, prefix: impl Into<String>, iri: impl Into<String>) {
        let prefix = prefix.into();
        let iri = iri.into();
        // Remove stale reverse mapping if prefix was registered before
        if let Some(old_iri) = self.prefix_map.get(&prefix).cloned() {
            self.iri_map.remove(&old_iri);
            self.namespaces.retain(|n| n.prefix != prefix);
        }
        self.insert_internal(prefix, iri);
    }

    /// Remove a namespace by prefix.  Returns the removed `Namespace` if it
    /// existed.
    pub fn remove(&mut self, prefix: &str) -> Option<Namespace> {
        if let Some(iri) = self.prefix_map.remove(prefix) {
            self.iri_map.remove(&iri);
            let pos = self.namespaces.iter().position(|n| n.prefix == prefix);
            if let Some(idx) = pos {
                return Some(self.namespaces.remove(idx));
            }
        }
        None
    }

    /// Clear all namespaces.
    pub fn clear(&mut self) {
        self.namespaces.clear();
        self.prefix_map.clear();
        self.iri_map.clear();
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Get the IRI for a given prefix.
    pub fn get_iri(&self, prefix: &str) -> Option<&str> {
        self.prefix_map.get(prefix).map(String::as_str)
    }

    /// Get the prefix registered for an IRI.
    pub fn get_prefix(&self, iri: &str) -> Option<&str> {
        self.iri_map.get(iri).map(String::as_str)
    }

    /// Returns `true` if the prefix is registered.
    pub fn contains_prefix(&self, prefix: &str) -> bool {
        self.prefix_map.contains_key(prefix)
    }

    /// Returns `true` if the IRI is registered as a namespace.
    pub fn contains_iri(&self, iri: &str) -> bool {
        self.iri_map.contains_key(iri)
    }

    /// Number of registered namespaces.
    pub fn prefix_count(&self) -> usize {
        self.namespaces.len()
    }

    /// Slice of all registered `Namespace` values in insertion order.
    pub fn all_namespaces(&self) -> &[Namespace] {
        &self.namespaces
    }

    // -----------------------------------------------------------------------
    // Abbreviation / expansion
    // -----------------------------------------------------------------------

    /// Abbreviate a full IRI to a prefixed name using the longest matching
    /// namespace.
    ///
    /// If no namespace matches, returns `AbbrevResult::FullIri`.
    /// If `iri` is empty, returns `AbbrevResult::Invalid`.
    pub fn abbreviate(&self, iri: &str) -> AbbrevResult {
        if iri.is_empty() {
            return AbbrevResult::Invalid;
        }

        // Find longest registered IRI that is a prefix of `iri`
        let mut best_match: Option<(&str, &str)> = None; // (ns_iri, prefix)
        for ns in &self.namespaces {
            if iri.starts_with(&ns.iri) {
                let is_longer = best_match
                    .map(|(best_iri, _)| ns.iri.len() > best_iri.len())
                    .unwrap_or(true);
                if is_longer {
                    best_match = Some((&ns.iri, &ns.prefix));
                }
            }
        }

        match best_match {
            Some((ns_iri, prefix)) => {
                let local = &iri[ns_iri.len()..];
                AbbrevResult::PrefixedName {
                    prefix: prefix.to_string(),
                    local: local.to_string(),
                }
            }
            None => AbbrevResult::FullIri(iri.to_string()),
        }
    }

    /// Expand a prefixed name to a full IRI.
    ///
    /// # Errors
    /// - `PrefixNotFound` if the prefix is not registered.
    pub fn expand(&self, prefixed: &str) -> Result<String, PrefixError> {
        let colon_pos = prefixed
            .find(':')
            .ok_or_else(|| PrefixError::PrefixNotFound(format!("no colon in '{}'", prefixed)))?;
        let prefix = &prefixed[..colon_pos];
        let local = &prefixed[colon_pos + 1..];

        let ns_iri = self
            .prefix_map
            .get(prefix)
            .ok_or_else(|| PrefixError::PrefixNotFound(prefix.to_string()))?;
        Ok(format!("{}{}", ns_iri, local))
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Generate Turtle `@prefix` declarations for all registered namespaces.
    pub fn turtle_declarations(&self) -> String {
        let mut out = String::new();
        for ns in &self.namespaces {
            out.push_str(&format!("@prefix {}: <{}> .\n", ns.prefix, ns.iri));
        }
        out
    }

    /// Generate SPARQL `PREFIX` declarations for all registered namespaces.
    pub fn sparql_declarations(&self) -> String {
        let mut out = String::new();
        for ns in &self.namespaces {
            out.push_str(&format!("PREFIX {}: <{}>\n", ns.prefix, ns.iri));
        }
        out
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn insert_internal(&mut self, prefix: String, iri: String) {
        self.prefix_map.insert(prefix.clone(), iri.clone());
        self.iri_map.insert(iri.clone(), prefix.clone());
        self.namespaces.push(Namespace { prefix, iri });
    }

    /// Validate a prefix string (must be non-empty and contain no whitespace).
    fn validate_prefix(prefix: &str) -> Result<(), PrefixError> {
        if prefix.contains(char::is_whitespace) {
            return Err(PrefixError::InvalidPrefix(format!(
                "prefix '{}' contains whitespace",
                prefix
            )));
        }
        // We allow the empty string as the default (empty) prefix in Turtle,
        // so we only flag embedded spaces, not the empty string.
        Ok(())
    }

    /// Validate an IRI string (must be non-empty).
    fn validate_iri(iri: &str) -> Result<(), PrefixError> {
        if iri.is_empty() {
            return Err(PrefixError::InvalidIri("IRI must not be empty".to_string()));
        }
        Ok(())
    }
}

impl Default for PrefixManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- construction -------------------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let pm = PrefixManager::new();
        assert_eq!(pm.prefix_count(), 0);
        assert!(pm.all_namespaces().is_empty());
    }

    #[test]
    fn test_with_defaults_has_standard_prefixes() {
        let pm = PrefixManager::with_defaults();
        assert!(pm.contains_prefix("rdf"), "missing rdf");
        assert!(pm.contains_prefix("rdfs"), "missing rdfs");
        assert!(pm.contains_prefix("owl"), "missing owl");
        assert!(pm.contains_prefix("xsd"), "missing xsd");
        assert!(pm.contains_prefix("schema"), "missing schema");
        assert!(pm.contains_prefix("skos"), "missing skos");
    }

    #[test]
    fn test_with_defaults_correct_iris() {
        let pm = PrefixManager::with_defaults();
        assert_eq!(pm.get_iri("rdf"), Some(RDF_IRI));
        assert_eq!(pm.get_iri("rdfs"), Some(RDFS_IRI));
        assert_eq!(pm.get_iri("owl"), Some(OWL_IRI));
        assert_eq!(pm.get_iri("xsd"), Some(XSD_IRI));
    }

    // --- add ----------------------------------------------------------------

    #[test]
    fn test_add_prefix_ok() {
        let mut pm = PrefixManager::new();
        pm.add("ex", "http://example.org/").expect("ok");
        assert_eq!(pm.prefix_count(), 1);
        assert_eq!(pm.get_iri("ex"), Some("http://example.org/"));
    }

    #[test]
    fn test_add_duplicate_prefix_error() {
        let mut pm = PrefixManager::new();
        pm.add("ex", "http://example.org/").expect("first ok");
        let err = pm.add("ex", "http://other.org/");
        assert!(matches!(err, Err(PrefixError::DuplicatePrefix(_))));
    }

    #[test]
    fn test_add_invalid_iri_error() {
        let mut pm = PrefixManager::new();
        let err = pm.add("ex", "");
        assert!(matches!(err, Err(PrefixError::InvalidIri(_))));
    }

    #[test]
    fn test_add_invalid_prefix_whitespace_error() {
        let mut pm = PrefixManager::new();
        let err = pm.add("ex foo", "http://example.org/");
        assert!(matches!(err, Err(PrefixError::InvalidPrefix(_))));
    }

    // --- add_or_update -------------------------------------------------------

    #[test]
    fn test_add_or_update_replaces_existing() {
        let mut pm = PrefixManager::new();
        pm.add_or_update("ex", "http://example.org/");
        pm.add_or_update("ex", "http://example.com/");
        assert_eq!(pm.get_iri("ex"), Some("http://example.com/"));
        // Count should still be 1
        assert_eq!(pm.prefix_count(), 1);
    }

    #[test]
    fn test_add_or_update_new_entry() {
        let mut pm = PrefixManager::new();
        pm.add_or_update("ex", "http://example.org/");
        assert_eq!(pm.prefix_count(), 1);
    }

    // --- remove -------------------------------------------------------------

    #[test]
    fn test_remove_existing() {
        let mut pm = PrefixManager::new();
        pm.add("ex", "http://example.org/").expect("ok");
        let removed = pm.remove("ex");
        assert!(removed.is_some());
        assert_eq!(pm.prefix_count(), 0);
        assert!(pm.get_iri("ex").is_none());
    }

    #[test]
    fn test_remove_non_existing_returns_none() {
        let mut pm = PrefixManager::new();
        assert!(pm.remove("nonexistent").is_none());
    }

    // --- get_iri / get_prefix -----------------------------------------------

    #[test]
    fn test_get_iri_found() {
        let pm = PrefixManager::with_defaults();
        assert_eq!(pm.get_iri("rdf"), Some(RDF_IRI));
    }

    #[test]
    fn test_get_iri_not_found() {
        let pm = PrefixManager::new();
        assert!(pm.get_iri("unknown").is_none());
    }

    #[test]
    fn test_get_prefix_found() {
        let pm = PrefixManager::with_defaults();
        assert_eq!(pm.get_prefix(RDF_IRI), Some("rdf"));
    }

    #[test]
    fn test_get_prefix_not_found() {
        let pm = PrefixManager::new();
        assert!(pm.get_prefix("http://nowhere.org/").is_none());
    }

    // --- abbreviate ---------------------------------------------------------

    #[test]
    fn test_abbreviate_rdf_type() {
        let pm = PrefixManager::with_defaults();
        let result = pm.abbreviate("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert!(result.is_abbreviated());
        assert_eq!(result.to_str(), "rdf:type");
    }

    #[test]
    fn test_abbreviate_no_match_returns_full_iri() {
        let pm = PrefixManager::with_defaults();
        let iri = "http://unknown.example.org/something";
        let result = pm.abbreviate(iri);
        assert!(!result.is_abbreviated());
        assert!(matches!(result, AbbrevResult::FullIri(_)));
    }

    #[test]
    fn test_abbreviate_empty_returns_invalid() {
        let pm = PrefixManager::with_defaults();
        assert_eq!(pm.abbreviate(""), AbbrevResult::Invalid);
    }

    #[test]
    fn test_abbreviate_longest_prefix_match() {
        // Register two namespaces where one is a prefix of the other
        let mut pm = PrefixManager::new();
        pm.add_or_update("ex", "http://example.org/");
        pm.add_or_update("ex2", "http://example.org/sub/");
        let result = pm.abbreviate("http://example.org/sub/foo");
        // Should match the longer namespace
        assert_eq!(result.to_str(), "ex2:foo");
    }

    #[test]
    fn test_abbreviate_schema_class() {
        let pm = PrefixManager::with_defaults();
        let result = pm.abbreviate("https://schema.org/Person");
        assert_eq!(result.to_str(), "schema:Person");
    }

    // --- expand -------------------------------------------------------------

    #[test]
    fn test_expand_rdf_type() {
        let pm = PrefixManager::with_defaults();
        let full = pm.expand("rdf:type").expect("ok");
        assert_eq!(full, format!("{}type", RDF_IRI));
    }

    #[test]
    fn test_expand_unknown_prefix_error() {
        let pm = PrefixManager::new();
        let err = pm.expand("unknown:foo");
        assert!(matches!(err, Err(PrefixError::PrefixNotFound(_))));
    }

    #[test]
    fn test_expand_no_colon_error() {
        let pm = PrefixManager::with_defaults();
        let err = pm.expand("rdfstype");
        assert!(matches!(err, Err(PrefixError::PrefixNotFound(_))));
    }

    #[test]
    fn test_expand_rdfs_label() {
        let pm = PrefixManager::with_defaults();
        let full = pm.expand("rdfs:label").expect("ok");
        assert_eq!(full, format!("{}label", RDFS_IRI));
    }

    // --- turtle_declarations -------------------------------------------------

    #[test]
    fn test_turtle_declarations_format() {
        let mut pm = PrefixManager::new();
        pm.add("ex", "http://example.org/").expect("ok");
        let decls = pm.turtle_declarations();
        assert!(decls.contains("@prefix ex: <http://example.org/> ."));
    }

    #[test]
    fn test_turtle_declarations_all_defaults() {
        let pm = PrefixManager::with_defaults();
        let decls = pm.turtle_declarations();
        assert!(decls.contains("@prefix rdf:"));
        assert!(decls.contains("@prefix rdfs:"));
        assert!(decls.contains("@prefix owl:"));
        assert!(decls.contains("@prefix xsd:"));
    }

    // --- sparql_declarations -------------------------------------------------

    #[test]
    fn test_sparql_declarations_format() {
        let mut pm = PrefixManager::new();
        pm.add("ex", "http://example.org/").expect("ok");
        let decls = pm.sparql_declarations();
        assert!(decls.contains("PREFIX ex: <http://example.org/>"));
    }

    #[test]
    fn test_sparql_declarations_all_defaults() {
        let pm = PrefixManager::with_defaults();
        let decls = pm.sparql_declarations();
        assert!(decls.contains("PREFIX rdf:"));
        assert!(decls.contains("PREFIX owl:"));
    }

    // --- contains_prefix / contains_iri -------------------------------------

    #[test]
    fn test_contains_prefix() {
        let pm = PrefixManager::with_defaults();
        assert!(pm.contains_prefix("rdf"));
        assert!(!pm.contains_prefix("dc"));
    }

    #[test]
    fn test_contains_iri() {
        let pm = PrefixManager::with_defaults();
        assert!(pm.contains_iri(RDF_IRI));
        assert!(!pm.contains_iri("http://purl.org/dc/elements/1.1/"));
    }

    // --- prefix_count -------------------------------------------------------

    #[test]
    fn test_prefix_count_default() {
        let pm = PrefixManager::with_defaults();
        assert_eq!(pm.prefix_count(), 6); // rdf, rdfs, owl, xsd, schema, skos
    }

    // --- clear ---------------------------------------------------------------

    #[test]
    fn test_clear_empties_manager() {
        let mut pm = PrefixManager::with_defaults();
        pm.clear();
        assert_eq!(pm.prefix_count(), 0);
        assert!(pm.all_namespaces().is_empty());
        assert!(pm.get_iri("rdf").is_none());
    }

    // --- all_namespaces insertion order -------------------------------------

    #[test]
    fn test_all_namespaces_insertion_order() {
        let mut pm = PrefixManager::new();
        pm.add_or_update("z", "http://z.org/");
        pm.add_or_update("a", "http://a.org/");
        let names: Vec<&str> = pm
            .all_namespaces()
            .iter()
            .map(|n| n.prefix.as_str())
            .collect();
        assert_eq!(names, vec!["z", "a"]);
    }

    // --- AbbrevResult display -----------------------------------------------

    #[test]
    fn test_abbrev_result_to_str_prefixed() {
        let r = AbbrevResult::PrefixedName {
            prefix: "rdf".to_string(),
            local: "type".to_string(),
        };
        assert_eq!(r.to_str(), "rdf:type");
    }

    #[test]
    fn test_abbrev_result_to_str_full_iri() {
        let r = AbbrevResult::FullIri("http://example.org/foo".to_string());
        assert_eq!(r.to_str(), "<http://example.org/foo>");
    }

    #[test]
    fn test_abbrev_result_to_str_invalid() {
        assert_eq!(AbbrevResult::Invalid.to_str(), "<invalid>");
    }

    // --- PrefixError display ------------------------------------------------

    #[test]
    fn test_prefix_error_display() {
        assert!(PrefixError::DuplicatePrefix("rdf".to_string())
            .to_string()
            .contains("rdf"));
        assert!(PrefixError::InvalidIri("".to_string())
            .to_string()
            .contains("IRI"));
        assert!(PrefixError::PrefixNotFound("dc".to_string())
            .to_string()
            .contains("dc"));
    }

    // --- round-trip abbreviate/expand ---------------------------------------

    #[test]
    fn test_round_trip_abbreviate_expand() {
        let pm = PrefixManager::with_defaults();
        let original = format!("{}Class", OWL_IRI);
        let abbrev = pm.abbreviate(&original);
        assert!(abbrev.is_abbreviated());
        if let AbbrevResult::PrefixedName { prefix, local } = abbrev {
            let expanded = pm.expand(&format!("{}:{}", prefix, local)).expect("ok");
            assert_eq!(expanded, original);
        }
    }
}
