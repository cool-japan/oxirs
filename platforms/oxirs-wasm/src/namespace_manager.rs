//! # Namespace Manager
//!
//! RDF namespace prefix management for WASM bindings.
//!
//! This module maintains a mapping of short prefixes to full IRI namespaces
//! (e.g. `rdf` → `http://www.w3.org/1999/02/22-rdf-syntax-ns#`) and provides
//! expand / compact utilities that are commonly needed when serialising or
//! parsing RDF data inside a WebAssembly environment.
//!
//! A set of well-known standard prefixes can be loaded via
//! [`NamespaceManager::with_standard_prefixes`].
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::namespace_manager::NamespaceManager;
//!
//! let mut mgr = NamespaceManager::with_standard_prefixes();
//! mgr.add_prefix("ex", "http://example.org/").unwrap();
//!
//! assert_eq!(
//!     mgr.expand("ex:Thing"),
//!     Some("http://example.org/Thing".to_string())
//! );
//! assert_eq!(
//!     mgr.compact("http://example.org/Thing"),
//!     Some("ex:Thing".to_string())
//! );
//! ```

use std::collections::HashMap;

// ─── Standard namespaces ──────────────────────────────────────────────────────

/// Well-known standard RDF namespace IRIs.
const NS_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const NS_RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
const NS_OWL: &str = "http://www.w3.org/2002/07/owl#";
const NS_XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const NS_FOAF: &str = "http://xmlns.com/foaf/0.1/";
const NS_SCHEMA: &str = "https://schema.org/";

// ─── PrefixBinding ────────────────────────────────────────────────────────────

/// A single namespace prefix-to-IRI binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixBinding {
    /// The short prefix string (e.g. `"rdf"`, `"ex"`).
    pub prefix: String,
    /// The full namespace IRI (e.g. `"http://www.w3.org/1999/02/22-rdf-syntax-ns#"`).
    pub namespace: String,
}

// ─── NamespaceManager ─────────────────────────────────────────────────────────

/// Manages RDF namespace prefix declarations.
///
/// Internally stores a `prefix → namespace` map.  All lookup operations are
/// O(n) in the number of registered prefixes; this is acceptable for the
/// typical use-case of a few dozen prefixes.
pub struct NamespaceManager {
    bindings: HashMap<String, String>,
}

impl NamespaceManager {
    /// Creates an empty [`NamespaceManager`] with no registered prefixes.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Creates a [`NamespaceManager`] pre-loaded with the six standard
    /// prefixes: `rdf`, `rdfs`, `owl`, `xsd`, `foaf`, `schema`.
    pub fn with_standard_prefixes() -> Self {
        let mut mgr = Self::new();
        // These are canonical namespaces; they cannot fail, so errors are swallowed.
        let _ = mgr.add_prefix("rdf", NS_RDF);
        let _ = mgr.add_prefix("rdfs", NS_RDFS);
        let _ = mgr.add_prefix("owl", NS_OWL);
        let _ = mgr.add_prefix("xsd", NS_XSD);
        let _ = mgr.add_prefix("foaf", NS_FOAF);
        let _ = mgr.add_prefix("schema", NS_SCHEMA);
        mgr
    }

    /// Registers a new prefix binding or **updates** an existing one.
    ///
    /// Returns `Err` when either `prefix` or `namespace` is empty.
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) -> Result<(), String> {
        if prefix.is_empty() {
            return Err("prefix must not be empty".to_string());
        }
        if namespace.is_empty() {
            return Err("namespace must not be empty".to_string());
        }
        self.bindings
            .insert(prefix.to_string(), namespace.to_string());
        Ok(())
    }

    /// Removes the binding for `prefix`.
    ///
    /// Returns `true` when a binding was removed, `false` when no such
    /// binding existed.
    pub fn remove_prefix(&mut self, prefix: &str) -> bool {
        self.bindings.remove(prefix).is_some()
    }

    /// Returns the namespace IRI for `prefix`, or `None` if unknown.
    pub fn resolve_prefix(&self, prefix: &str) -> Option<&str> {
        self.bindings.get(prefix).map(String::as_str)
    }

    /// Expands a CURIE of the form `"prefix:local"` to the full IRI.
    ///
    /// Returns `None` when:
    /// - the input contains no `':'` separator, or
    /// - the prefix part is not registered.
    pub fn expand(&self, curie: &str) -> Option<String> {
        let colon_pos = curie.find(':')?;
        let prefix = &curie[..colon_pos];
        let local = &curie[colon_pos + 1..];
        let ns = self.bindings.get(prefix)?;
        Some(format!("{}{}", ns, local))
    }

    /// Compacts a full IRI to a CURIE using the **longest** matching
    /// registered namespace prefix.
    ///
    /// Returns `None` when no registered namespace is a prefix of `iri`.
    pub fn compact(&self, iri: &str) -> Option<String> {
        // Find the binding whose namespace is the longest prefix of `iri`.
        let mut best: Option<(&str, &str)> = None; // (prefix, namespace)
        for (pfx, ns) in &self.bindings {
            if iri.starts_with(ns.as_str()) {
                let is_longer = best
                    .as_ref()
                    .map_or(true, |(_, best_ns)| ns.len() > best_ns.len());
                if is_longer {
                    best = Some((pfx.as_str(), ns.as_str()));
                }
            }
        }
        best.map(|(pfx, ns)| {
            let local = &iri[ns.len()..];
            format!("{}:{}", pfx, local)
        })
    }

    /// Returns all registered prefix bindings in an unspecified order.
    pub fn all_prefixes(&self) -> Vec<PrefixBinding> {
        self.bindings
            .iter()
            .map(|(p, n)| PrefixBinding {
                prefix: p.clone(),
                namespace: n.clone(),
            })
            .collect()
    }

    /// Returns the number of registered prefix bindings.
    pub fn prefix_count(&self) -> usize {
        self.bindings.len()
    }

    /// Generates Turtle `@prefix` declaration lines for all registered
    /// bindings, sorted alphabetically by prefix.
    ///
    /// Example output line: `@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .`
    pub fn to_turtle_prefixes(&self) -> String {
        let mut pairs: Vec<(&str, &str)> = self
            .bindings
            .iter()
            .map(|(p, n)| (p.as_str(), n.as_str()))
            .collect();
        pairs.sort_by_key(|(p, _)| *p);
        pairs
            .iter()
            .map(|(p, n)| format!("@prefix {}: <{}> .", p, n))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for NamespaceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── add_prefix ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_prefix_valid() {
        let mut mgr = NamespaceManager::new();
        assert!(mgr.add_prefix("ex", "http://example.org/").is_ok());
        assert_eq!(mgr.resolve_prefix("ex"), Some("http://example.org/"));
    }

    #[test]
    fn test_add_prefix_empty_prefix_error() {
        let mut mgr = NamespaceManager::new();
        assert!(mgr.add_prefix("", "http://example.org/").is_err());
    }

    #[test]
    fn test_add_prefix_empty_namespace_error() {
        let mut mgr = NamespaceManager::new();
        assert!(mgr.add_prefix("ex", "").is_err());
    }

    #[test]
    fn test_add_prefix_update_existing() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://old.org/").unwrap();
        mgr.add_prefix("ex", "http://new.org/").unwrap();
        assert_eq!(mgr.resolve_prefix("ex"), Some("http://new.org/"));
        assert_eq!(mgr.prefix_count(), 1, "update must not increase count");
    }

    #[test]
    fn test_add_multiple_prefixes() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("a", "http://a.org/").unwrap();
        mgr.add_prefix("b", "http://b.org/").unwrap();
        assert_eq!(mgr.prefix_count(), 2);
    }

    // ── remove_prefix ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_prefix_existing() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        assert!(mgr.remove_prefix("ex"));
        assert!(mgr.resolve_prefix("ex").is_none());
    }

    #[test]
    fn test_remove_prefix_nonexistent() {
        let mut mgr = NamespaceManager::new();
        assert!(!mgr.remove_prefix("nonexistent"));
    }

    #[test]
    fn test_remove_prefix_decrements_count() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        mgr.remove_prefix("ex");
        assert_eq!(mgr.prefix_count(), 0);
    }

    // ── resolve_prefix ────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_prefix_known() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("foaf", NS_FOAF).unwrap();
        assert_eq!(mgr.resolve_prefix("foaf"), Some(NS_FOAF));
    }

    #[test]
    fn test_resolve_prefix_unknown() {
        let mgr = NamespaceManager::new();
        assert!(mgr.resolve_prefix("unknown").is_none());
    }

    // ── expand ────────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_valid_curie() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        assert_eq!(
            mgr.expand("ex:Person"),
            Some("http://example.org/Person".to_string())
        );
    }

    #[test]
    fn test_expand_empty_local_name() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        assert_eq!(mgr.expand("ex:"), Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_expand_unknown_prefix() {
        let mgr = NamespaceManager::new();
        assert!(mgr.expand("ex:Person").is_none());
    }

    #[test]
    fn test_expand_no_colon_returns_none() {
        let mgr = NamespaceManager::new();
        assert!(mgr.expand("nocolon").is_none());
    }

    #[test]
    fn test_expand_rdf_type() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.expand("rdf:type"), Some(format!("{}type", NS_RDF)));
    }

    // ── compact ───────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_known_iri() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        assert_eq!(
            mgr.compact("http://example.org/Thing"),
            Some("ex:Thing".to_string())
        );
    }

    #[test]
    fn test_compact_unknown_namespace() {
        let mgr = NamespaceManager::new();
        assert!(mgr.compact("http://unknown.org/Thing").is_none());
    }

    #[test]
    fn test_compact_longest_prefix_wins() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("base", "http://example.org/").unwrap();
        mgr.add_prefix("sub", "http://example.org/sub/").unwrap();
        // Should use "sub" (longer match) not "base"
        let result = mgr.compact("http://example.org/sub/Thing");
        assert_eq!(result, Some("sub:Thing".to_string()));
    }

    #[test]
    fn test_compact_rdfs_iri() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(
            mgr.compact(&format!("{}label", NS_RDFS)),
            Some("rdfs:label".to_string())
        );
    }

    // ── with_standard_prefixes ────────────────────────────────────────────────

    #[test]
    fn test_standard_prefixes_has_rdf() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("rdf"), Some(NS_RDF));
    }

    #[test]
    fn test_standard_prefixes_has_rdfs() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("rdfs"), Some(NS_RDFS));
    }

    #[test]
    fn test_standard_prefixes_has_owl() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("owl"), Some(NS_OWL));
    }

    #[test]
    fn test_standard_prefixes_has_xsd() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("xsd"), Some(NS_XSD));
    }

    #[test]
    fn test_standard_prefixes_has_foaf() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("foaf"), Some(NS_FOAF));
    }

    #[test]
    fn test_standard_prefixes_has_schema() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.resolve_prefix("schema"), Some(NS_SCHEMA));
    }

    #[test]
    fn test_standard_prefixes_count() {
        let mgr = NamespaceManager::with_standard_prefixes();
        assert_eq!(mgr.prefix_count(), 6);
    }

    // ── all_prefixes ──────────────────────────────────────────────────────────

    #[test]
    fn test_all_prefixes_empty() {
        let mgr = NamespaceManager::new();
        assert!(mgr.all_prefixes().is_empty());
    }

    #[test]
    fn test_all_prefixes_returns_all() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("a", "http://a.org/").unwrap();
        mgr.add_prefix("b", "http://b.org/").unwrap();
        let all = mgr.all_prefixes();
        assert_eq!(all.len(), 2);
        let prefixes: Vec<_> = all.iter().map(|b| b.prefix.as_str()).collect();
        assert!(prefixes.contains(&"a"));
        assert!(prefixes.contains(&"b"));
    }

    #[test]
    fn test_all_prefixes_correct_namespaces() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        let all = mgr.all_prefixes();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].namespace, "http://example.org/");
    }

    // ── prefix_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_prefix_count_zero_initially() {
        assert_eq!(NamespaceManager::new().prefix_count(), 0);
    }

    #[test]
    fn test_prefix_count_after_add() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("p1", "http://p1.org/").unwrap();
        mgr.add_prefix("p2", "http://p2.org/").unwrap();
        assert_eq!(mgr.prefix_count(), 2);
    }

    // ── to_turtle_prefixes ────────────────────────────────────────────────────

    #[test]
    fn test_to_turtle_prefixes_contains_at_prefix() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        let turtle = mgr.to_turtle_prefixes();
        assert!(turtle.contains("@prefix"));
    }

    #[test]
    fn test_to_turtle_prefixes_format() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("ex", "http://example.org/").unwrap();
        let turtle = mgr.to_turtle_prefixes();
        assert!(turtle.contains("@prefix ex: <http://example.org/> ."));
    }

    #[test]
    fn test_to_turtle_prefixes_empty() {
        let mgr = NamespaceManager::new();
        assert_eq!(mgr.to_turtle_prefixes(), "");
    }

    #[test]
    fn test_to_turtle_prefixes_sorted_alphabetically() {
        let mut mgr = NamespaceManager::new();
        mgr.add_prefix("z", "http://z.org/").unwrap();
        mgr.add_prefix("a", "http://a.org/").unwrap();
        mgr.add_prefix("m", "http://m.org/").unwrap();
        let turtle = mgr.to_turtle_prefixes();
        let lines: Vec<&str> = turtle.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("@prefix a:"));
        assert!(lines[1].contains("@prefix m:"));
        assert!(lines[2].contains("@prefix z:"));
    }

    #[test]
    fn test_to_turtle_prefixes_standard_contains_rdf() {
        let mgr = NamespaceManager::with_standard_prefixes();
        let turtle = mgr.to_turtle_prefixes();
        assert!(turtle.contains(&format!("@prefix rdf: <{}> .", NS_RDF)));
    }

    // ── Default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let mgr = NamespaceManager::default();
        assert_eq!(mgr.prefix_count(), 0);
    }

    // ── PrefixBinding equality ────────────────────────────────────────────────

    #[test]
    fn test_prefix_binding_equality() {
        let a = PrefixBinding {
            prefix: "ex".to_string(),
            namespace: "http://example.org/".to_string(),
        };
        let b = PrefixBinding {
            prefix: "ex".to_string(),
            namespace: "http://example.org/".to_string(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefix_binding_inequality() {
        let a = PrefixBinding {
            prefix: "ex".to_string(),
            namespace: "http://example.org/".to_string(),
        };
        let b = PrefixBinding {
            prefix: "ex".to_string(),
            namespace: "http://other.org/".to_string(),
        };
        assert_ne!(a, b);
    }
}
