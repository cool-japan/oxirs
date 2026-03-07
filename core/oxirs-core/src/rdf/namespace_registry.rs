//! Global namespace/prefix registry with IRI expansion and compression.
//!
//! Provides a bidirectional mapping between namespace prefixes (e.g., `rdf`)
//! and their full IRI bases (e.g., `http://www.w3.org/1999/02/22-rdf-syntax-ns#`).

use std::collections::HashMap;

/// A single namespace entry mapping a prefix to an IRI base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceEntry {
    /// The short prefix (e.g., `"rdf"`).
    pub prefix: String,
    /// The full IRI base (e.g., `"http://www.w3.org/1999/02/22-rdf-syntax-ns#"`).
    pub iri: String,
}

/// Well-known default namespace prefixes pre-registered via [`NamespaceRegistry::with_defaults`].
pub const DEFAULT_PREFIXES: &[(&str, &str)] = &[
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("schema", "https://schema.org/"),
];

/// Bidirectional registry mapping namespace prefixes to IRI bases and back.
///
/// # Example
/// ```
/// use oxirs_core::rdf::namespace_registry::NamespaceRegistry;
///
/// let mut reg = NamespaceRegistry::with_defaults();
/// assert_eq!(
///     reg.expand("rdf:type"),
///     Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
/// );
/// ```
#[derive(Debug, Clone, Default)]
pub struct NamespaceRegistry {
    prefix_to_iri: HashMap<String, String>,
    iri_to_prefix: HashMap<String, String>,
    /// Ordered list of entries for `all_entries`.
    entries: Vec<NamespaceEntry>,
}

impl NamespaceRegistry {
    /// Create an empty registry with no predefined prefixes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with the seven standard prefixes:
    /// `rdf`, `rdfs`, `owl`, `xsd`, `dc`, `foaf`, `schema`.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        for (prefix, iri) in DEFAULT_PREFIXES {
            reg.register_or_replace(prefix, iri);
        }
        reg
    }

    /// Register a new prefix→IRI mapping.
    ///
    /// Returns `false` and leaves the registry unchanged if the prefix already
    /// exists.  Returns `true` on success.
    pub fn register(&mut self, prefix: impl Into<String>, iri: impl Into<String>) -> bool {
        let prefix = prefix.into();
        let iri = iri.into();
        if self.prefix_to_iri.contains_key(&prefix) {
            return false;
        }
        self.iri_to_prefix.insert(iri.clone(), prefix.clone());
        self.entries.push(NamespaceEntry {
            prefix: prefix.clone(),
            iri: iri.clone(),
        });
        self.prefix_to_iri.insert(prefix, iri);
        true
    }

    /// Register or overwrite a prefix→IRI mapping unconditionally.
    pub fn register_or_replace(&mut self, prefix: &str, iri: &str) {
        // Remove old reverse mapping if the prefix existed before.
        if let Some(old_iri) = self.prefix_to_iri.get(prefix) {
            let old_iri = old_iri.clone();
            self.iri_to_prefix.remove(&old_iri);
            // Remove old entry from ordered list.
            self.entries.retain(|e| e.prefix != prefix);
        }
        self.prefix_to_iri
            .insert(prefix.to_string(), iri.to_string());
        self.iri_to_prefix
            .insert(iri.to_string(), prefix.to_string());
        self.entries.push(NamespaceEntry {
            prefix: prefix.to_string(),
            iri: iri.to_string(),
        });
    }

    /// Expand a prefixed name such as `"rdf:type"` into its full IRI string.
    ///
    /// Returns `None` if the prefix is not registered or the input does not
    /// contain a `:` separator.
    pub fn expand(&self, prefixed: &str) -> Option<String> {
        let colon = prefixed.find(':')?;
        let prefix = &prefixed[..colon];
        let local = &prefixed[colon + 1..];
        let base = self.prefix_to_iri.get(prefix)?;
        Some(format!("{base}{local}"))
    }

    /// Compress a full IRI string into a prefixed name such as `"rdf:type"`.
    ///
    /// Returns `None` if no registered IRI base is a prefix of `iri`.
    pub fn compress(&self, iri: &str) -> Option<String> {
        // Find the longest matching IRI base.
        let mut best: Option<(&str, &str)> = None;
        for (base, prefix) in &self.iri_to_prefix {
            if iri.starts_with(base.as_str()) {
                match best {
                    None => best = Some((base.as_str(), prefix.as_str())),
                    Some((b, _)) if base.len() > b.len() => {
                        best = Some((base.as_str(), prefix.as_str()));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(base, prefix)| {
            let local = &iri[base.len()..];
            format!("{prefix}:{local}")
        })
    }

    /// Return the prefix registered for an IRI base, if any.
    pub fn prefix_for(&self, iri: &str) -> Option<&str> {
        self.iri_to_prefix.get(iri).map(String::as_str)
    }

    /// Return the IRI base registered for a prefix, if any.
    pub fn iri_for(&self, prefix: &str) -> Option<&str> {
        self.prefix_to_iri.get(prefix).map(String::as_str)
    }

    /// Return all registered entries in insertion order.
    pub fn all_entries(&self) -> Vec<&NamespaceEntry> {
        self.entries.iter().collect()
    }

    /// Remove the mapping for a prefix.
    ///
    /// Returns `true` if the prefix existed and was removed.
    pub fn remove(&mut self, prefix: &str) -> bool {
        if let Some(iri) = self.prefix_to_iri.remove(prefix) {
            self.iri_to_prefix.remove(&iri);
            self.entries.retain(|e| e.prefix != prefix);
            true
        } else {
            false
        }
    }

    /// Total number of registered prefix→IRI mappings.
    pub fn count(&self) -> usize {
        self.prefix_to_iri.len()
    }

    /// Merge all entries from `other` into this registry.
    ///
    /// Entries in `other` whose prefix already exists in `self` are **skipped**
    /// (non-destructive merge).  Use `register_or_replace` manually for
    /// overwrite semantics.
    pub fn merge(&mut self, other: &NamespaceRegistry) {
        for entry in &other.entries {
            self.register(entry.prefix.clone(), entry.iri.clone());
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rdf_base() -> &'static str {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    }
    fn rdfs_base() -> &'static str {
        "http://www.w3.org/2000/01/rdf-schema#"
    }
    fn owl_base() -> &'static str {
        "http://www.w3.org/2002/07/owl#"
    }
    fn xsd_base() -> &'static str {
        "http://www.w3.org/2001/XMLSchema#"
    }

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let reg = NamespaceRegistry::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_with_defaults_count() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.count(), DEFAULT_PREFIXES.len());
    }

    #[test]
    fn test_with_defaults_rdf() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("rdf"), Some(rdf_base()));
    }

    #[test]
    fn test_with_defaults_rdfs() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("rdfs"), Some(rdfs_base()));
    }

    #[test]
    fn test_with_defaults_owl() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("owl"), Some(owl_base()));
    }

    #[test]
    fn test_with_defaults_xsd() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("xsd"), Some(xsd_base()));
    }

    #[test]
    fn test_with_defaults_dc() {
        let reg = NamespaceRegistry::with_defaults();
        assert!(reg.iri_for("dc").is_some());
    }

    #[test]
    fn test_with_defaults_foaf() {
        let reg = NamespaceRegistry::with_defaults();
        assert!(reg.iri_for("foaf").is_some());
    }

    #[test]
    fn test_with_defaults_schema() {
        let reg = NamespaceRegistry::with_defaults();
        assert!(reg.iri_for("schema").is_some());
    }

    // ── register ──────────────────────────────────────────────────────────────

    #[test]
    fn test_register_success() {
        let mut reg = NamespaceRegistry::new();
        assert!(reg.register("ex", "http://example.org/"));
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_register_duplicate_returns_false() {
        let mut reg = NamespaceRegistry::new();
        assert!(reg.register("ex", "http://example.org/"));
        assert!(!reg.register("ex", "http://other.org/"));
        assert_eq!(reg.count(), 1);
        // Original IRI must remain.
        assert_eq!(reg.iri_for("ex"), Some("http://example.org/"));
    }

    #[test]
    fn test_register_or_replace_new() {
        let mut reg = NamespaceRegistry::new();
        reg.register_or_replace("ex", "http://example.org/");
        assert_eq!(reg.iri_for("ex"), Some("http://example.org/"));
    }

    #[test]
    fn test_register_or_replace_overwrites() {
        let mut reg = NamespaceRegistry::new();
        reg.register_or_replace("ex", "http://example.org/");
        reg.register_or_replace("ex", "http://updated.org/");
        assert_eq!(reg.iri_for("ex"), Some("http://updated.org/"));
        assert_eq!(reg.count(), 1);
    }

    // ── expand ────────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_rdf_type() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.expand("rdf:type"), Some(format!("{}type", rdf_base())));
    }

    #[test]
    fn test_expand_xsd_string() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(
            reg.expand("xsd:string"),
            Some(format!("{}string", xsd_base()))
        );
    }

    #[test]
    fn test_expand_unknown_prefix_returns_none() {
        let reg = NamespaceRegistry::new();
        assert_eq!(reg.expand("ex:foo"), None);
    }

    #[test]
    fn test_expand_no_colon_returns_none() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.expand("rdftype"), None);
    }

    #[test]
    fn test_expand_owl_class() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(
            reg.expand("owl:Class"),
            Some(format!("{}Class", owl_base()))
        );
    }

    // ── compress ──────────────────────────────────────────────────────────────

    #[test]
    fn test_compress_rdf_type() {
        let reg = NamespaceRegistry::with_defaults();
        let iri = format!("{}type", rdf_base());
        assert_eq!(reg.compress(&iri), Some("rdf:type".to_string()));
    }

    #[test]
    fn test_compress_rdfs_label() {
        let reg = NamespaceRegistry::with_defaults();
        let iri = format!("{}label", rdfs_base());
        assert_eq!(reg.compress(&iri), Some("rdfs:label".to_string()));
    }

    #[test]
    fn test_compress_unknown_iri_returns_none() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.compress("http://unknown.example/foo"), None);
    }

    #[test]
    fn test_compress_custom_prefix() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/");
        assert_eq!(
            reg.compress("http://example.org/Person"),
            Some("ex:Person".to_string())
        );
    }

    #[test]
    fn test_compress_longest_base_wins() {
        // Register two overlapping bases; compression must pick the longer one.
        let mut reg = NamespaceRegistry::new();
        reg.register("a", "http://example.org/");
        reg.register("b", "http://example.org/ns/");
        assert_eq!(
            reg.compress("http://example.org/ns/Foo"),
            Some("b:Foo".to_string())
        );
    }

    // ── prefix_for / iri_for ──────────────────────────────────────────────────

    #[test]
    fn test_prefix_for_known() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.prefix_for(rdf_base()), Some("rdf"));
    }

    #[test]
    fn test_prefix_for_unknown() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.prefix_for("http://nothing.example/"), None);
    }

    #[test]
    fn test_iri_for_known() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("rdfs"), Some(rdfs_base()));
    }

    #[test]
    fn test_iri_for_unknown() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.iri_for("notregistered"), None);
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing() {
        let mut reg = NamespaceRegistry::with_defaults();
        let before = reg.count();
        assert!(reg.remove("rdf"));
        assert_eq!(reg.count(), before - 1);
    }

    #[test]
    fn test_remove_clears_reverse_lookup() {
        let mut reg = NamespaceRegistry::with_defaults();
        reg.remove("rdf");
        assert_eq!(reg.prefix_for(rdf_base()), None);
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut reg = NamespaceRegistry::with_defaults();
        assert!(!reg.remove("nonexistent"));
    }

    #[test]
    fn test_remove_then_re_register() {
        let mut reg = NamespaceRegistry::with_defaults();
        reg.remove("rdf");
        assert!(reg.register("rdf", "http://example.org/new-rdf#"));
        assert_eq!(reg.iri_for("rdf"), Some("http://example.org/new-rdf#"));
    }

    // ── all_entries ───────────────────────────────────────────────────────────

    #[test]
    fn test_all_entries_count_matches() {
        let reg = NamespaceRegistry::with_defaults();
        assert_eq!(reg.all_entries().len(), reg.count());
    }

    #[test]
    fn test_all_entries_insertion_order() {
        let mut reg = NamespaceRegistry::new();
        reg.register("a", "http://a.example/");
        reg.register("b", "http://b.example/");
        reg.register("c", "http://c.example/");
        let entries = reg.all_entries();
        assert_eq!(entries[0].prefix, "a");
        assert_eq!(entries[1].prefix, "b");
        assert_eq!(entries[2].prefix, "c");
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_adds_new_entries() {
        let mut a = NamespaceRegistry::new();
        a.register("ex1", "http://ex1.org/");
        let mut b = NamespaceRegistry::new();
        b.register("ex2", "http://ex2.org/");
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert!(a.iri_for("ex2").is_some());
    }

    #[test]
    fn test_merge_does_not_overwrite_existing() {
        let mut a = NamespaceRegistry::new();
        a.register("ex", "http://original.org/");
        let mut b = NamespaceRegistry::new();
        b.register("ex", "http://other.org/");
        a.merge(&b);
        // Original must survive.
        assert_eq!(a.iri_for("ex"), Some("http://original.org/"));
    }

    #[test]
    fn test_merge_with_defaults() {
        let mut a = NamespaceRegistry::new();
        let defaults = NamespaceRegistry::with_defaults();
        a.merge(&defaults);
        assert_eq!(a.count(), DEFAULT_PREFIXES.len());
    }

    #[test]
    fn test_merge_empty_other_is_noop() {
        let mut a = NamespaceRegistry::with_defaults();
        let before = a.count();
        let empty = NamespaceRegistry::new();
        a.merge(&empty);
        assert_eq!(a.count(), before);
    }

    // ── roundtrip ─────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_compress_roundtrip() {
        let reg = NamespaceRegistry::with_defaults();
        let prefixed = "owl:Class";
        let expanded = reg.expand(prefixed).expect("expand failed");
        let compressed = reg.compress(&expanded).expect("compress failed");
        assert_eq!(compressed, prefixed);
    }

    #[test]
    fn test_custom_prefix_roundtrip() {
        let mut reg = NamespaceRegistry::new();
        reg.register("my", "http://my.example.org/vocab#");
        let expanded = reg.expand("my:Term").expect("expand");
        let compressed = reg.compress(&expanded).expect("compress");
        assert_eq!(compressed, "my:Term");
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_empty_local_name() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/");
        assert_eq!(reg.expand("ex:"), Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_count_reflects_adds_and_removes() {
        let mut reg = NamespaceRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.register("a", "http://a/");
        assert_eq!(reg.count(), 1);
        reg.register("b", "http://b/");
        assert_eq!(reg.count(), 2);
        reg.remove("a");
        assert_eq!(reg.count(), 1);
    }

    // ── Extra edge-case tests ─────────────────────────────────────────────────

    #[test]
    fn test_namespace_entry_fields() {
        let e = NamespaceEntry {
            prefix: "rdf".to_string(),
            iri: "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        };
        assert_eq!(e.prefix, "rdf");
        assert_eq!(e.iri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
    }

    #[test]
    fn test_default_prefixes_len() {
        assert_eq!(DEFAULT_PREFIXES.len(), 7);
    }

    #[test]
    fn test_register_or_replace_count_stays_same() {
        let mut reg = NamespaceRegistry::new();
        reg.register("x", "http://x.org/");
        reg.register_or_replace("x", "http://x2.org/");
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_all_entries_empty_registry() {
        let reg = NamespaceRegistry::new();
        assert!(reg.all_entries().is_empty());
    }

    #[test]
    fn test_merge_partial_overlap() {
        let mut a = NamespaceRegistry::new();
        a.register("p1", "http://p1/");
        let mut b = NamespaceRegistry::new();
        b.register("p1", "http://other/");
        b.register("p2", "http://p2/");
        a.merge(&b);
        // p1 should keep original; p2 gets added.
        assert_eq!(a.iri_for("p1"), Some("http://p1/"));
        assert_eq!(a.iri_for("p2"), Some("http://p2/"));
        assert_eq!(a.count(), 2);
    }
}
