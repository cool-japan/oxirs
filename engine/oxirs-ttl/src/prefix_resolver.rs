/// Prefix/CURIE resolver for Turtle and TriG documents.
///
/// Manages `@prefix` and `@base` declarations and resolves both CURIE
/// (prefixed name) and relative IRI references to full absolute IRIs.
use std::collections::HashMap;

/// A single prefix declaration with source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixDeclaration {
    /// The prefix label (e.g. "ex", "rdf").
    pub prefix: String,
    /// The namespace IRI (e.g. `"http://www.w3.org/1999/02/22-rdf-syntax-ns#"`).
    pub namespace: String,
    /// The line number where the declaration appeared (1-based).
    pub line: usize,
}

/// How an IRI was resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveSource {
    /// Resolved via a prefix declaration (carries the prefix label).
    Prefix(String),
    /// Resolved against the base IRI.
    Base,
    /// The term was already an absolute IRI.
    Absolute,
}

/// The outcome of resolving a term.
#[derive(Debug, Clone)]
pub struct ResolveResult {
    /// The fully resolved IRI.
    pub iri: String,
    /// How it was resolved.
    pub source: ResolveSource,
}

/// Errors produced by prefix resolution.
#[derive(Debug, Clone)]
pub enum ResolveError {
    /// The prefix is not declared.
    UnknownPrefix(String),
    /// No base IRI is set and a relative IRI was encountered.
    NoBaseIri,
    /// The CURIE syntax is invalid (e.g. missing colon).
    InvalidCurie(String),
    /// The IRI structure is invalid.
    InvalidIri(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPrefix(p) => write!(f, "Unknown prefix: {p}"),
            Self::NoBaseIri => write!(f, "No base IRI set"),
            Self::InvalidCurie(c) => write!(f, "Invalid CURIE: {c}"),
            Self::InvalidIri(i) => write!(f, "Invalid IRI: {i}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// A prefix/CURIE resolver for Turtle documents.
#[derive(Debug, Default)]
pub struct PrefixResolver {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
    declarations: Vec<PrefixDeclaration>,
}

impl PrefixResolver {
    /// Create an empty resolver.
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            declarations: Vec::new(),
        }
    }

    /// Create a resolver with a pre-set base IRI.
    pub fn with_base(base_iri: impl Into<String>) -> Self {
        let base = base_iri.into();
        Self {
            prefixes: HashMap::new(),
            base_iri: Some(base),
            declarations: Vec::new(),
        }
    }

    /// Add a prefix declaration.
    ///
    /// Does not treat re-declarations as errors; the latest wins.
    pub fn add_prefix(
        &mut self,
        prefix: impl Into<String>,
        namespace: impl Into<String>,
        line: usize,
    ) -> Result<(), ResolveError> {
        let prefix = prefix.into();
        let namespace = namespace.into();
        self.prefixes.insert(prefix.clone(), namespace.clone());
        self.declarations.push(PrefixDeclaration {
            prefix,
            namespace,
            line,
        });
        Ok(())
    }

    /// Update (or set) the base IRI.
    pub fn set_base(&mut self, base_iri: impl Into<String>) {
        self.base_iri = Some(base_iri.into());
    }

    /// Resolve a term that is either a CURIE (`prefix:local`) or a relative IRI.
    ///
    /// Absolute IRIs (with a scheme like `http://`) are passed through unchanged.
    /// Registered prefix names are resolved via their namespace.
    /// Unregistered prefix-like names are treated as unknown prefix errors.
    pub fn resolve(&self, term: &str) -> Result<ResolveResult, ResolveError> {
        // Check for CURIE pattern first (prefix:local where prefix is registered).
        if let Some(colon) = term.find(':') {
            let prefix = &term[..colon];
            let after_colon = &term[colon + 1..];

            // A registered prefix always wins.
            if !prefix.is_empty() && self.prefixes.contains_key(prefix) {
                let ns = &self.prefixes[prefix];
                return Ok(ResolveResult {
                    iri: format!("{ns}{after_colon}"),
                    source: ResolveSource::Prefix(prefix.to_string()),
                });
            }

            // If the part after ':' starts with '//', this is an absolute IRI (http://, ftp://, etc.)
            if after_colon.starts_with("//") {
                return Ok(ResolveResult {
                    iri: term.to_string(),
                    source: ResolveSource::Absolute,
                });
            }

            // Known absolute URI schemes without authority (urn:, mailto:, data:, file:, etc.)
            if !prefix.is_empty() && Self::is_known_scheme(prefix) {
                return Ok(ResolveResult {
                    iri: term.to_string(),
                    source: ResolveSource::Absolute,
                });
            }

            // Non-empty, non-registered prefix — report as unknown.
            if !prefix.is_empty() {
                return Err(ResolveError::UnknownPrefix(prefix.to_string()));
            }
        } else if Self::is_absolute_iri(term) {
            return Ok(ResolveResult {
                iri: term.to_string(),
                source: ResolveSource::Absolute,
            });
        }

        // Treat as relative IRI.
        self.resolve_relative(term).map(|iri| ResolveResult {
            iri,
            source: ResolveSource::Base,
        })
    }

    /// Resolve a CURIE string (`prefix:local`) into a full IRI.
    pub fn resolve_curie(&self, curie: &str) -> Result<String, ResolveError> {
        let colon = curie
            .find(':')
            .ok_or_else(|| ResolveError::InvalidCurie(curie.to_string()))?;
        let prefix = &curie[..colon];
        let local = &curie[colon + 1..];
        let ns = self
            .prefixes
            .get(prefix)
            .ok_or_else(|| ResolveError::UnknownPrefix(prefix.to_string()))?;
        Ok(format!("{ns}{local}"))
    }

    /// Resolve a relative IRI against the current base IRI.
    pub fn resolve_relative(&self, relative: &str) -> Result<String, ResolveError> {
        let base = self.base_iri.as_deref().ok_or(ResolveError::NoBaseIri)?;

        if relative.is_empty() {
            return Ok(base.to_string());
        }

        // Absolute fragment reference
        if relative.starts_with('#') {
            return Ok(format!("{base}{relative}"));
        }

        // Strip base to its document root (last '/') then join
        let base_path = if let Some(idx) = base.rfind('/') {
            &base[..=idx]
        } else {
            base
        };
        Ok(format!("{base_path}{relative}"))
    }

    /// Try to abbreviate a full IRI back to `prefix:local` using the registered prefixes.
    /// Returns `None` if no matching prefix is found.
    pub fn abbreviate(&self, iri: &str) -> Option<String> {
        // Find the longest matching namespace
        let mut best: Option<(&str, &str)> = None;
        for (prefix, ns) in &self.prefixes {
            if iri.starts_with(ns.as_str())
                && best.map_or(true, |(_, best_ns)| ns.len() > best_ns.len())
            {
                best = Some((prefix.as_str(), ns.as_str()));
            }
        }
        best.map(|(prefix, ns)| {
            let local = &iri[ns.len()..];
            format!("{prefix}:{local}")
        })
    }

    /// Return the number of declared prefixes (most recent per label).
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }

    /// Return true if a base IRI is set.
    pub fn has_base(&self) -> bool {
        self.base_iri.is_some()
    }

    /// Return all declarations in declaration order (may include historical re-declarations).
    pub fn declarations(&self) -> &[PrefixDeclaration] {
        &self.declarations
    }

    /// Return true if the string has a URI scheme (e.g. `http://`, `https://`, `urn:`, `ftp://`).
    ///
    /// Uses heuristics: the scheme must be followed by `//` (hierarchy) or be a known
    /// hierarchical/non-hierarchical scheme. CURIEs like `ex:Foo` are NOT absolute IRIs.
    pub fn is_absolute_iri(s: &str) -> bool {
        if s.contains(' ') {
            return false;
        }
        if let Some(colon_pos) = s.find(':') {
            let scheme = &s[..colon_pos];
            let after = &s[colon_pos + 1..];
            if scheme.is_empty() {
                return false;
            }
            let scheme_chars_valid = scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.');
            if !scheme_chars_valid {
                return false;
            }
            // Authority-based IRIs (http://, ftp://, etc.)
            if after.starts_with("//") {
                return true;
            }
            // Known schemeless-authority schemes
            Self::is_known_scheme(scheme)
        } else {
            false
        }
    }

    /// Return true if `scheme` is a well-known absolute URI scheme that does not use `://`.
    fn is_known_scheme(scheme: &str) -> bool {
        matches!(
            scheme.to_ascii_lowercase().as_str(),
            "urn"
                | "mailto"
                | "data"
                | "file"
                | "tel"
                | "fax"
                | "news"
                | "http"
                | "https"
                | "ftp"
                | "ftps"
                | "ldap"
                | "ldaps"
                | "irc"
                | "ircs"
                | "xmpp"
                | "sip"
                | "sips"
                | "coap"
                | "coaps"
                | "ws"
                | "wss"
                | "urn:ietf"
                | "tag"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- add_prefix / resolve_curie ---

    #[test]
    fn test_add_prefix_and_resolve_curie() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 1)
            .expect("should succeed");
        assert_eq!(
            r.resolve_curie("ex:Person").expect("should succeed"),
            "http://example.org/Person"
        );
    }

    #[test]
    fn test_resolve_curie_unknown_prefix() {
        let r = PrefixResolver::new();
        let err = r.resolve_curie("ex:Thing");
        assert!(matches!(err, Err(ResolveError::UnknownPrefix(_))));
    }

    #[test]
    fn test_resolve_curie_no_colon() {
        let r = PrefixResolver::new();
        let err = r.resolve_curie("nocolon");
        assert!(matches!(err, Err(ResolveError::InvalidCurie(_))));
    }

    #[test]
    fn test_resolve_curie_empty_local() {
        let mut r = PrefixResolver::new();
        r.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", 1)
            .expect("should succeed");
        assert_eq!(
            r.resolve_curie("rdf:").expect("should succeed"),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        );
    }

    // --- resolve ---

    #[test]
    fn test_resolve_absolute_iri_passthrough() {
        let r = PrefixResolver::new();
        let result = r.resolve("http://example.org/foo").expect("should succeed");
        assert_eq!(result.iri, "http://example.org/foo");
        assert_eq!(result.source, ResolveSource::Absolute);
    }

    #[test]
    fn test_resolve_curie_via_resolve() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 1)
            .expect("should succeed");
        let result = r.resolve("ex:Cat").expect("should succeed");
        assert_eq!(result.iri, "http://example.org/Cat");
        assert!(matches!(result.source, ResolveSource::Prefix(_)));
    }

    #[test]
    fn test_resolve_relative_via_resolve() {
        let r = PrefixResolver::with_base("http://example.org/doc/page.ttl");
        let result = r.resolve("other.ttl").expect("should succeed");
        assert_eq!(result.iri, "http://example.org/doc/other.ttl");
        assert_eq!(result.source, ResolveSource::Base);
    }

    // --- resolve_relative ---

    #[test]
    fn test_resolve_relative_simple() {
        let r = PrefixResolver::with_base("http://example.org/base/");
        let iri = r.resolve_relative("foo").expect("should succeed");
        assert_eq!(iri, "http://example.org/base/foo");
    }

    #[test]
    fn test_resolve_relative_no_base() {
        let r = PrefixResolver::new();
        let err = r.resolve_relative("foo");
        assert!(matches!(err, Err(ResolveError::NoBaseIri)));
    }

    #[test]
    fn test_resolve_relative_fragment() {
        let r = PrefixResolver::with_base("http://example.org/ont");
        let iri = r.resolve_relative("#Alice").expect("should succeed");
        assert_eq!(iri, "http://example.org/ont#Alice");
    }

    #[test]
    fn test_resolve_relative_empty_string_returns_base() {
        let r = PrefixResolver::with_base("http://example.org/doc");
        let iri = r.resolve_relative("").expect("should succeed");
        assert_eq!(iri, "http://example.org/doc");
    }

    // --- set_base ---

    #[test]
    fn test_set_base_updates() {
        let mut r = PrefixResolver::new();
        r.set_base("http://first.org/");
        r.set_base("http://second.org/");
        let iri = r.resolve_relative("x").expect("should succeed");
        assert!(iri.contains("second.org"));
    }

    #[test]
    fn test_has_base_false() {
        let r = PrefixResolver::new();
        assert!(!r.has_base());
    }

    #[test]
    fn test_has_base_true() {
        let r = PrefixResolver::with_base("http://base.org/");
        assert!(r.has_base());
    }

    // --- abbreviate ---

    #[test]
    fn test_abbreviate_success() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 1)
            .expect("should succeed");
        let abbrev = r.abbreviate("http://example.org/Person");
        assert_eq!(abbrev.as_deref(), Some("ex:Person"));
    }

    #[test]
    fn test_abbreviate_no_match() {
        let r = PrefixResolver::new();
        assert!(r.abbreviate("http://unknown.org/foo").is_none());
    }

    #[test]
    fn test_abbreviate_prefers_longest_prefix() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 1)
            .expect("should succeed");
        r.add_prefix("exv", "http://example.org/vocab/", 2)
            .expect("should succeed");
        let abbrev = r
            .abbreviate("http://example.org/vocab/Foo")
            .expect("should succeed");
        assert_eq!(abbrev, "exv:Foo");
    }

    // --- prefix_count ---

    #[test]
    fn test_prefix_count_zero() {
        let r = PrefixResolver::new();
        assert_eq!(r.prefix_count(), 0);
    }

    #[test]
    fn test_prefix_count_after_adds() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 1)
            .expect("should succeed");
        r.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", 2)
            .expect("should succeed");
        assert_eq!(r.prefix_count(), 2);
    }

    #[test]
    fn test_prefix_count_redeclaration_same() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://a.org/", 1)
            .expect("should succeed");
        r.add_prefix("ex", "http://b.org/", 2)
            .expect("should succeed");
        // HashMap replaces; count stays 1
        assert_eq!(r.prefix_count(), 1);
    }

    // --- declarations ---

    #[test]
    fn test_declarations_list() {
        let mut r = PrefixResolver::new();
        r.add_prefix("ex", "http://example.org/", 5)
            .expect("should succeed");
        let decls = r.declarations();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].prefix, "ex");
        assert_eq!(decls[0].line, 5);
    }

    #[test]
    fn test_declarations_order() {
        let mut r = PrefixResolver::new();
        r.add_prefix("a", "http://a.org/", 1)
            .expect("should succeed");
        r.add_prefix("b", "http://b.org/", 2)
            .expect("should succeed");
        r.add_prefix("c", "http://c.org/", 3)
            .expect("should succeed");
        assert_eq!(r.declarations().len(), 3);
        assert_eq!(r.declarations()[0].prefix, "a");
        assert_eq!(r.declarations()[2].prefix, "c");
    }

    // --- is_absolute_iri ---

    #[test]
    fn test_is_absolute_http() {
        assert!(PrefixResolver::is_absolute_iri("http://example.org/foo"));
    }

    #[test]
    fn test_is_absolute_https() {
        assert!(PrefixResolver::is_absolute_iri("https://example.org/"));
    }

    #[test]
    fn test_is_absolute_urn() {
        assert!(PrefixResolver::is_absolute_iri("urn:example:a123,z456"));
    }

    #[test]
    fn test_is_absolute_relative() {
        assert!(!PrefixResolver::is_absolute_iri("foo/bar"));
    }

    #[test]
    fn test_is_absolute_no_scheme() {
        assert!(!PrefixResolver::is_absolute_iri("no-colon-here"));
    }

    // --- ResolveSource variants ---

    #[test]
    fn test_resolve_source_absolute_variant() {
        let r = PrefixResolver::new();
        let result = r.resolve("http://x.org/").expect("should succeed");
        assert_eq!(result.source, ResolveSource::Absolute);
    }

    #[test]
    fn test_resolve_source_prefix_variant() {
        let mut r = PrefixResolver::new();
        r.add_prefix("x", "http://x.org/", 1)
            .expect("should succeed");
        let result = r.resolve("x:Foo").expect("should succeed");
        assert!(matches!(result.source, ResolveSource::Prefix(_)));
    }

    #[test]
    fn test_resolve_source_base_variant() {
        let r = PrefixResolver::with_base("http://base.org/");
        let result = r.resolve("relative").expect("should succeed");
        assert_eq!(result.source, ResolveSource::Base);
    }

    // --- error display ---

    #[test]
    fn test_unknown_prefix_display() {
        let e = ResolveError::UnknownPrefix("xyz".to_string());
        assert!(format!("{e}").contains("xyz"));
    }

    #[test]
    fn test_no_base_iri_display() {
        let e = ResolveError::NoBaseIri;
        assert!(format!("{e}").contains("base"));
    }

    #[test]
    fn test_invalid_curie_display() {
        let e = ResolveError::InvalidCurie("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn test_resolver_default() {
        let r = PrefixResolver::default();
        assert_eq!(r.prefix_count(), 0);
        assert!(!r.has_base());
    }

    #[test]
    fn test_resolve_unknown_curie_prefix() {
        let r = PrefixResolver::new();
        // "ex:Foo" has no registered prefix
        let err = r.resolve("ex:Foo");
        assert!(matches!(err, Err(ResolveError::UnknownPrefix(_))));
    }
}
