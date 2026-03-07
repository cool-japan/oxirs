//! @base / @prefix IRI resolution for Turtle and TriG.
//!
//! Implements RFC 3986-compatible relative IRI resolution and Turtle prefix
//! expansion. Supports both Turtle (`@base`/`@prefix`) and SPARQL
//! (`BASE`/`PREFIX`) syntaxes.

use std::collections::HashMap;
use std::fmt;

/// A base IRI directive.
#[derive(Debug, Clone, PartialEq)]
pub struct BaseDirective {
    /// The absolute base IRI string.
    pub base_iri: String,
}

impl BaseDirective {
    /// Create a new base directive from a validated absolute IRI.
    pub fn new(iri: String) -> Self {
        Self { base_iri: iri }
    }

    /// Return the IRI string.
    pub fn as_str(&self) -> &str {
        &self.base_iri
    }
}

/// Error type for IRI resolution.
#[derive(Debug, Clone, PartialEq)]
pub enum IriError {
    /// No base IRI has been set and a relative IRI was encountered.
    NoBase,
    /// The given base IRI is not an absolute IRI.
    InvalidBase(String),
    /// The requested prefix has not been declared.
    UnknownPrefix(String),
    /// The IRI string is malformed.
    MalformedIri(String),
}

impl fmt::Display for IriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IriError::NoBase => write!(f, "No base IRI set for relative IRI resolution"),
            IriError::InvalidBase(s) => write!(f, "Invalid base IRI: '{s}'"),
            IriError::UnknownPrefix(p) => write!(f, "Unknown prefix: '{p}:'"),
            IriError::MalformedIri(s) => write!(f, "Malformed IRI: '{s}'"),
        }
    }
}

impl std::error::Error for IriError {}

/// IRI resolver for a Turtle/TriG document.
#[derive(Debug, Default, Clone)]
pub struct IriResolver {
    /// The current base IRI directive, if set.
    pub base: Option<BaseDirective>,
    /// Declared namespace prefixes (prefix → IRI).
    pub prefixes: HashMap<String, String>,
}

impl IriResolver {
    /// Create an empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check whether `iri` is absolute (has a scheme).
    fn is_absolute(iri: &str) -> bool {
        iri.contains("://") || iri.starts_with("urn:")
    }

    /// Validate and set the base IRI.
    pub fn set_base(&mut self, iri: &str) -> Result<(), IriError> {
        if iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:") {
            self.base = Some(BaseDirective::new(iri.to_string()));
            Ok(())
        } else {
            Err(IriError::InvalidBase(iri.to_string()))
        }
    }

    /// Declare a namespace prefix.
    pub fn set_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
    }

    /// Resolve a (possibly relative) IRI against the current base.
    ///
    /// If the IRI is wrapped in angle brackets (`<...>`), they are stripped first.
    pub fn resolve_relative(&self, iri: &str) -> Result<String, IriError> {
        // Strip angle brackets
        let iri = if iri.starts_with('<') && iri.ends_with('>') {
            &iri[1..iri.len() - 1]
        } else {
            iri
        };

        // Already absolute
        if Self::is_absolute(iri) {
            return Ok(iri.to_string());
        }

        let base = self.base.as_ref().ok_or(IriError::NoBase)?;
        let base_str = base.as_str();

        // Protocol-relative: //host/path
        if iri.starts_with("//") {
            let scheme = base_str
                .split("://")
                .next()
                .ok_or_else(|| IriError::InvalidBase(base_str.to_string()))?;
            return Ok(format!("{scheme}:{iri}"));
        }

        // Fragment: #fragment → append to base (without existing fragment)
        if iri.starts_with('#') {
            let base_no_fragment = base_str.split('#').next().unwrap_or(base_str);
            return Ok(format!("{base_no_fragment}{iri}"));
        }

        // Query: ?query → replace query in base
        if iri.starts_with('?') {
            let base_no_query = base_str.split('?').next().unwrap_or(base_str);
            let base_no_frag = base_no_query.split('#').next().unwrap_or(base_no_query);
            return Ok(format!("{base_no_frag}{iri}"));
        }

        // Absolute path: /path → replace path component
        if iri.starts_with('/') {
            let authority_end = Self::authority_end(base_str);
            let authority = &base_str[..authority_end];
            return Ok(format!("{authority}{iri}"));
        }

        // Relative path: resolve against base directory
        let base_dir = Self::base_directory(base_str);
        let merged = format!("{base_dir}{iri}");
        Ok(Self::remove_dot_segments(&merged))
    }

    /// Find the end of the authority component (scheme + "://" + host [+ port]).
    fn authority_end(base: &str) -> usize {
        // Find "://" then skip to next "/"
        if let Some(sep) = base.find("://") {
            let after_scheme = sep + 3;
            if let Some(slash) = base[after_scheme..].find('/') {
                after_scheme + slash
            } else {
                base.len()
            }
        } else if base.starts_with("urn:") {
            // For URNs there is no path separator in the authority sense
            base.len()
        } else {
            0
        }
    }

    /// Return the "directory" part of a base IRI (everything up to and including the last `/`).
    fn base_directory(base: &str) -> &str {
        if let Some(pos) = base.rfind('/') {
            &base[..=pos]
        } else {
            base
        }
    }

    /// Remove `.` and `..` segments from a path string.
    fn remove_dot_segments(path: &str) -> String {
        // Find scheme prefix (keep it intact)
        let (prefix, rest) = if let Some(sep) = path.find("://") {
            let after = sep + 3;
            if let Some(slash) = path[after..].find('/') {
                let split = after + slash;
                (&path[..split], &path[split..])
            } else {
                (path, "")
            }
        } else {
            ("", path)
        };

        // rest starts with '/' — split will produce an empty first segment; skip it
        let mut output: Vec<&str> = Vec::new();
        for seg in rest.split('/') {
            match seg {
                "" if output.is_empty() => {
                    // Leading '/' — do not add empty segment, reconstruct with leading slash later
                }
                "." => {}
                ".." => {
                    output.pop();
                }
                s => output.push(s),
            }
        }

        if prefix.is_empty() {
            // No scheme — just a relative-style path
            format!("/{}", output.join("/"))
        } else {
            format!("{prefix}/{}", output.join("/"))
        }
    }

    /// Expand a prefixed name (`prefix:local`) to a full IRI.
    pub fn resolve_prefixed(&self, qname: &str) -> Result<String, IriError> {
        let colon = qname
            .find(':')
            .ok_or_else(|| IriError::MalformedIri(format!("No ':' in prefixed name: '{qname}'")))?;
        let prefix = &qname[..colon];
        let local = &qname[colon + 1..];

        let ns = self
            .prefixes
            .get(prefix)
            .ok_or_else(|| IriError::UnknownPrefix(prefix.to_string()))?;
        Ok(format!("{ns}{local}"))
    }

    /// Resolve a token: absolute IRI, prefixed name, or relative IRI.
    pub fn resolve(&self, token: &str) -> Result<String, IriError> {
        // Strip angle brackets → absolute or relative
        let stripped = if token.starts_with('<') && token.ends_with('>') {
            &token[1..token.len() - 1]
        } else {
            token
        };

        // Absolute
        if Self::is_absolute(stripped) {
            return Ok(stripped.to_string());
        }

        // Prefixed name (contains ':' and no '<')
        if !token.starts_with('<') && token.contains(':') {
            return self.resolve_prefixed(token);
        }

        // Relative
        self.resolve_relative(token)
    }

    /// Parse a `@base` or `BASE` directive line and return the IRI if matched.
    pub fn parse_base_directive(line: &str) -> Option<String> {
        let trimmed = line.trim();
        let upper = trimmed.to_ascii_uppercase();

        // Turtle: @base <IRI> .
        // SPARQL: BASE <IRI>
        let rest = if upper.starts_with("@BASE") {
            &trimmed[5..]
        } else if upper.starts_with("BASE") {
            &trimmed[4..]
        } else {
            return None;
        };

        Self::extract_iri(rest.trim())
    }

    /// Parse a `@prefix` or `PREFIX` directive line.
    ///
    /// Returns `Some((prefix, iri))` on success.
    pub fn parse_prefix_directive(line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();
        let upper = trimmed.to_ascii_uppercase();

        let rest = if upper.starts_with("@PREFIX") {
            &trimmed[7..]
        } else if upper.starts_with("PREFIX") {
            &trimmed[6..]
        } else {
            return None;
        };

        let rest = rest.trim();

        // Find prefix (ends at ':')
        let colon = rest.find(':')?;
        let prefix = rest[..colon].trim().to_string();
        let after_colon = rest[colon + 1..].trim();

        let iri = Self::extract_iri(after_colon)?;
        Some((prefix, iri))
    }

    /// Extract an IRI from `<IRI>` (optionally followed by whitespace and `.`).
    fn extract_iri(s: &str) -> Option<String> {
        let s = s.trim();
        if !s.starts_with('<') {
            return None;
        }
        let end = s.find('>')?;
        Some(s[1..end].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------ set_base ------

    #[test]
    fn test_set_base_http() {
        let mut r = IriResolver::new();
        assert!(r.set_base("http://example.org/").is_ok());
        assert_eq!(
            r.base.as_ref().map(|b| b.as_str()),
            Some("http://example.org/")
        );
    }

    #[test]
    fn test_set_base_https() {
        let mut r = IriResolver::new();
        assert!(r.set_base("https://example.org/").is_ok());
    }

    #[test]
    fn test_set_base_urn() {
        let mut r = IriResolver::new();
        assert!(r.set_base("urn:example:foo").is_ok());
    }

    #[test]
    fn test_set_base_invalid() {
        let mut r = IriResolver::new();
        let e = r.set_base("relative/path").expect_err("should fail");
        assert!(matches!(e, IriError::InvalidBase(_)));
    }

    // ------ resolve_relative ------

    #[test]
    fn test_resolve_absolute_passthrough() {
        let r = IriResolver::new();
        let result = r.resolve_relative("http://other.org/foo").expect("resolve");
        assert_eq!(result, "http://other.org/foo");
    }

    #[test]
    fn test_resolve_angle_bracket_absolute() {
        let r = IriResolver::new();
        let result = r
            .resolve_relative("<http://other.org/foo>")
            .expect("resolve");
        assert_eq!(result, "http://other.org/foo");
    }

    #[test]
    fn test_resolve_relative_path() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/base/").expect("set_base");
        let result = r.resolve_relative("foo").expect("resolve");
        assert_eq!(result, "http://example.org/base/foo");
    }

    #[test]
    fn test_resolve_relative_path_no_trailing_slash() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/base/doc").expect("set_base");
        let result = r.resolve_relative("other").expect("resolve");
        // base directory is http://example.org/base/, so resolves to .../base/other
        assert!(result.contains("base/other"));
    }

    #[test]
    fn test_resolve_fragment() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/doc").expect("set_base");
        let result = r.resolve_relative("#section1").expect("resolve");
        assert_eq!(result, "http://example.org/doc#section1");
    }

    #[test]
    fn test_resolve_fragment_strips_existing_fragment() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/doc#old").expect("set_base");
        let result = r.resolve_relative("#new").expect("resolve");
        assert_eq!(result, "http://example.org/doc#new");
    }

    #[test]
    fn test_resolve_query() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/search?q=foo")
            .expect("set_base");
        let result = r.resolve_relative("?q=bar").expect("resolve");
        assert_eq!(result, "http://example.org/search?q=bar");
    }

    #[test]
    fn test_resolve_absolute_path() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/a/b/c").expect("set_base");
        let result = r.resolve_relative("/root").expect("resolve");
        assert_eq!(result, "http://example.org/root");
    }

    #[test]
    fn test_resolve_no_base_error() {
        let r = IriResolver::new();
        let e = r.resolve_relative("relative").expect_err("no base");
        assert_eq!(e, IriError::NoBase);
    }

    #[test]
    fn test_resolve_dot_dot_segments() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/a/b/c/").expect("set_base");
        let result = r.resolve_relative("../../d").expect("resolve");
        assert!(result.contains("/a/d"));
    }

    #[test]
    fn test_resolve_urn_absolute() {
        let r = IriResolver::new();
        let result = r.resolve_relative("urn:example:thing").expect("resolve");
        assert_eq!(result, "urn:example:thing");
    }

    // ------ resolve_prefixed ------

    #[test]
    fn test_resolve_prefixed_basic() {
        let mut r = IriResolver::new();
        r.set_prefix("ex", "http://example.org/");
        let result = r.resolve_prefixed("ex:Person").expect("resolve");
        assert_eq!(result, "http://example.org/Person");
    }

    #[test]
    fn test_resolve_prefixed_empty_local() {
        let mut r = IriResolver::new();
        r.set_prefix("ex", "http://example.org/");
        let result = r.resolve_prefixed("ex:").expect("resolve");
        assert_eq!(result, "http://example.org/");
    }

    #[test]
    fn test_resolve_prefixed_unknown_prefix() {
        let r = IriResolver::new();
        let e = r.resolve_prefixed("unknown:thing").expect_err("error");
        assert!(matches!(e, IriError::UnknownPrefix(_)));
    }

    #[test]
    fn test_resolve_prefixed_no_colon() {
        let r = IriResolver::new();
        let e = r.resolve_prefixed("nocolon").expect_err("error");
        assert!(matches!(e, IriError::MalformedIri(_)));
    }

    // ------ resolve (auto-dispatch) ------

    #[test]
    fn test_resolve_absolute_iri() {
        let r = IriResolver::new();
        let result = r.resolve("http://example.org/foo").expect("resolve");
        assert_eq!(result, "http://example.org/foo");
    }

    #[test]
    fn test_resolve_prefixed_via_resolve() {
        let mut r = IriResolver::new();
        r.set_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let result = r.resolve("rdf:type").expect("resolve");
        assert_eq!(result, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    #[test]
    fn test_resolve_bracketed_iri() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/").expect("set_base");
        let result = r.resolve("<relative>").expect("resolve");
        assert_eq!(result, "http://example.org/relative");
    }

    // ------ parse_base_directive ------

    #[test]
    fn test_parse_base_turtle_style() {
        let result = IriResolver::parse_base_directive("@base <http://example.org/> .");
        assert_eq!(result, Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_parse_base_sparql_style() {
        let result = IriResolver::parse_base_directive("BASE <http://example.org/>");
        assert_eq!(result, Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_parse_base_case_insensitive() {
        let result = IriResolver::parse_base_directive("base <http://example.org/>");
        assert_eq!(result, Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_parse_base_no_match() {
        let result = IriResolver::parse_base_directive("PREFIX ex: <http://example.org/>");
        assert!(result.is_none());
    }

    // ------ parse_prefix_directive ------

    #[test]
    fn test_parse_prefix_turtle_style() {
        let result = IriResolver::parse_prefix_directive("@prefix ex: <http://example.org/> .");
        assert_eq!(
            result,
            Some(("ex".to_string(), "http://example.org/".to_string()))
        );
    }

    #[test]
    fn test_parse_prefix_sparql_style() {
        let result = IriResolver::parse_prefix_directive(
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
        );
        assert_eq!(
            result,
            Some((
                "rdf".to_string(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string()
            ))
        );
    }

    #[test]
    fn test_parse_prefix_case_insensitive() {
        let result = IriResolver::parse_prefix_directive("prefix ex: <http://example.org/>");
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_prefix_empty_prefix() {
        let result = IriResolver::parse_prefix_directive("@prefix : <http://default.org/> .");
        assert_eq!(
            result,
            Some(("".to_string(), "http://default.org/".to_string()))
        );
    }

    #[test]
    fn test_parse_prefix_no_match() {
        let result = IriResolver::parse_prefix_directive("@base <http://example.org/>");
        assert!(result.is_none());
    }

    // ------ IriError Display ------

    #[test]
    fn test_iri_error_no_base_display() {
        let e = IriError::NoBase;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_iri_error_invalid_base_display() {
        let e = IriError::InvalidBase("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_iri_error_unknown_prefix_display() {
        let e = IriError::UnknownPrefix("ex".to_string());
        assert!(e.to_string().contains("ex"));
    }

    #[test]
    fn test_iri_error_malformed_display() {
        let e = IriError::MalformedIri(":::".to_string());
        assert!(e.to_string().contains(":::"));
    }

    // ------ integration ------

    #[test]
    fn test_resolve_document_iris() {
        let mut r = IriResolver::new();
        r.set_base("http://example.org/ontology/")
            .expect("set_base");
        r.set_prefix("owl", "http://www.w3.org/2002/07/owl#");
        r.set_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");

        // Prefixed
        let c = r.resolve("owl:Class").expect("owl:Class");
        assert_eq!(c, "http://www.w3.org/2002/07/owl#Class");

        // Relative
        let rel = r.resolve("<Person>").expect("<Person>");
        assert_eq!(rel, "http://example.org/ontology/Person");

        // Absolute
        let abs = r.resolve("http://other.org/Thing").expect("absolute");
        assert_eq!(abs, "http://other.org/Thing");
    }
}
