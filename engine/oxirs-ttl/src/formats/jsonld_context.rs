//! JSON-LD Context Processing
//!
//! Implements JSON-LD `@context` parsing, term expansion, IRI compaction,
//! context merging, and node processing as defined by the W3C JSON-LD 1.1
//! specification.
//!
//! # Context Syntax
//!
//! A JSON-LD context may define:
//! - `@base`: the base IRI for relative IRI resolution
//! - `@vocab`: the default vocabulary IRI
//! - Term definitions: map term name → IRI or term definition object
//!
//! # Example
//!
//! ```json
//! {
//!   "@context": {
//!     "@base": "http://example.org/",
//!     "@vocab": "http://schema.org/",
//!     "name": "http://schema.org/name",
//!     "age": { "@id": "http://schema.org/age", "@type": "xsd:integer" }
//!   }
//! }
//! ```
//!
//! # References
//!
//! - <https://www.w3.org/TR/json-ld11/>
//! - <https://www.w3.org/TR/json-ld11-api/>

use std::collections::HashMap;
use std::fmt;

// ─── Container types ─────────────────────────────────────────────────────────

/// JSON-LD container type for a term definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Container {
    /// `@list` — ordered collection
    List,
    /// `@set` — unordered collection
    Set,
    /// `@index` — index map
    Index,
    /// `@language` — language map
    Language,
    /// `@type` — type map
    Type,
    /// `@graph` — named graph
    Graph,
}

impl Container {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "@list" => Some(Self::List),
            "@set" => Some(Self::Set),
            "@index" => Some(Self::Index),
            "@language" => Some(Self::Language),
            "@type" => Some(Self::Type),
            "@graph" => Some(Self::Graph),
            _ => None,
        }
    }
}

// ─── TermDef ─────────────────────────────────────────────────────────────────

/// A term definition within a JSON-LD context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermDef {
    /// The expanded IRI for this term
    pub iri: String,
    /// Type coercion (`@type` field in the term definition)
    pub type_coerce: Option<String>,
    /// Language tag for plain literals
    pub language: Option<String>,
    /// Container type mapping
    pub container: Option<Container>,
    /// Whether this term is protected (cannot be overridden)
    pub protected: bool,
}

impl TermDef {
    /// Create a simple term definition from an IRI.
    pub fn simple(iri: impl Into<String>) -> Self {
        Self {
            iri: iri.into(),
            type_coerce: None,
            language: None,
            container: None,
            protected: false,
        }
    }
}

// ─── Context ─────────────────────────────────────────────────────────────────

/// A parsed JSON-LD `@context` document.
#[derive(Debug, Clone, Default)]
pub struct JsonLdContext {
    /// The base IRI (`@base`)
    pub base: Option<String>,
    /// The default vocabulary IRI (`@vocab`)
    pub vocab: Option<String>,
    /// Term definitions
    pub terms: HashMap<String, TermDef>,
}

impl JsonLdContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a term definition.
    pub fn define_term(&mut self, name: impl Into<String>, def: TermDef) {
        self.terms.insert(name.into(), def);
    }

    /// Add a simple IRI mapping.
    pub fn define_iri(&mut self, name: impl Into<String>, iri: impl Into<String>) {
        self.terms.insert(name.into(), TermDef::simple(iri));
    }
}

// ─── Error ───────────────────────────────────────────────────────────────────

/// Error from JSON-LD context processing.
#[derive(Debug, Clone)]
pub struct ContextError(pub String);

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContextError: {}", self.0)
    }
}

impl std::error::Error for ContextError {}

// ─── Minimal JSON parser ──────────────────────────────────────────────────────

/// Parse a JSON string value from a JSON text.
/// Returns `(value, remaining)` where remaining starts after the closing `"`.
fn parse_json_string(input: &str) -> Option<(String, &str)> {
    let input = input.trim_start();
    if !input.starts_with('"') {
        return None;
    }
    let input = &input[1..];
    let mut out = String::new();
    let mut chars = input.char_indices();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '"' => return Some((out, &input[i + 1..])),
            '\\' => {
                if let Some((_, esc)) = chars.next() {
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        other => {
                            out.push('\\');
                            out.push(other);
                        }
                    }
                }
            }
            other => out.push(other),
        }
    }
    None
}

/// Parse a boolean value from a JSON text.
fn parse_json_bool(input: &str) -> Option<(bool, &str)> {
    let input = input.trim_start();
    if let Some(rest) = input.strip_prefix("true") {
        Some((true, rest))
    } else if let Some(rest) = input.strip_prefix("false") {
        Some((false, rest))
    } else {
        None
    }
}

/// Find the matching closing `}` for an object that starts at the current position
/// (after the opening `{` has been consumed).
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 1i32;
    let mut in_string = false;
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_string => in_string = true,
            b'"' if in_string => {
                // Check for escape
                let escapes = bytes[..i].iter().rev().take_while(|&&b| b == b'\\').count();
                if escapes % 2 == 0 {
                    in_string = false;
                }
            }
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Very lightweight JSON object parser: extracts top-level key→raw-value pairs.
/// Only parses shallow JSON objects (one level deep for the keys we care about).
fn parse_json_object(json: &str) -> Result<HashMap<String, String>, ContextError> {
    let s = json.trim();
    if !s.starts_with('{') {
        return Err(ContextError(format!(
            "expected '{{', got: '{}'",
            &s[..s.len().min(20)]
        )));
    }
    let s = &s[1..]; // skip '{'

    let mut map = HashMap::new();
    let mut rest = s;

    loop {
        rest = rest.trim_start();
        if rest.starts_with('}') {
            break;
        }
        if rest.starts_with(',') {
            rest = &rest[1..];
            rest = rest.trim_start();
        }
        if rest.starts_with('}') {
            break;
        }
        if rest.is_empty() {
            break;
        }

        // Parse key
        let (key, after_key) = parse_json_string(rest).ok_or_else(|| {
            ContextError(format!(
                "expected key string in: '{}'",
                &rest[..rest.len().min(30)]
            ))
        })?;
        rest = after_key.trim_start();

        if !rest.starts_with(':') {
            return Err(ContextError(format!(
                "expected ':' after key '{}', got: '{}'",
                key,
                &rest[..rest.len().min(10)]
            )));
        }
        rest = &rest[1..];
        rest = rest.trim_start();

        // Parse value (string, boolean, or nested object)
        let raw_value = if rest.starts_with('"') {
            let (val, after) =
                parse_json_string(rest).ok_or_else(|| ContextError("bad string value".into()))?;
            rest = after;
            val
        } else if rest.starts_with('{') {
            let end = find_matching_brace(&rest[1..])
                .ok_or_else(|| ContextError("unclosed '{' in value".into()))?;
            let obj_str = &rest[..end + 2]; // include both braces
            rest = &rest[end + 2..];
            obj_str.to_string()
        } else if rest.starts_with("true") || rest.starts_with("false") {
            let (b, after) = parse_json_bool(rest).expect("checked starts_with");
            rest = after;
            b.to_string()
        } else if rest.starts_with("null") {
            rest = &rest[4..];
            String::new()
        } else {
            return Err(ContextError(format!(
                "unsupported value type at: '{}'",
                &rest[..rest.len().min(20)]
            )));
        };

        map.insert(key, raw_value);
    }

    Ok(map)
}

// ─── Processor ────────────────────────────────────────────────────────────────

/// JSON-LD context processor.
///
/// Provides parsing, expansion, compaction, and node processing.
#[derive(Debug, Default, Clone)]
pub struct ContextProcessor;

impl ContextProcessor {
    /// Create a new processor.
    pub fn new() -> Self {
        Self
    }

    /// Parse a `@context` value from a JSON string.
    ///
    /// The input may be either:
    /// - A bare `@context` object: `{ "@base": "...", "term": "..." }`
    /// - A full JSON-LD document with a `@context` key (unwrapped automatically)
    pub fn parse_context(&self, json: &str) -> Result<JsonLdContext, ContextError> {
        let json = json.trim();

        // If the JSON has a "@context" wrapper, extract that value
        let context_json = if let Ok(outer) = parse_json_object(json) {
            if let Some(ctx_raw) = outer.get("@context") {
                ctx_raw.clone()
            } else {
                json.to_string()
            }
        } else {
            json.to_string()
        };

        let obj = parse_json_object(&context_json)?;
        let mut ctx = JsonLdContext::new();

        for (key, value) in &obj {
            match key.as_str() {
                "@base" => {
                    ctx.base = Some(value.clone());
                }
                "@vocab" => {
                    ctx.vocab = Some(value.clone());
                }
                k if k.starts_with('@') => {
                    // Ignore other JSON-LD keywords
                }
                term => {
                    // Parse term definition
                    let def = if value.starts_with('{') {
                        self.parse_term_def(value)?
                    } else {
                        TermDef::simple(value)
                    };
                    ctx.terms.insert(term.to_string(), def);
                }
            }
        }

        Ok(ctx)
    }

    fn parse_term_def(&self, json: &str) -> Result<TermDef, ContextError> {
        let obj = parse_json_object(json)?;

        let iri = obj.get("@id").cloned().unwrap_or_default();

        let type_coerce = obj.get("@type").cloned().filter(|s| !s.is_empty());
        let language = obj.get("@language").cloned().filter(|s| !s.is_empty());
        let container = obj.get("@container").and_then(|c| Container::from_str(c));
        let protected = obj
            .get("@protected")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        Ok(TermDef {
            iri,
            type_coerce,
            language,
            container,
            protected,
        })
    }

    /// Compact an absolute IRI to its shortest form given the context.
    ///
    /// Algorithm (in order):
    /// 1. If an exact term maps to this IRI, return the term name
    /// 2. If `@vocab` is set and the IRI starts with it, return the suffix
    /// 3. If `@base` is set and the IRI starts with it, return the suffix
    /// 4. Otherwise return the IRI unchanged
    pub fn compact_iri(&self, iri: &str, ctx: &JsonLdContext) -> String {
        // 1. Exact term match
        for (term, def) in &ctx.terms {
            if def.iri == iri {
                return term.clone();
            }
        }

        // 2. Vocab prefix
        if let Some(vocab) = &ctx.vocab {
            if iri.starts_with(vocab.as_str()) && iri.len() > vocab.len() {
                return iri[vocab.len()..].to_string();
            }
        }

        // 3. Base prefix
        if let Some(base) = &ctx.base {
            if iri.starts_with(base.as_str()) && iri.len() > base.len() {
                return iri[base.len()..].to_string();
            }
        }

        // 4. No compaction possible
        iri.to_string()
    }

    /// Expand a term to its full IRI using the context.
    ///
    /// Algorithm:
    /// 1. If `term` is in the context terms, return `def.iri`
    /// 2. If term contains `:`, treat as a CURIE — look up the prefix part
    /// 3. If `@vocab` is set, return `vocab + term`
    /// 4. If `@base` is set, return `base + term`
    /// 5. Otherwise return `None`
    pub fn expand_term(&self, term: &str, ctx: &JsonLdContext) -> Option<String> {
        // 1. Exact term
        if let Some(def) = ctx.terms.get(term) {
            if !def.iri.is_empty() {
                return Some(def.iri.clone());
            }
        }

        // 2. CURIE (prefix:local)
        if let Some(colon) = term.find(':') {
            let prefix = &term[..colon];
            let local = &term[colon + 1..];
            // Skip if the part before ':' is "http", "https", etc. (it's already a full IRI)
            if prefix == "http" || prefix == "https" || prefix == "urn" || prefix == "ftp" {
                return Some(term.to_string());
            }
            if let Some(def) = ctx.terms.get(prefix) {
                return Some(format!("{}{}", def.iri, local));
            }
        }

        // 3. Vocab
        if let Some(vocab) = &ctx.vocab {
            return Some(format!("{vocab}{term}"));
        }

        // 4. Base
        if let Some(base) = &ctx.base {
            return Some(format!("{base}{term}"));
        }

        None
    }

    /// Merge two contexts: terms in `overlay` override terms in `base`,
    /// unless the base term is `protected`.
    ///
    /// `@base` and `@vocab` from `overlay` take precedence.
    pub fn merge_contexts(&self, base: &JsonLdContext, overlay: &JsonLdContext) -> JsonLdContext {
        let mut merged = base.clone();

        if overlay.base.is_some() {
            merged.base = overlay.base.clone();
        }
        if overlay.vocab.is_some() {
            merged.vocab = overlay.vocab.clone();
        }

        for (term, def) in &overlay.terms {
            if let Some(existing) = merged.terms.get(term) {
                if existing.protected {
                    // Protected term cannot be overridden
                    continue;
                }
            }
            merged.terms.insert(term.clone(), def.clone());
        }

        merged
    }

    /// Process a JSON-LD node object, expanding all keys using the context.
    ///
    /// Returns a `HashMap<expanded_key, raw_value_string>`.
    pub fn process_node(
        &self,
        node_json: &str,
        ctx: &JsonLdContext,
    ) -> Result<HashMap<String, String>, ContextError> {
        let obj = parse_json_object(node_json)?;
        let mut result = HashMap::new();

        for (key, value) in obj {
            let expanded_key = if key.starts_with('@') {
                key.clone()
            } else if let Some(expanded) = self.expand_term(&key, ctx) {
                expanded
            } else {
                key.clone()
            };
            result.insert(expanded_key, value);
        }

        Ok(result)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn proc() -> ContextProcessor {
        ContextProcessor::new()
    }

    fn simple_ctx() -> JsonLdContext {
        let mut ctx = JsonLdContext::new();
        ctx.base = Some("http://example.org/".to_string());
        ctx.vocab = Some("http://schema.org/".to_string());
        ctx.define_iri("name", "http://schema.org/name");
        ctx.define_iri("age", "http://schema.org/age");
        ctx
    }

    // ── Container ─────────────────────────────────────────────────────────────

    #[test]
    fn test_container_from_str_list() {
        assert_eq!(Container::from_str("@list"), Some(Container::List));
    }

    #[test]
    fn test_container_from_str_set() {
        assert_eq!(Container::from_str("@set"), Some(Container::Set));
    }

    #[test]
    fn test_container_from_str_index() {
        assert_eq!(Container::from_str("@index"), Some(Container::Index));
    }

    #[test]
    fn test_container_from_str_language() {
        assert_eq!(Container::from_str("@language"), Some(Container::Language));
    }

    #[test]
    fn test_container_from_str_type() {
        assert_eq!(Container::from_str("@type"), Some(Container::Type));
    }

    #[test]
    fn test_container_from_str_graph() {
        assert_eq!(Container::from_str("@graph"), Some(Container::Graph));
    }

    #[test]
    fn test_container_from_str_unknown() {
        assert_eq!(Container::from_str("@unknown"), None);
    }

    // ── TermDef ───────────────────────────────────────────────────────────────

    #[test]
    fn test_term_def_simple() {
        let def = TermDef::simple("http://schema.org/name");
        assert_eq!(def.iri, "http://schema.org/name");
        assert!(!def.protected);
        assert!(def.type_coerce.is_none());
    }

    // ── JsonLdContext ─────────────────────────────────────────────────────────

    #[test]
    fn test_context_new_empty() {
        let ctx = JsonLdContext::new();
        assert!(ctx.base.is_none());
        assert!(ctx.vocab.is_none());
        assert!(ctx.terms.is_empty());
    }

    #[test]
    fn test_context_define_iri() {
        let mut ctx = JsonLdContext::new();
        ctx.define_iri("name", "http://schema.org/name");
        assert!(ctx.terms.contains_key("name"));
        assert_eq!(ctx.terms["name"].iri, "http://schema.org/name");
    }

    // ── parse_context ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_context_simple() {
        let json = r#"{"@base": "http://example.org/", "@vocab": "http://schema.org/"}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert_eq!(ctx.base.as_deref(), Some("http://example.org/"));
        assert_eq!(ctx.vocab.as_deref(), Some("http://schema.org/"));
    }

    #[test]
    fn test_parse_context_with_terms() {
        let json = r#"{"name": "http://schema.org/name", "age": "http://schema.org/age"}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert!(ctx.terms.contains_key("name"));
        assert_eq!(ctx.terms["name"].iri, "http://schema.org/name");
    }

    #[test]
    fn test_parse_context_term_definition_object() {
        let json = r#"{"age": {"@id": "http://schema.org/age", "@type": "xsd:integer"}}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert_eq!(ctx.terms["age"].iri, "http://schema.org/age");
        assert_eq!(ctx.terms["age"].type_coerce.as_deref(), Some("xsd:integer"));
    }

    #[test]
    fn test_parse_context_protected_term() {
        let json = r#"{"myTerm": {"@id": "http://ex.org/term", "@protected": true}}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert!(ctx.terms["myTerm"].protected);
    }

    #[test]
    fn test_parse_context_container() {
        let json = r#"{"tags": {"@id": "http://ex.org/tags", "@container": "@set"}}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert_eq!(ctx.terms["tags"].container, Some(Container::Set));
    }

    #[test]
    fn test_parse_context_language() {
        let json = r#"{"label": {"@id": "http://ex.org/label", "@language": "en"}}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert_eq!(ctx.terms["label"].language.as_deref(), Some("en"));
    }

    #[test]
    fn test_parse_context_full_document() {
        let json =
            r#"{"@context": {"@vocab": "http://schema.org/", "name": "http://schema.org/name"}}"#;
        let ctx = proc().parse_context(json).expect("ok");
        assert_eq!(ctx.vocab.as_deref(), Some("http://schema.org/"));
        assert!(ctx.terms.contains_key("name"));
    }

    #[test]
    fn test_parse_context_error_invalid_json() {
        let result = proc().parse_context("this is not json");
        assert!(result.is_err());
    }

    // ── compact_iri ───────────────────────────────────────────────────────────

    #[test]
    fn test_compact_exact_term_match() {
        let ctx = simple_ctx();
        let compacted = proc().compact_iri("http://schema.org/name", &ctx);
        assert_eq!(compacted, "name");
    }

    #[test]
    fn test_compact_via_vocab() {
        let ctx = simple_ctx();
        let compacted = proc().compact_iri("http://schema.org/description", &ctx);
        assert_eq!(compacted, "description");
    }

    #[test]
    fn test_compact_via_base() {
        let ctx = simple_ctx();
        let compacted = proc().compact_iri("http://example.org/resource", &ctx);
        assert_eq!(compacted, "resource");
    }

    #[test]
    fn test_compact_no_match() {
        let ctx = simple_ctx();
        let iri = "http://other.org/thing";
        let compacted = proc().compact_iri(iri, &ctx);
        assert_eq!(compacted, iri);
    }

    #[test]
    fn test_compact_empty_context() {
        let ctx = JsonLdContext::new();
        let iri = "http://schema.org/name";
        assert_eq!(proc().compact_iri(iri, &ctx), iri);
    }

    // ── expand_term ──────────────────────────────────────────────────────────

    #[test]
    fn test_expand_exact_term() {
        let ctx = simple_ctx();
        let expanded = proc().expand_term("name", &ctx);
        assert_eq!(expanded, Some("http://schema.org/name".to_string()));
    }

    #[test]
    fn test_expand_via_vocab() {
        let ctx = simple_ctx();
        let expanded = proc().expand_term("description", &ctx);
        assert_eq!(expanded, Some("http://schema.org/description".to_string()));
    }

    #[test]
    fn test_expand_via_base() {
        let mut ctx = JsonLdContext::new();
        ctx.base = Some("http://example.org/".to_string());
        let expanded = proc().expand_term("resource", &ctx);
        assert_eq!(expanded, Some("http://example.org/resource".to_string()));
    }

    #[test]
    fn test_expand_curie() {
        let mut ctx = JsonLdContext::new();
        ctx.define_iri("schema", "http://schema.org/");
        let expanded = proc().expand_term("schema:name", &ctx);
        assert_eq!(expanded, Some("http://schema.org/name".to_string()));
    }

    #[test]
    fn test_expand_full_iri_preserved() {
        let ctx = JsonLdContext::new();
        let iri = "http://schema.org/name";
        let expanded = proc().expand_term(iri, &ctx);
        assert_eq!(expanded, Some(iri.to_string()));
    }

    #[test]
    fn test_expand_unknown_no_vocab() {
        let ctx = JsonLdContext::new();
        let result = proc().expand_term("unknown", &ctx);
        assert!(result.is_none());
    }

    // ── merge_contexts ────────────────────────────────────────────────────────

    #[test]
    fn test_merge_adds_overlay_terms() {
        let base = simple_ctx();
        let mut overlay = JsonLdContext::new();
        overlay.define_iri("email", "http://schema.org/email");
        let merged = proc().merge_contexts(&base, &overlay);
        assert!(merged.terms.contains_key("email"));
        assert!(merged.terms.contains_key("name")); // preserved from base
    }

    #[test]
    fn test_merge_overlay_overrides_base() {
        let mut base = JsonLdContext::new();
        base.define_iri("name", "http://old.org/name");
        let mut overlay = JsonLdContext::new();
        overlay.define_iri("name", "http://new.org/name");
        let merged = proc().merge_contexts(&base, &overlay);
        assert_eq!(merged.terms["name"].iri, "http://new.org/name");
    }

    #[test]
    fn test_merge_protected_term_not_overridden() {
        let mut base = JsonLdContext::new();
        base.define_term(
            "name",
            TermDef {
                iri: "http://protected.org/name".to_string(),
                type_coerce: None,
                language: None,
                container: None,
                protected: true,
            },
        );
        let mut overlay = JsonLdContext::new();
        overlay.define_iri("name", "http://new.org/name");
        let merged = proc().merge_contexts(&base, &overlay);
        // Protected term should not be overridden
        assert_eq!(merged.terms["name"].iri, "http://protected.org/name");
    }

    #[test]
    fn test_merge_overlay_vocab_takes_precedence() {
        let mut base = JsonLdContext::new();
        base.vocab = Some("http://base.org/".to_string());
        let mut overlay = JsonLdContext::new();
        overlay.vocab = Some("http://overlay.org/".to_string());
        let merged = proc().merge_contexts(&base, &overlay);
        assert_eq!(merged.vocab.as_deref(), Some("http://overlay.org/"));
    }

    #[test]
    fn test_merge_overlay_base_takes_precedence() {
        let mut base = JsonLdContext::new();
        base.base = Some("http://base.org/".to_string());
        let mut overlay = JsonLdContext::new();
        overlay.base = Some("http://new.org/".to_string());
        let merged = proc().merge_contexts(&base, &overlay);
        assert_eq!(merged.base.as_deref(), Some("http://new.org/"));
    }

    // ── process_node ─────────────────────────────────────────────────────────

    #[test]
    fn test_process_node_expands_keys() {
        let ctx = simple_ctx();
        let node = r#"{"name": "Alice", "age": "30"}"#;
        let result = proc().process_node(node, &ctx).expect("ok");
        assert!(result.contains_key("http://schema.org/name"));
        assert!(result.contains_key("http://schema.org/age"));
    }

    #[test]
    fn test_process_node_preserves_at_keywords() {
        let ctx = simple_ctx();
        let node = r#"{"@id": "http://ex.org/alice", "name": "Alice"}"#;
        let result = proc().process_node(node, &ctx).expect("ok");
        assert!(result.contains_key("@id"));
    }

    #[test]
    fn test_process_node_values_preserved() {
        let ctx = simple_ctx();
        let node = r#"{"name": "Alice"}"#;
        let result = proc().process_node(node, &ctx).expect("ok");
        assert_eq!(result["http://schema.org/name"], "Alice");
    }

    #[test]
    fn test_process_node_error_invalid_json() {
        let ctx = simple_ctx();
        let result = proc().process_node("not json", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_node_unknown_term() {
        let ctx = JsonLdContext::new();
        let node = r#"{"unknownTerm": "value"}"#;
        let result = proc().process_node(node, &ctx).expect("ok");
        // Without vocab/base, unknown terms stay as-is
        assert!(result.contains_key("unknownTerm"));
    }

    // ── Round-trip ────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_compact_roundtrip() {
        let ctx = simple_ctx();
        let original_term = "name";
        let expanded = proc().expand_term(original_term, &ctx).expect("ok");
        let compacted = proc().compact_iri(&expanded, &ctx);
        assert_eq!(compacted, original_term);
    }

    #[test]
    fn test_compact_expand_roundtrip_vocab() {
        let ctx = simple_ctx();
        let iri = "http://schema.org/description";
        let compacted = proc().compact_iri(iri, &ctx);
        let expanded = proc().expand_term(&compacted, &ctx).expect("ok");
        assert_eq!(expanded, iri);
    }

    // ── ContextError ─────────────────────────────────────────────────────────

    #[test]
    fn test_context_error_display() {
        let e = ContextError("test error".to_string());
        assert!(e.to_string().contains("test error"));
    }
}
