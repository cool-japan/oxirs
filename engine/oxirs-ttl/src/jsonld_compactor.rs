//! JSON-LD compaction: converts expanded JSON-LD to compact form using a context.
//!
//! # Overview
//!
//! JSON-LD compaction transforms a verbose expanded representation into a
//! concise compact form by applying prefix substitutions defined in a context.
//! This module implements the core subset of the JSON-LD 1.1 Compaction
//! Algorithm (W3C Recommendation) covering IRI compaction/expansion, nested
//! objects, and value recursion.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// JSON-LD Value enum
// ---------------------------------------------------------------------------

/// A JSON value that may appear inside a JSON-LD document.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonLdValue {
    /// JSON null
    Null,
    /// JSON boolean
    Bool(bool),
    /// JSON number (stored as f64)
    Number(f64),
    /// JSON string
    Str(String),
    /// JSON array
    Array(Vec<JsonLdValue>),
    /// JSON object (map)
    Object(HashMap<String, JsonLdValue>),
}

impl JsonLdValue {
    /// Return `true` if this value is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, JsonLdValue::Null)
    }

    /// Return the inner string slice, if any.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonLdValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the object map, if any.
    pub fn as_object(&self) -> Option<&HashMap<String, JsonLdValue>> {
        match self {
            JsonLdValue::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Return the array slice, if any.
    pub fn as_array(&self) -> Option<&[JsonLdValue]> {
        match self {
            JsonLdValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// A JSON-LD processing context holding prefix-to-IRI mappings and an optional
/// default vocabulary IRI.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Maps compact prefix (e.g. `"schema"`) to base IRI (e.g. `"http://schema.org/"`).
    pub prefix_map: HashMap<String, String>,
    /// Default vocabulary IRI — used when no matching prefix exists.
    pub default_vocab: Option<String>,
}

impl Context {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a prefix → IRI mapping.
    ///
    /// # Arguments
    /// * `prefix` — compact prefix (without trailing colon), e.g. `"schema"`
    /// * `iri`    — base IRI, e.g. `"http://schema.org/"`
    pub fn add_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefix_map.insert(prefix.to_string(), iri.to_string());
    }

    /// Set the default vocabulary IRI.
    pub fn set_default_vocab(&mut self, iri: &str) {
        self.default_vocab = Some(iri.to_string());
    }

    /// Compact a full IRI to its compact form using the registered prefixes.
    ///
    /// Attempts every registered prefix in order and returns the shortest
    /// compact form found.  Falls back to the original IRI when no prefix
    /// matches.
    ///
    /// # Example
    /// ```
    /// # use oxirs_ttl::jsonld_compactor::Context;
    /// let mut ctx = Context::new();
    /// ctx.add_prefix("schema", "http://schema.org/");
    /// assert_eq!(ctx.compact_iri("http://schema.org/name"), "schema:name");
    /// ```
    pub fn compact_iri(&self, iri: &str) -> String {
        // Strip angle brackets if present
        let iri = strip_angle_brackets(iri);

        let mut best: Option<String> = None;

        for (prefix, base) in &self.prefix_map {
            if iri.starts_with(base.as_str()) {
                let local = &iri[base.len()..];
                let compact = format!("{}:{}", prefix, local);
                best = Some(match best {
                    None => compact,
                    Some(prev) if compact.len() < prev.len() => compact,
                    Some(prev) => prev,
                });
            }
        }

        if let Some(c) = best {
            return c;
        }

        // Try default vocab
        if let Some(vocab) = &self.default_vocab {
            if iri.starts_with(vocab.as_str()) {
                return iri[vocab.len()..].to_string();
            }
        }

        iri.to_string()
    }

    /// Expand a compact IRI (e.g. `"schema:name"`) to its full form.
    ///
    /// Returns the original string if no matching prefix is found.
    ///
    /// # Example
    /// ```
    /// # use oxirs_ttl::jsonld_compactor::Context;
    /// let mut ctx = Context::new();
    /// ctx.add_prefix("schema", "http://schema.org/");
    /// assert_eq!(ctx.expand_iri("schema:name"), "http://schema.org/name");
    /// ```
    pub fn expand_iri(&self, compact: &str) -> String {
        // If it looks like an absolute IRI already, return as-is.
        if compact.starts_with("http://")
            || compact.starts_with("https://")
            || compact.starts_with("urn:")
        {
            return compact.to_string();
        }

        if let Some(colon_pos) = compact.find(':') {
            let prefix = &compact[..colon_pos];
            let local = &compact[colon_pos + 1..];

            if let Some(base) = self.prefix_map.get(prefix) {
                return format!("{}{}", base, local);
            }
        }

        // Try default vocab as expansion target
        if let Some(vocab) = &self.default_vocab {
            if !compact.contains(':') {
                return format!("{}{}", vocab, compact);
            }
        }

        compact.to_string()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn strip_angle_brackets(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('<') && s.ends_with('>') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Keywords that should be treated as JSON-LD keyword strings (never compacted
/// as if they were IRIs).
fn is_jsonld_keyword(key: &str) -> bool {
    matches!(
        key,
        "@context"
            | "@id"
            | "@type"
            | "@value"
            | "@language"
            | "@container"
            | "@graph"
            | "@set"
            | "@list"
            | "@reverse"
            | "@base"
            | "@vocab"
            | "@none"
            | "@included"
            | "@direction"
            | "@prefix"
            | "@protected"
            | "@propagate"
            | "@import"
            | "@nest"
            | "@version"
    )
}

// ---------------------------------------------------------------------------
// JsonLdCompactor
// ---------------------------------------------------------------------------

/// Compacts and expands JSON-LD documents according to a context.
pub struct JsonLdCompactor {
    context: Context,
}

impl JsonLdCompactor {
    /// Create a new compactor using the given context.
    pub fn new(context: Context) -> Self {
        Self { context }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Compact an expanded JSON-LD value (root entry point).
    pub fn compact(&self, expanded: &JsonLdValue) -> JsonLdValue {
        self.compact_value(expanded)
    }

    /// Expand a compact JSON-LD value (root entry point).
    pub fn expand(&self, compact: &JsonLdValue) -> JsonLdValue {
        self.expand_value(compact)
    }

    // -----------------------------------------------------------------------
    // Compaction internals
    // -----------------------------------------------------------------------

    /// Recursively compact a value.
    pub fn compact_value(&self, value: &JsonLdValue) -> JsonLdValue {
        match value {
            JsonLdValue::Object(map) => JsonLdValue::Object(self.compact_node(map)),
            JsonLdValue::Array(arr) => {
                JsonLdValue::Array(arr.iter().map(|v| self.compact_value(v)).collect())
            }
            // Scalars pass through unchanged
            other => other.clone(),
        }
    }

    /// Compact one JSON object node.
    pub fn compact_node(
        &self,
        node: &HashMap<String, JsonLdValue>,
    ) -> HashMap<String, JsonLdValue> {
        let mut out: HashMap<String, JsonLdValue> = HashMap::new();

        for (key, val) in node {
            let compacted_key = self.compact_key(key);
            let compacted_val = self.compact_node_value(key, val);
            out.insert(compacted_key, compacted_val);
        }

        out
    }

    /// Compact a key string.
    fn compact_key(&self, key: &str) -> String {
        if is_jsonld_keyword(key) {
            // Keep keywords as-is
            key.to_string()
        } else {
            self.context.compact_iri(key)
        }
    }

    /// Compact the value for a given key.
    fn compact_node_value(&self, key: &str, val: &JsonLdValue) -> JsonLdValue {
        match key {
            "@id" | "@type" => {
                // IRI values under @id and @type should be compacted
                self.compact_iri_value(val)
            }
            _ => self.compact_value(val),
        }
    }

    /// Compact an IRI value (string or array of strings).
    fn compact_iri_value(&self, val: &JsonLdValue) -> JsonLdValue {
        match val {
            JsonLdValue::Str(iri) => JsonLdValue::Str(self.context.compact_iri(iri)),
            JsonLdValue::Array(arr) => JsonLdValue::Array(
                arr.iter()
                    .map(|v| match v {
                        JsonLdValue::Str(iri) => JsonLdValue::Str(self.context.compact_iri(iri)),
                        other => other.clone(),
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Expansion internals
    // -----------------------------------------------------------------------

    fn expand_value(&self, value: &JsonLdValue) -> JsonLdValue {
        match value {
            JsonLdValue::Object(map) => JsonLdValue::Object(self.expand_node(map)),
            JsonLdValue::Array(arr) => {
                JsonLdValue::Array(arr.iter().map(|v| self.expand_value(v)).collect())
            }
            other => other.clone(),
        }
    }

    fn expand_node(&self, node: &HashMap<String, JsonLdValue>) -> HashMap<String, JsonLdValue> {
        let mut out: HashMap<String, JsonLdValue> = HashMap::new();

        for (key, val) in node {
            let expanded_key = self.expand_key(key);
            let expanded_val = self.expand_node_value(key, val);
            out.insert(expanded_key, expanded_val);
        }

        out
    }

    fn expand_key(&self, key: &str) -> String {
        if is_jsonld_keyword(key) {
            key.to_string()
        } else {
            self.context.expand_iri(key)
        }
    }

    fn expand_node_value(&self, key: &str, val: &JsonLdValue) -> JsonLdValue {
        match key {
            "@id" | "@type" => self.expand_iri_value(val),
            _ => self.expand_value(val),
        }
    }

    fn expand_iri_value(&self, val: &JsonLdValue) -> JsonLdValue {
        match val {
            JsonLdValue::Str(s) => JsonLdValue::Str(self.context.expand_iri(s)),
            JsonLdValue::Array(arr) => JsonLdValue::Array(
                arr.iter()
                    .map(|v| match v {
                        JsonLdValue::Str(s) => JsonLdValue::Str(self.context.expand_iri(s)),
                        other => other.clone(),
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Context helpers
    // -----------------------------------------------------------------------

    fn schema_context() -> Context {
        let mut ctx = Context::new();
        ctx.add_prefix("schema", "http://schema.org/");
        ctx.add_prefix("ex", "http://example.org/");
        ctx.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        ctx.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        ctx
    }

    // -----------------------------------------------------------------------
    // Context: compact_iri
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_iri_known_prefix() {
        let ctx = schema_context();
        assert_eq!(ctx.compact_iri("http://schema.org/name"), "schema:name");
    }

    #[test]
    fn test_compact_iri_example_prefix() {
        let ctx = schema_context();
        assert_eq!(ctx.compact_iri("http://example.org/Person"), "ex:Person");
    }

    #[test]
    fn test_compact_iri_unknown_prefix() {
        let ctx = schema_context();
        let unknown = "http://unknown.org/foo";
        assert_eq!(ctx.compact_iri(unknown), unknown);
    }

    #[test]
    fn test_compact_iri_empty_string() {
        let ctx = schema_context();
        assert_eq!(ctx.compact_iri(""), "");
    }

    #[test]
    fn test_compact_iri_angle_bracket_iri() {
        let ctx = schema_context();
        assert_eq!(ctx.compact_iri("<http://schema.org/name>"), "schema:name");
    }

    #[test]
    fn test_compact_iri_rdf_type() {
        let ctx = schema_context();
        assert_eq!(
            ctx.compact_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            "rdf:type"
        );
    }

    #[test]
    fn test_compact_iri_xsd_string() {
        let ctx = schema_context();
        assert_eq!(
            ctx.compact_iri("http://www.w3.org/2001/XMLSchema#string"),
            "xsd:string"
        );
    }

    #[test]
    fn test_compact_iri_multiple_candidates_picks_shortest() {
        let mut ctx = Context::new();
        ctx.add_prefix("a", "http://example.org/");
        ctx.add_prefix("ab", "http://example.org/b/");
        // "http://example.org/b/c" → "ab:c" (shorter than "a:b/c")
        let result = ctx.compact_iri("http://example.org/b/c");
        assert_eq!(result, "ab:c");
    }

    // -----------------------------------------------------------------------
    // Context: expand_iri
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_iri_known_prefix() {
        let ctx = schema_context();
        assert_eq!(ctx.expand_iri("schema:name"), "http://schema.org/name");
    }

    #[test]
    fn test_expand_iri_unknown_prefix() {
        let ctx = schema_context();
        assert_eq!(ctx.expand_iri("foo:bar"), "foo:bar");
    }

    #[test]
    fn test_expand_iri_absolute_http_passthrough() {
        let ctx = schema_context();
        let full = "http://schema.org/name";
        assert_eq!(ctx.expand_iri(full), full);
    }

    #[test]
    fn test_expand_iri_absolute_https_passthrough() {
        let ctx = schema_context();
        let full = "https://example.org/resource";
        assert_eq!(ctx.expand_iri(full), full);
    }

    #[test]
    fn test_expand_iri_urn_passthrough() {
        let ctx = schema_context();
        let urn = "urn:isbn:9780306406157";
        assert_eq!(ctx.expand_iri(urn), urn);
    }

    #[test]
    fn test_expand_iri_rdf_type() {
        let ctx = schema_context();
        assert_eq!(
            ctx.expand_iri("rdf:type"),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
    }

    // -----------------------------------------------------------------------
    // Context: round-trip compact → expand
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_schema_name() {
        let ctx = schema_context();
        let original = "http://schema.org/name";
        let compacted = ctx.compact_iri(original);
        let expanded = ctx.expand_iri(&compacted);
        assert_eq!(expanded, original);
    }

    #[test]
    fn test_round_trip_ex_person() {
        let ctx = schema_context();
        let original = "http://example.org/Person";
        let compacted = ctx.compact_iri(original);
        let expanded = ctx.expand_iri(&compacted);
        assert_eq!(expanded, original);
    }

    // -----------------------------------------------------------------------
    // Default vocab
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_vocab_compact() {
        let mut ctx = Context::new();
        ctx.set_default_vocab("http://schema.org/");
        assert_eq!(ctx.compact_iri("http://schema.org/name"), "name");
    }

    #[test]
    fn test_default_vocab_expand() {
        let mut ctx = Context::new();
        ctx.set_default_vocab("http://schema.org/");
        assert_eq!(ctx.expand_iri("name"), "http://schema.org/name");
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: scalar passthrough
    // -----------------------------------------------------------------------

    fn make_compactor() -> JsonLdCompactor {
        JsonLdCompactor::new(schema_context())
    }

    #[test]
    fn test_compact_null() {
        let c = make_compactor();
        assert_eq!(c.compact(&JsonLdValue::Null), JsonLdValue::Null);
    }

    #[test]
    fn test_compact_bool_true() {
        let c = make_compactor();
        assert_eq!(c.compact(&JsonLdValue::Bool(true)), JsonLdValue::Bool(true));
    }

    #[test]
    fn test_compact_bool_false() {
        let c = make_compactor();
        assert_eq!(
            c.compact(&JsonLdValue::Bool(false)),
            JsonLdValue::Bool(false)
        );
    }

    #[test]
    fn test_compact_number() {
        let c = make_compactor();
        assert_eq!(
            c.compact(&JsonLdValue::Number(42.0)),
            JsonLdValue::Number(42.0)
        );
    }

    #[test]
    fn test_compact_plain_string() {
        let c = make_compactor();
        let s = JsonLdValue::Str("hello".to_string());
        assert_eq!(c.compact(&s), s);
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: @id compaction
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_object_with_id() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/name".to_string()),
        );
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert_eq!(
                m.get("@id"),
                Some(&JsonLdValue::Str("schema:name".to_string()))
            );
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_compact_object_with_type_array() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@type".to_string(),
            JsonLdValue::Array(vec![
                JsonLdValue::Str("http://schema.org/Person".to_string()),
                JsonLdValue::Str("http://example.org/Employee".to_string()),
            ]),
        );
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            if let Some(JsonLdValue::Array(types)) = m.get("@type") {
                assert_eq!(types[0], JsonLdValue::Str("schema:Person".to_string()));
                assert_eq!(types[1], JsonLdValue::Str("ex:Employee".to_string()));
            } else {
                panic!("Expected array for @type");
            }
        } else {
            panic!("Expected object");
        }
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: predicate key compaction
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_predicate_key() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "http://schema.org/name".to_string(),
            JsonLdValue::Str("Alice".to_string()),
        );
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert!(m.contains_key("schema:name"));
            assert_eq!(
                m.get("schema:name"),
                Some(&JsonLdValue::Str("Alice".to_string()))
            );
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_compact_keywords_preserved() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@context".to_string(),
            JsonLdValue::Str("https://schema.org".to_string()),
        );
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert!(m.contains_key("@context"));
        } else {
            panic!("Expected object");
        }
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: nested objects
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_nested_object() {
        let c = make_compactor();
        let mut inner = HashMap::new();
        inner.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/Alice".to_string()),
        );
        let mut outer = HashMap::new();
        outer.insert(
            "http://schema.org/knows".to_string(),
            JsonLdValue::Object(inner),
        );
        let result = c.compact(&JsonLdValue::Object(outer));
        if let JsonLdValue::Object(m) = result {
            assert!(m.contains_key("schema:knows"));
            if let Some(JsonLdValue::Object(inner_m)) = m.get("schema:knows") {
                assert_eq!(
                    inner_m.get("@id"),
                    Some(&JsonLdValue::Str("ex:Alice".to_string()))
                );
            } else {
                panic!("Expected nested object");
            }
        } else {
            panic!("Expected object");
        }
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_array_of_objects() {
        let c = make_compactor();
        let mut node1 = HashMap::new();
        node1.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/A".to_string()),
        );
        let mut node2 = HashMap::new();
        node2.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/B".to_string()),
        );
        let arr = JsonLdValue::Array(vec![JsonLdValue::Object(node1), JsonLdValue::Object(node2)]);
        let result = c.compact(&arr);
        if let JsonLdValue::Array(items) = result {
            assert_eq!(items.len(), 2);
            for item in &items {
                if let JsonLdValue::Object(m) = item {
                    let id = m.get("@id").and_then(|v| v.as_str()).unwrap_or("");
                    assert!(id == "ex:A" || id == "ex:B");
                } else {
                    panic!("Expected object");
                }
            }
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_compact_array_of_scalars() {
        let c = make_compactor();
        let arr = JsonLdValue::Array(vec![
            JsonLdValue::Null,
            JsonLdValue::Bool(true),
            JsonLdValue::Number(1.0),
        ]);
        let result = c.compact(&arr);
        assert_eq!(
            result,
            JsonLdValue::Array(vec![
                JsonLdValue::Null,
                JsonLdValue::Bool(true),
                JsonLdValue::Number(1.0),
            ])
        );
    }

    // -----------------------------------------------------------------------
    // JsonLdCompactor: expand
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_object_with_id() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@id".to_string(),
            JsonLdValue::Str("schema:name".to_string()),
        );
        let result = c.expand(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert_eq!(
                m.get("@id"),
                Some(&JsonLdValue::Str("http://schema.org/name".to_string()))
            );
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_expand_predicate_key() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "schema:name".to_string(),
            JsonLdValue::Str("Alice".to_string()),
        );
        let result = c.expand(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert!(m.contains_key("http://schema.org/name"));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_expand_null() {
        let c = make_compactor();
        assert_eq!(c.expand(&JsonLdValue::Null), JsonLdValue::Null);
    }

    #[test]
    fn test_expand_bool() {
        let c = make_compactor();
        assert_eq!(
            c.expand(&JsonLdValue::Bool(false)),
            JsonLdValue::Bool(false)
        );
    }

    #[test]
    fn test_expand_number() {
        let c = make_compactor();
        assert_eq!(
            c.expand(&JsonLdValue::Number(2.71)),
            JsonLdValue::Number(2.71)
        );
    }

    // -----------------------------------------------------------------------
    // Round-trip compact → expand
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_object() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/name".to_string()),
        );
        node.insert(
            "http://schema.org/label".to_string(),
            JsonLdValue::Str("Name".to_string()),
        );
        let original = JsonLdValue::Object(node);
        let compacted = c.compact(&original);
        let expanded = c.expand(&compacted);

        // @id should round-trip
        if let (JsonLdValue::Object(orig_m), JsonLdValue::Object(exp_m)) = (&original, &expanded) {
            assert_eq!(orig_m.get("@id"), exp_m.get("@id"));
        } else {
            panic!("Expected objects");
        }
    }

    #[test]
    fn test_round_trip_nested() {
        let c = make_compactor();
        let mut inner = HashMap::new();
        inner.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/Alice".to_string()),
        );
        let mut outer = HashMap::new();
        outer.insert(
            "http://schema.org/knows".to_string(),
            JsonLdValue::Object(inner),
        );

        let original = JsonLdValue::Object(outer);
        let compacted = c.compact(&original);
        let expanded = c.expand(&compacted);

        // The expanded form should have "http://schema.org/knows" key
        if let JsonLdValue::Object(m) = &expanded {
            assert!(m.contains_key("http://schema.org/knows"));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_compact_then_expand_id_roundtrip() {
        let c = make_compactor();
        let full_iri = "http://example.org/Person";
        let compacted = c.context.compact_iri(full_iri);
        let expanded = c.context.expand_iri(&compacted);
        assert_eq!(expanded, full_iri);
    }

    #[test]
    fn test_compact_empty_object() {
        let c = make_compactor();
        let node: HashMap<String, JsonLdValue> = HashMap::new();
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert!(m.is_empty());
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_expand_empty_object() {
        let c = make_compactor();
        let node: HashMap<String, JsonLdValue> = HashMap::new();
        let result = c.expand(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert!(m.is_empty());
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_compact_empty_array() {
        let c = make_compactor();
        let arr = JsonLdValue::Array(vec![]);
        let result = c.compact(&arr);
        assert_eq!(result, JsonLdValue::Array(vec![]));
    }

    #[test]
    fn test_expand_empty_array() {
        let c = make_compactor();
        let arr = JsonLdValue::Array(vec![]);
        let result = c.expand(&arr);
        assert_eq!(result, JsonLdValue::Array(vec![]));
    }

    #[test]
    fn test_compact_type_single_string() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@type".to_string(),
            JsonLdValue::Str("http://schema.org/Person".to_string()),
        );
        let result = c.compact(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            assert_eq!(
                m.get("@type"),
                Some(&JsonLdValue::Str("schema:Person".to_string()))
            );
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_no_prefix_context() {
        let c = JsonLdCompactor::new(Context::new());
        let iri = "http://schema.org/name";
        assert_eq!(c.context.compact_iri(iri), iri);
        assert_eq!(c.context.expand_iri(iri), iri);
    }

    #[test]
    fn test_context_add_prefix_overwrite() {
        let mut ctx = Context::new();
        ctx.add_prefix("schema", "http://schema.org/");
        ctx.add_prefix("schema", "http://schema2.org/");
        assert_eq!(ctx.compact_iri("http://schema2.org/name"), "schema:name");
    }

    #[test]
    fn test_is_null_helper() {
        assert!(JsonLdValue::Null.is_null());
        assert!(!JsonLdValue::Bool(true).is_null());
    }

    #[test]
    fn test_as_str_helper() {
        let s = JsonLdValue::Str("hello".to_string());
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(JsonLdValue::Null.as_str(), None);
    }

    #[test]
    fn test_as_object_helper() {
        let m: HashMap<String, JsonLdValue> = HashMap::new();
        let obj = JsonLdValue::Object(m);
        assert!(obj.as_object().is_some());
        assert!(JsonLdValue::Null.as_object().is_none());
    }

    #[test]
    fn test_as_array_helper() {
        let arr = JsonLdValue::Array(vec![]);
        assert!(arr.as_array().is_some());
        assert!(JsonLdValue::Null.as_array().is_none());
    }

    #[test]
    fn test_expand_type_array() {
        let c = make_compactor();
        let mut node = HashMap::new();
        node.insert(
            "@type".to_string(),
            JsonLdValue::Array(vec![
                JsonLdValue::Str("schema:Person".to_string()),
                JsonLdValue::Str("ex:Employee".to_string()),
            ]),
        );
        let result = c.expand(&JsonLdValue::Object(node));
        if let JsonLdValue::Object(m) = result {
            if let Some(JsonLdValue::Array(types)) = m.get("@type") {
                assert_eq!(
                    types[0],
                    JsonLdValue::Str("http://schema.org/Person".to_string())
                );
                assert_eq!(
                    types[1],
                    JsonLdValue::Str("http://example.org/Employee".to_string())
                );
            } else {
                panic!("Expected array");
            }
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_compact_deep_nesting() {
        let c = make_compactor();
        let mut level3 = HashMap::new();
        level3.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/Z".to_string()),
        );
        let mut level2 = HashMap::new();
        level2.insert(
            "http://schema.org/address".to_string(),
            JsonLdValue::Object(level3),
        );
        let mut level1 = HashMap::new();
        level1.insert(
            "http://schema.org/person".to_string(),
            JsonLdValue::Object(level2),
        );
        let result = c.compact(&JsonLdValue::Object(level1));
        if let JsonLdValue::Object(m) = &result {
            assert!(m.contains_key("schema:person"));
            if let Some(JsonLdValue::Object(l2)) = m.get("schema:person") {
                assert!(l2.contains_key("schema:address"));
                if let Some(JsonLdValue::Object(l3)) = l2.get("schema:address") {
                    assert_eq!(l3.get("@id"), Some(&JsonLdValue::Str("ex:Z".to_string())));
                } else {
                    panic!("level3 not object");
                }
            } else {
                panic!("level2 not object");
            }
        } else {
            panic!("level1 not object");
        }
    }

    #[test]
    fn test_compact_mixed_array() {
        let c = make_compactor();
        let mut obj = HashMap::new();
        obj.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/name".to_string()),
        );
        let arr = JsonLdValue::Array(vec![
            JsonLdValue::Null,
            JsonLdValue::Number(1.0),
            JsonLdValue::Object(obj),
        ]);
        let result = c.compact(&arr);
        if let JsonLdValue::Array(items) = result {
            assert_eq!(items[0], JsonLdValue::Null);
            assert_eq!(items[1], JsonLdValue::Number(1.0));
            if let JsonLdValue::Object(m) = &items[2] {
                assert_eq!(
                    m.get("@id"),
                    Some(&JsonLdValue::Str("schema:name".to_string()))
                );
            } else {
                panic!("Expected object");
            }
        } else {
            panic!("Expected array");
        }
    }
}
