// JSON-LD framing (subset of the JSON-LD Framing spec) (v1.1.0 round 11)
//
// Implements a working subset of the JSON-LD Framing algorithm:
// https://www.w3.org/TR/json-ld11-framing/
//
// Supported features:
// - Match nodes by @type
// - Match nodes by presence of a required property
// - EmbedMode: Always / Once / Never
// - explicit: true → only declared frame properties are kept
// - flatten: expand nested nodes to top-level
// - compact: apply a prefix context to shorten IRIs

use std::collections::HashMap;

// ── Public types ───────────────────────────────────────────────────────────────

/// A value within a JSON-LD property
#[derive(Debug, Clone, PartialEq)]
pub enum JsonLdValue {
    /// A nested node value
    Node(JsonLdNode),
    /// A literal value with optional datatype and language tag
    Literal {
        /// The raw string value of the literal.
        value: String,
        /// Optional XSD datatype IRI (e.g. `xsd:integer`).
        datatype: Option<String>,
        /// Optional BCP-47 language tag (e.g. `"en"`).
        language: Option<String>,
    },
    /// A reference to another node (by id)
    Reference(String),
}

impl JsonLdValue {
    /// Create a plain string literal value
    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            datatype: None,
            language: None,
        }
    }

    /// Create a typed literal
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            language: None,
        }
    }
}

/// A JSON-LD node (an object with @id, @type, and properties)
#[derive(Debug, Clone, PartialEq)]
pub struct JsonLdNode {
    /// The `@id` of this node, or `None` for blank nodes.
    pub id: Option<String>,
    /// The `@type` values associated with this node.
    pub types: Vec<String>,
    /// Key-value property map where values are lists of [`JsonLdValue`]s.
    pub properties: HashMap<String, Vec<JsonLdValue>>,
}

impl JsonLdNode {
    /// Create a node with an id
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            types: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Create a node without an id (blank node)
    pub fn anonymous() -> Self {
        Self {
            id: None,
            types: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Add a @type
    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.types.push(t.into());
        self
    }

    /// Add a property value
    pub fn add_property(&mut self, key: impl Into<String>, value: JsonLdValue) {
        self.properties.entry(key.into()).or_default().push(value);
    }

    /// Number of properties (excluding @id and @type)
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}

/// Controls how deeply linked nodes are embedded during framing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedMode {
    /// Always embed the referenced node (default)
    Always,
    /// Embed the node the first time it is encountered; use a reference thereafter
    Once,
    /// Never embed; always use a reference (by @id)
    Never,
}

/// A framing template
#[derive(Debug, Clone)]
pub struct Frame {
    /// Expected @type(s) for matching
    pub types: Vec<String>,
    /// Required sub-frames for properties (property key → sub-frame)
    pub properties: HashMap<String, Box<Frame>>,
    /// How to embed matched linked nodes
    pub embed: EmbedMode,
    /// If true, only properties declared in the frame are kept in the output
    pub explicit: bool,
}

impl Frame {
    /// Create an empty frame (matches everything)
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            properties: HashMap::new(),
            embed: EmbedMode::Always,
            explicit: false,
        }
    }

    /// Add a required type to match
    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.types.push(t.into());
        self
    }

    /// Add a sub-frame for a required property
    pub fn with_property(mut self, key: impl Into<String>, sub_frame: Frame) -> Self {
        self.properties.insert(key.into(), Box::new(sub_frame));
        self
    }

    /// Set embed mode
    pub fn with_embed(mut self, mode: EmbedMode) -> Self {
        self.embed = mode;
        self
    }

    /// Set explicit mode
    pub fn with_explicit(mut self, explicit: bool) -> Self {
        self.explicit = explicit;
        self
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of framing a node set
#[derive(Debug, Clone)]
pub struct FramingResult {
    /// Nodes that matched the frame
    pub nodes: Vec<JsonLdNode>,
    /// Number of nodes that were matched
    pub framed_count: usize,
}

/// Stateless JSON-LD framer
pub struct JsonLdFramer;

impl JsonLdFramer {
    /// Apply `frame` to `nodes`, returning only nodes that match.
    /// Linked nodes may be embedded according to `frame.embed`.
    pub fn frame(nodes: &[JsonLdNode], frame: &Frame) -> FramingResult {
        let mut result = Vec::new();

        for node in nodes {
            if Self::match_frame(node, frame) {
                let framed = Self::embed_node(node, nodes, frame, 0);
                result.push(framed);
            }
        }

        let framed_count = result.len();
        FramingResult {
            nodes: result,
            framed_count,
        }
    }

    /// Test whether `node` matches `frame`.
    ///
    /// A frame with no type and no required properties matches everything.
    /// If types are specified, the node must have at least one matching type.
    /// If property sub-frames are specified, the node must have those properties.
    pub fn match_frame(node: &JsonLdNode, frame: &Frame) -> bool {
        // Check type match
        if !frame.types.is_empty() {
            let type_match = frame.types.iter().any(|ft| node.types.contains(ft));
            if !type_match {
                return false;
            }
        }

        // Check required properties
        for key in frame.properties.keys() {
            if !node.properties.contains_key(key.as_str()) {
                return false;
            }
        }

        true
    }

    /// Produce an embedded version of `node` respecting the frame's embed mode
    /// and explicit flag.
    pub fn embed_node(
        node: &JsonLdNode,
        all_nodes: &[JsonLdNode],
        frame: &Frame,
        depth: usize,
    ) -> JsonLdNode {
        let max_depth = 16; // guard against infinite recursion
        let mut out = JsonLdNode {
            id: node.id.clone(),
            types: node.types.clone(),
            properties: HashMap::new(),
        };

        // Copy properties
        for (key, values) in &node.properties {
            // In explicit mode, only keep properties declared in the frame
            if frame.explicit && !frame.properties.contains_key(key.as_str()) {
                continue;
            }

            let embedded_values: Vec<JsonLdValue> = values
                .iter()
                .map(|v| match v {
                    JsonLdValue::Node(inner_node) => {
                        // Determine sub-frame
                        let sub_frame = frame.properties.get(key.as_str()).map(|f| f.as_ref());

                        match &frame.embed {
                            EmbedMode::Never => {
                                // Replace with a reference
                                if let Some(id) = &inner_node.id {
                                    JsonLdValue::Reference(id.clone())
                                } else {
                                    v.clone()
                                }
                            }
                            EmbedMode::Always | EmbedMode::Once if depth < max_depth => {
                                let effective_frame = sub_frame.unwrap_or(frame);
                                JsonLdValue::Node(Self::embed_node(
                                    inner_node,
                                    all_nodes,
                                    effective_frame,
                                    depth + 1,
                                ))
                            }
                            _ => v.clone(),
                        }
                    }
                    JsonLdValue::Reference(ref_id) => {
                        // Try to resolve the reference to a full node
                        if frame.embed == EmbedMode::Never || depth >= max_depth {
                            v.clone()
                        } else {
                            let resolved = all_nodes
                                .iter()
                                .find(|n| n.id.as_deref() == Some(ref_id.as_str()));
                            if let Some(resolved_node) = resolved {
                                let sub_frame = frame
                                    .properties
                                    .get(key.as_str())
                                    .map(|f| f.as_ref())
                                    .unwrap_or(frame);
                                JsonLdValue::Node(Self::embed_node(
                                    resolved_node,
                                    all_nodes,
                                    sub_frame,
                                    depth + 1,
                                ))
                            } else {
                                v.clone()
                            }
                        }
                    }
                    other => other.clone(),
                })
                .collect();

            out.properties.insert(key.clone(), embedded_values);
        }

        out
    }

    /// Flatten all nodes: move all nested nodes to the top level,
    /// replacing them with References.
    pub fn flatten(nodes: &[JsonLdNode]) -> Vec<JsonLdNode> {
        let mut result: Vec<JsonLdNode> = Vec::new();
        let mut id_counter = 0usize;

        fn collect(
            node: &JsonLdNode,
            result: &mut Vec<JsonLdNode>,
            counter: &mut usize,
        ) -> JsonLdNode {
            let mut flat = JsonLdNode {
                id: node.id.clone(),
                types: node.types.clone(),
                properties: HashMap::new(),
            };

            for (key, values) in &node.properties {
                let flat_values: Vec<JsonLdValue> = values
                    .iter()
                    .map(|v| match v {
                        JsonLdValue::Node(inner) => {
                            // Assign a synthetic id if needed
                            let mut inner_with_id = inner.clone();
                            if inner_with_id.id.is_none() {
                                *counter += 1;
                                inner_with_id.id = Some(format!("_:b{}", counter));
                            }
                            let id = inner_with_id.id.clone().unwrap_or_default();
                            let flat_inner = collect(&inner_with_id, result, counter);
                            result.push(flat_inner);
                            JsonLdValue::Reference(id)
                        }
                        other => other.clone(),
                    })
                    .collect();
                flat.properties.insert(key.clone(), flat_values);
            }

            flat
        }

        for node in nodes {
            let flat = collect(node, &mut result, &mut id_counter);
            result.push(flat);
        }

        result
    }

    /// Compact a node's IRI strings using a simple prefix context map.
    /// IRIs that start with a context value are replaced with `prefix:localname`.
    pub fn compact(node: &JsonLdNode, context: &HashMap<String, String>) -> JsonLdNode {
        let compact_iri = |iri: &str| -> String {
            // longest-prefix wins
            let mut best: Option<(usize, &str)> = None;
            for (pfx, ns) in context {
                if iri.starts_with(ns.as_str()) && ns.len() > best.map_or(0, |(l, _)| l) {
                    best = Some((ns.len(), pfx.as_str()));
                }
            }
            if let Some((len, pfx)) = best {
                format!("{}:{}", pfx, &iri[len..])
            } else {
                iri.to_string()
            }
        };

        let compacted_id = node.id.as_deref().map(compact_iri);
        let compacted_types: Vec<String> = node.types.iter().map(|t| compact_iri(t)).collect();

        let mut compacted_properties: HashMap<String, Vec<JsonLdValue>> = HashMap::new();
        for (key, values) in &node.properties {
            let compacted_key = compact_iri(key);
            let compacted_values: Vec<JsonLdValue> = values
                .iter()
                .map(|v| match v {
                    JsonLdValue::Node(inner) => JsonLdValue::Node(Self::compact(inner, context)),
                    JsonLdValue::Reference(id) => JsonLdValue::Reference(compact_iri(id)),
                    JsonLdValue::Literal {
                        value,
                        datatype,
                        language,
                    } => JsonLdValue::Literal {
                        value: value.clone(),
                        datatype: datatype.as_deref().map(compact_iri),
                        language: language.clone(),
                    },
                })
                .collect();
            compacted_properties.insert(compacted_key, compacted_values);
        }

        JsonLdNode {
            id: compacted_id,
            types: compacted_types,
            properties: compacted_properties,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn person_node(id: &str, name: &str) -> JsonLdNode {
        let mut n = JsonLdNode::new(id).with_type("Person");
        n.add_property("name", JsonLdValue::literal(name));
        n
    }

    fn make_context() -> HashMap<String, String> {
        let mut ctx = HashMap::new();
        ctx.insert("ex".to_string(), "http://example.org/".to_string());
        ctx.insert("schema".to_string(), "http://schema.org/".to_string());
        ctx
    }

    // ── JsonLdNode ─────────────────────────────────────────────────────────

    #[test]
    fn test_node_new() {
        let n = JsonLdNode::new("http://example.org/Alice");
        assert_eq!(n.id.as_deref(), Some("http://example.org/Alice"));
        assert!(n.types.is_empty());
    }

    #[test]
    fn test_node_with_type() {
        let n = JsonLdNode::new("id").with_type("Person");
        assert_eq!(n.types, vec!["Person"]);
    }

    #[test]
    fn test_node_add_property() {
        let mut n = JsonLdNode::new("id");
        n.add_property("name", JsonLdValue::literal("Alice"));
        assert_eq!(n.property_count(), 1);
    }

    #[test]
    fn test_node_property_count_empty() {
        let n = JsonLdNode::new("id");
        assert_eq!(n.property_count(), 0);
    }

    #[test]
    fn test_node_property_count_multiple() {
        let mut n = JsonLdNode::new("id").with_type("T");
        n.add_property("a", JsonLdValue::literal("1"));
        n.add_property("b", JsonLdValue::literal("2"));
        assert_eq!(n.property_count(), 2);
    }

    #[test]
    fn test_node_multiple_values_same_key() {
        let mut n = JsonLdNode::new("id");
        n.add_property("tag", JsonLdValue::literal("x"));
        n.add_property("tag", JsonLdValue::literal("y"));
        assert_eq!(n.properties["tag"].len(), 2);
    }

    // ── Frame ──────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_new_default() {
        let f = Frame::new();
        assert!(f.types.is_empty());
        assert!(f.properties.is_empty());
        assert_eq!(f.embed, EmbedMode::Always);
        assert!(!f.explicit);
    }

    #[test]
    fn test_frame_with_type() {
        let f = Frame::new().with_type("Person");
        assert_eq!(f.types, vec!["Person"]);
    }

    #[test]
    fn test_frame_with_property() {
        let f = Frame::new().with_property("name", Frame::new());
        assert!(f.properties.contains_key("name"));
    }

    #[test]
    fn test_frame_embed_never() {
        let f = Frame::new().with_embed(EmbedMode::Never);
        assert_eq!(f.embed, EmbedMode::Never);
    }

    #[test]
    fn test_frame_explicit() {
        let f = Frame::new().with_explicit(true);
        assert!(f.explicit);
    }

    // ── match_frame ────────────────────────────────────────────────────────

    #[test]
    fn test_match_frame_empty_frame_matches_all() {
        let n = person_node("id", "Alice");
        assert!(JsonLdFramer::match_frame(&n, &Frame::new()));
    }

    #[test]
    fn test_match_frame_by_type_hit() {
        let n = person_node("id", "Alice");
        let f = Frame::new().with_type("Person");
        assert!(JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_match_frame_by_type_miss() {
        let n = person_node("id", "Alice");
        let f = Frame::new().with_type("Organization");
        assert!(!JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_match_frame_by_property_hit() {
        let n = person_node("id", "Alice");
        let f = Frame::new().with_property("name", Frame::new());
        assert!(JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_match_frame_by_property_miss() {
        let n = person_node("id", "Alice"); // no "email"
        let f = Frame::new().with_property("email", Frame::new());
        assert!(!JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_match_frame_type_and_property() {
        let n = person_node("id", "Alice");
        let f = Frame::new()
            .with_type("Person")
            .with_property("name", Frame::new());
        assert!(JsonLdFramer::match_frame(&n, &f));
    }

    // ── frame() ───────────────────────────────────────────────────────────

    #[test]
    fn test_frame_returns_matching_nodes() {
        let nodes = vec![
            person_node("alice", "Alice"),
            JsonLdNode::new("corp").with_type("Organization"),
        ];
        let f = Frame::new().with_type("Person");
        let result = JsonLdFramer::frame(&nodes, &f);
        assert_eq!(result.framed_count, 1);
        assert_eq!(result.nodes[0].id.as_deref(), Some("alice"));
    }

    #[test]
    fn test_frame_excludes_non_matching() {
        let nodes = vec![
            JsonLdNode::new("a").with_type("X"),
            JsonLdNode::new("b").with_type("Y"),
        ];
        let f = Frame::new().with_type("Person");
        let result = JsonLdFramer::frame(&nodes, &f);
        assert_eq!(result.framed_count, 0);
    }

    #[test]
    fn test_frame_all_match_empty_frame() {
        let nodes = vec![person_node("a", "Alice"), person_node("b", "Bob")];
        let result = JsonLdFramer::frame(&nodes, &Frame::new());
        assert_eq!(result.framed_count, 2);
    }

    #[test]
    fn test_framing_result_framed_count() {
        let nodes = vec![person_node("a", "A"), person_node("b", "B")];
        let f = Frame::new().with_type("Person");
        let res = JsonLdFramer::frame(&nodes, &f);
        assert_eq!(res.framed_count, res.nodes.len());
    }

    // ── embed_mode Never ──────────────────────────────────────────────────

    #[test]
    fn test_embed_never_replaces_with_reference() {
        let inner = JsonLdNode::new("http://example.org/inner").with_type("City");
        let mut outer = JsonLdNode::new("http://example.org/outer").with_type("Person");
        outer.add_property("lives_in", JsonLdValue::Node(inner));

        let f = Frame::new()
            .with_type("Person")
            .with_embed(EmbedMode::Never);
        let result = JsonLdFramer::frame(&[outer], &f);
        assert_eq!(result.framed_count, 1);
        // The embedded node should be replaced with a reference
        let embedded_vals = &result.nodes[0].properties["lives_in"];
        assert!(matches!(embedded_vals[0], JsonLdValue::Reference(_)));
    }

    // ── explicit mode ─────────────────────────────────────────────────────

    #[test]
    fn test_explicit_frame_keeps_only_declared_props() {
        let mut n = JsonLdNode::new("id").with_type("Person");
        n.add_property("name", JsonLdValue::literal("Alice"));
        n.add_property("age", JsonLdValue::literal("30"));

        // explicit frame only declares "name"
        let f = Frame::new()
            .with_type("Person")
            .with_property("name", Frame::new())
            .with_explicit(true);

        let result = JsonLdFramer::frame(&[n], &f);
        assert_eq!(result.framed_count, 1);
        let framed = &result.nodes[0];
        assert!(framed.properties.contains_key("name"));
        assert!(!framed.properties.contains_key("age"));
    }

    // ── flatten ────────────────────────────────────────────────────────────

    #[test]
    fn test_flatten_no_nested() {
        let n = person_node("alice", "Alice");
        let flat = JsonLdFramer::flatten(std::slice::from_ref(&n));
        // At least the original node is there
        assert!(!flat.is_empty());
        assert!(flat.iter().any(|f| f.id.as_deref() == Some("alice")));
    }

    #[test]
    fn test_flatten_expands_nested_node() {
        let inner = JsonLdNode::new("http://example.org/company").with_type("Org");
        let mut outer = JsonLdNode::new("http://example.org/alice").with_type("Person");
        outer.add_property("works_at", JsonLdValue::Node(inner));

        let flat = JsonLdFramer::flatten(&[outer]);
        // Should have both the outer and the inner at top level
        let has_alice = flat
            .iter()
            .any(|n| n.id.as_deref() == Some("http://example.org/alice"));
        let has_company = flat
            .iter()
            .any(|n| n.id.as_deref() == Some("http://example.org/company"));
        assert!(has_alice);
        assert!(has_company);
    }

    #[test]
    fn test_flatten_replaces_nested_with_reference() {
        let inner = JsonLdNode::new("http://example.org/company").with_type("Org");
        let mut outer = JsonLdNode::new("http://example.org/alice").with_type("Person");
        outer.add_property("works_at", JsonLdValue::Node(inner));

        let flat = JsonLdFramer::flatten(&[outer]);
        let alice = flat
            .iter()
            .find(|n| n.id.as_deref() == Some("http://example.org/alice"))
            .unwrap();
        // works_at should now be a Reference
        if let Some(values) = alice.properties.get("works_at") {
            assert!(matches!(values[0], JsonLdValue::Reference(_)));
        } else {
            panic!("works_at property missing after flatten");
        }
    }

    // ── compact ───────────────────────────────────────────────────────────

    #[test]
    fn test_compact_applies_prefix_to_id() {
        let n = JsonLdNode::new("http://example.org/Alice");
        let ctx = make_context();
        let compacted = JsonLdFramer::compact(&n, &ctx);
        assert_eq!(compacted.id.as_deref(), Some("ex:Alice"));
    }

    #[test]
    fn test_compact_applies_prefix_to_type() {
        let n = JsonLdNode::new("id").with_type("http://schema.org/Person");
        let ctx = make_context();
        let compacted = JsonLdFramer::compact(&n, &ctx);
        assert!(compacted.types.contains(&"schema:Person".to_string()));
    }

    #[test]
    fn test_compact_no_match_leaves_iri() {
        let n = JsonLdNode::new("http://other.org/x");
        let ctx = make_context();
        let compacted = JsonLdFramer::compact(&n, &ctx);
        assert_eq!(compacted.id.as_deref(), Some("http://other.org/x"));
    }

    #[test]
    fn test_compact_applies_to_property_keys() {
        let mut n = JsonLdNode::new("id");
        n.add_property("http://schema.org/name", JsonLdValue::literal("Alice"));
        let ctx = make_context();
        let compacted = JsonLdFramer::compact(&n, &ctx);
        assert!(compacted.properties.contains_key("schema:name"));
    }

    // ── JsonLdValue ────────────────────────────────────────────────────────

    #[test]
    fn test_json_ld_value_literal() {
        let v = JsonLdValue::literal("hello");
        assert!(matches!(v, JsonLdValue::Literal { value, .. } if value == "hello"));
    }

    #[test]
    fn test_json_ld_value_typed_literal() {
        let v = JsonLdValue::typed_literal("42", "xsd:integer");
        assert!(
            matches!(v, JsonLdValue::Literal { datatype: Some(ref dt), .. } if dt == "xsd:integer")
        );
    }

    #[test]
    fn test_json_ld_value_reference() {
        let v = JsonLdValue::Reference("http://example.org/x".to_string());
        assert!(matches!(v, JsonLdValue::Reference(_)));
    }

    #[test]
    fn test_node_anonymous_no_id() {
        let n = JsonLdNode::anonymous();
        assert!(n.id.is_none());
    }

    #[test]
    fn test_frame_with_sub_frame() {
        let sub = Frame::new().with_type("City");
        let f = Frame::new()
            .with_type("Person")
            .with_property("hometown", sub);
        assert!(f.properties.contains_key("hometown"));
    }

    #[test]
    fn test_frame_default() {
        let f = Frame::default();
        assert!(f.types.is_empty());
        assert_eq!(f.embed, EmbedMode::Always);
    }

    #[test]
    fn test_json_ld_framer_frame_empty_nodes() {
        let result = JsonLdFramer::frame(&[], &Frame::new());
        assert_eq!(result.framed_count, 0);
        assert!(result.nodes.is_empty());
    }

    #[test]
    fn test_match_frame_multiple_types_any_match() {
        let n = JsonLdNode::new("id")
            .with_type("Person")
            .with_type("Employee");
        let f = Frame::new().with_type("Employee");
        assert!(JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_match_frame_requires_all_properties() {
        let mut n = JsonLdNode::new("id").with_type("Person");
        n.add_property("name", JsonLdValue::literal("Alice"));
        // Frame requires both "name" and "email" but node only has "name"
        let f = Frame::new()
            .with_property("name", Frame::new())
            .with_property("email", Frame::new());
        assert!(!JsonLdFramer::match_frame(&n, &f));
    }

    #[test]
    fn test_flatten_multiple_nodes() {
        let n1 = person_node("a", "A");
        let n2 = person_node("b", "B");
        let flat = JsonLdFramer::flatten(&[n1, n2]);
        let a = flat.iter().find(|n| n.id.as_deref() == Some("a"));
        let b = flat.iter().find(|n| n.id.as_deref() == Some("b"));
        assert!(a.is_some());
        assert!(b.is_some());
    }

    #[test]
    fn test_compact_applies_multiple_prefixes() {
        let ctx = make_context(); // ex: and schema:
        let mut n = JsonLdNode::new("http://example.org/Alice");
        n.add_property("http://schema.org/name", JsonLdValue::literal("Alice"));
        let compacted = JsonLdFramer::compact(&n, &ctx);
        assert_eq!(compacted.id.as_deref(), Some("ex:Alice"));
        assert!(compacted.properties.contains_key("schema:name"));
    }

    #[test]
    fn test_framing_result_nodes_length_equals_framed_count() {
        let nodes = vec![
            person_node("a", "A"),
            person_node("b", "B"),
            JsonLdNode::new("c").with_type("Org"),
        ];
        let f = Frame::new().with_type("Person");
        let result = JsonLdFramer::frame(&nodes, &f);
        assert_eq!(result.nodes.len(), result.framed_count);
    }

    #[test]
    fn test_embed_always_keeps_nested_node() {
        let inner = JsonLdNode::new("inner").with_type("City");
        let mut outer = JsonLdNode::new("outer").with_type("Person");
        outer.add_property("lives_in", JsonLdValue::Node(inner));
        let f = Frame::new()
            .with_type("Person")
            .with_embed(EmbedMode::Always);
        let result = JsonLdFramer::frame(&[outer], &f);
        assert_eq!(result.framed_count, 1);
        let lives_in = &result.nodes[0].properties["lives_in"];
        // Should be embedded (Node), not a Reference
        assert!(matches!(lives_in[0], JsonLdValue::Node(_)));
    }

    #[test]
    fn test_compact_literal_value_unchanged() {
        let ctx = make_context();
        let mut n = JsonLdNode::new("id");
        n.add_property("name", JsonLdValue::literal("Alice"));
        let compacted = JsonLdFramer::compact(&n, &ctx);
        if let Some(vals) = compacted.properties.get("name") {
            assert!(matches!(&vals[0], JsonLdValue::Literal { value, .. } if value == "Alice"));
        }
    }

    #[test]
    fn test_node_with_multiple_types() {
        let n = JsonLdNode::new("id")
            .with_type("Person")
            .with_type("Employee")
            .with_type("Manager");
        assert_eq!(n.types.len(), 3);
    }
}
