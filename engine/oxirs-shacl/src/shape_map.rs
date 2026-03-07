//! W3C ShapeMap Validation
//!
//! Implements the W3C ShapeMap specification for associating RDF nodes with
//! SHACL shapes and producing structured validation results.
//!
//! # ShapeMap Compact Syntax
//!
//! The compact syntax associates nodes with shapes using the `@` operator:
//!
//! ```text
//! <http://example.org/alice>@<http://schema.org/PersonShape>
//! <http://example.org/bob>@START
//! *@<http://schema.org/EntityShape>
//! ```
//!
//! # References
//!
//! - <https://shex.io/shape-map/>
//! - <https://www.w3.org/TR/shacl/>

use std::collections::HashMap;
use std::fmt;

// ─── Node selector ───────────────────────────────────────────────────────────

/// Selects which RDF nodes an association applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeSelector {
    /// A specific IRI node
    NodeIri(String),
    /// A regex pattern matched against node IRIs
    Pattern(String),
    /// Matches all nodes in the graph
    Wildcard,
    /// The focus node (used in validation contexts)
    Focus,
}

impl fmt::Display for NodeSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeIri(iri) => write!(f, "<{iri}>"),
            Self::Pattern(p) => write!(f, "/{p}/"),
            Self::Wildcard => write!(f, "*"),
            Self::Focus => write!(f, "FOCUS"),
        }
    }
}

// ─── Shape label ─────────────────────────────────────────────────────────────

/// Identifies which shape(s) a node should be validated against.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeLabel {
    /// A specific shape identified by IRI
    Iri(String),
    /// The start shape of the schema
    Start,
    /// All shapes in the schema
    All,
}

impl fmt::Display for ShapeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{iri}>"),
            Self::Start => write!(f, "START"),
            Self::All => write!(f, "*"),
        }
    }
}

// ─── Association ─────────────────────────────────────────────────────────────

/// A single node → shape association in a ShapeMap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeAssociation {
    /// The node selector
    pub node: NodeSelector,
    /// The shape label
    pub shape: ShapeLabel,
}

impl fmt::Display for ShapeAssociation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.node, self.shape)
    }
}

// ─── ShapeMap ─────────────────────────────────────────────────────────────────

/// A collection of node → shape associations.
#[derive(Debug, Clone, Default)]
pub struct ShapeMap {
    /// The ordered list of associations
    pub associations: Vec<ShapeAssociation>,
}

impl ShapeMap {
    /// Create an empty ShapeMap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an association.
    pub fn add(&mut self, node: NodeSelector, shape: ShapeLabel) {
        self.associations.push(ShapeAssociation { node, shape });
    }

    /// Number of associations.
    pub fn len(&self) -> usize {
        self.associations.len()
    }

    /// Returns `true` if there are no associations.
    pub fn is_empty(&self) -> bool {
        self.associations.is_empty()
    }
}

// ─── Validation status ────────────────────────────────────────────────────────

/// Result of validating a node against a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationStatus {
    /// The node conforms to the shape
    Conformant,
    /// The node does not conform to the shape
    NonConformant,
    /// The validation could not be completed (e.g. missing data)
    Incomplete,
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conformant => write!(f, "conformant"),
            Self::NonConformant => write!(f, "nonConformant"),
            Self::Incomplete => write!(f, "incomplete"),
        }
    }
}

// ─── Validation result ────────────────────────────────────────────────────────

/// Result of validating a single node against a single shape.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The node IRI that was validated
    pub node: String,
    /// The shape IRI that was applied
    pub shape: String,
    /// The conformance status
    pub status: ValidationStatus,
    /// Human-readable messages (empty if conformant)
    pub messages: Vec<String>,
}

impl ValidationResult {
    /// Create a conformant result.
    pub fn conformant(node: impl Into<String>, shape: impl Into<String>) -> Self {
        Self {
            node: node.into(),
            shape: shape.into(),
            status: ValidationStatus::Conformant,
            messages: Vec::new(),
        }
    }

    /// Create a non-conformant result with messages.
    pub fn non_conformant(
        node: impl Into<String>,
        shape: impl Into<String>,
        messages: Vec<String>,
    ) -> Self {
        Self {
            node: node.into(),
            shape: shape.into(),
            status: ValidationStatus::NonConformant,
            messages,
        }
    }

    /// Create an incomplete result.
    pub fn incomplete(
        node: impl Into<String>,
        shape: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            node: node.into(),
            shape: shape.into(),
            status: ValidationStatus::Incomplete,
            messages: vec![reason.into()],
        }
    }
}

// ─── Shape-map parse error ────────────────────────────────────────────────────

/// Error returned when parsing a ShapeMap compact syntax string.
#[derive(Debug, Clone)]
pub struct ShapeMapError(pub String);

impl fmt::Display for ShapeMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShapeMapError: {}", self.0)
    }
}

impl std::error::Error for ShapeMapError {}

// ─── Parser ───────────────────────────────────────────────────────────────────

/// Parse a ShapeMap compact-syntax string into a `ShapeMap`.
///
/// Supported syntax per association (one per line or semicolon-separated):
///
/// ```text
/// <nodeIRI>@<shapeIRI>
/// <nodeIRI>@START
/// <nodeIRI>@*
/// *@<shapeIRI>
/// FOCUS@<shapeIRI>
/// /<regex>/@<shapeIRI>
/// ```
pub fn parse_shape_map(text: &str) -> Result<ShapeMap, ShapeMapError> {
    let mut map = ShapeMap::new();

    for raw_line in text.lines() {
        // Also split on ';'
        for part in raw_line.split(';') {
            let part = part.trim();
            if part.is_empty() || part.starts_with('#') {
                continue;
            }

            let at_pos = find_at_separator(part)
                .ok_or_else(|| ShapeMapError(format!("missing '@' separator in: '{part}'")))?;

            let node_str = part[..at_pos].trim();
            let shape_str = part[at_pos + 1..].trim();

            let node = parse_node_selector(node_str)?;
            let shape = parse_shape_label(shape_str)?;

            map.add(node, shape);
        }
    }

    Ok(map)
}

/// Find the position of the `@` separator that divides node from shape.
///
/// We need to skip `@` characters inside `<...>` and `/.../`.
fn find_at_separator(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
            }
            b'/' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'/' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'@' => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

fn parse_node_selector(s: &str) -> Result<NodeSelector, ShapeMapError> {
    if s == "*" {
        Ok(NodeSelector::Wildcard)
    } else if s.eq_ignore_ascii_case("FOCUS") {
        Ok(NodeSelector::Focus)
    } else if s.starts_with('<') && s.ends_with('>') {
        Ok(NodeSelector::NodeIri(s[1..s.len() - 1].to_string()))
    } else if s.starts_with('/') && s.ends_with('/') && s.len() >= 2 {
        Ok(NodeSelector::Pattern(s[1..s.len() - 1].to_string()))
    } else {
        Err(ShapeMapError(format!("unrecognised node selector: '{s}'")))
    }
}

fn parse_shape_label(s: &str) -> Result<ShapeLabel, ShapeMapError> {
    if s == "*" {
        Ok(ShapeLabel::All)
    } else if s.eq_ignore_ascii_case("START") {
        Ok(ShapeLabel::Start)
    } else if s.starts_with('<') && s.ends_with('>') {
        Ok(ShapeLabel::Iri(s[1..s.len() - 1].to_string()))
    } else {
        Err(ShapeMapError(format!("unrecognised shape label: '{s}'")))
    }
}

// ─── Validator ────────────────────────────────────────────────────────────────

/// A simple ShapeMap-based validation engine.
///
/// Validation logic is necessarily lightweight here since full SHACL evaluation
/// is provided by the rest of the `oxirs-shacl` crate. The validator applies
/// built-in heuristics:
///
/// - Wildcard node selectors expand to all nodes in the graph
/// - Shape conformance is checked by looking for a matching constraint triple
/// - If a shape IRI is referenced but has no constraints in the graph, the
///   result is `Incomplete`
pub struct ShapeMapValidator {
    /// Map from shape IRI → set of (property, value) constraints
    /// (extracted from graph triples in SHACL vocabulary)
    constraints: HashMap<String, Vec<(String, String)>>,
}

impl ShapeMapValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
        }
    }

    /// Register a simple property-value constraint for a shape.
    ///
    /// Used in tests / simple scenarios. For full SHACL support use the
    /// `oxirs-shacl` validation engine directly.
    pub fn add_constraint(
        &mut self,
        shape_iri: impl Into<String>,
        property: impl Into<String>,
        value: impl Into<String>,
    ) {
        self.constraints
            .entry(shape_iri.into())
            .or_default()
            .push((property.into(), value.into()));
    }

    /// Validate a ShapeMap against a set of graph triples.
    ///
    /// Graph triples are passed as `(subject, predicate, object)` string tuples.
    pub fn validate(
        &self,
        shape_map: &ShapeMap,
        graph_triples: &[(String, String, String)],
    ) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        // Collect all distinct subjects in the graph
        let all_nodes: Vec<String> = {
            let mut nodes: Vec<String> = graph_triples.iter().map(|(s, _, _)| s.clone()).collect();
            nodes.sort();
            nodes.dedup();
            nodes
        };

        // Build a per-subject property map for fast lookup
        let mut subj_props: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for (s, p, o) in graph_triples {
            subj_props
                .entry(s.as_str())
                .or_default()
                .push((p.as_str(), o.as_str()));
        }

        for assoc in &shape_map.associations {
            // Expand node selector to concrete IRIs
            let node_iris: Vec<String> = match &assoc.node {
                NodeSelector::NodeIri(iri) => vec![iri.clone()],
                NodeSelector::Wildcard => all_nodes.clone(),
                NodeSelector::Focus => {
                    // In a standalone validator there is no focus node context —
                    // report Incomplete for each node in the graph
                    for node in &all_nodes {
                        results.push(ValidationResult::incomplete(
                            node,
                            assoc.shape.to_string(),
                            "FOCUS selector requires a query context".to_string(),
                        ));
                    }
                    continue;
                }
                NodeSelector::Pattern(pat) => {
                    // Simple substring match for pattern
                    all_nodes
                        .iter()
                        .filter(|n| n.contains(pat.as_str()))
                        .cloned()
                        .collect()
                }
            };

            // Expand shape label
            let shape_iris: Vec<String> = match &assoc.shape {
                ShapeLabel::Iri(iri) => vec![iri.clone()],
                ShapeLabel::Start => {
                    // Use the first registered constraint shape as START, if any
                    self.constraints.keys().take(1).cloned().collect()
                }
                ShapeLabel::All => self.constraints.keys().cloned().collect(),
            };

            for node_iri in &node_iris {
                for shape_iri in &shape_iris {
                    let result = self.validate_node_shape(node_iri, shape_iri, &subj_props);
                    results.push(result);
                }
            }
        }

        results
    }

    fn validate_node_shape(
        &self,
        node: &str,
        shape: &str,
        subj_props: &HashMap<&str, Vec<(&str, &str)>>,
    ) -> ValidationResult {
        let Some(constraints) = self.constraints.get(shape) else {
            return ValidationResult::incomplete(
                node,
                shape,
                format!("shape '{shape}' has no registered constraints"),
            );
        };

        let props = subj_props.get(node).cloned().unwrap_or_default();
        let mut messages = Vec::new();

        for (req_prop, req_val) in constraints {
            let found = props.iter().any(|(p, o)| {
                *p == req_prop.as_str() && (req_val.is_empty() || *o == req_val.as_str())
            });
            if !found {
                messages.push(format!(
                    "Node <{node}> missing required property <{req_prop}> = '{req_val}' for shape <{shape}>"
                ));
            }
        }

        if messages.is_empty() {
            ValidationResult::conformant(node, shape)
        } else {
            ValidationResult::non_conformant(node, shape, messages)
        }
    }
}

impl Default for ShapeMapValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── JSON rendering ───────────────────────────────────────────────────────────

/// Render validation results to a JSON string following the W3C ShapeMap spec.
///
/// Each result is an object with `node`, `shape`, `status`, and optionally `reason`.
pub fn render_json(results: &[ValidationResult]) -> String {
    let mut out = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!("    \"node\": \"{}\",\n", json_escape(&r.node)));
        out.push_str(&format!("    \"shape\": \"{}\",\n", json_escape(&r.shape)));
        out.push_str(&format!("    \"status\": \"{}\"", r.status));
        if !r.messages.is_empty() {
            out.push_str(",\n    \"reason\": [");
            for (j, msg) in r.messages.iter().enumerate() {
                out.push_str(&format!("\"{}\"", json_escape(msg)));
                if j + 1 < r.messages.len() {
                    out.push_str(", ");
                }
            }
            out.push(']');
        }
        out.push_str("\n  }");
        if i + 1 < results.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push(']');
    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triples(v: &[(&str, &str, &str)]) -> Vec<(String, String, String)> {
        v.iter()
            .map(|(s, p, o)| (s.to_string(), p.to_string(), o.to_string()))
            .collect()
    }

    // ── NodeSelector ────────────────────────────────────────────────────────

    #[test]
    fn test_node_selector_wildcard() {
        assert_eq!(NodeSelector::Wildcard.to_string(), "*");
    }

    #[test]
    fn test_node_selector_iri() {
        let ns = NodeSelector::NodeIri("http://example.org/alice".to_string());
        assert_eq!(ns.to_string(), "<http://example.org/alice>");
    }

    #[test]
    fn test_node_selector_pattern() {
        let ns = NodeSelector::Pattern("example".to_string());
        assert_eq!(ns.to_string(), "/example/");
    }

    #[test]
    fn test_node_selector_focus() {
        assert_eq!(NodeSelector::Focus.to_string(), "FOCUS");
    }

    // ── ShapeLabel ───────────────────────────────────────────────────────────

    #[test]
    fn test_shape_label_iri() {
        let sl = ShapeLabel::Iri("http://schema.org/PersonShape".to_string());
        assert_eq!(sl.to_string(), "<http://schema.org/PersonShape>");
    }

    #[test]
    fn test_shape_label_start() {
        assert_eq!(ShapeLabel::Start.to_string(), "START");
    }

    #[test]
    fn test_shape_label_all() {
        assert_eq!(ShapeLabel::All.to_string(), "*");
    }

    // ── ShapeMap ─────────────────────────────────────────────────────────────

    #[test]
    fn test_shape_map_new_empty() {
        let sm = ShapeMap::new();
        assert!(sm.is_empty());
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn test_shape_map_add() {
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://example.org/alice".to_string()),
            ShapeLabel::Iri("http://schema.org/PersonShape".to_string()),
        );
        assert_eq!(sm.len(), 1);
        assert!(!sm.is_empty());
    }

    #[test]
    fn test_association_display() {
        let assoc = ShapeAssociation {
            node: NodeSelector::NodeIri("http://ex.org/n".to_string()),
            shape: ShapeLabel::Iri("http://ex.org/s".to_string()),
        };
        assert!(assoc.to_string().contains('@'));
    }

    // ── Parsing ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_iri_association() {
        let text = "<http://example.org/alice>@<http://schema.org/PersonShape>";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.len(), 1);
        assert_eq!(
            sm.associations[0].node,
            NodeSelector::NodeIri("http://example.org/alice".to_string())
        );
        assert_eq!(
            sm.associations[0].shape,
            ShapeLabel::Iri("http://schema.org/PersonShape".to_string())
        );
    }

    #[test]
    fn test_parse_wildcard_node() {
        let text = "*@<http://schema.org/EntityShape>";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.len(), 1);
        assert_eq!(sm.associations[0].node, NodeSelector::Wildcard);
    }

    #[test]
    fn test_parse_start_shape() {
        let text = "<http://example.org/bob>@START";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.associations[0].shape, ShapeLabel::Start);
    }

    #[test]
    fn test_parse_all_shape() {
        let text = "<http://example.org/x>@*";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.associations[0].shape, ShapeLabel::All);
    }

    #[test]
    fn test_parse_pattern_node() {
        let text = "/alice/@<http://schema.org/PersonShape>";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(
            sm.associations[0].node,
            NodeSelector::Pattern("alice".to_string())
        );
    }

    #[test]
    fn test_parse_focus_node() {
        let text = "FOCUS@<http://schema.org/PersonShape>";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.associations[0].node, NodeSelector::Focus);
    }

    #[test]
    fn test_parse_multiple_lines() {
        let text = "<http://ex.org/a>@<http://ex.org/S>\n<http://ex.org/b>@START";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.len(), 2);
    }

    #[test]
    fn test_parse_semicolon_separated() {
        let text = "<http://ex.org/a>@<http://ex.org/S>;<http://ex.org/b>@START";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.len(), 2);
    }

    #[test]
    fn test_parse_empty_string() {
        let sm = parse_shape_map("").expect("parse ok");
        assert!(sm.is_empty());
    }

    #[test]
    fn test_parse_comments() {
        let text = "# This is a comment\n<http://ex.org/a>@<http://ex.org/S>";
        let sm = parse_shape_map(text).expect("parse ok");
        assert_eq!(sm.len(), 1);
    }

    #[test]
    fn test_parse_error_missing_at() {
        let result = parse_shape_map("<http://ex.org/a><http://ex.org/S>");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_bad_node() {
        let result = parse_shape_map("badnode@<http://ex.org/S>");
        assert!(result.is_err());
    }

    // ── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn test_validate_conformant() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint(
            "http://schema.org/PersonShape",
            "http://schema.org/name",
            "",
        );
        let graph = triples(&[(
            "http://example.org/alice",
            "http://schema.org/name",
            "Alice",
        )]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://example.org/alice".to_string()),
            ShapeLabel::Iri("http://schema.org/PersonShape".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, ValidationStatus::Conformant);
    }

    #[test]
    fn test_validate_non_conformant() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint(
            "http://schema.org/PersonShape",
            "http://schema.org/name",
            "",
        );
        let graph = triples(&[("http://example.org/bob", "http://schema.org/age", "30")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://example.org/bob".to_string()),
            ShapeLabel::Iri("http://schema.org/PersonShape".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results[0].status, ValidationStatus::NonConformant);
        assert!(!results[0].messages.is_empty());
    }

    #[test]
    fn test_validate_incomplete_unknown_shape() {
        let validator = ShapeMapValidator::new();
        let graph = triples(&[("http://ex.org/a", "http://ex.org/p", "v")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://ex.org/a".to_string()),
            ShapeLabel::Iri("http://ex.org/UnknownShape".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results[0].status, ValidationStatus::Incomplete);
    }

    #[test]
    fn test_validate_wildcard_expands_to_all_nodes() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint("http://ex.org/S", "http://ex.org/p", "");
        let graph = triples(&[
            ("http://ex.org/a", "http://ex.org/p", "x"),
            ("http://ex.org/b", "http://ex.org/p", "y"),
        ]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::Wildcard,
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_validate_pattern_selector() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint("http://ex.org/S", "http://ex.org/p", "");
        let graph = triples(&[
            ("http://ex.org/alice_1", "http://ex.org/p", "v"),
            ("http://ex.org/bob_1", "http://ex.org/p", "v"),
            ("http://other.org/charlie", "http://ex.org/p", "v"),
        ]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::Pattern("ex.org/alice".to_string()),
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        // Only alice matches
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node, "http://ex.org/alice_1");
        assert_eq!(results[0].status, ValidationStatus::Conformant);
    }

    #[test]
    fn test_validate_multiple_constraints() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint("http://ex.org/S", "http://ex.org/name", "");
        validator.add_constraint("http://ex.org/S", "http://ex.org/age", "");
        let graph = triples(&[("http://ex.org/n", "http://ex.org/name", "N")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://ex.org/n".to_string()),
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results[0].status, ValidationStatus::NonConformant);
        // Missing age
        assert_eq!(results[0].messages.len(), 1);
    }

    #[test]
    fn test_validate_focus_returns_incomplete() {
        let validator = ShapeMapValidator::new();
        let graph = triples(&[("http://ex.org/a", "http://ex.org/p", "v")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::Focus,
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        for r in &results {
            assert_eq!(r.status, ValidationStatus::Incomplete);
        }
    }

    // ── JSON rendering ────────────────────────────────────────────────────────

    #[test]
    fn test_render_json_conformant() {
        let results = vec![ValidationResult::conformant(
            "http://ex.org/alice",
            "http://ex.org/S",
        )];
        let json = render_json(&results);
        assert!(json.contains("conformant"));
        assert!(json.contains("alice"));
        assert!(json.contains('['));
        assert!(json.contains(']'));
    }

    #[test]
    fn test_render_json_non_conformant() {
        let results = vec![ValidationResult::non_conformant(
            "http://ex.org/bob",
            "http://ex.org/S",
            vec!["Missing required property".to_string()],
        )];
        let json = render_json(&results);
        assert!(json.contains("nonConformant"));
        assert!(json.contains("reason"));
    }

    #[test]
    fn test_render_json_empty() {
        let json = render_json(&[]);
        assert!(json.contains('['));
        assert!(json.contains(']'));
    }

    #[test]
    fn test_render_json_multiple() {
        let results = vec![
            ValidationResult::conformant("http://ex.org/a", "http://ex.org/S"),
            ValidationResult::non_conformant(
                "http://ex.org/b",
                "http://ex.org/S",
                vec!["err".to_string()],
            ),
        ];
        let json = render_json(&results);
        assert!(json.contains("conformant"));
        assert!(json.contains("nonConformant"));
    }

    #[test]
    fn test_render_json_escapes() {
        let results = vec![ValidationResult::non_conformant(
            r#"http://ex.org/n"ode"#,
            "http://ex.org/S",
            vec!["line1\nline2".to_string()],
        )];
        let json = render_json(&results);
        // Quotes in IRI should be escaped
        assert!(json.contains("\\\""));
    }

    #[test]
    fn test_validation_status_display() {
        assert_eq!(ValidationStatus::Conformant.to_string(), "conformant");
        assert_eq!(ValidationStatus::NonConformant.to_string(), "nonConformant");
        assert_eq!(ValidationStatus::Incomplete.to_string(), "incomplete");
    }

    #[test]
    fn test_specific_value_constraint() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint("http://ex.org/S", "http://ex.org/type", "Person");
        let graph = triples(&[("http://ex.org/a", "http://ex.org/type", "Person")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://ex.org/a".to_string()),
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results[0].status, ValidationStatus::Conformant);
    }

    #[test]
    fn test_specific_value_constraint_wrong_value() {
        let mut validator = ShapeMapValidator::new();
        validator.add_constraint("http://ex.org/S", "http://ex.org/type", "Person");
        let graph = triples(&[("http://ex.org/a", "http://ex.org/type", "Organization")]);
        let mut sm = ShapeMap::new();
        sm.add(
            NodeSelector::NodeIri("http://ex.org/a".to_string()),
            ShapeLabel::Iri("http://ex.org/S".to_string()),
        );
        let results = validator.validate(&sm, &graph);
        assert_eq!(results[0].status, ValidationStatus::NonConformant);
    }
}
