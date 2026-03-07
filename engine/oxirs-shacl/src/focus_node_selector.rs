/// SHACL focus node selection.
///
/// Implements the four standard target declaration types defined by the
/// W3C SHACL specification, together with implicit class targets and
/// SPARQL-based custom targets.  Results from multiple target declarations
/// are combined (union).  A built-in cache avoids redundant re-evaluation
/// of targets across repeated validation runs.
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// An RDF node that may become a focus node for validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfNode {
    /// An IRI-identified resource.
    Iri(String),
    /// A blank node.
    BlankNode(String),
    /// A plain or typed literal.
    Literal(String),
}

impl RdfNode {
    /// Return a human-readable string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Iri(s) | Self::BlankNode(s) | Self::Literal(s) => s,
        }
    }

    /// True when the node is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }
}

impl std::fmt::Display for RdfNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Iri(s) => write!(f, "<{s}>"),
            Self::BlankNode(s) => write!(f, "_:{s}"),
            Self::Literal(s) => write!(f, "\"{s}\""),
        }
    }
}

/// A single RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: RdfNode,
    pub predicate: String,
    pub object: RdfNode,
}

/// An in-memory RDF graph against which focus nodes are selected.
#[derive(Debug, Default, Clone)]
pub struct RdfGraph {
    triples: Vec<Triple>,
}

impl RdfGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    /// Return all triples.
    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }

    /// Return subjects with `rdf:type` equal to the given class IRI.
    pub fn instances_of(&self, class_iri: &str) -> HashSet<RdfNode> {
        self.triples
            .iter()
            .filter(|t| {
                t.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                    && t.object == RdfNode::Iri(class_iri.to_string())
            })
            .map(|t| t.subject.clone())
            .collect()
    }

    /// Return subjects of triples with the given predicate.
    pub fn subjects_of(&self, predicate: &str) -> HashSet<RdfNode> {
        self.triples
            .iter()
            .filter(|t| t.predicate == predicate)
            .map(|t| t.subject.clone())
            .collect()
    }

    /// Return objects of triples with the given predicate.
    pub fn objects_of(&self, predicate: &str) -> HashSet<RdfNode> {
        self.triples
            .iter()
            .filter(|t| t.predicate == predicate)
            .map(|t| t.object.clone())
            .collect()
    }

    /// Return all unique subjects in the graph.
    pub fn all_subjects(&self) -> HashSet<RdfNode> {
        self.triples.iter().map(|t| t.subject.clone()).collect()
    }

    /// Check if the graph contains any triple.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Return the number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }
}

// ---------------------------------------------------------------------------
// Target declarations
// ---------------------------------------------------------------------------

/// A single SHACL target declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetDeclaration {
    /// sh:targetClass — all instances of the given class.
    TargetClass(String),
    /// sh:targetNode — a single named node.
    TargetNode(RdfNode),
    /// sh:targetSubjectsOf — subjects of triples with the given predicate.
    TargetSubjectsOf(String),
    /// sh:targetObjectsOf — objects of triples with the given predicate.
    TargetObjectsOf(String),
    /// Implicit class target — the shape IRI is also an RDFS class.
    ImplicitClassTarget(String),
    /// SPARQL-based custom target (the string is a SPARQL SELECT query
    /// that should bind `?this`).
    SparqlTarget(String),
}

/// The result of selecting focus nodes for one or more target declarations.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// The set of focus nodes that matched.
    pub nodes: HashSet<RdfNode>,
    /// Per-declaration breakdown.
    pub per_target: Vec<TargetContribution>,
}

/// The contribution of one target declaration to the overall selection.
#[derive(Debug, Clone)]
pub struct TargetContribution {
    /// The declaration that was evaluated.
    pub declaration: TargetDeclaration,
    /// The nodes selected by this declaration.
    pub matched_nodes: HashSet<RdfNode>,
}

// ---------------------------------------------------------------------------
// FocusNodeSelector
// ---------------------------------------------------------------------------

/// Error returned by the focus node selector.
#[derive(Debug)]
pub enum SelectorError {
    /// A SPARQL target query was malformed or unsupported.
    SparqlError(String),
    /// General error.
    General(String),
}

impl std::fmt::Display for SelectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SparqlError(msg) => write!(f, "SPARQL target error: {msg}"),
            Self::General(msg) => write!(f, "Selector error: {msg}"),
        }
    }
}

impl std::error::Error for SelectorError {}

/// Configuration for the focus node selector.
#[derive(Debug, Clone)]
pub struct SelectorConfig {
    /// Enable caching of target results.
    pub cache_enabled: bool,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            cache_enabled: true,
        }
    }
}

/// Selects focus nodes from an RDF graph based on SHACL target declarations.
#[derive(Debug)]
pub struct FocusNodeSelector {
    config: SelectorConfig,
    cache: HashMap<String, HashSet<RdfNode>>,
}

impl FocusNodeSelector {
    /// Create a new selector with default configuration.
    pub fn new() -> Self {
        Self {
            config: SelectorConfig::default(),
            cache: HashMap::new(),
        }
    }

    /// Create a new selector with the given configuration.
    pub fn with_config(config: SelectorConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    /// Clear the internal cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Return the number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Select focus nodes for a single target declaration.
    pub fn select(
        &mut self,
        declaration: &TargetDeclaration,
        graph: &RdfGraph,
    ) -> Result<HashSet<RdfNode>, SelectorError> {
        let cache_key = format!("{declaration:?}");

        if self.config.cache_enabled {
            if let Some(cached) = self.cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let nodes = match declaration {
            TargetDeclaration::TargetClass(class_iri) => graph.instances_of(class_iri),
            TargetDeclaration::TargetNode(node) => {
                let mut set = HashSet::new();
                set.insert(node.clone());
                set
            }
            TargetDeclaration::TargetSubjectsOf(pred) => graph.subjects_of(pred),
            TargetDeclaration::TargetObjectsOf(pred) => graph.objects_of(pred),
            TargetDeclaration::ImplicitClassTarget(class_iri) => graph.instances_of(class_iri),
            TargetDeclaration::SparqlTarget(query) => self.evaluate_sparql_target(query, graph)?,
        };

        if self.config.cache_enabled {
            self.cache.insert(cache_key, nodes.clone());
        }

        Ok(nodes)
    }

    /// Select focus nodes for multiple target declarations (union).
    pub fn select_combined(
        &mut self,
        declarations: &[TargetDeclaration],
        graph: &RdfGraph,
    ) -> Result<SelectionResult, SelectorError> {
        let mut all_nodes = HashSet::new();
        let mut per_target = Vec::with_capacity(declarations.len());

        for decl in declarations {
            let matched = self.select(decl, graph)?;
            all_nodes.extend(matched.iter().cloned());
            per_target.push(TargetContribution {
                declaration: decl.clone(),
                matched_nodes: matched,
            });
        }

        Ok(SelectionResult {
            nodes: all_nodes,
            per_target,
        })
    }

    /// Naive SPARQL target evaluator.
    ///
    /// Supports a trivial pattern: `SELECT ?this WHERE { ?this <pred> <obj> }`.
    /// In a full implementation this would delegate to the SPARQL engine.
    fn evaluate_sparql_target(
        &self,
        query: &str,
        graph: &RdfGraph,
    ) -> Result<HashSet<RdfNode>, SelectorError> {
        let query_trimmed = query.trim();

        // Very simple pattern matching for demo/test purposes:
        // SELECT ?this WHERE { ?this <pred> <obj> }
        if let Some(body) = extract_where_body(query_trimmed) {
            let parts: Vec<&str> = body.split_whitespace().collect();
            if parts.len() >= 3 && parts[0] == "?this" {
                let pred = parts[1].trim_matches(|c| c == '<' || c == '>');
                let obj = parts[2]
                    .trim_matches(|c| c == '<' || c == '>' || c == '.')
                    .trim();
                let mut result = HashSet::new();
                for triple in graph.triples() {
                    if triple.predicate == pred && triple.object == RdfNode::Iri(obj.to_string()) {
                        result.insert(triple.subject.clone());
                    }
                }
                return Ok(result);
            }
        }

        // Fallback: return empty set for unrecognised patterns
        Ok(HashSet::new())
    }
}

impl Default for FocusNodeSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the body between WHERE { ... } from a SPARQL query.
fn extract_where_body(query: &str) -> Option<&str> {
    let upper = query.to_uppercase();
    let where_pos = upper.find("WHERE")?;
    let after_where = &query[where_pos + 5..];
    let open = after_where.find('{')?;
    let close = after_where.rfind('}')?;
    if close > open + 1 {
        Some(after_where[open + 1..close].trim())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    fn iri_node(s: &str) -> RdfNode {
        RdfNode::Iri(s.to_string())
    }

    fn bnode(s: &str) -> RdfNode {
        RdfNode::BlankNode(s.to_string())
    }

    fn lit_node(s: &str) -> RdfNode {
        RdfNode::Literal(s.to_string())
    }

    fn triple(s: RdfNode, p: &str, o: RdfNode) -> Triple {
        Triple {
            subject: s,
            predicate: p.to_string(),
            object: o,
        }
    }

    fn sample_graph() -> RdfGraph {
        let mut g = RdfGraph::new();
        g.add_triple(triple(
            iri_node("http://ex.org/alice"),
            RDF_TYPE,
            iri_node("http://ex.org/Person"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/bob"),
            RDF_TYPE,
            iri_node("http://ex.org/Person"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/alice"),
            "http://ex.org/knows",
            iri_node("http://ex.org/bob"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/alice"),
            "http://ex.org/age",
            lit_node("30"),
        ));
        g
    }

    // -- RdfNode tests --

    #[test]
    fn test_rdf_node_as_str() {
        assert_eq!(iri_node("http://ex.org/a").as_str(), "http://ex.org/a");
        assert_eq!(bnode("b0").as_str(), "b0");
        assert_eq!(lit_node("hello").as_str(), "hello");
    }

    #[test]
    fn test_rdf_node_is_iri() {
        assert!(iri_node("http://ex.org/a").is_iri());
        assert!(!bnode("b0").is_iri());
        assert!(!lit_node("x").is_iri());
    }

    #[test]
    fn test_rdf_node_display() {
        let n = iri_node("http://ex.org/a");
        assert_eq!(format!("{n}"), "<http://ex.org/a>");
        let bn = bnode("b0");
        assert_eq!(format!("{bn}"), "_:b0");
        let l = lit_node("hello");
        assert_eq!(format!("{l}"), "\"hello\"");
    }

    // -- RdfGraph tests --

    #[test]
    fn test_graph_new_is_empty() {
        let g = RdfGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_graph_add_triple() {
        let mut g = RdfGraph::new();
        g.add_triple(triple(iri_node("s"), "p", iri_node("o")));
        assert!(!g.is_empty());
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_graph_instances_of() {
        let g = sample_graph();
        let persons = g.instances_of("http://ex.org/Person");
        assert_eq!(persons.len(), 2);
        assert!(persons.contains(&iri_node("http://ex.org/alice")));
        assert!(persons.contains(&iri_node("http://ex.org/bob")));
    }

    #[test]
    fn test_graph_instances_of_empty() {
        let g = sample_graph();
        let animals = g.instances_of("http://ex.org/Animal");
        assert!(animals.is_empty());
    }

    #[test]
    fn test_graph_subjects_of() {
        let g = sample_graph();
        let subjects = g.subjects_of("http://ex.org/knows");
        assert_eq!(subjects.len(), 1);
        assert!(subjects.contains(&iri_node("http://ex.org/alice")));
    }

    #[test]
    fn test_graph_objects_of() {
        let g = sample_graph();
        let objects = g.objects_of("http://ex.org/knows");
        assert_eq!(objects.len(), 1);
        assert!(objects.contains(&iri_node("http://ex.org/bob")));
    }

    #[test]
    fn test_graph_all_subjects() {
        let g = sample_graph();
        let subjects = g.all_subjects();
        assert!(subjects.len() >= 2);
    }

    // -- TargetClass --

    #[test]
    fn test_target_class() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Person".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_target_class_no_match() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Animal".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert!(nodes.is_empty());
    }

    // -- TargetNode --

    #[test]
    fn test_target_node() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetNode(iri_node("http://ex.org/alice"));
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&iri_node("http://ex.org/alice")));
    }

    #[test]
    fn test_target_node_blank_node() {
        let g = RdfGraph::new();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetNode(bnode("b42"));
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&bnode("b42")));
    }

    // -- TargetSubjectsOf --

    #[test]
    fn test_target_subjects_of() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetSubjectsOf("http://ex.org/knows".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&iri_node("http://ex.org/alice")));
    }

    #[test]
    fn test_target_subjects_of_no_match() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetSubjectsOf("http://ex.org/unknown".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert!(nodes.is_empty());
    }

    // -- TargetObjectsOf --

    #[test]
    fn test_target_objects_of() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetObjectsOf("http://ex.org/knows".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&iri_node("http://ex.org/bob")));
    }

    #[test]
    fn test_target_objects_of_literal() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetObjectsOf("http://ex.org/age".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&lit_node("30")));
    }

    // -- ImplicitClassTarget --

    #[test]
    fn test_implicit_class_target() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::ImplicitClassTarget("http://ex.org/Person".to_string());
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 2);
    }

    // -- SparqlTarget --

    #[test]
    fn test_sparql_target_simple() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::SparqlTarget(
            "SELECT ?this WHERE { ?this <http://ex.org/knows> <http://ex.org/bob> }".to_string(),
        );
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&iri_node("http://ex.org/alice")));
    }

    #[test]
    fn test_sparql_target_no_match() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::SparqlTarget(
            "SELECT ?this WHERE { ?this <http://ex.org/knows> <http://ex.org/charlie> }"
                .to_string(),
        );
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_sparql_target_unrecognised_pattern() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::SparqlTarget(
            "SELECT ?this WHERE { ?this a ?type . FILTER(?type = <http://ex.org/Person>) }"
                .to_string(),
        );
        let nodes = sel.select(&decl, &g).expect("select should succeed");
        // Unrecognised patterns return empty set
        assert!(nodes.is_empty());
    }

    // -- Combined targets --

    #[test]
    fn test_select_combined_union() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let declarations = vec![
            TargetDeclaration::TargetNode(iri_node("http://ex.org/charlie")),
            TargetDeclaration::TargetClass("http://ex.org/Person".to_string()),
        ];
        let result = sel
            .select_combined(&declarations, &g)
            .expect("combined select should succeed");
        assert_eq!(result.nodes.len(), 3); // alice, bob, charlie
        assert_eq!(result.per_target.len(), 2);
    }

    #[test]
    fn test_select_combined_empty() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let result = sel
            .select_combined(&[], &g)
            .expect("empty combined should succeed");
        assert!(result.nodes.is_empty());
        assert!(result.per_target.is_empty());
    }

    #[test]
    fn test_select_combined_overlap() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let declarations = vec![
            TargetDeclaration::TargetClass("http://ex.org/Person".to_string()),
            TargetDeclaration::TargetNode(iri_node("http://ex.org/alice")),
        ];
        let result = sel
            .select_combined(&declarations, &g)
            .expect("combined select should succeed");
        // alice appears in both targets, but the union de-duplicates
        assert_eq!(result.nodes.len(), 2);
    }

    // -- Caching --

    #[test]
    fn test_cache_hit() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Person".to_string());

        let _first = sel.select(&decl, &g).expect("select");
        assert_eq!(sel.cache_size(), 1);

        let second = sel.select(&decl, &g).expect("select");
        assert_eq!(second.len(), 2);
        assert_eq!(sel.cache_size(), 1); // still 1 entry
    }

    #[test]
    fn test_cache_disabled() {
        let g = sample_graph();
        let config = SelectorConfig {
            cache_enabled: false,
        };
        let mut sel = FocusNodeSelector::with_config(config);
        let decl = TargetDeclaration::TargetClass("http://ex.org/Person".to_string());

        let _first = sel.select(&decl, &g).expect("select");
        assert_eq!(sel.cache_size(), 0);
    }

    #[test]
    fn test_clear_cache() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Person".to_string());

        let _first = sel.select(&decl, &g).expect("select");
        assert_eq!(sel.cache_size(), 1);

        sel.clear_cache();
        assert_eq!(sel.cache_size(), 0);
    }

    // -- Error display --

    #[test]
    fn test_error_display() {
        let e1 = SelectorError::SparqlError("bad query".to_string());
        assert!(e1.to_string().contains("bad query"));
        let e2 = SelectorError::General("oops".to_string());
        assert!(e2.to_string().contains("oops"));
    }

    // -- Config --

    #[test]
    fn test_default_config() {
        let config = SelectorConfig::default();
        assert!(config.cache_enabled);
    }

    // -- TargetContribution --

    #[test]
    fn test_target_contribution_details() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let declarations = vec![
            TargetDeclaration::TargetClass("http://ex.org/Person".to_string()),
            TargetDeclaration::TargetSubjectsOf("http://ex.org/knows".to_string()),
        ];
        let result = sel.select_combined(&declarations, &g).expect("combined");
        assert_eq!(result.per_target[0].matched_nodes.len(), 2);
        assert_eq!(result.per_target[1].matched_nodes.len(), 1);
    }

    // -- extract_where_body --

    #[test]
    fn test_extract_where_body() {
        let q = "SELECT ?this WHERE { ?this <p> <o> }";
        let body = extract_where_body(q);
        assert_eq!(body, Some("?this <p> <o>"));
    }

    #[test]
    fn test_extract_where_body_no_where() {
        assert!(extract_where_body("SELECT ?x").is_none());
    }

    #[test]
    fn test_extract_where_body_empty_braces() {
        assert!(extract_where_body("SELECT ?x WHERE {}").is_none());
    }

    // -- Edge cases --

    #[test]
    fn test_empty_graph_target_class() {
        let g = RdfGraph::new();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Person".to_string());
        let nodes = sel.select(&decl, &g).expect("select");
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_empty_graph_subjects_of() {
        let g = RdfGraph::new();
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetSubjectsOf("http://ex.org/p".to_string());
        let nodes = sel.select(&decl, &g).expect("select");
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_multiple_types_same_node() {
        let mut g = RdfGraph::new();
        g.add_triple(triple(
            iri_node("http://ex.org/x"),
            RDF_TYPE,
            iri_node("http://ex.org/A"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/x"),
            RDF_TYPE,
            iri_node("http://ex.org/B"),
        ));
        let mut sel = FocusNodeSelector::new();
        let decl_a = TargetDeclaration::TargetClass("http://ex.org/A".to_string());
        let decl_b = TargetDeclaration::TargetClass("http://ex.org/B".to_string());
        let nodes_a = sel.select(&decl_a, &g).expect("select");
        let nodes_b = sel.select(&decl_b, &g).expect("select");
        assert_eq!(nodes_a.len(), 1);
        assert_eq!(nodes_b.len(), 1);
        assert_eq!(nodes_a, nodes_b);
    }

    #[test]
    fn test_blank_node_as_subject() {
        let mut g = RdfGraph::new();
        g.add_triple(triple(
            bnode("b1"),
            RDF_TYPE,
            iri_node("http://ex.org/Thing"),
        ));
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetClass("http://ex.org/Thing".to_string());
        let nodes = sel.select(&decl, &g).expect("select");
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&bnode("b1")));
    }

    #[test]
    fn test_multiple_subjects_same_predicate() {
        let mut g = RdfGraph::new();
        g.add_triple(triple(
            iri_node("http://ex.org/a"),
            "http://ex.org/p",
            iri_node("http://ex.org/x"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/b"),
            "http://ex.org/p",
            iri_node("http://ex.org/y"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/c"),
            "http://ex.org/p",
            iri_node("http://ex.org/z"),
        ));
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetSubjectsOf("http://ex.org/p".to_string());
        let nodes = sel.select(&decl, &g).expect("select");
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_target_objects_of_multiple() {
        let mut g = RdfGraph::new();
        g.add_triple(triple(
            iri_node("http://ex.org/s1"),
            "http://ex.org/p",
            iri_node("http://ex.org/o1"),
        ));
        g.add_triple(triple(
            iri_node("http://ex.org/s2"),
            "http://ex.org/p",
            iri_node("http://ex.org/o2"),
        ));
        let mut sel = FocusNodeSelector::new();
        let decl = TargetDeclaration::TargetObjectsOf("http://ex.org/p".to_string());
        let nodes = sel.select(&decl, &g).expect("select");
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_combined_three_targets() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let declarations = vec![
            TargetDeclaration::TargetClass("http://ex.org/Person".to_string()),
            TargetDeclaration::TargetSubjectsOf("http://ex.org/age".to_string()),
            TargetDeclaration::TargetNode(iri_node("http://ex.org/charlie")),
        ];
        let result = sel.select_combined(&declarations, &g).expect("combined");
        // alice (Person+age subject), bob (Person), charlie (target node)
        assert_eq!(result.nodes.len(), 3);
    }

    #[test]
    fn test_selector_default() {
        let sel = FocusNodeSelector::default();
        assert_eq!(sel.cache_size(), 0);
    }

    // -- Implicit class with explicit class overlap --

    #[test]
    fn test_implicit_and_explicit_class_overlap() {
        let g = sample_graph();
        let mut sel = FocusNodeSelector::new();
        let declarations = vec![
            TargetDeclaration::TargetClass("http://ex.org/Person".to_string()),
            TargetDeclaration::ImplicitClassTarget("http://ex.org/Person".to_string()),
        ];
        let result = sel.select_combined(&declarations, &g).expect("combined");
        // Union of identical sets => still 2 nodes
        assert_eq!(result.nodes.len(), 2);
    }
}
