/// SHACL property path execution.
///
/// Evaluates the full set of SHACL/SPARQL property path operators over an
/// in-memory RDF graph, with cycle prevention and result collection.
use std::collections::{HashSet, VecDeque};

// ── Graph representation ──────────────────────────────────────────────────────

/// A ground RDF triple used as the graph data-source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl GraphTriple {
    /// Create a new triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        GraphTriple {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A borrowed view of a set of RDF triples used as the evaluation graph.
pub struct RdfGraph<'a> {
    triples: &'a [GraphTriple],
}

impl<'a> RdfGraph<'a> {
    /// Wrap a slice of triples as an RDF graph.
    pub fn new(triples: &'a [GraphTriple]) -> Self {
        RdfGraph { triples }
    }

    /// Return all objects reachable from `subject` via `predicate`.
    pub fn objects_of(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject && t.predicate == predicate)
            .map(|t| t.object.clone())
            .collect()
    }

    /// Return all subjects that have `predicate` pointing to `object`.
    pub fn subjects_of(&self, predicate: &str, object: &str) -> Vec<String> {
        self.triples
            .iter()
            .filter(|t| t.predicate == predicate && t.object == object)
            .map(|t| t.subject.clone())
            .collect()
    }
}

// ── Property path data type ───────────────────────────────────────────────────

/// A SHACL/SPARQL property path expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyPath {
    /// Direct predicate IRI — `sh:path <iri>`.
    Predicate(String),
    /// Inverse path — `sh:inversePath path`.
    Inverse(Box<PropertyPath>),
    /// Sequence path — `(path1 / path2 / ...)`.
    Sequence(Vec<PropertyPath>),
    /// Alternative path — `(path1 | path2 | ...)`.
    Alternative(Vec<PropertyPath>),
    /// Zero-or-more path — `path*`.
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more path — `path+`.
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one path — `path?`.
    ZeroOrOne(Box<PropertyPath>),
}

// ── Evaluation result ─────────────────────────────────────────────────────────

/// The result of evaluating a property path from a start node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathResult {
    /// All distinct nodes reachable from the start node via the path.
    pub nodes: Vec<String>,
    /// Total number of raw path steps traversed (may exceed `nodes.len()` when
    /// cycles are detected and deduplicated).
    pub steps_taken: usize,
}

impl PathResult {
    /// Returns `true` when no nodes were reached.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of reachable nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when `node` is among the reachable nodes.
    pub fn contains(&self, node: &str) -> bool {
        self.nodes.iter().any(|n| n == node)
    }
}

// ── PathExecutor ──────────────────────────────────────────────────────────────

/// Evaluates SHACL property paths over an RDF graph.
pub struct PathExecutor;

impl PathExecutor {
    /// Evaluate `path` starting from `start` over `graph`.
    ///
    /// Returns all reachable nodes plus traversal statistics.
    pub fn evaluate(path: &PropertyPath, start: &str, graph: &RdfGraph<'_>) -> PathResult {
        let mut visited: HashSet<String> = HashSet::new();
        let mut steps = 0usize;
        let nodes = eval_path(path, start, graph, &mut visited, &mut steps);
        PathResult {
            nodes,
            steps_taken: steps,
        }
    }

    /// Evaluate `path` and return only whether at least one node is reachable.
    pub fn has_any(path: &PropertyPath, start: &str, graph: &RdfGraph<'_>) -> bool {
        !Self::evaluate(path, start, graph).is_empty()
    }

    /// Evaluate `path` and check whether a specific target node is reachable.
    pub fn reaches(path: &PropertyPath, start: &str, target: &str, graph: &RdfGraph<'_>) -> bool {
        Self::evaluate(path, start, graph).contains(target)
    }
}

// ── Core recursive evaluator ──────────────────────────────────────────────────

fn eval_path(
    path: &PropertyPath,
    start: &str,
    graph: &RdfGraph<'_>,
    global_visited: &mut HashSet<String>,
    steps: &mut usize,
) -> Vec<String> {
    match path {
        PropertyPath::Predicate(iri) => eval_predicate(start, iri, graph, steps),
        PropertyPath::Inverse(inner) => eval_inverse(inner, start, graph, steps),
        PropertyPath::Sequence(parts) => eval_sequence(parts, start, graph, global_visited, steps),
        PropertyPath::Alternative(alts) => {
            eval_alternative(alts, start, graph, global_visited, steps)
        }
        PropertyPath::ZeroOrMore(inner) => eval_zero_or_more(inner, start, graph, steps),
        PropertyPath::OneOrMore(inner) => eval_one_or_more(inner, start, graph, steps),
        PropertyPath::ZeroOrOne(inner) => eval_zero_or_one(inner, start, graph, steps),
    }
}

/// Direct predicate path: return all `o` such that `start pred o` ∈ graph.
fn eval_predicate(
    start: &str,
    predicate: &str,
    graph: &RdfGraph<'_>,
    steps: &mut usize,
) -> Vec<String> {
    let nodes = graph.objects_of(start, predicate);
    *steps += nodes.len();
    dedup(nodes)
}

/// Inverse path: return all `s` such that `s pred start` ∈ graph.
fn eval_inverse(
    inner: &PropertyPath,
    start: &str,
    graph: &RdfGraph<'_>,
    steps: &mut usize,
) -> Vec<String> {
    match inner {
        PropertyPath::Predicate(iri) => {
            let nodes = graph.subjects_of(iri, start);
            *steps += nodes.len();
            dedup(nodes)
        }
        other => {
            // For complex inverse paths, collect forward results and swap direction.
            // This is a best-effort approximation for nested inverse expressions.
            let mut candidates: HashSet<String> = HashSet::new();
            for triple in graph.triples {
                // Try: can we reach `start` from triple.subject via `other`?
                let mut vis = HashSet::new();
                let reachable = eval_path(other, &triple.subject, graph, &mut vis, steps);
                if reachable.iter().any(|n| n == start) {
                    candidates.insert(triple.subject.clone());
                }
            }
            candidates.into_iter().collect()
        }
    }
}

/// Sequence path: chain each path segment, feeding the output of one into the next.
fn eval_sequence(
    parts: &[PropertyPath],
    start: &str,
    graph: &RdfGraph<'_>,
    global_visited: &mut HashSet<String>,
    steps: &mut usize,
) -> Vec<String> {
    let mut frontier: Vec<String> = vec![start.to_owned()];
    for segment in parts {
        let mut next: HashSet<String> = HashSet::new();
        for node in &frontier {
            let reached = eval_path(segment, node, graph, global_visited, steps);
            next.extend(reached);
        }
        frontier = next.into_iter().collect();
    }
    frontier
}

/// Alternative path: union of results from each alternative.
fn eval_alternative(
    alts: &[PropertyPath],
    start: &str,
    graph: &RdfGraph<'_>,
    global_visited: &mut HashSet<String>,
    steps: &mut usize,
) -> Vec<String> {
    let mut result: HashSet<String> = HashSet::new();
    for alt in alts {
        let reached = eval_path(alt, start, graph, global_visited, steps);
        result.extend(reached);
    }
    result.into_iter().collect()
}

/// Zero-or-more path: BFS transitive closure, including the start node itself.
fn eval_zero_or_more(
    inner: &PropertyPath,
    start: &str,
    graph: &RdfGraph<'_>,
    steps: &mut usize,
) -> Vec<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    // ZeroOrMore always includes the start node (zero steps = identity).
    visited.insert(start.to_owned());
    queue.push_back(start.to_owned());

    while let Some(current) = queue.pop_front() {
        let mut inner_vis = HashSet::new();
        let reached = eval_path(inner, &current, graph, &mut inner_vis, steps);
        for node in reached {
            if visited.insert(node.clone()) {
                queue.push_back(node);
            }
        }
    }
    visited.into_iter().collect()
}

/// One-or-more path: BFS transitive closure, excluding the start node unless revisited.
fn eval_one_or_more(
    inner: &PropertyPath,
    start: &str,
    graph: &RdfGraph<'_>,
    steps: &mut usize,
) -> Vec<String> {
    // First, collect the direct successors (one step).
    let mut expanded: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    // Seed with one-step successors.
    let mut seed_vis = HashSet::new();
    let first = eval_path(inner, start, graph, &mut seed_vis, steps);
    for node in first {
        if expanded.insert(node.clone()) {
            queue.push_back(node);
        }
    }

    // BFS for additional steps.
    while let Some(current) = queue.pop_front() {
        let mut inner_vis = HashSet::new();
        let reached = eval_path(inner, &current, graph, &mut inner_vis, steps);
        for node in reached {
            if expanded.insert(node.clone()) {
                queue.push_back(node);
            }
        }
    }
    expanded.into_iter().collect()
}

/// Zero-or-one path: either the start node itself or a single-step successor.
fn eval_zero_or_one(
    inner: &PropertyPath,
    start: &str,
    graph: &RdfGraph<'_>,
    steps: &mut usize,
) -> Vec<String> {
    let mut result: HashSet<String> = HashSet::new();
    // Zero steps → include start.
    result.insert(start.to_owned());
    // One step.
    let mut vis = HashSet::new();
    let one_step = eval_path(inner, start, graph, &mut vis, steps);
    result.extend(one_step);
    result.into_iter().collect()
}

/// Remove duplicates while preserving order of first occurrence.
fn dedup(nodes: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    nodes
        .into_iter()
        .filter(|n| seen.insert(n.clone()))
        .collect()
}

// ── Builder helpers for PropertyPath ─────────────────────────────────────────

impl PropertyPath {
    /// Convenience constructor for a predicate path.
    pub fn predicate(iri: impl Into<String>) -> Self {
        PropertyPath::Predicate(iri.into())
    }

    /// Convenience constructor for an inverse path.
    pub fn inverse(inner: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(inner))
    }

    /// Convenience constructor for a sequence path.
    pub fn sequence(parts: Vec<PropertyPath>) -> Self {
        PropertyPath::Sequence(parts)
    }

    /// Convenience constructor for an alternative path.
    pub fn alternative(alts: Vec<PropertyPath>) -> Self {
        PropertyPath::Alternative(alts)
    }

    /// Convenience constructor for zero-or-more.
    pub fn zero_or_more(inner: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(inner))
    }

    /// Convenience constructor for one-or-more.
    pub fn one_or_more(inner: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(inner))
    }

    /// Convenience constructor for zero-or-one.
    pub fn zero_or_one(inner: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(inner))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small RDF graph for testing.
    fn build_graph() -> Vec<GraphTriple> {
        vec![
            // type hierarchy: A → B → C → D
            GraphTriple::new("A", "parent", "B"),
            GraphTriple::new("B", "parent", "C"),
            GraphTriple::new("C", "parent", "D"),
            // label triples
            GraphTriple::new("A", "label", "Alice"),
            GraphTriple::new("B", "label", "Bob"),
            GraphTriple::new("C", "label", "Carol"),
            // sibling link A → E
            GraphTriple::new("A", "sibling", "E"),
            // back edge for cycle testing
            GraphTriple::new("D", "parent", "A"),
        ]
    }

    // ── Direct predicate path ─────────────────────────────────────────────────

    #[test]
    fn test_predicate_path_basic() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::predicate("parent");
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("B"), "A parent B");
        assert!(!result.contains("C"), "A does not directly parent C");
    }

    #[test]
    fn test_predicate_path_no_match() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::predicate("nonexistent");
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.is_empty());
    }

    // ── Inverse path ──────────────────────────────────────────────────────────

    #[test]
    fn test_inverse_path() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        // ^parent from B should give A (the node that has B as parent).
        let path = PropertyPath::inverse(PropertyPath::predicate("parent"));
        let result = PathExecutor::evaluate(&path, "B", &graph);
        assert!(result.contains("A"), "^parent from B gives A");
    }

    #[test]
    fn test_inverse_path_chain() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        // ^parent from C should give B.
        let path = PropertyPath::inverse(PropertyPath::predicate("parent"));
        let result = PathExecutor::evaluate(&path, "C", &graph);
        assert!(result.contains("B"));
    }

    // ── Sequence path ─────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_path_two_hops() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        // parent / parent from A should reach C.
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate("parent"),
            PropertyPath::predicate("parent"),
        ]);
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("C"), "A parent/parent C");
        assert!(!result.contains("B"), "B not in two-hop result");
    }

    #[test]
    fn test_sequence_path_three_hops() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate("parent"),
            PropertyPath::predicate("parent"),
            PropertyPath::predicate("parent"),
        ]);
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("D"), "A →³ D");
    }

    // ── Alternative path ──────────────────────────────────────────────────────

    #[test]
    fn test_alternative_path() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        // parent | sibling from A should give B and E.
        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate("parent"),
            PropertyPath::predicate("sibling"),
        ]);
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("B"), "parent gives B");
        assert!(result.contains("E"), "sibling gives E");
    }

    #[test]
    fn test_alternative_path_deduplication() {
        // Both alternatives point to the same node → should appear only once.
        let triples = vec![
            GraphTriple::new("X", "rel1", "Y"),
            GraphTriple::new("X", "rel2", "Y"),
        ];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate("rel1"),
            PropertyPath::predicate("rel2"),
        ]);
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert_eq!(result.len(), 1);
        assert!(result.contains("Y"));
    }

    // ── Zero-or-more path ─────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_more_includes_start() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("parent"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        // ZeroOrMore always includes the start.
        assert!(result.contains("A"), "start node included");
        assert!(result.contains("B"));
        assert!(result.contains("C"));
        assert!(result.contains("D"));
    }

    #[test]
    fn test_zero_or_more_cycle_prevention() {
        // Graph has a cycle A → B → C → D → A.
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("parent"));
        // Should terminate even with the cycle and not loop forever.
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.len() >= 4, "all nodes reachable: {:?}", result);
    }

    // ── One-or-more path ──────────────────────────────────────────────────────

    #[test]
    fn test_one_or_more_excludes_start() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::one_or_more(PropertyPath::predicate("parent"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        // A may be reachable via the cycle, but direct `A` from zero steps should not appear
        // unless it is genuinely reached via one+ steps (here it is due to D→A).
        assert!(result.contains("B"), "B reachable in one step");
        assert!(result.contains("C"), "C reachable in two steps");
        assert!(result.contains("D"), "D reachable in three steps");
    }

    #[test]
    fn test_one_or_more_no_successors() {
        let triples = vec![GraphTriple::new("X", "p", "Y")];
        let graph = RdfGraph::new(&triples);
        // From Y, one+ step of 'p' yields nothing (Y has no outgoing 'p').
        let path = PropertyPath::one_or_more(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "Y", &graph);
        assert!(result.is_empty());
    }

    // ── Zero-or-one path ──────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_one_includes_start_and_successor() {
        let triples = vec![GraphTriple::new("X", "rel", "Y")];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_one(PropertyPath::predicate("rel"));
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert!(result.contains("X"), "zero steps → start");
        assert!(result.contains("Y"), "one step → Y");
    }

    #[test]
    fn test_zero_or_one_no_successors() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_one(PropertyPath::predicate("rel"));
        let result = PathExecutor::evaluate(&path, "X", &graph);
        // Zero steps → start node only.
        assert_eq!(result.len(), 1);
        assert!(result.contains("X"));
    }

    // ── Reachability helpers ───────────────────────────────────────────────────

    #[test]
    fn test_has_any_true() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        assert!(PathExecutor::has_any(
            &PropertyPath::predicate("parent"),
            "A",
            &graph
        ));
    }

    #[test]
    fn test_has_any_false() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        assert!(!PathExecutor::has_any(
            &PropertyPath::predicate("anything"),
            "X",
            &graph
        ));
    }

    #[test]
    fn test_reaches_target() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("parent"));
        assert!(PathExecutor::reaches(&path, "A", "D", &graph));
    }

    #[test]
    fn test_reaches_target_false() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::predicate("parent");
        // A direct parent step from A does not reach D.
        assert!(!PathExecutor::reaches(&path, "A", "D", &graph));
    }

    // ── Steps counter ─────────────────────────────────────────────────────────

    #[test]
    fn test_steps_taken_nonzero_for_traversal() {
        let triples = build_graph();
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("parent"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        // Steps should be > 0 as the graph was actually traversed.
        assert!(result.steps_taken > 0, "steps_taken={}", result.steps_taken);
    }

    // ── Nested paths ──────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_then_alternative() {
        let triples = vec![
            GraphTriple::new("S", "hop", "M"),
            GraphTriple::new("M", "a", "X"),
            GraphTriple::new("M", "b", "Y"),
        ];
        let graph = RdfGraph::new(&triples);
        // hop / (a | b)
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate("hop"),
            PropertyPath::alternative(vec![
                PropertyPath::predicate("a"),
                PropertyPath::predicate("b"),
            ]),
        ]);
        let result = PathExecutor::evaluate(&path, "S", &graph);
        assert!(result.contains("X"));
        assert!(result.contains("Y"));
    }

    #[test]
    fn test_inverse_then_zero_or_more() {
        // Graph: A → B → C  with labels
        let triples = vec![
            GraphTriple::new("A", "child", "B"),
            GraphTriple::new("B", "child", "C"),
        ];
        let graph = RdfGraph::new(&triples);
        // ^child* from B → B, A  (inverse step then zero-or-more inverse)
        let path =
            PropertyPath::zero_or_more(PropertyPath::inverse(PropertyPath::predicate("child")));
        let result = PathExecutor::evaluate(&path, "B", &graph);
        assert!(result.contains("B"), "start included");
        assert!(result.contains("A"), "A reaches B via child");
    }

    // ── Graph result collection ────────────────────────────────────────────────

    #[test]
    fn test_path_result_len_and_is_empty() {
        let r = PathResult {
            nodes: vec!["X".into(), "Y".into()],
            steps_taken: 2,
        };
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        let empty = PathResult {
            nodes: vec![],
            steps_taken: 0,
        };
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_path_result_contains() {
        let r = PathResult {
            nodes: vec!["Alpha".into(), "Beta".into()],
            steps_taken: 2,
        };
        assert!(r.contains("Alpha"));
        assert!(!r.contains("Gamma"));
    }

    // ── Graph triple construction ─────────────────────────────────────────────

    #[test]
    fn test_graph_triple_new() {
        let t = GraphTriple::new("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_rdf_graph_objects_of() {
        let triples = vec![
            GraphTriple::new("A", "p", "B"),
            GraphTriple::new("A", "p", "C"),
            GraphTriple::new("B", "p", "D"),
        ];
        let graph = RdfGraph::new(&triples);
        let objs = graph.objects_of("A", "p");
        assert_eq!(objs.len(), 2);
        assert!(objs.contains(&"B".to_owned()));
        assert!(objs.contains(&"C".to_owned()));
    }

    #[test]
    fn test_rdf_graph_subjects_of() {
        let triples = vec![
            GraphTriple::new("A", "p", "Z"),
            GraphTriple::new("B", "p", "Z"),
        ];
        let graph = RdfGraph::new(&triples);
        let subs = graph.subjects_of("p", "Z");
        assert_eq!(subs.len(), 2);
        assert!(subs.contains(&"A".to_owned()));
        assert!(subs.contains(&"B".to_owned()));
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_predicate_path_multiple_objects() {
        let triples = vec![
            GraphTriple::new("X", "rel", "Y1"),
            GraphTriple::new("X", "rel", "Y2"),
            GraphTriple::new("X", "rel", "Y3"),
        ];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::predicate("rel");
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_zero_or_more_single_hop() {
        let triples = vec![GraphTriple::new("A", "p", "B")];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("A"));
        assert!(result.contains("B"));
    }

    #[test]
    fn test_one_or_more_single_hop() {
        let triples = vec![GraphTriple::new("A", "p", "B")];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::one_or_more(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.contains("B"));
        assert!(
            !result.contains("A"),
            "start excluded for one-or-more unless reachable via cycle"
        );
    }

    #[test]
    fn test_zero_or_one_at_most_one_step() {
        let triples = vec![
            GraphTriple::new("A", "p", "B"),
            GraphTriple::new("B", "p", "C"),
        ];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_one(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        // Zero steps → A, one step → B; NOT two steps → C.
        assert!(result.contains("A"));
        assert!(result.contains("B"));
        assert!(!result.contains("C"));
    }

    #[test]
    fn test_alternative_path_single_element() {
        let triples = vec![GraphTriple::new("X", "q", "Y")];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::alternative(vec![PropertyPath::predicate("q")]);
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert!(result.contains("Y"));
    }

    #[test]
    fn test_sequence_path_single_segment() {
        let triples = vec![GraphTriple::new("X", "rel", "Y")];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::sequence(vec![PropertyPath::predicate("rel")]);
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert!(result.contains("Y"));
    }

    #[test]
    fn test_empty_graph_always_empty_result() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::one_or_more(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert!(result.is_empty());
    }

    #[test]
    fn test_inverse_path_no_subjects() {
        let triples = vec![GraphTriple::new("A", "p", "B")];
        let graph = RdfGraph::new(&triples);
        // ^p from A should find nodes that have A as object via p.  None here.
        let path = PropertyPath::inverse(PropertyPath::predicate("p"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert!(result.is_empty());
    }

    #[test]
    fn test_zero_or_more_deep_chain() {
        // Linear chain depth 5: A → B → C → D → E → F.
        let triples = vec![
            GraphTriple::new("A", "next", "B"),
            GraphTriple::new("B", "next", "C"),
            GraphTriple::new("C", "next", "D"),
            GraphTriple::new("D", "next", "E"),
            GraphTriple::new("E", "next", "F"),
        ];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("next"));
        let result = PathExecutor::evaluate(&path, "A", &graph);
        assert_eq!(result.len(), 6, "all 6 nodes reachable: {:?}", result);
    }

    #[test]
    fn test_property_path_equality() {
        let p1 = PropertyPath::predicate("http://example.org/p");
        let p2 = PropertyPath::predicate("http://example.org/p");
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_property_path_inverse_equality() {
        let p = PropertyPath::inverse(PropertyPath::predicate("p"));
        let q = PropertyPath::inverse(PropertyPath::predicate("p"));
        assert_eq!(p, q);
    }

    #[test]
    fn test_graph_triple_hash_equality() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GraphTriple::new("s", "p", "o"));
        set.insert(GraphTriple::new("s", "p", "o")); // duplicate
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_sequence_empty_gives_start_node() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::sequence(vec![]);
        // An empty sequence resolves to just the start node.
        let result = PathExecutor::evaluate(&path, "S", &graph);
        // frontier starts as ["S"], no segments to process → ["S"].
        assert!(result.contains("S"));
    }

    #[test]
    fn test_alternative_empty_gives_empty() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::alternative(vec![PropertyPath::predicate("p")]);
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert!(result.is_empty());
    }

    #[test]
    fn test_reaches_self_via_zero_or_more() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::zero_or_more(PropertyPath::predicate("any"));
        // ZeroOrMore always includes the start (zero steps).
        assert!(PathExecutor::reaches(&path, "X", "X", &graph));
    }

    #[test]
    fn test_steps_taken_zero_for_empty_predicate() {
        let triples: Vec<GraphTriple> = vec![];
        let graph = RdfGraph::new(&triples);
        let path = PropertyPath::predicate("p");
        let result = PathExecutor::evaluate(&path, "X", &graph);
        assert_eq!(result.steps_taken, 0);
    }
}
