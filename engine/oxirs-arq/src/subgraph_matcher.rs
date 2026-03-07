//! Subgraph isomorphism / pattern matching for RDF graphs.
//!
//! Implements a VF2-inspired algorithm for finding subgraph matches
//! between a pattern graph and a data graph.

use std::collections::HashMap;

/// A node in an RDF graph with a unique id and label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfNode {
    pub id: u32,
    pub label: String,
}

/// A directed edge in an RDF graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfEdge {
    pub from: u32,
    pub to: u32,
    pub label: String,
}

/// An RDF graph represented as adjacency maps.
#[derive(Debug, Clone, Default)]
pub struct RdfGraph {
    pub nodes: HashMap<u32, RdfNode>,
    pub edges: Vec<RdfEdge>,
}

/// A single match result: mapping from pattern node id → data node id,
/// plus the indices into `data.edges` for each matched pattern edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// pattern_node_id -> data_node_id
    pub node_mapping: HashMap<u32, u32>,
    /// For each pattern edge index, the corresponding data edge index.
    pub edge_mapping: Vec<usize>,
}

/// Subgraph matcher using VF2-inspired backtracking.
pub struct SubgraphMatcher {
    max_results: usize,
}

impl RdfGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node with the given id and label.
    pub fn add_node(&mut self, id: u32, label: &str) {
        self.nodes.insert(
            id,
            RdfNode {
                id,
                label: label.to_string(),
            },
        );
    }

    /// Add a directed edge from `from` to `to` with the given label.
    pub fn add_edge(&mut self, from: u32, to: u32, label: &str) {
        self.edges.push(RdfEdge {
            from,
            to,
            label: label.to_string(),
        });
    }

    /// Return outgoing neighbour ids of `node_id`.
    pub fn out_neighbors(&self, node_id: u32) -> Vec<u32> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id)
            .map(|e| e.to)
            .collect()
    }

    /// Return incoming neighbour ids of `node_id`.
    pub fn in_neighbors(&self, node_id: u32) -> Vec<u32> {
        self.edges
            .iter()
            .filter(|e| e.to == node_id)
            .map(|e| e.from)
            .collect()
    }

    /// Return all node ids in deterministic order.
    fn node_ids_sorted(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.nodes.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

// ── Internal state used during backtracking ──────────────────────────────────

struct MatchState<'a> {
    pattern: &'a RdfGraph,
    data: &'a RdfGraph,
    /// pattern_node_id -> data_node_id
    node_map: HashMap<u32, u32>,
    /// data_node_id -> pattern_node_id (reverse)
    rev_map: HashMap<u32, u32>,
    /// Ordered list of pattern node ids that we match one at a time.
    pattern_order: Vec<u32>,
    results: Vec<MatchResult>,
    max_results: usize,
}

impl<'a> MatchState<'a> {
    fn new(
        pattern: &'a RdfGraph,
        data: &'a RdfGraph,
        pattern_order: Vec<u32>,
        max_results: usize,
    ) -> Self {
        Self {
            pattern,
            data,
            node_map: HashMap::new(),
            rev_map: HashMap::new(),
            pattern_order,
            results: Vec::new(),
            max_results,
        }
    }

    /// Depth-first search: try to map `pattern_order[depth]` to some data node.
    fn dfs(&mut self, depth: usize) {
        if self.results.len() >= self.max_results {
            return;
        }
        if depth == self.pattern_order.len() {
            // All pattern nodes mapped — record the result.
            if let Some(result) = self.build_result() {
                self.results.push(result);
            }
            return;
        }

        let p_node_id = self.pattern_order[depth];
        let p_label = match self.pattern.nodes.get(&p_node_id) {
            Some(n) => n.label.clone(),
            None => return,
        };

        let data_candidates: Vec<u32> = self.data.node_ids_sorted();

        for d_node_id in data_candidates {
            if self.results.len() >= self.max_results {
                break;
            }
            // Must not already be mapped.
            if self.rev_map.contains_key(&d_node_id) {
                continue;
            }
            // Labels must match.
            match self.data.nodes.get(&d_node_id) {
                Some(dn) if dn.label == p_label => {}
                _ => continue,
            }
            // Structural feasibility: edges already in the partial mapping must be satisfied.
            if !self.is_feasible(p_node_id, d_node_id) {
                continue;
            }

            // Extend state.
            self.node_map.insert(p_node_id, d_node_id);
            self.rev_map.insert(d_node_id, p_node_id);

            self.dfs(depth + 1);

            // Retract.
            self.node_map.remove(&p_node_id);
            self.rev_map.remove(&d_node_id);
        }
    }

    /// Check that mapping p_node → d_node is consistent with edges already mapped.
    fn is_feasible(&self, p_node: u32, d_node: u32) -> bool {
        // For every already-mapped predecessor of p_node, the corresponding
        // data predecessor of d_node must have the same edge label.
        for p_edge in &self.pattern.edges {
            if p_edge.to == p_node {
                // p_edge.from → p_node  (incoming to p_node)
                if let Some(&d_from) = self.node_map.get(&p_edge.from) {
                    // There must be an edge d_from → d_node with the same label.
                    if !self
                        .data
                        .edges
                        .iter()
                        .any(|e| e.from == d_from && e.to == d_node && e.label == p_edge.label)
                    {
                        return false;
                    }
                }
            }
            if p_edge.from == p_node {
                // p_node → p_edge.to  (outgoing from p_node)
                if let Some(&d_to) = self.node_map.get(&p_edge.to) {
                    // There must be an edge d_node → d_to with the same label.
                    if !self
                        .data
                        .edges
                        .iter()
                        .any(|e| e.from == d_node && e.to == d_to && e.label == p_edge.label)
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Build a MatchResult from the current full node mapping.
    fn build_result(&self) -> Option<MatchResult> {
        let mut edge_mapping = Vec::with_capacity(self.pattern.edges.len());
        for p_edge in &self.pattern.edges {
            let d_from = *self.node_map.get(&p_edge.from)?;
            let d_to = *self.node_map.get(&p_edge.to)?;
            let idx = self
                .data
                .edges
                .iter()
                .position(|e| e.from == d_from && e.to == d_to && e.label == p_edge.label)?;
            edge_mapping.push(idx);
        }
        Some(MatchResult {
            node_mapping: self.node_map.clone(),
            edge_mapping,
        })
    }
}

impl SubgraphMatcher {
    /// Create a new matcher that returns at most `max_results` matches.
    pub fn new(max_results: usize) -> Self {
        Self { max_results }
    }

    /// Find all subgraph isomorphism matches of `pattern` in `data`.
    pub fn find_matches(&self, pattern: &RdfGraph, data: &RdfGraph) -> Vec<MatchResult> {
        if pattern.nodes.is_empty() {
            // Empty pattern matches trivially once.
            return vec![MatchResult {
                node_mapping: HashMap::new(),
                edge_mapping: Vec::new(),
            }];
        }
        // Order pattern nodes: start with highest-degree node.
        let order = Self::compute_match_order(pattern);
        let mut state = MatchState::new(pattern, data, order, self.max_results);
        state.dfs(0);
        state.results
    }

    /// Test whether `g1` and `g2` are exactly isomorphic (same size, bijective mapping).
    pub fn is_isomorphic(&self, g1: &RdfGraph, g2: &RdfGraph) -> bool {
        if g1.nodes.len() != g2.nodes.len() || g1.edges.len() != g2.edges.len() {
            return false;
        }
        let matcher = SubgraphMatcher::new(1);
        !matcher.find_matches(g1, g2).is_empty()
    }

    /// Count the number of subgraph matches of `pattern` in `data`.
    pub fn count_matches(&self, pattern: &RdfGraph, data: &RdfGraph) -> usize {
        let unlimited = SubgraphMatcher::new(usize::MAX);
        unlimited.find_matches(pattern, data).len()
    }

    /// Compute a good matching order: start from the node with the highest
    /// combined degree, then extend to connected neighbours.
    fn compute_match_order(pattern: &RdfGraph) -> Vec<u32> {
        if pattern.nodes.is_empty() {
            return Vec::new();
        }
        // Degree = out-degree + in-degree.
        let mut degree: HashMap<u32, usize> = pattern.nodes.keys().map(|&k| (k, 0)).collect();
        for edge in &pattern.edges {
            *degree.entry(edge.from).or_insert(0) += 1;
            *degree.entry(edge.to).or_insert(0) += 1;
        }
        let mut sorted: Vec<u32> = degree.keys().copied().collect();
        sorted.sort_unstable_by(|a, b| degree[b].cmp(&degree[a]).then_with(|| a.cmp(b)));

        // BFS-style extension so the order stays connected.
        let mut order = Vec::with_capacity(sorted.len());
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        if let Some(&first) = sorted.first() {
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(first);
            visited.insert(first);
            while let Some(node) = queue.pop_front() {
                order.push(node);
                // Add neighbours (out and in) in sorted order.
                let mut neighbours: Vec<u32> = pattern
                    .edges
                    .iter()
                    .filter(|e| e.from == node || e.to == node)
                    .flat_map(|e| [e.from, e.to])
                    .filter(|n| !visited.contains(n))
                    .collect();
                neighbours.sort_unstable();
                neighbours.dedup();
                for n in neighbours {
                    if visited.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }
        // Append any isolated nodes not reachable from BFS start.
        for n in sorted {
            if !visited.contains(&n) {
                order.push(n);
            }
        }
        order
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matcher() -> SubgraphMatcher {
        SubgraphMatcher::new(100)
    }

    // ── Basic graph helpers ───────────────────────────────────────────────

    fn single_node_graph(label: &str) -> RdfGraph {
        let mut g = RdfGraph::new();
        g.add_node(0, label);
        g
    }

    fn single_edge_graph(from_label: &str, to_label: &str, edge_label: &str) -> RdfGraph {
        let mut g = RdfGraph::new();
        g.add_node(0, from_label);
        g.add_node(1, to_label);
        g.add_edge(0, 1, edge_label);
        g
    }

    // ── Empty / trivial cases ─────────────────────────────────────────────

    #[test]
    fn test_empty_pattern_empty_data() {
        let m = make_matcher();
        let p = RdfGraph::new();
        let d = RdfGraph::new();
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
        assert!(r[0].node_mapping.is_empty());
    }

    #[test]
    fn test_empty_pattern_nonempty_data() {
        let m = make_matcher();
        let p = RdfGraph::new();
        let mut d = RdfGraph::new();
        d.add_node(0, "A");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_single_node_match() {
        let m = make_matcher();
        let p = single_node_graph("A");
        let d = single_node_graph("A");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].node_mapping[&0], 0);
    }

    #[test]
    fn test_single_node_label_mismatch() {
        let m = make_matcher();
        let p = single_node_graph("A");
        let d = single_node_graph("B");
        assert!(m.find_matches(&p, &d).is_empty());
    }

    #[test]
    fn test_single_node_multiple_candidates() {
        let m = make_matcher();
        let p = single_node_graph("A");
        let mut d = RdfGraph::new();
        d.add_node(0, "A");
        d.add_node(1, "B");
        d.add_node(2, "A");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 2);
    }

    // ── Single-edge patterns ──────────────────────────────────────────────

    #[test]
    fn test_single_edge_exact_match() {
        let m = make_matcher();
        let p = single_edge_graph("S", "O", "p");
        let d = single_edge_graph("S", "O", "p");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_single_edge_label_mismatch() {
        let m = make_matcher();
        let p = single_edge_graph("S", "O", "p");
        let d = single_edge_graph("S", "O", "q");
        assert!(m.find_matches(&p, &d).is_empty());
    }

    #[test]
    fn test_single_edge_node_label_mismatch() {
        let m = make_matcher();
        let p = single_edge_graph("S", "O", "p");
        let d = single_edge_graph("X", "O", "p");
        assert!(m.find_matches(&p, &d).is_empty());
    }

    #[test]
    fn test_single_edge_multiple_matches() {
        let m = make_matcher();
        let p = single_edge_graph("A", "B", "r");
        let mut d = RdfGraph::new();
        d.add_node(0, "A");
        d.add_node(1, "B");
        d.add_node(2, "A");
        d.add_node(3, "B");
        d.add_edge(0, 1, "r");
        d.add_edge(2, 3, "r");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 2);
    }

    // ── Path patterns ─────────────────────────────────────────────────────

    #[test]
    fn test_path_pattern_match() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "A");
        p.add_node(1, "B");
        p.add_node(2, "C");
        p.add_edge(0, 1, "e1");
        p.add_edge(1, 2, "e2");

        let mut d = RdfGraph::new();
        d.add_node(10, "A");
        d.add_node(11, "B");
        d.add_node(12, "C");
        d.add_edge(10, 11, "e1");
        d.add_edge(11, 12, "e2");

        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_path_pattern_no_match_wrong_order() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "A");
        p.add_node(1, "B");
        p.add_edge(0, 1, "e");

        let mut d = RdfGraph::new();
        d.add_node(0, "B");
        d.add_node(1, "A");
        d.add_edge(0, 1, "e"); // B→A not A→B

        assert!(m.find_matches(&p, &d).is_empty());
    }

    // ── Star patterns ─────────────────────────────────────────────────────

    #[test]
    fn test_star_pattern_match() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "hub");
        p.add_node(1, "leaf");
        p.add_node(2, "leaf");
        p.add_edge(0, 1, "spoke");
        p.add_edge(0, 2, "spoke");

        let mut d = RdfGraph::new();
        d.add_node(0, "hub");
        d.add_node(1, "leaf");
        d.add_node(2, "leaf");
        d.add_edge(0, 1, "spoke");
        d.add_edge(0, 2, "spoke");

        let r = m.find_matches(&p, &d);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_star_pattern_insufficient_leaves() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "hub");
        p.add_node(1, "leaf");
        p.add_node(2, "leaf");
        p.add_node(3, "leaf");
        p.add_edge(0, 1, "spoke");
        p.add_edge(0, 2, "spoke");
        p.add_edge(0, 3, "spoke");

        let mut d = RdfGraph::new();
        d.add_node(0, "hub");
        d.add_node(1, "leaf");
        d.add_node(2, "leaf"); // only 2 leaves
        d.add_edge(0, 1, "spoke");
        d.add_edge(0, 2, "spoke");

        assert!(m.find_matches(&p, &d).is_empty());
    }

    // ── Cycle patterns ────────────────────────────────────────────────────

    #[test]
    fn test_triangle_cycle_match() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "N");
        p.add_node(1, "N");
        p.add_node(2, "N");
        p.add_edge(0, 1, "e");
        p.add_edge(1, 2, "e");
        p.add_edge(2, 0, "e");

        let mut d = RdfGraph::new();
        d.add_node(0, "N");
        d.add_node(1, "N");
        d.add_node(2, "N");
        d.add_edge(0, 1, "e");
        d.add_edge(1, 2, "e");
        d.add_edge(2, 0, "e");

        let r = m.find_matches(&p, &d);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_cycle_not_in_path() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "N");
        p.add_node(1, "N");
        p.add_edge(0, 1, "e");
        p.add_edge(1, 0, "e"); // bidirectional = cycle

        let mut d = RdfGraph::new();
        d.add_node(0, "N");
        d.add_node(1, "N");
        d.add_edge(0, 1, "e"); // only one direction

        assert!(m.find_matches(&p, &d).is_empty());
    }

    // ── max_results limit ─────────────────────────────────────────────────

    #[test]
    fn test_max_results_limit() {
        let m = SubgraphMatcher::new(2);
        let p = single_node_graph("X");
        let mut d = RdfGraph::new();
        for i in 0..10u32 {
            d.add_node(i, "X");
        }
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_max_results_zero() {
        let m = SubgraphMatcher::new(0);
        let p = single_node_graph("X");
        let d = single_node_graph("X");
        // With max_results=0, the empty-pattern branch always returns 1 trivial result,
        // but for non-empty patterns we expect 0.
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 0);
    }

    // ── is_isomorphic ─────────────────────────────────────────────────────

    #[test]
    fn test_isomorphic_empty_graphs() {
        let m = make_matcher();
        assert!(m.is_isomorphic(&RdfGraph::new(), &RdfGraph::new()));
    }

    #[test]
    fn test_isomorphic_single_node() {
        let m = make_matcher();
        let g1 = single_node_graph("A");
        let g2 = single_node_graph("A");
        assert!(m.is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_isomorphic_different_labels() {
        let m = make_matcher();
        let g1 = single_node_graph("A");
        let g2 = single_node_graph("B");
        assert!(!m.is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_isomorphic_different_sizes() {
        let m = make_matcher();
        let mut g1 = RdfGraph::new();
        g1.add_node(0, "A");
        g1.add_node(1, "B");
        let g2 = single_node_graph("A");
        assert!(!m.is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_isomorphic_edge_graphs() {
        let m = make_matcher();
        let g1 = single_edge_graph("A", "B", "p");
        let g2 = single_edge_graph("A", "B", "p");
        assert!(m.is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_not_isomorphic_edge_label_differs() {
        let m = make_matcher();
        let g1 = single_edge_graph("A", "B", "p");
        let g2 = single_edge_graph("A", "B", "q");
        assert!(!m.is_isomorphic(&g1, &g2));
    }

    // ── count_matches ─────────────────────────────────────────────────────

    #[test]
    fn test_count_matches_zero() {
        let m = make_matcher();
        let p = single_node_graph("Z");
        let d = single_node_graph("A");
        assert_eq!(m.count_matches(&p, &d), 0);
    }

    #[test]
    fn test_count_matches_many() {
        let m = make_matcher();
        let p = single_node_graph("N");
        let mut d = RdfGraph::new();
        for i in 0..5u32 {
            d.add_node(i, "N");
        }
        assert_eq!(m.count_matches(&p, &d), 5);
    }

    // ── out/in neighbor methods ───────────────────────────────────────────

    #[test]
    fn test_out_neighbors() {
        let g = single_edge_graph("A", "B", "p");
        let out = g.out_neighbors(0);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn test_in_neighbors() {
        let g = single_edge_graph("A", "B", "p");
        let inn = g.in_neighbors(1);
        assert_eq!(inn, vec![0]);
    }

    #[test]
    fn test_no_neighbors() {
        let g = single_node_graph("A");
        assert!(g.out_neighbors(0).is_empty());
        assert!(g.in_neighbors(0).is_empty());
    }

    // ── edge_mapping correctness ──────────────────────────────────────────

    #[test]
    fn test_edge_mapping_single_edge() {
        let m = make_matcher();
        let p = single_edge_graph("A", "B", "p");
        let d = single_edge_graph("A", "B", "p");
        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].edge_mapping, vec![0usize]);
    }

    #[test]
    fn test_edge_mapping_two_edges() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "A");
        p.add_node(1, "B");
        p.add_node(2, "C");
        p.add_edge(0, 1, "e");
        p.add_edge(1, 2, "e");

        let mut d = RdfGraph::new();
        d.add_node(10, "A");
        d.add_node(11, "B");
        d.add_node(12, "C");
        d.add_edge(10, 11, "e");
        d.add_edge(11, 12, "e");

        let r = m.find_matches(&p, &d);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].edge_mapping.len(), 2);
    }

    // ── Subgraph match (pattern smaller than data) ────────────────────────

    #[test]
    fn test_subgraph_match_smaller_pattern() {
        let m = make_matcher();
        // Pattern: node(N) → node(N) with edge label "knows"
        // Both endpoints have label "N" so multiple matches are possible.
        let mut p = RdfGraph::new();
        p.add_node(0, "N");
        p.add_node(1, "N");
        p.add_edge(0, 1, "knows");

        let mut d = RdfGraph::new();
        d.add_node(0, "N");
        d.add_node(1, "N");
        d.add_node(2, "N");
        d.add_edge(0, 1, "knows");
        d.add_edge(1, 2, "knows");

        let r = m.find_matches(&p, &d);
        // Should find 0→1 and 1→2 (and also 0→2 if that edge existed, but it doesn't).
        // Two directed edges N→N means 2 matches.
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_large_data_no_match() {
        let m = make_matcher();
        let p = single_edge_graph("X", "Y", "special");
        let mut d = RdfGraph::new();
        for i in 0..10u32 {
            d.add_node(i, if i % 2 == 0 { "X" } else { "Y" });
        }
        for i in (0..10u32).step_by(2) {
            d.add_edge(i, i + 1, "other"); // wrong edge label
        }
        assert!(m.find_matches(&p, &d).is_empty());
    }

    #[test]
    fn test_isolated_nodes_in_pattern() {
        let m = make_matcher();
        let mut p = RdfGraph::new();
        p.add_node(0, "A");
        p.add_node(1, "B"); // isolated

        let mut d = RdfGraph::new();
        d.add_node(0, "A");
        d.add_node(1, "B");
        d.add_node(2, "C");

        let r = m.find_matches(&p, &d);
        // (0→0,1→1) and (0→0,1→2 — but label C≠B so no) → only 1
        assert!(!r.is_empty());
    }
}
