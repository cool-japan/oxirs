// Rule dependency graph for optimization (v1.1.0 round 11)
//
// Implements a directed graph of rule dependencies with:
// - Topological sort (Kahn's algorithm) for execution ordering
// - Strongly connected component detection (Tarjan's algorithm)
// - Stratifiability check (no negative cycles)

use std::collections::{HashMap, HashSet, VecDeque};

/// Type of dependency between two rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyType {
    /// Rule `to` uses a predicate produced by rule `from`
    Positive,
    /// Rule `to` uses a predicate negated that is produced by rule `from`
    Negative,
    /// Dependency exists but may be absent without breaking correctness
    Optional,
}

/// A directed dependency edge in the rule graph
#[derive(Debug, Clone)]
pub struct RuleDependency {
    pub from_rule: String,
    pub to_rule: String,
    pub dep_type: DependencyType,
}

impl RuleDependency {
    /// Create a new rule dependency
    pub fn new(from: impl Into<String>, to: impl Into<String>, dep_type: DependencyType) -> Self {
        Self {
            from_rule: from.into(),
            to_rule: to.into(),
            dep_type,
        }
    }
}

/// A node in the rule graph representing a single rule
#[derive(Debug, Clone)]
pub struct RuleNode {
    pub id: String,
    /// Higher priority = executed earlier within the same topological level
    pub priority: i32,
    /// Estimated number of firings per inference cycle
    pub estimated_firings: usize,
}

impl RuleNode {
    /// Create a new rule node
    pub fn new(id: impl Into<String>, priority: i32, estimated_firings: usize) -> Self {
        Self {
            id: id.into(),
            priority,
            estimated_firings,
        }
    }
}

/// Errors that can arise from rule graph operations
#[derive(Debug)]
pub enum GraphError {
    /// A cycle was detected; the first element is one member of the cycle
    CycleDetected(Vec<String>),
    /// A requested rule was not found in the graph
    RuleNotFound(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::CycleDetected(nodes) => {
                write!(f, "Cycle detected involving: {:?}", nodes)
            }
            GraphError::RuleNotFound(id) => write!(f, "Rule not found: {id}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Directed graph of rule dependencies
pub struct RuleGraph {
    nodes: HashMap<String, RuleNode>,
    edges: Vec<RuleDependency>,
}

impl RuleGraph {
    /// Create an empty rule graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a rule node; replaces any existing node with the same id
    pub fn add_rule(&mut self, node: RuleNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add a dependency edge. Both rules must be added via `add_rule` first,
    /// but this method is permissive and simply records the edge regardless.
    pub fn add_dependency(&mut self, from: &str, to: &str, dep_type: DependencyType) {
        self.edges.push(RuleDependency::new(from, to, dep_type));
    }

    /// Return the number of rule nodes in the graph
    pub fn rule_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of dependency edges in the graph
    pub fn dependency_count(&self) -> usize {
        self.edges.len()
    }

    /// Return all dependency edges where `from_rule == rule_id`
    pub fn dependencies_of(&self, rule_id: &str) -> Vec<&RuleDependency> {
        self.edges
            .iter()
            .filter(|e| e.from_rule == rule_id)
            .collect()
    }

    /// Return all dependency edges where `to_rule == rule_id`
    pub fn dependents_of(&self, rule_id: &str) -> Vec<&RuleDependency> {
        self.edges.iter().filter(|e| e.to_rule == rule_id).collect()
    }

    /// Return true if the graph contains any cycle (among the nodes actually in `self.nodes`)
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().is_err()
    }

    /// Perform a topological sort using Kahn's algorithm.
    ///
    /// Returns `Ok(Vec<String>)` with nodes in topological order,
    /// or `Err(GraphError::CycleDetected(_))` if a cycle exists.
    pub fn topological_sort(&self) -> Result<Vec<String>, GraphError> {
        // Only include nodes that are in self.nodes
        let node_ids: HashSet<&str> = self.nodes.keys().map(String::as_str).collect();

        // Build adjacency list and in-degree map
        let mut in_degree: HashMap<&str, usize> = node_ids.iter().map(|id| (*id, 0)).collect();
        let mut adj: HashMap<&str, Vec<&str>> = node_ids.iter().map(|id| (*id, vec![])).collect();

        for edge in &self.edges {
            if node_ids.contains(edge.from_rule.as_str())
                && node_ids.contains(edge.to_rule.as_str())
            {
                adj.entry(edge.from_rule.as_str())
                    .or_default()
                    .push(edge.to_rule.as_str());
                *in_degree.entry(edge.to_rule.as_str()).or_insert(0) += 1;
            }
        }

        // Collect nodes with in-degree 0, sorted for determinism (by id)
        let mut queue: VecDeque<&str> = {
            let mut zero: Vec<&str> = in_degree
                .iter()
                .filter(|(_, &d)| d == 0)
                .map(|(id, _)| *id)
                .collect();
            zero.sort_unstable();
            zero.into_iter().collect()
        };

        let mut sorted: Vec<String> = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            sorted.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                let mut next: Vec<&str> = neighbors.clone();
                next.sort_unstable();
                for neighbor in next {
                    let deg = in_degree.entry(neighbor).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if sorted.len() == self.nodes.len() {
            Ok(sorted)
        } else {
            // Find nodes not in sorted (cycle members)
            let sorted_set: HashSet<&str> = sorted.iter().map(String::as_str).collect();
            let cycle_members: Vec<String> = self
                .nodes
                .keys()
                .filter(|id| !sorted_set.contains(id.as_str()))
                .cloned()
                .collect();
            Err(GraphError::CycleDetected(cycle_members))
        }
    }

    /// Compute strongly connected components using Tarjan's algorithm.
    ///
    /// Returns a Vec of SCCs, each SCC is a `Vec<String>` of rule ids.
    /// SCCs are returned in reverse topological order (a leaf SCC appears first).
    pub fn strongly_connected_components(&self) -> Vec<Vec<String>> {
        let node_ids: Vec<&str> = {
            let mut ids: Vec<&str> = self.nodes.keys().map(String::as_str).collect();
            ids.sort_unstable();
            ids
        };

        let n = node_ids.len();
        let id_index: HashMap<&str, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        // Adjacency list (indices)
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for edge in &self.edges {
            if let (Some(&fi), Some(&ti)) = (
                id_index.get(edge.from_rule.as_str()),
                id_index.get(edge.to_rule.as_str()),
            ) {
                adj[fi].push(ti);
            }
        }

        // Tarjan's algorithm state
        let mut index_counter = 0usize;
        let mut stack: Vec<usize> = Vec::new();
        let mut on_stack = vec![false; n];
        let mut indices: Vec<Option<usize>> = vec![None; n];
        let mut lowlinks: Vec<usize> = vec![0; n];
        let mut sccs: Vec<Vec<String>> = Vec::new();

        #[allow(clippy::too_many_arguments)]
        fn strongconnect(
            v: usize,
            adj: &Vec<Vec<usize>>,
            index_counter: &mut usize,
            stack: &mut Vec<usize>,
            on_stack: &mut Vec<bool>,
            indices: &mut Vec<Option<usize>>,
            lowlinks: &mut Vec<usize>,
            sccs: &mut Vec<Vec<String>>,
            node_ids: &[&str],
        ) {
            indices[v] = Some(*index_counter);
            lowlinks[v] = *index_counter;
            *index_counter += 1;
            stack.push(v);
            on_stack[v] = true;

            for &w in &adj[v] {
                if indices[w].is_none() {
                    strongconnect(
                        w,
                        adj,
                        index_counter,
                        stack,
                        on_stack,
                        indices,
                        lowlinks,
                        sccs,
                        node_ids,
                    );
                    lowlinks[v] = lowlinks[v].min(lowlinks[w]);
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w].unwrap_or(0));
                }
            }

            // Root of an SCC
            if lowlinks[v] == indices[v].unwrap_or(usize::MAX) {
                let mut scc = Vec::new();
                loop {
                    let w = stack.pop().unwrap_or(0);
                    on_stack[w] = false;
                    scc.push(node_ids[w].to_string());
                    if w == v {
                        break;
                    }
                }
                sccs.push(scc);
            }
        }

        for i in 0..n {
            if indices[i].is_none() {
                strongconnect(
                    i,
                    &adj,
                    &mut index_counter,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut sccs,
                    &node_ids,
                );
            }
        }

        sccs
    }

    /// Return a total execution order that respects topological ordering and
    /// within each level, sorts by descending priority (higher first).
    ///
    /// Returns an error if the graph has a cycle.
    pub fn execution_order(&self) -> Vec<String> {
        // Build level sets from Kahn's algorithm
        let node_ids: HashSet<&str> = self.nodes.keys().map(String::as_str).collect();

        let mut in_degree: HashMap<&str, usize> = node_ids.iter().map(|id| (*id, 0)).collect();
        let mut adj: HashMap<&str, Vec<&str>> = node_ids.iter().map(|id| (*id, vec![])).collect();

        for edge in &self.edges {
            if node_ids.contains(edge.from_rule.as_str())
                && node_ids.contains(edge.to_rule.as_str())
            {
                adj.entry(edge.from_rule.as_str())
                    .or_default()
                    .push(edge.to_rule.as_str());
                *in_degree.entry(edge.to_rule.as_str()).or_insert(0) += 1;
            }
        }

        let mut result: Vec<String> = Vec::with_capacity(self.nodes.len());
        let mut remaining: HashSet<&str> = node_ids.clone();

        while !remaining.is_empty() {
            // Collect all nodes with in_degree 0
            let mut level: Vec<&str> = remaining
                .iter()
                .copied()
                .filter(|id| *in_degree.get(*id).unwrap_or(&0) == 0)
                .collect();

            if level.is_empty() {
                // Cycle: just drain remaining in arbitrary order for graceful degradation
                let mut rest: Vec<&str> = remaining.iter().copied().collect();
                rest.sort_unstable_by(|a, b| {
                    let pa = self.nodes.get(*a).map(|n| n.priority).unwrap_or(0);
                    let pb = self.nodes.get(*b).map(|n| n.priority).unwrap_or(0);
                    pb.cmp(&pa)
                });
                for id in rest {
                    result.push(id.to_string());
                    remaining.remove(id);
                }
                break;
            }

            // Sort level by descending priority, then by id for determinism
            level.sort_unstable_by(|a, b| {
                let pa = self.nodes.get(*a).map(|n| n.priority).unwrap_or(0);
                let pb = self.nodes.get(*b).map(|n| n.priority).unwrap_or(0);
                pb.cmp(&pa).then_with(|| a.cmp(b))
            });

            for id in level {
                result.push(id.to_string());
                remaining.remove(id);
                if let Some(neighbors) = adj.get(id) {
                    for neighbor in neighbors {
                        let deg = in_degree.entry(neighbor).or_insert(0);
                        *deg = deg.saturating_sub(1);
                    }
                }
            }
        }

        result
    }

    /// Return true if the rule graph is stratifiable.
    ///
    /// A graph is stratifiable iff there are no negative cycles — i.e., no cycle
    /// contains a `Negative` dependency edge.
    pub fn is_stratum_stratifiable(&self) -> bool {
        // Build a graph of only negative edges and check for cycles in that sub-graph,
        // but also considering that a negative edge must not lie on any cycle.
        //
        // Proper definition: a rule graph is stratifiable iff there is no cycle
        // containing a negative dependency.
        //
        // Algorithm:
        // 1. Find all SCCs.
        // 2. An SCC with >1 node is a cycle.
        //    Also an SCC with 1 node that has a self-loop is a cycle.
        // 3. Check if any cycle SCC contains a negative edge.

        let sccs = self.strongly_connected_components();

        for scc in &sccs {
            let is_cycle = if scc.len() > 1 {
                true
            } else {
                // Check for self-loop
                let id = &scc[0];
                self.edges
                    .iter()
                    .any(|e| e.from_rule == *id && e.to_rule == *id)
            };

            if is_cycle {
                let scc_set: HashSet<&str> = scc.iter().map(String::as_str).collect();
                // Check if any negative edge lies within this SCC
                let has_negative = self.edges.iter().any(|e| {
                    e.dep_type == DependencyType::Negative
                        && scc_set.contains(e.from_rule.as_str())
                        && scc_set.contains(e.to_rule.as_str())
                });
                if has_negative {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for RuleGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, priority: i32) -> RuleNode {
        RuleNode::new(id, priority, 1)
    }

    // ── Basic construction ─────────────────────────────────────────────────

    #[test]
    fn test_new_graph_empty() {
        let g = RuleGraph::new();
        assert_eq!(g.rule_count(), 0);
        assert_eq!(g.dependency_count(), 0);
    }

    #[test]
    fn test_default_graph_empty() {
        let g = RuleGraph::default();
        assert_eq!(g.rule_count(), 0);
    }

    #[test]
    fn test_add_single_rule() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        assert_eq!(g.rule_count(), 1);
    }

    #[test]
    fn test_add_multiple_rules() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        g.add_rule(make_node("r2", 1));
        g.add_rule(make_node("r3", 2));
        assert_eq!(g.rule_count(), 3);
    }

    #[test]
    fn test_add_rule_overwrites_existing() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        g.add_rule(RuleNode::new("r1", 5, 10));
        assert_eq!(g.rule_count(), 1);
        assert_eq!(g.nodes["r1"].priority, 5);
    }

    #[test]
    fn test_add_dependency() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        g.add_rule(make_node("r2", 0));
        g.add_dependency("r1", "r2", DependencyType::Positive);
        assert_eq!(g.dependency_count(), 1);
    }

    #[test]
    fn test_rule_count() {
        let mut g = RuleGraph::new();
        for i in 0..5 {
            g.add_rule(make_node(&format!("r{i}"), 0));
        }
        assert_eq!(g.rule_count(), 5);
    }

    #[test]
    fn test_dependency_count() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_rule(make_node("c", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Negative);
        assert_eq!(g.dependency_count(), 2);
    }

    // ── dependencies_of / dependents_of ───────────────────────────────────

    #[test]
    fn test_dependencies_of() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        g.add_rule(make_node("r2", 0));
        g.add_rule(make_node("r3", 0));
        g.add_dependency("r1", "r2", DependencyType::Positive);
        g.add_dependency("r1", "r3", DependencyType::Negative);
        let deps = g.dependencies_of("r1");
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_dependencies_of_empty() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        assert_eq!(g.dependencies_of("r1").len(), 0);
    }

    #[test]
    fn test_dependents_of() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        g.add_rule(make_node("r2", 0));
        g.add_rule(make_node("r3", 0));
        g.add_dependency("r1", "r3", DependencyType::Positive);
        g.add_dependency("r2", "r3", DependencyType::Optional);
        let deps = g.dependents_of("r3");
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_dependents_of_empty() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("r1", 0));
        assert_eq!(g.dependents_of("r1").len(), 0);
    }

    // ── topological_sort ──────────────────────────────────────────────────

    #[test]
    fn test_topological_sort_linear_chain() -> Result<(), Box<dyn std::error::Error>> {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_rule(make_node("c", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Positive);
        let sorted = g.topological_sort().expect("No cycle");
        let ia = sorted
            .iter()
            .position(|x| x == "a")
            .ok_or("expected Some value")?;
        let ib = sorted
            .iter()
            .position(|x| x == "b")
            .ok_or("expected Some value")?;
        let ic = sorted
            .iter()
            .position(|x| x == "c")
            .ok_or("expected Some value")?;
        assert!(ia < ib);
        assert!(ib < ic);
        Ok(())
    }

    #[test]
    fn test_topological_sort_single_node() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("only", 0));
        let sorted = g.topological_sort().expect("No cycle");
        assert_eq!(sorted, vec!["only"]);
    }

    #[test]
    fn test_topological_sort_detects_cycle() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "a", DependencyType::Positive);
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_topological_sort_empty_graph() {
        let g = RuleGraph::new();
        let sorted = g.topological_sort().expect("No cycle");
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_topological_sort_diamond() -> Result<(), Box<dyn std::error::Error>> {
        let mut g = RuleGraph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("a", "c", DependencyType::Positive);
        g.add_dependency("b", "d", DependencyType::Positive);
        g.add_dependency("c", "d", DependencyType::Positive);
        let sorted = g.topological_sort().expect("No cycle");
        let ia = sorted
            .iter()
            .position(|x| x == "a")
            .ok_or("expected Some value")?;
        let id = sorted
            .iter()
            .position(|x| x == "d")
            .ok_or("expected Some value")?;
        assert!(ia < id);
        Ok(())
    }

    // ── has_cycle ─────────────────────────────────────────────────────────

    #[test]
    fn test_has_cycle_false() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_has_cycle_true() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "a", DependencyType::Positive);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_has_cycle_self_loop() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_dependency("a", "a", DependencyType::Positive);
        assert!(g.has_cycle());
    }

    // ── strongly_connected_components ─────────────────────────────────────

    #[test]
    fn test_scc_single_node_no_self_loop() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        let sccs = g.strongly_connected_components();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec!["a"]);
    }

    #[test]
    fn test_scc_no_cycle_each_is_own() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_rule(make_node("c", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Positive);
        let sccs = g.strongly_connected_components();
        // Each node is its own SCC
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn test_scc_full_cycle() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_rule(make_node("c", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Positive);
        g.add_dependency("c", "a", DependencyType::Positive);
        let sccs = g.strongly_connected_components();
        // Should be one SCC with 3 members
        let big: Vec<_> = sccs.iter().filter(|s| s.len() > 1).collect();
        assert_eq!(big.len(), 1);
        assert_eq!(big[0].len(), 3);
    }

    #[test]
    fn test_scc_two_separate_cycles() {
        let mut g = RuleGraph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "a", DependencyType::Positive);
        g.add_dependency("c", "d", DependencyType::Positive);
        g.add_dependency("d", "c", DependencyType::Positive);
        let sccs = g.strongly_connected_components();
        let big: Vec<_> = sccs.iter().filter(|s| s.len() > 1).collect();
        assert_eq!(big.len(), 2);
    }

    // ── execution_order ───────────────────────────────────────────────────

    #[test]
    fn test_execution_order_respects_priority() -> Result<(), Box<dyn std::error::Error>> {
        let mut g = RuleGraph::new();
        // a and b have no dependencies between them → same level
        g.add_rule(RuleNode::new("a", 10, 1)); // high priority
        g.add_rule(RuleNode::new("b", 1, 1)); // low priority
        g.add_rule(RuleNode::new("c", 5, 1));
        let order = g.execution_order();
        let ia = order
            .iter()
            .position(|x| x == "a")
            .ok_or("expected Some value")?;
        let ib = order
            .iter()
            .position(|x| x == "b")
            .ok_or("expected Some value")?;
        // a (priority 10) should come before b (priority 1)
        assert!(ia < ib);
        Ok(())
    }

    #[test]
    fn test_execution_order_topological_before_priority() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut g = RuleGraph::new();
        g.add_rule(RuleNode::new("a", 1, 1)); // low priority but no deps
        g.add_rule(RuleNode::new("b", 100, 1)); // high priority but depends on a
        g.add_dependency("a", "b", DependencyType::Positive);
        let order = g.execution_order();
        let ia = order
            .iter()
            .position(|x| x == "a")
            .ok_or("expected Some value")?;
        let ib = order
            .iter()
            .position(|x| x == "b")
            .ok_or("expected Some value")?;
        assert!(ia < ib);
        Ok(())
    }

    #[test]
    fn test_execution_order_includes_all_nodes() {
        let mut g = RuleGraph::new();
        for i in 0..5 {
            g.add_rule(make_node(&format!("r{i}"), i));
        }
        let order = g.execution_order();
        assert_eq!(order.len(), 5);
    }

    // ── is_stratum_stratifiable ────────────────────────────────────────────

    #[test]
    fn test_stratifiable_acyclic() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Negative);
        assert!(g.is_stratum_stratifiable());
    }

    #[test]
    fn test_stratifiable_positive_cycle() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "a", DependencyType::Positive);
        // Positive-only cycle is OK for stratifiability
        assert!(g.is_stratum_stratifiable());
    }

    #[test]
    fn test_not_stratifiable_negative_cycle() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Negative);
        g.add_dependency("b", "a", DependencyType::Positive);
        // Negative edge in a cycle → not stratifiable
        assert!(!g.is_stratum_stratifiable());
    }

    #[test]
    fn test_stratifiable_empty_graph() {
        let g = RuleGraph::new();
        assert!(g.is_stratum_stratifiable());
    }

    #[test]
    fn test_stratifiable_self_negative_loop() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_dependency("a", "a", DependencyType::Negative);
        assert!(!g.is_stratum_stratifiable());
    }

    #[test]
    fn test_stratifiable_complex_positive_cycle() {
        let mut g = RuleGraph::new();
        for id in ["a", "b", "c"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Positive);
        g.add_dependency("c", "a", DependencyType::Positive);
        assert!(g.is_stratum_stratifiable());
    }

    // ── GraphError ─────────────────────────────────────────────────────────

    #[test]
    fn test_graph_error_cycle_detected_display() {
        let err = GraphError::CycleDetected(vec!["a".to_string(), "b".to_string()]);
        let msg = format!("{err}");
        assert!(msg.contains("Cycle"));
    }

    #[test]
    fn test_graph_error_rule_not_found_display() {
        let err = GraphError::RuleNotFound("xyz".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn test_dependency_type_eq() {
        assert_eq!(DependencyType::Positive, DependencyType::Positive);
        assert_ne!(DependencyType::Positive, DependencyType::Negative);
    }

    #[test]
    fn test_rule_node_fields() {
        let n = RuleNode::new("test_rule", 42, 100);
        assert_eq!(n.id, "test_rule");
        assert_eq!(n.priority, 42);
        assert_eq!(n.estimated_firings, 100);
    }

    #[test]
    fn test_rule_dependency_fields() {
        let d = RuleDependency::new("from", "to", DependencyType::Optional);
        assert_eq!(d.from_rule, "from");
        assert_eq!(d.to_rule, "to");
        assert_eq!(d.dep_type, DependencyType::Optional);
    }

    #[test]
    fn test_topological_sort_parallel_branches() -> Result<(), Box<dyn std::error::Error>> {
        let mut g = RuleGraph::new();
        for id in ["root", "b1", "b2", "leaf"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("root", "b1", DependencyType::Positive);
        g.add_dependency("root", "b2", DependencyType::Positive);
        g.add_dependency("b1", "leaf", DependencyType::Positive);
        g.add_dependency("b2", "leaf", DependencyType::Positive);
        let sorted = g.topological_sort().expect("No cycle");
        let i_root = sorted
            .iter()
            .position(|x| x == "root")
            .ok_or("expected Some value")?;
        let i_leaf = sorted
            .iter()
            .position(|x| x == "leaf")
            .ok_or("expected Some value")?;
        assert!(i_root < i_leaf);
        Ok(())
    }

    #[test]
    fn test_execution_order_single_node() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("only", 42));
        let order = g.execution_order();
        assert_eq!(order, vec!["only"]);
    }

    #[test]
    fn test_scc_empty_graph() {
        let g = RuleGraph::new();
        let sccs = g.strongly_connected_components();
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_dependencies_of_includes_type() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Optional);
        let deps = g.dependencies_of("a");
        assert_eq!(deps[0].dep_type, DependencyType::Optional);
    }

    #[test]
    fn test_has_cycle_three_node_cycle() {
        let mut g = RuleGraph::new();
        for id in ["a", "b", "c"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("a", "b", DependencyType::Positive);
        g.add_dependency("b", "c", DependencyType::Positive);
        g.add_dependency("c", "a", DependencyType::Positive);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_stratifiable_optional_cycle() {
        let mut g = RuleGraph::new();
        g.add_rule(make_node("a", 0));
        g.add_rule(make_node("b", 0));
        g.add_dependency("a", "b", DependencyType::Optional);
        g.add_dependency("b", "a", DependencyType::Optional);
        // Optional-only cycle is stratifiable
        assert!(g.is_stratum_stratifiable());
    }

    #[test]
    fn test_dependents_of_includes_all() {
        let mut g = RuleGraph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_rule(make_node(id, 0));
        }
        g.add_dependency("a", "d", DependencyType::Positive);
        g.add_dependency("b", "d", DependencyType::Positive);
        g.add_dependency("c", "d", DependencyType::Negative);
        let deps = g.dependents_of("d");
        assert_eq!(deps.len(), 3);
    }
}
