/// Truth Maintenance System (TMS) for belief revision.
///
/// A classical JTMS (Justification-based Truth Maintenance System) that tracks
/// beliefs (nodes) and their justifications. Retracting an axiom propagates
/// the "unsupported" status to all transitively dependent nodes.
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How a TMS node is justified.
#[derive(Debug, Clone, PartialEq)]
pub enum Justification {
    /// The node is an axiom — always supported unless explicitly retracted.
    Axiom,
    /// The node is derived from one or more other nodes.
    Derived {
        /// Indices of nodes this node depends on.
        from: Vec<usize>,
    },
}

/// A single belief node in the TMS.
#[derive(Debug, Clone)]
pub struct TmsNode {
    /// Unique identifier.
    pub id: usize,
    /// The statement this node represents (e.g. a fact, rule conclusion).
    pub statement: String,
    /// Whether this node is currently supported (believed).
    pub supported: bool,
    /// How this node is justified.
    pub justification: Justification,
}

/// Justification-based Truth Maintenance System.
///
/// Nodes can be axioms (always-on unless retracted) or derived (supported
/// only when all antecedent nodes are supported).
#[derive(Debug, Default)]
pub struct TruthMaintenanceSystem {
    nodes: Vec<TmsNode>,
    /// Set of node IDs that have been explicitly retracted.
    assumptions: HashSet<usize>,
}

impl TruthMaintenanceSystem {
    /// Create an empty TMS.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            assumptions: HashSet::new(),
        }
    }

    /// Add an axiom node.  Axiom nodes are always supported until retracted.
    ///
    /// Returns the new node's `id`.
    pub fn add_axiom(&mut self, statement: impl Into<String>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TmsNode {
            id,
            statement: statement.into(),
            supported: true,
            justification: Justification::Axiom,
        });
        id
    }

    /// Add a derived node that is supported exactly when all nodes in `from`
    /// are supported.
    ///
    /// Returns the new node's `id`.
    pub fn add_derived(&mut self, statement: impl Into<String>, from: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let supported = from
            .iter()
            .all(|&dep| self.nodes.get(dep).map(|n| n.supported).unwrap_or(false));
        self.nodes.push(TmsNode {
            id,
            statement: statement.into(),
            supported,
            justification: Justification::Derived { from },
        });
        id
    }

    /// Retract (un-support) a node and propagate unsupport to all dependents.
    pub fn retract(&mut self, node_id: usize) {
        if node_id >= self.nodes.len() {
            return;
        }
        self.assumptions.insert(node_id);
        self.nodes[node_id].supported = false;
        // Propagate transitively to all reachable dependents
        let affected = self.reachable(node_id);
        for dep_id in affected {
            self.nodes[dep_id].supported = false;
        }
    }

    /// Attempt to re-support a node that was previously retracted.
    ///
    /// For an axiom: re-supports unless still in the retraction set (but we
    /// remove it from the set here).  For a derived node: re-supports only
    /// when all antecedents are supported.
    pub fn restore(&mut self, node_id: usize) {
        if node_id >= self.nodes.len() {
            return;
        }
        self.assumptions.remove(&node_id);
        let supported = self.compute_supported(node_id);
        self.nodes[node_id].supported = supported;
        // Re-evaluate downstream nodes transitively
        let downstream = self.reachable(node_id);
        for dep_id in downstream {
            let s = self.compute_supported(dep_id);
            self.nodes[dep_id].supported = s;
        }
    }

    /// Return whether node `node_id` is currently supported.
    pub fn is_supported(&self, node_id: usize) -> bool {
        self.nodes
            .get(node_id)
            .map(|n| n.supported)
            .unwrap_or(false)
    }

    /// Count supported nodes.
    pub fn supported_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.supported).count()
    }

    /// Count unsupported nodes.
    pub fn unsupported_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.supported).count()
    }

    /// Return all node IDs that **directly** depend on `node_id`.
    pub fn dependents(&self, node_id: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| match &n.justification {
                Justification::Derived { from } => from.contains(&node_id),
                _ => false,
            })
            .map(|n| n.id)
            .collect()
    }

    /// Return a slice of all nodes.
    pub fn all_nodes(&self) -> &[TmsNode] {
        &self.nodes
    }

    /// Return all node IDs transitively reachable from `root_id` via the
    /// "depended-upon-by" relation (i.e. nodes that would be invalidated if
    /// `root_id` were retracted).
    pub fn reachable(&self, root_id: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut stack = vec![root_id];
        while let Some(current) = stack.pop() {
            for dep in self.dependents(current) {
                if visited.insert(dep) {
                    stack.push(dep);
                }
            }
        }
        let mut result: Vec<usize> = visited.into_iter().collect();
        result.sort_unstable();
        result
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Recompute whether node `id` should be supported given current state.
    fn compute_supported(&self, id: usize) -> bool {
        if self.assumptions.contains(&id) {
            return false;
        }
        match self.nodes.get(id) {
            None => false,
            Some(node) => match &node.justification {
                Justification::Axiom => true,
                Justification::Derived { from } => from.iter().all(|&dep| self.is_supported(dep)),
            },
        }
    }

    /// Build an index of node_id → indices of nodes that depend on it.
    #[allow(dead_code)]
    fn build_dependent_index(&self) -> HashMap<usize, Vec<usize>> {
        let mut index: HashMap<usize, Vec<usize>> = HashMap::new();
        for node in &self.nodes {
            if let Justification::Derived { from } = &node.justification {
                for &dep in from {
                    index.entry(dep).or_default().push(node.id);
                }
            }
        }
        index
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic construction ---
    #[test]
    fn test_new_tms_is_empty() {
        let tms = TruthMaintenanceSystem::new();
        assert_eq!(tms.all_nodes().len(), 0);
        assert_eq!(tms.supported_count(), 0);
        assert_eq!(tms.unsupported_count(), 0);
    }

    #[test]
    fn test_add_axiom_returns_id() {
        let mut tms = TruthMaintenanceSystem::new();
        let id = tms.add_axiom("A");
        assert_eq!(id, 0);
    }

    #[test]
    fn test_axiom_is_supported() {
        let mut tms = TruthMaintenanceSystem::new();
        let id = tms.add_axiom("A");
        assert!(tms.is_supported(id));
    }

    #[test]
    fn test_multiple_axioms_ids_sequential() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_axiom("B");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // --- Derived nodes ---
    #[test]
    fn test_derived_node_supported_when_deps_supported() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_axiom("B");
        let c = tms.add_derived("C", vec![a, b]);
        assert!(tms.is_supported(c));
    }

    #[test]
    fn test_derived_node_unsupported_when_dep_unsupported() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B", vec![a]);
        tms.retract(a);
        assert!(!tms.is_supported(b));
    }

    #[test]
    fn test_derived_node_with_no_deps_is_supported() {
        let mut tms = TruthMaintenanceSystem::new();
        let c = tms.add_derived("C", vec![]);
        assert!(tms.is_supported(c));
    }

    // --- Retraction ---
    #[test]
    fn test_retract_axiom() {
        let mut tms = TruthMaintenanceSystem::new();
        let id = tms.add_axiom("A");
        tms.retract(id);
        assert!(!tms.is_supported(id));
    }

    #[test]
    fn test_retract_propagates_one_level() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        tms.retract(a);
        assert!(!tms.is_supported(a));
        assert!(!tms.is_supported(b));
    }

    #[test]
    fn test_retract_propagates_two_levels() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        let c = tms.add_derived("C <- B", vec![b]);
        tms.retract(a);
        assert!(!tms.is_supported(b));
        assert!(!tms.is_supported(c));
    }

    #[test]
    fn test_retract_does_not_affect_unrelated() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_axiom("B");
        tms.retract(a);
        assert!(!tms.is_supported(a));
        assert!(tms.is_supported(b));
    }

    #[test]
    fn test_retract_invalid_id_is_noop() {
        let mut tms = TruthMaintenanceSystem::new();
        tms.retract(999); // Should not panic
    }

    // --- Restore ---
    #[test]
    fn test_restore_axiom() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        tms.retract(a);
        tms.restore(a);
        assert!(tms.is_supported(a));
    }

    #[test]
    fn test_restore_propagates_to_derived() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        tms.retract(a);
        assert!(!tms.is_supported(b));
        tms.restore(a);
        assert!(tms.is_supported(a));
        assert!(tms.is_supported(b));
    }

    #[test]
    fn test_restore_does_not_restore_if_other_dep_still_retracted() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_axiom("B");
        let c = tms.add_derived("C <- A, B", vec![a, b]);
        tms.retract(a);
        tms.retract(b);
        tms.restore(a);
        // c still depends on b which is retracted
        assert!(!tms.is_supported(c));
    }

    // --- supported_count / unsupported_count ---
    #[test]
    fn test_counts_after_retract() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let _b = tms.add_axiom("B");
        let _c = tms.add_derived("C <- A", vec![a]);
        tms.retract(a);
        assert_eq!(tms.supported_count(), 1); // only B
        assert_eq!(tms.unsupported_count(), 2); // A and C
    }

    #[test]
    fn test_counts_all_supported() {
        let mut tms = TruthMaintenanceSystem::new();
        let _a = tms.add_axiom("A");
        let _b = tms.add_axiom("B");
        assert_eq!(tms.supported_count(), 2);
        assert_eq!(tms.unsupported_count(), 0);
    }

    // --- dependents ---
    #[test]
    fn test_dependents_empty_when_none() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        assert!(tms.dependents(a).is_empty());
    }

    #[test]
    fn test_dependents_returns_derived_nodes() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        let c = tms.add_derived("C <- A", vec![a]);
        let mut deps = tms.dependents(a);
        deps.sort_unstable();
        assert_eq!(deps, vec![b, c]);
    }

    // --- reachable ---
    #[test]
    fn test_reachable_empty() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        assert!(tms.reachable(a).is_empty());
    }

    #[test]
    fn test_reachable_chain() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        let c = tms.add_derived("C <- B", vec![b]);
        let reachable = tms.reachable(a);
        assert!(reachable.contains(&b));
        assert!(reachable.contains(&c));
    }

    #[test]
    fn test_reachable_diamond() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B <- A", vec![a]);
        let c = tms.add_derived("C <- A", vec![a]);
        let d = tms.add_derived("D <- B, C", vec![b, c]);
        let reachable = tms.reachable(a);
        assert!(reachable.contains(&b));
        assert!(reachable.contains(&c));
        assert!(reachable.contains(&d));
        assert_eq!(reachable.len(), 3);
    }

    // --- all_nodes ---
    #[test]
    fn test_all_nodes_returns_correct_slice() {
        let mut tms = TruthMaintenanceSystem::new();
        tms.add_axiom("A");
        tms.add_axiom("B");
        let nodes = tms.all_nodes();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].statement, "A");
        assert_eq!(nodes[1].statement, "B");
    }

    // --- justification content ---
    #[test]
    fn test_axiom_justification() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        assert_eq!(tms.all_nodes()[a].justification, Justification::Axiom);
    }

    #[test]
    fn test_derived_justification_from_ids() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        let b = tms.add_derived("B", vec![a]);
        assert_eq!(
            tms.all_nodes()[b].justification,
            Justification::Derived { from: vec![a] }
        );
    }

    // --- is_supported for out-of-range id ---
    #[test]
    fn test_is_supported_out_of_range() {
        let tms = TruthMaintenanceSystem::new();
        assert!(!tms.is_supported(999));
    }

    // --- Double retract is safe ---
    #[test]
    fn test_double_retract_is_idempotent() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        tms.retract(a);
        tms.retract(a);
        assert!(!tms.is_supported(a));
    }

    // --- Restore then retract ---
    #[test]
    fn test_restore_then_retract() {
        let mut tms = TruthMaintenanceSystem::new();
        let a = tms.add_axiom("A");
        tms.retract(a);
        tms.restore(a);
        assert!(tms.is_supported(a));
        tms.retract(a);
        assert!(!tms.is_supported(a));
    }
}
