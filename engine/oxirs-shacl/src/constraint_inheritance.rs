/// SHACL constraint inheritance via sh:and / sh:or / sh:not / sh:xone.
///
/// Provides a hierarchy of shapes linked by logical constraints so that
/// validators can determine which shapes must all be satisfied (sh:and),
/// at least one of (sh:or), none of (sh:not), or exactly one of (sh:xone).
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A reference to another SHACL shape by its string identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeRef {
    pub id: String,
}

impl ShapeRef {
    /// Create a new `ShapeRef` with the given `id`.
    pub fn new(id: impl Into<String>) -> Self {
        ShapeRef { id: id.into() }
    }
}

/// A logical combination constraint attached to a shape.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalConstraint {
    /// sh:and — all referenced shapes must be satisfied.
    And(Vec<ShapeRef>),
    /// sh:or — at least one referenced shape must be satisfied.
    Or(Vec<ShapeRef>),
    /// sh:not — the referenced shape must NOT be satisfied.
    Not(Box<ShapeRef>),
    /// sh:xone — exactly one of the referenced shapes must be satisfied.
    Xone(Vec<ShapeRef>),
}

/// An inherited constraint as seen from a specific source shape.
#[derive(Debug, Clone, PartialEq)]
pub struct InheritedConstraint {
    /// The shape that carries this logical constraint.
    pub source_shape: String,
    /// The constraint itself.
    pub constraint: LogicalConstraint,
}

/// Registry of shapes and their logical (inheritance) constraints.
#[derive(Debug, Default)]
pub struct ShapeHierarchy {
    /// shape_id → list of logical constraints declared on that shape.
    shapes: HashMap<String, Vec<LogicalConstraint>>,
}

impl ShapeHierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    /// Register a shape with no constraints (creates it if absent).
    pub fn add_shape(&mut self, shape_id: impl Into<String>) {
        self.shapes.entry(shape_id.into()).or_default();
    }

    /// Add an sh:and constraint to `shape_id`.
    pub fn add_and(&mut self, shape_id: &str, refs: Vec<ShapeRef>) {
        self.shapes
            .entry(shape_id.to_string())
            .or_default()
            .push(LogicalConstraint::And(refs));
    }

    /// Add an sh:or constraint to `shape_id`.
    pub fn add_or(&mut self, shape_id: &str, refs: Vec<ShapeRef>) {
        self.shapes
            .entry(shape_id.to_string())
            .or_default()
            .push(LogicalConstraint::Or(refs));
    }

    /// Add an sh:not constraint to `shape_id`.
    pub fn add_not(&mut self, shape_id: &str, shape_ref: ShapeRef) {
        self.shapes
            .entry(shape_id.to_string())
            .or_default()
            .push(LogicalConstraint::Not(Box::new(shape_ref)));
    }

    /// Add an sh:xone constraint to `shape_id`.
    pub fn add_xone(&mut self, shape_id: &str, refs: Vec<ShapeRef>) {
        self.shapes
            .entry(shape_id.to_string())
            .or_default()
            .push(LogicalConstraint::Xone(refs));
    }

    /// Return all shapes reachable via sh:and links starting from `shape_id`
    /// (i.e. the "ancestors" in the sense of shapes that must be co-satisfied).
    ///
    /// Traversal is breadth-first; the starting shape is not included.
    pub fn ancestors(&self, shape_id: &str) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(shape_id.to_string());
        visited.insert(shape_id.to_string());

        let mut result = Vec::new();
        while let Some(current) = queue.pop_front() {
            if let Some(constraints) = self.shapes.get(&current) {
                for constraint in constraints {
                    if let LogicalConstraint::And(refs) = constraint {
                        for shape_ref in refs {
                            if visited.insert(shape_ref.id.clone()) {
                                result.push(shape_ref.id.clone());
                                queue.push_back(shape_ref.id.clone());
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Evaluate sh:and for `shape_id`: all shapes in each And constraint must
    /// be present in `satisfied_shapes`.
    pub fn validate_and(&self, shape_id: &str, satisfied_shapes: &HashSet<String>) -> bool {
        let constraints = match self.shapes.get(shape_id) {
            None => return true, // no constraints → trivially valid
            Some(c) => c,
        };
        for constraint in constraints {
            if let LogicalConstraint::And(refs) = constraint {
                if !refs.iter().all(|r| satisfied_shapes.contains(&r.id)) {
                    return false;
                }
            }
        }
        true
    }

    /// Evaluate sh:or for `shape_id`: at least one shape in each Or constraint
    /// must be present in `satisfied_shapes`.
    pub fn validate_or(&self, shape_id: &str, satisfied_shapes: &HashSet<String>) -> bool {
        let constraints = match self.shapes.get(shape_id) {
            None => return true,
            Some(c) => c,
        };
        for constraint in constraints {
            if let LogicalConstraint::Or(refs) = constraint {
                if refs.is_empty() {
                    return false;
                }
                if !refs.iter().any(|r| satisfied_shapes.contains(&r.id)) {
                    return false;
                }
            }
        }
        true
    }

    /// Evaluate sh:xone for `shape_id`: exactly one shape in each Xone
    /// constraint must be present in `satisfied_shapes`.
    pub fn validate_xone(&self, shape_id: &str, satisfied_shapes: &HashSet<String>) -> bool {
        let constraints = match self.shapes.get(shape_id) {
            None => return true,
            Some(c) => c,
        };
        for constraint in constraints {
            if let LogicalConstraint::Xone(refs) = constraint {
                let count = refs
                    .iter()
                    .filter(|r| satisfied_shapes.contains(&r.id))
                    .count();
                if count != 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Return all shapes transitively required by `shape_id` via sh:and
    /// (including those reached through further And links).
    pub fn flatten_and(&self, shape_id: &str) -> Vec<String> {
        self.ancestors(shape_id)
    }

    /// Total number of registered shapes.
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// Return all constraints registered for a shape (for inspection/testing).
    pub fn constraints_for(&self, shape_id: &str) -> Vec<&LogicalConstraint> {
        self.shapes
            .get(shape_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_refs(ids: &[&str]) -> Vec<ShapeRef> {
        ids.iter().map(|id| ShapeRef::new(*id)).collect()
    }

    fn satisfied(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    // --- ShapeRef ---
    #[test]
    fn test_shape_ref_new() {
        let r = ShapeRef::new("ex:PersonShape");
        assert_eq!(r.id, "ex:PersonShape");
    }

    // --- ShapeHierarchy construction ---
    #[test]
    fn test_new_empty() {
        let h = ShapeHierarchy::new();
        assert_eq!(h.shape_count(), 0);
    }

    #[test]
    fn test_add_shape_increments_count() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        h.add_shape("B");
        assert_eq!(h.shape_count(), 2);
    }

    #[test]
    fn test_add_shape_idempotent() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        h.add_shape("A");
        assert_eq!(h.shape_count(), 1);
    }

    // --- add_and ---
    #[test]
    fn test_add_and_stored() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        let cs = h.constraints_for("A");
        assert_eq!(cs.len(), 1);
        assert!(matches!(cs[0], LogicalConstraint::And(_)));
    }

    // --- add_or ---
    #[test]
    fn test_add_or_stored() {
        let mut h = ShapeHierarchy::new();
        h.add_or("A", mk_refs(&["B", "C"]));
        let cs = h.constraints_for("A");
        assert!(matches!(cs[0], LogicalConstraint::Or(_)));
    }

    // --- add_not ---
    #[test]
    fn test_add_not_stored() {
        let mut h = ShapeHierarchy::new();
        h.add_not("A", ShapeRef::new("B"));
        let cs = h.constraints_for("A");
        assert!(matches!(cs[0], LogicalConstraint::Not(_)));
    }

    // --- add_xone ---
    #[test]
    fn test_add_xone_stored() {
        let mut h = ShapeHierarchy::new();
        h.add_xone("A", mk_refs(&["B", "C"]));
        let cs = h.constraints_for("A");
        assert!(matches!(cs[0], LogicalConstraint::Xone(_)));
    }

    // --- ancestors ---
    #[test]
    fn test_ancestors_empty_when_no_and() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        assert!(h.ancestors("A").is_empty());
    }

    #[test]
    fn test_ancestors_direct_and() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        let ancestors = h.ancestors("A");
        assert!(ancestors.contains(&"B".to_string()));
        assert!(ancestors.contains(&"C".to_string()));
    }

    #[test]
    fn test_ancestors_transitive() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        h.add_and("B", mk_refs(&["C"]));
        let ancestors = h.ancestors("A");
        assert!(ancestors.contains(&"B".to_string()));
        assert!(ancestors.contains(&"C".to_string()));
    }

    #[test]
    fn test_ancestors_no_cycles_with_loop() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        h.add_and("B", mk_refs(&["A"])); // cycle
        let ancestors = h.ancestors("A");
        // Should terminate; B is reachable, A is excluded (already visited)
        assert!(ancestors.contains(&"B".to_string()));
        // A should not appear again
        assert!(!ancestors.contains(&"A".to_string()));
    }

    // --- validate_and ---
    #[test]
    fn test_validate_and_no_constraints_true() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        assert!(h.validate_and("A", &satisfied(&[])));
    }

    #[test]
    fn test_validate_and_all_satisfied() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        assert!(h.validate_and("A", &satisfied(&["B", "C"])));
    }

    #[test]
    fn test_validate_and_partial_failure() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        assert!(!h.validate_and("A", &satisfied(&["B"])));
    }

    #[test]
    fn test_validate_and_none_satisfied_failure() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        assert!(!h.validate_and("A", &satisfied(&[])));
    }

    // --- validate_or ---
    #[test]
    fn test_validate_or_no_constraints_true() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        assert!(h.validate_or("A", &satisfied(&[])));
    }

    #[test]
    fn test_validate_or_one_satisfied() {
        let mut h = ShapeHierarchy::new();
        h.add_or("A", mk_refs(&["B", "C"]));
        assert!(h.validate_or("A", &satisfied(&["B"])));
    }

    #[test]
    fn test_validate_or_none_satisfied_failure() {
        let mut h = ShapeHierarchy::new();
        h.add_or("A", mk_refs(&["B", "C"]));
        assert!(!h.validate_or("A", &satisfied(&[])));
    }

    #[test]
    fn test_validate_or_all_satisfied() {
        let mut h = ShapeHierarchy::new();
        h.add_or("A", mk_refs(&["B", "C"]));
        assert!(h.validate_or("A", &satisfied(&["B", "C"])));
    }

    // --- validate_xone ---
    #[test]
    fn test_validate_xone_exactly_one() {
        let mut h = ShapeHierarchy::new();
        h.add_xone("A", mk_refs(&["B", "C", "D"]));
        assert!(h.validate_xone("A", &satisfied(&["B"])));
    }

    #[test]
    fn test_validate_xone_zero_fails() {
        let mut h = ShapeHierarchy::new();
        h.add_xone("A", mk_refs(&["B", "C"]));
        assert!(!h.validate_xone("A", &satisfied(&[])));
    }

    #[test]
    fn test_validate_xone_two_fails() {
        let mut h = ShapeHierarchy::new();
        h.add_xone("A", mk_refs(&["B", "C"]));
        assert!(!h.validate_xone("A", &satisfied(&["B", "C"])));
    }

    #[test]
    fn test_validate_xone_no_constraints_true() {
        let mut h = ShapeHierarchy::new();
        h.add_shape("A");
        assert!(h.validate_xone("A", &satisfied(&["B"])));
    }

    // --- flatten_and ---
    #[test]
    fn test_flatten_and_single_level() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B", "C"]));
        let flat = h.flatten_and("A");
        assert!(flat.contains(&"B".to_string()));
        assert!(flat.contains(&"C".to_string()));
    }

    #[test]
    fn test_flatten_and_transitive() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        h.add_and("B", mk_refs(&["C"]));
        let flat = h.flatten_and("A");
        assert!(flat.contains(&"B".to_string()));
        assert!(flat.contains(&"C".to_string()));
    }

    // --- shape_count ---
    #[test]
    fn test_shape_count_with_constraints() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        // "A" and "B" both get entries
        assert_eq!(h.shape_count(), 1); // only "A" is explicitly added
    }

    // --- unknown shape ---
    #[test]
    fn test_constraints_for_unknown_shape() {
        let h = ShapeHierarchy::new();
        assert!(h.constraints_for("Unknown").is_empty());
    }

    #[test]
    fn test_ancestors_unknown_shape() {
        let h = ShapeHierarchy::new();
        assert!(h.ancestors("NonExistent").is_empty());
    }

    // --- multiple constraints on same shape ---
    #[test]
    fn test_multiple_constraints_on_same_shape() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        h.add_or("A", mk_refs(&["C", "D"]));
        assert_eq!(h.constraints_for("A").len(), 2);
    }

    #[test]
    fn test_validate_and_and_or_both_must_pass() {
        let mut h = ShapeHierarchy::new();
        h.add_and("A", mk_refs(&["B"]));
        h.add_or("A", mk_refs(&["C", "D"]));
        // B satisfied, C satisfied → and passes, or passes
        assert!(h.validate_and("A", &satisfied(&["B", "C"])));
        assert!(h.validate_or("A", &satisfied(&["B", "C"])));
    }
}
