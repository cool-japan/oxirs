//! SHACL Advanced Features - Conditional Constraints
//!
//! Implementation of sh:if, sh:then, sh:else for conditional constraint validation.
//! Based on the W3C SHACL Advanced Features specification.

use serde::{Deserialize, Serialize};

use oxirs_core::{model::Term, Store};

use crate::{validation::ValidationEngine, Result, ShaclError, Shape, ShapeId, ValidationConfig};

/// Conditional constraint structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalConstraint {
    /// The condition shape (sh:if)
    pub if_shape: ShapeId,

    /// The consequent shape to apply if condition is satisfied (sh:then)
    pub then_shape: Option<ShapeId>,

    /// The alternative shape to apply if condition is not satisfied (sh:else)
    pub else_shape: Option<ShapeId>,
}

impl ConditionalConstraint {
    /// Create a new conditional constraint
    pub fn new(if_shape: ShapeId) -> Self {
        Self {
            if_shape,
            then_shape: None,
            else_shape: None,
        }
    }

    /// Set the then shape
    pub fn with_then(mut self, then_shape: ShapeId) -> Self {
        self.then_shape = Some(then_shape);
        self
    }

    /// Set the else shape
    pub fn with_else(mut self, else_shape: ShapeId) -> Self {
        self.else_shape = Some(else_shape);
        self
    }

    /// Check if this conditional has a then branch
    pub fn has_then(&self) -> bool {
        self.then_shape.is_some()
    }

    /// Check if this conditional has an else branch
    pub fn has_else(&self) -> bool {
        self.else_shape.is_some()
    }
}

/// Result of conditional evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionalResult {
    /// Condition was satisfied, then branch should be applied
    ThenBranch,
    /// Condition was not satisfied, else branch should be applied
    ElseBranch,
    /// No applicable branch (condition not satisfied and no else branch)
    NoBranch,
}

/// Evaluator for conditional constraints
pub struct ConditionalEvaluator {
    /// Cache of evaluated conditions
    condition_cache: std::collections::HashMap<String, bool>,
    /// Number of cache hits served from `condition_cache`
    cache_hits: usize,
}

impl ConditionalEvaluator {
    /// Create a new conditional evaluator
    pub fn new() -> Self {
        Self {
            condition_cache: std::collections::HashMap::new(),
            cache_hits: 0,
        }
    }

    /// Evaluate a conditional constraint for a focus node.
    ///
    /// This drives `sh:if` / `sh:then` / `sh:else`: the `if_shape` is validated
    /// against the focus node, and the appropriate branch is selected based on
    /// real conformance. Results are memoized per (if-shape, focus-node) pair.
    pub fn evaluate_conditional(
        &mut self,
        conditional: &ConditionalConstraint,
        focus_node: &Term,
        store: &dyn Store,
        shape_registry: &ShapeRegistry,
    ) -> Result<ConditionalResult> {
        // Check cache first
        let cache_key = self.cache_key(conditional, focus_node);
        if let Some(&cached) = self.condition_cache.get(&cache_key) {
            self.cache_hits += 1;
            return Ok(Self::branch_for(conditional, cached));
        }

        // Evaluate the condition (if shape)
        let condition_satisfied = self.evaluate_condition_shape(
            &conditional.if_shape,
            focus_node,
            store,
            shape_registry,
        )?;

        // Cache the result
        self.condition_cache.insert(cache_key, condition_satisfied);

        Ok(Self::branch_for(conditional, condition_satisfied))
    }

    /// Map a condition result to the branch that SHACL prescribes.
    ///
    /// When the condition holds, the `sh:then` branch applies. When it does not,
    /// the `sh:else` branch applies if present, otherwise no branch is taken.
    fn branch_for(
        conditional: &ConditionalConstraint,
        condition_satisfied: bool,
    ) -> ConditionalResult {
        if condition_satisfied {
            ConditionalResult::ThenBranch
        } else if conditional.has_else() {
            ConditionalResult::ElseBranch
        } else {
            ConditionalResult::NoBranch
        }
    }

    /// Evaluate the `sh:if` condition shape against a focus node.
    ///
    /// Resolves the shape from the registry and validates the focus node against
    /// it using a dedicated [`ValidationEngine`] (the registry's shapes are copied
    /// into a temporary [`IndexMap`] so nested `sh:property` / `sh:node` references
    /// resolve correctly). The condition is *satisfied* iff the produced report
    /// conforms.
    fn evaluate_condition_shape(
        &self,
        shape_id: &ShapeId,
        focus_node: &Term,
        store: &dyn Store,
        shape_registry: &ShapeRegistry,
    ) -> Result<bool> {
        let condition_shape = shape_registry.get(shape_id).ok_or_else(|| {
            ShaclError::ShapeValidation(format!("Conditional if-shape not found: {shape_id}"))
        })?;

        // Build a temporary shapes map from the whole registry so that any shapes
        // referenced by the condition shape (sh:property, sh:node, ...) resolve.
        let mut temp_shapes = indexmap::IndexMap::new();
        for shape in shape_registry.all_shapes() {
            temp_shapes.insert(shape.id.clone(), shape.clone());
        }

        let config = ValidationConfig::default();
        let mut validator = ValidationEngine::new(&temp_shapes, config);

        match validator.validate_node_against_shape(store, condition_shape, focus_node, None) {
            Ok(report) => Ok(report.conforms()),
            Err(e) => {
                tracing::warn!("Conditional if-shape validation error: {e}");
                Ok(false)
            }
        }
    }

    /// Generate a cache key for a conditional evaluation
    fn cache_key(&self, conditional: &ConditionalConstraint, focus_node: &Term) -> String {
        format!("{:?}:{:?}", conditional.if_shape, focus_node)
    }

    /// Clear the condition cache
    pub fn clear_cache(&mut self) {
        self.condition_cache.clear();
        self.cache_hits = 0;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> ConditionalCacheStats {
        ConditionalCacheStats {
            entries: self.condition_cache.len(),
            hits: self.cache_hits,
        }
    }
}

impl Default for ConditionalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics for conditional evaluation
#[derive(Debug, Clone)]
pub struct ConditionalCacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Number of cache hits
    pub hits: usize,
}

/// Registry for managing shapes (stub)
pub struct ShapeRegistry {
    shapes: std::collections::HashMap<ShapeId, Shape>,
}

impl ShapeRegistry {
    /// Create a new shape registry
    pub fn new() -> Self {
        Self {
            shapes: std::collections::HashMap::new(),
        }
    }

    /// Register a shape
    pub fn register(&mut self, shape: Shape) {
        self.shapes.insert(shape.id.clone(), shape);
    }

    /// Get a shape by ID
    pub fn get(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.shapes.get(shape_id)
    }

    /// Get all registered shapes
    pub fn all_shapes(&self) -> Vec<&Shape> {
        self.shapes.values().collect()
    }
}

impl Default for ShapeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for conditional constraints
pub struct ConditionalBuilder {
    if_shape: Option<ShapeId>,
    then_shape: Option<ShapeId>,
    else_shape: Option<ShapeId>,
}

impl ConditionalBuilder {
    /// Create a new conditional builder
    pub fn new() -> Self {
        Self {
            if_shape: None,
            then_shape: None,
            else_shape: None,
        }
    }

    /// Set the if shape
    pub fn if_shape(mut self, shape_id: ShapeId) -> Self {
        self.if_shape = Some(shape_id);
        self
    }

    /// Set the then shape
    pub fn then_shape(mut self, shape_id: ShapeId) -> Self {
        self.then_shape = Some(shape_id);
        self
    }

    /// Set the else shape
    pub fn else_shape(mut self, shape_id: ShapeId) -> Self {
        self.else_shape = Some(shape_id);
        self
    }

    /// Build the conditional constraint
    pub fn build(self) -> Result<ConditionalConstraint> {
        let if_shape = self.if_shape.ok_or_else(|| {
            ShaclError::Configuration("Conditional constraint requires an if shape".to_string())
        })?;

        Ok(ConditionalConstraint {
            if_shape,
            then_shape: self.then_shape,
            else_shape: self.else_shape,
        })
    }
}

impl Default for ConditionalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::value_constraints::ClassConstraint;
    use crate::{Constraint, ConstraintComponentId};
    use oxirs_core::{
        model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject},
        ConcreteStore,
    };

    const EX: &str = "http://example.org/";
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    fn iri(local: &str) -> NamedNode {
        NamedNode::new(format!("{EX}{local}")).expect("valid IRI")
    }

    fn term(local: &str) -> Term {
        Term::NamedNode(iri(local))
    }

    fn insert_type(store: &ConcreteStore, subject: &str, type_local: &str) {
        let quad = Quad::new(
            Subject::from(iri(subject)),
            Predicate::from(NamedNode::new(RDF_TYPE).expect("rdf:type")),
            Object::from(iri(type_local)),
            GraphName::DefaultGraph,
        );
        store.insert_quad(quad).expect("insert type triple");
    }

    fn insert_name(store: &ConcreteStore, subject: &str, name: &str) {
        let quad = Quad::new(
            Subject::from(iri(subject)),
            Predicate::from(iri("name")),
            Object::from(Literal::new(name)),
            GraphName::DefaultGraph,
        );
        store.insert_quad(quad).expect("insert name triple");
    }

    /// Build a node shape requiring `sh:class :Person`.
    fn person_class_shape(id: &str) -> Shape {
        let mut shape = Shape::node_shape(ShapeId::new(id));
        let constraint = Constraint::Class(ClassConstraint {
            class_iri: iri("Person"),
        });
        shape.add_constraint(
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            constraint,
        );
        shape
    }

    #[test]
    fn test_if_true_selects_then_branch() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "alice", "Person");

        let mut registry = ShapeRegistry::new();
        registry.register(person_class_shape("ifPersonShape"));

        let conditional = ConditionalConstraint::new(ShapeId::new("ifPersonShape"))
            .with_then(ShapeId::new("thenShape"))
            .with_else(ShapeId::new("elseShape"));

        let mut evaluator = ConditionalEvaluator::new();
        let result = evaluator
            .evaluate_conditional(&conditional, &term("alice"), &store, &registry)
            .expect("evaluate");
        assert_eq!(
            result,
            ConditionalResult::ThenBranch,
            "if-shape conforms => then branch"
        );
    }

    #[test]
    fn test_if_false_selects_else_branch() {
        let store = ConcreteStore::new().expect("store");
        // bob has no rdf:type Person => if-shape does NOT conform.
        insert_name(&store, "bob", "Bob");

        let mut registry = ShapeRegistry::new();
        registry.register(person_class_shape("ifPersonShape"));

        let conditional = ConditionalConstraint::new(ShapeId::new("ifPersonShape"))
            .with_then(ShapeId::new("thenShape"))
            .with_else(ShapeId::new("elseShape"));

        let mut evaluator = ConditionalEvaluator::new();
        let result = evaluator
            .evaluate_conditional(&conditional, &term("bob"), &store, &registry)
            .expect("evaluate");
        assert_eq!(
            result,
            ConditionalResult::ElseBranch,
            "if-shape fails => else branch"
        );
    }

    #[test]
    fn test_if_false_no_else_yields_no_branch() {
        let store = ConcreteStore::new().expect("store");
        insert_name(&store, "carol", "Carol");

        let mut registry = ShapeRegistry::new();
        registry.register(person_class_shape("ifPersonShape"));

        // No else branch configured.
        let conditional = ConditionalConstraint::new(ShapeId::new("ifPersonShape"))
            .with_then(ShapeId::new("thenShape"));

        let mut evaluator = ConditionalEvaluator::new();
        let result = evaluator
            .evaluate_conditional(&conditional, &term("carol"), &store, &registry)
            .expect("evaluate");
        assert_eq!(
            result,
            ConditionalResult::NoBranch,
            "if-shape fails and no else => no branch"
        );
    }

    #[test]
    fn test_cache_hit_increments_stats() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "alice", "Person");

        let mut registry = ShapeRegistry::new();
        registry.register(person_class_shape("ifPersonShape"));

        let conditional = ConditionalConstraint::new(ShapeId::new("ifPersonShape"))
            .with_then(ShapeId::new("thenShape"));

        let mut evaluator = ConditionalEvaluator::new();

        // First evaluation: cache miss, populates the cache.
        let _ = evaluator
            .evaluate_conditional(&conditional, &term("alice"), &store, &registry)
            .expect("evaluate");
        assert_eq!(evaluator.cache_stats().hits, 0, "first call is a miss");
        assert_eq!(evaluator.cache_stats().entries, 1);

        // Second identical evaluation: served from cache => hit count increments.
        let result = evaluator
            .evaluate_conditional(&conditional, &term("alice"), &store, &registry)
            .expect("evaluate");
        assert_eq!(result, ConditionalResult::ThenBranch);
        assert_eq!(
            evaluator.cache_stats().hits,
            1,
            "second identical call must be a cache hit"
        );

        // A third identical call increments again.
        let _ = evaluator
            .evaluate_conditional(&conditional, &term("alice"), &store, &registry)
            .expect("evaluate");
        assert_eq!(evaluator.cache_stats().hits, 2);
    }

    #[test]
    fn test_missing_if_shape_errors() {
        let store = ConcreteStore::new().expect("store");
        let registry = ShapeRegistry::new(); // empty: if-shape unresolvable

        let conditional = ConditionalConstraint::new(ShapeId::new("missingShape"))
            .with_then(ShapeId::new("thenShape"));

        let mut evaluator = ConditionalEvaluator::new();
        let result = evaluator.evaluate_conditional(&conditional, &term("x"), &store, &registry);
        assert!(result.is_err(), "unresolvable if-shape must error");
    }

    #[test]
    fn test_conditional_constraint_creation() {
        let conditional = ConditionalConstraint::new(ShapeId::new("ifShape"))
            .with_then(ShapeId::new("thenShape"))
            .with_else(ShapeId::new("elseShape"));

        assert_eq!(conditional.if_shape.as_str(), "ifShape");
        assert!(conditional.has_then());
        assert!(conditional.has_else());
    }

    #[test]
    fn test_conditional_builder() {
        let conditional = ConditionalBuilder::new()
            .if_shape(ShapeId::new("condition"))
            .then_shape(ShapeId::new("consequent"))
            .build()
            .expect("operation should succeed");

        assert_eq!(conditional.if_shape.as_str(), "condition");
        assert!(conditional.has_then());
        assert!(!conditional.has_else());
    }

    #[test]
    fn test_conditional_builder_missing_if() {
        let result = ConditionalBuilder::new()
            .then_shape(ShapeId::new("thenShape"))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_conditional_evaluator() {
        let evaluator = ConditionalEvaluator::new();
        let stats = evaluator.cache_stats();

        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_shape_registry() {
        let mut registry = ShapeRegistry::new();
        let shape = Shape::node_shape(ShapeId::new("testShape"));

        registry.register(shape);
        assert!(registry.get(&ShapeId::new("testShape")).is_some());
        assert_eq!(registry.all_shapes().len(), 1);
    }
}
