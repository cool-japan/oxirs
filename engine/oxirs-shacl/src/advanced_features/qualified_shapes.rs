//! SHACL Advanced Features - Qualified Value Shapes
//!
//! Implementation of qualified value shape constraints including:
//! - sh:qualifiedValueShape - constrains values based on their conformance to a shape
//! - sh:qualifiedMinCount - minimum number of values conforming to the qualified shape
//! - sh:qualifiedMaxCount - maximum number of values conforming to the qualified shape
//! - sh:qualifiedValueShapesDisjoint - ensures qualified shapes are disjoint
//!
//! Based on the W3C SHACL specification for qualified value shapes.

use crate::{PropertyPath, Result, ShaclError, Shape, ShapeId};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};

/// Qualified value shape constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifiedValueShapeConstraint {
    /// The shape that values must conform to
    pub qualified_shape: QualifiedShape,
    /// Minimum count of values that must conform (inclusive)
    pub min_count: Option<usize>,
    /// Maximum count of values that may conform (inclusive)
    pub max_count: Option<usize>,
    /// Whether qualified shapes must be disjoint
    pub disjoint: bool,
}

/// A qualified shape reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualifiedShape {
    /// Reference to a named shape by ID
    ShapeRef(ShapeId),
    /// Inline shape definition
    InlineShape(Box<Shape>),
}

impl PartialEq for QualifiedShape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QualifiedShape::ShapeRef(id1), QualifiedShape::ShapeRef(id2)) => id1 == id2,
            (QualifiedShape::InlineShape(s1), QualifiedShape::InlineShape(s2)) => {
                // Compare shape IDs for inline shapes
                s1.id == s2.id
            }
            _ => false,
        }
    }
}

impl Eq for QualifiedShape {}

impl QualifiedValueShapeConstraint {
    /// Create a new qualified value shape constraint
    pub fn new(qualified_shape: QualifiedShape) -> Self {
        Self {
            qualified_shape,
            min_count: None,
            max_count: None,
            disjoint: false,
        }
    }

    /// Set minimum count
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.min_count = Some(min_count);
        self
    }

    /// Set maximum count
    pub fn with_max_count(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    /// Set disjoint flag
    pub fn with_disjoint(mut self, disjoint: bool) -> Self {
        self.disjoint = disjoint;
        self
    }

    /// Validate this constraint
    pub fn validate(
        &self,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<QualifiedValidationResult> {
        // Get the qualified shape
        let shape = match &self.qualified_shape {
            QualifiedShape::ShapeRef(shape_id) => shape_registry
                .get_shape(shape_id)
                .ok_or_else(|| {
                    ShaclError::ShapeValidation(format!("Qualified shape not found: {}", shape_id))
                })?
                .clone(),
            QualifiedShape::InlineShape(shape) => (**shape).clone(),
        };

        // Count how many values conform to the qualified shape
        let mut conforming_values = Vec::new();
        let mut non_conforming_values = Vec::new();

        for value in values {
            // TODO: Validate value against shape
            // For now, we'll use a simple placeholder
            let conforms = self.value_conforms_to_shape(value, &shape, store)?;

            if conforms {
                conforming_values.push(value.clone());
            } else {
                non_conforming_values.push(value.clone());
            }
        }

        let conforming_count = conforming_values.len();

        // Check minimum count constraint
        if let Some(min_count) = self.min_count {
            if conforming_count < min_count {
                return Ok(QualifiedValidationResult::violation(
                    format!(
                        "Qualified value shape requires at least {} conforming values, found {}",
                        min_count, conforming_count
                    ),
                    conforming_values,
                    non_conforming_values,
                ));
            }
        }

        // Check maximum count constraint
        if let Some(max_count) = self.max_count {
            if conforming_count > max_count {
                return Ok(QualifiedValidationResult::violation(
                    format!(
                        "Qualified value shape allows at most {} conforming values, found {}",
                        max_count, conforming_count
                    ),
                    conforming_values,
                    non_conforming_values,
                ));
            }
        }

        Ok(QualifiedValidationResult::success(
            conforming_values,
            non_conforming_values,
        ))
    }

    /// Check if a value conforms to a shape (placeholder)
    fn value_conforms_to_shape(
        &self,
        value: &Term,
        shape: &Shape,
        _store: &dyn Store,
    ) -> Result<bool> {
        // TODO: Implement full shape validation
        // This requires integration with the validation engine
        tracing::debug!("Checking conformance of {:?} to shape {}", value, shape.id);
        Ok(true) // Placeholder
    }
}

/// Result of qualified value shape validation
#[derive(Debug, Clone)]
pub struct QualifiedValidationResult {
    /// Whether validation passed
    pub conforms: bool,
    /// Error message if validation failed
    pub message: Option<String>,
    /// Values that conformed to the qualified shape
    pub conforming_values: Vec<Term>,
    /// Values that did not conform
    pub non_conforming_values: Vec<Term>,
}

impl QualifiedValidationResult {
    /// Create a success result
    pub fn success(conforming_values: Vec<Term>, non_conforming_values: Vec<Term>) -> Self {
        Self {
            conforms: true,
            message: None,
            conforming_values,
            non_conforming_values,
        }
    }

    /// Create a violation result
    pub fn violation(
        message: String,
        conforming_values: Vec<Term>,
        non_conforming_values: Vec<Term>,
    ) -> Self {
        Self {
            conforms: false,
            message: Some(message),
            conforming_values,
            non_conforming_values,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.conforms
    }

    /// Get conforming count
    pub fn conforming_count(&self) -> usize {
        self.conforming_values.len()
    }
}

/// Trait for accessing shapes in qualified value shape validation
pub trait ShapeRegistry {
    /// Get a shape by ID
    fn get_shape(&self, id: &ShapeId) -> Option<&Shape>;

    /// Check if shapes are disjoint
    fn are_shapes_disjoint(&self, shape1: &ShapeId, shape2: &ShapeId) -> Result<bool>;
}

/// Qualified shapes validator with disjointness checking
pub struct QualifiedShapesValidator {
    /// Cache for disjointness checks
    disjointness_cache: std::collections::HashMap<(ShapeId, ShapeId), bool>,
}

impl QualifiedShapesValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            disjointness_cache: std::collections::HashMap::new(),
        }
    }

    /// Validate multiple qualified value shapes with disjointness checking
    pub fn validate_with_disjointness(
        &mut self,
        constraints: &[QualifiedValueShapeConstraint],
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<Vec<QualifiedValidationResult>> {
        // First, check if any constraints require disjoint shapes
        let has_disjoint_requirement = constraints.iter().any(|c| c.disjoint);

        if has_disjoint_requirement {
            // Verify that all qualified shapes are pairwise disjoint
            self.verify_disjointness(constraints, shape_registry)?;
        }

        // Validate each constraint
        let mut results = Vec::new();
        for constraint in constraints {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Verify that all qualified shapes in the constraints are pairwise disjoint
    fn verify_disjointness(
        &mut self,
        constraints: &[QualifiedValueShapeConstraint],
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<()> {
        // Get all shape IDs
        let shape_ids: Vec<ShapeId> = constraints
            .iter()
            .filter_map(|c| match &c.qualified_shape {
                QualifiedShape::ShapeRef(id) => Some(id.clone()),
                QualifiedShape::InlineShape(_) => None, // Skip inline shapes for now
            })
            .collect();

        // Check pairwise disjointness
        for i in 0..shape_ids.len() {
            for j in (i + 1)..shape_ids.len() {
                let shape1 = &shape_ids[i];
                let shape2 = &shape_ids[j];

                // Check cache
                let cache_key = if shape1 < shape2 {
                    (shape1.clone(), shape2.clone())
                } else {
                    (shape2.clone(), shape1.clone())
                };

                let disjoint = if let Some(&cached) = self.disjointness_cache.get(&cache_key) {
                    cached
                } else {
                    let disjoint = shape_registry.are_shapes_disjoint(shape1, shape2)?;
                    self.disjointness_cache.insert(cache_key, disjoint);
                    disjoint
                };

                if !disjoint {
                    return Err(ShaclError::ConstraintValidation(format!(
                        "Qualified shapes {} and {} are not disjoint",
                        shape1, shape2
                    )));
                }
            }
        }

        Ok(())
    }

    /// Clear the disjointness cache
    pub fn clear_cache(&mut self) {
        self.disjointness_cache.clear();
    }
}

impl Default for QualifiedShapesValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex qualified value shapes with multiple conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexQualifiedConstraint {
    /// All of these qualified shapes must be satisfied (AND)
    pub all_of: Vec<QualifiedValueShapeConstraint>,
    /// At least one of these qualified shapes must be satisfied (OR)
    pub any_of: Vec<QualifiedValueShapeConstraint>,
    /// None of these qualified shapes should be satisfied (NOT)
    pub none_of: Vec<QualifiedValueShapeConstraint>,
    /// Exactly one of these qualified shapes must be satisfied (XOR)
    pub one_of: Vec<QualifiedValueShapeConstraint>,
}

impl ComplexQualifiedConstraint {
    /// Create a new complex constraint
    pub fn new() -> Self {
        Self {
            all_of: Vec::new(),
            any_of: Vec::new(),
            none_of: Vec::new(),
            one_of: Vec::new(),
        }
    }

    /// Add an ALL OF constraint
    pub fn add_all_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.all_of.push(constraint);
        self
    }

    /// Add an ANY OF constraint
    pub fn add_any_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.any_of.push(constraint);
        self
    }

    /// Add a NONE OF constraint
    pub fn add_none_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.none_of.push(constraint);
        self
    }

    /// Add a ONE OF constraint
    pub fn add_one_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.one_of.push(constraint);
        self
    }

    /// Validate this complex constraint
    pub fn validate(
        &self,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<ComplexValidationResult> {
        let mut errors = Vec::new();

        // Validate ALL OF constraints
        for constraint in &self.all_of {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            if !result.is_valid() {
                errors.push(format!(
                    "ALL OF constraint failed: {}",
                    result.message.unwrap_or_default()
                ));
            }
        }

        // Validate ANY OF constraints (at least one must pass)
        if !self.any_of.is_empty() {
            let any_passed = self.any_of.iter().any(|c| {
                c.validate(focus_node, path, values, store, shape_registry)
                    .ok()
                    .and_then(|r| if r.is_valid() { Some(()) } else { None })
                    .is_some()
            });

            if !any_passed {
                errors.push("None of the ANY OF constraints were satisfied".to_string());
            }
        }

        // Validate NONE OF constraints (all must fail)
        for constraint in &self.none_of {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            if result.is_valid() {
                errors.push("NONE OF constraint unexpectedly passed".to_string());
            }
        }

        // Validate ONE OF constraints (exactly one must pass)
        if !self.one_of.is_empty() {
            let passed_count = self
                .one_of
                .iter()
                .filter(|c| {
                    c.validate(focus_node, path, values, store, shape_registry)
                        .ok()
                        .and_then(|r| if r.is_valid() { Some(()) } else { None })
                        .is_some()
                })
                .count();

            if passed_count != 1 {
                errors.push(format!(
                    "ONE OF constraint requires exactly 1 match, found {}",
                    passed_count
                ));
            }
        }

        if errors.is_empty() {
            Ok(ComplexValidationResult::success())
        } else {
            Ok(ComplexValidationResult::violation(errors))
        }
    }
}

impl Default for ComplexQualifiedConstraint {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of complex qualified constraint validation
#[derive(Debug, Clone)]
pub struct ComplexValidationResult {
    /// Whether validation passed
    pub conforms: bool,
    /// Error messages if validation failed
    pub errors: Vec<String>,
}

impl ComplexValidationResult {
    /// Create a success result
    pub fn success() -> Self {
        Self {
            conforms: true,
            errors: Vec::new(),
        }
    }

    /// Create a violation result
    pub fn violation(errors: Vec<String>) -> Self {
        Self {
            conforms: false,
            errors,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.conforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualified_constraint_creation() {
        let shape_id = ShapeId::new("test:shape");
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(shape_id))
            .with_min_count(1)
            .with_max_count(5);

        assert_eq!(constraint.min_count, Some(1));
        assert_eq!(constraint.max_count, Some(5));
    }

    #[test]
    fn test_complex_constraint_creation() {
        let shape_id = ShapeId::new("test:shape");
        let qualified = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(shape_id));

        let complex = ComplexQualifiedConstraint::new()
            .add_all_of(qualified.clone())
            .add_any_of(qualified);

        assert_eq!(complex.all_of.len(), 1);
        assert_eq!(complex.any_of.len(), 1);
    }

    #[test]
    fn test_validation_result() {
        let result = QualifiedValidationResult::success(vec![], vec![]);
        assert!(result.is_valid());
        assert_eq!(result.conforming_count(), 0);
    }

    #[test]
    fn test_complex_validation_result() {
        let result = ComplexValidationResult::success();
        assert!(result.is_valid());
        assert!(result.errors.is_empty());

        let result = ComplexValidationResult::violation(vec!["error".to_string()]);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
    }
}
