//! Shape building and constraint utilities
//!
//! This module provides utilities for creating and manipulating SHACL shapes
//! with AI-powered enhancements.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use oxirs_core::model::NamedNode;

use crate::{Result, ShaclAiError};

// Simplified shape types for AI operations
// We'll expand this as the SHACL crate stabilizes

/// Enhanced shape wrapper with AI capabilities
#[derive(Debug, Clone)]
pub struct Shape {
    pub id: String,
    pub target_classes: Vec<String>,
    pub property_constraints: Vec<PropertyConstraint>,
    pub confidence: f64,
    pub ai_generated: bool,
    pub metrics: ShapeMetrics,
}

impl Shape {
    /// Create a new shape with the given IRI
    pub fn new(iri: String) -> Self {
        Self {
            id: iri,
            target_classes: Vec::new(),
            property_constraints: Vec::new(),
            confidence: 1.0,
            ai_generated: false,
            metrics: ShapeMetrics::default(),
        }
    }

    /// Set the target class for this shape
    pub fn set_target_class(&mut self, class_iri: String) {
        self.target_classes.push(class_iri);
    }

    /// Add a property constraint to this shape
    pub fn add_property_constraint(&mut self, constraint: PropertyConstraint) {
        self.property_constraints.push(constraint);
    }

    /// Set confidence score for AI-generated shapes
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence;
    }

    /// Mark shape as AI-generated
    pub fn mark_ai_generated(&mut self) {
        self.ai_generated = true;
    }

    /// Get shape confidence
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Check if shape was AI-generated
    pub fn is_ai_generated(&self) -> bool {
        self.ai_generated
    }

    /// Get shape metrics
    pub fn metrics(&self) -> &ShapeMetrics {
        &self.metrics
    }

    /// Update shape metrics
    pub fn update_metrics(&mut self, metrics: ShapeMetrics) {
        self.metrics = metrics;
    }

    /// Get the shape ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get target classes
    pub fn target_classes(&self) -> &[String] {
        &self.target_classes
    }

    /// Get property constraints
    pub fn property_constraints(&self) -> &[PropertyConstraint] {
        &self.property_constraints
    }
}

/// Property constraint builder with AI enhancements
#[derive(Debug, Clone)]
pub struct PropertyConstraint {
    pub path: String,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
    pub datatype: Option<String>,
    pub node_kind: Option<String>,
    pub min_length: Option<u32>,
    pub max_length: Option<u32>,
    pub pattern: Option<String>,
    pub class: Option<String>,
    pub node: Option<String>,
    pub has_value: Option<String>,
    pub in_values: Vec<String>,
    pub confidence: f64,
    pub ai_generated: bool,
}

impl PropertyConstraint {
    /// Create a new property constraint
    pub fn new(path: String) -> Self {
        Self {
            path,
            min_count: None,
            max_count: None,
            datatype: None,
            node_kind: None,
            min_length: None,
            max_length: None,
            pattern: None,
            class: None,
            node: None,
            has_value: None,
            in_values: Vec::new(),
            confidence: 1.0,
            ai_generated: false,
        }
    }

    /// Set minimum cardinality
    pub fn with_min_count(mut self, min_count: u32) -> Self {
        self.min_count = Some(min_count);
        self
    }

    /// Set maximum cardinality
    pub fn with_max_count(mut self, max_count: u32) -> Self {
        self.max_count = Some(max_count);
        self
    }

    /// Set datatype constraint
    pub fn with_datatype(mut self, datatype: String) -> Self {
        self.datatype = Some(datatype);
        self
    }

    /// Set node kind constraint
    pub fn with_node_kind(mut self, node_kind: String) -> Self {
        self.node_kind = Some(node_kind);
        self
    }

    /// Set minimum string length
    pub fn with_min_length(mut self, min_length: u32) -> Self {
        self.min_length = Some(min_length);
        self
    }

    /// Set maximum string length
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set regex pattern constraint
    pub fn with_pattern(mut self, pattern: String) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set class constraint
    pub fn with_class(mut self, class: String) -> Self {
        self.class = Some(class);
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Mark as AI-generated
    pub fn mark_ai_generated(mut self) -> Self {
        self.ai_generated = true;
        self
    }

    /// Validate constraint completeness
    pub fn is_valid(&self) -> bool {
        // Basic validation - has path and at least one constraint
        !self.path.is_empty()
            && (self.min_count.is_some()
                || self.max_count.is_some()
                || self.datatype.is_some()
                || self.node_kind.is_some()
                || self.class.is_some())
    }

    /// Get constraint summary
    pub fn constraint_summary(&self) -> String {
        let mut parts = Vec::new();

        if let Some(min_count) = self.min_count {
            parts.push(format!("minCount:{}", min_count));
        }
        if let Some(max_count) = self.max_count {
            parts.push(format!("maxCount:{}", max_count));
        }
        if let Some(ref datatype) = self.datatype {
            parts.push(format!("datatype:{}", datatype));
        }
        if let Some(ref class) = self.class {
            parts.push(format!("class:{}", class));
        }

        if parts.is_empty() {
            "no constraints".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Metrics for shape quality and performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeMetrics {
    /// Number of validation runs
    pub validation_runs: usize,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Average validation time in milliseconds
    pub avg_validation_time_ms: f64,

    /// Number of violations detected
    pub violations_detected: usize,

    /// False positive rate
    pub false_positive_rate: f64,

    /// Coverage percentage
    pub coverage_percentage: f64,

    /// Precision score
    pub precision: f64,

    /// Recall score
    pub recall: f64,

    /// F1 score
    pub f1_score: f64,
}

impl ShapeMetrics {
    /// Create new metrics with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Update success rate
    pub fn update_success_rate(&mut self, rate: f64) {
        self.success_rate = rate;
    }

    /// Add validation run result
    pub fn add_validation_result(&mut self, success: bool, time_ms: f64) {
        self.validation_runs += 1;

        // Update average time
        let total_time = self.avg_validation_time_ms * (self.validation_runs - 1) as f64 + time_ms;
        self.avg_validation_time_ms = total_time / self.validation_runs as f64;

        // Update success rate
        let total_successes = (self.success_rate * (self.validation_runs - 1) as f64)
            + if success { 1.0 } else { 0.0 };
        self.success_rate = total_successes / self.validation_runs as f64;
    }

    /// Calculate F1 score from precision and recall
    pub fn calculate_f1_score(&mut self) {
        if self.precision + self.recall > 0.0 {
            self.f1_score = 2.0 * (self.precision * self.recall) / (self.precision + self.recall);
        } else {
            self.f1_score = 0.0;
        }
    }
}

/// Shape builder for fluent API
#[derive(Debug)]
pub struct ShapeBuilder {
    shape: Shape,
}

impl ShapeBuilder {
    /// Create a new shape builder
    pub fn new(iri: String) -> Self {
        Self {
            shape: Shape::new(iri),
        }
    }

    /// Set target class
    pub fn target_class(mut self, class_iri: String) -> Self {
        self.shape.set_target_class(class_iri);
        self
    }

    /// Add property constraint
    pub fn property(mut self, constraint: PropertyConstraint) -> Self {
        self.shape.add_property_constraint(constraint);
        self
    }

    /// Set confidence
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.shape.set_confidence(confidence);
        self
    }

    /// Mark as AI-generated
    pub fn ai_generated(mut self) -> Self {
        self.shape.mark_ai_generated();
        self
    }

    /// Build the shape
    pub fn build(self) -> Shape {
        self.shape
    }
}

/// Utility functions for shape operations
pub mod utils {
    use super::*;

    /// Create a simple property constraint with cardinality
    pub fn simple_property(
        path: String,
        min_count: Option<u32>,
        max_count: Option<u32>,
    ) -> PropertyConstraint {
        let mut constraint = PropertyConstraint::new(path);
        constraint.min_count = min_count;
        constraint.max_count = max_count;
        constraint
    }

    /// Create a datatype property constraint
    pub fn datatype_property(
        path: String,
        datatype: String,
        min_count: Option<u32>,
    ) -> PropertyConstraint {
        PropertyConstraint::new(path)
            .with_datatype(datatype)
            .with_min_count(min_count.unwrap_or(0))
    }

    /// Create a class property constraint
    pub fn class_property(
        path: String,
        class: String,
        min_count: Option<u32>,
    ) -> PropertyConstraint {
        PropertyConstraint::new(path)
            .with_class(class)
            .with_min_count(min_count.unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let mut shape = Shape::new("http://example.org/TestShape".to_string());
        shape.set_target_class("http://example.org/TestClass".to_string());

        assert!(!shape.is_ai_generated());
        assert_eq!(shape.confidence(), 1.0);
    }

    #[test]
    fn test_property_constraint() {
        let constraint = PropertyConstraint::new("http://example.org/name".to_string())
            .with_min_count(1)
            .with_max_count(1)
            .with_datatype("http://www.w3.org/2001/XMLSchema#string".to_string());

        assert_eq!(constraint.min_count, Some(1));
        assert_eq!(constraint.max_count, Some(1));
        assert_eq!(
            constraint.datatype,
            Some("http://www.w3.org/2001/XMLSchema#string".to_string())
        );
    }

    #[test]
    fn test_shape_builder() {
        let shape = ShapeBuilder::new("http://example.org/PersonShape".to_string())
            .target_class("http://example.org/Person".to_string())
            .property(
                PropertyConstraint::new("http://example.org/name".to_string())
                    .with_min_count(1)
                    .with_datatype("http://www.w3.org/2001/XMLSchema#string".to_string()),
            )
            .confidence(0.95)
            .ai_generated()
            .build();

        assert!(shape.is_ai_generated());
        assert_eq!(shape.confidence(), 0.95);
    }

    #[test]
    fn test_shape_metrics() {
        let mut metrics = ShapeMetrics::new();

        metrics.add_validation_result(true, 100.0);
        metrics.add_validation_result(false, 150.0);

        assert_eq!(metrics.validation_runs, 2);
        assert_eq!(metrics.success_rate, 0.5);
        assert_eq!(metrics.avg_validation_time_ms, 125.0);
    }
}
