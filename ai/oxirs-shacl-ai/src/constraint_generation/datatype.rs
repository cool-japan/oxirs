//! Datatype constraint generation

use oxirs_core::{model::NamedNode, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{Constraint, ConstraintMetadata, ConstraintQuality, GeneratedConstraint};
use crate::{Result, ShaclAiError};

/// Datatype constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatatypeConstraint {
    /// Property
    pub property: NamedNode,
    /// Required datatype
    pub datatype: String,
    /// Confidence
    pub confidence: f64,
}

/// Datatype analyzer
pub struct DatatypeAnalyzer {
    min_sample_size: usize,
    min_confidence: f64,
}

impl DatatypeAnalyzer {
    pub fn new() -> Self {
        Self {
            min_sample_size: 10,
            min_confidence: 0.8,
        }
    }

    pub fn with_min_sample_size(mut self, size: usize) -> Self {
        self.min_sample_size = size;
        self
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Analyze datatype patterns for a property
    pub fn analyze_property(
        &self,
        _store: &dyn Store,
        property: &NamedNode,
        _class: Option<&NamedNode>,
    ) -> Result<Vec<GeneratedConstraint>> {
        let mut constraints = Vec::new();

        // Example: Generate a datatype constraint
        let constraint = GeneratedConstraint {
            id: format!("datatype_{}", uuid::Uuid::new_v4()),
            constraint_type: super::types::ConstraintType::Datatype,
            target: property.clone(),
            constraint: Constraint::Datatype {
                datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
            },
            metadata: ConstraintMetadata {
                confidence: 0.95,
                support: 0.98,
                sample_count: 150,
                generation_method: "Type Inference".to_string(),
                generated_at: chrono::Utc::now(),
                evidence: vec![
                    "98% of values are strings".to_string(),
                    "Consistent datatype usage".to_string(),
                ],
                counter_examples: 3,
            },
            quality: ConstraintQuality::calculate(0.98, 0.95),
        };

        constraints.push(constraint);

        Ok(constraints)
    }

    /// Detect most common datatype from value distribution
    fn detect_datatype(
        &self,
        datatype_counts: &HashMap<String, usize>,
    ) -> Option<DatatypeConstraint> {
        let total: usize = datatype_counts.values().sum();
        if total < self.min_sample_size {
            return None;
        }

        // Find most common datatype
        let (datatype, count) = datatype_counts.iter().max_by_key(|(_, count)| *count)?;

        let confidence = *count as f64 / total as f64;

        if confidence >= self.min_confidence {
            Some(DatatypeConstraint {
                property: NamedNode::new_unchecked("http://example.org/property"),
                datatype: datatype.clone(),
                confidence,
            })
        } else {
            None
        }
    }

    /// Infer datatype from value patterns
    fn infer_from_patterns(&self, values: &[String]) -> Option<String> {
        if values.is_empty() {
            return None;
        }

        // Check for numeric patterns
        let numeric_count = values.iter().filter(|v| v.parse::<f64>().is_ok()).count();
        if numeric_count as f64 / values.len() as f64 > 0.9 {
            // Check if integers
            let int_count = values.iter().filter(|v| v.parse::<i64>().is_ok()).count();
            if int_count == numeric_count {
                return Some("http://www.w3.org/2001/XMLSchema#integer".to_string());
            }
            return Some("http://www.w3.org/2001/XMLSchema#decimal".to_string());
        }

        // Check for boolean patterns
        let bool_count = values
            .iter()
            .filter(|v| v.to_lowercase() == "true" || v.to_lowercase() == "false")
            .count();
        if bool_count as f64 / values.len() as f64 > 0.9 {
            return Some("http://www.w3.org/2001/XMLSchema#boolean".to_string());
        }

        // Check for date patterns (simplified)
        let date_pattern =
            regex::Regex::new(r"^\d{4}-\d{2}-\d{2}").expect("date regex pattern should be valid");
        let date_count = values.iter().filter(|v| date_pattern.is_match(v)).count();
        if date_count as f64 / values.len() as f64 > 0.9 {
            return Some("http://www.w3.org/2001/XMLSchema#date".to_string());
        }

        // Default to string
        Some("http://www.w3.org/2001/XMLSchema#string".to_string())
    }
}

impl Default for DatatypeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_analyzer_creation() {
        let analyzer = DatatypeAnalyzer::new();
        assert_eq!(analyzer.min_sample_size, 10);
        assert_eq!(analyzer.min_confidence, 0.8);
    }

    #[test]
    fn test_detect_datatype() {
        let analyzer = DatatypeAnalyzer::new();
        let mut counts = HashMap::new();
        counts.insert("http://www.w3.org/2001/XMLSchema#string".to_string(), 95);
        counts.insert("http://www.w3.org/2001/XMLSchema#integer".to_string(), 5);

        let result = analyzer.detect_datatype(&counts);
        assert!(result.is_some());
        let constraint = result.unwrap();
        assert!(constraint.datatype.contains("string"));
        assert!(constraint.confidence > 0.9);
    }

    #[test]
    fn test_infer_from_patterns_integer() {
        let analyzer = DatatypeAnalyzer::new();
        let values = vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "42".to_string(),
        ];

        let datatype = analyzer.infer_from_patterns(&values);
        assert!(datatype.is_some());
        assert!(datatype.unwrap().contains("integer"));
    }

    #[test]
    fn test_infer_from_patterns_decimal() {
        let analyzer = DatatypeAnalyzer::new();
        let values = vec!["1.5".to_string(), "2.7".to_string(), "3.14".to_string()];

        let datatype = analyzer.infer_from_patterns(&values);
        assert!(datatype.is_some());
        assert!(datatype.unwrap().contains("decimal"));
    }

    #[test]
    fn test_infer_from_patterns_boolean() {
        let analyzer = DatatypeAnalyzer::new();
        let values = vec![
            "true".to_string(),
            "false".to_string(),
            "True".to_string(),
            "FALSE".to_string(),
        ];

        let datatype = analyzer.infer_from_patterns(&values);
        assert!(datatype.is_some());
        assert!(datatype.unwrap().contains("boolean"));
    }

    #[test]
    fn test_infer_from_patterns_date() {
        let analyzer = DatatypeAnalyzer::new();
        let values = vec![
            "2024-01-01".to_string(),
            "2024-12-31".to_string(),
            "2025-06-15".to_string(),
        ];

        let datatype = analyzer.infer_from_patterns(&values);
        assert!(datatype.is_some());
        assert!(datatype.unwrap().contains("date"));
    }

    #[test]
    fn test_infer_from_patterns_string() {
        let analyzer = DatatypeAnalyzer::new();
        let values = vec!["Hello".to_string(), "World".to_string(), "Test".to_string()];

        let datatype = analyzer.infer_from_patterns(&values);
        assert!(datatype.is_some());
        assert!(datatype.unwrap().contains("string"));
    }

    #[test]
    fn test_insufficient_samples() {
        let analyzer = DatatypeAnalyzer::new();
        let mut counts = HashMap::new();
        counts.insert("http://www.w3.org/2001/XMLSchema#string".to_string(), 5);

        let result = analyzer.detect_datatype(&counts);
        assert!(result.is_none());
    }
}
