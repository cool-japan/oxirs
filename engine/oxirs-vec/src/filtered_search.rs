//! Filtered search capabilities for vector indices
//!
//! This module provides advanced filtering capabilities for vector search,
//! allowing searches to be constrained by metadata predicates, value ranges,
//! and complex logical conditions.

use crate::{Vector, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata filter for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// Exact match on a metadata field
    Equals { field: String, value: FilterValue },
    /// Field value is not equal to the given value
    NotEquals { field: String, value: FilterValue },
    /// Field value is greater than the given value
    GreaterThan { field: String, value: FilterValue },
    /// Field value is greater than or equal to the given value
    GreaterThanOrEqual { field: String, value: FilterValue },
    /// Field value is less than the given value
    LessThan { field: String, value: FilterValue },
    /// Field value is less than or equal to the given value
    LessThanOrEqual { field: String, value: FilterValue },
    /// Field value is in the given set
    In {
        field: String,
        values: Vec<FilterValue>,
    },
    /// Field value is not in the given set
    NotIn {
        field: String,
        values: Vec<FilterValue>,
    },
    /// Field value contains the given substring
    Contains { field: String, substring: String },
    /// Field value matches the given regex pattern
    Regex { field: String, pattern: String },
    /// Field exists (has any value)
    Exists { field: String },
    /// Field does not exist or is null
    NotExists { field: String },
    /// Logical AND of multiple filters
    And(Vec<MetadataFilter>),
    /// Logical OR of multiple filters
    Or(Vec<MetadataFilter>),
    /// Logical NOT of a filter
    Not(Box<MetadataFilter>),
}

/// Value type for filter predicates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
}

impl FilterValue {
    /// Compare two filter values
    fn compare(&self, other: &FilterValue) -> std::cmp::Ordering {
        match (self, other) {
            (FilterValue::String(a), FilterValue::String(b)) => a.cmp(b),
            (FilterValue::Integer(a), FilterValue::Integer(b)) => a.cmp(b),
            (FilterValue::Float(a), FilterValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (FilterValue::Boolean(a), FilterValue::Boolean(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl MetadataFilter {
    /// Evaluate the filter against a metadata map
    pub fn evaluate(&self, metadata: &HashMap<String, String>) -> bool {
        match self {
            MetadataFilter::Equals { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    &parsed_value == value
                } else {
                    false
                }
            }
            MetadataFilter::NotEquals { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    &parsed_value != value
                } else {
                    true
                }
            }
            MetadataFilter::GreaterThan { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    parsed_value.compare(value) == std::cmp::Ordering::Greater
                } else {
                    false
                }
            }
            MetadataFilter::GreaterThanOrEqual { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    matches!(
                        parsed_value.compare(value),
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                    )
                } else {
                    false
                }
            }
            MetadataFilter::LessThan { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    parsed_value.compare(value) == std::cmp::Ordering::Less
                } else {
                    false
                }
            }
            MetadataFilter::LessThanOrEqual { field, value } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    matches!(
                        parsed_value.compare(value),
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                    )
                } else {
                    false
                }
            }
            MetadataFilter::In { field, values } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    values.contains(&parsed_value)
                } else {
                    false
                }
            }
            MetadataFilter::NotIn { field, values } => {
                if let Some(field_value) = metadata.get(field) {
                    let parsed_value = Self::parse_value(field_value);
                    !values.contains(&parsed_value)
                } else {
                    true
                }
            }
            MetadataFilter::Contains { field, substring } => {
                if let Some(field_value) = metadata.get(field) {
                    field_value.contains(substring)
                } else {
                    false
                }
            }
            MetadataFilter::Regex { field, pattern } => {
                if let Some(field_value) = metadata.get(field) {
                    if let Ok(regex) = regex::Regex::new(pattern) {
                        regex.is_match(field_value)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataFilter::Exists { field } => metadata.contains_key(field),
            MetadataFilter::NotExists { field } => !metadata.contains_key(field),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.evaluate(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.evaluate(metadata)),
            MetadataFilter::Not(filter) => !filter.evaluate(metadata),
        }
    }

    /// Parse a string value into a FilterValue
    fn parse_value(s: &str) -> FilterValue {
        // Try to parse as integer
        if let Ok(i) = s.parse::<i64>() {
            return FilterValue::Integer(i);
        }

        // Try to parse as float
        if let Ok(f) = s.parse::<f64>() {
            return FilterValue::Float(f);
        }

        // Try to parse as boolean
        if let Ok(b) = s.parse::<bool>() {
            return FilterValue::Boolean(b);
        }

        // Check for null
        if s == "null" || s.is_empty() {
            return FilterValue::Null;
        }

        // Default to string
        FilterValue::String(s.to_string())
    }
}

/// Search filter combining distance and metadata constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Maximum distance threshold
    pub max_distance: Option<f32>,
    /// Minimum distance threshold
    pub min_distance: Option<f32>,
    /// Metadata filter predicates
    pub metadata_filter: Option<MetadataFilter>,
    /// Vector dimension constraints
    pub dimension_constraints: Option<Vec<DimensionConstraint>>,
}

/// Constraint on specific vector dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionConstraint {
    /// Dimension index
    pub dimension: usize,
    /// Minimum value for this dimension
    pub min_value: Option<f32>,
    /// Maximum value for this dimension
    pub max_value: Option<f32>,
}

impl DimensionConstraint {
    /// Check if a vector satisfies this dimension constraint
    pub fn satisfies(&self, vector: &Vector) -> bool {
        let values = vector.as_f32();

        if self.dimension >= values.len() {
            return false;
        }

        let value = values[self.dimension];

        if let Some(min) = self.min_value {
            if value < min {
                return false;
            }
        }

        if let Some(max) = self.max_value {
            if value > max {
                return false;
            }
        }

        true
    }
}

impl SearchFilter {
    /// Create a new empty search filter
    pub fn new() -> Self {
        Self {
            max_distance: None,
            min_distance: None,
            metadata_filter: None,
            dimension_constraints: None,
        }
    }

    /// Set maximum distance threshold
    pub fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = Some(max_distance);
        self
    }

    /// Set minimum distance threshold
    pub fn with_min_distance(mut self, min_distance: f32) -> Self {
        self.min_distance = Some(min_distance);
        self
    }

    /// Set metadata filter
    pub fn with_metadata_filter(mut self, filter: MetadataFilter) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Set dimension constraints
    pub fn with_dimension_constraints(mut self, constraints: Vec<DimensionConstraint>) -> Self {
        self.dimension_constraints = Some(constraints);
        self
    }

    /// Check if a search result satisfies this filter
    pub fn satisfies(
        &self,
        distance: f32,
        vector: &Vector,
        metadata: &HashMap<String, String>,
    ) -> bool {
        // Check distance constraints
        if let Some(max) = self.max_distance {
            if distance > max {
                return false;
            }
        }

        if let Some(min) = self.min_distance {
            if distance < min {
                return false;
            }
        }

        // Check metadata filter
        if let Some(ref filter) = self.metadata_filter {
            if !filter.evaluate(metadata) {
                return false;
            }
        }

        // Check dimension constraints
        if let Some(ref constraints) = self.dimension_constraints {
            for constraint in constraints {
                if !constraint.satisfies(vector) {
                    return false;
                }
            }
        }

        true
    }

    /// Filter a list of search results
    pub fn filter_results(
        &self,
        results: Vec<(VectorId, f32, Vector, HashMap<String, String>)>,
    ) -> Vec<(VectorId, f32)> {
        results
            .into_iter()
            .filter(|(_, distance, vector, metadata)| self.satisfies(*distance, vector, metadata))
            .map(|(id, distance, _, _)| (id, distance))
            .collect()
    }
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for complex filter expressions
pub struct FilterBuilder {
    filters: Vec<MetadataFilter>,
}

impl FilterBuilder {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn equals(mut self, field: impl Into<String>, value: FilterValue) -> Self {
        self.filters.push(MetadataFilter::Equals {
            field: field.into(),
            value,
        });
        self
    }

    pub fn not_equals(mut self, field: impl Into<String>, value: FilterValue) -> Self {
        self.filters.push(MetadataFilter::NotEquals {
            field: field.into(),
            value,
        });
        self
    }

    pub fn greater_than(mut self, field: impl Into<String>, value: FilterValue) -> Self {
        self.filters.push(MetadataFilter::GreaterThan {
            field: field.into(),
            value,
        });
        self
    }

    pub fn less_than(mut self, field: impl Into<String>, value: FilterValue) -> Self {
        self.filters.push(MetadataFilter::LessThan {
            field: field.into(),
            value,
        });
        self
    }

    pub fn contains(mut self, field: impl Into<String>, substring: impl Into<String>) -> Self {
        self.filters.push(MetadataFilter::Contains {
            field: field.into(),
            substring: substring.into(),
        });
        self
    }

    pub fn regex(mut self, field: impl Into<String>, pattern: impl Into<String>) -> Self {
        self.filters.push(MetadataFilter::Regex {
            field: field.into(),
            pattern: pattern.into(),
        });
        self
    }

    pub fn exists(mut self, field: impl Into<String>) -> Self {
        self.filters.push(MetadataFilter::Exists {
            field: field.into(),
        });
        self
    }

    pub fn build_and(self) -> MetadataFilter {
        if self.filters.len() == 1 {
            self.filters.into_iter().next().unwrap()
        } else {
            MetadataFilter::And(self.filters)
        }
    }

    pub fn build_or(self) -> MetadataFilter {
        if self.filters.len() == 1 {
            self.filters.into_iter().next().unwrap()
        } else {
            MetadataFilter::Or(self.filters)
        }
    }
}

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equals_filter() {
        let filter = MetadataFilter::Equals {
            field: "category".to_string(),
            value: FilterValue::String("news".to_string()),
        };

        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "news".to_string());

        assert!(filter.evaluate(&metadata));

        metadata.insert("category".to_string(), "sports".to_string());
        assert!(!filter.evaluate(&metadata));
    }

    #[test]
    fn test_greater_than_filter() {
        let filter = MetadataFilter::GreaterThan {
            field: "score".to_string(),
            value: FilterValue::Integer(50),
        };

        let mut metadata = HashMap::new();
        metadata.insert("score".to_string(), "75".to_string());
        assert!(filter.evaluate(&metadata));

        metadata.insert("score".to_string(), "25".to_string());
        assert!(!filter.evaluate(&metadata));
    }

    #[test]
    fn test_and_filter() {
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Equals {
                field: "status".to_string(),
                value: FilterValue::String("active".to_string()),
            },
            MetadataFilter::GreaterThan {
                field: "priority".to_string(),
                value: FilterValue::Integer(5),
            },
        ]);

        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), "active".to_string());
        metadata.insert("priority".to_string(), "8".to_string());
        assert!(filter.evaluate(&metadata));

        metadata.insert("priority".to_string(), "3".to_string());
        assert!(!filter.evaluate(&metadata));
    }

    #[test]
    fn test_or_filter() {
        let filter = MetadataFilter::Or(vec![
            MetadataFilter::Equals {
                field: "type".to_string(),
                value: FilterValue::String("urgent".to_string()),
            },
            MetadataFilter::Equals {
                field: "type".to_string(),
                value: FilterValue::String("critical".to_string()),
            },
        ]);

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "urgent".to_string());
        assert!(filter.evaluate(&metadata));

        metadata.insert("type".to_string(), "critical".to_string());
        assert!(filter.evaluate(&metadata));

        metadata.insert("type".to_string(), "normal".to_string());
        assert!(!filter.evaluate(&metadata));
    }

    #[test]
    fn test_contains_filter() {
        let filter = MetadataFilter::Contains {
            field: "description".to_string(),
            substring: "important".to_string(),
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "This is an important message".to_string(),
        );
        assert!(filter.evaluate(&metadata));

        metadata.insert("description".to_string(), "Regular message".to_string());
        assert!(!filter.evaluate(&metadata));
    }

    #[test]
    fn test_filter_builder() {
        let filter = FilterBuilder::new()
            .equals("category", FilterValue::String("tech".to_string()))
            .greater_than("score", FilterValue::Integer(70))
            .build_and();

        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "tech".to_string());
        metadata.insert("score".to_string(), "85".to_string());
        assert!(filter.evaluate(&metadata));
    }

    #[test]
    fn test_dimension_constraint() {
        let constraint = DimensionConstraint {
            dimension: 0,
            min_value: Some(0.0),
            max_value: Some(1.0),
        };

        let vec1 = Vector::new(vec![0.5, 0.3, 0.7]);
        assert!(constraint.satisfies(&vec1));

        let vec2 = Vector::new(vec![1.5, 0.3, 0.7]);
        assert!(!constraint.satisfies(&vec2));
    }

    #[test]
    fn test_search_filter() {
        let filter = SearchFilter::new()
            .with_max_distance(0.5)
            .with_metadata_filter(MetadataFilter::Equals {
                field: "category".to_string(),
                value: FilterValue::String("approved".to_string()),
            });

        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "approved".to_string());

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(filter.satisfies(0.3, &vector, &metadata));
        assert!(!filter.satisfies(0.7, &vector, &metadata)); // distance too high

        metadata.insert("category".to_string(), "pending".to_string());
        assert!(!filter.satisfies(0.3, &vector, &metadata)); // metadata doesn't match
    }
}
