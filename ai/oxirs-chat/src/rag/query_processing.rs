//! Query constraint processing and analysis utilities
//!
//! Provides query analysis, constraint extraction, and processing utilities for RAG queries.

use super::*;

/// Query processor for analyzing and extracting constraints from queries
pub struct QueryProcessor;

impl QueryProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Extract constraints from query based on entities
    pub async fn extract_constraints(
        &self,
        query: &str,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<QueryConstraint>> {
        let mut constraints = Vec::new();
        let query_lower = query.to_lowercase();

        // Temporal constraints
        let temporal_patterns = [
            (
                r"(?:in|during|from|since|before|after)\s+(\d{4})",
                ConstraintType::Temporal,
                "year",
            ),
            (
                r"(?:today|yesterday|tomorrow|now|recent)",
                ConstraintType::Temporal,
                "relative_time",
            ),
        ];

        for (pattern, constraint_type, operator) in temporal_patterns {
            let regex = Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let Some(value) = cap.get(1) {
                    constraints.push(QueryConstraint {
                        constraint_type,
                        value: value.as_str().to_string(),
                        operator: operator.to_string(),
                    });
                } else if cap.get(0).is_some() {
                    constraints.push(QueryConstraint {
                        constraint_type,
                        value: cap.get(0).unwrap().as_str().to_string(),
                        operator: operator.to_string(),
                    });
                }
            }
        }

        // Type constraints
        if query_lower.contains("type")
            || query_lower.contains("kind")
            || query_lower.contains("class")
        {
            constraints.push(QueryConstraint {
                constraint_type: ConstraintType::Type,
                value: "type_constraint".to_string(),
                operator: "equals".to_string(),
            });
        }

        // Value constraints (numeric, comparison)
        let value_patterns = [
            (r"(?:greater than|more than|>\s*)(\d+)", "greater_than"),
            (r"(?:less than|fewer than|<\s*)(\d+)", "less_than"),
            (r"(?:equals?|is|=\s*)(\d+)", "equals"),
        ];

        for (pattern, operator) in value_patterns {
            let regex = Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let Some(value) = cap.get(1) {
                    constraints.push(QueryConstraint {
                        constraint_type: ConstraintType::Value,
                        value: value.as_str().to_string(),
                        operator: operator.to_string(),
                    });
                }
            }
        }

        // Entity-based constraints
        for entity in entities {
            match entity.entity_type {
                EntityType::Person => {
                    constraints.push(QueryConstraint {
                        constraint_type: ConstraintType::Entity,
                        value: entity.text.clone(),
                        operator: "person_filter".to_string(),
                    });
                }
                EntityType::Location => {
                    constraints.push(QueryConstraint {
                        constraint_type: ConstraintType::Spatial,
                        value: entity.text.clone(),
                        operator: "location_filter".to_string(),
                    });
                }
                _ => {}
            }
        }

        debug!("Extracted {} constraints from query", constraints.len());
        Ok(constraints)
    }

    /// Analyze query intent and complexity
    pub fn analyze_query_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        if query_lower.contains("how many") || query_lower.contains("count") {
            QueryIntent::Counting
        } else if query_lower.contains("what is") || query_lower.contains("define") {
            QueryIntent::Definition
        } else if query_lower.contains("compare") || query_lower.contains("difference") {
            QueryIntent::Comparison
        } else if query_lower.contains("list") || query_lower.contains("show all") {
            QueryIntent::Listing
        } else if query_lower.contains("why") || query_lower.contains("because") {
            QueryIntent::Explanation
        } else {
            QueryIntent::General
        }
    }

    /// Calculate query complexity score
    pub fn calculate_query_complexity(&self, query: &str) -> f64 {
        let word_count = query.split_whitespace().count();
        let unique_words = query.split_whitespace().collect::<HashSet<_>>().len();
        let question_words = ["what", "how", "why", "when", "where", "who", "which"];
        let query_lower = query.to_lowercase();
        
        let question_word_count = question_words.iter()
            .filter(|word| query_lower.contains(*word))
            .count();

        let complexity = (word_count as f64 * 0.05) + 
                        (unique_words as f64 * 0.1) +
                        (question_word_count as f64 * 0.2);
        
        complexity.min(1.0)
    }
}

/// Query constraint for filtering and processing
#[derive(Debug, Clone)]
pub struct QueryConstraint {
    pub constraint_type: ConstraintType,
    pub value: String,
    pub operator: String,
}

/// Types of constraints that can be extracted from queries
#[derive(Debug, Clone, Copy)]
pub enum ConstraintType {
    Temporal,
    Spatial,
    Type,
    Value,
    Entity,
    Relationship,
}

/// Query intent classification
#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    Definition,
    Comparison,
    Counting,
    Listing,
    Explanation,
    General,
}

use super::graph_traversal::{ExtractedEntity, EntityType};