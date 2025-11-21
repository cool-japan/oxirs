//! Semantic Query Understanding for NL2SPARQL
//!
//! This module provides advanced semantic understanding for natural language queries,
//! integrating with the NLP pipeline for better SPARQL generation.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::nlp::{EntityType, ExtractedEntity, IntentResult, IntentType};
use crate::schema_introspection::DiscoveredSchema;

/// Semantic query understanding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticUnderstanding {
    /// Recognized intent
    pub intent: IntentType,
    /// Intent confidence
    pub intent_confidence: f32,
    /// Extracted entities with types
    pub entities: Vec<ExtractedEntity>,
    /// Query type (SELECT, ASK, CONSTRUCT, DESCRIBE)
    pub query_type: QueryType,
    /// Required triple patterns
    pub triple_patterns: Vec<TriplePattern>,
    /// Filters to apply
    pub filters: Vec<Filter>,
    /// Aggregations requested
    pub aggregations: Vec<Aggregation>,
    /// Ordering preferences
    pub ordering: Option<Ordering>,
    /// Limit/offset hints
    pub pagination: Option<Pagination>,
    /// Schema elements to use
    pub schema_hints: SchemaHints,
}

/// SPARQL query type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
}

/// Triple pattern for SPARQL generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: TripleElement,
    pub predicate: TripleElement,
    pub object: TripleElement,
    pub optional: bool,
}

/// Element in a triple pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TripleElement {
    Variable(String),
    URI(String),
    Literal(String),
    BNode(String),
}

/// Filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub variable: String,
    pub operator: FilterOperator,
    pub value: String,
}

/// Filter operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    Regex,
    Lang,
}

/// Aggregation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation {
    pub function: AggregationFunction,
    pub variable: String,
    pub alias: Option<String>,
}

/// Aggregation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    GroupConcat,
    Sample,
}

/// Query ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ordering {
    pub variable: String,
    pub direction: OrderDirection,
}

/// Order direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderDirection {
    Ascending,
    Descending,
}

/// Pagination hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Schema hints for query generation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaHints {
    /// Relevant RDF classes
    pub classes: Vec<String>,
    /// Relevant properties
    pub properties: Vec<String>,
    /// Prefixes to use
    pub prefixes: HashMap<String, String>,
}

/// Semantic query analyzer
pub struct SemanticQueryAnalyzer {
    /// Cached schema for schema-aware generation
    schema: Option<DiscoveredSchema>,
}

impl SemanticQueryAnalyzer {
    /// Create a new semantic query analyzer
    pub fn new() -> Self {
        info!("Initialized semantic query analyzer");
        Self { schema: None }
    }

    /// Set the schema for schema-aware analysis
    pub fn set_schema(&mut self, schema: DiscoveredSchema) {
        self.schema = Some(schema);
    }

    /// Analyze a query with NLP results
    pub fn analyze(
        &self,
        query: &str,
        intent: IntentResult,
        entities: Vec<ExtractedEntity>,
    ) -> Result<SemanticUnderstanding> {
        debug!("Analyzing query semantics: {}", query);

        // Determine query type from intent
        let query_type = self.infer_query_type(&intent, query);

        // Build triple patterns from entities and intent
        let triple_patterns = self.build_triple_patterns(&entities, &intent, query);

        // Extract filters from query
        let filters = self.extract_filters(query, &entities);

        // Detect aggregations
        let aggregations = self.detect_aggregations(query, &intent);

        // Detect ordering preferences
        let ordering = self.detect_ordering(query);

        // Extract pagination hints
        let pagination = self.extract_pagination(query);

        // Generate schema hints
        let schema_hints = self.generate_schema_hints(&entities, &triple_patterns);

        Ok(SemanticUnderstanding {
            intent: intent.primary_intent,
            intent_confidence: intent.confidence,
            entities,
            query_type,
            triple_patterns,
            filters,
            aggregations,
            ordering,
            pagination,
            schema_hints,
        })
    }

    /// Infer SPARQL query type from intent
    fn infer_query_type(&self, intent: &IntentResult, query: &str) -> QueryType {
        let lowercase = query.to_lowercase();

        // Explicit query type keywords
        if lowercase.contains("ask") || lowercase.contains("is there") || lowercase.contains("does")
        {
            return QueryType::Ask;
        }

        if lowercase.contains("describe") || lowercase.contains("tell me about") {
            return QueryType::Describe;
        }

        if lowercase.contains("construct") {
            return QueryType::Construct;
        }

        // Intent-based inference
        match intent.primary_intent {
            IntentType::Query | IntentType::Exploration => QueryType::Select,
            IntentType::Analytics => QueryType::Select,
            IntentType::Explanation => QueryType::Describe,
            _ => QueryType::Select,
        }
    }

    /// Build triple patterns from entities and intent
    fn build_triple_patterns(
        &self,
        entities: &[ExtractedEntity],
        intent: &IntentResult,
        query: &str,
    ) -> Vec<TriplePattern> {
        let mut patterns = Vec::new();
        let lowercase = query.to_lowercase();

        // Basic pattern: entity type queries
        for entity in entities {
            match entity.entity_type {
                EntityType::RDFResource | EntityType::Class => {
                    // ?s rdf:type <entity>
                    patterns.push(TriplePattern {
                        subject: TripleElement::Variable("s".to_string()),
                        predicate: TripleElement::URI(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                        ),
                        object: TripleElement::URI(entity.text.clone()),
                        optional: false,
                    });
                }
                EntityType::Property => {
                    // ?s <property> ?o
                    patterns.push(TriplePattern {
                        subject: TripleElement::Variable("s".to_string()),
                        predicate: TripleElement::URI(entity.text.clone()),
                        object: TripleElement::Variable("o".to_string()),
                        optional: false,
                    });
                }
                _ => {
                    // General pattern for other entities
                    if intent.primary_intent == IntentType::Analytics && lowercase.contains("count")
                    {
                        // For count queries, add type pattern
                        patterns.push(TriplePattern {
                            subject: TripleElement::Variable("s".to_string()),
                            predicate: TripleElement::URI(
                                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                            ),
                            object: TripleElement::Variable("type".to_string()),
                            optional: false,
                        });
                    }
                }
            }
        }

        // If no patterns generated, add a default triple pattern
        if patterns.is_empty() {
            patterns.push(TriplePattern {
                subject: TripleElement::Variable("s".to_string()),
                predicate: TripleElement::Variable("p".to_string()),
                object: TripleElement::Variable("o".to_string()),
                optional: false,
            });
        }

        patterns
    }

    /// Extract filters from query
    fn extract_filters(&self, query: &str, entities: &[ExtractedEntity]) -> Vec<Filter> {
        let mut filters = Vec::new();

        // Number filters
        for entity in entities {
            if entity.entity_type == EntityType::Number {
                // Try to infer filter type from context
                let filter = if query.contains("greater than") || query.contains(">") {
                    Filter {
                        variable: "value".to_string(),
                        operator: FilterOperator::GreaterThan,
                        value: entity.text.clone(),
                    }
                } else if query.contains("less than") || query.contains("<") {
                    Filter {
                        variable: "value".to_string(),
                        operator: FilterOperator::LessThan,
                        value: entity.text.clone(),
                    }
                } else if query.contains("equals") || query.contains("=") {
                    Filter {
                        variable: "value".to_string(),
                        operator: FilterOperator::Equals,
                        value: entity.text.clone(),
                    }
                } else {
                    continue;
                };

                filters.push(filter);
            }
        }

        // DateTime filters
        for entity in entities {
            if entity.entity_type == EntityType::DateTime {
                let lowercase = query.to_lowercase();
                if lowercase.contains("after") {
                    filters.push(Filter {
                        variable: "date".to_string(),
                        operator: FilterOperator::GreaterThan,
                        value: format!("\"{}\"", entity.text),
                    });
                } else if lowercase.contains("before") {
                    filters.push(Filter {
                        variable: "date".to_string(),
                        operator: FilterOperator::LessThan,
                        value: format!("\"{}\"", entity.text),
                    });
                }
            }
        }

        // String filters with contains
        if query.to_lowercase().contains("contains") || query.to_lowercase().contains("includes") {
            filters.push(Filter {
                variable: "label".to_string(),
                operator: FilterOperator::Contains,
                value: "search_term".to_string(),
            });
        }

        filters
    }

    /// Detect aggregations in query
    fn detect_aggregations(&self, query: &str, intent: &IntentResult) -> Vec<Aggregation> {
        let mut aggregations = Vec::new();
        let lowercase = query.to_lowercase();

        // Count
        if lowercase.contains("count") || lowercase.contains("how many") {
            aggregations.push(Aggregation {
                function: AggregationFunction::Count,
                variable: "s".to_string(),
                alias: Some("count".to_string()),
            });
        }

        // Sum
        if lowercase.contains("sum") || lowercase.contains("total") {
            aggregations.push(Aggregation {
                function: AggregationFunction::Sum,
                variable: "value".to_string(),
                alias: Some("sum".to_string()),
            });
        }

        // Average
        if lowercase.contains("average") || lowercase.contains("avg") || lowercase.contains("mean")
        {
            aggregations.push(Aggregation {
                function: AggregationFunction::Avg,
                variable: "value".to_string(),
                alias: Some("avg".to_string()),
            });
        }

        // Min/Max
        if lowercase.contains("minimum") || lowercase.contains("smallest") {
            aggregations.push(Aggregation {
                function: AggregationFunction::Min,
                variable: "value".to_string(),
                alias: Some("min".to_string()),
            });
        }

        if lowercase.contains("maximum")
            || lowercase.contains("largest")
            || lowercase.contains("biggest")
        {
            aggregations.push(Aggregation {
                function: AggregationFunction::Max,
                variable: "value".to_string(),
                alias: Some("max".to_string()),
            });
        }

        // Analytics intent suggests aggregation
        if intent.primary_intent == IntentType::Analytics && aggregations.is_empty() {
            aggregations.push(Aggregation {
                function: AggregationFunction::Count,
                variable: "s".to_string(),
                alias: Some("count".to_string()),
            });
        }

        aggregations
    }

    /// Detect ordering preferences
    fn detect_ordering(&self, query: &str) -> Option<Ordering> {
        let lowercase = query.to_lowercase();

        if lowercase.contains("order by") || lowercase.contains("sort by") {
            let direction = if lowercase.contains("descending") || lowercase.contains("desc") {
                OrderDirection::Descending
            } else {
                OrderDirection::Ascending
            };

            Some(Ordering {
                variable: "s".to_string(),
                direction,
            })
        } else if lowercase.contains("sorted") || lowercase.contains("ordered") {
            Some(Ordering {
                variable: "s".to_string(),
                direction: OrderDirection::Ascending,
            })
        } else {
            None
        }
    }

    /// Extract pagination hints
    fn extract_pagination(&self, query: &str) -> Option<Pagination> {
        let lowercase = query.to_lowercase();
        let mut limit = None;
        let offset = None;

        // Extract limit
        if let Some(limit_match) = lowercase.find("limit") {
            let after_limit = &lowercase[limit_match + 5..];
            let number_str = after_limit
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok());
            limit = number_str;
        }

        // Extract from patterns like "top 10", "first 5"
        if let Some(top_match) = lowercase.find("top ") {
            let after_top = &lowercase[top_match + 4..];
            let number_str = after_top
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok());
            limit = number_str.or(limit);
        }

        if let Some(first_match) = lowercase.find("first ") {
            let after_first = &lowercase[first_match + 6..];
            let number_str = after_first
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok());
            limit = number_str.or(limit);
        }

        // Default limit if none specified
        if limit.is_none() && (lowercase.contains("list") || lowercase.contains("show")) {
            limit = Some(100);
        }

        if limit.is_some() || offset.is_some() {
            Some(Pagination { limit, offset })
        } else {
            None
        }
    }

    /// Generate schema hints from entities and patterns
    fn generate_schema_hints(
        &self,
        entities: &[ExtractedEntity],
        _patterns: &[TriplePattern],
    ) -> SchemaHints {
        let mut hints = SchemaHints::default();

        // Extract classes and properties from entities
        for entity in entities {
            match entity.entity_type {
                EntityType::Class => hints.classes.push(entity.text.clone()),
                EntityType::Property => hints.properties.push(entity.text.clone()),
                EntityType::RDFResource => {
                    // Try to determine if it's a class or property
                    if entity
                        .text
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        hints.classes.push(entity.text.clone());
                    }
                }
                _ => {}
            }
        }

        // Add common prefixes
        hints.prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        hints.prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        hints.prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        // Use schema if available
        if let Some(ref schema) = self.schema {
            for class in &schema.classes {
                if !hints.classes.contains(&class.uri) {
                    hints.classes.push(class.uri.clone());
                }
            }

            for (prefix, uri) in &schema.prefixes {
                if !hints.prefixes.contains_key(prefix as &str) {
                    hints.prefixes.insert(prefix.clone(), uri.clone());
                }
            }
        }

        hints
    }
}

impl Default for SemanticQueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlp::IntentRecognitionConfig;

    #[test]
    fn test_query_type_inference() {
        let analyzer = SemanticQueryAnalyzer::new();
        let intent = crate::nlp::IntentRecognizer::new(IntentRecognitionConfig::default())
            .unwrap()
            .recognize("How many movies are there?")
            .unwrap();

        let query_type = analyzer.infer_query_type(&intent, "How many movies are there?");
        assert_eq!(query_type, QueryType::Select);
    }

    #[test]
    fn test_aggregation_detection() {
        let analyzer = SemanticQueryAnalyzer::new();
        let intent = crate::nlp::IntentRecognizer::new(IntentRecognitionConfig::default())
            .unwrap()
            .recognize("Count all users")
            .unwrap();

        let aggregations = analyzer.detect_aggregations("Count all users", &intent);
        assert!(!aggregations.is_empty());
        assert_eq!(aggregations[0].function, AggregationFunction::Count);
    }

    #[test]
    fn test_pagination_extraction() {
        let analyzer = SemanticQueryAnalyzer::new();

        let pagination = analyzer.extract_pagination("Show me the top 10 results");
        assert!(pagination.is_some());
        assert_eq!(pagination.unwrap().limit, Some(10));
    }
}
