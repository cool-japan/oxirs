//! GraphQL-Specific Span Attributes for Distributed Tracing
//!
//! This module provides standardized span attributes for GraphQL operations,
//! enabling detailed observability and performance analysis.
//!
//! # Features
//!
//! - **Operation Attributes**: Type, name, and document tracking
//! - **Field Resolution**: Per-field timing and error tracking
//! - **Complexity Metrics**: Query complexity, depth, and breadth
//! - **Cache Metrics**: Hit/miss rates and cache effectiveness
//! - **Error Attribution**: Detailed error context and categorization
//! - **Client Tracking**: Client identification and versioning
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::graphql_span_attributes::{GraphQLSpanAttributes, OperationType};
//!
//! let mut attrs = GraphQLSpanAttributes::new()
//!     .with_operation_type(OperationType::Query)
//!     .with_operation_name("GetUser")
//!     .with_field_path(vec!["user".to_string(), "profile".to_string()]);
//!
//! // Record field resolution
//! attrs.record_field_resolution("user", 25);
//! attrs.record_cache_hit("user");
//!
//! // Convert to span attributes
//! let span_attrs = attrs.to_attribute_map();
//! ```

use crate::trace_correlation::AttributeValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// GraphQL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Query operation
    Query,
    /// Mutation operation
    Mutation,
    /// Subscription operation
    Subscription,
}

impl OperationType {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Mutation => "mutation",
            Self::Subscription => "subscription",
        }
    }
}

/// Field resolution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMetrics {
    /// Field path (e.g., "user.profile.name")
    pub path: String,
    /// Resolution duration in microseconds
    pub duration_us: u64,
    /// Whether this field was cached
    pub cached: bool,
    /// Number of items resolved (for lists)
    pub item_count: Option<usize>,
    /// Error message if resolution failed
    pub error: Option<String>,
}

/// Cache metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Total cache lookups
    pub total_lookups: usize,
}

impl CacheMetrics {
    /// Update metrics
    pub fn update(&mut self) {
        self.total_lookups = self.hits + self.misses;
        if self.total_lookups > 0 {
            self.hit_rate = self.hits as f64 / self.total_lookups as f64;
        }
    }
}

/// GraphQL complexity metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Query complexity score
    pub complexity_score: u32,
    /// Query depth
    pub depth: u32,
    /// Query breadth (max fields at any level)
    pub breadth: u32,
    /// Number of fields requested
    pub field_count: u32,
    /// Number of aliases used
    pub alias_count: u32,
    /// Number of fragments used
    pub fragment_count: u32,
}

/// Error categorization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Validation error (syntax, schema)
    Validation,
    /// Authorization error
    Authorization,
    /// Field resolution error
    Resolution,
    /// Data source error
    DataSource,
    /// Internal server error
    Internal,
    /// Timeout error
    Timeout,
    /// Rate limit exceeded
    RateLimit,
}

impl ErrorCategory {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Authorization => "authorization",
            Self::Resolution => "resolution",
            Self::DataSource => "data_source",
            Self::Internal => "internal",
            Self::Timeout => "timeout",
            Self::RateLimit => "rate_limit",
        }
    }
}

/// GraphQL error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    /// Error message
    pub message: String,
    /// Error category
    pub category: ErrorCategory,
    /// Field path where error occurred
    pub path: Vec<String>,
    /// Error code
    pub code: Option<String>,
}

/// GraphQL-specific span attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLSpanAttributes {
    /// Operation type
    pub operation_type: Option<OperationType>,
    /// Operation name
    pub operation_name: Option<String>,
    /// GraphQL document (query string)
    pub document: Option<String>,
    /// Current field path
    pub field_path: Vec<String>,
    /// Field resolution metrics
    pub field_metrics: Vec<FieldMetrics>,
    /// Cache metrics
    pub cache_metrics: CacheMetrics,
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
    /// Errors encountered
    pub errors: Vec<GraphQLError>,
    /// Client name
    pub client_name: Option<String>,
    /// Client version
    pub client_version: Option<String>,
    /// Schema version
    pub schema_version: Option<String>,
    /// Variables used in query
    pub variables_count: usize,
    /// Whether query was persisted (APQ)
    pub persisted_query: bool,
    /// Start time for duration calculation
    #[serde(skip)]
    start_time: Option<Instant>,
}

impl GraphQLSpanAttributes {
    /// Create new GraphQL span attributes
    pub fn new() -> Self {
        Self {
            operation_type: None,
            operation_name: None,
            document: None,
            field_path: Vec::new(),
            field_metrics: Vec::new(),
            cache_metrics: CacheMetrics::default(),
            complexity_metrics: ComplexityMetrics::default(),
            errors: Vec::new(),
            client_name: None,
            client_version: None,
            schema_version: None,
            variables_count: 0,
            persisted_query: false,
            start_time: None,
        }
    }

    /// Set operation type
    pub fn with_operation_type(mut self, op_type: OperationType) -> Self {
        self.operation_type = Some(op_type);
        self
    }

    /// Set operation name
    pub fn with_operation_name(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    /// Set GraphQL document
    pub fn with_document(mut self, doc: impl Into<String>) -> Self {
        self.document = Some(doc.into());
        self
    }

    /// Set field path
    pub fn with_field_path(mut self, path: Vec<String>) -> Self {
        self.field_path = path;
        self
    }

    /// Set client information
    pub fn with_client(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.client_name = Some(name.into());
        self.client_version = Some(version.into());
        self
    }

    /// Set schema version
    pub fn with_schema_version(mut self, version: impl Into<String>) -> Self {
        self.schema_version = Some(version.into());
        self
    }

    /// Set variables count
    pub fn with_variables_count(mut self, count: usize) -> Self {
        self.variables_count = count;
        self
    }

    /// Mark as persisted query
    pub fn with_persisted_query(mut self, persisted: bool) -> Self {
        self.persisted_query = persisted;
        self
    }

    /// Set complexity metrics
    pub fn with_complexity(mut self, metrics: ComplexityMetrics) -> Self {
        self.complexity_metrics = metrics;
        self
    }

    /// Start timing
    pub fn start_timing(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record field resolution
    pub fn record_field_resolution(&mut self, field: impl Into<String>, duration_us: u64) {
        let field_str = field.into();
        self.field_metrics.push(FieldMetrics {
            path: field_str,
            duration_us,
            cached: false,
            item_count: None,
            error: None,
        });
    }

    /// Record field resolution with item count
    pub fn record_field_resolution_with_count(
        &mut self,
        field: impl Into<String>,
        duration_us: u64,
        count: usize,
    ) {
        let field_str = field.into();
        self.field_metrics.push(FieldMetrics {
            path: field_str,
            duration_us,
            cached: false,
            item_count: Some(count),
            error: None,
        });
    }

    /// Record field error
    pub fn record_field_error(&mut self, field: impl Into<String>, error: impl Into<String>) {
        let field_str = field.into();
        self.field_metrics.push(FieldMetrics {
            path: field_str,
            duration_us: 0,
            cached: false,
            item_count: None,
            error: Some(error.into()),
        });
    }

    /// Record cache hit
    pub fn record_cache_hit(&mut self, field: impl Into<String>) {
        self.cache_metrics.hits += 1;
        self.cache_metrics.update();

        // Mark latest field metric as cached
        if let Some(last_metric) = self.field_metrics.last_mut() {
            if last_metric.path == field.into() {
                last_metric.cached = true;
            }
        }
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_metrics.misses += 1;
        self.cache_metrics.update();
    }

    /// Add error
    pub fn add_error(&mut self, error: GraphQLError) {
        self.errors.push(error);
    }

    /// Convert to attribute map for span
    pub fn to_attribute_map(&self) -> HashMap<String, AttributeValue> {
        let mut attrs = HashMap::new();

        // Operation attributes
        if let Some(op_type) = &self.operation_type {
            attrs.insert(
                "graphql.operation.type".to_string(),
                AttributeValue::String(op_type.as_str().to_string()),
            );
        }
        if let Some(op_name) = &self.operation_name {
            attrs.insert(
                "graphql.operation.name".to_string(),
                AttributeValue::String(op_name.clone()),
            );
        }
        if let Some(doc) = &self.document {
            // Truncate document if too long
            let truncated = if doc.len() > 1000 {
                format!("{}...", &doc[..1000])
            } else {
                doc.clone()
            };
            attrs.insert(
                "graphql.document".to_string(),
                AttributeValue::String(truncated),
            );
        }

        // Field path
        if !self.field_path.is_empty() {
            attrs.insert(
                "graphql.field.path".to_string(),
                AttributeValue::String(self.field_path.join(".")),
            );
        }

        // Field metrics
        attrs.insert(
            "graphql.field.count".to_string(),
            AttributeValue::Int(self.field_metrics.len() as i64),
        );

        if !self.field_metrics.is_empty() {
            let total_duration: u64 = self.field_metrics.iter().map(|m| m.duration_us).sum();
            let avg_duration = total_duration / self.field_metrics.len() as u64;
            attrs.insert(
                "graphql.field.avg_duration_us".to_string(),
                AttributeValue::Int(avg_duration as i64),
            );

            let max_duration = self
                .field_metrics
                .iter()
                .map(|m| m.duration_us)
                .max()
                .unwrap_or(0);
            attrs.insert(
                "graphql.field.max_duration_us".to_string(),
                AttributeValue::Int(max_duration as i64),
            );
        }

        // Cache metrics
        attrs.insert(
            "graphql.cache.hits".to_string(),
            AttributeValue::Int(self.cache_metrics.hits as i64),
        );
        attrs.insert(
            "graphql.cache.misses".to_string(),
            AttributeValue::Int(self.cache_metrics.misses as i64),
        );
        attrs.insert(
            "graphql.cache.hit_rate".to_string(),
            AttributeValue::Float(self.cache_metrics.hit_rate),
        );

        // Complexity metrics
        attrs.insert(
            "graphql.complexity.score".to_string(),
            AttributeValue::Int(self.complexity_metrics.complexity_score as i64),
        );
        attrs.insert(
            "graphql.complexity.depth".to_string(),
            AttributeValue::Int(self.complexity_metrics.depth as i64),
        );
        attrs.insert(
            "graphql.complexity.breadth".to_string(),
            AttributeValue::Int(self.complexity_metrics.breadth as i64),
        );

        // Error attributes
        if !self.errors.is_empty() {
            attrs.insert(
                "graphql.error.count".to_string(),
                AttributeValue::Int(self.errors.len() as i64),
            );

            let error_categories: Vec<String> = self
                .errors
                .iter()
                .map(|e| e.category.as_str().to_string())
                .collect();
            attrs.insert(
                "graphql.error.categories".to_string(),
                AttributeValue::StringArray(error_categories),
            );
        }

        // Client attributes
        if let Some(client_name) = &self.client_name {
            attrs.insert(
                "graphql.client.name".to_string(),
                AttributeValue::String(client_name.clone()),
            );
        }
        if let Some(client_version) = &self.client_version {
            attrs.insert(
                "graphql.client.version".to_string(),
                AttributeValue::String(client_version.clone()),
            );
        }

        // Schema version
        if let Some(schema_version) = &self.schema_version {
            attrs.insert(
                "graphql.schema.version".to_string(),
                AttributeValue::String(schema_version.clone()),
            );
        }

        // Variables
        attrs.insert(
            "graphql.variables.count".to_string(),
            AttributeValue::Int(self.variables_count as i64),
        );

        // Persisted query
        attrs.insert(
            "graphql.persisted_query".to_string(),
            AttributeValue::Bool(self.persisted_query),
        );

        attrs
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> AttributeSummary {
        let total_duration: u64 = self.field_metrics.iter().map(|m| m.duration_us).sum();
        let field_count = self.field_metrics.len();
        let error_count = self.errors.len();

        AttributeSummary {
            operation_type: self.operation_type.map(|t| t.as_str().to_string()),
            operation_name: self.operation_name.clone(),
            field_count,
            total_duration_us: total_duration,
            avg_duration_us: if field_count > 0 {
                total_duration / field_count as u64
            } else {
                0
            },
            cache_hit_rate: self.cache_metrics.hit_rate,
            error_count,
            complexity_score: self.complexity_metrics.complexity_score,
        }
    }
}

impl Default for GraphQLSpanAttributes {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of span attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeSummary {
    /// Operation type
    pub operation_type: Option<String>,
    /// Operation name
    pub operation_name: Option<String>,
    /// Number of fields resolved
    pub field_count: usize,
    /// Total resolution duration
    pub total_duration_us: u64,
    /// Average resolution duration
    pub avg_duration_us: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of errors
    pub error_count: usize,
    /// Complexity score
    pub complexity_score: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_type() {
        assert_eq!(OperationType::Query.as_str(), "query");
        assert_eq!(OperationType::Mutation.as_str(), "mutation");
        assert_eq!(OperationType::Subscription.as_str(), "subscription");
    }

    #[test]
    fn test_error_category() {
        assert_eq!(ErrorCategory::Validation.as_str(), "validation");
        assert_eq!(ErrorCategory::Authorization.as_str(), "authorization");
        assert_eq!(ErrorCategory::Resolution.as_str(), "resolution");
    }

    #[test]
    fn test_cache_metrics_update() {
        let mut metrics = CacheMetrics {
            hits: 8,
            misses: 2,
            ..Default::default()
        };
        metrics.update();

        assert_eq!(metrics.total_lookups, 10);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_graphql_span_attributes_creation() {
        let attrs = GraphQLSpanAttributes::new()
            .with_operation_type(OperationType::Query)
            .with_operation_name("GetUser")
            .with_field_path(vec!["user".to_string(), "profile".to_string()]);

        assert_eq!(attrs.operation_type, Some(OperationType::Query));
        assert_eq!(attrs.operation_name, Some("GetUser".to_string()));
        assert_eq!(attrs.field_path.len(), 2);
    }

    #[test]
    fn test_record_field_resolution() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.record_field_resolution("user", 1000);
        attrs.record_field_resolution("posts", 2000);

        assert_eq!(attrs.field_metrics.len(), 2);
        assert_eq!(attrs.field_metrics[0].path, "user");
        assert_eq!(attrs.field_metrics[0].duration_us, 1000);
        assert_eq!(attrs.field_metrics[1].path, "posts");
        assert_eq!(attrs.field_metrics[1].duration_us, 2000);
    }

    #[test]
    fn test_record_field_resolution_with_count() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.record_field_resolution_with_count("posts", 3000, 10);

        assert_eq!(attrs.field_metrics.len(), 1);
        assert_eq!(attrs.field_metrics[0].item_count, Some(10));
    }

    #[test]
    fn test_record_field_error() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.record_field_error("user", "Not found");

        assert_eq!(attrs.field_metrics.len(), 1);
        assert_eq!(attrs.field_metrics[0].error, Some("Not found".to_string()));
    }

    #[test]
    fn test_cache_hit_miss() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.record_field_resolution("user", 1000);
        attrs.record_cache_hit("user");
        attrs.record_cache_miss();

        assert_eq!(attrs.cache_metrics.hits, 1);
        assert_eq!(attrs.cache_metrics.misses, 1);
        assert!((attrs.cache_metrics.hit_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_add_error() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.add_error(GraphQLError {
            message: "Unauthorized".to_string(),
            category: ErrorCategory::Authorization,
            path: vec!["user".to_string()],
            code: Some("AUTH_ERROR".to_string()),
        });

        assert_eq!(attrs.errors.len(), 1);
        assert_eq!(attrs.errors[0].category, ErrorCategory::Authorization);
    }

    #[test]
    fn test_to_attribute_map() {
        let mut attrs = GraphQLSpanAttributes::new()
            .with_operation_type(OperationType::Query)
            .with_operation_name("GetUser")
            .with_client("apollo-client", "3.0.0")
            .with_schema_version("1.2.3")
            .with_variables_count(5)
            .with_persisted_query(true);

        attrs.record_field_resolution("user", 1000);
        attrs.record_cache_hit("user");

        let attr_map = attrs.to_attribute_map();

        assert!(attr_map.contains_key("graphql.operation.type"));
        assert!(attr_map.contains_key("graphql.operation.name"));
        assert!(attr_map.contains_key("graphql.client.name"));
        assert!(attr_map.contains_key("graphql.cache.hit_rate"));
        assert!(attr_map.contains_key("graphql.persisted_query"));
    }

    #[test]
    fn test_complexity_metrics() {
        let complexity = ComplexityMetrics {
            complexity_score: 100,
            depth: 5,
            breadth: 10,
            field_count: 20,
            alias_count: 2,
            fragment_count: 3,
        };

        let attrs = GraphQLSpanAttributes::new().with_complexity(complexity);

        assert_eq!(attrs.complexity_metrics.complexity_score, 100);
        assert_eq!(attrs.complexity_metrics.depth, 5);
        assert_eq!(attrs.complexity_metrics.breadth, 10);
    }

    #[test]
    fn test_get_summary() {
        let mut attrs = GraphQLSpanAttributes::new()
            .with_operation_type(OperationType::Query)
            .with_operation_name("GetPosts");

        attrs.record_field_resolution("posts", 1000);
        attrs.record_field_resolution("author", 500);
        attrs.record_cache_hit("posts");
        attrs.record_cache_miss();

        let summary = attrs.get_summary();

        assert_eq!(summary.operation_type, Some("query".to_string()));
        assert_eq!(summary.operation_name, Some("GetPosts".to_string()));
        assert_eq!(summary.field_count, 2);
        assert_eq!(summary.total_duration_us, 1500);
        assert_eq!(summary.avg_duration_us, 750);
        assert!((summary.cache_hit_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_document_truncation() {
        let long_doc = "a".repeat(1500);
        let attrs = GraphQLSpanAttributes::new().with_document(long_doc);

        let attr_map = attrs.to_attribute_map();

        if let Some(AttributeValue::String(doc)) = attr_map.get("graphql.document") {
            assert!(doc.len() <= 1003); // 1000 + "..."
            assert!(doc.ends_with("..."));
        } else {
            panic!("Document attribute not found or wrong type");
        }
    }

    #[test]
    fn test_field_metrics_aggregation() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.record_field_resolution("field1", 100);
        attrs.record_field_resolution("field2", 200);
        attrs.record_field_resolution("field3", 300);

        let attr_map = attrs.to_attribute_map();

        match attr_map.get("graphql.field.avg_duration_us") {
            Some(AttributeValue::Int(avg)) => assert_eq!(*avg, 200),
            _ => panic!("Avg duration not found"),
        }

        match attr_map.get("graphql.field.max_duration_us") {
            Some(AttributeValue::Int(max)) => assert_eq!(*max, 300),
            _ => panic!("Max duration not found"),
        }
    }

    #[test]
    fn test_error_categories_in_attributes() {
        let mut attrs = GraphQLSpanAttributes::new();

        attrs.add_error(GraphQLError {
            message: "Validation failed".to_string(),
            category: ErrorCategory::Validation,
            path: vec!["query".to_string()],
            code: None,
        });

        attrs.add_error(GraphQLError {
            message: "Unauthorized".to_string(),
            category: ErrorCategory::Authorization,
            path: vec!["user".to_string()],
            code: Some("AUTH_ERROR".to_string()),
        });

        let attr_map = attrs.to_attribute_map();

        match attr_map.get("graphql.error.categories") {
            Some(AttributeValue::StringArray(categories)) => {
                assert_eq!(categories.len(), 2);
                assert!(categories.contains(&"validation".to_string()));
                assert!(categories.contains(&"authorization".to_string()));
            }
            _ => panic!("Error categories not found"),
        }
    }

    #[test]
    fn test_persisted_query_attribute() {
        let attrs = GraphQLSpanAttributes::new().with_persisted_query(true);

        let attr_map = attrs.to_attribute_map();

        match attr_map.get("graphql.persisted_query") {
            Some(AttributeValue::Bool(true)) => {}
            _ => panic!("Persisted query attribute incorrect"),
        }
    }
}
