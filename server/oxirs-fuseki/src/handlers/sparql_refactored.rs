//! Refactored SPARQL 1.1/1.2 Protocol implementation with modular architecture
//!
//! This module provides a clean, modular implementation of the SPARQL protocol
//! with all components properly separated for maintainability.

// Re-export core functionality from submodules (using specific imports to avoid conflicts)
pub use crate::handlers::sparql::content_types::*;
pub use crate::handlers::sparql::core::*;
pub use crate::handlers::sparql::optimizers::*;

// Import specific items from modules that have conflicts
pub use crate::handlers::sparql::aggregation_engine::{
    EnhancedAggregationProcessor, AggregationFunction
};
pub use crate::handlers::sparql::bind_processor::{
    EnhancedBindProcessor, EnhancedValuesProcessor
};
pub use crate::handlers::sparql::service_delegation::{
    ServiceDelegationManager, ParallelServiceExecutor, ServiceResultMerger
};
pub use crate::handlers::sparql::sparql12_features::{
    Sparql12Features, AggregationEngine
};

use crate::{
    auth::AuthUser,
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    response::IntoResponse,
    Form,
};
use std::sync::Arc;

// Functions are already available via the glob re-export from core module

/// Main SPARQL query endpoint for GET requests
pub async fn query_handler_get(
    Query(params): Query<SparqlQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    // Delegate to the SPARQL query handler
    crate::handlers::sparql::core::sparql_query(
        Query(params),
        State(state),
        headers,
        user,
    )
    .await
}

/// Main SPARQL query endpoint for POST requests
pub async fn query_handler_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    // Check content type and handle body accordingly
    let content_type = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    
    tracing::debug!("POST /sparql request with content-type: {}", content_type);

    let params = if content_type.contains("application/x-www-form-urlencoded") {
        // Parse form data manually from body
        let body_str = String::from_utf8_lossy(&body);
        let mut query = None;
        let mut default_graph_uri = None;
        let mut named_graph_uri = None;
        
        for part in body_str.split('&') {
            if let Some((key, value)) = part.split_once('=') {
                let decoded_value = urlencoding::decode(value).unwrap_or_default().to_string();
                match key {
                    "query" => query = Some(decoded_value),
                    "default-graph-uri" => {
                        default_graph_uri = Some(vec![decoded_value]);
                    }
                    "named-graph-uri" => {
                        named_graph_uri = Some(vec![decoded_value]);
                    }
                    _ => {}
                }
            }
        }
        
        SparqlQueryParams {
            query,
            default_graph_uri,
            named_graph_uri,
            timeout: None,
            format: None,
        }
    } else if content_type.contains("application/sparql-query") {
        // Direct SPARQL query in body
        let query_string = String::from_utf8_lossy(&body).to_string();
        SparqlQueryParams {
            query: Some(query_string),
            default_graph_uri: None,
            named_graph_uri: None,
            timeout: None,
            format: None,
        }
    } else {
        // Invalid content type - return error
        tracing::debug!("Invalid content type detected: {}", content_type);
        return FusekiError::bad_request(format!(
            "Unsupported content type: {content_type}. Expected 'application/sparql-query' or 'application/x-www-form-urlencoded'"
        )).into_response();
    };

    // Delegate to the SPARQL query handler
    crate::handlers::sparql::core::sparql_query(
        Query(params),
        State(state),
        headers,
        user,
    )
    .await
    .into_response()
}

/// Main SPARQL query endpoint (backwards compatibility)
pub async fn query_handler(
    Query(params): Query<SparqlQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    // For backwards compatibility, delegate directly to sparql_query
    crate::handlers::sparql::core::sparql_query(
        Query(params),
        State(state),
        headers,
        user,
    )
    .await
}

/// Main SPARQL update endpoint with enhanced features
pub async fn update_handler(
    State(state): State<Arc<AppState>>,
    Form(params): Form<SparqlUpdateParams>,
) -> impl IntoResponse {
    use axum::Json;
    

    // Create query context
    let context = QueryContext::default();

    // Execute update
    match execute_sparql_update(&params.update, context, &state).await {
        Ok(result) => Json(serde_json::json!({
            "success": true,
            "message": "Update executed successfully",
            "modified_count": result.affected_triples.unwrap_or(0)
        }))
        .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "update_execution_failed",
                "message": e.to_string()
            })),
        )
            .into_response(),
    }
}

/// Enhanced SPARQL service with all advanced features
#[derive(Debug, Clone)]
pub struct EnhancedSparqlService {
    pub sparql12_features: Sparql12Features,
    pub service_delegator: ServiceDelegationManager,
    pub aggregation_processor: EnhancedAggregationProcessor,
    pub bind_processor: EnhancedBindProcessor,
    pub values_processor: EnhancedValuesProcessor,
    pub content_negotiator: ContentNegotiator,
    pub injection_detector: InjectionDetector,
    pub complexity_analyzer: ComplexityAnalyzer,
    pub performance_optimizer: PerformanceOptimizer,
}

impl EnhancedSparqlService {
    /// Create a new enhanced SPARQL service
    pub fn new() -> Self {
        Self {
            sparql12_features: Sparql12Features::new(),
            service_delegator: ServiceDelegationManager::new(),
            aggregation_processor: EnhancedAggregationProcessor::new(),
            bind_processor: EnhancedBindProcessor::new(),
            values_processor: EnhancedValuesProcessor::new(),
            content_negotiator: ContentNegotiator::new(),
            injection_detector: InjectionDetector::new(),
            complexity_analyzer: ComplexityAnalyzer::new(),
            performance_optimizer: PerformanceOptimizer::new(),
        }
    }

    /// Process a SPARQL query with all enhancements
    pub async fn process_query(
        &mut self,
        query: &str,
        context: QueryContext,
        headers: &HeaderMap,
    ) -> FusekiResult<String> {
        // 1. Security validation
        if self.injection_detector.detect_injection(query)? {
            return Err(FusekiError::authorization(
                "Potential SPARQL injection detected",
            ));
        }

        // 2. Complexity analysis
        let complexity = self.complexity_analyzer.analyze_complexity(query)?;
        if complexity.is_complex {
            tracing::warn!("Complex query detected: {:?}", complexity);
        }

        // 3. Apply optimizations
        let mut optimized_query = query.to_string();

        // Apply SPARQL 1.2 features and optimizations
        optimized_query = self
            .sparql12_features
            .optimize_query(&optimized_query)
            .await?;

        // Apply performance optimizations
        optimized_query = self.performance_optimizer.optimize(&optimized_query)?;

        // Process SERVICE clauses if present
        if optimized_query.to_uppercase().contains("SERVICE") {
            optimized_query = self
                .service_delegator
                .process_service_clauses(&optimized_query)
                .await?;
        }

        // Process aggregations
        if contains_aggregation_functions(&optimized_query) {
            optimized_query = self
                .aggregation_processor
                .process_aggregations(&optimized_query)?;
        }

        // Process BIND clauses
        if optimized_query.to_uppercase().contains("BIND") {
            optimized_query = self.bind_processor.process_bind_clauses(&optimized_query)?;
        }

        // Process VALUES clauses
        if optimized_query.to_uppercase().contains("VALUES") {
            optimized_query = self
                .values_processor
                .process_values_clauses(&optimized_query)?;
        }

        Ok(optimized_query)
    }

    /// Process a SPARQL update with all enhancements
    pub async fn process_update(
        &mut self,
        update: &str,
        context: QueryContext,
    ) -> FusekiResult<String> {
        // Security validation for updates
        if self.injection_detector.detect_injection(update)? {
            return Err(FusekiError::authorization(
                "Potential SPARQL injection detected in update",
            ));
        }

        // Apply update-specific optimizations
        let optimized_update = self.performance_optimizer.optimize(update)?;

        Ok(optimized_update)
    }

    /// Negotiate content type for response
    pub fn negotiate_content_type(&self, headers: &HeaderMap) -> String {
        self.content_negotiator.negotiate(headers)
    }

    /// Format response according to content type
    pub fn format_response<T: serde::Serialize + std::fmt::Debug>(
        &self,
        data: &T,
        content_type: &str,
    ) -> FusekiResult<String> {
        ResponseFormatter::format(data, content_type)
            .map_err(|e| FusekiError::response_formatting(e.to_string()))
    }
}

impl Default for EnhancedSparqlService {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions moved to appropriate modules but re-exported for compatibility

pub fn validate_sparql_query(query: &str) -> FusekiResult<()> {
    // Basic SPARQL validation - check for basic syntax
    if query.trim().is_empty() {
        return Err(crate::error::FusekiError::bad_request(
            "Empty query".to_string(),
        ));
    }
    Ok(())
}

pub fn contains_aggregation_functions(query: &str) -> bool {
    let upper = query.to_uppercase();
    [
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "GROUP_CONCAT",
        "SAMPLE",
    ]
    .iter()
    .any(|func| upper.contains(&format!("{func}(")))
}

pub fn contains_sparql_star_features(query: &str) -> bool {
    let upper = query.to_uppercase();
    
    // Check for quoted triple syntax
    let has_quoted_triples = upper.contains("<<") && upper.contains(">>");
    
    // Check for SPARQL-star functions
    let has_star_functions = upper.contains("SUBJECT(") 
        || upper.contains("PREDICATE(")
        || upper.contains("OBJECT(")
        || upper.contains("ISTRIPLE(");
    
    // Check for annotation syntax
    let has_annotations = query.contains("{|") && query.contains("|}");
    
    has_quoted_triples || has_star_functions || has_annotations
}

pub fn contains_property_paths(query: &str) -> bool {
    // Property path operators in SPARQL: / | + * ?
    // We need to be careful not to match variables (starting with ?)
    // Look for typical property path patterns

    // Simple pattern detection without regex for performance
    let chars: Vec<char> = query.chars().collect();
    for i in 0..chars.len() {
        match chars[i] {
            // Property path sequence: prop1/prop2
            '/' => {
                // Check if it's not a URI scheme (like http://)
                if i > 0
                    && chars[i - 1].is_alphanumeric()
                    && i + 1 < chars.len()
                    && chars[i + 1].is_alphanumeric()
                {
                    return true;
                }
            }
            // Property path alternative: prop1|prop2
            '|' => {
                if i > 0
                    && chars[i - 1].is_alphanumeric()
                    && i + 1 < chars.len()
                    && chars[i + 1].is_alphanumeric()
                {
                    return true;
                }
            }
            // One or more path: prop+
            '+' => {
                if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == ':') {
                    return true;
                }
            }
            // Zero or more path: prop*
            '*' => {
                if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == ':') {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

pub fn contains_subqueries(query: &str) -> bool {
    let upper = query.to_uppercase();
    upper.matches("SELECT").count() > 1 || upper.contains("ASK") || upper.contains("CONSTRUCT")
}

pub fn contains_bind_values(query: &str) -> bool {
    let upper = query.to_uppercase();
    upper.contains("BIND(") || upper.contains("VALUES")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_sparql_service() {
        let mut service = EnhancedSparqlService::new();
        let context = QueryContext::default();
        let headers = HeaderMap::new();

        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let result = service.process_query(query, context, &headers).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_query_validation() {
        assert!(validate_sparql_query("SELECT ?s WHERE { ?s ?p ?o }").is_ok());
        assert!(validate_sparql_query("").is_err());
    }

    #[test]
    fn test_feature_detection() {
        assert!(contains_aggregation_functions(
            "SELECT (COUNT(?s) as ?count) WHERE { ?s ?p ?o }"
        ));
        assert!(!contains_aggregation_functions(
            "SELECT ?s WHERE { ?s ?p ?o }"
        ));

        assert!(contains_property_paths(
            "SELECT ?s WHERE { ?s foaf:knows+ ?friend }"
        ));
        assert!(!contains_property_paths(
            "SELECT ?s WHERE { ?s foaf:knows ?friend }"
        ));

        assert!(contains_sparql_star_features(
            "SELECT ?s WHERE { <<?s ?p ?o>> ?meta ?value }"
        ));
        assert!(!contains_sparql_star_features(
            "SELECT ?s WHERE { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_content_negotiation() {
        let service = EnhancedSparqlService::new();
        let mut headers = HeaderMap::new();

        headers.insert("accept", "application/json".parse().unwrap());
        let content_type = service.negotiate_content_type(&headers);
        assert_eq!(content_type, "application/json");
    }
}
