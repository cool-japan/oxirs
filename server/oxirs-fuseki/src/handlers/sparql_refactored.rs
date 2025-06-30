//! Refactored SPARQL 1.1/1.2 Protocol implementation with modular architecture
//!
//! This module provides a clean, modular implementation of the SPARQL protocol
//! with all components properly separated for maintainability.

// Re-export all the core functionality from submodules
pub use sparql::aggregation_engine::*;
pub use sparql::bind_processor::*;
pub use sparql::content_types::*;
pub use sparql::core::*;
pub use sparql::optimizers::*;
pub use sparql::service_delegation::*;
pub use sparql::sparql12_features::*;

// Import the submodules
pub mod sparql;

use crate::{
    auth::{AuthUser, Permission},
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
use tracing::{info, instrument};

/// Main SPARQL query endpoint with enhanced features
#[instrument(skip(state))]
pub async fn query_handler(
    Query(params): Query<SparqlQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    info!("Processing SPARQL query request");
    
    // Delegate to the core implementation
    sparql_query(Query(params), State(state), headers, user).await
}

/// Main SPARQL update endpoint with enhanced features
#[instrument(skip(state))]
pub async fn update_handler(
    Form(params): Form<SparqlUpdateParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    info!("Processing SPARQL update request");
    
    // Check permissions first
    if let Some(ref user_info) = user {
        if !user_info.0.permissions.contains(&Permission::SparqlUpdate) {
            return axum::response::Response::builder()
                .status(403)
                .body("Insufficient permissions for SPARQL updates".into())
                .unwrap()
                .into_response();
        }
    }
    
    // Delegate to the core implementation
    sparql_update(Form(params), State(state), headers, user).await
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
            return Err(FusekiError::security("Potential SPARQL injection detected"));
        }

        // 2. Complexity analysis
        let complexity = self.complexity_analyzer.analyze_complexity(query)?;
        if complexity.is_complex {
            tracing::warn!("Complex query detected: {:?}", complexity);
        }

        // 3. Apply optimizations
        let mut optimized_query = query.to_string();
        
        // Apply SPARQL 1.2 features and optimizations
        optimized_query = self.sparql12_features.optimize_query(&optimized_query).await?;
        
        // Apply performance optimizations
        optimized_query = self.performance_optimizer.optimize(&optimized_query)?;
        
        // Process SERVICE clauses if present
        if optimized_query.to_uppercase().contains("SERVICE") {
            optimized_query = self.service_delegator.process_service_clauses(&optimized_query).await?;
        }
        
        // Process aggregations
        if contains_aggregation_functions(&optimized_query) {
            optimized_query = self.aggregation_processor.process_aggregations(&optimized_query)?;
        }
        
        // Process BIND clauses
        if optimized_query.to_uppercase().contains("BIND") {
            optimized_query = self.bind_processor.process_bind_clauses(&optimized_query)?;
        }
        
        // Process VALUES clauses
        if optimized_query.to_uppercase().contains("VALUES") {
            optimized_query = self.values_processor.process_values_clauses(&optimized_query)?;
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
            return Err(FusekiError::security("Potential SPARQL injection detected in update"));
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
    pub fn format_response<T: serde::Serialize>(
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
    sparql::core::validate_sparql_query(query)
}

pub fn contains_aggregation_functions(query: &str) -> bool {
    sparql::sparql12_features::contains_aggregation_functions(query)
}

pub fn contains_sparql_star_features(query: &str) -> bool {
    let upper = query.to_uppercase();
    upper.contains("<<") && upper.contains(">>")
}

pub fn contains_property_paths(query: &str) -> bool {
    sparql::sparql12_features::contains_property_paths(query)
}

pub fn contains_subqueries(query: &str) -> bool {
    sparql::sparql12_features::contains_subqueries(query)
}

pub fn contains_bind_values(query: &str) -> bool {
    sparql::sparql12_features::contains_bind_values(query)
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
        assert!(contains_aggregation_functions("SELECT (COUNT(?s) as ?count) WHERE { ?s ?p ?o }"));
        assert!(!contains_aggregation_functions("SELECT ?s WHERE { ?s ?p ?o }"));
        
        assert!(contains_property_paths("SELECT ?s WHERE { ?s foaf:knows+ ?friend }"));
        assert!(!contains_property_paths("SELECT ?s WHERE { ?s foaf:knows ?friend }"));
        
        assert!(contains_sparql_star_features("SELECT ?s WHERE { <<?s ?p ?o>> ?meta ?value }"));
        assert!(!contains_sparql_star_features("SELECT ?s WHERE { ?s ?p ?o }"));
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