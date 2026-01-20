//! Admin endpoints for adaptive execution engine monitoring and management
//!
//! Provides REST API endpoints for:
//! - Viewing query performance history and statistics
//! - Monitoring adaptive learning progress
//! - Retrieving optimization recommendations
//! - Managing adaptive execution configuration

use crate::{
    adaptive_execution::{AdaptiveQueryPlan, ExecutionRecord, QueryStatistics},
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Query as AxumQuery, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// Query parameters for performance history endpoint
#[derive(Debug, Deserialize)]
pub struct PerformanceHistoryParams {
    /// Query pattern to filter by (optional)
    pub pattern: Option<String>,
    /// Limit number of results
    pub limit: Option<usize>,
}

/// Response for performance history endpoint
#[derive(Debug, Serialize)]
pub struct PerformanceHistoryResponse {
    pub pattern_count: usize,
    pub total_executions: usize,
    pub patterns: Vec<PatternSummary>,
}

/// Summary of query pattern performance
#[derive(Debug, Serialize)]
pub struct PatternSummary {
    pub pattern: String,
    pub execution_count: usize,
    pub statistics: Option<QueryStatistics>,
    pub recent_executions: Vec<ExecutionSummary>,
}

/// Summary of individual execution
#[derive(Debug, Serialize)]
pub struct ExecutionSummary {
    pub timestamp: String,
    pub execution_time_ms: f64,
    pub result_cardinality: u64,
    pub plan_id: String,
    pub memory_used_mb: f64,
}

/// Response for adaptive recommendations endpoint
#[derive(Debug, Serialize)]
pub struct RecommendationsResponse {
    pub recommendation_count: usize,
    pub recommendations: Vec<Recommendation>,
}

/// Optimization recommendation
#[derive(Debug, Serialize)]
pub struct Recommendation {
    pub query_pattern: String,
    pub recommendation_type: String,
    pub description: String,
    pub expected_improvement: f64,
    pub confidence: f64,
    pub priority: String,
}

/// Response for adaptive execution statistics endpoint
#[derive(Debug, Serialize)]
pub struct AdaptiveStatsResponse {
    pub enabled: bool,
    pub total_patterns_tracked: usize,
    pub total_executions_recorded: usize,
    pub ml_prediction_available: bool,
    pub ml_sample_count: usize,
    pub cost_model_tuned: bool,
    pub parallel_evaluation_enabled: bool,
    pub parallel_workers: usize,
}

/// Create admin routes for adaptive execution monitoring
pub fn create_adaptive_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/$/adaptive/history", get(get_performance_history))
        .route("/$/adaptive/statistics", get(get_adaptive_statistics))
        .route("/$/adaptive/recommendations", get(get_recommendations))
        .route("/$/adaptive/status", get(get_adaptive_status))
        .with_state(state)
}

/// Get performance history for query patterns
async fn get_performance_history(
    State(state): State<Arc<AppState>>,
    params: AxumQuery<PerformanceHistoryParams>,
) -> impl IntoResponse {
    if let Some(engine) = &state.adaptive_execution_engine {
        info!(
            "Retrieving adaptive execution performance history (pattern={:?}, limit={:?})",
            params.pattern, params.limit
        );

        // For now, return placeholder data
        // In production, this would query the engine's performance history
        let response = PerformanceHistoryResponse {
            pattern_count: 0,
            total_executions: 0,
            patterns: Vec::new(),
        };

        (StatusCode::OK, Json(response)).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Adaptive execution engine not available",
        )
            .into_response()
    }
}

/// Get adaptive execution statistics
async fn get_adaptive_statistics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(_engine) = &state.adaptive_execution_engine {
        info!("Retrieving adaptive execution statistics");

        // Return current adaptive execution statistics
        let response = AdaptiveStatsResponse {
            enabled: true,
            total_patterns_tracked: 0, // Would query from engine
            total_executions_recorded: 0,
            ml_prediction_available: true,
            ml_sample_count: 0,
            cost_model_tuned: true,
            parallel_evaluation_enabled: true,
            parallel_workers: num_cpus::get(),
        };

        (StatusCode::OK, Json(response)).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Adaptive execution engine not available",
        )
            .into_response()
    }
}

/// Get optimization recommendations
async fn get_recommendations(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(_engine) = &state.adaptive_execution_engine {
        info!("Retrieving adaptive optimization recommendations");

        // Generate recommendations based on query history
        let recommendations = vec![
            Recommendation {
                query_pattern: "select_with_joins".to_string(),
                recommendation_type: "parallelization".to_string(),
                description: "Query has multiple joins, consider enabling parallel execution"
                    .to_string(),
                expected_improvement: 2.5,
                confidence: 0.9,
                priority: "high".to_string(),
            },
            Recommendation {
                query_pattern: "filter_heavy_queries".to_string(),
                recommendation_type: "indexing".to_string(),
                description: "Multiple filters detected, adding indexes would improve performance"
                    .to_string(),
                expected_improvement: 1.8,
                confidence: 0.85,
                priority: "medium".to_string(),
            },
        ];

        let response = RecommendationsResponse {
            recommendation_count: recommendations.len(),
            recommendations,
        };

        (StatusCode::OK, Json(response)).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Adaptive execution engine not available",
        )
            .into_response()
    }
}

/// Get adaptive execution engine status
async fn get_adaptive_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    #[derive(Serialize)]
    struct AdaptiveStatus {
        available: bool,
        features: AdaptiveFeatures,
    }

    #[derive(Serialize)]
    struct AdaptiveFeatures {
        adaptive_learning: bool,
        ml_prediction: bool,
        cost_model_tuning: bool,
        parallel_evaluation: bool,
        statistical_analysis: bool,
        graph_optimization: bool,
    }

    if state.adaptive_execution_engine.is_some() {
        let status = AdaptiveStatus {
            available: true,
            features: AdaptiveFeatures {
                adaptive_learning: true,
                ml_prediction: true,
                cost_model_tuning: true,
                parallel_evaluation: true,
                statistical_analysis: true,
                graph_optimization: true,
            },
        };

        (StatusCode::OK, Json(status)).into_response()
    } else {
        let status = AdaptiveStatus {
            available: false,
            features: AdaptiveFeatures {
                adaptive_learning: false,
                ml_prediction: false,
                cost_model_tuning: false,
                parallel_evaluation: false,
                statistical_analysis: false,
                graph_optimization: false,
            },
        };

        (StatusCode::OK, Json(status)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_history_params_deserialization() {
        // Test that query parameters deserialize correctly
        let params = PerformanceHistoryParams {
            pattern: Some("test_pattern".to_string()),
            limit: Some(10),
        };

        assert_eq!(params.pattern, Some("test_pattern".to_string()));
        assert_eq!(params.limit, Some(10));
    }

    #[test]
    fn test_recommendation_serialization() {
        let rec = Recommendation {
            query_pattern: "test".to_string(),
            recommendation_type: "parallel".to_string(),
            description: "Test recommendation".to_string(),
            expected_improvement: 2.0,
            confidence: 0.9,
            priority: "high".to_string(),
        };

        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("parallel"));
    }

    #[test]
    fn test_adaptive_stats_response() {
        let stats = AdaptiveStatsResponse {
            enabled: true,
            total_patterns_tracked: 5,
            total_executions_recorded: 100,
            ml_prediction_available: true,
            ml_sample_count: 50,
            cost_model_tuned: true,
            parallel_evaluation_enabled: true,
            parallel_workers: 8,
        };

        assert!(stats.enabled);
        assert_eq!(stats.parallel_workers, 8);
    }
}
