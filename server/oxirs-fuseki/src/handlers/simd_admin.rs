//! Admin endpoints for SIMD triple matcher management
//!
//! This module provides HTTP endpoints for managing and monitoring the
//! SIMD-accelerated triple pattern matcher.

use crate::error::FusekiResult;
use crate::simd_triple_matcher::{SimdTripleMatcher, Triple, TriplePattern};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Global SIMD matcher instance
pub type SimdMatcherState = Arc<RwLock<SimdTripleMatcher>>;

/// Request to add triples to the SIMD matcher
#[derive(Debug, Deserialize)]
pub struct AddTriplesRequest {
    pub triples: Vec<TripleData>,
}

/// Triple data for API
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TripleData {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Request to match a triple pattern
#[derive(Debug, Deserialize)]
pub struct MatchPatternRequest {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Response from pattern matching
#[derive(Debug, Serialize)]
pub struct MatchPatternResponse {
    pub matches: Vec<TripleData>,
    pub match_count: usize,
    pub execution_time_ms: f64,
    pub used_simd: bool,
}

/// SIMD matcher statistics response
#[derive(Debug, Serialize)]
pub struct SimdMatcherStats {
    pub total_triples: usize,
    pub total_matches: u64,
    pub simd_accelerated_matches: u64,
    pub fallback_matches: u64,
    pub simd_percentage: f64,
    pub index_sizes: IndexSizesResponse,
}

/// Index sizes for response
#[derive(Debug, Serialize)]
pub struct IndexSizesResponse {
    pub subject_index_size: usize,
    pub predicate_index_size: usize,
    pub object_index_size: usize,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub total_triples: usize,
    pub ready: bool,
}

/// Handler: GET /$/simd/stats
///
/// Get SIMD matcher statistics
pub async fn get_simd_stats(
    State(matcher): State<SimdMatcherState>,
) -> FusekiResult<impl IntoResponse> {
    debug!("Getting SIMD matcher statistics");

    let matcher = matcher
        .read()
        .map_err(|e| crate::error::FusekiError::Internal {
            message: format!("Failed to acquire lock: {}", e),
        })?;

    let stats = matcher.get_statistics();

    let simd_percentage = if stats.total_matches > 0 {
        (stats.simd_accelerated_matches as f64 / stats.total_matches as f64) * 100.0
    } else {
        0.0
    };

    let response = SimdMatcherStats {
        total_triples: stats.total_triples,
        total_matches: stats.total_matches,
        simd_accelerated_matches: stats.simd_accelerated_matches,
        fallback_matches: stats.fallback_matches,
        simd_percentage,
        index_sizes: IndexSizesResponse {
            subject_index_size: stats.index_sizes.subject_index_size,
            predicate_index_size: stats.index_sizes.predicate_index_size,
            object_index_size: stats.index_sizes.object_index_size,
        },
    };

    Ok(Json(response))
}

/// Handler: POST /$/simd/add-triples
///
/// Add triples to the SIMD matcher
pub async fn add_triples(
    State(matcher): State<SimdMatcherState>,
    Json(request): Json<AddTriplesRequest>,
) -> FusekiResult<impl IntoResponse> {
    info!("Adding {} triples to SIMD matcher", request.triples.len());

    let mut matcher = matcher
        .write()
        .map_err(|e| crate::error::FusekiError::Internal {
            message: format!("Failed to acquire lock: {}", e),
        })?;

    let triples: Vec<Triple> = request
        .triples
        .into_iter()
        .map(|t| Triple::new(t.subject, t.predicate, t.object))
        .collect();

    matcher.add_triples(triples);

    let stats = matcher.get_statistics();

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "status": "success",
            "total_triples": stats.total_triples,
            "message": "Triples added successfully"
        })),
    ))
}

/// Handler: POST /$/simd/match
///
/// Match a triple pattern using SIMD acceleration
pub async fn match_pattern(
    State(matcher): State<SimdMatcherState>,
    Json(request): Json<MatchPatternRequest>,
) -> FusekiResult<impl IntoResponse> {
    debug!("Matching pattern: {:?}", request);

    let matcher = matcher
        .read()
        .map_err(|e| crate::error::FusekiError::Internal {
            message: format!("Failed to acquire lock: {}", e),
        })?;

    let pattern = TriplePattern {
        subject: request.subject,
        predicate: request.predicate,
        object: request.object,
    };

    let start = std::time::Instant::now();
    let results = matcher.match_pattern(&pattern)?;
    let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let matches: Vec<TripleData> = results
        .iter()
        .map(|t| TripleData {
            subject: t.subject.clone(),
            predicate: t.predicate.clone(),
            object: t.object.clone(),
        })
        .collect();

    let match_count = matches.len();
    let used_simd = match_count >= 32;

    Ok(Json(MatchPatternResponse {
        matches,
        match_count,
        execution_time_ms,
        used_simd,
    }))
}

/// Handler: DELETE /$/simd/clear
///
/// Clear all triples from the SIMD matcher
pub async fn clear_matcher(
    State(matcher): State<SimdMatcherState>,
) -> FusekiResult<impl IntoResponse> {
    info!("Clearing SIMD matcher");

    let mut matcher = matcher
        .write()
        .map_err(|e| crate::error::FusekiError::Internal {
            message: format!("Failed to acquire lock: {}", e),
        })?;

    matcher.clear();

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "success",
            "message": "SIMD matcher cleared"
        })),
    ))
}

/// Handler: GET /$/simd/health
///
/// Check SIMD matcher health
pub async fn health_check(
    State(matcher): State<SimdMatcherState>,
) -> FusekiResult<impl IntoResponse> {
    let matcher = matcher
        .read()
        .map_err(|e| crate::error::FusekiError::Internal {
            message: format!("Failed to acquire lock: {}", e),
        })?;

    let stats = matcher.get_statistics();

    let response = HealthCheckResponse {
        status: "healthy".to_string(),
        total_triples: stats.total_triples,
        ready: stats.total_triples > 0,
    };

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simd_matcher_stats() {
        let matcher = Arc::new(RwLock::new(SimdTripleMatcher::new()));

        // Add some test triples
        {
            let mut m = matcher.write().expect("lock should not be poisoned");
            for i in 0..10 {
                m.add_triple(Triple::new(
                    format!("s{}", i),
                    "p".to_string(),
                    format!("o{}", i),
                ));
            }
        }

        let response = get_simd_stats(State(matcher)).await.unwrap();
        // Response handling would be tested in integration tests
    }

    #[tokio::test]
    async fn test_add_triples_request() {
        let matcher = Arc::new(RwLock::new(SimdTripleMatcher::new()));

        let request = AddTriplesRequest {
            triples: vec![
                TripleData {
                    subject: "s1".to_string(),
                    predicate: "p1".to_string(),
                    object: "o1".to_string(),
                },
                TripleData {
                    subject: "s2".to_string(),
                    predicate: "p2".to_string(),
                    object: "o2".to_string(),
                },
            ],
        };

        let response = add_triples(State(matcher.clone()), Json(request))
            .await
            .unwrap();

        let m = matcher.read().expect("lock should not be poisoned");
        let stats = m.get_statistics();
        assert_eq!(stats.total_triples, 2);
    }

    #[tokio::test]
    async fn test_match_pattern_request() {
        let matcher = Arc::new(RwLock::new(SimdTripleMatcher::new()));

        // Add triples
        {
            let mut m = matcher.write().expect("lock should not be poisoned");
            m.add_triple(Triple::new(
                "s1".to_string(),
                "p1".to_string(),
                "o1".to_string(),
            ));
            m.add_triple(Triple::new(
                "s2".to_string(),
                "p1".to_string(),
                "o2".to_string(),
            ));
        }

        let request = MatchPatternRequest {
            subject: None,
            predicate: Some("p1".to_string()),
            object: None,
        };

        let response = match_pattern(State(matcher), Json(request)).await.unwrap();
        // Response handling would be tested in integration tests
    }

    #[tokio::test]
    async fn test_clear_matcher() {
        let matcher = Arc::new(RwLock::new(SimdTripleMatcher::new()));

        // Add triples
        {
            let mut m = matcher.write().expect("lock should not be poisoned");
            for i in 0..5 {
                m.add_triple(Triple::new(
                    format!("s{}", i),
                    "p".to_string(),
                    format!("o{}", i),
                ));
            }
        }

        assert_eq!(
            matcher
                .read()
                .expect("lock should not be poisoned")
                .get_statistics()
                .total_triples,
            5
        );

        clear_matcher(State(matcher.clone())).await.unwrap();

        assert_eq!(
            matcher
                .read()
                .expect("lock should not be poisoned")
                .get_statistics()
                .total_triples,
            0
        );
    }

    #[tokio::test]
    async fn test_health_check() {
        let matcher = Arc::new(RwLock::new(SimdTripleMatcher::new()));

        let response = health_check(State(matcher)).await.unwrap();
        // Response handling would be tested in integration tests
    }
}
