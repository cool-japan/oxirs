//! # Validation Endpoints
//!
//! Apache Fuseki-compatible validation services for SPARQL queries,
//! SPARQL Update operations, IRIs, RDF data, and language tags.
//!
//! ## Endpoints
//!
//! - `POST /$/validate/query` - Validate SPARQL queries
//! - `POST /$/validate/update` - Validate SPARQL Update operations
//! - `POST /$/validate/iri` - Validate IRIs
//! - `POST /$/validate/data` - Validate RDF data
//! - `POST /$/validate/langtag` - Validate BCP 47 language tags

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;

use crate::server::AppState;

// Re-export all types and internal functions from sibling modules
pub use crate::handlers::validation_core::{
    detect_query_type, extract_affected_graphs, extract_error_location, extract_prefixes,
    extract_query_variables, extract_update_operations, format_query, format_update,
    generate_algebra_representation, generate_optimized_algebra, is_deprecated_language,
    normalize_format, parse_language_tag, parse_rdf_data, validate_iris_internal,
    validate_langtags_internal, validate_rdf_data_internal, validate_sparql_query_internal,
    validate_sparql_update_internal, validate_sparql_update_syntax, ParsedLanguageTag,
    RdfParseResult,
};
pub use crate::handlers::validation_types::{
    default_format, default_sparql_syntax, default_true, DataValidationRequest,
    DataValidationResponse, IriValidationParams, IriValidationRequest, IriValidationResponse,
    IriValidationResult, LangTagValidationParams, LangTagValidationRequest,
    LangTagValidationResponse, LangTagValidationResult, PrefixMapping, QueryValidationParams,
    QueryValidationRequest, QueryValidationResponse, UpdateValidationParams,
    UpdateValidationRequest, UpdateValidationResponse, ValidationError, ValidationSummary,
    ValidationWarning,
};

// ============================================================================
// Validation Handlers
// ============================================================================

/// Validate SPARQL query (POST)
pub async fn validate_query(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<QueryValidationRequest>,
) -> Response {
    let result = validate_sparql_query_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate SPARQL query (GET)
pub async fn validate_query_get(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<QueryValidationParams>,
) -> Response {
    let query = match params.query {
        Some(q) => q,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(QueryValidationResponse {
                    valid: false,
                    input: String::new(),
                    formatted: None,
                    query_type: None,
                    algebra: None,
                    algebra_optimized: None,
                    variables: None,
                    prefixes: None,
                    errors: vec![ValidationError {
                        message: "Missing 'query' parameter".to_string(),
                        line: None,
                        column: None,
                        code: Some("MISSING_PARAM".to_string()),
                    }],
                    warnings: vec![],
                }),
            )
                .into_response();
        }
    };

    let request = QueryValidationRequest {
        query,
        syntax: params.syntax.unwrap_or_else(default_sparql_syntax),
        include_algebra: true,
        include_optimized: true,
    };

    let result = validate_sparql_query_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate SPARQL Update (POST)
pub async fn validate_update(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<UpdateValidationRequest>,
) -> Response {
    let result = validate_sparql_update_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate SPARQL Update (GET)
pub async fn validate_update_get(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<UpdateValidationParams>,
) -> Response {
    let update = match params.update {
        Some(u) => u,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(UpdateValidationResponse {
                    valid: false,
                    input: String::new(),
                    formatted: None,
                    operations: vec![],
                    affected_graphs: vec![],
                    errors: vec![ValidationError {
                        message: "Missing 'update' parameter".to_string(),
                        line: None,
                        column: None,
                        code: Some("MISSING_PARAM".to_string()),
                    }],
                    warnings: vec![],
                }),
            )
                .into_response();
        }
    };

    let request = UpdateValidationRequest {
        update,
        syntax: params.syntax.unwrap_or_else(default_sparql_syntax),
    };

    let result = validate_sparql_update_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate IRIs (POST)
pub async fn validate_iri(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<IriValidationRequest>,
) -> Response {
    let result = validate_iris_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate IRIs (GET)
pub async fn validate_iri_get(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<IriValidationParams>,
) -> Response {
    let iris = match params.iri {
        Some(iri_str) => iri_str.split(',').map(|s| s.trim().to_string()).collect(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(IriValidationResponse {
                    results: vec![],
                    summary: ValidationSummary {
                        total: 0,
                        valid: 0,
                        invalid: 0,
                        warnings: 0,
                    },
                }),
            )
                .into_response();
        }
    };

    let request = IriValidationRequest {
        iris,
        check_relative: true,
    };

    let result = validate_iris_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate RDF data (POST)
pub async fn validate_data(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<DataValidationRequest>,
) -> Response {
    let result = validate_rdf_data_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate language tags (POST)
pub async fn validate_langtag(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<LangTagValidationRequest>,
) -> Response {
    let result = validate_langtags_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}

/// Validate language tags (GET)
pub async fn validate_langtag_get(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<LangTagValidationParams>,
) -> Response {
    let tags = match params.tag {
        Some(tag_str) => tag_str.split(',').map(|s| s.trim().to_string()).collect(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(LangTagValidationResponse {
                    results: vec![],
                    summary: ValidationSummary {
                        total: 0,
                        valid: 0,
                        invalid: 0,
                        warnings: 0,
                    },
                }),
            )
                .into_response();
        }
    };

    let request = LangTagValidationRequest { tags };

    let result = validate_langtags_internal(&request);
    (StatusCode::OK, Json(result)).into_response()
}
