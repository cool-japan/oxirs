//! SPARQL function bindings for multimodal search
//!
//! This module provides SPARQL integration for multimodal search fusion,
//! allowing queries to combine text, vector, and spatial search modalities.

use super::config::{VectorServiceArg, VectorServiceResult};
use crate::hybrid_search::multimodal_fusion::{
    FusedResult, FusionConfig, FusionStrategy, Modality, MultimodalFusion, NormalizationMethod,
};
use crate::hybrid_search::types::DocumentScore;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Multimodal search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSearchConfig {
    /// Default weights for weighted fusion [text, vector, spatial]
    pub default_weights: Vec<f64>,
    /// Default fusion strategy
    pub default_strategy: String,
    /// Score normalization method
    pub normalization: String,
    /// Cascade thresholds [text, vector, spatial]
    pub cascade_thresholds: Vec<f64>,
}

impl Default for MultimodalSearchConfig {
    fn default() -> Self {
        Self {
            default_weights: vec![0.33, 0.33, 0.34],
            default_strategy: "rankfusion".to_string(),
            normalization: "minmax".to_string(),
            cascade_thresholds: vec![0.5, 0.7, 0.8],
        }
    }
}

/// Multimodal search result for SPARQL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlMultimodalResult {
    /// Resource URI
    pub uri: String,
    /// Combined score
    pub score: f64,
    /// Individual modality scores
    pub text_score: Option<f64>,
    pub vector_score: Option<f64>,
    pub spatial_score: Option<f64>,
}

impl From<FusedResult> for SparqlMultimodalResult {
    fn from(result: FusedResult) -> Self {
        let text_score = result.get_score(Modality::Text);
        let vector_score = result.get_score(Modality::Vector);
        let spatial_score = result.get_score(Modality::Spatial);

        Self {
            uri: result.uri,
            score: result.total_score,
            text_score,
            vector_score,
            spatial_score,
        }
    }
}

/// Execute multimodal search with multiple modalities
///
/// # Arguments
/// * `text_query` - Optional text/keyword query string
/// * `vector_query` - Optional vector embedding (comma-separated)
/// * `spatial_query` - Optional WKT geometry (e.g., "POINT(10.0 20.0)")
/// * `weights` - Optional weights [text, vector, spatial] (comma-separated)
/// * `strategy` - Optional fusion strategy: "weighted", "sequential", "cascade", "rankfusion"
/// * `limit` - Maximum number of results
/// * `config` - Multimodal search configuration
///
/// # Returns
/// Vector of multimodal search results with combined scores
pub fn sparql_multimodal_search(
    text_query: Option<String>,
    vector_query: Option<String>,
    spatial_query: Option<String>,
    weights: Option<String>,
    strategy: Option<String>,
    limit: usize,
    config: &MultimodalSearchConfig,
) -> Result<Vec<SparqlMultimodalResult>> {
    // Parse fusion strategy
    let fusion_strategy = parse_fusion_strategy(
        strategy.as_deref(),
        weights.as_deref(),
        &config.default_weights,
        &config.cascade_thresholds,
    )?;

    // Parse normalization method
    let normalization = parse_normalization(&config.normalization)?;

    // Create fusion engine
    let fusion_config = FusionConfig {
        default_strategy: fusion_strategy.clone(),
        score_normalization: normalization,
    };
    let fusion = MultimodalFusion::new(fusion_config);

    // Execute individual searches
    let text_results = if let Some(query) = text_query {
        execute_text_search(&query, limit * 2)?
    } else {
        Vec::new()
    };

    let vector_results = if let Some(query) = vector_query {
        let embedding = parse_vector(&query)?;
        execute_vector_search(&embedding, limit * 2)?
    } else {
        Vec::new()
    };

    let spatial_results = if let Some(query) = spatial_query {
        execute_spatial_search(&query, limit * 2)?
    } else {
        Vec::new()
    };

    // Fuse results
    let fused = fusion.fuse(
        &text_results,
        &vector_results,
        &spatial_results,
        Some(fusion_strategy),
    )?;

    // Convert to SPARQL results and limit
    let results: Vec<SparqlMultimodalResult> = fused
        .into_iter()
        .take(limit)
        .map(SparqlMultimodalResult::from)
        .collect();

    Ok(results)
}

/// Parse fusion strategy from string
fn parse_fusion_strategy(
    strategy: Option<&str>,
    weights: Option<&str>,
    default_weights: &[f64],
    cascade_thresholds: &[f64],
) -> Result<FusionStrategy> {
    match strategy {
        Some("weighted") => {
            let w = if let Some(weights_str) = weights {
                parse_weights(weights_str)?
            } else {
                default_weights.to_vec()
            };
            Ok(FusionStrategy::Weighted { weights: w })
        }
        Some("sequential") => {
            // Default order: Text → Vector
            Ok(FusionStrategy::Sequential {
                order: vec![Modality::Text, Modality::Vector],
            })
        }
        Some("cascade") => Ok(FusionStrategy::Cascade {
            thresholds: cascade_thresholds.to_vec(),
        }),
        Some("rankfusion") | None => Ok(FusionStrategy::RankFusion),
        Some(s) => anyhow::bail!("Unknown fusion strategy: {}", s),
    }
}

/// Parse normalization method from string
fn parse_normalization(normalization: &str) -> Result<NormalizationMethod> {
    match normalization.to_lowercase().as_str() {
        "minmax" => Ok(NormalizationMethod::MinMax),
        "zscore" => Ok(NormalizationMethod::ZScore),
        "sigmoid" => Ok(NormalizationMethod::Sigmoid),
        _ => anyhow::bail!("Unknown normalization method: {}", normalization),
    }
}

/// Parse weights from comma-separated string
fn parse_weights(weights_str: &str) -> Result<Vec<f64>> {
    weights_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .context("Failed to parse weight value")
        })
        .collect()
}

/// Parse vector embedding from comma-separated string
fn parse_vector(vector_str: &str) -> Result<Vec<f32>> {
    vector_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .context("Failed to parse vector value")
        })
        .collect()
}

/// Execute text/keyword search (placeholder - integrate with actual text search)
fn execute_text_search(query: &str, limit: usize) -> Result<Vec<DocumentScore>> {
    // This is a placeholder implementation
    // In production, integrate with Tantivy or BM25 search
    Ok(vec![
        DocumentScore {
            doc_id: format!("text_result_1_{}", query),
            score: 10.0,
            rank: 0,
        },
        DocumentScore {
            doc_id: format!("text_result_2_{}", query),
            score: 8.0,
            rank: 1,
        },
    ]
    .into_iter()
    .take(limit)
    .collect())
}

/// Execute vector/semantic search (placeholder - integrate with actual vector search)
fn execute_vector_search(embedding: &[f32], limit: usize) -> Result<Vec<DocumentScore>> {
    // This is a placeholder implementation
    // In production, integrate with HNSW or vector index
    Ok(vec![
        DocumentScore {
            doc_id: format!("vector_result_1_dim{}", embedding.len()),
            score: 0.95,
            rank: 0,
        },
        DocumentScore {
            doc_id: format!("vector_result_2_dim{}", embedding.len()),
            score: 0.90,
            rank: 1,
        },
    ]
    .into_iter()
    .take(limit)
    .collect())
}

/// Execute spatial/geographic search (placeholder - integrate with actual spatial search)
fn execute_spatial_search(wkt: &str, limit: usize) -> Result<Vec<DocumentScore>> {
    // This is a placeholder implementation
    // In production, integrate with GeoSPARQL or spatial index
    Ok(vec![
        DocumentScore {
            doc_id: format!("spatial_result_1_{}", wkt),
            score: 0.99,
            rank: 0,
        },
        DocumentScore {
            doc_id: format!("spatial_result_2_{}", wkt),
            score: 0.92,
            rank: 1,
        },
    ]
    .into_iter()
    .take(limit)
    .collect())
}

/// Convert SPARQL arguments to multimodal search
pub fn sparql_multimodal_search_from_args(
    args: &[VectorServiceArg],
    config: &MultimodalSearchConfig,
) -> Result<VectorServiceResult> {
    // Parse arguments
    let mut text_query: Option<String> = None;
    let vector_query: Option<String> = None;
    let spatial_query: Option<String> = None;
    let weights: Option<String> = None;
    let strategy: Option<String> = None;
    let mut limit: usize = 10;

    // Extract named arguments (simplified parsing)
    for arg in args {
        match arg {
            VectorServiceArg::String(s) => {
                if text_query.is_none() {
                    text_query = Some(s.clone());
                }
            }
            VectorServiceArg::Number(n) => {
                limit = *n as usize;
            }
            _ => {}
        }
    }

    // Execute search
    let results = sparql_multimodal_search(
        text_query,
        vector_query,
        spatial_query,
        weights,
        strategy,
        limit,
        config,
    )?;

    // Convert to SPARQL result format
    let similarity_list: Vec<(String, f32)> = results
        .into_iter()
        .map(|r| (r.uri, r.score as f32))
        .collect();

    Ok(VectorServiceResult::SimilarityList(similarity_list))
}

/// Generate SPARQL function definition for multimodal search
pub fn generate_multimodal_sparql_function() -> String {
    r#"
PREFIX vec: <http://oxirs.org/vec#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>

# Multimodal Search Function
# Combines text, vector, and spatial search with intelligent fusion
#
# Usage:
# SELECT ?entity ?score WHERE {
#   ?entity vec:multimodal_search(
#     text: "machine learning conference",
#     vector: "0.1,0.2,0.3,...",
#     spatial: "POINT(10.0 20.0)",
#     weights: "0.4,0.4,0.2",
#     strategy: "rankfusion",
#     limit: 10
#   ) .
#   BIND(vec:score(?entity) AS ?score)
# }
# ORDER BY DESC(?score)
#
# Parameters:
#   - text: Text/keyword query (optional)
#   - vector: Comma-separated embedding values (optional)
#   - spatial: WKT geometry string (optional)
#   - weights: Comma-separated weights [text, vector, spatial] (optional)
#   - strategy: Fusion strategy - "weighted", "sequential", "cascade", "rankfusion" (optional)
#   - limit: Maximum results (default: 10)
#
# Fusion Strategies:
#   - weighted: Linear combination of normalized scores
#   - sequential: Filter with one modality, rank with another
#   - cascade: Progressive filtering (fast → expensive)
#   - rankfusion: Reciprocal Rank Fusion (position-based)
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fusion_strategy_weighted() {
        let strategy = parse_fusion_strategy(
            Some("weighted"),
            Some("0.5,0.3,0.2"),
            &[0.33, 0.33, 0.34],
            &[0.5, 0.7, 0.8],
        )
        .unwrap();

        match strategy {
            FusionStrategy::Weighted { weights } => {
                assert_eq!(weights.len(), 3);
                assert!((weights[0] - 0.5).abs() < 1e-6);
                assert!((weights[1] - 0.3).abs() < 1e-6);
                assert!((weights[2] - 0.2).abs() < 1e-6);
            }
            _ => panic!("Expected Weighted strategy"),
        }
    }

    #[test]
    fn test_parse_fusion_strategy_default() {
        let strategy =
            parse_fusion_strategy(None, None, &[0.33, 0.33, 0.34], &[0.5, 0.7, 0.8]).unwrap();

        match strategy {
            FusionStrategy::RankFusion => {}
            _ => panic!("Expected RankFusion as default"),
        }
    }

    #[test]
    fn test_parse_normalization() {
        assert!(matches!(
            parse_normalization("minmax").unwrap(),
            NormalizationMethod::MinMax
        ));
        assert!(matches!(
            parse_normalization("zscore").unwrap(),
            NormalizationMethod::ZScore
        ));
        assert!(matches!(
            parse_normalization("sigmoid").unwrap(),
            NormalizationMethod::Sigmoid
        ));
    }

    #[test]
    fn test_parse_weights() {
        let weights = parse_weights("0.4, 0.35, 0.25").unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 0.4).abs() < 1e-6);
        assert!((weights[1] - 0.35).abs() < 1e-6);
        assert!((weights[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_parse_vector() {
        let vector = parse_vector("0.1, 0.2, 0.3").unwrap();
        assert_eq!(vector.len(), 3);
        assert!((vector[0] - 0.1).abs() < 1e-6);
        assert!((vector[1] - 0.2).abs() < 1e-6);
        assert!((vector[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_multimodal_search_config_default() {
        let config = MultimodalSearchConfig::default();
        assert_eq!(config.default_weights.len(), 3);
        assert_eq!(config.default_strategy, "rankfusion");
        assert_eq!(config.normalization, "minmax");
        assert_eq!(config.cascade_thresholds.len(), 3);
    }

    #[test]
    fn test_sparql_multimodal_result_conversion() {
        let mut fused = FusedResult::new("test_doc".to_string());
        fused.add_score(Modality::Text, 0.5);
        fused.add_score(Modality::Vector, 0.3);
        fused.calculate_total();

        let sparql_result: SparqlMultimodalResult = fused.into();

        assert_eq!(sparql_result.uri, "test_doc");
        assert!((sparql_result.score - 0.8).abs() < 1e-6);
        assert_eq!(sparql_result.text_score, Some(0.5));
        assert_eq!(sparql_result.vector_score, Some(0.3));
        assert_eq!(sparql_result.spatial_score, None);
    }

    #[test]
    fn test_execute_text_search() {
        let results = execute_text_search("test query", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].doc_id.contains("test query"));
    }

    #[test]
    fn test_execute_vector_search() {
        let embedding = vec![0.1, 0.2, 0.3];
        let results = execute_vector_search(&embedding, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].doc_id.contains("dim3"));
    }

    #[test]
    fn test_execute_spatial_search() {
        let results = execute_spatial_search("POINT(10.0 20.0)", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].doc_id.contains("POINT"));
    }

    #[test]
    fn test_sparql_multimodal_search_integration() {
        let config = MultimodalSearchConfig::default();

        let results = sparql_multimodal_search(
            Some("machine learning".to_string()),
            Some("0.1,0.2,0.3".to_string()),
            Some("POINT(10.0 20.0)".to_string()),
            Some("0.4,0.4,0.2".to_string()),
            Some("rankfusion".to_string()),
            10,
            &config,
        )
        .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_generate_multimodal_sparql_function() {
        let sparql = generate_multimodal_sparql_function();
        assert!(sparql.contains("vec:multimodal_search"));
        assert!(sparql.contains("text:"));
        assert!(sparql.contains("vector:"));
        assert!(sparql.contains("spatial:"));
    }
}
