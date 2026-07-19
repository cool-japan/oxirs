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

/// Query inputs for a multimodal search, grouped so that
/// [`sparql_multimodal_search`] stays within clippy's argument-count limit.
#[derive(Debug, Clone, Default)]
pub struct MultimodalSearchQuery {
    /// Optional text/keyword query string
    pub text_query: Option<String>,
    /// Optional vector embedding (comma-separated)
    pub vector_query: Option<String>,
    /// Optional WKT geometry (e.g., "POINT(10.0 20.0)")
    pub spatial_query: Option<String>,
    /// Optional weights \[text, vector, spatial\] (comma-separated)
    pub weights: Option<String>,
    /// Optional fusion strategy: "weighted", "sequential", "cascade", "rankfusion"
    pub strategy: Option<String>,
}

/// Execute multimodal search with multiple modalities
///
/// # Arguments
/// * `query` - Text/vector/spatial query inputs plus weighting/strategy, see
///   [`MultimodalSearchQuery`]
/// * `limit` - Maximum number of results
/// * `config` - Multimodal search configuration
/// * `vector_store` - Optional backing store for the vector modality
///
/// # Returns
/// Vector of multimodal search results with combined scores
///
/// # Real backends
/// * `vector_query` is served by a real [`crate::VectorStore::similarity_search_vector`]
///   call when `vector_store` is `Some`; when it is `None` the vector modality
///   is excluded (empty results, logged warning) rather than fabricated.
/// * `text_query` and `spatial_query` currently have no in-process, stateless
///   full-text or spatial search backend wired into this free function (see
///   [`execute_text_search`] / [`execute_spatial_search`] docs), so they are
///   always excluded with a logged warning instead of returning fake scores.
pub fn sparql_multimodal_search(
    query: MultimodalSearchQuery,
    limit: usize,
    config: &MultimodalSearchConfig,
    vector_store: Option<&crate::VectorStore>,
) -> Result<Vec<SparqlMultimodalResult>> {
    let MultimodalSearchQuery {
        text_query,
        vector_query,
        spatial_query,
        weights,
        strategy,
    } = query;

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
        execute_vector_search(vector_store, &embedding, limit * 2)?
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

/// Execute text/keyword search.
///
/// This crate's full-text engine ([`crate::hybrid_search::tantivy_searcher::TantivySearcher`])
/// requires a persistent, pre-built index handle (documents must have been
/// indexed ahead of time) that this stateless free function has no way to
/// obtain. Rather than fabricating fake results, the text modality is
/// excluded (empty result set) with a logged warning until a real searcher
/// handle is threaded through the public API.
fn execute_text_search(query: &str, _limit: usize) -> Result<Vec<DocumentScore>> {
    tracing::warn!(
        "sparql_multimodal_search: no full-text search backend is wired up; \
         excluding the text modality from fusion (query = {:?})",
        query
    );
    Ok(Vec::new())
}

/// Execute vector/semantic search against the real [`crate::VectorStore`]
/// when one is supplied; excludes the vector modality (with a logged
/// warning) instead of fabricating results when no store is configured.
fn execute_vector_search(
    vector_store: Option<&crate::VectorStore>,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<DocumentScore>> {
    let Some(store) = vector_store else {
        tracing::warn!(
            "sparql_multimodal_search: vector modality requested but no VectorStore \
             was supplied; excluding it from fusion"
        );
        return Ok(Vec::new());
    };

    let query_vector = crate::Vector::new(embedding.to_vec());
    let results = store.similarity_search_vector(&query_vector, limit)?;

    Ok(results
        .into_iter()
        .enumerate()
        .map(|(rank, (uri, score))| DocumentScore {
            doc_id: uri,
            score,
            rank,
        })
        .collect())
}

/// Execute spatial/geographic search.
///
/// This crate has no in-process spatial index backend (GeoSPARQL/spatial
/// indexing lives in the separate `oxirs-geosparql` crate, which is not a
/// dependency here). Rather than fabricating fake results, the spatial
/// modality is always excluded (empty result set) with a logged warning.
fn execute_spatial_search(wkt: &str, _limit: usize) -> Result<Vec<DocumentScore>> {
    tracing::warn!(
        "sparql_multimodal_search: no spatial search backend is wired up; \
         excluding the spatial modality from fusion (query = {:?})",
        wkt
    );
    Ok(Vec::new())
}

/// Convert SPARQL arguments to multimodal search
pub fn sparql_multimodal_search_from_args(
    args: &[VectorServiceArg],
    config: &MultimodalSearchConfig,
    vector_store: Option<&crate::VectorStore>,
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
            VectorServiceArg::String(s) if text_query.is_none() => {
                text_query = Some(s.clone());
            }
            VectorServiceArg::Number(n) => {
                limit = *n as usize;
            }
            _ => {}
        }
    }

    // Execute search
    let results = sparql_multimodal_search(
        MultimodalSearchQuery {
            text_query,
            vector_query,
            spatial_query,
            weights,
            strategy,
        },
        limit,
        config,
        vector_store,
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
    use anyhow::Result;

    #[test]
    fn test_parse_fusion_strategy_weighted() -> Result<()> {
        let strategy = parse_fusion_strategy(
            Some("weighted"),
            Some("0.5,0.3,0.2"),
            &[0.33, 0.33, 0.34],
            &[0.5, 0.7, 0.8],
        )?;

        match strategy {
            FusionStrategy::Weighted { weights } => {
                assert_eq!(weights.len(), 3);
                assert!((weights[0] - 0.5).abs() < 1e-6);
                assert!((weights[1] - 0.3).abs() < 1e-6);
                assert!((weights[2] - 0.2).abs() < 1e-6);
            }
            _ => panic!("Expected Weighted strategy"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_fusion_strategy_default() -> Result<()> {
        let strategy = parse_fusion_strategy(None, None, &[0.33, 0.33, 0.34], &[0.5, 0.7, 0.8])?;

        match strategy {
            FusionStrategy::RankFusion => {}
            _ => panic!("Expected RankFusion as default"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_normalization() -> Result<()> {
        assert!(matches!(
            parse_normalization("minmax")?,
            NormalizationMethod::MinMax
        ));
        assert!(matches!(
            parse_normalization("zscore")?,
            NormalizationMethod::ZScore
        ));
        assert!(matches!(
            parse_normalization("sigmoid")?,
            NormalizationMethod::Sigmoid
        ));
        Ok(())
    }

    #[test]
    fn test_parse_weights() -> Result<()> {
        let weights = parse_weights("0.4, 0.35, 0.25")?;
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 0.4).abs() < 1e-6);
        assert!((weights[1] - 0.35).abs() < 1e-6);
        assert!((weights[2] - 0.25).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_parse_vector() -> Result<()> {
        let vector = parse_vector("0.1, 0.2, 0.3")?;
        assert_eq!(vector.len(), 3);
        assert!((vector[0] - 0.1).abs() < 1e-6);
        assert!((vector[1] - 0.2).abs() < 1e-6);
        assert!((vector[2] - 0.3).abs() < 1e-6);
        Ok(())
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

    /// Regression test for the P1 finding: `execute_text_search` used to
    /// fabricate fake `DocumentScore`s regardless of what was indexed. It
    /// must now exclude the (unimplemented) text modality instead of
    /// inventing results.
    #[test]
    fn test_execute_text_search_excludes_instead_of_fabricating() -> Result<()> {
        let results = execute_text_search("test query", 10)?;
        assert!(
            results.is_empty(),
            "text search has no real backend and must return no results, not fabricated ones"
        );
        Ok(())
    }

    /// Regression test: `execute_vector_search` must use the real
    /// `VectorStore::similarity_search_vector` path and return actual
    /// indexed results, not synthetic `vector_result_*` placeholders.
    #[test]
    fn test_execute_vector_search_uses_real_store() -> Result<()> {
        use crate::VectorStore;

        let mut store = VectorStore::new();
        store.index_vector("doc_a".to_string(), crate::Vector::new(vec![1.0, 0.0, 0.0]))?;
        store.index_vector("doc_b".to_string(), crate::Vector::new(vec![0.0, 1.0, 0.0]))?;

        let embedding = vec![1.0, 0.0, 0.0];
        let results = execute_vector_search(Some(&store), &embedding, 10)?;

        assert!(!results.is_empty());
        // The closest match to [1,0,0] must be doc_a, not a fabricated ID.
        assert_eq!(results[0].doc_id, "doc_a");
        assert!(!results
            .iter()
            .any(|r| r.doc_id.starts_with("vector_result_")));
        Ok(())
    }

    /// Without a store, the vector modality must be excluded, not fabricated.
    #[test]
    fn test_execute_vector_search_excludes_without_store() -> Result<()> {
        let embedding = vec![0.1, 0.2, 0.3];
        let results = execute_vector_search(None, &embedding, 10)?;
        assert!(results.is_empty());
        Ok(())
    }

    /// Regression test for the P1 finding: `execute_spatial_search` used to
    /// fabricate fake `DocumentScore`s. There is no in-crate spatial backend,
    /// so it must exclude the modality instead.
    #[test]
    fn test_execute_spatial_search_excludes_instead_of_fabricating() -> Result<()> {
        let results = execute_spatial_search("POINT(10.0 20.0)", 10)?;
        assert!(
            results.is_empty(),
            "spatial search has no in-crate backend and must return no results"
        );
        Ok(())
    }

    #[test]
    fn test_sparql_multimodal_search_integration() -> Result<()> {
        use crate::VectorStore;

        let config = MultimodalSearchConfig::default();
        let mut store = VectorStore::new();
        store.index_vector("doc_a".to_string(), crate::Vector::new(vec![0.1, 0.2, 0.3]))?;

        let results = sparql_multimodal_search(
            MultimodalSearchQuery {
                text_query: Some("machine learning".to_string()),
                vector_query: Some("0.1,0.2,0.3".to_string()),
                spatial_query: Some("POINT(10.0 20.0)".to_string()),
                weights: Some("0.4,0.4,0.2".to_string()),
                strategy: Some("rankfusion".to_string()),
            },
            10,
            &config,
            Some(&store),
        )?;

        // Only the real (vector) modality contributes; text/spatial are
        // excluded rather than fabricated, so results come solely from the
        // indexed vector store.
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
        assert_eq!(results[0].uri, "doc_a");
        Ok(())
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
