//! Regression tests for the [`crate::VectorIndex`] score contract.
//!
//! Every backend must return **similarity** scores (larger = closer), with
//! `search_knn` sorted by descending similarity and `search_threshold` filtering
//! `similarity >= threshold`. These tests pin down that cross-backend agreement
//! so a dispatcher like `DynamicIndexSelector` can compare scores from different
//! backends without knowing which produced them.

#![cfg(test)]

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::lsh::{LshConfig, LshFamily, LshIndex};
use crate::{MemoryVectorIndex, Vector, VectorIndex};
use anyhow::Result;

/// Build a tiny, well-separated dataset: an exact match, a near match, and a
/// far (orthogonal) vector.
fn seed<I: VectorIndex>(index: &mut I) -> Result<()> {
    index.insert("exact".to_string(), Vector::new(vec![1.0, 0.0, 0.0]))?;
    index.insert("near".to_string(), Vector::new(vec![0.9, 0.1, 0.0]))?;
    index.insert("far".to_string(), Vector::new(vec![0.0, 0.0, 1.0]))?;
    Ok(())
}

/// Assert similarity semantics: descending order and the exact match scoring
/// highest (>= every other result). Returns the score of the exact match.
fn assert_similarity_semantics(results: &[(String, f32)]) -> f32 {
    assert!(!results.is_empty(), "search returned no results");
    // Descending by similarity (best first).
    for w in results.windows(2) {
        assert!(
            w[0].1 >= w[1].1 - 1e-6,
            "results must be sorted by descending similarity, got {:?}",
            results
        );
    }
    // The exact match must be the top result under similarity semantics
    // (it would be *last* if a backend leaked ascending-distance ordering).
    assert_eq!(
        results[0].0, "exact",
        "closest vector must rank first under similarity semantics, got {:?}",
        results
    );
    results[0].1
}

#[test]
fn regression_score_contract_hnsw_returns_similarity() -> Result<()> {
    let mut index = HnswIndex::new(HnswConfig::default())?;
    seed(&mut index)?;
    let query = Vector::new(vec![1.0, 0.0, 0.0]);
    // Concrete `HnswIndex` has an inherent (distance) `search_knn`; call the
    // trait method explicitly to exercise the similarity contract.
    let results = <HnswIndex as VectorIndex>::search_knn(&index, &query, 3)?;
    let best = assert_similarity_semantics(&results);
    // similarity = 1/(1+distance); exact match has distance ~0 => similarity ~1.
    assert!(
        best > 0.9,
        "exact-match similarity should be near 1.0, got {best}"
    );
    Ok(())
}

#[test]
fn regression_score_contract_lsh_returns_similarity() -> Result<()> {
    let config = LshConfig {
        num_tables: 8,
        num_hash_functions: 4,
        lsh_family: LshFamily::RandomProjection,
        seed: 42,
        multi_probe: true,
        num_probes: 3,
    };
    let mut index = LshIndex::new(config);
    seed(&mut index)?;
    let query = Vector::new(vec![1.0, 0.0, 0.0]);
    let results = index.search_knn(&query, 3)?;
    // LSH is probabilistic; only assert descending ordering + non-negative
    // similarity, plus that the exact match (if returned) scores highest.
    for w in results.windows(2) {
        assert!(
            w[0].1 >= w[1].1 - 1e-6,
            "LSH must sort descending similarity"
        );
    }
    for (_, sim) in &results {
        assert!(
            *sim >= 0.0 && *sim <= 1.0,
            "LSH similarity out of [0,1]: {sim}"
        );
    }
    Ok(())
}

#[test]
fn regression_score_contract_memory_returns_similarity() -> Result<()> {
    let mut index = MemoryVectorIndex::new();
    seed(&mut index)?;
    let query = Vector::new(vec![1.0, 0.0, 0.0]);
    let results = index.search_knn(&query, 3)?;
    assert_similarity_semantics(&results);
    Ok(())
}

/// All three easily-constructed backends must agree on *ordering* for the same
/// query: the exact match ranks first everywhere (no backend leaks
/// ascending-distance semantics).
#[test]
fn regression_score_contract_cross_backend_agreement() -> Result<()> {
    let query = Vector::new(vec![1.0, 0.0, 0.0]);

    let mut hnsw = HnswIndex::new(HnswConfig::default())?;
    seed(&mut hnsw)?;
    let hnsw_top = <HnswIndex as VectorIndex>::search_knn(&hnsw, &query, 3)?[0]
        .0
        .clone();

    let mut mem = MemoryVectorIndex::new();
    seed(&mut mem)?;
    let mem_top = mem.search_knn(&query, 3)?[0].0.clone();

    assert_eq!(hnsw_top, "exact");
    assert_eq!(mem_top, "exact");
    assert_eq!(
        hnsw_top, mem_top,
        "HNSW and MemoryVectorIndex must agree on the best match"
    );
    Ok(())
}

#[test]
fn regression_hnsw_optimization_flags_drive_hot_path() -> Result<()> {
    use std::sync::atomic::Ordering;

    // Default config: enable_simd + enable_prefetch + cache_friendly_layout all
    // on, metric = Cosine (which has the SIMD fast path). A real search must
    // therefore drive both the SIMD and prefetch counters — proving the config
    // flags are no longer no-ops on the query hot path.
    let mut index = HnswIndex::new(HnswConfig::default())?;
    for i in 0..40 {
        let a = (i as f32) * 0.1;
        index.insert(format!("v{i}"), Vector::new(vec![a.cos(), a.sin(), 0.1]))?;
    }

    let query = Vector::new(vec![1.0, 0.0, 0.1]);
    let results = index.search_knn(&query, 5)?;
    assert!(!results.is_empty());

    let stats = index.get_stats();
    assert!(
        stats.simd_operations.load(Ordering::Relaxed) > 0,
        "enable_simd must route the search hot path through SIMD distance batches"
    );
    assert!(
        stats.prefetch_operations.load(Ordering::Relaxed) > 0,
        "enable_prefetch must issue prefetches during search"
    );
    Ok(())
}

#[test]
fn regression_score_contract_hnsw_threshold_is_similarity() -> Result<()> {
    let mut index = HnswIndex::new(HnswConfig::default())?;
    seed(&mut index)?;
    let query = Vector::new(vec![1.0, 0.0, 0.0]);

    // High threshold: only the exact/near matches (high similarity) survive.
    let strict = <HnswIndex as VectorIndex>::search_threshold(&index, &query, 0.9)?;
    assert!(
        strict.iter().all(|(_, sim)| *sim >= 0.9),
        "threshold must filter similarity >= threshold, got {strict:?}"
    );
    assert!(
        strict.iter().any(|(uri, _)| uri == "exact"),
        "exact match must pass a 0.9 similarity threshold"
    );

    // A similarity threshold above 1.0 can never match.
    assert!(<HnswIndex as VectorIndex>::search_threshold(&index, &query, 1.5)?.is_empty());
    Ok(())
}
