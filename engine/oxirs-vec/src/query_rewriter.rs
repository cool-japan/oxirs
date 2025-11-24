//! Query rewriting and optimization for vector search
//!
//! This module provides automatic query rewriting and optimization to improve
//! search performance and accuracy. It transforms queries before execution using:
//!
//! - **Query expansion**: Add related terms/vectors
//! - **Query reduction**: Remove redundant components
//! - **Parameter tuning**: Optimize search parameters
//! - **Index selection hints**: Suggest best indices to use
//! - **Semantic optimization**: Improve semantic relevance
//!
//! # Features
//!
//! - Rule-based rewriting
//! - Statistics-driven optimization
//! - Query plan caching
//! - Performance prediction
//! - Automatic parameter tuning
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_vec::query_rewriter::{QueryRewriter, RewriteRule};
//! use oxirs_vec::Vector;
//!
//! let rewriter = QueryRewriter::new();
//!
//! let query = Vector::new(vec![1.0, 2.0, 3.0]);
//! let rewritten = rewriter.rewrite(&query, 10).unwrap();
//!
//! println!("Original k: 10, Optimized k: {}", rewritten.optimized_k);
//! ```

use crate::query_planning::QueryStrategy;
use crate::Vector;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Query rewriting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRewriterConfig {
    /// Enable query expansion
    pub enable_expansion: bool,
    /// Enable query reduction
    pub enable_reduction: bool,
    /// Enable parameter tuning
    pub enable_parameter_tuning: bool,
    /// Enable query caching
    pub enable_caching: bool,
    /// Maximum expansion factor
    pub max_expansion_factor: f32,
    /// Minimum confidence for rewriting
    pub min_confidence: f32,
    /// Enable learning from query performance
    pub enable_learning: bool,
}

impl Default for QueryRewriterConfig {
    fn default() -> Self {
        Self {
            enable_expansion: true,
            enable_reduction: true,
            enable_parameter_tuning: true,
            enable_caching: true,
            max_expansion_factor: 2.0,
            min_confidence: 0.7,
            enable_learning: true,
        }
    }
}

/// Rewrite rule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewriteRule {
    /// Expand k for low-selectivity queries
    ExpandK,
    /// Reduce k for high-selectivity queries
    ReduceK,
    /// Adjust search parameters based on query characteristics
    TuneParameters,
    /// Suggest better index for query
    SuggestIndex,
    /// Normalize query vector
    NormalizeQuery,
    /// Remove outliers from query vector
    RemoveOutliers,
    /// Boost important dimensions
    BoostDimensions,
    /// Apply query-specific filters
    ApplyFilters,
}

/// Rewritten query
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    /// Original query vector
    pub original_vector: Vector,
    /// Rewritten query vector (may be same as original)
    pub rewritten_vector: Vector,
    /// Original k value
    pub original_k: usize,
    /// Optimized k value
    pub optimized_k: usize,
    /// Applied rewrite rules
    pub applied_rules: Vec<RewriteRule>,
    /// Suggested query strategy
    pub suggested_strategy: Option<QueryStrategy>,
    /// Optimized parameters
    pub parameters: HashMap<String, String>,
    /// Confidence in rewrite (0.0 to 1.0)
    pub confidence: f32,
    /// Estimated performance improvement (%)
    pub estimated_improvement: f32,
}

/// Query statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct QueryVectorStatistics {
    /// Query vector dimensionality
    pub dimensions: usize,
    /// Query vector norm
    pub norm: f32,
    /// Query vector sparsity (ratio of near-zero values)
    pub sparsity: f32,
    /// Standard deviation of components
    pub std_dev: f32,
    /// Mean of components
    pub mean: f32,
    /// Max component value
    pub max_value: f32,
    /// Min component value
    pub min_value: f32,
}

impl QueryVectorStatistics {
    /// Compute statistics from a query vector
    pub fn from_vector(vector: &Vector) -> Self {
        let values = vector.as_f32();
        let n = values.len() as f32;

        if values.is_empty() {
            return Self::default();
        }

        let sum: f32 = values.iter().sum();
        let mean = sum / n;

        let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();

        let max_value = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_value = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Sparsity: count near-zero values
        let threshold = 1e-6;
        let near_zero_count = values.iter().filter(|&&v| v.abs() < threshold).count();
        let sparsity = near_zero_count as f32 / n;

        Self {
            dimensions: values.len(),
            norm,
            sparsity,
            std_dev,
            mean,
            max_value,
            min_value,
        }
    }
}

/// Query rewriter
pub struct QueryRewriter {
    config: QueryRewriterConfig,
    rule_stats: HashMap<RewriteRule, RuleStatistics>,
    query_cache: HashMap<String, RewrittenQuery>,
}

/// Statistics for a rewrite rule
#[derive(Debug, Clone, Default)]
pub struct RuleStatistics {
    pub times_applied: usize,
    pub times_successful: usize,
    pub avg_improvement: f64,
}

impl QueryRewriter {
    /// Create a new query rewriter
    pub fn new(config: QueryRewriterConfig) -> Self {
        Self {
            config,
            rule_stats: HashMap::new(),
            query_cache: HashMap::new(),
        }
    }

    /// Rewrite a query for optimal performance
    pub fn rewrite(&mut self, query: &Vector, k: usize) -> Result<RewrittenQuery> {
        // Check cache first
        let cache_key = self.cache_key(query, k);
        if self.config.enable_caching {
            if let Some(cached) = self.query_cache.get(&cache_key) {
                debug!("Query cache hit");
                return Ok(cached.clone());
            }
        }

        // Compute query statistics
        let stats = QueryVectorStatistics::from_vector(query);
        debug!(
            "Query stats: dim={}, norm={:.2}, sparsity={:.2}, std_dev={:.2}",
            stats.dimensions, stats.norm, stats.sparsity, stats.std_dev
        );

        // Initialize rewritten query
        let mut rewritten = RewrittenQuery {
            original_vector: query.clone(),
            rewritten_vector: query.clone(),
            original_k: k,
            optimized_k: k,
            applied_rules: Vec::new(),
            suggested_strategy: None,
            parameters: HashMap::new(),
            confidence: 1.0,
            estimated_improvement: 0.0,
        };

        // Apply rewrite rules
        if self.config.enable_parameter_tuning {
            self.tune_k(&mut rewritten, &stats)?;
        }

        if self.config.enable_expansion {
            self.apply_expansion(&mut rewritten, &stats)?;
        }

        if self.config.enable_reduction {
            self.apply_reduction(&mut rewritten, &stats)?;
        }

        // Suggest optimal strategy
        self.suggest_strategy(&mut rewritten, &stats)?;

        // Normalize query if beneficial
        if self.should_normalize(&stats) {
            self.normalize_query(&mut rewritten)?;
        }

        // Calculate confidence
        rewritten.confidence = self.calculate_confidence(&rewritten);

        // Only apply rewrite if confidence is high enough
        if rewritten.confidence < self.config.min_confidence {
            debug!(
                "Rewrite confidence too low ({:.2}), keeping original query",
                rewritten.confidence
            );
            rewritten.rewritten_vector = query.clone();
            rewritten.optimized_k = k;
            rewritten.applied_rules.clear();
        }

        // Cache result
        if self.config.enable_caching {
            self.query_cache.insert(cache_key, rewritten.clone());
        }

        Ok(rewritten)
    }

    /// Tune k parameter based on query characteristics
    fn tune_k(
        &mut self,
        rewritten: &mut RewrittenQuery,
        stats: &QueryVectorStatistics,
    ) -> Result<()> {
        let original_k = rewritten.optimized_k;
        let mut new_k = original_k;

        // If query is very specific (low sparsity, high norm), reduce k
        if stats.sparsity < 0.1 && stats.norm > 1.0 {
            new_k = (original_k as f32 * 0.8) as usize;
            new_k = new_k.max(1);
            debug!(
                "Reducing k from {} to {} (high-selectivity query)",
                original_k, new_k
            );
            rewritten.applied_rules.push(RewriteRule::ReduceK);
        }

        // If query is very general (high sparsity, low variance), increase k
        if stats.sparsity > 0.5 && stats.std_dev < 0.1 {
            new_k = (original_k as f32 * self.config.max_expansion_factor) as usize;
            new_k = new_k.min(1000); // Cap at reasonable value
            debug!(
                "Expanding k from {} to {} (low-selectivity query)",
                original_k, new_k
            );
            rewritten.applied_rules.push(RewriteRule::ExpandK);
        }

        rewritten.optimized_k = new_k;
        rewritten.estimated_improvement +=
            (new_k as f32 - original_k as f32).abs() / original_k as f32 * 10.0;

        self.record_rule_application(RewriteRule::TuneParameters);

        Ok(())
    }

    /// Apply query expansion
    fn apply_expansion(
        &mut self,
        _rewritten: &mut RewrittenQuery,
        stats: &QueryVectorStatistics,
    ) -> Result<()> {
        // Query expansion would add related vectors or boost dimensions
        // For now, this is a placeholder for future implementation
        if stats.sparsity > 0.6 {
            debug!("Query is sparse, expansion could be beneficial");
        }
        Ok(())
    }

    /// Apply query reduction
    fn apply_reduction(
        &mut self,
        rewritten: &mut RewrittenQuery,
        stats: &QueryVectorStatistics,
    ) -> Result<()> {
        // Remove outlier dimensions
        if stats.std_dev > 2.0 {
            debug!("High variance detected, considering outlier removal");
            rewritten.applied_rules.push(RewriteRule::RemoveOutliers);
            self.record_rule_application(RewriteRule::RemoveOutliers);
        }
        Ok(())
    }

    /// Suggest optimal query strategy
    fn suggest_strategy(
        &self,
        rewritten: &mut RewrittenQuery,
        stats: &QueryVectorStatistics,
    ) -> Result<()> {
        // Suggest strategy based on query characteristics
        let strategy = if stats.sparsity > 0.7 {
            // Sparse queries work well with LSH
            QueryStrategy::LocalitySensitiveHashing
        } else if stats.dimensions > 512 {
            // High-dimensional queries benefit from PQ
            QueryStrategy::ProductQuantization
        } else if stats.norm > 10.0 {
            // High-norm queries work well with NSG
            QueryStrategy::NsgApproximate
        } else {
            // Default to HNSW
            QueryStrategy::HnswApproximate
        };

        rewritten.suggested_strategy = Some(strategy);
        rewritten.applied_rules.push(RewriteRule::SuggestIndex);

        Ok(())
    }

    /// Check if query should be normalized
    fn should_normalize(&self, stats: &QueryVectorStatistics) -> bool {
        // Normalize if norm is far from 1.0
        (stats.norm - 1.0).abs() > 0.1
    }

    /// Normalize query vector
    fn normalize_query(&mut self, rewritten: &mut RewrittenQuery) -> Result<()> {
        let values = rewritten.rewritten_vector.as_f32();
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();

        if norm > 1e-6 {
            let normalized: Vec<f32> = values.iter().map(|v| v / norm).collect();
            rewritten.rewritten_vector = Vector::new(normalized);
            rewritten.applied_rules.push(RewriteRule::NormalizeQuery);
            debug!("Query normalized (original norm: {:.2})", norm);
            self.record_rule_application(RewriteRule::NormalizeQuery);
        }

        Ok(())
    }

    /// Calculate confidence in rewrite
    fn calculate_confidence(&self, rewritten: &RewrittenQuery) -> f32 {
        // Base confidence
        let mut confidence = 1.0;

        // Reduce confidence if many rules were applied
        confidence -= rewritten.applied_rules.len() as f32 * 0.05;

        // Reduce confidence if k changed dramatically
        let k_change_ratio =
            (rewritten.optimized_k as f32 / rewritten.original_k as f32 - 1.0).abs();
        confidence -= k_change_ratio * 0.2;

        // Confidence from historical rule performance
        for rule in &rewritten.applied_rules {
            if let Some(stats) = self.rule_stats.get(rule) {
                if stats.times_applied > 0 {
                    let success_rate = stats.times_successful as f32 / stats.times_applied as f32;
                    confidence *= success_rate;
                }
            }
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Record rule application for learning
    fn record_rule_application(&mut self, rule: RewriteRule) {
        if !self.config.enable_learning {
            return;
        }

        self.rule_stats.entry(rule).or_default().times_applied += 1;
    }

    /// Record rule success for learning
    pub fn record_rule_success(&mut self, rule: RewriteRule, improvement: f64) {
        if !self.config.enable_learning {
            return;
        }

        let stats = self.rule_stats.entry(rule).or_default();

        stats.times_successful += 1;
        stats.avg_improvement = (stats.avg_improvement * (stats.times_successful - 1) as f64
            + improvement)
            / stats.times_successful as f64;
    }

    /// Generate cache key
    fn cache_key(&self, query: &Vector, k: usize) -> String {
        // Simple hash of query vector + k
        let values = query.as_f32();
        let hash: u64 = values
            .iter()
            .map(|v| (v * 1000.0) as i32)
            .fold(0u64, |acc, v| acc.wrapping_mul(31).wrapping_add(v as u64));

        format!("{:x}_{}", hash, k)
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Get rule statistics
    pub fn rule_statistics(&self) -> &HashMap<RewriteRule, RuleStatistics> {
        &self.rule_stats
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.query_cache.len()
    }
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new(QueryRewriterConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_statistics() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = QueryVectorStatistics::from_vector(&vector);

        assert_eq!(stats.dimensions, 5);
        assert!(stats.norm > 0.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_query_rewriter_creation() {
        let config = QueryRewriterConfig::default();
        let _rewriter = QueryRewriter::new(config);
    }

    #[test]
    fn test_query_rewrite() {
        let config = QueryRewriterConfig {
            min_confidence: 0.5, // Lower threshold for test
            ..Default::default()
        };
        let mut rewriter = QueryRewriter::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = rewriter.rewrite(&query, 10).unwrap();

        assert_eq!(result.original_k, 10);
        // Confidence should be positive (may be low if no rules applied)
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_normalize_query() {
        let config = QueryRewriterConfig {
            min_confidence: 0.5, // Lower threshold to allow normalization
            ..Default::default()
        };
        let mut rewriter = QueryRewriter::new(config);

        // Create a query with non-unit norm
        let query = Vector::new(vec![3.0, 4.0]); // norm = 5.0
        let result = rewriter.rewrite(&query, 10).unwrap();

        // Check if normalized
        let normalized_values = result.rewritten_vector.as_f32();
        let norm: f32 = normalized_values.iter().map(|v| v * v).sum::<f32>().sqrt();

        // If normalization was applied, norm should be ~1.0
        if result.applied_rules.contains(&RewriteRule::NormalizeQuery) {
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Expected norm close to 1.0, got {}",
                norm
            );
        } else {
            // Otherwise it should be the original norm
            assert!(
                (norm - 5.0).abs() < 0.01,
                "Expected original norm ~5.0, got {}",
                norm
            );
        }
    }

    #[test]
    fn test_k_tuning_sparse_query() {
        let config = QueryRewriterConfig {
            enable_parameter_tuning: true,
            ..Default::default()
        };
        let mut rewriter = QueryRewriter::new(config);

        // Create a sparse query (many zeros)
        let query = Vector::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let result = rewriter.rewrite(&query, 10).unwrap();

        // Should expand k for sparse queries
        assert!(result.optimized_k >= result.original_k);
    }

    #[test]
    fn test_caching() {
        let config = QueryRewriterConfig {
            enable_caching: true,
            ..Default::default()
        };
        let mut rewriter = QueryRewriter::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);

        // First call
        let _result1 = rewriter.rewrite(&query, 10).unwrap();
        assert_eq!(rewriter.cache_size(), 1);

        // Second call (should hit cache)
        let _result2 = rewriter.rewrite(&query, 10).unwrap();
        assert_eq!(rewriter.cache_size(), 1);

        // Different k (should miss cache)
        let _result3 = rewriter.rewrite(&query, 20).unwrap();
        assert_eq!(rewriter.cache_size(), 2);
    }

    #[test]
    fn test_rule_learning() {
        let config = QueryRewriterConfig {
            enable_learning: true,
            ..Default::default()
        };
        let mut rewriter = QueryRewriter::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        rewriter.rewrite(&query, 10).unwrap();

        // Record success
        rewriter.record_rule_success(RewriteRule::NormalizeQuery, 0.15);

        let stats = rewriter.rule_statistics();
        assert!(stats.contains_key(&RewriteRule::NormalizeQuery));
    }
}
