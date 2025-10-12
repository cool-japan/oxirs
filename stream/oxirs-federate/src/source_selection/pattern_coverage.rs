//! Pattern coverage analyzer implementation
//!
//! Analyzes which services can answer specific triple patterns

use crate::source_selection::types::*;
use crate::ServiceRegistry;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

impl Default for PatternCoverageAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternCoverageAnalyzer {
    pub fn new() -> Self {
        Self {
            coverage_cache: Arc::new(RwLock::new(HashMap::new())),
            service_statistics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn analyze_coverage(
        &self,
        patterns: &[TriplePattern],
        registry: &ServiceRegistry,
    ) -> Result<Vec<PatternCoverageResult>> {
        let mut results = Vec::new();

        for pattern in patterns {
            let pattern_key = format!(
                "{}:{}:{}",
                pattern.subject, pattern.predicate, pattern.object
            );

            // Check cache first
            if let Some(cached) = self.coverage_cache.read().await.get(&pattern_key) {
                results.push(cached.clone());
                continue;
            }

            // Analyze coverage for this pattern
            let mut covering_sources = Vec::new();
            let services: Vec<_> = registry.get_all_services().into_iter().collect();

            for service in &services {
                if let Some(stats) = self.service_statistics.read().await.get(&service.endpoint) {
                    let coverage = self.calculate_pattern_coverage(pattern, stats).await?;
                    if coverage.coverage_score > 0.0 {
                        covering_sources.push(coverage);
                    }
                }
            }

            // Sort by coverage score
            covering_sources
                .sort_by(|a, b| b.coverage_score.partial_cmp(&a.coverage_score).unwrap());

            let overall_coverage = if covering_sources.is_empty() {
                0.0
            } else {
                covering_sources
                    .iter()
                    .map(|c| c.coverage_score)
                    .sum::<f64>()
                    / covering_sources.len() as f64
            };

            let confidence = self
                .calculate_confidence(&covering_sources, services.len())
                .await?;
            let estimated_result_size = self
                .estimate_result_size(pattern, &covering_sources)
                .await?;

            let result = PatternCoverageResult {
                pattern: pattern.clone(),
                total_sources: services.len(),
                covering_sources,
                coverage_score: overall_coverage,
                confidence,
                estimated_result_size,
            };

            // Cache the result
            self.coverage_cache
                .write()
                .await
                .insert(pattern_key, result.clone());
            results.push(result);
        }

        Ok(results)
    }

    async fn calculate_pattern_coverage(
        &self,
        pattern: &TriplePattern,
        stats: &ServiceStatistics,
    ) -> Result<SourceCoverage> {
        let mut score = 0.0;

        // Check predicate coverage
        if let Some(freq) = stats.predicate_frequency.get(&pattern.predicate) {
            score += (*freq as f64) / (stats.total_triples as f64);
        }

        // Check subject coverage (if not a variable)
        if !pattern.subject.starts_with('?') {
            if let Some(freq) = stats.subject_frequency.get(&pattern.subject) {
                score += (*freq as f64) / (stats.total_triples as f64) * 0.5;
            }
        }

        // Check object coverage (if not a variable)
        if !pattern.object.starts_with('?') {
            if let Some(freq) = stats.object_frequency.get(&pattern.object) {
                score += (*freq as f64) / (stats.total_triples as f64) * 0.5;
            }
        }

        Ok(SourceCoverage {
            service_endpoint: "".to_string(), // Will be filled by caller
            coverage_score: score.min(1.0),
            selectivity: score * 0.8,
            estimated_results: (score * stats.total_triples as f64) as u64,
            data_quality: stats.data_quality_score,
            response_time_estimate: self.estimate_response_time(score, stats).await?,
        })
    }

    pub async fn update_service_statistics(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        let mut predicate_freq = HashMap::new();
        let mut subject_freq = HashMap::new();
        let mut object_freq = HashMap::new();

        for (s, p, o) in triples {
            *predicate_freq.entry(p.clone()).or_insert(0) += 1;
            *subject_freq.entry(s.clone()).or_insert(0) += 1;
            *object_freq.entry(o.clone()).or_insert(0) += 1;
        }

        let stats = ServiceStatistics {
            total_triples: triples.len() as u64,
            unique_predicates: predicate_freq.len() as u64,
            unique_subjects: subject_freq.len() as u64,
            unique_objects: object_freq.len() as u64,
            predicate_frequency: predicate_freq,
            subject_frequency: subject_freq,
            object_frequency: object_freq,
            last_updated: Utc::now(),
            data_quality_score: self.assess_data_quality(triples).await?,
        };

        self.service_statistics
            .write()
            .await
            .insert(service_endpoint.to_string(), stats);

        // Clear cache to force recalculation
        self.coverage_cache.write().await.clear();

        Ok(())
    }

    /// Calculate confidence score for coverage results
    async fn calculate_confidence(
        &self,
        covering_sources: &[SourceCoverage],
        total_services: usize,
    ) -> Result<f64> {
        if covering_sources.is_empty() {
            return Ok(0.0);
        }

        let coverage_ratio = covering_sources.len() as f64 / total_services.max(1) as f64;
        let avg_coverage = covering_sources
            .iter()
            .map(|c| c.coverage_score)
            .sum::<f64>()
            / covering_sources.len() as f64;

        // Confidence based on both coverage quality and source availability
        let confidence = (avg_coverage * 0.7) + (coverage_ratio * 0.3);
        Ok(confidence.min(1.0))
    }

    /// Estimate result size for a pattern
    async fn estimate_result_size(
        &self,
        _pattern: &TriplePattern,
        covering_sources: &[SourceCoverage],
    ) -> Result<u64> {
        if covering_sources.is_empty() {
            return Ok(0);
        }

        let total_estimated = covering_sources
            .iter()
            .map(|c| c.estimated_results)
            .sum::<u64>();

        // Apply deduplication factor (assume 20% overlap between sources)
        Ok((total_estimated as f64 * 0.8) as u64)
    }

    /// Estimate response time based on coverage and statistics
    async fn estimate_response_time(
        &self,
        coverage_score: f64,
        stats: &ServiceStatistics,
    ) -> Result<u64> {
        // Base response time of 100ms
        let base_time = 100.0;

        // Factor in data size (more data = longer response)
        let size_factor = (stats.total_triples as f64).log10() / 6.0; // Log scale

        // Factor in coverage (better coverage = faster response)
        let coverage_factor = 1.0 / (coverage_score + 0.1);

        // Factor in data quality (better quality = more predictable performance)
        let quality_factor = 2.0 - stats.data_quality_score;

        let estimated_time = base_time * size_factor * coverage_factor * quality_factor;
        Ok(estimated_time.max(10.0) as u64) // Minimum 10ms
    }

    /// Assess data quality of triples
    async fn assess_data_quality(&self, triples: &[(String, String, String)]) -> Result<f64> {
        if triples.is_empty() {
            return Ok(0.0);
        }

        let mut valid_uris = 0;
        let mut total_uris = 0;
        let mut non_empty_values = 0;

        for (s, p, o) in triples {
            // Check for empty values
            if !s.trim().is_empty() && !p.trim().is_empty() && !o.trim().is_empty() {
                non_empty_values += 1;
            }

            // Check URI validity (simple heuristic)
            for value in [s, p, o] {
                if value.starts_with("http://") || value.starts_with("https://") {
                    total_uris += 1;
                    if value.contains("://") && value.len() > 10 {
                        valid_uris += 1;
                    }
                }
            }
        }

        // Calculate completeness ratio
        let completeness = non_empty_values as f64 / triples.len() as f64;

        // Calculate URI validity ratio
        let uri_validity = if total_uris > 0 {
            valid_uris as f64 / total_uris as f64
        } else {
            1.0 // No URIs to validate
        };

        // Combine metrics
        let quality_score = (completeness * 0.6) + (uri_validity * 0.4);

        Ok(quality_score.clamp(0.1, 1.0)) // Clamp between 10% and 100%
    }
}
