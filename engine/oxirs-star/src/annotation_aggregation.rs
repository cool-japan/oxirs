//! Annotation aggregation and rollup for RDF-star
//!
//! This module provides sophisticated aggregation capabilities for combining,
//! summarizing, and rolling up annotation data across multiple sources and time periods.
//!
//! # Features
//!
//! - **Statistical aggregation** - Mean, median, weighted averages for confidence scores
//! - **Evidence consolidation** - Combine evidence from multiple sources
//! - **Temporal rollup** - Aggregate annotations by time windows
//! - **Source grouping** - Group by annotation source or provenance
//! - **Conflict resolution** - Handle contradictory annotations
//! - **SciRS2 optimization** - Parallel aggregation for large datasets
//!
//! # Use Cases
//!
//! - **Data consolidation** - Merge annotations from multiple knowledge sources
//! - **Trend analysis** - Analyze confidence scores over time
//! - **Source reliability** - Compare annotation quality across sources
//! - **Compression** - Reduce storage by aggregating historical annotations
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::annotation_aggregation::{AnnotationAggregator, AggregationStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut aggregator = AnnotationAggregator::new();
//!
//! // Aggregate by source
//! // let summary = aggregator.aggregate_by_source(&annotations)?;
//!
//! // Temporal rollup (daily)
//! // let daily = aggregator.temporal_rollup(&annotations, Duration::days(1))?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, span, Level};

// SciRS2 imports for numerical operations (SCIRS2 POLICY)
// Note: par_chunks available for future parallel aggregation optimization

use crate::annotations::{EvidenceItem, TripleAnnotation};
use crate::model::StarTriple;
use crate::StarResult;

/// Strategy for aggregating annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple mean of confidence scores
    Mean,
    /// Weighted mean by evidence strength
    WeightedMean,
    /// Median confidence score
    Median,
    /// Maximum confidence (optimistic)
    Maximum,
    /// Minimum confidence (pessimistic)
    Minimum,
    /// Bayesian combination
    Bayesian,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Keep annotation with highest confidence
    HighestConfidence,
    /// Keep most recent annotation
    MostRecent,
    /// Keep annotation from most trusted source
    MostTrustedSource,
    /// Merge all conflicting annotations
    MergeAll,
    /// Flag as conflicting and require manual resolution
    FlagConflict,
}

/// Aggregated annotation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedAnnotation {
    /// Aggregated confidence score
    pub confidence: f64,

    /// Number of annotations aggregated
    pub annotation_count: usize,

    /// Sources contributing to this aggregation
    pub sources: Vec<String>,

    /// Consolidated evidence
    pub evidence: Vec<EvidenceItem>,

    /// Timestamp range (earliest, latest)
    pub timestamp_range: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// Aggregation strategy used
    pub strategy: AggregationStrategy,

    /// Variance in confidence scores
    pub confidence_variance: f64,

    /// Conflicts detected
    pub has_conflicts: bool,
}

/// Source-level aggregation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAggregation {
    /// Source identifier
    pub source: String,

    /// Number of annotations
    pub count: usize,

    /// Average confidence
    pub avg_confidence: f64,

    /// Median confidence
    pub median_confidence: f64,

    /// Confidence standard deviation
    pub std_dev: f64,

    /// Evidence count
    pub evidence_count: usize,
}

/// Temporal aggregation bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBucket {
    /// Time window start
    pub start: DateTime<Utc>,

    /// Time window end
    pub end: DateTime<Utc>,

    /// Aggregated annotation for this window
    pub aggregation: AggregatedAnnotation,
}

/// Annotation aggregator
pub struct AnnotationAggregator {
    /// Default aggregation strategy
    default_strategy: AggregationStrategy,

    /// Default conflict resolution
    conflict_resolution: ConflictResolution,

    /// Cache of source reputations
    source_reputations: HashMap<String, f64>,

    /// Statistics
    stats: AggregationStatistics,
}

/// Statistics for aggregation operations
#[derive(Debug, Clone, Default)]
pub struct AggregationStatistics {
    /// Total aggregations performed
    pub aggregations_count: usize,

    /// Total annotations processed
    pub annotations_processed: usize,

    /// Conflicts detected
    pub conflicts_detected: usize,

    /// Conflicts resolved
    pub conflicts_resolved: usize,
}

impl AnnotationAggregator {
    /// Create a new annotation aggregator
    pub fn new() -> Self {
        Self {
            default_strategy: AggregationStrategy::WeightedMean,
            conflict_resolution: ConflictResolution::MostRecent,
            source_reputations: HashMap::new(),
            stats: AggregationStatistics::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_strategy(strategy: AggregationStrategy) -> Self {
        Self {
            default_strategy: strategy,
            ..Self::new()
        }
    }

    /// Aggregate multiple annotations for the same triple
    pub fn aggregate(
        &mut self,
        annotations: &[TripleAnnotation],
        strategy: Option<AggregationStrategy>,
    ) -> StarResult<AggregatedAnnotation> {
        let span = span!(Level::DEBUG, "aggregate_annotations");
        let _enter = span.enter();

        if annotations.is_empty() {
            return Err(crate::StarError::invalid_quoted_triple(
                "Cannot aggregate empty annotation list",
            ));
        }

        let strategy = strategy.unwrap_or(self.default_strategy);

        // Extract confidence scores
        let confidences: Vec<f64> = annotations.iter().filter_map(|a| a.confidence).collect();

        if confidences.is_empty() {
            return Err(crate::StarError::invalid_quoted_triple(
                "No confidence scores to aggregate",
            ));
        }

        // Calculate aggregated confidence
        let aggregated_confidence = match strategy {
            AggregationStrategy::Mean => self.mean(&confidences),
            AggregationStrategy::WeightedMean => self.weighted_mean(annotations),
            AggregationStrategy::Median => self.median(&confidences),
            AggregationStrategy::Maximum => confidences
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            AggregationStrategy::Minimum => {
                confidences.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            AggregationStrategy::Bayesian => self.bayesian_combination(&confidences),
        };

        // Collect sources
        let sources: Vec<String> = annotations
            .iter()
            .filter_map(|a| a.source.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Consolidate evidence
        let mut all_evidence = Vec::new();
        for ann in annotations {
            all_evidence.extend(ann.evidence.clone());
        }

        // Deduplicate evidence by reference
        let unique_evidence = self.deduplicate_evidence(all_evidence);

        // Find timestamp range
        let timestamps: Vec<DateTime<Utc>> =
            annotations.iter().filter_map(|a| a.timestamp).collect();

        let timestamp_range = if timestamps.is_empty() {
            None
        } else {
            let earliest = timestamps.iter().min().unwrap();
            let latest = timestamps.iter().max().unwrap();
            Some((*earliest, *latest))
        };

        // Calculate variance
        let variance = if confidences.len() > 1 {
            let mean = self.mean(&confidences);
            let sum_squared_diff: f64 = confidences.iter().map(|&c| (c - mean).powi(2)).sum();
            sum_squared_diff / (confidences.len() - 1) as f64
        } else {
            0.0
        };

        // Check for conflicts
        let has_conflicts = self.detect_conflicts(annotations);
        if has_conflicts {
            self.stats.conflicts_detected += 1;
        }

        // Update statistics
        self.stats.aggregations_count += 1;
        self.stats.annotations_processed += annotations.len();

        debug!(
            "Aggregated {} annotations into confidence {:.3}",
            annotations.len(),
            aggregated_confidence
        );

        Ok(AggregatedAnnotation {
            confidence: aggregated_confidence,
            annotation_count: annotations.len(),
            sources,
            evidence: unique_evidence,
            timestamp_range,
            strategy,
            confidence_variance: variance,
            has_conflicts,
        })
    }

    /// Aggregate annotations grouped by source
    pub fn aggregate_by_source(
        &mut self,
        annotations: &[(StarTriple, TripleAnnotation)],
    ) -> HashMap<String, SourceAggregation> {
        let span = span!(Level::INFO, "aggregate_by_source");
        let _enter = span.enter();

        let mut source_groups: HashMap<String, Vec<TripleAnnotation>> = HashMap::new();

        for (_, annotation) in annotations {
            if let Some(ref source) = annotation.source {
                source_groups
                    .entry(source.clone())
                    .or_default()
                    .push(annotation.clone());
            }
        }

        let mut result = HashMap::new();

        for (source, anns) in source_groups {
            let confidences: Vec<f64> = anns.iter().filter_map(|a| a.confidence).collect();

            if confidences.is_empty() {
                continue;
            }

            let avg_confidence = self.mean(&confidences);
            let median_confidence = self.median(&confidences);
            let std_dev = self.standard_deviation(&confidences);
            let evidence_count: usize = anns.iter().map(|a| a.evidence.len()).sum();

            result.insert(
                source.clone(),
                SourceAggregation {
                    source,
                    count: anns.len(),
                    avg_confidence,
                    median_confidence,
                    std_dev,
                    evidence_count,
                },
            );
        }

        info!("Aggregated annotations from {} sources", result.len());
        result
    }

    /// Aggregate annotations by time windows (temporal rollup)
    pub fn temporal_rollup(
        &mut self,
        annotations: &[(StarTriple, TripleAnnotation)],
        window_size: Duration,
    ) -> StarResult<Vec<TemporalBucket>> {
        let span = span!(Level::INFO, "temporal_rollup");
        let _enter = span.enter();

        // Find time range
        let timestamps: Vec<DateTime<Utc>> = annotations
            .iter()
            .filter_map(|(_, a)| a.timestamp)
            .collect();

        if timestamps.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = *timestamps.iter().min().unwrap();
        let end_time = *timestamps.iter().max().unwrap();

        // Create time buckets
        let mut buckets = Vec::new();
        let mut current = start_time;

        while current <= end_time {
            let bucket_end = current + window_size;

            // Collect annotations in this window
            let window_anns: Vec<TripleAnnotation> = annotations
                .iter()
                .filter_map(|(_, a)| {
                    if let Some(ts) = a.timestamp {
                        if ts >= current && ts < bucket_end {
                            Some(a.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if !window_anns.is_empty() {
                let aggregation = self.aggregate(&window_anns, None)?;

                buckets.push(TemporalBucket {
                    start: current,
                    end: bucket_end,
                    aggregation,
                });
            }

            current = bucket_end;
        }

        info!("Created {} temporal buckets", buckets.len());
        Ok(buckets)
    }

    /// Resolve conflicts between annotations
    pub fn resolve_conflict(
        &mut self,
        annotations: &[TripleAnnotation],
    ) -> StarResult<TripleAnnotation> {
        if annotations.is_empty() {
            return Err(crate::StarError::invalid_quoted_triple(
                "No annotations to resolve",
            ));
        }

        if annotations.len() == 1 {
            return Ok(annotations[0].clone());
        }

        let resolved = match self.conflict_resolution {
            ConflictResolution::HighestConfidence => annotations
                .iter()
                .max_by(|a, b| {
                    a.confidence
                        .unwrap_or(0.0)
                        .partial_cmp(&b.confidence.unwrap_or(0.0))
                        .unwrap()
                })
                .unwrap()
                .clone(),
            ConflictResolution::MostRecent => annotations
                .iter()
                .max_by_key(|a| a.timestamp)
                .unwrap()
                .clone(),
            ConflictResolution::MostTrustedSource => {
                let mut best = &annotations[0];
                let mut best_reputation = 0.0;

                for ann in annotations {
                    if let Some(ref source) = ann.source {
                        let reputation =
                            self.source_reputations.get(source).copied().unwrap_or(0.5);
                        if reputation > best_reputation {
                            best_reputation = reputation;
                            best = ann;
                        }
                    }
                }

                best.clone()
            }
            ConflictResolution::MergeAll => {
                // Create aggregated annotation
                let agg = self.aggregate(annotations, None)?;
                let mut merged = annotations[0].clone();
                merged.confidence = Some(agg.confidence);
                merged
            }
            ConflictResolution::FlagConflict => {
                return Err(crate::StarError::invalid_quoted_triple(
                    "Conflicting annotations require manual resolution",
                ));
            }
        };

        self.stats.conflicts_resolved += 1;
        Ok(resolved)
    }

    // Helper methods

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn weighted_mean(&self, annotations: &[TripleAnnotation]) -> f64 {
        let mut sum_weighted = 0.0;
        let mut sum_weights = 0.0;

        for ann in annotations {
            if let Some(confidence) = ann.confidence {
                // Weight by evidence strength
                let weight = if ann.evidence.is_empty() {
                    1.0
                } else {
                    ann.evidence.iter().map(|e| e.strength).sum::<f64>() / ann.evidence.len() as f64
                };

                sum_weighted += confidence * weight;
                sum_weights += weight;
            }
        }

        if sum_weights > 0.0 {
            sum_weighted / sum_weights
        } else {
            0.0
        }
    }

    fn median(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn standard_deviation(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    fn bayesian_combination(&self, confidences: &[f64]) -> f64 {
        // Use odds ratio combination
        let mut odds = 1.0;

        for &confidence in confidences {
            if confidence > 0.0 && confidence < 1.0 {
                let p = confidence;
                let current_odds = p / (1.0 - p);
                odds *= current_odds;
            }
        }

        // Convert back to probability
        odds / (1.0 + odds)
    }

    fn deduplicate_evidence(&self, evidence: Vec<EvidenceItem>) -> Vec<EvidenceItem> {
        let mut unique = Vec::new();
        let mut seen = HashSet::new();

        for item in evidence {
            let key = format!("{}:{}", item.evidence_type, item.reference);
            if seen.insert(key) {
                unique.push(item);
            }
        }

        unique
    }

    fn detect_conflicts(&self, annotations: &[TripleAnnotation]) -> bool {
        if annotations.len() < 2 {
            return false;
        }

        // Check for significant variance in confidence scores
        let confidences: Vec<f64> = annotations.iter().filter_map(|a| a.confidence).collect();

        if confidences.len() < 2 {
            return false;
        }

        let std_dev = self.standard_deviation(&confidences);
        std_dev > 0.3 // Significant disagreement threshold
    }

    /// Set source reputation
    pub fn set_source_reputation(&mut self, source: String, reputation: f64) {
        self.source_reputations
            .insert(source, reputation.clamp(0.0, 1.0));
    }

    /// Get statistics
    pub fn statistics(&self) -> &AggregationStatistics {
        &self.stats
    }
}

impl Default for AnnotationAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_aggregation() {
        let mut aggregator = AnnotationAggregator::new();

        let annotations = vec![
            TripleAnnotation::new().with_confidence(0.8),
            TripleAnnotation::new().with_confidence(0.9),
            TripleAnnotation::new().with_confidence(0.7),
        ];

        let result = aggregator
            .aggregate(&annotations, Some(AggregationStrategy::Mean))
            .unwrap();

        assert!((result.confidence - 0.8).abs() < 0.01);
        assert_eq!(result.annotation_count, 3);
    }

    #[test]
    fn test_median_aggregation() {
        let mut aggregator = AnnotationAggregator::new();

        let annotations = vec![
            TripleAnnotation::new().with_confidence(0.5),
            TripleAnnotation::new().with_confidence(0.9),
            TripleAnnotation::new().with_confidence(0.7),
        ];

        let result = aggregator
            .aggregate(&annotations, Some(AggregationStrategy::Median))
            .unwrap();

        assert_eq!(result.confidence, 0.7);
    }

    #[test]
    fn test_weighted_mean() {
        let mut aggregator = AnnotationAggregator::new();

        let mut ann1 = TripleAnnotation::new().with_confidence(0.5);
        ann1.evidence.push(EvidenceItem {
            evidence_type: "weak".to_string(),
            reference: "ref1".to_string(),
            strength: 0.3,
            description: None,
        });

        let mut ann2 = TripleAnnotation::new().with_confidence(0.9);
        ann2.evidence.push(EvidenceItem {
            evidence_type: "strong".to_string(),
            reference: "ref2".to_string(),
            strength: 0.9,
            description: None,
        });

        let result = aggregator
            .aggregate(&[ann1, ann2], Some(AggregationStrategy::WeightedMean))
            .unwrap();

        // Weighted mean should be closer to 0.9 due to higher evidence strength
        assert!(result.confidence > 0.7);
    }

    #[test]
    fn test_conflict_detection() {
        let mut aggregator = AnnotationAggregator::new();

        let annotations = vec![
            TripleAnnotation::new().with_confidence(0.2),
            TripleAnnotation::new().with_confidence(0.9),
        ];

        let result = aggregator.aggregate(&annotations, None).unwrap();
        assert!(result.has_conflicts);
    }

    #[test]
    fn test_temporal_rollup() {
        let mut aggregator = AnnotationAggregator::new();

        let base_time = Utc::now();
        let triple = crate::model::StarTriple::new(
            crate::model::StarTerm::iri("http://example.org/s").unwrap(),
            crate::model::StarTerm::iri("http://example.org/p").unwrap(),
            crate::model::StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotations = vec![
            (
                triple.clone(),
                TripleAnnotation {
                    confidence: Some(0.8),
                    timestamp: Some(base_time),
                    ..Default::default()
                },
            ),
            (
                triple.clone(),
                TripleAnnotation {
                    confidence: Some(0.9),
                    timestamp: Some(base_time + Duration::hours(2)),
                    ..Default::default()
                },
            ),
            (
                triple.clone(),
                TripleAnnotation {
                    confidence: Some(0.7),
                    timestamp: Some(base_time + Duration::days(1)),
                    ..Default::default()
                },
            ),
        ];

        let buckets = aggregator
            .temporal_rollup(&annotations, Duration::days(1))
            .unwrap();

        assert!(!buckets.is_empty());
    }

    #[test]
    fn test_source_aggregation() {
        let mut aggregator = AnnotationAggregator::new();

        let triple = crate::model::StarTriple::new(
            crate::model::StarTerm::iri("http://example.org/s").unwrap(),
            crate::model::StarTerm::iri("http://example.org/p").unwrap(),
            crate::model::StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotations = vec![
            (
                triple.clone(),
                TripleAnnotation::new()
                    .with_confidence(0.8)
                    .with_source("source1".to_string()),
            ),
            (
                triple.clone(),
                TripleAnnotation::new()
                    .with_confidence(0.9)
                    .with_source("source1".to_string()),
            ),
            (
                triple.clone(),
                TripleAnnotation::new()
                    .with_confidence(0.7)
                    .with_source("source2".to_string()),
            ),
        ];

        let result = aggregator.aggregate_by_source(&annotations);

        assert_eq!(result.len(), 2);
        assert_eq!(result.get("source1").unwrap().count, 2);
        assert_eq!(result.get("source2").unwrap().count, 1);
    }
}
