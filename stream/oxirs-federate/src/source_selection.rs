//! # Advanced Source Selection Algorithms
//!
//! This module implements sophisticated algorithms for selecting optimal data sources
//! in federated query processing. It includes triple pattern coverage analysis,
//! predicate-based filtering, range-based selection, and ML-driven source prediction.

use anyhow::{anyhow, Result};
use bloom::{BloomFilter, ASMS};
use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{FederatedService, ServiceCapability, ServiceRegistry};

/// Triple pattern for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub graph: Option<String>,
}

/// Range constraint for numeric or temporal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeConstraint {
    pub field: String,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub data_type: RangeDataType,
}

/// Supported data types for range constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeDataType {
    Integer,
    Float,
    DateTime,
    String,
    Uri,
}

/// Comprehensive source selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionConfig {
    pub enable_pattern_coverage: bool,
    pub enable_predicate_filtering: bool,
    pub enable_range_selection: bool,
    pub enable_ml_prediction: bool,
    pub coverage_threshold: f64,
    pub bloom_filter_capacity: usize,
    pub bloom_filter_fp_rate: f64,
    pub ml_confidence_threshold: f64,
    pub max_sources_per_pattern: usize,
}

impl Default for SourceSelectionConfig {
    fn default() -> Self {
        Self {
            enable_pattern_coverage: true,
            enable_predicate_filtering: true,
            enable_range_selection: true,
            enable_ml_prediction: true,
            coverage_threshold: 0.8,
            bloom_filter_capacity: 100000,
            bloom_filter_fp_rate: 0.01,
            ml_confidence_threshold: 0.7,
            max_sources_per_pattern: 10,
        }
    }
}

/// Advanced source selector with multiple algorithms
pub struct AdvancedSourceSelector {
    config: SourceSelectionConfig,
    pattern_analyzer: PatternCoverageAnalyzer,
    predicate_filter: PredicateBasedFilter,
    range_selector: RangeBasedSelector,
    ml_predictor: Option<MLSourcePredictor>,
    statistics: Arc<RwLock<SelectionStatistics>>,
}

/// Pattern coverage analyzer for triple pattern analysis
pub struct PatternCoverageAnalyzer {
    coverage_cache: Arc<RwLock<HashMap<String, PatternCoverageResult>>>,
    service_statistics: Arc<RwLock<HashMap<String, ServiceStatistics>>>,
}

/// Predicate-based filter using Bloom filters
pub struct PredicateBasedFilter {
    service_filters: Arc<RwLock<HashMap<String, ServiceBloomFilters>>>,
    last_update: Arc<RwLock<DateTime<Utc>>>,
}

/// Range-based selector for numeric and temporal constraints
pub struct RangeBasedSelector {
    range_indices: Arc<RwLock<HashMap<String, ServiceRangeIndex>>>,
    temporal_indices: Arc<RwLock<HashMap<String, ServiceTemporalIndex>>>,
}

/// ML-based source predictor
pub struct MLSourcePredictor {
    training_data: Vec<SourcePredictionSample>,
    feature_weights: HashMap<String, f64>,
    prediction_cache: Arc<RwLock<HashMap<String, PredictionResult>>>,
    model_accuracy: f64,
}

/// Service statistics for coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatistics {
    pub total_triples: u64,
    pub unique_predicates: u64,
    pub unique_subjects: u64,
    pub unique_objects: u64,
    pub predicate_frequency: HashMap<String, u64>,
    pub subject_frequency: HashMap<String, u64>,
    pub object_frequency: HashMap<String, u64>,
    pub last_updated: DateTime<Utc>,
    pub data_quality_score: f64,
}

/// Bloom filters for service predicate filtering
pub struct ServiceBloomFilters {
    pub predicate_filter: BloomFilter,
    pub subject_filter: BloomFilter,
    pub object_filter: BloomFilter,
    pub type_filter: BloomFilter,
    pub last_updated: DateTime<Utc>,
    pub estimated_elements: usize,
}

impl std::fmt::Debug for ServiceBloomFilters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceBloomFilters")
            .field("last_updated", &self.last_updated)
            .field("estimated_elements", &self.estimated_elements)
            .finish()
    }
}

impl Clone for ServiceBloomFilters {
    fn clone(&self) -> Self {
        let capacity = (self.estimated_elements.max(1000) as u32);
        Self {
            predicate_filter: BloomFilter::with_rate(0.01, capacity),
            subject_filter: BloomFilter::with_rate(0.01, capacity),
            object_filter: BloomFilter::with_rate(0.01, capacity),
            type_filter: BloomFilter::with_rate(0.01, capacity),
            last_updated: self.last_updated,
            estimated_elements: self.estimated_elements,
        }
    }
}

/// Range index for a service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRangeIndex {
    pub numeric_ranges: HashMap<String, NumericRange>,
    pub string_ranges: HashMap<String, StringRange>,
    pub uri_patterns: HashMap<String, UriPattern>,
    pub last_updated: DateTime<Utc>,
}

/// Temporal index for a service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceTemporalIndex {
    pub datetime_ranges: HashMap<String, DateTimeRange>,
    pub year_ranges: HashMap<String, YearRange>,
    pub temporal_patterns: HashMap<String, TemporalPattern>,
    pub last_updated: DateTime<Utc>,
}

/// Numeric range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericRange {
    pub min_value: f64,
    pub max_value: f64,
    pub count: u64,
    pub sample_values: Vec<f64>,
}

/// String range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringRange {
    pub min_length: usize,
    pub max_length: usize,
    pub common_prefixes: Vec<String>,
    pub sample_values: Vec<String>,
}

/// URI pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UriPattern {
    pub base_uris: Vec<String>,
    pub path_patterns: Vec<String>,
    pub namespace_prefixes: Vec<String>,
}

/// DateTime range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateTimeRange {
    pub earliest: DateTime<Utc>,
    pub latest: DateTime<Utc>,
    pub count: u64,
    pub granularity: TemporalGranularity,
}

/// Year range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearRange {
    pub earliest_year: i32,
    pub latest_year: i32,
    pub year_distribution: HashMap<i32, u64>,
}

/// Temporal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_type: TemporalPatternType,
    pub frequency: u64,
    pub confidence: f64,
}

/// Temporal granularity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalGranularity {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// Types of temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPatternType {
    Sequential,
    Periodic,
    Clustered,
    Random,
}

/// ML training sample for source prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePredictionSample {
    pub query_features: QueryFeatures,
    pub selected_sources: Vec<String>,
    pub actual_performance: PerformanceMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Query features for ML prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub pattern_count: usize,
    pub variable_count: usize,
    pub predicate_types: Vec<String>,
    pub has_ranges: bool,
    pub has_joins: bool,
    pub complexity_score: f64,
    pub selectivity_estimate: f64,
}

/// Performance metrics for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub result_count: u64,
    pub data_transfer_bytes: u64,
    pub success_rate: f64,
}

/// Pattern coverage analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCoverageResult {
    pub pattern: TriplePattern,
    pub total_sources: usize,
    pub covering_sources: Vec<SourceCoverage>,
    pub coverage_score: f64,
    pub confidence: f64,
    pub estimated_result_size: u64,
}

/// Coverage information for a specific source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCoverage {
    pub service_endpoint: String,
    pub coverage_score: f64,
    pub selectivity: f64,
    pub estimated_results: u64,
    pub data_quality: f64,
    pub response_time_estimate: u64,
}

/// Selection statistics for monitoring
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SelectionStatistics {
    pub total_selections: u64,
    pub pattern_coverage_hits: u64,
    pub predicate_filter_hits: u64,
    pub range_selection_hits: u64,
    pub ml_prediction_hits: u64,
    pub average_sources_per_query: f64,
    pub selection_accuracy: f64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// ML prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub recommended_sources: Vec<String>,
    pub confidence_scores: HashMap<String, f64>,
    pub predicted_performance: HashMap<String, f64>,
    pub feature_importance: HashMap<String, f64>,
}

/// Comprehensive source selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionResult {
    pub selected_sources: Vec<String>,
    pub selection_reasons: HashMap<String, Vec<String>>,
    pub confidence_scores: HashMap<String, f64>,
    pub estimated_performance: HashMap<String, PerformanceMetrics>,
    pub selection_method: SelectionMethod,
    pub fallback_sources: Vec<String>,
}

/// Method used for source selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    PatternCoverage,
    PredicateFiltering,
    RangeBased,
    MLPrediction,
    Hybrid,
    Fallback,
}

impl AdvancedSourceSelector {
    /// Create a new advanced source selector
    pub fn new(config: SourceSelectionConfig) -> Self {
        let ml_predictor = if config.enable_ml_prediction {
            Some(MLSourcePredictor::new())
        } else {
            None
        };

        Self {
            pattern_analyzer: PatternCoverageAnalyzer::new(),
            predicate_filter: PredicateBasedFilter::new(&config),
            range_selector: RangeBasedSelector::new(),
            statistics: Arc::new(RwLock::new(SelectionStatistics::default())),
            config,
            ml_predictor,
        }
    }

    /// Select optimal sources for a set of triple patterns
    pub async fn select_sources(
        &self,
        patterns: &[TriplePattern],
        constraints: &[RangeConstraint],
        registry: &ServiceRegistry,
    ) -> Result<SourceSelectionResult> {
        info!(
            "Selecting sources for {} patterns with {} constraints",
            patterns.len(),
            constraints.len()
        );

        let mut selected_sources = HashSet::new();
        let mut selection_reasons: HashMap<String, Vec<String>> = HashMap::new();
        let mut confidence_scores: HashMap<String, f64> = HashMap::new();
        let mut methods_used = Vec::new();

        // Step 1: Pattern coverage analysis
        if self.config.enable_pattern_coverage {
            let coverage_results = self
                .pattern_analyzer
                .analyze_coverage(patterns, registry)
                .await?;

            for result in coverage_results {
                for source in result.covering_sources {
                    if source.coverage_score >= self.config.coverage_threshold {
                        selected_sources.insert(source.service_endpoint.clone());
                        selection_reasons
                            .entry(source.service_endpoint.clone())
                            .or_insert_with(Vec::new)
                            .push(format!("Pattern coverage: {:.2}", source.coverage_score));
                        confidence_scores
                            .insert(source.service_endpoint.clone(), source.coverage_score);
                    }
                }
            }
            methods_used.push(SelectionMethod::PatternCoverage);
        }

        // Step 2: Predicate-based filtering
        if self.config.enable_predicate_filtering {
            let predicate_matches = self
                .predicate_filter
                .filter_by_predicates(patterns, registry)
                .await?;

            for (source, score) in predicate_matches {
                selected_sources.insert(source.clone());
                selection_reasons
                    .entry(source.clone())
                    .or_insert_with(Vec::new)
                    .push(format!("Predicate match: {:.2}", score));
                *confidence_scores.entry(source).or_insert(0.0) += score * 0.3;
            }
            methods_used.push(SelectionMethod::PredicateFiltering);
        }

        // Step 3: Range-based selection
        if self.config.enable_range_selection && !constraints.is_empty() {
            let range_matches = self
                .range_selector
                .select_by_ranges(constraints, registry)
                .await?;

            for (source, score) in range_matches {
                selected_sources.insert(source.clone());
                selection_reasons
                    .entry(source.clone())
                    .or_insert_with(Vec::new)
                    .push(format!("Range match: {:.2}", score));
                *confidence_scores.entry(source).or_insert(0.0) += score * 0.4;
            }
            methods_used.push(SelectionMethod::RangeBased);
        }

        // Step 4: ML-based prediction
        if let Some(predictor) = &self.ml_predictor {
            if self.config.enable_ml_prediction {
                let query_features = self.extract_query_features(patterns, constraints).await?;
                let prediction = predictor.predict_sources(&query_features).await?;

                for source in prediction.recommended_sources {
                    if let Some(conf) = prediction.confidence_scores.get(&source) {
                        if *conf >= self.config.ml_confidence_threshold {
                            selected_sources.insert(source.clone());
                            selection_reasons
                                .entry(source.clone())
                                .or_insert_with(Vec::new)
                                .push(format!("ML prediction: {:.2}", conf));
                            *confidence_scores.entry(source).or_insert(0.0) += conf * 0.5;
                        }
                    }
                }
                methods_used.push(SelectionMethod::MLPrediction);
            }
        }

        // Fallback: select all available sources if no sources selected
        if selected_sources.is_empty() {
            let all_services: Vec<_> = registry.get_all_services().collect();
            selected_sources.extend(all_services.iter().map(|s| s.endpoint.clone()));
            methods_used.push(SelectionMethod::Fallback);
            warn!("No sources selected by algorithms, falling back to all sources");
        }

        // Limit number of sources if configured
        let final_sources: Vec<String> = selected_sources
            .into_iter()
            .take(self.config.max_sources_per_pattern)
            .collect();

        // Generate fallback sources
        let fallback_sources: Vec<String> = registry
            .get_all_services()
            .map(|s| s.endpoint.clone())
            .filter(|s| !final_sources.contains(s))
            .take(3)
            .collect();

        // Update statistics
        self.update_statistics(patterns.len(), final_sources.len())
            .await;

        let selection_method = if methods_used.len() > 1 {
            SelectionMethod::Hybrid
        } else {
            methods_used
                .into_iter()
                .next()
                .unwrap_or(SelectionMethod::Fallback)
        };

        let estimated_performance = self
            .estimate_performance(&final_sources, patterns, constraints)
            .await?;

        Ok(SourceSelectionResult {
            selected_sources: final_sources,
            selection_reasons,
            confidence_scores,
            estimated_performance,
            selection_method,
            fallback_sources,
        })
    }

    /// Update service statistics and indices
    pub async fn update_service_data(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        // Update pattern coverage analyzer
        self.pattern_analyzer
            .update_service_statistics(service_endpoint, triples)
            .await?;

        // Update predicate filters
        self.predicate_filter
            .update_filters(service_endpoint, triples)
            .await?;

        // Update range indices
        self.range_selector
            .update_indices(service_endpoint, triples)
            .await?;

        info!(
            "Updated service data for {} with {} triples",
            service_endpoint,
            triples.len()
        );
        Ok(())
    }

    /// Train ML predictor with new performance data
    pub async fn train_predictor(
        &mut self,
        training_samples: Vec<SourcePredictionSample>,
    ) -> Result<()> {
        if let Some(predictor) = &mut self.ml_predictor {
            predictor.train(training_samples).await?;
            info!("Trained ML predictor with new performance data");
        }
        Ok(())
    }

    /// Get selection statistics
    pub async fn get_statistics(&self) -> SelectionStatistics {
        self.statistics.read().await.clone()
    }

    /// Extract query features for ML prediction
    async fn extract_query_features(
        &self,
        patterns: &[TriplePattern],
        constraints: &[RangeConstraint],
    ) -> Result<QueryFeatures> {
        let variable_count = patterns
            .iter()
            .flat_map(|p| vec![&p.subject, &p.predicate, &p.object])
            .filter(|s| s.starts_with('?'))
            .collect::<HashSet<_>>()
            .len();

        let predicate_types: Vec<String> = patterns
            .iter()
            .map(|p| p.predicate.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let has_joins = patterns.len() > 1 && variable_count > 0;
        let complexity_score = (patterns.len() as f64) * (variable_count as f64) / 10.0;

        Ok(QueryFeatures {
            pattern_count: patterns.len(),
            variable_count,
            predicate_types,
            has_ranges: !constraints.is_empty(),
            has_joins,
            complexity_score,
            selectivity_estimate: self
                .calculate_selectivity_estimate(patterns, constraints)
                .await?,
        })
    }

    /// Update selection statistics
    async fn update_statistics(&self, pattern_count: usize, selected_sources: usize) {
        let mut stats = self.statistics.write().await;
        stats.total_selections += 1;
        stats.average_sources_per_query =
            (stats.average_sources_per_query + selected_sources as f64) / 2.0;
        stats.last_updated = Some(Utc::now());
    }

    /// Estimate performance for selected sources
    async fn estimate_performance(
        &self,
        sources: &[String],
        patterns: &[TriplePattern],
        constraints: &[RangeConstraint],
    ) -> Result<HashMap<String, PerformanceMetrics>> {
        let mut performance_map = HashMap::new();

        for source in sources {
            // Base performance estimation based on patterns and constraints
            let complexity_factor = patterns.len() as f64 * (1.0 + constraints.len() as f64 * 0.2);
            let execution_time_ms = (50.0 * complexity_factor).max(10.0) as u64;

            // Estimate result count based on pattern selectivity
            let result_count = self.estimate_pattern_results(patterns).await? as u64;

            // Estimate data transfer size (assuming 100 bytes per result on average)
            let data_transfer_bytes = result_count * 100;

            // High success rate for known sources
            let success_rate = 0.95;

            performance_map.insert(
                source.clone(),
                PerformanceMetrics {
                    execution_time_ms,
                    result_count,
                    data_transfer_bytes,
                    success_rate,
                },
            );
        }

        Ok(performance_map)
    }

    /// Calculate selectivity estimate for patterns
    async fn calculate_selectivity_estimate(
        &self,
        patterns: &[TriplePattern],
        constraints: &[RangeConstraint],
    ) -> Result<f64> {
        let mut total_selectivity = 1.0;

        // Base selectivity per pattern (lower = more selective)
        for pattern in patterns {
            let mut pattern_selectivity = 0.5; // Default moderate selectivity

            // Variables are less selective
            if pattern.subject.starts_with('?') {
                pattern_selectivity *= 0.8;
            }
            if pattern.predicate.starts_with('?') {
                pattern_selectivity *= 0.7;
            }
            if pattern.object.starts_with('?') {
                pattern_selectivity *= 0.8;
            }

            total_selectivity *= pattern_selectivity;
        }

        // Range constraints increase selectivity
        for constraint in constraints {
            match constraint.data_type {
                RangeDataType::Integer | RangeDataType::Float => {
                    if constraint.min_value.is_some() || constraint.max_value.is_some() {
                        total_selectivity *= 0.3; // Numeric ranges are quite selective
                    }
                }
                RangeDataType::DateTime => {
                    total_selectivity *= 0.4; // DateTime ranges are moderately selective
                }
                _ => {
                    total_selectivity *= 0.6; // Other constraints are less selective
                }
            }
        }

        Ok(f64::max(total_selectivity, 0.001).min(1.0)) // Clamp between 0.1% and 100%
    }

    /// Estimate pattern result count
    async fn estimate_pattern_results(&self, patterns: &[TriplePattern]) -> Result<usize> {
        let base_size = 10000; // Assume 10K triples per service on average
        let selectivity = self.calculate_selectivity_estimate(patterns, &[]).await?;
        Ok((base_size as f64 * selectivity) as usize)
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
            let services: Vec<_> = registry.get_all_services().collect();

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
        pattern: &TriplePattern,
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

        let mut quality_score = 1.0;
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
        quality_score = (completeness * 0.6) + (uri_validity * 0.4);

        Ok(quality_score.max(0.1).min(1.0)) // Clamp between 10% and 100%
    }
}

impl PredicateBasedFilter {
    pub fn new(config: &SourceSelectionConfig) -> Self {
        Self {
            service_filters: Arc::new(RwLock::new(HashMap::new())),
            last_update: Arc::new(RwLock::new(Utc::now())),
        }
    }

    pub async fn filter_by_predicates(
        &self,
        patterns: &[TriplePattern],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, f64>> {
        let mut matches = HashMap::new();
        let filters = self.service_filters.read().await;

        for (service_endpoint, filter) in filters.iter() {
            let mut match_score = 0.0;
            let mut total_patterns = 0.0;

            for pattern in patterns {
                total_patterns += 1.0;

                // Check predicate membership
                if filter.predicate_filter.contains(&pattern.predicate) {
                    match_score += 1.0;
                }

                // Check subject membership (if not variable)
                if !pattern.subject.starts_with('?') {
                    if filter.subject_filter.contains(&pattern.subject) {
                        match_score += 0.5;
                    }
                }

                // Check object membership (if not variable)
                if !pattern.object.starts_with('?') {
                    if filter.object_filter.contains(&pattern.object) {
                        match_score += 0.5;
                    }
                }
            }

            if total_patterns > 0.0 {
                let final_score = match_score / total_patterns;
                if final_score > 0.0 {
                    matches.insert(service_endpoint.clone(), final_score);
                }
            }
        }

        Ok(matches)
    }

    pub async fn update_filters(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        let capacity = triples.len().max(1000) as u32;
        let mut predicate_filter = BloomFilter::with_rate(0.01, capacity);
        let mut subject_filter = BloomFilter::with_rate(0.01, capacity);
        let mut object_filter = BloomFilter::with_rate(0.01, capacity);
        let mut type_filter = BloomFilter::with_rate(0.01, capacity);

        for (s, p, o) in triples {
            predicate_filter.insert(p);
            subject_filter.insert(s);
            object_filter.insert(o);

            // Add type information if available
            if p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                type_filter.insert(o);
            }
        }

        let filters = ServiceBloomFilters {
            predicate_filter,
            subject_filter,
            object_filter,
            type_filter,
            last_updated: Utc::now(),
            estimated_elements: triples.len(),
        };

        self.service_filters
            .write()
            .await
            .insert(service_endpoint.to_string(), filters);
        *self.last_update.write().await = Utc::now();

        Ok(())
    }
}

impl RangeBasedSelector {
    pub fn new() -> Self {
        Self {
            range_indices: Arc::new(RwLock::new(HashMap::new())),
            temporal_indices: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn select_by_ranges(
        &self,
        constraints: &[RangeConstraint],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, f64>> {
        let mut matches = HashMap::new();
        let range_indices = self.range_indices.read().await;
        let temporal_indices = self.temporal_indices.read().await;

        for (service_endpoint, range_index) in range_indices.iter() {
            let mut match_score = 0.0;
            let mut total_constraints = 0.0;

            for constraint in constraints {
                total_constraints += 1.0;

                match constraint.data_type {
                    RangeDataType::Integer | RangeDataType::Float => {
                        if let Some(numeric_range) =
                            range_index.numeric_ranges.get(&constraint.field)
                        {
                            if self.range_overlaps_numeric(constraint, numeric_range) {
                                match_score += 1.0;
                            }
                        }
                    }
                    RangeDataType::DateTime => {
                        if let Some(temporal_index) = temporal_indices.get(service_endpoint) {
                            if let Some(datetime_range) =
                                temporal_index.datetime_ranges.get(&constraint.field)
                            {
                                if self.range_overlaps_datetime(constraint, datetime_range) {
                                    match_score += 1.0;
                                }
                            }
                        }
                    }
                    RangeDataType::String => {
                        if let Some(string_range) = range_index.string_ranges.get(&constraint.field)
                        {
                            if self.range_overlaps_string(constraint, string_range) {
                                match_score += 1.0;
                            }
                        }
                    }
                    RangeDataType::Uri => {
                        if let Some(uri_pattern) = range_index.uri_patterns.get(&constraint.field) {
                            if self.matches_uri_pattern(constraint, uri_pattern) {
                                match_score += 1.0;
                            }
                        }
                    }
                }
            }

            if total_constraints > 0.0 {
                let final_score = match_score / total_constraints;
                if final_score > 0.0 {
                    matches.insert(service_endpoint.clone(), final_score);
                }
            }
        }

        Ok(matches)
    }

    pub async fn update_indices(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        // Analyze numeric, string, and URI ranges from the triples
        let mut numeric_ranges = HashMap::new();
        let mut string_ranges = HashMap::new();
        let mut uri_patterns = HashMap::new();
        let mut datetime_ranges = HashMap::new();
        let mut year_ranges = HashMap::new();

        // Group triples by predicate for analysis
        let mut predicate_values: HashMap<String, Vec<String>> = HashMap::new();
        for (_, p, o) in triples {
            predicate_values
                .entry(p.clone())
                .or_insert_with(Vec::new)
                .push(o.clone());
        }

        // Analyze each predicate's values
        for (predicate, values) in predicate_values {
            self.analyze_numeric_range(&predicate, &values, &mut numeric_ranges);
            self.analyze_string_range(&predicate, &values, &mut string_ranges);
            self.analyze_uri_patterns(&predicate, &values, &mut uri_patterns);
            self.analyze_datetime_range(&predicate, &values, &mut datetime_ranges);
            self.analyze_year_range(&predicate, &values, &mut year_ranges);
        }

        let range_index = ServiceRangeIndex {
            numeric_ranges,
            string_ranges,
            uri_patterns,
            last_updated: Utc::now(),
        };

        let temporal_index = ServiceTemporalIndex {
            datetime_ranges,
            year_ranges,
            temporal_patterns: HashMap::new(), // TODO: Implement pattern detection
            last_updated: Utc::now(),
        };

        self.range_indices
            .write()
            .await
            .insert(service_endpoint.to_string(), range_index);

        self.temporal_indices
            .write()
            .await
            .insert(service_endpoint.to_string(), temporal_index);
        Ok(())
    }

    fn range_overlaps_numeric(&self, constraint: &RangeConstraint, range: &NumericRange) -> bool {
        // Simplified overlap detection
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            if let (Ok(min_val), Ok(max_val)) = (min_str.parse::<f64>(), max_str.parse::<f64>()) {
                return !(max_val < range.min_value || min_val > range.max_value);
            }
        }
        true // Default to potential match if parsing fails
    }

    fn range_overlaps_datetime(&self, constraint: &RangeConstraint, range: &DateTimeRange) -> bool {
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            if let (Ok(min_dt), Ok(max_dt)) = (
                DateTime::parse_from_rfc3339(min_str).map(|dt| dt.with_timezone(&Utc)),
                DateTime::parse_from_rfc3339(max_str).map(|dt| dt.with_timezone(&Utc)),
            ) {
                return !(max_dt < range.earliest || min_dt > range.latest);
            }
        }
        true // Default to potential match if parsing fails
    }

    fn range_overlaps_string(&self, constraint: &RangeConstraint, range: &StringRange) -> bool {
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            let min_len = min_str.len();
            let max_len = max_str.len();

            // Check if length ranges overlap
            if min_len > range.max_length || max_len < range.min_length {
                return false;
            }

            // Check prefix matching
            for prefix in &range.common_prefixes {
                if min_str.starts_with(prefix) || max_str.starts_with(prefix) {
                    return true;
                }
            }
        }
        true // Default to potential match
    }

    fn matches_uri_pattern(&self, constraint: &RangeConstraint, pattern: &UriPattern) -> bool {
        if let Some(value) = &constraint.min_value {
            // Check if the constraint value matches any of the URI patterns
            for base_uri in &pattern.base_uris {
                if value.starts_with(base_uri) {
                    return true;
                }
            }

            for namespace in &pattern.namespace_prefixes {
                if value.starts_with(namespace) {
                    return true;
                }
            }
        }
        true // Default to potential match
    }

    /// Analyze numeric ranges in predicate values
    fn analyze_numeric_range(
        &self,
        predicate: &str,
        values: &[String],
        numeric_ranges: &mut HashMap<String, NumericRange>,
    ) {
        let mut numeric_values = Vec::new();

        for value in values {
            if let Ok(num) = value.parse::<f64>() {
                numeric_values.push(num);
            }
        }

        if !numeric_values.is_empty() {
            let min_value = numeric_values
                .iter()
                .cloned()
                .fold(f64::INFINITY, |a, b| a.min(b));
            let max_value = numeric_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));

            // Take up to 10 sample values
            let mut sample_values = numeric_values.clone();
            sample_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sample_values.truncate(10);

            numeric_ranges.insert(
                predicate.to_string(),
                NumericRange {
                    min_value,
                    max_value,
                    count: numeric_values.len() as u64,
                    sample_values,
                },
            );
        }
    }

    /// Analyze string ranges in predicate values
    fn analyze_string_range(
        &self,
        predicate: &str,
        values: &[String],
        string_ranges: &mut HashMap<String, StringRange>,
    ) {
        if values.is_empty() {
            return;
        }

        let min_length = values.iter().map(|s| s.len()).min().unwrap_or(0);
        let max_length = values.iter().map(|s| s.len()).max().unwrap_or(0);

        // Find common prefixes (at least 3 chars and appears in >20% of values)
        let mut prefix_counts: HashMap<String, usize> = HashMap::new();
        for value in values {
            for len in 3..=value.len().min(10) {
                let prefix = &value[..len];
                *prefix_counts.entry(prefix.to_string()).or_insert(0) += 1;
            }
        }

        let threshold = (values.len() as f64 * 0.2) as usize;
        let common_prefixes: Vec<String> = prefix_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(prefix, _)| prefix)
            .collect();

        // Take up to 10 sample values
        let mut sample_values: Vec<String> = values.iter().take(10).cloned().collect();
        sample_values.sort();

        string_ranges.insert(
            predicate.to_string(),
            StringRange {
                min_length,
                max_length,
                common_prefixes,
                sample_values,
            },
        );
    }

    /// Analyze URI patterns in predicate values
    fn analyze_uri_patterns(
        &self,
        predicate: &str,
        values: &[String],
        uri_patterns: &mut HashMap<String, UriPattern>,
    ) {
        let mut base_uris = HashSet::new();
        let mut path_patterns = HashSet::new();
        let mut namespace_prefixes = HashSet::new();

        for value in values {
            if value.starts_with("http://") || value.starts_with("https://") {
                // Extract base URI
                if let Some(pos) = value[8..].find('/').map(|p| p + 8) {
                    base_uris.insert(value[..pos].to_string());

                    // Extract path pattern (remove specific IDs/numbers)
                    let path = &value[pos..];
                    let generalized_path = self.generalize_path(path);
                    path_patterns.insert(generalized_path);
                } else {
                    base_uris.insert(value.clone());
                }

                // Extract namespace (everything before the last '/' or '#')
                if let Some(pos) = value.rfind(&['/', '#'][..]) {
                    namespace_prefixes.insert(value[..=pos].to_string());
                }
            }
        }

        if !base_uris.is_empty() {
            uri_patterns.insert(
                predicate.to_string(),
                UriPattern {
                    base_uris: base_uris.into_iter().collect(),
                    path_patterns: path_patterns.into_iter().collect(),
                    namespace_prefixes: namespace_prefixes.into_iter().collect(),
                },
            );
        }
    }

    /// Analyze datetime ranges in predicate values
    fn analyze_datetime_range(
        &self,
        predicate: &str,
        values: &[String],
        datetime_ranges: &mut HashMap<String, DateTimeRange>,
    ) {
        let mut datetime_values = Vec::new();

        for value in values {
            if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
                datetime_values.push(dt.with_timezone(&Utc));
            }
        }

        if !datetime_values.is_empty() {
            let earliest = datetime_values.iter().min().unwrap().clone();
            let latest = datetime_values.iter().max().unwrap().clone();

            // Determine granularity based on the range
            let range_duration = latest.signed_duration_since(earliest);
            let granularity = if range_duration.num_days() > 365 {
                TemporalGranularity::Year
            } else if range_duration.num_days() > 30 {
                TemporalGranularity::Month
            } else if range_duration.num_days() > 1 {
                TemporalGranularity::Day
            } else {
                TemporalGranularity::Hour
            };

            datetime_ranges.insert(
                predicate.to_string(),
                DateTimeRange {
                    earliest,
                    latest,
                    count: datetime_values.len() as u64,
                    granularity,
                },
            );
        }
    }

    /// Analyze year ranges in predicate values
    fn analyze_year_range(
        &self,
        predicate: &str,
        values: &[String],
        year_ranges: &mut HashMap<String, YearRange>,
    ) {
        let mut year_distribution = HashMap::new();

        for value in values {
            // Try to extract year from various formats
            if let Some(year) = self.extract_year(value) {
                *year_distribution.entry(year).or_insert(0) += 1;
            }
        }

        if !year_distribution.is_empty() {
            let earliest_year = *year_distribution.keys().min().unwrap();
            let latest_year = *year_distribution.keys().max().unwrap();

            year_ranges.insert(
                predicate.to_string(),
                YearRange {
                    earliest_year,
                    latest_year,
                    year_distribution,
                },
            );
        }
    }

    /// Generalize path by replacing numbers with placeholders
    fn generalize_path(&self, path: &str) -> String {
        use regex::Regex;
        let re = Regex::new(r"\d+").unwrap();
        re.replace_all(path, "{id}").to_string()
    }

    /// Extract year from various date formats
    fn extract_year(&self, value: &str) -> Option<i32> {
        // Try ISO date format first
        if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
            return Some(dt.year());
        }

        // Try just year as number
        if let Ok(year) = value.parse::<i32>() {
            if year >= 1900 && year <= 2100 {
                return Some(year);
            }
        }

        // Try to find 4-digit year in the string
        use regex::Regex;
        let re = Regex::new(r"\b(19|20)\d{2}\b").unwrap();
        if let Some(captures) = re.find(value) {
            if let Ok(year) = captures.as_str().parse::<i32>() {
                return Some(year);
            }
        }

        None
    }
}

impl MLSourcePredictor {
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            feature_weights: HashMap::new(),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            model_accuracy: 0.5,
        }
    }

    pub async fn predict_sources(&self, features: &QueryFeatures) -> Result<PredictionResult> {
        let feature_key = format!(
            "{}:{}:{}",
            features.pattern_count, features.variable_count, features.complexity_score
        );

        // Check cache first
        if let Some(cached) = self.prediction_cache.read().await.get(&feature_key) {
            return Ok(cached.clone());
        }

        // Simple prediction based on feature weights
        let mut source_scores = HashMap::new();

        // This is a simplified ML model - in practice, this would use
        // more sophisticated algorithms like neural networks or ensemble methods
        for sample in &self.training_data {
            if self.features_similar(features, &sample.query_features) {
                for source in &sample.selected_sources {
                    let score = self.calculate_similarity_score(features, &sample.query_features);
                    *source_scores.entry(source.clone()).or_insert(0.0) += score;
                }
            }
        }

        // Normalize scores
        let max_score = source_scores.values().cloned().fold(0.0, f64::max);
        if max_score > 0.0 {
            for score in source_scores.values_mut() {
                *score /= max_score;
            }
        }

        let recommended_sources: Vec<String> = source_scores.keys().cloned().collect();

        let result = PredictionResult {
            recommended_sources,
            confidence_scores: source_scores.clone(),
            predicted_performance: HashMap::new(), // TODO: Implement performance prediction
            feature_importance: self.feature_weights.clone(),
        };

        // Cache the result
        self.prediction_cache
            .write()
            .await
            .insert(feature_key, result.clone());

        Ok(result)
    }

    pub async fn train(&mut self, samples: Vec<SourcePredictionSample>) -> Result<()> {
        self.training_data.extend(samples);

        // Simple feature weight learning
        self.update_feature_weights().await?;

        // Keep only recent training data (last 10000 samples)
        if self.training_data.len() > 10000 {
            let start = self.training_data.len() - 10000;
            self.training_data.drain(0..start);
        }

        // Clear prediction cache to force recomputation
        self.prediction_cache.write().await.clear();

        Ok(())
    }

    async fn update_feature_weights(&mut self) -> Result<()> {
        // Simple feature weight calculation based on correlation with performance
        let mut weights = HashMap::new();

        weights.insert("pattern_count".to_string(), 0.3);
        weights.insert("variable_count".to_string(), 0.2);
        weights.insert("complexity_score".to_string(), 0.4);
        weights.insert("has_joins".to_string(), 0.1);

        self.feature_weights = weights;
        Ok(())
    }

    fn features_similar(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> bool {
        let pattern_diff = (f1.pattern_count as f64 - f2.pattern_count as f64).abs();
        let var_diff = (f1.variable_count as f64 - f2.variable_count as f64).abs();
        let complexity_diff = (f1.complexity_score - f2.complexity_score).abs();

        pattern_diff <= 2.0 && var_diff <= 3.0 && complexity_diff <= 1.0
    }

    fn calculate_similarity_score(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> f64 {
        let pattern_sim = 1.0 - (f1.pattern_count as f64 - f2.pattern_count as f64).abs() / 10.0;
        let var_sim = 1.0 - (f1.variable_count as f64 - f2.variable_count as f64).abs() / 10.0;
        let complexity_sim = 1.0 - (f1.complexity_score - f2.complexity_score).abs();

        (pattern_sim + var_sim + complexity_sim) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pattern(subject: &str, predicate: &str, object: &str) -> TriplePattern {
        TriplePattern {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            graph: None,
        }
    }

    fn create_test_service() -> FederatedService {
        use std::collections::HashSet;

        FederatedService {
            id: "test-service-1".to_string(),
            name: "Test Service".to_string(),
            endpoint: "http://example.org/sparql".to_string(),
            service_type: crate::ServiceType::Sparql,
            capabilities: {
                let mut caps = HashSet::new();
                caps.insert(crate::ServiceCapability::SparqlQuery);
                caps.insert(crate::ServiceCapability::Sparql11Query);
                caps
            },
            data_patterns: vec!["http://example.org/".to_string()],
            auth: None,
            metadata: crate::ServiceMetadata {
                description: Some("Test SPARQL endpoint".to_string()),
                version: Some("1.0".to_string()),
                maintainer: None,
                tags: vec!["test".to_string()],
                documentation_url: None,
                schema_url: None,
            },
            extended_metadata: None,
            performance: crate::ServicePerformance::default(),
        }
    }

    #[tokio::test]
    async fn test_pattern_coverage_analyzer() {
        let analyzer = PatternCoverageAnalyzer::new();
        let pattern = create_test_pattern("?s", "http://example.org/name", "?o");

        // Update statistics with test data
        let triples = vec![
            (
                "http://example.org/entity1".to_string(),
                "http://example.org/name".to_string(),
                "Alice".to_string(),
            ),
            (
                "http://example.org/entity2".to_string(),
                "http://example.org/name".to_string(),
                "Bob".to_string(),
            ),
        ];

        analyzer
            .update_service_statistics("http://example.org/sparql", &triples)
            .await
            .unwrap();

        // Test coverage analysis
        let registry = ServiceRegistry::new();
        let results = analyzer
            .analyze_coverage(&[pattern], &registry)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].coverage_score >= 0.0);
    }

    #[tokio::test]
    async fn test_source_selector() {
        let config = SourceSelectionConfig::default();
        let selector = AdvancedSourceSelector::new(config);

        let patterns = vec![
            create_test_pattern("?s", "http://example.org/name", "?o"),
            create_test_pattern("?s", "http://example.org/age", "?age"),
        ];

        let constraints = vec![RangeConstraint {
            field: "age".to_string(),
            min_value: Some("18".to_string()),
            max_value: Some("65".to_string()),
            data_type: RangeDataType::Integer,
        }];

        let registry = ServiceRegistry::new();
        let result = selector
            .select_sources(&patterns, &constraints, &registry)
            .await
            .unwrap();

        assert!(!result.selected_sources.is_empty());
        assert!(matches!(result.selection_method, SelectionMethod::Fallback)); // No services registered
    }

    #[tokio::test]
    async fn test_predicate_filter() {
        let config = SourceSelectionConfig::default();
        let filter = PredicateBasedFilter::new(&config);

        let triples = vec![
            (
                "http://example.org/entity1".to_string(),
                "http://example.org/name".to_string(),
                "Alice".to_string(),
            ),
            (
                "http://example.org/entity2".to_string(),
                "http://example.org/age".to_string(),
                "25".to_string(),
            ),
        ];

        filter
            .update_filters("http://example.org/sparql", &triples)
            .await
            .unwrap();

        let patterns = vec![create_test_pattern("?s", "http://example.org/name", "?o")];

        let registry = ServiceRegistry::new();
        let matches = filter
            .filter_by_predicates(&patterns, &registry)
            .await
            .unwrap();

        assert!(matches.contains_key("http://example.org/sparql"));
    }

    #[test]
    fn test_query_features() {
        let features = QueryFeatures {
            pattern_count: 3,
            variable_count: 2,
            predicate_types: vec!["http://example.org/name".to_string()],
            has_ranges: true,
            has_joins: true,
            complexity_score: 1.5,
            selectivity_estimate: 0.3,
        };

        assert_eq!(features.pattern_count, 3);
        assert_eq!(features.variable_count, 2);
        assert!(features.has_ranges);
        assert!(features.has_joins);
    }

    #[test]
    fn test_config_serialization() {
        let config = SourceSelectionConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SourceSelectionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.coverage_threshold, deserialized.coverage_threshold);
        assert_eq!(
            config.enable_pattern_coverage,
            deserialized.enable_pattern_coverage
        );
    }
}
