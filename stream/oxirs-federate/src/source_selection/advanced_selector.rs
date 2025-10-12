//! Advanced source selector implementation
//!
//! Coordinates multiple selection strategies (coverage, predicate, range, ML)

use crate::source_selection::types::*;
use crate::ServiceRegistry;
use anyhow::Result;
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

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
                            .or_default()
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
                    .or_default()
                    .push(format!("Predicate match: {score:.2}"));
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
                    .or_default()
                    .push(format!("Range match: {score:.2}"));
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
                                .or_default()
                                .push(format!("ML prediction: {conf:.2}"));
                            *confidence_scores.entry(source).or_insert(0.0) += conf * 0.5;
                        }
                    }
                }
                methods_used.push(SelectionMethod::MLPrediction);
            }
        }

        // Fallback: select all available sources if no sources selected
        if selected_sources.is_empty() {
            let all_services: Vec<_> = registry.get_all_services().into_iter().collect();
            if !all_services.is_empty() {
                selected_sources.extend(all_services.iter().map(|s| s.endpoint.clone()));
                methods_used.push(SelectionMethod::Fallback);
                warn!("No sources selected by algorithms, falling back to all sources");
            } else {
                // Secondary fallback: create default sources when registry is empty
                selected_sources.insert("http://localhost:3030/sparql".to_string());
                selected_sources.insert("http://localhost:8080/sparql".to_string());
                methods_used.push(SelectionMethod::Fallback);
                warn!("No services in registry, using default fallback sources");
            }
        }

        // Limit number of sources if configured
        let final_sources: Vec<String> = selected_sources
            .into_iter()
            .take(self.config.max_sources_per_pattern)
            .collect();

        // Generate fallback sources
        let fallback_sources: Vec<String> = registry
            .get_all_services()
            .into_iter()
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
    async fn update_statistics(&self, _pattern_count: usize, selected_sources: usize) {
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
