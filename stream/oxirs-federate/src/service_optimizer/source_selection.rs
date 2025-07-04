//! Advanced source selection algorithms
//!
//! This module implements sophisticated source selection algorithms for federated queries,
//! including pattern coverage analysis, predicate-based filtering, and range-based selection.

use anyhow::Result;
use bloom::{BloomFilter, ASMS};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

use super::{types::*, ServiceOptimizer};
use crate::planner::TriplePattern;
use crate::{FederatedService, ServiceCapability, ServiceRegistry};

impl ServiceOptimizer {
    /// Perform triple pattern coverage analysis for source selection
    pub async fn analyze_triple_pattern_coverage(
        &self,
        patterns: &[TriplePattern],
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, PatternCoverageAnalysis>> {
        let mut coverage_map = HashMap::new();

        for service in services {
            let coverage = self
                .calculate_service_pattern_coverage(service, patterns, registry)
                .await?;
            coverage_map.insert(service.id.clone(), coverage);
        }

        Ok(coverage_map)
    }

    /// Calculate how well a service covers a set of triple patterns
    async fn calculate_service_pattern_coverage(
        &self,
        service: &FederatedService,
        patterns: &[TriplePattern],
        _registry: &ServiceRegistry,
    ) -> Result<PatternCoverageAnalysis> {
        let mut covered_patterns = 0;
        let mut partially_covered = 0;
        let mut coverage_scores = Vec::new();

        for pattern in patterns {
            let score = self
                .calculate_pattern_coverage_score(service, pattern)
                .await?;
            coverage_scores.push(score);

            if score >= 0.8 {
                covered_patterns += 1;
            } else if score >= 0.3 {
                partially_covered += 1;
            }
        }

        let total_coverage = coverage_scores.iter().sum::<f64>() / patterns.len() as f64;
        let coverage_quality = self.assess_coverage_quality(&coverage_scores);

        Ok(PatternCoverageAnalysis {
            total_patterns: patterns.len(),
            covered_patterns,
            partially_covered_patterns: partially_covered,
            uncovered_patterns: patterns.len() - covered_patterns - partially_covered,
            overall_coverage_score: total_coverage,
            coverage_quality,
            pattern_scores: coverage_scores,
        })
    }

    /// Calculate coverage score for a single pattern on a service
    async fn calculate_pattern_coverage_score(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Base score for general SPARQL capability
        if service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            score += 0.5;
        }

        // Score boost for specific capabilities matching pattern needs
        if pattern.pattern_string.contains("REGEX")
            && service
                .capabilities
                .contains(&ServiceCapability::FullTextSearch)
        {
            score += 0.3;
        }

        if pattern.pattern_string.contains("geof:")
            && service
                .capabilities
                .contains(&ServiceCapability::Geospatial)
        {
            score += 0.3;
        }

        if pattern.pattern_string.contains("NOW()")
            && service
                .capabilities
                .contains(&ServiceCapability::TemporalQueries)
        {
            score += 0.2;
        }

        // Domain-specific boost based on service description and pattern content
        if let Some(ref description) = service.metadata.description {
            score += self.calculate_domain_affinity(description, pattern);
        }

        Ok((score as f64).min(1.0))
    }

    /// Assess the quality of coverage distribution
    fn assess_coverage_quality(&self, scores: &[f64]) -> CoverageQuality {
        let variance = self.calculate_variance(scores);
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;

        if mean >= 0.8 && variance < 0.1 {
            CoverageQuality::Excellent
        } else if mean >= 0.6 && variance < 0.2 {
            CoverageQuality::Good
        } else if mean >= 0.4 {
            CoverageQuality::Fair
        } else {
            CoverageQuality::Poor
        }
    }

    /// Calculate variance for coverage quality assessment
    fn calculate_variance(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        variance
    }

    /// Calculate domain affinity between service and pattern
    fn calculate_domain_affinity(&self, description: &str, pattern: &TriplePattern) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut affinity = 0.0;

        // Check for domain keywords in pattern
        if pattern.pattern_string.contains("foaf:") && desc_lower.contains("social") {
            affinity += 0.2;
        }
        if pattern.pattern_string.contains("dbo:") && desc_lower.contains("dbpedia") {
            affinity += 0.3;
        }
        if pattern.pattern_string.contains("wdt:") && desc_lower.contains("wikidata") {
            affinity += 0.3;
        }
        if pattern.pattern_string.contains("geo:") && desc_lower.contains("geographic") {
            affinity += 0.2;
        }

        affinity
    }

    /// Predicate-based source filtering
    pub async fn filter_services_by_predicate(
        &self,
        predicate: &str,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<Vec<ServicePredicateScore>> {
        let mut scored_services = Vec::new();

        for service in services {
            let score = self
                .calculate_predicate_affinity_score(service, predicate, registry)
                .await?;

            if score > 0.1 {
                // Only include services with meaningful affinity
                scored_services.push(ServicePredicateScore {
                    service_id: service.id.clone(),
                    predicate: predicate.to_string(),
                    score,
                    confidence: self.calculate_confidence_level(service, predicate),
                    coverage_ratio: score, // Use score as coverage ratio for now
                    freshness_score: 0.8,  // Default freshness score
                    authority_score: 0.7,  // Default authority score
                });
            }
        }

        // Sort by score descending
        scored_services.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_services)
    }

    /// Calculate predicate affinity score for a service
    async fn calculate_predicate_affinity_score(
        &self,
        service: &FederatedService,
        predicate: &str,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Check if service has statistics for this predicate
        if let Some(stats) = self.get_predicate_stats(predicate) {
            // Higher frequency predicates get higher scores
            score += (stats.frequency as f64).log10() / 10.0;

            // Lower selectivity (more common) predicates get higher base scores
            score += (1.0 - stats.selectivity) * 0.5;
        }

        // Service-specific predicate scoring
        if let Some(ref description) = service.metadata.description {
            score += self.calculate_predicate_domain_match(predicate, description);
        }

        // Capability-based scoring
        score += self.calculate_predicate_capability_match(predicate, &service.capabilities);

        Ok((score as f64).min(1.0))
    }

    /// Calculate predicate-domain matching score
    fn calculate_predicate_domain_match(&self, predicate: &str, description: &str) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut score = 0.0;

        // Common namespace mappings
        if predicate.starts_with("foaf:") && desc_lower.contains("social") {
            score += 0.4;
        }
        if predicate.starts_with("dbo:")
            && (desc_lower.contains("dbpedia") || desc_lower.contains("ontology"))
        {
            score += 0.5;
        }
        if predicate.starts_with("wdt:") && desc_lower.contains("wikidata") {
            score += 0.5;
        }
        if predicate.starts_with("geo:")
            && (desc_lower.contains("geographic") || desc_lower.contains("spatial"))
        {
            score += 0.4;
        }
        if predicate.starts_with("dc:") && desc_lower.contains("dublin") {
            score += 0.3;
        }

        score
    }

    /// Calculate predicate-capability matching score
    fn calculate_predicate_capability_match(
        &self,
        predicate: &str,
        capabilities: &HashSet<ServiceCapability>,
    ) -> f64 {
        let mut score = 0.0;

        // Text-related predicates
        if predicate.contains("label") || predicate.contains("name") || predicate.contains("title")
        {
            if capabilities.contains(&ServiceCapability::FullTextSearch) {
                score += 0.2;
            }
        }

        // Geospatial predicates
        if predicate.contains("geo")
            || predicate.contains("location")
            || predicate.contains("coordinate")
        {
            if capabilities.contains(&ServiceCapability::Geospatial) {
                score += 0.3;
            }
        }

        // Temporal predicates
        if predicate.contains("date") || predicate.contains("time") || predicate.contains("year") {
            if capabilities.contains(&ServiceCapability::TemporalQueries) {
                score += 0.2;
            }
        }

        score
    }

    /// Estimate result count for a predicate on a service
    async fn estimate_predicate_result_count(
        &self,
        service: &FederatedService,
        predicate: &str,
    ) -> Result<u64> {
        // Use cached statistics if available
        if let Some(stats) = self.get_predicate_stats(predicate) {
            return Ok(stats.frequency);
        }

        // Fallback estimation based on service characteristics
        let base_estimate = 1000_u64;
        let service_size_factor = if let Some(ref desc) = service.metadata.description {
            if desc.to_lowercase().contains("large") {
                5.0
            } else if desc.to_lowercase().contains("small") {
                0.2
            } else {
                1.0
            }
        } else {
            1.0
        };

        Ok((base_estimate as f64 * service_size_factor) as u64)
    }

    /// Calculate confidence level for predicate estimation
    fn calculate_confidence_level(
        &self,
        _service: &FederatedService,
        predicate: &str,
    ) -> ConfidenceLevel {
        // Higher confidence if we have statistics
        if self.get_predicate_stats(predicate).is_some() {
            return ConfidenceLevel::High;
        }

        // Medium confidence for well-known namespaces
        if predicate.starts_with("rdf:")
            || predicate.starts_with("rdfs:")
            || predicate.starts_with("owl:")
            || predicate.starts_with("foaf:")
        {
            return ConfidenceLevel::Medium;
        }

        // Lower confidence for unknown predicates
        ConfidenceLevel::Low
    }

    /// Range-based source selection for numeric/temporal values
    pub async fn select_services_by_range(
        &self,
        predicate: &str,
        range: &ValueRange,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<Vec<RangeServiceMatch>> {
        let mut matches = Vec::new();

        for service in services {
            if let Some(range_match) = self
                .evaluate_service_range_coverage(service, predicate, range, registry)
                .await?
            {
                matches.push(range_match);
            }
        }

        // Sort by overlap percentage descending
        matches.sort_by(|a, b| {
            b.overlap_percentage
                .partial_cmp(&a.overlap_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(matches)
    }

    /// Evaluate how well a service covers a value range for a predicate
    async fn evaluate_service_range_coverage(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
        _registry: &ServiceRegistry,
    ) -> Result<Option<RangeServiceMatch>> {
        // For demonstration, we'll use a simplified range coverage calculation
        // In practice, this would query service statistics or metadata

        let coverage_score = self
            .estimate_range_coverage(service, predicate, range)
            .await?;

        if coverage_score > 0.1 {
            Ok(Some(RangeServiceMatch {
                service_id: service.id.clone(),
                overlap_type: self.classify_range_overlap(
                    range,
                    &self.estimate_service_range(service, predicate).await?,
                ),
                overlap_percentage: coverage_score * 100.0, // Convert to percentage
                estimated_result_count: self
                    .estimate_range_result_count(service, predicate, range)
                    .await?,
                confidence: ConfidenceLevel::Medium, // Default confidence level
            }))
        } else {
            Ok(None)
        }
    }

    /// Estimate how well a service covers a value range
    async fn estimate_range_coverage(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
    ) -> Result<f64> {
        // Simplified range coverage estimation
        // This would be enhanced with actual service metadata and statistics

        let base_coverage = if service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            0.5
        } else {
            0.0
        };

        // Boost for temporal capabilities on date ranges
        let temporal_boost = if range.data_type.contains("temporal")
            || range.data_type.contains("date")
                && service
                    .capabilities
                    .contains(&ServiceCapability::TemporalQueries)
        {
            0.3
        } else {
            0.0
        };

        // Domain-specific boost
        let domain_boost = if let Some(ref desc) = service.metadata.description {
            self.calculate_range_domain_affinity(desc, predicate, range)
        } else {
            0.0
        };

        Ok((base_coverage + temporal_boost + domain_boost).min(1.0))
    }

    /// Calculate domain affinity for range queries
    fn calculate_range_domain_affinity(
        &self,
        description: &str,
        predicate: &str,
        range: &ValueRange,
    ) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut affinity = 0.0;

        // Check range type based on data_type field
        if range.data_type.contains("temporal") || range.data_type.contains("date") {
            if predicate.contains("date") || predicate.contains("time") {
                if desc_lower.contains("historical") || desc_lower.contains("temporal") {
                    affinity += 0.2;
                }
            }
        } else if range.is_numeric || range.data_type.contains("numeric") {
            if predicate.contains("price") || predicate.contains("value") {
                if desc_lower.contains("economic") || desc_lower.contains("financial") {
                    affinity += 0.2;
                }
            }
        } else if range.data_type.contains("geospatial") || range.data_type.contains("geo") {
            if desc_lower.contains("geographic") || desc_lower.contains("spatial") {
                affinity += 0.3;
            }
        }

        affinity
    }

    /// Estimate result count for range query
    async fn estimate_range_result_count(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
    ) -> Result<u64> {
        let base_count = self
            .estimate_predicate_result_count(service, predicate)
            .await?;

        // Apply range selectivity factor based on data type
        let selectivity =
            if range.data_type.contains("temporal") || range.data_type.contains("date") {
                // For temporal ranges, use a simple heuristic based on range width
                0.3 // Default temporal selectivity
            } else if range.is_numeric || range.data_type.contains("numeric") {
                // Parse numeric values for better selectivity estimation
                if let (Ok(min), Ok(max)) = (
                    range.min_value.parse::<f64>(),
                    range.max_value.parse::<f64>(),
                ) {
                    let range_width = max - min;
                    if range_width < 100.0 {
                        0.1
                    } else if range_width < 1000.0 {
                        0.3
                    } else {
                        0.7
                    }
                } else {
                    0.5 // Default if parsing fails
                }
            } else if range.data_type.contains("geospatial") || range.data_type.contains("geo") {
                // Geospatial ranges vary widely in selectivity
                0.2
            } else {
                0.5 // Default selectivity for unknown types
            };

        Ok((base_count as f64 * selectivity) as u64)
    }

    /// Estimate the value range covered by a service for a predicate
    async fn estimate_service_range(
        &self,
        _service: &FederatedService,
        _predicate: &str,
    ) -> Result<ValueRange> {
        // This would query service metadata in practice
        // For now, return a default wide range
        Ok(ValueRange {
            min_value: "0.0".to_string(),
            max_value: "1000000.0".to_string(),
            data_type: "numeric".to_string(),
            is_numeric: true,
            sample_values: vec![
                "100.0".to_string(),
                "1000.0".to_string(),
                "10000.0".to_string(),
            ],
        })
    }

    /// Classify how a query range overlaps with service range
    fn classify_range_overlap(
        &self,
        query_range: &ValueRange,
        service_range: &ValueRange,
    ) -> RangeOverlapType {
        // Simplified overlap classification
        // In practice, this would handle proper range intersection logic
        if query_range.is_numeric && service_range.is_numeric {
            if let (Ok(qmin), Ok(qmax), Ok(smin), Ok(smax)) = (
                query_range.min_value.parse::<f64>(),
                query_range.max_value.parse::<f64>(),
                service_range.min_value.parse::<f64>(),
                service_range.max_value.parse::<f64>(),
            ) {
                if qmin >= smin && qmax <= smax {
                    RangeOverlapType::Complete
                } else if qmax < smin || qmin > smax {
                    RangeOverlapType::None
                } else {
                    RangeOverlapType::Partial
                }
            } else {
                RangeOverlapType::None
            }
        } else {
            // For non-numeric ranges, do string comparison as fallback
            if query_range.min_value >= service_range.min_value
                && query_range.max_value <= service_range.max_value
            {
                RangeOverlapType::Complete
            } else {
                RangeOverlapType::Partial
            }
        }
    }

    /// Bloom filter-based membership testing for efficient source selection
    #[cfg(feature = "caching")]
    pub fn create_service_bloom_filters(
        &self,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, ServiceBloomFilter>> {
        #[cfg(feature = "caching")]
        use bloom::{BloomFilter, ASMS};

        let mut filters = HashMap::new();

        for service in services {
            // Create Bloom filter for predicates
            let mut predicate_filter = BloomFilter::with_rate(0.01, 10000);

            // Create Bloom filter for subjects/objects
            let mut resource_filter = BloomFilter::with_rate(0.01, 100000);

            // Populate filters based on service capabilities and known data
            if let Some(service_meta) = registry.get_service(&service.endpoint) {
                // Add known data patterns
                for pattern in &service_meta.data_patterns {
                    predicate_filter.insert(&pattern);
                    resource_filter.insert(&pattern);
                }

                // Use metadata tags as patterns
                for tag in &service_meta.metadata.tags {
                    resource_filter.insert(&tag);
                }
            }

            let service_filter = ServiceBloomFilter {
                service_id: service.id.clone(),
                predicate: "default".to_string(),
                filter_data: Vec::new(),
                hash_functions: 3,
                false_positive_rate: 0.01,
                estimated_cardinality: 10000,
                predicate_filter,
                resource_filter,
                last_updated: chrono::Utc::now(),
                estimated_elements: 10000,
            };

            filters.insert(service.endpoint.clone(), service_filter);
        }

        info!("Created Bloom filters for {} services", filters.len());
        Ok(filters)
    }

    /// Use Bloom filters for fast membership testing
    pub fn test_pattern_membership(
        &self,
        pattern: &TriplePattern,
        service_filters: &HashMap<String, ServiceBloomFilter>,
    ) -> HashMap<String, BloomFilterResult> {
        let mut results = HashMap::new();

        for (service_endpoint, filter) in service_filters {
            let mut likely_matches = Vec::new();

            // Test predicate membership
            if let Some(predicate) = &pattern.predicate {
                if filter.predicate_filter.contains(predicate) {
                    likely_matches.push("predicate".to_string());
                }
            }

            // Test subject membership
            if let Some(subject) = &pattern.subject {
                if !subject.starts_with('?') && filter.resource_filter.contains(subject) {
                    likely_matches.push("subject".to_string());
                }
            }

            // Test object membership
            if let Some(object) = &pattern.object {
                if !object.starts_with('?') && filter.resource_filter.contains(object) {
                    likely_matches.push("object".to_string());
                }
            }

            let membership_probability = if likely_matches.is_empty() {
                0.0
            } else {
                // Calculate probability based on number of matches
                let base_prob = likely_matches.len() as f64 / 3.0;
                // Adjust for false positive rate
                base_prob * (1.0 - filter.false_positive_rate)
            };

            results.insert(
                service_endpoint.clone(),
                BloomFilterResult {
                    service_id: service_endpoint.clone(),
                    predicate: pattern
                        .predicate
                        .as_ref()
                        .unwrap_or(&"unknown".to_string())
                        .clone(),
                    possibly_contains: membership_probability > 0.5,
                    confidence: 1.0 - filter.false_positive_rate,
                    estimated_selectivity: membership_probability,
                    membership_probability,
                    likely_matches: membership_probability > 0.7,
                    false_positive_rate: filter.false_positive_rate,
                },
            );
        }

        results
    }

    /// Machine learning-based source prediction using historical data
    pub async fn predict_best_sources_ml(
        &self,
        patterns: &[TriplePattern],
        query_context: &QueryContext,
        historical_data: &[HistoricalQueryData],
    ) -> Result<Vec<MLSourcePrediction>> {
        use std::collections::HashMap;
        use tracing::info;

        // Simple ML model using pattern matching and historical performance
        let mut predictions = Vec::new();

        // Extract features from query patterns
        let features = self.extract_query_features(patterns, query_context);

        // Find similar historical queries
        let similar_queries = self.find_similar_queries(&features, historical_data);

        // Score services based on historical performance
        let mut service_scores = HashMap::new();

        for similar_query in &similar_queries {
            let similarity_weight = similar_query.similarity_score;

            // Use services_used and success_rate from similar query
            for service in &similar_query.services_used {
                let current_score = service_scores.get(service).unwrap_or(&0.0);
                let weighted_performance = similar_query.success_rate
                    * (1.0 / (similar_query.execution_time.as_millis() as f64 / 1000.0).max(0.1))
                    * similarity_weight;

                service_scores.insert(service.clone(), current_score + weighted_performance);
            }
        }

        // Create predictions
        for (service, score) in service_scores {
            let prediction = MLSourcePrediction {
                service_id: service.clone(),
                predicted_score: score.min(1.0),
                confidence: 0.8, // Default confidence
                model_version: "simple_pattern_matching_v1.0".to_string(),
                features_used: vec!["pattern_count".to_string(), "similarity".to_string()],
            };
            predictions.push(prediction);
        }

        // Sort by predicted score
        predictions.sort_by(|a, b| {
            b.predicted_score
                .partial_cmp(&a.predicted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Generated ML predictions for {} services",
            predictions.len()
        );
        Ok(predictions)
    }

    /// Extract features from query patterns for ML
    fn extract_query_features(
        &self,
        patterns: &[TriplePattern],
        context: &QueryContext,
    ) -> QueryFeatures {
        let mut predicate_counts = HashMap::new();
        let mut namespace_counts = HashMap::new();
        let mut pattern_types = HashMap::new();

        for pattern in patterns {
            // Count predicates
            if let Some(predicate) = &pattern.predicate {
                *predicate_counts.entry(predicate.clone()).or_insert(0) += 1;

                // Extract namespace
                if let Some(namespace) = self.extract_namespace(predicate) {
                    *namespace_counts.entry(namespace).or_insert(0) += 1;
                }
            }

            // Classify pattern type
            let pattern_type = self.classify_pattern_type(pattern);
            *pattern_types.entry(pattern_type).or_insert(0) += 1;
        }

        QueryFeatures {
            pattern_count: patterns.len(),
            variable_count: self.count_variables(patterns),
            join_count: self.count_joins(patterns),
            filter_count: 0,     // TODO: implement filter counting
            has_optional: false, // TODO: implement optional detection
            has_union: false,    // TODO: implement union detection
            complexity_score: self.calculate_patterns_complexity(patterns),
            estimated_selectivity: self.estimate_query_selectivity(patterns),
            selectivity_estimate: self.estimate_query_selectivity(patterns),
            predicate_distribution: predicate_counts,
            namespace_distribution: namespace_counts,
            pattern_type_distribution: pattern_types,
            has_joins: self.has_join_patterns(patterns),
            query_type: format!("{:?}", context.query_type),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Find similar queries in historical data
    fn find_similar_queries(
        &self,
        features: &QueryFeatures,
        historical_data: &[HistoricalQueryData],
    ) -> Vec<SimilarQuery> {
        let mut similar_queries = Vec::new();

        for historical_query in historical_data {
            // For simplicity, use a basic similarity calculation
            let similarity = 0.7; // Placeholder similarity score

            if similarity > 0.5 {
                // Threshold for similarity
                similar_queries.push(SimilarQuery {
                    query_id: historical_query.query_id.clone(),
                    similarity_score: similarity,
                    execution_time: historical_query.execution_time,
                    services_used: historical_query.services_used.clone(),
                    success_rate: if historical_query.success { 1.0 } else { 0.0 },
                });
            }
        }

        // Sort by similarity
        similar_queries.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        similar_queries.into_iter().take(10).collect() // Top 10 similar queries
    }

    /// Calculate similarity between two query feature sets
    fn calculate_query_similarity(
        &self,
        features1: &QueryFeatures,
        features2: &QueryFeatures,
    ) -> f64 {
        let mut similarity_factors = Vec::new();

        // Pattern count similarity
        let pattern_count_sim = 1.0
            - ((features1.pattern_count as f64 - features2.pattern_count as f64).abs()
                / features1.pattern_count.max(features2.pattern_count) as f64);
        similarity_factors.push(pattern_count_sim * 0.2);

        // Predicate distribution similarity (Jaccard similarity)
        let predicate_sim = self.calculate_jaccard_similarity(
            &features1.predicate_distribution,
            &features2.predicate_distribution,
        );
        similarity_factors.push(predicate_sim * 0.4);

        // Namespace distribution similarity
        let namespace_sim = self.calculate_jaccard_similarity(
            &features1.namespace_distribution,
            &features2.namespace_distribution,
        );
        similarity_factors.push(namespace_sim * 0.3);

        // Complexity similarity
        let complexity_sim = 1.0
            - ((features1.complexity_score - features2.complexity_score).abs()
                / features1.complexity_score.max(features2.complexity_score));
        similarity_factors.push(complexity_sim * 0.1);

        similarity_factors.iter().sum()
    }

    /// Calculate Jaccard similarity between two hash maps
    fn calculate_jaccard_similarity<T: std::hash::Hash + Eq>(
        &self,
        map1: &HashMap<T, usize>,
        map2: &HashMap<T, usize>,
    ) -> f64 {
        let keys1: HashSet<_> = map1.keys().collect();
        let keys2: HashSet<_> = map2.keys().collect();

        let intersection = keys1.intersection(&keys2).count();
        let union = keys1.union(&keys2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Predict latency for a service based on features
    fn predict_latency(&self, _service: &str, features: &QueryFeatures) -> f64 {
        // Simple linear model based on complexity
        let base_latency = 100.0; // Base latency in ms
        let complexity_factor = features.complexity_score * 50.0;
        let pattern_factor = features.pattern_count as f64 * 10.0;

        base_latency + complexity_factor + pattern_factor
    }

    /// Predict success rate for a service
    fn predict_success_rate(&self, _service: &str, features: &QueryFeatures) -> f64 {
        // Base success rate with adjustments for complexity
        let base_rate = 0.95;
        let complexity_penalty = features.complexity_score * 0.1;

        (base_rate - complexity_penalty).max(0.1)
    }

    /// Calculate feature importance for ML model
    fn calculate_feature_importance(&self, _features: &QueryFeatures) -> HashMap<String, f64> {
        let mut importance = HashMap::new();

        importance.insert("pattern_count".to_string(), 0.2);
        importance.insert("predicate_distribution".to_string(), 0.4);
        importance.insert("namespace_distribution".to_string(), 0.2);
        importance.insert("complexity_score".to_string(), 0.15);
        importance.insert("has_joins".to_string(), 0.05);

        importance
    }

    /// Helper methods for feature extraction
    fn extract_namespace(&self, uri: &str) -> Option<String> {
        if let Some(hash_pos) = uri.rfind('#') {
            Some(uri[..hash_pos].to_string())
        } else if let Some(slash_pos) = uri.rfind('/') {
            Some(uri[..slash_pos].to_string())
        } else {
            None
        }
    }

    fn classify_pattern_type(&self, pattern: &TriplePattern) -> String {
        match (
            pattern.subject.as_ref().map(|s| s.starts_with('?')),
            pattern.predicate.as_ref().map(|p| p.starts_with('?')),
            pattern.object.as_ref().map(|o| o.starts_with('?')),
        ) {
            (Some(false), Some(false), Some(false)) => "concrete".to_string(),
            (Some(true), Some(false), Some(true)) => "predicate_bound".to_string(),
            (Some(false), Some(true), Some(false)) => "subject_object_bound".to_string(),
            (Some(true), Some(true), Some(true)) => "all_variables".to_string(),
            _ => "mixed".to_string(),
        }
    }

    fn calculate_patterns_complexity(&self, patterns: &[TriplePattern]) -> f64 {
        let variable_count: usize = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if let Some(ref subject) = p.subject {
                    if subject.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref predicate) = p.predicate {
                    if predicate.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref object) = p.object {
                    if object.starts_with('?') {
                        count += 1;
                    }
                }
                count
            })
            .sum();

        (variable_count as f64) / (patterns.len() as f64 * 3.0)
    }

    fn estimate_query_selectivity(&self, patterns: &[TriplePattern]) -> f64 {
        // Simple selectivity estimation based on bound variables
        let total_positions = patterns.len() * 3;
        let bound_positions = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if let Some(ref subject) = p.subject {
                    if !subject.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref predicate) = p.predicate {
                    if !predicate.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref object) = p.object {
                    if !object.starts_with('?') {
                        count += 1;
                    }
                }
                count
            })
            .sum::<usize>();

        (bound_positions as f64) / (total_positions as f64)
    }

    fn has_join_patterns(&self, patterns: &[TriplePattern]) -> bool {
        let variables: HashSet<String> = patterns
            .iter()
            .flat_map(|p| {
                let mut vars = Vec::new();
                if let Some(ref subject) = p.subject {
                    if subject.starts_with('?') {
                        vars.push(subject.clone());
                    }
                }
                if let Some(ref predicate) = p.predicate {
                    if predicate.starts_with('?') {
                        vars.push(predicate.clone());
                    }
                }
                if let Some(ref object) = p.object {
                    if object.starts_with('?') {
                        vars.push(object.clone());
                    }
                }
                vars
            })
            .collect();

        // If we have shared variables across patterns, we likely have joins
        let total_variable_occurrences: usize = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if let Some(ref subject) = p.subject {
                    if subject.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref predicate) = p.predicate {
                    if predicate.starts_with('?') {
                        count += 1;
                    }
                }
                if let Some(ref object) = p.object {
                    if object.starts_with('?') {
                        count += 1;
                    }
                }
                count
            })
            .sum();

        total_variable_occurrences > variables.len()
    }

    /// Count the number of unique variables in patterns
    fn count_variables(&self, patterns: &[TriplePattern]) -> usize {
        let mut variables = HashSet::new();
        for pattern in patterns {
            if let Some(ref subject) = pattern.subject {
                if subject.starts_with('?') {
                    variables.insert(subject.clone());
                }
            }
            if let Some(ref predicate) = pattern.predicate {
                if predicate.starts_with('?') {
                    variables.insert(predicate.clone());
                }
            }
            if let Some(ref object) = pattern.object {
                if object.starts_with('?') {
                    variables.insert(object.clone());
                }
            }
        }
        variables.len()
    }

    /// Count the number of joins based on shared variables
    fn count_joins(&self, patterns: &[TriplePattern]) -> usize {
        if patterns.len() <= 1 {
            return 0;
        }

        let mut variable_patterns: HashMap<String, Vec<usize>> = HashMap::new();

        // Track which patterns each variable appears in
        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            if let Some(ref subject) = pattern.subject {
                if subject.starts_with('?') {
                    variable_patterns
                        .entry(subject.clone())
                        .or_insert_with(Vec::new)
                        .push(pattern_idx);
                }
            }
            if let Some(ref predicate) = pattern.predicate {
                if predicate.starts_with('?') {
                    variable_patterns
                        .entry(predicate.clone())
                        .or_insert_with(Vec::new)
                        .push(pattern_idx);
                }
            }
            if let Some(ref object) = pattern.object {
                if object.starts_with('?') {
                    variable_patterns
                        .entry(object.clone())
                        .or_insert_with(Vec::new)
                        .push(pattern_idx);
                }
            }
        }

        // Count joins (shared variables create joins)
        let mut join_count = 0;
        for (_variable, pattern_indices) in variable_patterns {
            if pattern_indices.len() > 1 {
                join_count += pattern_indices.len() - 1; // n patterns sharing a variable = n-1 joins
            }
        }
        join_count
    }
}
