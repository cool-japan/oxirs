//! Advanced cost-based selection algorithms
//!
//! This module implements sophisticated cost-based optimization techniques for federated queries,
//! including ML-based estimation, network latency modeling, and multi-objective optimization.

use anyhow::Result;
use chrono::Timelike;
use std::time::Duration;
use tracing::debug;

use crate::planner::TriplePattern;
use crate::{service_registry::ServiceRegistry, FederatedService};

use super::{types::*, ServiceOptimizer};

impl ServiceOptimizer {
    /// Advanced result size estimation using statistical models
    pub fn estimate_result_size_advanced(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
    ) -> Result<u64> {
        let mut estimated_size = 1000; // Base estimate

        // Use historical statistics if available
        if let Some(predicate) = &pattern.predicate {
            if let Some(stats) = self.get_predicate_stats(predicate) {
                estimated_size = stats.avg_result_size;

                // Adjust based on pattern selectivity
                let selectivity =
                    self.calculate_pattern_selectivity_local(pattern, service, registry);
                estimated_size = (estimated_size as f64 * selectivity) as u64;
            }
        }

        // Apply service-specific factors
        let service_factor = self.get_service_result_size_factor_local(service, registry);
        estimated_size = (estimated_size as f64 * service_factor) as u64;

        // Consider triple pattern complexity
        let complexity_factor = match self.calculate_pattern_complexity(pattern) {
            PatternComplexity::Simple => 0.8, // Simple patterns typically return fewer results
            PatternComplexity::Medium => 1.0,
            PatternComplexity::Complex => 1.5, // Complex patterns may return more results
        };

        estimated_size = (estimated_size as f64 * complexity_factor) as u64;

        // Apply machine learning-based size prediction if available
        if let Ok(ml_estimate) = self.estimate_result_size_ml_local(pattern, service, registry) {
            // Blend statistical and ML estimates
            estimated_size = ((estimated_size as f64 * 0.6) + (ml_estimate as f64 * 0.4)) as u64;
        }

        // Apply range-based adjustments for numeric/temporal predicates
        if let Ok(range_factor) = self.estimate_range_selectivity_factor_local(pattern, service) {
            estimated_size = (estimated_size as f64 * range_factor) as u64;
        }

        Ok(estimated_size.max(1)) // At least 1 result
    }

    /// Machine learning-based result size estimation
    pub fn estimate_result_size_ml_local(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<u64> {
        // Extract pattern features for ML model
        let features = self.extract_pattern_features(pattern, service)?;

        // Simple linear model - in production this would use a trained model
        let mut predicted_size = 1000.0; // Base prediction

        // Feature weights learned from historical data
        predicted_size += features.predicate_frequency * 200.0;
        predicted_size *= features.subject_specificity;
        predicted_size *= features.object_specificity;
        predicted_size += features.service_data_size_factor * 500.0;

        // Apply pattern type multipliers
        if pattern.subject.as_ref().is_some_and(|s| s.starts_with('?'))
            && pattern.object.as_ref().is_some_and(|o| o.starts_with('?'))
        {
            predicted_size *= 2.0; // More variables = more results
        }

        Ok(predicted_size.max(1.0) as u64)
    }

    /// Extract features for ML-based estimation
    fn extract_pattern_features(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> Result<PatternFeatures> {
        let predicate_frequency = if let Some(predicate) = &pattern.predicate {
            if let Some(stats) = self.get_predicate_stats(predicate) {
                stats.frequency as f64 / 10000.0 // Normalize
            } else {
                0.5 // Default moderate frequency
            }
        } else {
            0.5
        };

        let subject_specificity = if let Some(ref subject) = pattern.subject {
            if subject.starts_with('?') {
                0.8 // Variable is less specific
            } else if subject.starts_with("http://") {
                0.3 // URI is very specific
            } else {
                0.5 // Literal has medium specificity
            }
        } else {
            1.0 // No subject means very specific
        };

        let object_specificity = if let Some(ref object) = pattern.object {
            if object.starts_with('?') {
                0.8 // Variable is less specific
            } else if object.starts_with("http://") {
                0.3 // URI is very specific
            } else {
                0.5 // Literal has medium specificity
            }
        } else {
            1.0 // No object means very specific
        };

        // Estimate service data size factor based on performance metrics
        let service_data_size_factor =
            if let Some(avg_time) = service.performance.average_response_time {
                (avg_time.as_millis() as f64 / 1000.0).min(2.0) // Cap at 2x factor
            } else {
                1.0
            };

        Ok(PatternFeatures {
            predicate_frequency,
            subject_specificity,
            object_specificity,
            service_data_size_factor,
            pattern_complexity: PatternComplexity::Medium, // Default complexity
            has_variables: true,                           // Default assumption for patterns
            is_star_pattern: false,                        // Default to false, can be enhanced
        })
    }

    /// Estimate range selectivity factor for numeric/temporal predicates
    pub fn estimate_range_selectivity_factor_local(
        &self,
        pattern: &TriplePattern,
        _service: &FederatedService,
    ) -> Result<f64> {
        // Check if predicate suggests numeric or temporal data
        let predicate_lower = pattern
            .predicate
            .as_ref()
            .map_or(String::new(), |p| p.to_lowercase());

        if predicate_lower.contains("age")
            || predicate_lower.contains("year")
            || predicate_lower.contains("date")
            || predicate_lower.contains("time")
            || predicate_lower.contains("count")
            || predicate_lower.contains("number")
        {
            // These predicates typically have range constraints
            return Ok(0.3); // Higher selectivity due to range filtering
        }

        if predicate_lower.contains("name")
            || predicate_lower.contains("title")
            || predicate_lower.contains("label")
        {
            // Text predicates are less selective
            return Ok(0.8);
        }

        Ok(1.0) // Default - no range adjustment
    }

    /// Calculate pattern selectivity (0.0 to 1.0)
    fn calculate_pattern_selectivity_local(
        &self,
        pattern: &TriplePattern,
        _service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut selectivity: f64 = 1.0;

        // More specific patterns are more selective (return fewer results)
        if let Some(subject) = &pattern.subject {
            if !subject.starts_with('?') {
                selectivity *= 0.1; // Specific subject reduces results significantly
            }
        }

        if let Some(predicate) = &pattern.predicate {
            if !predicate.starts_with('?') {
                selectivity *= 0.3; // Specific predicate is moderately selective
            }
        }

        if let Some(object) = &pattern.object {
            if !object.starts_with('?') {
                selectivity *= 0.2; // Specific object is quite selective
            }
        }

        selectivity
    }

    /// Multi-objective optimization for cost vs quality trade-offs  
    pub async fn optimize_source_selection_multi_objective(
        &self,
        patterns: &[TriplePattern],
        candidate_services: &[FederatedService],
        registry: &ServiceRegistry,
        weights: &OptimizationWeights,
    ) -> Result<Vec<OptimizedServiceSelection>> {
        debug!("Starting multi-objective source selection optimization");

        let mut optimized_selections = Vec::new();

        for pattern in patterns {
            let mut service_scores = Vec::new();

            for service in candidate_services {
                // Calculate multiple objectives
                let cost_score = self
                    .calculate_cost_objective(pattern, service, registry)
                    .await?;
                let quality_score = self.calculate_quality_objective(pattern, service, registry)?;
                let latency_score = self.calculate_latency_objective(pattern, service, registry)?;
                let reliability_score =
                    self.calculate_reliability_objective(pattern, service, registry)?;

                // Weighted combination of objectives
                let combined_score = (cost_score * weights.network_cost_weight)
                    + (quality_score * weights.result_quality_weight)
                    + (latency_score * weights.execution_time_weight)
                    + (reliability_score * weights.service_reliability_weight);

                service_scores.push(ServiceObjectiveScore {
                    service_id: service.id.clone(),
                    execution_time_score: latency_score,
                    quality_score,
                    cost_score,
                    reliability_score,
                    latency_score,
                    total_score: combined_score,
                });
            }

            // Sort by total score (higher is better)
            service_scores.sort_by(|a, b| {
                b.total_score
                    .partial_cmp(&a.total_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Select top services based on Pareto optimality
            let pareto_optimal = self.find_pareto_optimal_services(&service_scores);

            optimized_selections.push(OptimizedServiceSelection {
                selected_services: pareto_optimal
                    .iter()
                    .map(|s| s.service_id.clone())
                    .collect(),
                total_score: pareto_optimal.first().map(|s| s.total_score).unwrap_or(0.0),
                metadata: OptimizationMetadata {
                    algorithm_used: "multi_objective_weighted".to_string(),
                    optimization_time: std::time::Instant::now().elapsed(),
                    alternatives_considered: candidate_services.len(),
                    confidence_level: ConfidenceLevel::High,
                    factors_considered: vec![
                        "cost".to_string(),
                        "quality".to_string(),
                        "latency".to_string(),
                        "reliability".to_string(),
                    ],
                },
                execution_plan: format!(
                    "Multi-objective optimization selected {} services",
                    pareto_optimal.len()
                ),
                estimated_cost: pareto_optimal.iter().map(|s| s.cost_score).sum(),
            });
        }

        Ok(optimized_selections)
    }

    /// Dynamic source ranking with real-time updates
    pub fn update_dynamic_source_rankings(
        &mut self,
        performance_updates: &[ServicePerformanceUpdate],
        registry: &ServiceRegistry,
    ) -> Result<()> {
        debug!(
            "Updating dynamic source rankings with {} performance updates",
            performance_updates.len()
        );

        for update in performance_updates {
            // Create performance metrics from update data
            let metrics = ServicePerformanceMetrics {
                response_time_ms: update.execution_time.as_millis() as f64,
                success_rate: if update.success { 1.0 } else { 0.0 },
                throughput_qps: 0.0, // Default value
                error_count: if update.success { 0 } else { 1 },
                last_updated: update.timestamp,
                data_quality_score: 1.0, // Default value
                availability_score: if update.success { 1.0 } else { 0.0 },
                cpu_utilization: None,
                memory_utilization: None,
            };

            // Update service performance metrics
            self.update_service_performance(&update.service_id, &metrics);

            // Get previous ranking for comparison (default to 0.5 until public API is available)
            let previous_ranking = 0.5;

            // Recalculate ranking scores
            if let Some(service) = registry.get_service(&update.service_id) {
                let new_ranking = self.calculate_dynamic_ranking_score(&service, registry)?;
                self.update_service_ranking(&update.service_id, new_ranking);

                // Trigger re-ranking of related services if significant change
                if (new_ranking - previous_ranking).abs() > 0.1 {
                    self.trigger_related_service_reranking(&update.service_id, registry)?;
                }
            }
        }

        Ok(())
    }

    /// Calculate cost objective (lower cost = higher score)
    async fn calculate_cost_objective(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let estimated_cost = self
            .estimate_service_cost(service, std::slice::from_ref(pattern), &[])
            .await;
        // Normalize and invert (lower cost = higher score)
        Ok(1.0 / (1.0 + estimated_cost / 1000.0))
    }

    /// Calculate quality objective based on data completeness and accuracy
    fn calculate_quality_objective(
        &self,
        _pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        // Use service metadata to determine quality score
        let mut quality_score = 0.7; // Base quality

        // Consider historical success rate
        if let Some(success_rate) = service.performance.success_rate {
            quality_score += success_rate * 0.2;
        }

        // Consider data freshness
        if let Some(last_update) = service.performance.last_updated {
            let age_hours = chrono::Utc::now()
                .signed_duration_since(last_update)
                .num_hours();
            let freshness_factor = (1.0 / (1.0 + age_hours as f64 / 24.0)).min(1.0);
            quality_score += freshness_factor * 0.1;
        }

        Ok(quality_score.min(1.0))
    }

    /// Calculate latency objective (lower latency = higher score)
    fn calculate_latency_objective(
        &self,
        _pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let avg_latency_ms = service
            .performance
            .average_response_time
            .map(|d| d.as_millis() as f64)
            .unwrap_or(1000.0);

        // Normalize latency (lower latency = higher score)
        Ok(1.0 / (1.0 + avg_latency_ms / 1000.0))
    }

    /// Calculate reliability objective based on uptime and error rates
    fn calculate_reliability_objective(
        &self,
        _pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut reliability_score = 0.8; // Base reliability

        // Factor in success rate
        if let Some(success_rate) = service.performance.success_rate {
            reliability_score = success_rate;
        }

        // Factor in error rate
        if let Some(error_rate) = service.performance.error_rate {
            reliability_score *= 1.0 - error_rate;
        }

        Ok(reliability_score.min(1.0))
    }

    /// Find Pareto optimal services (non-dominated solutions)
    fn find_pareto_optimal_services(
        &self,
        service_scores: &[ServiceObjectiveScore],
    ) -> Vec<ServiceObjectiveScore> {
        let mut pareto_optimal = Vec::new();

        for candidate in service_scores {
            let mut is_dominated = false;

            // Check if any other service dominates this candidate
            for other in service_scores {
                if other.service_id != candidate.service_id {
                    // A service dominates if it's better in all objectives
                    if other.cost_score >= candidate.cost_score
                        && other.quality_score >= candidate.quality_score
                        && other.latency_score >= candidate.latency_score
                        && other.reliability_score >= candidate.reliability_score
                        && (other.cost_score > candidate.cost_score
                            || other.quality_score > candidate.quality_score
                            || other.latency_score > candidate.latency_score
                            || other.reliability_score > candidate.reliability_score)
                    {
                        is_dominated = true;
                        break;
                    }
                }
            }

            if !is_dominated {
                pareto_optimal.push(candidate.clone());
            }
        }

        // Sort by total score for easier selection
        pareto_optimal.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pareto_optimal
    }

    /// Calculate dynamic ranking score based on current performance
    fn calculate_dynamic_ranking_score(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut score = 0.5; // Base score

        // Factor in recent performance
        if let Some(avg_time) = service.performance.average_response_time {
            let time_factor = 1.0 / (1.0 + avg_time.as_millis() as f64 / 1000.0);
            score += time_factor * 0.3;
        }

        // Factor in success rate
        if let Some(success_rate) = service.performance.success_rate {
            score += success_rate * 0.2;
        }

        Ok(score.min(1.0))
    }

    /// Trigger re-ranking of services related to the updated service
    fn trigger_related_service_reranking(
        &mut self,
        _service_id: &str,
        _registry: &ServiceRegistry,
    ) -> Result<()> {
        // In a full implementation, this would identify related services
        // (e.g., services with similar data patterns or that are often used together)
        // and trigger their re-ranking as well
        debug!("Triggering related service re-ranking");
        Ok(())
    }

    /// Calculate selectivity estimate for a pattern
    #[allow(dead_code)]
    fn calculate_selectivity_estimate(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> f64 {
        let mut selectivity: f64 = 0.8; // Base selectivity

        if let Some(predicate) = &pattern.predicate {
            if !predicate.starts_with('?') {
                selectivity *= 0.3; // Specific predicate reduces results moderately
            }
        }

        if let Some(object) = &pattern.object {
            if !object.starts_with('?') {
                selectivity *= 0.2; // Specific object reduces results significantly
            }
        }

        // Consider service characteristics
        if let Some(ref description) = service.metadata.description {
            // Specialized services might have higher selectivity for their domain
            if let Some(predicate) = &pattern.predicate {
                if predicate.contains("foaf:") && description.to_lowercase().contains("social") {
                    selectivity *= 0.5;
                }
            }
        }

        selectivity.max(0.001) // Minimum selectivity to avoid zero results
    }

    /// Get service-specific result size factor
    fn get_service_result_size_factor_local(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut factor = 1.0;

        // Large services might return more results
        if let Some(ref description) = service.metadata.description {
            if description.to_lowercase().contains("large")
                || description.to_lowercase().contains("comprehensive")
            {
                factor *= 2.0;
            } else if description.to_lowercase().contains("small")
                || description.to_lowercase().contains("specialized")
            {
                factor *= 0.5;
            }
        }

        // Consider service performance as indicator of dataset size
        if let Some(avg_time) = service.performance.average_response_time {
            // Slower services might have larger datasets
            let time_millis = avg_time.as_millis() as f64;
            if time_millis > 1000.0 {
                factor *= 1.5;
            } else if time_millis < 100.0 {
                factor *= 0.7;
            }
        }

        factor
    }

    /// Advanced network latency modeling
    pub fn estimate_network_latency_advanced(
        &self,
        service: &FederatedService,
        request_size: u64,
        expected_response_size: u64,
        registry: &ServiceRegistry,
    ) -> Result<Duration> {
        // Base latency from service performance data
        let mut base_latency_ms = if let Some(avg_time) = service.performance.average_response_time
        {
            avg_time.as_millis() as f64
        } else {
            100.0 // Default estimate
        };

        // Adjust for request size (larger queries take longer)
        let request_factor = 1.0 + (request_size as f64 / 1000.0) * 0.1;
        base_latency_ms *= request_factor;

        // Adjust for expected response size (more data to transfer)
        let response_factor = 1.0 + (expected_response_size as f64 / 10000.0) * 0.2;
        base_latency_ms *= response_factor;

        // Consider network conditions based on service location
        let network_factor = self.estimate_network_conditions(service, registry);
        base_latency_ms *= network_factor;

        // Apply current load factor
        let load_factor = self.get_service_load_factor(service, registry);
        base_latency_ms *= load_factor;

        Ok(Duration::from_millis(base_latency_ms as u64))
    }

    /// Enhanced network latency estimation with geographic and temporal factors
    pub fn estimate_network_latency_enhanced(
        &self,
        service: &FederatedService,
        request_size: u64,
        expected_response_size: u64,
        _registry: &ServiceRegistry,
        current_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<Duration> {
        // Base latency from service performance data
        let mut base_latency_ms = if let Some(avg_time) = service.performance.average_response_time
        {
            avg_time.as_millis() as f64
        } else {
            100.0 // Default estimate
        };

        // Geographic distance estimation
        let geographic_factor = self.estimate_geographic_latency_factor(service)?;
        base_latency_ms *= geographic_factor;

        // Bandwidth and data transfer estimation
        let transfer_latency =
            self.estimate_transfer_latency(request_size, expected_response_size, service)?;
        base_latency_ms += transfer_latency;

        // Time-of-day network congestion factor
        let congestion_factor = self.estimate_network_congestion_factor(current_time, service);
        base_latency_ms *= congestion_factor;

        // Service tier and CDN optimization factor
        let service_tier_factor = self.estimate_service_tier_factor(service);
        base_latency_ms *= service_tier_factor;

        // Historical latency variance and confidence interval
        let (adjusted_latency, confidence) =
            self.apply_latency_confidence_adjustment(base_latency_ms, service);

        debug!(
            "Enhanced latency estimate for {}: {:.2}ms (confidence: {:.2})",
            service.endpoint, adjusted_latency, confidence
        );

        Ok(Duration::from_millis(adjusted_latency as u64))
    }

    /// Estimate geographic latency factor based on service location
    fn estimate_geographic_latency_factor(&self, service: &FederatedService) -> Result<f64> {
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Local/internal services
                if host == "localhost"
                    || host.starts_with("127.")
                    || host.starts_with("192.168.")
                    || host.starts_with("10.")
                {
                    return Ok(0.1); // Very low latency for local services
                }

                // Estimate by domain characteristics
                let domain_factor = self.estimate_domain_geographic_factor(host);
                return Ok(domain_factor);
            }
        }
        Ok(1.0) // Default factor for unknown locations
    }

    /// Estimate geographic factor based on domain characteristics
    fn estimate_domain_geographic_factor(&self, host: &str) -> f64 {
        // Country-specific TLDs (simplified estimation)
        if host.ends_with(".com") || host.ends_with(".org") || host.ends_with(".net") {
            return 1.0; // Assume global CDN for common domains
        }

        // Geographic hints in domain names
        let host_lower = host.to_lowercase();
        if host_lower.contains("us") || host_lower.contains("america") {
            return 0.8; // Assume same continent (North America)
        }
        if host_lower.contains("eu") || host_lower.contains("europe") {
            return 1.2; // Cross-Atlantic
        }
        if host_lower.contains("asia") || host_lower.contains("jp") || host_lower.contains("cn") {
            return 1.5; // Cross-Pacific
        }
        if host_lower.contains("au") || host_lower.contains("oceania") {
            return 1.8; // Long distance to Australia/Oceania
        }

        // CDN providers (typically optimized routing)
        if host_lower.contains("cloudflare")
            || host_lower.contains("fastly")
            || host_lower.contains("akamai")
            || host_lower.contains("amazonaws")
            || host_lower.contains("googleusercontent")
            || host_lower.contains("azure")
        {
            return 0.7; // CDN optimization
        }

        1.0 // Default for unknown geographic location
    }

    /// Estimate data transfer latency based on size and service characteristics
    fn estimate_transfer_latency(
        &self,
        request_size: u64,
        response_size: u64,
        service: &FederatedService,
    ) -> Result<f64> {
        // Estimate bandwidth based on service characteristics
        let estimated_bandwidth_mbps = self.estimate_service_bandwidth(service);

        // Convert sizes to megabits
        let request_mb = (request_size as f64) / 125000.0; // bytes to megabits
        let response_mb = (response_size as f64) / 125000.0;

        // Calculate transfer time in milliseconds
        let request_transfer_ms = (request_mb / estimated_bandwidth_mbps) * 1000.0;
        let response_transfer_ms = (response_mb / estimated_bandwidth_mbps) * 1000.0;

        // Add protocol overhead (TCP handshake, HTTP headers, etc.)
        let protocol_overhead_ms = 20.0;

        Ok(request_transfer_ms + response_transfer_ms + protocol_overhead_ms)
    }

    /// Estimate service bandwidth based on service characteristics
    fn estimate_service_bandwidth(&self, service: &FederatedService) -> f64 {
        // Default bandwidth assumption: 10 Mbps for typical service
        let mut bandwidth_mbps: f64 = 10.0;

        // Adjust based on performance metrics
        if let Some(avg_time) = service.performance.average_response_time {
            let time_ms = avg_time.as_millis() as f64;

            // Fast services likely have better infrastructure
            if time_ms < 100.0 {
                bandwidth_mbps *= 2.0; // High-performance service
            } else if time_ms > 1000.0 {
                bandwidth_mbps *= 0.5; // Potentially constrained service
            }
        }

        // Adjust based on endpoint characteristics
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                let host_lower = host.to_lowercase();

                // CDN services typically have higher bandwidth
                if host_lower.contains("cloudflare")
                    || host_lower.contains("fastly")
                    || host_lower.contains("amazonaws")
                    || host_lower.contains("azure")
                {
                    bandwidth_mbps *= 3.0;
                }

                // HTTPS might indicate better infrastructure
                if url.scheme() == "https" {
                    bandwidth_mbps *= 1.2;
                }
            }
        }

        bandwidth_mbps.max(1.0) // Minimum 1 Mbps
    }

    /// Estimate network congestion factor based on time of day
    fn estimate_network_congestion_factor(
        &self,
        current_time: chrono::DateTime<chrono::Utc>,
        service: &FederatedService,
    ) -> f64 {
        let hour = current_time.hour();

        // Estimate primary timezone based on service characteristics
        let primary_timezone_offset = self.estimate_service_timezone_offset(service);
        let local_hour = ((hour as i32) + primary_timezone_offset).rem_euclid(24) as u32;

        // Peak hours typically have more network congestion

        match local_hour {
            0..=5 => 0.8,   // Night: lower congestion
            6..=8 => 1.3,   // Morning peak
            9..=11 => 1.1,  // Business hours
            12..=13 => 1.4, // Lunch peak
            14..=17 => 1.2, // Afternoon business
            18..=20 => 1.5, // Evening peak
            21..=23 => 1.0, // Evening
            _ => 1.0,
        }
    }

    /// Estimate service timezone offset (simplified heuristic)
    fn estimate_service_timezone_offset(&self, service: &FederatedService) -> i32 {
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                let host_lower = host.to_lowercase();

                // Simple geographic timezone estimation
                if host_lower.contains("eu") || host_lower.contains("europe") {
                    return 1; // CET
                }
                if host_lower.contains("asia") || host_lower.contains("jp") {
                    return 9; // JST
                }
                if host_lower.contains("au") {
                    return 10; // AEST
                }
                if host_lower.contains("us") || host_lower.contains("america") {
                    return -5; // EST (approximate)
                }
            }
        }
        0 // Default to UTC
    }

    /// Estimate service tier and optimization factor
    fn estimate_service_tier_factor(&self, service: &FederatedService) -> f64 {
        let mut factor = 1.0;

        // Check for enterprise/premium indicators
        if let Some(description) = &service.metadata.description {
            let desc_lower = description.to_lowercase();
            if desc_lower.contains("enterprise")
                || desc_lower.contains("premium")
                || desc_lower.contains("pro")
            {
                factor *= 0.8; // Premium services likely have better performance
            }
            if desc_lower.contains("free") || desc_lower.contains("trial") {
                factor *= 1.3; // Free services may have limitations
            }
        }

        // Check endpoint for service tier indicators
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Dedicated domains suggest better performance
                if host.contains("api") || host.contains("sparql") {
                    factor *= 0.9;
                }
                // Subdomains might indicate shared infrastructure
                if host.split('.').count() > 2 {
                    factor *= 1.1;
                }
            }
        }

        factor
    }

    /// Apply latency confidence adjustment based on historical variance
    fn apply_latency_confidence_adjustment(
        &self,
        base_latency: f64,
        service: &FederatedService,
    ) -> (f64, f64) {
        // Simple confidence model - in production would use historical variance data
        let confidence = if service.performance.average_response_time.is_some() {
            0.8 // Good confidence if we have performance data
        } else {
            0.5 // Lower confidence for unknown services
        };

        // Apply confidence interval - add padding for uncertainty
        let uncertainty_padding = base_latency * (1.0 - confidence) * 0.5;
        let adjusted_latency = base_latency + uncertainty_padding;

        (adjusted_latency, confidence)
    }

    /// Estimate network conditions based on service characteristics
    fn estimate_network_conditions(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        // Parse service endpoint to estimate network distance/quality
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Local/internal services are faster
                if host == "localhost" || host.starts_with("127.") || host.starts_with("192.168.") {
                    return 0.5;
                }

                // Well-known fast CDN services
                if host.contains("cloudflare")
                    || host.contains("fastly")
                    || host.contains("amazonaws")
                {
                    return 0.8;
                }

                // Assume reasonable network for HTTPS, slower for HTTP
                if url.scheme() == "https" {
                    return 1.0;
                } else {
                    return 1.2;
                }
            }
        }

        1.1 // Default slightly higher latency for unknown services
    }

    /// Get service load factor based on current capacity utilization
    fn get_service_load_factor(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        // Use average response time as proxy for current load
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;

            // Higher response times suggest higher load
            if millis > 2000.0 {
                return 2.0; // Very high load
            } else if millis > 1000.0 {
                return 1.5; // High load
            } else if millis > 500.0 {
                return 1.2; // Medium load
            } else {
                return 1.0; // Normal load
            }
        }

        // Use max concurrent requests as additional load indicator
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 10 {
                return 1.3; // Low capacity suggests potential issues
            }
        }

        1.0 // Default normal load
    }

    /// Advanced service capacity analysis
    pub fn analyze_service_capacity(
        &self,
        service: &FederatedService,
        current_query_load: u32,
        _registry: &ServiceRegistry,
    ) -> Result<ServiceCapacityAnalysis> {
        let mut analysis = ServiceCapacityAnalysis {
            service_id: service.id.clone(),
            current_load: current_query_load as f64,
            max_capacity: 100.0, // Default estimate
            utilization_percentage: (current_query_load as f64 / 100.0) * 100.0,
            projected_capacity: 100.0 * 1.2, // 20% growth projection
            bottleneck_factors: Vec::new(),
            max_concurrent_queries: 100, // Default estimate
            current_utilization: 0.0,
            scaling_suggestions: Vec::new(),
            recommended_max_load: 80,
        };

        // Estimate maximum capacity based on performance characteristics
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;

            // Faster services can handle more concurrent requests
            analysis.max_concurrent_queries = if millis < 100.0 {
                200
            } else if millis < 500.0 {
                100
            } else if millis < 1000.0 {
                50
            } else {
                20
            };
        }

        // Calculate current utilization
        analysis.current_utilization =
            current_query_load as f64 / analysis.max_concurrent_queries as f64;

        // Identify bottleneck factors
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 20 {
                analysis.bottleneck_factors.push(
                    "Low concurrent request capacity indicates potential bottleneck".to_string(),
                );
            }
        }

        if let Some(avg_time) = service.performance.average_response_time {
            if avg_time.as_millis() > 2000 {
                analysis
                    .bottleneck_factors
                    .push("High response time indicates performance bottleneck".to_string());
            }
        }

        // Generate scaling suggestions
        if analysis.current_utilization > 0.8 {
            analysis
                .scaling_suggestions
                .push("Consider adding replica services".to_string());
            analysis
                .scaling_suggestions
                .push("Implement query caching".to_string());
        }

        if analysis.current_utilization > 0.9 {
            analysis
                .scaling_suggestions
                .push("URGENT: Service approaching capacity limit".to_string());
        }

        // Adjust recommended max load based on service capacity
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 50 {
                analysis.recommended_max_load = 60; // Be more conservative with low-capacity services
            }
        }

        Ok(analysis)
    }

    /// Multi-objective cost optimization combining multiple factors
    pub fn calculate_multi_objective_cost(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
        objectives: &CostObjectives,
    ) -> Result<CostScore> {
        let mut score = CostScore {
            service_id: service.id.clone(),
            execution_cost: 0.0,
            network_cost: 0.0,
            quality_penalty: 0.0,
            reliability_bonus: 0.0,
            total_cost: 0.0,
        };

        // Execution time cost
        let estimated_time = self.estimate_execution_time(pattern, service, registry)?;
        score.execution_cost =
            estimated_time.as_millis() as f64 * objectives.weight_balance.execution_time_weight;

        // Network latency cost
        let request_size = self.estimate_request_size(pattern)?;
        let response_size = self.estimate_result_size_advanced(pattern, service, registry)?;
        let network_latency =
            self.estimate_network_latency_advanced(service, request_size, response_size, registry)?;
        score.network_cost =
            network_latency.as_millis() as f64 * objectives.weight_balance.network_cost_weight;

        // Resource usage cost (based on query complexity and service load)
        let complexity_enum = self.calculate_pattern_complexity_cost(pattern);
        let complexity_cost = match complexity_enum {
            PatternComplexity::Simple => 1.0,
            PatternComplexity::Medium => 2.0,
            PatternComplexity::Complex => 4.0,
        };
        let load_factor = self.get_service_load_factor(service, registry);
        // Add resource usage as part of execution cost
        score.execution_cost +=
            complexity_cost * load_factor * objectives.weight_balance.execution_time_weight;

        // Reliability bonus (bonus for high-capacity services)
        let reliability_score =
            if let Some(max_requests) = service.performance.max_concurrent_requests {
                if max_requests < 50 {
                    0.0 // No bonus for low-capacity services
                } else {
                    50.0 // Bonus for high-capacity services
                }
            } else {
                0.0 // No bonus for unknown capacity
            };
        score.reliability_bonus = reliability_score
            * if objectives.maximize_reliability {
                1.0
            } else {
                0.0
            };

        // Quality penalty (penalty for low-quality services)
        let quality_score = self.calculate_service_quality_score(service, registry);
        score.quality_penalty = (100.0 - quality_score)
            * if objectives.maximize_quality {
                1.0
            } else {
                0.0
            };

        // Calculate total cost (lower is better)
        score.total_cost = score.execution_cost + score.network_cost + score.quality_penalty
            - score.reliability_bonus; // Subtract reliability bonus since it's a benefit

        Ok(score)
    }

    /// Estimate execution time for a pattern on a service
    fn estimate_execution_time(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<Duration> {
        let mut base_time_ms = if let Some(avg_time) = service.performance.average_response_time {
            avg_time.as_millis() as f64
        } else {
            200.0 // Default estimate
        };

        // Adjust for pattern complexity
        let complexity_factor = match self.calculate_pattern_complexity_cost(pattern) {
            PatternComplexity::Simple => 0.8,
            PatternComplexity::Medium => 1.0,
            PatternComplexity::Complex => 2.0,
        };

        base_time_ms *= complexity_factor;

        Ok(Duration::from_millis(base_time_ms as u64))
    }

    /// Estimate request size for network cost calculation
    fn estimate_request_size(&self, pattern: &TriplePattern) -> Result<u64> {
        // Base SPARQL query overhead
        let mut size = 100;

        // Add pattern string length
        size += pattern.pattern_string.len() as u64;

        // Add complexity overhead
        if pattern.pattern_string.contains("FILTER") {
            size += 50;
        }
        if pattern.pattern_string.contains("OPTIONAL") {
            size += 30;
        }

        Ok(size)
    }

    /// Calculate complexity cost for resource usage estimation
    #[allow(dead_code)]
    fn calculate_pattern_cost_factor(&self, pattern: &TriplePattern) -> f64 {
        let mut cost = 10.0; // Base cost

        // Variable patterns are more expensive
        if pattern.subject.as_ref().is_some_and(|s| s.starts_with('?')) {
            cost += 5.0;
        }
        if pattern
            .predicate
            .as_ref()
            .is_some_and(|p| p.starts_with('?'))
        {
            cost += 10.0; // Predicate variables are very expensive
        }
        if pattern.object.as_ref().is_some_and(|o| o.starts_with('?')) {
            cost += 5.0;
        }

        // Complex pattern features
        if pattern.pattern_string.contains("REGEX") {
            cost += 20.0;
        }
        if pattern.pattern_string.contains("PropertyPath") {
            cost += 15.0;
        }

        cost
    }

    /// Calculate service quality score
    fn calculate_service_quality_score(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut score = 0.0;

        // Capacity contribution (use max concurrent requests as quality indicator)
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            score += (max_requests as f64).min(100.0); // 0-100 points for capacity
        }

        // Response time contribution (faster is better)
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;
            let time_score = (2000.0 - millis.min(2000.0)) / 20.0; // 0-100 points, capped at 2 seconds
            score += time_score.max(0.0);
        }

        // Capability richness
        let capability_score = service.capabilities.len() as f64 * 5.0; // 5 points per capability
        score += capability_score;

        score
    }

    /// Calculate pattern complexity for cost analysis (renamed to avoid conflict)
    fn calculate_pattern_complexity_cost(&self, pattern: &TriplePattern) -> PatternComplexity {
        let mut complexity_score = 0;

        // Variable patterns increase complexity
        if pattern.subject.as_ref().is_some_and(|s| s.starts_with('?')) {
            complexity_score += 1;
        }
        if pattern
            .predicate
            .as_ref()
            .is_some_and(|p| p.starts_with('?'))
        {
            complexity_score += 2; // Predicate variables are more complex
        }
        if pattern.object.as_ref().is_some_and(|o| o.starts_with('?')) {
            complexity_score += 1;
        }

        // Check for complex patterns in the pattern string
        let pattern_str = &pattern.pattern_string;
        if pattern_str.contains("REGEX") || pattern_str.contains("FILTER") {
            complexity_score += 3;
        }
        if pattern_str.contains("OPTIONAL") || pattern_str.contains("UNION") {
            complexity_score += 2;
        }
        if pattern_str.contains("PropertyPath")
            || pattern_str.contains("*")
            || pattern_str.contains("+")
        {
            complexity_score += 4;
        }

        // Classify complexity based on score
        match complexity_score {
            0..=2 => PatternComplexity::Simple,
            3..=5 => PatternComplexity::Medium,
            _ => PatternComplexity::Complex,
        }
    }
}
