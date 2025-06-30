//! Enhanced Service Optimizer with ML-driven Pattern Analysis
//!
//! This module provides an enhanced service optimizer that integrates advanced pattern
//! analysis, ML-driven recommendations, and sophisticated optimization strategies.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    query_decomposition::advanced_pattern_analysis::{
        AdvancedPatternAnalyzer, PatternAnalysisResult, AdvancedAnalysisConfig,
        OptimizationOpportunity, ServiceRecommendation,
    },
    planner::planning::{TriplePattern, FilterExpression},
    service_optimizer::types::QueryFeatures,
    service_optimizer::{
        ServiceOptimizer, ServiceOptimizerConfig,
        types::{
            OptimizedQuery, OptimizedServiceClause, ServiceExecutionStrategy,
            ExecutionStrategy, CrossServiceJoin, JoinType, PatternFeatures,
            MLSourcePrediction, ServicePerformanceUpdate,
        },
    },
    FederatedService, ServiceCapability,
};

/// Enhanced service optimizer with ML-driven pattern analysis
#[derive(Debug)]
pub struct EnhancedServiceOptimizer {
    /// Base service optimizer
    base_optimizer: ServiceOptimizer,
    /// Advanced pattern analyzer
    pattern_analyzer: AdvancedPatternAnalyzer,
    /// Configuration
    config: EnhancedOptimizerConfig,
    /// Performance history for learning
    performance_history: Arc<RwLock<Vec<ServicePerformanceUpdate>>>,
    /// Optimization cache
    optimization_cache: Arc<RwLock<HashMap<String, CachedOptimization>>>,
}

impl EnhancedServiceOptimizer {
    /// Create a new enhanced service optimizer
    pub fn new() -> Self {
        Self {
            base_optimizer: ServiceOptimizer::new(),
            pattern_analyzer: AdvancedPatternAnalyzer::new(),
            config: EnhancedOptimizerConfig::default(),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        base_config: ServiceOptimizerConfig,
        analysis_config: AdvancedAnalysisConfig,
    ) -> Self {
        Self {
            base_optimizer: ServiceOptimizer::with_config(base_config),
            pattern_analyzer: AdvancedPatternAnalyzer::with_config(analysis_config),
            config: EnhancedOptimizerConfig::default(),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize query with advanced pattern analysis
    pub async fn optimize_query_with_analysis(
        &self,
        patterns: Vec<TriplePattern>,
        filters: Vec<FilterExpression>,
        services: &[FederatedService],
    ) -> Result<EnhancedOptimizationResult> {
        info!("Starting enhanced optimization for {} patterns across {} services", 
              patterns.len(), services.len());

        // Check optimization cache first
        let cache_key = self.generate_cache_key(&patterns, &filters);
        if let Some(cached) = self.get_cached_optimization(&cache_key).await {
            if !cached.is_expired() {
                info!("Using cached optimization result");
                return Ok(cached.result);
            }
        }

        // Perform advanced pattern analysis
        let analysis_result = self.pattern_analyzer
            .analyze_query_patterns(&patterns, &filters, services)
            .await?;

        // Apply ML-driven service selection
        let ml_recommendations = self.apply_ml_recommendations(&analysis_result, services).await?;

        // Generate optimized query using enhanced strategies
        let optimized_query = self.generate_enhanced_query(
            patterns.clone(),
            filters.clone(),
            &analysis_result,
            &ml_recommendations,
            services,
        ).await?;

        // Create execution plan with optimization recommendations
        let execution_plan = self.create_execution_plan(&optimized_query, &analysis_result).await?;

        // Create comprehensive result
        let result = EnhancedOptimizationResult {
            optimized_query,
            analysis_result,
            ml_recommendations,
            execution_plan: execution_plan.clone(),
            optimization_metadata: self.create_optimization_metadata(&patterns, &filters),
            performance_predictions: self.predict_performance(&execution_plan, services).await?,
        };

        // Cache the result
        self.cache_optimization(&cache_key, &result).await;

        info!("Enhanced optimization completed successfully");
        Ok(result)
    }

    /// Apply ML-driven service recommendations
    async fn apply_ml_recommendations(
        &self,
        analysis: &PatternAnalysisResult,
        services: &[FederatedService],
    ) -> Result<Vec<MLServiceRecommendation>> {
        let mut recommendations = Vec::new();

        for service_rec in &analysis.service_recommendations {
            let pattern_id = &service_rec.pattern_id;
            
            // Get the top recommended services for this pattern
            let top_services: Vec<_> = service_rec.recommended_services
                .iter()
                .take(self.config.max_services_per_pattern)
                .cloned()
                .collect();

            // Apply ML scoring if enabled
            let mut ml_scores = HashMap::new();
            if self.config.enable_ml_scoring {
                for (service_id, base_score) in &top_services {
                    if let Some(service) = services.iter().find(|s| &s.id == service_id) {
                        let ml_score = self.calculate_ml_score(service, pattern_id, *base_score).await?;
                        ml_scores.insert(service_id.clone(), ml_score);
                    }
                }
            }

            recommendations.push(MLServiceRecommendation {
                pattern_id: pattern_id.clone(),
                base_recommendations: top_services,
                ml_enhanced_scores: ml_scores,
                confidence: service_rec.confidence,
                reasoning: self.enhance_reasoning(&service_rec.reasoning, &analysis.optimization_opportunities),
            });
        }

        Ok(recommendations)
    }

    /// Generate enhanced optimized query
    async fn generate_enhanced_query(
        &self,
        patterns: Vec<TriplePattern>,
        filters: Vec<FilterExpression>,
        analysis: &PatternAnalysisResult,
        ml_recommendations: &[MLServiceRecommendation],
        services: &[FederatedService],
    ) -> Result<OptimizedQuery> {
        let mut optimized_services = Vec::new();
        let mut cross_service_joins = Vec::new();

        // Generate optimized service clauses based on ML recommendations
        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            let pattern_id = format!("pattern_{}", pattern_idx);
            
            if let Some(ml_rec) = ml_recommendations.iter().find(|r| r.pattern_id == pattern_id) {
                let best_service = self.select_best_service_for_pattern(ml_rec, services)?;
                
                let optimized_clause = OptimizedServiceClause {
                    service_id: best_service.id.clone(),
                    endpoint: best_service.endpoint.clone(),
                    patterns: vec![pattern.clone()],
                    filters: self.extract_applicable_filters(&filters, pattern),
                    pushed_filters: self.identify_pushdown_filters(&filters, pattern),
                    strategy: self.determine_execution_strategy(&best_service, pattern, analysis),
                    estimated_cost: self.estimate_service_cost(&best_service, pattern, analysis),
                    capabilities: best_service.capabilities.clone(),
                };

                optimized_services.push(optimized_clause);
            }
        }

        // Identify cross-service joins
        cross_service_joins = self.identify_cross_service_joins(&optimized_services, &analysis.join_graph_analysis);

        // Determine overall execution strategy
        let execution_strategy = self.determine_overall_execution_strategy(analysis, &optimized_services);

        // Calculate total estimated cost
        let estimated_cost: f64 = optimized_services.iter().map(|s| s.estimated_cost).sum();

        Ok(OptimizedQuery {
            services: optimized_services,
            global_filters: self.identify_global_filters(&filters),
            cross_service_joins,
            execution_strategy,
            estimated_cost,
        })
    }

    /// Create comprehensive execution plan
    async fn create_execution_plan(
        &self,
        optimized_query: &OptimizedQuery,
        analysis: &PatternAnalysisResult,
    ) -> Result<EnhancedExecutionPlan> {
        let mut steps = Vec::new();
        let mut parallelizable_groups = Vec::new();

        // Create service execution steps
        for (idx, service_clause) in optimized_query.services.iter().enumerate() {
            let step = ExecutionStep {
                step_id: format!("service_execution_{}", idx),
                step_type: ExecutionStepType::ServiceQuery,
                service_id: service_clause.service_id.clone(),
                estimated_duration: self.estimate_step_duration(&service_clause),
                dependencies: Vec::new(),
                parallelizable: self.is_step_parallelizable(&service_clause, &optimized_query.cross_service_joins),
                resource_requirements: self.estimate_resource_requirements(&service_clause),
                optimization_hints: self.generate_optimization_hints(&service_clause, analysis),
            };
            steps.push(step);
        }

        // Create join execution steps
        for (idx, join) in optimized_query.cross_service_joins.iter().enumerate() {
            let step = ExecutionStep {
                step_id: format!("join_execution_{}", idx),
                step_type: ExecutionStepType::Join,
                service_id: format!("{}+{}", join.left_service, join.right_service),
                estimated_duration: self.estimate_join_duration(join),
                dependencies: vec![
                    format!("service_execution_{}", idx),
                    format!("service_execution_{}", idx + 1),
                ],
                parallelizable: false, // Joins typically depend on their inputs
                resource_requirements: self.estimate_join_resource_requirements(join),
                optimization_hints: vec!["use_hash_join".to_string()],
            };
            steps.push(step);
        }

        // Identify parallelizable groups
        parallelizable_groups = self.identify_parallelizable_groups(&steps);

        let total_duration = self.estimate_total_duration(&steps);
        let resource_requirements = self.aggregate_resource_requirements(&steps);
        let risk_assessment = self.assess_execution_risks(&steps);
        
        Ok(EnhancedExecutionPlan {
            steps,
            parallelizable_groups,
            estimated_total_duration: total_duration,
            resource_requirements,
            optimization_strategy: optimized_query.execution_strategy.clone(),
            risk_assessment,
            fallback_strategies: self.generate_fallback_strategies(&optimized_query),
        })
    }

    /// Predict performance for the execution plan
    async fn predict_performance(
        &self,
        plan: &EnhancedExecutionPlan,
        services: &[FederatedService],
    ) -> Result<PerformancePrediction> {
        let mut service_predictions = HashMap::new();

        // Predict performance for each service
        for step in &plan.steps {
            if let ExecutionStepType::ServiceQuery = step.step_type {
                if let Some(service) = services.iter().find(|s| s.id == step.service_id) {
                    let prediction = self.predict_service_performance(service, &step).await?;
                    service_predictions.insert(step.service_id.clone(), prediction);
                }
            }
        }

        // Aggregate overall predictions
        let total_estimated_time = service_predictions.values()
            .map(|p| p.estimated_execution_time.as_secs_f64())
            .sum::<f64>();

        let success_probability = service_predictions.values()
            .map(|p| p.success_probability)
            .fold(1.0, |acc, prob| acc * prob);

        Ok(PerformancePrediction {
            service_predictions,
            total_estimated_time: std::time::Duration::from_secs_f64(total_estimated_time),
            success_probability,
            bottleneck_analysis: self.identify_bottlenecks(plan),
            confidence_interval: (0.8, 1.2), // ±20% confidence interval
        })
    }

    // Helper methods

    async fn calculate_ml_score(
        &self,
        service: &FederatedService,
        pattern_id: &str,
        base_score: f64,
    ) -> Result<f64> {
        // Simplified ML scoring - in practice would use actual ML model
        let performance_factor = (1000.0 - service.performance.avg_response_time_ms.min(1000.0)) / 1000.0;
        let reliability_factor = service.performance.reliability_score;
        
        let ml_score = base_score * 0.5 + performance_factor * 0.3 + reliability_factor * 0.2;
        Ok(ml_score.min(1.0).max(0.0))
    }

    fn enhance_reasoning(&self, base_reasoning: &str, opportunities: &[OptimizationOpportunity]) -> String {
        let mut enhanced = base_reasoning.to_string();
        
        if !opportunities.is_empty() {
            enhanced.push_str(" Additional optimization opportunities: ");
            for (idx, opp) in opportunities.iter().enumerate() {
                if idx > 0 { enhanced.push_str(", "); }
                enhanced.push_str(&opp.description);
            }
        }
        
        enhanced
    }

    fn select_best_service_for_pattern(
        &self,
        ml_rec: &MLServiceRecommendation,
        services: &[FederatedService],
    ) -> Result<FederatedService> {
        // Select based on ML-enhanced scores if available, otherwise use base recommendations
        let best_service_id = if !ml_rec.ml_enhanced_scores.is_empty() {
            ml_rec.ml_enhanced_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(id, _)| id.clone())
        } else {
            ml_rec.base_recommendations
                .first()
                .map(|(id, _)| id.clone())
        };

        if let Some(service_id) = best_service_id {
            services.iter()
                .find(|s| s.id == service_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Service {} not found", service_id))
        } else {
            Err(anyhow::anyhow!("No suitable service found for pattern"))
        }
    }

    fn extract_applicable_filters(&self, filters: &[FilterExpression], pattern: &TriplePattern) -> Vec<FilterExpression> {
        // Simple implementation - in practice would be more sophisticated
        filters.iter()
            .filter(|filter| self.filter_applies_to_pattern(filter, pattern))
            .cloned()
            .collect()
    }

    fn identify_pushdown_filters(&self, filters: &[FilterExpression], pattern: &TriplePattern) -> Vec<FilterExpression> {
        // Identify filters that can be pushed down to the service
        self.extract_applicable_filters(filters, pattern)
            .into_iter()
            .filter(|filter| self.can_pushdown_filter(filter))
            .collect()
    }

    fn determine_execution_strategy(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
        analysis: &PatternAnalysisResult,
    ) -> ServiceExecutionStrategy {
        let complexity = analysis.complexity_assessment.level.clone();
        let has_high_selectivity = analysis.estimated_selectivity < 0.1;

        ServiceExecutionStrategy {
            use_values_binding: has_high_selectivity && service.capabilities.contains(&ServiceCapability::SparqlValues),
            stream_results: matches!(complexity, crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::High | 
                                               crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::VeryHigh),
            use_subqueries: pattern.subject.is_none() && pattern.object.is_none(),
            batch_size: if has_high_selectivity { 50 } else { 100 },
            timeout_ms: analysis.complexity_assessment.estimated_execution_time.as_millis() as u64 * 2,
        }
    }

    fn estimate_service_cost(&self, service: &FederatedService, pattern: &TriplePattern, analysis: &PatternAnalysisResult) -> f64 {
        let base_cost = 10.0;
        let complexity_multiplier = match analysis.complexity_assessment.level {
            crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::Low => 0.5,
            crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::Medium => 1.0,
            crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::High => 2.0,
            crate::query_decomposition::advanced_pattern_analysis::ComplexityLevel::VeryHigh => 4.0,
        };
        let performance_factor = service.performance.avg_response_time_ms / 1000.0;
        
        base_cost * complexity_multiplier * (1.0 + performance_factor)
    }

    fn identify_cross_service_joins(
        &self,
        services: &[OptimizedServiceClause],
        join_analysis: &crate::query_decomposition::advanced_pattern_analysis::JoinGraphAnalysis,
    ) -> Vec<CrossServiceJoin> {
        let mut joins = Vec::new();

        // Simplified join identification based on shared variables
        for edge in &join_analysis.join_edges {
            if services.len() > edge.pattern1 && services.len() > edge.pattern2 {
                let left_service = &services[edge.pattern1].service_id;
                let right_service = &services[edge.pattern2].service_id;
                
                if left_service != right_service {
                    joins.push(CrossServiceJoin {
                        left_service: left_service.clone(),
                        right_service: right_service.clone(),
                        join_variables: vec![edge.shared_variable.clone()],
                        join_type: JoinType::Inner,
                        estimated_selectivity: edge.estimated_selectivity,
                    });
                }
            }
        }

        joins
    }

    fn determine_overall_execution_strategy(
        &self,
        analysis: &PatternAnalysisResult,
        services: &[OptimizedServiceClause],
    ) -> ExecutionStrategy {
        if analysis.complexity_assessment.parallelization_potential > 0.7 && services.len() > 1 {
            ExecutionStrategy::ParallelWithJoin
        } else if analysis.join_graph_analysis.complexity_score > 10.0 {
            ExecutionStrategy::Sequential
        } else {
            ExecutionStrategy::Adaptive
        }
    }

    fn identify_global_filters(&self, filters: &[FilterExpression]) -> Vec<FilterExpression> {
        // Filters that apply globally and cannot be pushed down
        filters.iter()
            .filter(|filter| !self.can_pushdown_filter(filter))
            .cloned()
            .collect()
    }

    // Cache management
    fn generate_cache_key(&self, patterns: &[TriplePattern], filters: &[FilterExpression]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for pattern in patterns {
            pattern.pattern_string.hash(&mut hasher);
        }
        for filter in filters {
            filter.expression.hash(&mut hasher);
        }
        
        format!("opt_{:x}", hasher.finish())
    }

    async fn get_cached_optimization(&self, cache_key: &str) -> Option<CachedOptimization> {
        let cache = self.optimization_cache.read().await;
        cache.get(cache_key).cloned()
    }

    async fn cache_optimization(&self, cache_key: &str, result: &EnhancedOptimizationResult) {
        let mut cache = self.optimization_cache.write().await;
        cache.insert(cache_key.to_string(), CachedOptimization {
            result: result.clone(),
            cached_at: chrono::Utc::now(),
            ttl: std::time::Duration::from_secs(3600), // 1 hour
        });

        // Clean up old entries
        if cache.len() > self.config.max_cache_entries {
            let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
            cache.retain(|_, cached| cached.cached_at > cutoff);
        }
    }

    // Placeholder implementations for helper methods
    fn filter_applies_to_pattern(&self, _filter: &FilterExpression, _pattern: &TriplePattern) -> bool { true }
    fn can_pushdown_filter(&self, _filter: &FilterExpression) -> bool { true }
    fn estimate_step_duration(&self, _clause: &OptimizedServiceClause) -> std::time::Duration { std::time::Duration::from_millis(100) }
    fn is_step_parallelizable(&self, _clause: &OptimizedServiceClause, _joins: &[CrossServiceJoin]) -> bool { true }
    fn estimate_resource_requirements(&self, _clause: &OptimizedServiceClause) -> ResourceRequirements { ResourceRequirements::default() }
    fn generate_optimization_hints(&self, _clause: &OptimizedServiceClause, _analysis: &PatternAnalysisResult) -> Vec<String> { vec![] }
    fn estimate_join_duration(&self, _join: &CrossServiceJoin) -> std::time::Duration { std::time::Duration::from_millis(50) }
    fn estimate_join_resource_requirements(&self, _join: &CrossServiceJoin) -> ResourceRequirements { ResourceRequirements::default() }
    fn identify_parallelizable_groups(&self, _steps: &[ExecutionStep]) -> Vec<Vec<String>> { vec![] }
    fn estimate_total_duration(&self, _steps: &[ExecutionStep]) -> std::time::Duration { std::time::Duration::from_secs(1) }
    fn aggregate_resource_requirements(&self, _steps: &[ExecutionStep]) -> ResourceRequirements { ResourceRequirements::default() }
    fn assess_execution_risks(&self, _steps: &[ExecutionStep]) -> RiskAssessment { RiskAssessment::default() }
    fn generate_fallback_strategies(&self, _query: &OptimizedQuery) -> Vec<String> { vec![] }
    async fn predict_service_performance(&self, _service: &FederatedService, _step: &ExecutionStep) -> Result<ServicePerformancePrediction> { Ok(ServicePerformancePrediction::default()) }
    fn identify_bottlenecks(&self, _plan: &EnhancedExecutionPlan) -> BottleneckAnalysis { BottleneckAnalysis::default() }
    fn create_optimization_metadata(&self, _patterns: &[TriplePattern], _filters: &[FilterExpression]) -> OptimizationMetadata { OptimizationMetadata::default() }
}

impl Default for EnhancedServiceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// Supporting types

#[derive(Debug, Clone)]
pub struct EnhancedOptimizerConfig {
    pub enable_ml_scoring: bool,
    pub max_services_per_pattern: usize,
    pub cache_ttl_hours: u64,
    pub max_cache_entries: usize,
    pub performance_weight: f64,
    pub cost_weight: f64,
    pub reliability_weight: f64,
}

impl Default for EnhancedOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_ml_scoring: true,
            max_services_per_pattern: 3,
            cache_ttl_hours: 1,
            max_cache_entries: 1000,
            performance_weight: 0.4,
            cost_weight: 0.3,
            reliability_weight: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedOptimizationResult {
    pub optimized_query: OptimizedQuery,
    pub analysis_result: PatternAnalysisResult,
    pub ml_recommendations: Vec<MLServiceRecommendation>,
    pub execution_plan: EnhancedExecutionPlan,
    pub optimization_metadata: OptimizationMetadata,
    pub performance_predictions: PerformancePrediction,
}

#[derive(Debug, Clone)]
pub struct MLServiceRecommendation {
    pub pattern_id: String,
    pub base_recommendations: Vec<(String, f64)>,
    pub ml_enhanced_scores: HashMap<String, f64>,
    pub confidence: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct EnhancedExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub parallelizable_groups: Vec<Vec<String>>,
    pub estimated_total_duration: std::time::Duration,
    pub resource_requirements: ResourceRequirements,
    pub optimization_strategy: ExecutionStrategy,
    pub risk_assessment: RiskAssessment,
    pub fallback_strategies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: String,
    pub step_type: ExecutionStepType,
    pub service_id: String,
    pub estimated_duration: std::time::Duration,
    pub dependencies: Vec<String>,
    pub parallelizable: bool,
    pub resource_requirements: ResourceRequirements,
    pub optimization_hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ExecutionStepType {
    ServiceQuery,
    Join,
    Filter,
    Aggregate,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    pub memory_mb: u64,
    pub cpu_cores: f64,
    pub network_bandwidth_mbps: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RiskAssessment {
    pub overall_risk_score: f64,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub service_predictions: HashMap<String, ServicePerformancePrediction>,
    pub total_estimated_time: std::time::Duration,
    pub success_probability: f64,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Default)]
pub struct ServicePerformancePrediction {
    pub estimated_execution_time: std::time::Duration,
    pub success_probability: f64,
    pub estimated_result_size: u64,
    pub resource_usage: ResourceRequirements,
}

#[derive(Debug, Clone, Default)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: Option<String>,
    pub bottleneck_severity: f64,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationMetadata {
    pub optimization_time: std::time::Duration,
    pub strategies_applied: Vec<String>,
    pub confidence_score: f64,
    pub alternatives_considered: usize,
}

#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub result: EnhancedOptimizationResult,
    pub cached_at: chrono::DateTime<chrono::Utc>,
    pub ttl: std::time::Duration,
}

impl CachedOptimization {
    pub fn is_expired(&self) -> bool {
        let age = chrono::Utc::now() - self.cached_at;
        age.to_std().unwrap_or_default() > self.ttl
    }
}