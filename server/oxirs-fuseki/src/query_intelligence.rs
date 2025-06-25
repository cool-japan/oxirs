//! Query intelligence and adaptive performance optimization
//!
//! This module provides advanced query analysis, pattern learning, and
//! intelligent optimization suggestions for SPARQL queries.

use crate::{
    error::{FusekiError, FusekiResult},
    store::Store,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn, instrument};

/// Query intelligence engine with machine learning capabilities
#[derive(Clone)]
pub struct QueryIntelligenceEngine {
    query_patterns: Arc<RwLock<HashMap<String, QueryPattern>>>,
    performance_history: Arc<RwLock<Vec<QueryExecution>>>,
    optimization_rules: Arc<RwLock<Vec<OptimizationRule>>>,
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    config: IntelligenceConfig,
}

/// Query pattern learned from historical executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    pub pattern_id: String,
    pub pattern_signature: String,
    pub execution_count: u64,
    pub average_execution_time: f64,
    pub success_rate: f64,
    pub common_optimizations: Vec<String>,
    pub typical_result_size: u64,
    pub parameter_ranges: HashMap<String, ParameterRange>,
    pub discovered_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

/// Parameter value ranges for query patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub common_values: Vec<String>,
    pub data_type: String,
}

/// Query execution record for learning
#[derive(Debug, Clone, Serialize)]
pub struct QueryExecution {
    pub execution_id: String,
    pub query_hash: String,
    pub query_text: String,
    pub execution_time_ms: u64,
    pub result_count: usize,
    pub memory_usage_mb: f64,
    pub success: bool,
    pub error_message: Option<String>,
    pub optimizations_applied: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub dataset_name: Option<String>,
}

/// Dynamic optimization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    pub rule_id: String,
    pub rule_name: String,
    pub condition: RuleCondition,
    pub action: OptimizationAction,
    pub confidence_score: f64,
    pub success_rate: f64,
    pub application_count: u64,
    pub created_at: DateTime<Utc>,
    pub enabled: bool,
}

/// Condition for applying optimization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub query_pattern: Option<String>,
    pub execution_time_threshold: Option<u64>,
    pub result_size_threshold: Option<usize>,
    pub query_complexity_score: Option<f64>,
    pub time_of_day: Option<TimeRange>,
    pub user_pattern: Option<String>,
}

/// Time range for rule conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_hour: u8,
    pub end_hour: u8,
}

/// Optimization action to apply
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationAction {
    RewriteQuery(String),
    AddIndex(String),
    UseCache,
    ParallelExecution,
    MemoryOptimization,
    TimeoutAdjustment(u64),
    SuggestAlternative(String),
}

/// Anomaly detection for query performance
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    baseline_metrics: HashMap<String, BaselineMetrics>,
    anomaly_threshold: f64,
    detection_window_hours: u64,
    recent_anomalies: Vec<QueryAnomaly>,
}

/// Baseline performance metrics
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    pub average_execution_time: f64,
    pub standard_deviation: f64,
    pub median_execution_time: f64,
    pub p95_execution_time: f64,
    pub typical_result_size: f64,
    pub last_updated: DateTime<Utc>,
}

/// Detected query anomaly
#[derive(Debug, Clone, Serialize)]
pub struct QueryAnomaly {
    pub anomaly_id: String,
    pub query_hash: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub current_value: f64,
    pub expected_value: f64,
    pub deviation_factor: f64,
}

/// Types of query anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyType {
    SlowExecution,
    HighMemoryUsage,
    UnexpectedResultSize,
    FrequentFailures,
    UnusualPattern,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Intelligence configuration
#[derive(Debug, Clone)]
pub struct IntelligenceConfig {
    pub enable_pattern_learning: bool,
    pub enable_anomaly_detection: bool,
    pub enable_automatic_optimization: bool,
    pub pattern_similarity_threshold: f64,
    pub anomaly_detection_threshold: f64,
    pub max_stored_executions: usize,
    pub learning_rate: f64,
    pub confidence_threshold: f64,
}

/// Query analysis result
#[derive(Debug, Clone, Serialize)]
pub struct QueryAnalysisResult {
    pub query_complexity: QueryComplexity,
    pub predicted_performance: PredictedPerformance,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub similar_patterns: Vec<String>,
    pub risk_assessment: RiskAssessment,
}

/// Query complexity analysis
#[derive(Debug, Clone, Serialize)]
pub struct QueryComplexity {
    pub complexity_score: f64,
    pub join_complexity: f64,
    pub filter_selectivity: f64,
    pub aggregation_complexity: f64,
    pub subquery_depth: u32,
    pub estimated_cardinality: u64,
}

/// Predicted query performance
#[derive(Debug, Clone, Serialize)]
pub struct PredictedPerformance {
    pub estimated_execution_time_ms: u64,
    pub confidence_interval: (u64, u64),
    pub estimated_memory_usage_mb: f64,
    pub estimated_result_size: u64,
    pub success_probability: f64,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_difficulty: DifficultyLevel,
    pub automatic_applicable: bool,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Risk assessment for query execution
#[derive(Debug, Clone, Serialize)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub performance_risk: RiskLevel,
    pub resource_risk: RiskLevel,
    pub failure_risk: RiskLevel,
    pub risk_factors: Vec<String>,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl QueryIntelligenceEngine {
    /// Create new query intelligence engine
    pub fn new(config: IntelligenceConfig) -> Self {
        QueryIntelligenceEngine {
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            optimization_rules: Arc::new(RwLock::new(Vec::new())),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new(
                config.anomaly_detection_threshold,
                24, // 24-hour detection window
            ))),
            config,
        }
    }

    /// Analyze query before execution
    #[instrument(skip(self, query))]
    pub async fn analyze_query(&self, query: &str) -> FusekiResult<QueryAnalysisResult> {
        debug!("Analyzing query intelligence for: {}", query.chars().take(100).collect::<String>());
        
        let query_hash = self.compute_query_hash(query);
        
        // Analyze query complexity
        let complexity = self.analyze_query_complexity(query).await;
        
        // Predict performance based on historical data
        let predicted_performance = self.predict_query_performance(query, &query_hash).await;
        
        // Generate optimization suggestions
        let suggestions = self.generate_optimization_suggestions(query, &complexity).await;
        
        // Find similar patterns
        let similar_patterns = self.find_similar_patterns(query, &query_hash).await;
        
        // Assess execution risks
        let risk_assessment = self.assess_query_risks(query, &complexity, &predicted_performance).await;
        
        Ok(QueryAnalysisResult {
            query_complexity: complexity,
            predicted_performance,
            optimization_suggestions: suggestions,
            similar_patterns,
            risk_assessment,
        })
    }

    /// Record query execution for learning
    #[instrument(skip(self, execution))]
    pub async fn record_execution(&self, execution: QueryExecution) -> FusekiResult<()> {
        // Update query patterns
        self.update_query_patterns(&execution).await?;
        
        // Update performance history
        self.update_performance_history(execution.clone()).await?;
        
        // Check for anomalies
        self.detect_anomalies(&execution).await?;
        
        // Learn optimization rules
        if self.config.enable_pattern_learning {
            self.learn_optimization_rules(&execution).await?;
        }
        
        debug!("Recorded query execution: {}", execution.execution_id);
        Ok(())
    }

    /// Get intelligent optimization suggestions
    pub async fn get_optimization_suggestions(&self, query: &str) -> FusekiResult<Vec<OptimizationSuggestion>> {
        let analysis = self.analyze_query(query).await?;
        Ok(analysis.optimization_suggestions)
    }

    /// Detect performance anomalies
    pub async fn detect_performance_anomalies(&self) -> FusekiResult<Vec<QueryAnomaly>> {
        let anomaly_detector = self.anomaly_detector.read().await;
        Ok(anomaly_detector.recent_anomalies.clone())
    }

    /// Get query performance predictions
    pub async fn predict_performance(&self, query: &str) -> FusekiResult<PredictedPerformance> {
        let query_hash = self.compute_query_hash(query);
        self.predict_query_performance(query, &query_hash).await
    }

    /// Get learned query patterns
    pub async fn get_query_patterns(&self) -> HashMap<String, QueryPattern> {
        let patterns = self.query_patterns.read().await;
        patterns.clone()
    }

    /// Get intelligence statistics
    pub async fn get_intelligence_statistics(&self) -> IntelligenceStatistics {
        let patterns = self.query_patterns.read().await;
        let history = self.performance_history.read().await;
        let rules = self.optimization_rules.read().await;
        let anomaly_detector = self.anomaly_detector.read().await;
        
        let total_executions = history.len();
        let successful_executions = history.iter().filter(|e| e.success).count();
        let success_rate = if total_executions > 0 {
            successful_executions as f64 / total_executions as f64
        } else {
            0.0
        };
        
        let average_execution_time = if !history.is_empty() {
            history.iter().map(|e| e.execution_time_ms as f64).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };
        
        IntelligenceStatistics {
            total_patterns: patterns.len(),
            total_executions,
            success_rate,
            average_execution_time,
            active_rules: rules.iter().filter(|r| r.enabled).count(),
            recent_anomalies: anomaly_detector.recent_anomalies.len(),
            learning_enabled: self.config.enable_pattern_learning,
        }
    }

    // Private implementation methods

    fn compute_query_hash(&self, query: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.trim().to_lowercase().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn analyze_query_complexity(&self, query: &str) -> QueryComplexity {
        let query_lower = query.to_lowercase();
        
        // Count different complexity factors
        let join_count = query.matches(" join ").count() + query.matches(" . ").count();
        let filter_count = query_lower.matches("filter").count();
        let optional_count = query_lower.matches("optional").count();
        let union_count = query_lower.matches("union").count();
        let subquery_count = query_lower.matches("select").count().saturating_sub(1);
        let aggregation_count = self.count_aggregations(query);
        
        // Calculate complexity scores
        let join_complexity = (join_count as f64) * 2.0;
        let filter_selectivity = (filter_count as f64) * 1.5;
        let aggregation_complexity = (aggregation_count as f64) * 3.0;
        let subquery_depth = subquery_count as u32;
        
        let complexity_score = join_complexity + filter_selectivity + aggregation_complexity +
                              (optional_count as f64) * 2.5 + (union_count as f64) * 2.0 +
                              (query.len() as f64 / 100.0);
        
        QueryComplexity {
            complexity_score,
            join_complexity,
            filter_selectivity,
            aggregation_complexity,
            subquery_depth,
            estimated_cardinality: self.estimate_result_cardinality(query).await,
        }
    }

    async fn predict_query_performance(&self, query: &str, query_hash: &str) -> PredictedPerformance {
        let patterns = self.query_patterns.read().await;
        let history = self.performance_history.read().await;
        
        // Look for exact match first
        if let Some(pattern) = patterns.get(query_hash) {
            let confidence_range = (
                (pattern.average_execution_time * 0.8) as u64,
                (pattern.average_execution_time * 1.2) as u64,
            );
            
            return PredictedPerformance {
                estimated_execution_time_ms: pattern.average_execution_time as u64,
                confidence_interval: confidence_range,
                estimated_memory_usage_mb: pattern.typical_result_size as f64 * 0.001, // Simple estimate
                estimated_result_size: pattern.typical_result_size,
                success_probability: pattern.success_rate,
            };
        }
        
        // Find similar patterns
        let similar_executions: Vec<_> = history.iter()
            .filter(|e| self.compute_similarity(query, &e.query_text) > self.config.pattern_similarity_threshold)
            .collect();
        
        if !similar_executions.is_empty() {
            let avg_time = similar_executions.iter()
                .map(|e| e.execution_time_ms as f64)
                .sum::<f64>() / similar_executions.len() as f64;
            
            let avg_results = similar_executions.iter()
                .map(|e| e.result_count as f64)
                .sum::<f64>() / similar_executions.len() as f64;
            
            let success_rate = similar_executions.iter()
                .filter(|e| e.success)
                .count() as f64 / similar_executions.len() as f64;
            
            PredictedPerformance {
                estimated_execution_time_ms: avg_time as u64,
                confidence_interval: ((avg_time * 0.7) as u64, (avg_time * 1.3) as u64),
                estimated_memory_usage_mb: avg_results * 0.001,
                estimated_result_size: avg_results as u64,
                success_probability: success_rate,
            }
        } else {
            // Fallback to complexity-based estimation
            let complexity = self.analyze_query_complexity(query).await;
            let estimated_time = (complexity.complexity_score * 10.0) as u64;
            
            PredictedPerformance {
                estimated_execution_time_ms: estimated_time,
                confidence_interval: (estimated_time / 2, estimated_time * 2),
                estimated_memory_usage_mb: complexity.estimated_cardinality as f64 * 0.001,
                estimated_result_size: complexity.estimated_cardinality,
                success_probability: 0.8, // Default assumption
            }
        }
    }

    async fn generate_optimization_suggestions(&self, query: &str, complexity: &QueryComplexity) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // High complexity query suggestions
        if complexity.complexity_score > 50.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: "Query Simplification".to_string(),
                description: "Consider breaking this complex query into smaller parts".to_string(),
                expected_improvement: 0.3,
                implementation_difficulty: DifficultyLevel::Medium,
                automatic_applicable: false,
            });
        }
        
        // High join complexity
        if complexity.join_complexity > 10.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: "Index Optimization".to_string(),
                description: "Consider adding indexes for join conditions".to_string(),
                expected_improvement: 0.4,
                implementation_difficulty: DifficultyLevel::Easy,
                automatic_applicable: true,
            });
        }
        
        // Many filters
        if complexity.filter_selectivity > 5.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: "Filter Ordering".to_string(),
                description: "Reorder filters by selectivity for better performance".to_string(),
                expected_improvement: 0.2,
                implementation_difficulty: DifficultyLevel::Easy,
                automatic_applicable: true,
            });
        }
        
        // Subqueries
        if complexity.subquery_depth > 2 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: "Subquery Optimization".to_string(),
                description: "Consider flattening subqueries or using joins instead".to_string(),
                expected_improvement: 0.25,
                implementation_difficulty: DifficultyLevel::Hard,
                automatic_applicable: false,
            });
        }
        
        // Large result sets
        if complexity.estimated_cardinality > 10000 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: "Result Limiting".to_string(),
                description: "Consider adding LIMIT clause to reduce result size".to_string(),
                expected_improvement: 0.5,
                implementation_difficulty: DifficultyLevel::Easy,
                automatic_applicable: false,
            });
        }
        
        suggestions
    }

    async fn find_similar_patterns(&self, query: &str, query_hash: &str) -> Vec<String> {
        let patterns = self.query_patterns.read().await;
        let mut similar = Vec::new();
        
        for (pattern_hash, pattern) in patterns.iter() {
            if pattern_hash != query_hash {
                let similarity = self.compute_similarity(query, &pattern.pattern_signature);
                if similarity > self.config.pattern_similarity_threshold {
                    similar.push(pattern.pattern_id.clone());
                }
            }
        }
        
        similar
    }

    async fn assess_query_risks(&self, query: &str, complexity: &QueryComplexity, performance: &PredictedPerformance) -> RiskAssessment {
        let mut risk_factors = Vec::new();
        
        // Performance risk
        let performance_risk = if performance.estimated_execution_time_ms > 30000 {
            risk_factors.push("Long execution time predicted".to_string());
            RiskLevel::High
        } else if performance.estimated_execution_time_ms > 10000 {
            risk_factors.push("Moderate execution time".to_string());
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        // Resource risk
        let resource_risk = if performance.estimated_memory_usage_mb > 1000.0 {
            risk_factors.push("High memory usage expected".to_string());
            RiskLevel::High
        } else if performance.estimated_memory_usage_mb > 100.0 {
            risk_factors.push("Moderate memory usage".to_string());
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        // Failure risk
        let failure_risk = if performance.success_probability < 0.5 {
            risk_factors.push("Low success probability".to_string());
            RiskLevel::High
        } else if performance.success_probability < 0.8 {
            risk_factors.push("Moderate failure risk".to_string());
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        // Overall risk (highest of individual risks)
        let overall_risk = match (&performance_risk, &resource_risk, &failure_risk) {
            (RiskLevel::High, _, _) | (_, RiskLevel::High, _) | (_, _, RiskLevel::High) => RiskLevel::High,
            (RiskLevel::Medium, _, _) | (_, RiskLevel::Medium, _) | (_, _, RiskLevel::Medium) => RiskLevel::Medium,
            _ => RiskLevel::Low,
        };
        
        RiskAssessment {
            overall_risk,
            performance_risk,
            resource_risk,
            failure_risk,
            risk_factors,
        }
    }

    async fn update_query_patterns(&self, execution: &QueryExecution) -> FusekiResult<()> {
        let mut patterns = self.query_patterns.write().await;
        
        if let Some(pattern) = patterns.get_mut(&execution.query_hash) {
            // Update existing pattern
            pattern.execution_count += 1;
            pattern.average_execution_time = 
                (pattern.average_execution_time * (pattern.execution_count - 1) as f64 + execution.execution_time_ms as f64) / pattern.execution_count as f64;
            
            if execution.success {
                pattern.success_rate = 
                    (pattern.success_rate * (pattern.execution_count - 1) as f64 + 1.0) / pattern.execution_count as f64;
            } else {
                pattern.success_rate = 
                    (pattern.success_rate * (pattern.execution_count - 1) as f64) / pattern.execution_count as f64;
            }
            
            pattern.last_seen = execution.timestamp;
            pattern.typical_result_size = 
                ((pattern.typical_result_size as f64 + execution.result_count as f64) / 2.0) as u64;
        } else {
            // Create new pattern
            let pattern = QueryPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_signature: execution.query_text.clone(),
                execution_count: 1,
                average_execution_time: execution.execution_time_ms as f64,
                success_rate: if execution.success { 1.0 } else { 0.0 },
                common_optimizations: execution.optimizations_applied.clone(),
                typical_result_size: execution.result_count as u64,
                parameter_ranges: HashMap::new(),
                discovered_at: execution.timestamp,
                last_seen: execution.timestamp,
            };
            
            patterns.insert(execution.query_hash.clone(), pattern);
        }
        
        Ok(())
    }

    async fn update_performance_history(&self, execution: QueryExecution) -> FusekiResult<()> {
        let mut history = self.performance_history.write().await;
        
        history.push(execution);
        
        // Keep only recent executions
        if history.len() > self.config.max_stored_executions {
            let excess = history.len() - self.config.max_stored_executions;
            history.drain(0..excess);
        }
        
        Ok(())
    }

    async fn detect_anomalies(&self, execution: &QueryExecution) -> FusekiResult<()> {
        if !self.config.enable_anomaly_detection {
            return Ok(());
        }
        
        let mut anomaly_detector = self.anomaly_detector.write().await;
        
        // Update baseline metrics
        anomaly_detector.update_baseline(&execution.query_hash, execution);
        
        // Check for anomalies
        if let Some(baseline) = anomaly_detector.baseline_metrics.get(&execution.query_hash) {
            let time_deviation = (execution.execution_time_ms as f64 - baseline.average_execution_time).abs() / baseline.standard_deviation;
            
            if time_deviation > anomaly_detector.anomaly_threshold {
                let anomaly = QueryAnomaly {
                    anomaly_id: uuid::Uuid::new_v4().to_string(),
                    query_hash: execution.query_hash.clone(),
                    anomaly_type: AnomalyType::SlowExecution,
                    severity: if time_deviation > 5.0 { AnomalySeverity::Critical } 
                             else if time_deviation > 3.0 { AnomalySeverity::High }
                             else { AnomalySeverity::Medium },
                    description: format!("Execution time {} ms is {:.1}x higher than expected", 
                                       execution.execution_time_ms, time_deviation),
                    detected_at: execution.timestamp,
                    current_value: execution.execution_time_ms as f64,
                    expected_value: baseline.average_execution_time,
                    deviation_factor: time_deviation,
                };
                
                anomaly_detector.recent_anomalies.push(anomaly);
                
                // Keep only recent anomalies
                let cutoff = Utc::now() - chrono::Duration::hours(anomaly_detector.detection_window_hours as i64);
                anomaly_detector.recent_anomalies.retain(|a| a.detected_at > cutoff);
            }
        }
        
        Ok(())
    }

    async fn learn_optimization_rules(&self, execution: &QueryExecution) -> FusekiResult<()> {
        // Simplified rule learning - in a full implementation this would use ML
        if !execution.optimizations_applied.is_empty() && execution.success {
            let mut rules = self.optimization_rules.write().await;
            
            // Create rule for successful optimizations
            for optimization in &execution.optimizations_applied {
                let rule_id = format!("{}_{}", execution.query_hash, optimization);
                
                if let Some(rule) = rules.iter_mut().find(|r| r.rule_id == rule_id) {
                    rule.application_count += 1;
                    rule.success_rate = (rule.success_rate + 1.0) / 2.0; // Simple update
                } else {
                    let rule = OptimizationRule {
                        rule_id,
                        rule_name: format!("Auto-learned: {}", optimization),
                        condition: RuleCondition {
                            query_pattern: Some(execution.query_hash.clone()),
                            execution_time_threshold: None,
                            result_size_threshold: None,
                            query_complexity_score: None,
                            time_of_day: None,
                            user_pattern: None,
                        },
                        action: OptimizationAction::UseCache, // Simplified
                        confidence_score: 0.7,
                        success_rate: 1.0,
                        application_count: 1,
                        created_at: execution.timestamp,
                        enabled: true,
                    };
                    
                    rules.push(rule);
                }
            }
        }
        
        Ok(())
    }

    // Helper methods

    fn count_aggregations(&self, query: &str) -> usize {
        let query_upper = query.to_uppercase();
        ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "GROUP_CONCAT(", "SAMPLE("]
            .iter()
            .map(|agg| query_upper.matches(agg).count())
            .sum()
    }

    async fn estimate_result_cardinality(&self, query: &str) -> u64 {
        // Simplified cardinality estimation
        let complexity_factors = query.matches("?").count() * 10 + 
                                query.matches(".").count() * 5;
        (complexity_factors as u64).max(1)
    }

    fn compute_similarity(&self, query1: &str, query2: &str) -> f64 {
        // Simple Jaccard similarity on tokens
        let tokens1: std::collections::HashSet<&str> = query1.split_whitespace().collect();
        let tokens2: std::collections::HashSet<&str> = query2.split_whitespace().collect();
        
        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
}

impl AnomalyDetector {
    fn new(threshold: f64, window_hours: u64) -> Self {
        AnomalyDetector {
            baseline_metrics: HashMap::new(),
            anomaly_threshold: threshold,
            detection_window_hours: window_hours,
            recent_anomalies: Vec::new(),
        }
    }

    fn update_baseline(&mut self, query_hash: &str, execution: &QueryExecution) {
        let metrics = self.baseline_metrics.entry(query_hash.to_string()).or_insert_with(|| {
            BaselineMetrics {
                average_execution_time: execution.execution_time_ms as f64,
                standard_deviation: 0.0,
                median_execution_time: execution.execution_time_ms as f64,
                p95_execution_time: execution.execution_time_ms as f64,
                typical_result_size: execution.result_count as f64,
                last_updated: execution.timestamp,
            }
        });

        // Simple exponential moving average
        let alpha = 0.1;
        metrics.average_execution_time = 
            alpha * execution.execution_time_ms as f64 + (1.0 - alpha) * metrics.average_execution_time;
        
        // Simple standard deviation estimate
        let diff = execution.execution_time_ms as f64 - metrics.average_execution_time;
        metrics.standard_deviation = 
            alpha * diff.powi(2) + (1.0 - alpha) * metrics.standard_deviation;
        
        metrics.last_updated = execution.timestamp;
    }
}

/// Intelligence statistics
#[derive(Debug, Clone, Serialize)]
pub struct IntelligenceStatistics {
    pub total_patterns: usize,
    pub total_executions: usize,
    pub success_rate: f64,
    pub average_execution_time: f64,
    pub active_rules: usize,
    pub recent_anomalies: usize,
    pub learning_enabled: bool,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        IntelligenceConfig {
            enable_pattern_learning: true,
            enable_anomaly_detection: true,
            enable_automatic_optimization: false,
            pattern_similarity_threshold: 0.7,
            anomaly_detection_threshold: 2.0,
            max_stored_executions: 10000,
            learning_rate: 0.1,
            confidence_threshold: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_intelligence_engine_creation() {
        let config = IntelligenceConfig::default();
        let engine = QueryIntelligenceEngine::new(config);
        
        let stats = engine.get_intelligence_statistics().await;
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.total_executions, 0);
    }

    #[tokio::test]
    async fn test_query_complexity_analysis() {
        let config = IntelligenceConfig::default();
        let engine = QueryIntelligenceEngine::new(config);
        
        let simple_query = "SELECT * WHERE { ?s ?p ?o }";
        let complex_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 } FILTER(?o > 10) }";
        
        let simple_complexity = engine.analyze_query_complexity(simple_query).await;
        let complex_complexity = engine.analyze_query_complexity(complex_query).await;
        
        assert!(complex_complexity.complexity_score > simple_complexity.complexity_score);
        assert!(complex_complexity.aggregation_complexity > 0.0);
    }

    #[tokio::test]
    async fn test_pattern_recording() {
        let config = IntelligenceConfig::default();
        let engine = QueryIntelligenceEngine::new(config);
        
        let execution = QueryExecution {
            execution_id: "test1".to_string(),
            query_hash: "hash123".to_string(),
            query_text: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            execution_time_ms: 100,
            result_count: 50,
            memory_usage_mb: 10.0,
            success: true,
            error_message: None,
            optimizations_applied: vec![],
            timestamp: Utc::now(),
            user_id: None,
            dataset_name: None,
        };
        
        engine.record_execution(execution).await.unwrap();
        
        let patterns = engine.get_query_patterns().await;
        assert_eq!(patterns.len(), 1);
        assert!(patterns.contains_key("hash123"));
    }

    #[test]
    fn test_similarity_computation() {
        let config = IntelligenceConfig::default();
        let engine = QueryIntelligenceEngine::new(config);
        
        let query1 = "SELECT * WHERE { ?s ?p ?o }";
        let query2 = "SELECT * WHERE { ?s ?p ?o }";
        let query3 = "SELECT ?name WHERE { ?person foaf:name ?name }";
        
        let similarity1 = engine.compute_similarity(query1, query2);
        let similarity2 = engine.compute_similarity(query1, query3);
        
        assert_eq!(similarity1, 1.0);
        assert!(similarity2 < 1.0);
        assert!(similarity2 > 0.0);
    }
}