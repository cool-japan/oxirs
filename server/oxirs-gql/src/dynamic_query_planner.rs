//! Dynamic Query Plan Adaptation
//!
//! This module provides intelligent, runtime-adaptive query planning that automatically
//! adjusts execution strategies based on real-time performance metrics, resource availability,
//! and historical query patterns.
//!
//! ## Features
//!
//! - **Adaptive Strategy Selection**: Dynamically chooses optimal execution strategy
//! - **Performance Monitoring**: Real-time tracking of query execution metrics
//! - **Resource-Aware Planning**: Adapts to CPU, memory, and network conditions
//! - **Cost-Based Optimization**: Uses historical cost data for better plans
//! - **Fallback Strategies**: Automatic degradation under high load
//! - **Learning-Based Adaptation**: Improves over time with ML-based predictions

use anyhow::Result;
use scirs2_core::metrics::{Counter, Gauge};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;

use crate::historical_cost_estimator::HistoricalCostEstimator;
use crate::ml_optimizer::MLQueryOptimizer;

/// Configuration for dynamic query planning
#[derive(Debug, Clone)]
pub struct DynamicPlannerConfig {
    /// Enable dynamic query plan adaptation
    pub enabled: bool,
    /// Minimum execution time to trigger adaptation (ms)
    pub min_adaptation_threshold_ms: f64,
    /// CPU usage threshold for degradation (0.0-1.0)
    pub cpu_threshold: f64,
    /// Memory usage threshold for degradation (0.0-1.0)
    pub memory_threshold: f64,
    /// Enable ML-based strategy prediction
    pub enable_ml_prediction: bool,
    /// Number of recent executions to track
    pub history_size: usize,
    /// Adaptation interval (how often to reassess strategy)
    pub adaptation_interval: Duration,
    /// Enable aggressive optimization under high load
    pub aggressive_mode_enabled: bool,
}

impl Default for DynamicPlannerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_adaptation_threshold_ms: 100.0,
            cpu_threshold: 0.80,    // 80% CPU usage triggers degradation
            memory_threshold: 0.85, // 85% memory usage triggers degradation
            enable_ml_prediction: true,
            history_size: 100,
            adaptation_interval: Duration::from_secs(10),
            aggressive_mode_enabled: true,
        }
    }
}

/// Query execution strategy
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Standard sequential execution
    Sequential,
    /// Parallel field resolution
    Parallel,
    /// Batched execution with DataLoader
    Batched,
    /// Streaming results as they become available
    Streaming,
    /// Cached result (no execution needed)
    Cached,
    /// Optimized with query rewriting
    Optimized,
    /// Degraded mode for high load
    Degraded,
}

impl ExecutionStrategy {
    /// Get the estimated overhead of this strategy
    pub fn overhead_multiplier(&self) -> f64 {
        match self {
            Self::Sequential => 1.0,
            Self::Parallel => 1.2,  // Slight overhead for parallelization
            Self::Batched => 0.8,   // Reduces N+1 queries
            Self::Streaming => 1.1, // Small overhead for streaming
            Self::Cached => 0.01,   // Minimal overhead
            Self::Optimized => 0.7, // Best performance with optimization
            Self::Degraded => 1.5,  // Higher overhead, but safer under load
        }
    }

    /// Check if this strategy is suitable for high load conditions
    pub fn is_high_load_safe(&self) -> bool {
        matches!(self, Self::Degraded | Self::Cached | Self::Sequential)
    }
}

/// System resource snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_queries: usize,
    pub avg_query_time_ms: f64,
}

impl ResourceSnapshot {
    /// Check if system is under high load
    pub fn is_high_load(&self, config: &DynamicPlannerConfig) -> bool {
        self.cpu_usage > config.cpu_threshold || self.memory_usage > config.memory_threshold
    }

    /// Get load severity (0.0 = no load, 1.0 = critical)
    pub fn load_severity(&self, config: &DynamicPlannerConfig) -> f64 {
        let cpu_severity = (self.cpu_usage / config.cpu_threshold).min(1.0);
        let memory_severity = (self.memory_usage / config.memory_threshold).min(1.0);

        cpu_severity.max(memory_severity)
    }
}

/// Query plan with adaptive strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePlan {
    pub query_fingerprint: String,
    pub strategy: ExecutionStrategy,
    pub estimated_cost: f64,
    pub estimated_time_ms: f64,
    pub confidence: f64,
    pub fallback_strategy: Option<ExecutionStrategy>,
    #[serde(skip, default = "Instant::now")]
    pub created_at: Instant,
    pub metadata: HashMap<String, String>,
}

impl AdaptivePlan {
    pub fn new(
        query_fingerprint: String,
        strategy: ExecutionStrategy,
        estimated_cost: f64,
    ) -> Self {
        Self {
            query_fingerprint,
            strategy,
            estimated_cost,
            estimated_time_ms: estimated_cost * 10.0, // Rough heuristic
            confidence: 0.5,
            fallback_strategy: None,
            created_at: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_fallback(mut self, fallback: ExecutionStrategy) -> Self {
        self.fallback_strategy = Some(fallback);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_estimated_time(mut self, time_ms: f64) -> Self {
        self.estimated_time_ms = time_ms;
        self
    }
}

/// Execution result for feedback
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub query_fingerprint: String,
    pub strategy_used: ExecutionStrategy,
    pub actual_time_ms: f64,
    pub success: bool,
    pub error_message: Option<String>,
    pub resource_snapshot: ResourceSnapshot,
}

/// Dynamic query planner with adaptive strategy selection
pub struct DynamicQueryPlanner {
    config: DynamicPlannerConfig,
    cost_estimator: Arc<RwLock<HistoricalCostEstimator>>,
    ml_optimizer: Option<Arc<RwLock<MLQueryOptimizer>>>,
    execution_history: Arc<RwLock<VecDeque<ExecutionResult>>>,
    strategy_performance: Arc<RwLock<HashMap<ExecutionStrategy, StrategyStats>>>,
    system: Arc<RwLock<System>>,

    // Metrics
    adaptations: Arc<Counter>,
    strategy_switches: Arc<Counter>,
    high_load_events: Arc<Counter>,
    avg_adaptation_quality: Arc<Gauge>,
}

/// Performance statistics for each strategy
#[derive(Debug, Clone)]
pub struct StrategyStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    #[allow(dead_code)]
    pub p95_time_ms: f64,
    pub error_rate: f64,
    #[allow(dead_code)]
    last_used: Instant,
}

impl Default for StrategyStats {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            total_time_ms: 0.0,
            avg_time_ms: 0.0,
            p95_time_ms: 0.0,
            error_rate: 0.0,
            last_used: Instant::now(),
        }
    }
}

impl DynamicQueryPlanner {
    pub fn new(config: DynamicPlannerConfig) -> Self {
        // Initialize ML optimizer if enabled
        // Note: ML optimizer requires performance tracker, so we disable it for now
        // TODO: Integrate with performance tracker when available
        let ml_optimizer: Option<Arc<RwLock<MLQueryOptimizer>>> = None;

        Self {
            cost_estimator: Arc::new(RwLock::new(HistoricalCostEstimator::new())),
            ml_optimizer,
            execution_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            strategy_performance: Arc::new(RwLock::new(HashMap::new())),
            system: Arc::new(RwLock::new(System::new_all())),
            adaptations: Arc::new(Counter::new("dynamic_planner_adaptations".to_string())),
            strategy_switches: Arc::new(Counter::new(
                "dynamic_planner_strategy_switches".to_string(),
            )),
            high_load_events: Arc::new(Counter::new(
                "dynamic_planner_high_load_events".to_string(),
            )),
            avg_adaptation_quality: Arc::new(Gauge::new(
                "dynamic_planner_adaptation_quality".to_string(),
            )),
            config,
        }
    }

    /// Create an adaptive plan for a query
    pub async fn create_plan(&self, query: &str, query_complexity: f64) -> Result<AdaptivePlan> {
        if !self.config.enabled {
            // Return basic sequential plan
            return Ok(AdaptivePlan::new(
                Self::fingerprint_query(query),
                ExecutionStrategy::Sequential,
                query_complexity,
            ));
        }

        let query_fingerprint = Self::fingerprint_query(query);

        // Get current resource snapshot
        let snapshot = self.get_resource_snapshot().await?;

        // Check for high load condition
        let is_high_load = snapshot.is_high_load(&self.config);
        if is_high_load {
            self.high_load_events.inc();
        }

        // Get historical cost estimate
        let cost_estimate = {
            let estimator = self.cost_estimator.read().await;
            estimator.estimate_cost(&query_fingerprint).ok()
        };

        // Determine optimal strategy
        let strategy = self
            .select_optimal_strategy(
                &query_fingerprint,
                query_complexity,
                &snapshot,
                cost_estimate.as_ref(),
            )
            .await?;

        // Select fallback strategy
        let fallback = self.select_fallback_strategy(strategy, &snapshot);

        // Build adaptive plan
        let mut plan = AdaptivePlan::new(query_fingerprint.clone(), strategy, query_complexity)
            .with_fallback(fallback);

        // Add estimated time from historical data
        if let Some(estimate) = cost_estimate {
            plan = plan
                .with_estimated_time(estimate.estimated_time_ms)
                .with_confidence(estimate.confidence);
        }

        self.adaptations.inc();

        Ok(plan)
    }

    /// Select the optimal execution strategy based on current conditions
    async fn select_optimal_strategy(
        &self,
        query_fingerprint: &str,
        complexity: f64,
        snapshot: &ResourceSnapshot,
        cost_estimate: Option<&crate::historical_cost_estimator::CostEstimate>,
    ) -> Result<ExecutionStrategy> {
        // High load: use safe strategies
        if snapshot.is_high_load(&self.config) {
            return Ok(self.select_high_load_strategy(snapshot).await);
        }

        // Check if ML prediction is available
        if let Some(ml_opt) = &self.ml_optimizer {
            let ml = ml_opt.read().await;
            if let Ok(strategy) = self
                .ml_predict_strategy(&ml, query_fingerprint, complexity, snapshot)
                .await
            {
                return Ok(strategy);
            }
        }

        // Use historical performance to select strategy
        if let Some(estimate) = cost_estimate {
            return Ok(self
                .select_strategy_from_history(estimate, complexity, snapshot)
                .await);
        }

        // Default selection based on complexity
        Ok(self.select_default_strategy(complexity, snapshot))
    }

    /// Select strategy using ML prediction
    async fn ml_predict_strategy(
        &self,
        _ml: &MLQueryOptimizer,
        _query_fingerprint: &str,
        complexity: f64,
        snapshot: &ResourceSnapshot,
    ) -> Result<ExecutionStrategy> {
        // Create feature vector for ML prediction (currently unused)
        let _features = [
            complexity,
            snapshot.cpu_usage,
            snapshot.memory_usage,
            snapshot.active_queries as f64,
            snapshot.avg_query_time_ms,
        ];

        // Get strategy performance stats
        let stats = self.strategy_performance.read().await;

        // Score each strategy
        let mut best_strategy = ExecutionStrategy::Sequential;
        let mut best_score = f64::MIN;

        for strategy in &[
            ExecutionStrategy::Sequential,
            ExecutionStrategy::Parallel,
            ExecutionStrategy::Batched,
            ExecutionStrategy::Streaming,
            ExecutionStrategy::Optimized,
        ] {
            let score = if let Some(stat) = stats.get(strategy) {
                // Factor in success rate and average time
                let success_rate =
                    stat.successful_executions as f64 / stat.total_executions.max(1) as f64;
                let time_penalty = stat.avg_time_ms / 1000.0; // Normalize
                (success_rate * 10.0) - time_penalty
            } else {
                // No history - use default scoring
                match strategy {
                    ExecutionStrategy::Optimized => 8.0,
                    ExecutionStrategy::Parallel => 7.0,
                    ExecutionStrategy::Batched => 6.0,
                    ExecutionStrategy::Streaming => 5.0,
                    ExecutionStrategy::Sequential => 4.0,
                    ExecutionStrategy::Cached => 10.0,
                    ExecutionStrategy::Degraded => 2.0,
                }
            };

            if score > best_score {
                best_score = score;
                best_strategy = *strategy;
            }
        }

        Ok(best_strategy)
    }

    /// Select strategy based on historical performance
    async fn select_strategy_from_history(
        &self,
        estimate: &crate::historical_cost_estimator::CostEstimate,
        complexity: f64,
        snapshot: &ResourceSnapshot,
    ) -> ExecutionStrategy {
        // If query is fast, use sequential
        if estimate.estimated_time_ms < self.config.min_adaptation_threshold_ms {
            return ExecutionStrategy::Sequential;
        }

        // If query is complex and system has capacity, use parallel
        if complexity > 50.0 && !snapshot.is_high_load(&self.config) {
            return ExecutionStrategy::Parallel;
        }

        // If confidence is low, use optimized strategy to learn
        if estimate.confidence < 0.5 {
            return ExecutionStrategy::Optimized;
        }

        // Default to optimized for medium-high complexity
        if complexity > 20.0 {
            ExecutionStrategy::Optimized
        } else {
            ExecutionStrategy::Sequential
        }
    }

    /// Default strategy selection based on complexity
    fn select_default_strategy(
        &self,
        complexity: f64,
        snapshot: &ResourceSnapshot,
    ) -> ExecutionStrategy {
        if snapshot.is_high_load(&self.config) {
            return ExecutionStrategy::Degraded;
        }

        match complexity {
            c if c < 10.0 => ExecutionStrategy::Sequential,
            c if c < 30.0 => ExecutionStrategy::Batched,
            c if c < 60.0 => ExecutionStrategy::Optimized,
            _ => ExecutionStrategy::Parallel,
        }
    }

    /// Select strategy for high load conditions
    async fn select_high_load_strategy(&self, snapshot: &ResourceSnapshot) -> ExecutionStrategy {
        let severity = snapshot.load_severity(&self.config);

        if severity > 0.95 {
            // Critical load - use degraded mode
            ExecutionStrategy::Degraded
        } else if severity > 0.85 {
            // High load - use sequential
            ExecutionStrategy::Sequential
        } else {
            // Moderate load - use batched
            ExecutionStrategy::Batched
        }
    }

    /// Select fallback strategy
    fn select_fallback_strategy(
        &self,
        primary: ExecutionStrategy,
        snapshot: &ResourceSnapshot,
    ) -> ExecutionStrategy {
        if snapshot.is_high_load(&self.config) {
            return ExecutionStrategy::Degraded;
        }

        match primary {
            ExecutionStrategy::Parallel => ExecutionStrategy::Sequential,
            ExecutionStrategy::Optimized => ExecutionStrategy::Batched,
            ExecutionStrategy::Batched => ExecutionStrategy::Sequential,
            ExecutionStrategy::Streaming => ExecutionStrategy::Sequential,
            _ => ExecutionStrategy::Sequential,
        }
    }

    /// Record execution result for adaptation
    pub async fn record_execution(&self, result: ExecutionResult) -> Result<()> {
        let query_fingerprint = result.query_fingerprint.clone();
        let strategy = result.strategy_used;
        let execution_time = result.actual_time_ms;
        let success = result.success;

        // Update historical cost estimator
        {
            let mut estimator = self.cost_estimator.write().await;
            let _ = estimator.record_execution(
                &query_fingerprint,
                execution_time,
                1,    // complexity placeholder
                1024, // memory placeholder (1KB)
            );
        }

        // Update strategy performance stats
        {
            let mut stats = self.strategy_performance.write().await;
            let strategy_stat = stats.entry(strategy).or_default();

            strategy_stat.total_executions += 1;
            if success {
                strategy_stat.successful_executions += 1;
            }
            strategy_stat.total_time_ms += execution_time;
            strategy_stat.avg_time_ms =
                strategy_stat.total_time_ms / strategy_stat.total_executions as f64;
            strategy_stat.error_rate = 1.0
                - (strategy_stat.successful_executions as f64
                    / strategy_stat.total_executions as f64);
            strategy_stat.last_used = Instant::now();
        }

        // Add to execution history
        {
            let mut history = self.execution_history.write().await;
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(result);
        }

        // Train ML optimizer if enabled (currently disabled)
        // TODO: Integrate ML training when performance tracker is available
        if let Some(_ml_opt) = &self.ml_optimizer {
            // Training would happen here with features and target
        }

        Ok(())
    }

    /// Get current resource snapshot
    async fn get_resource_snapshot(&self) -> Result<ResourceSnapshot> {
        let mut sys = self.system.write().await;
        sys.refresh_cpu_all();
        sys.refresh_memory();

        // Average CPU usage across all cores
        let cpu_usage = sys
            .cpus()
            .iter()
            .map(|cpu| cpu.cpu_usage() as f64)
            .sum::<f64>()
            / sys.cpus().len().max(1) as f64
            / 100.0;

        let memory_usage = {
            let total_mem = sys.total_memory();
            let used_mem = sys.used_memory();
            if total_mem > 0 {
                used_mem as f64 / total_mem as f64
            } else {
                0.0
            }
        };

        // Get active queries and avg time from history
        let (active_queries, avg_query_time) = {
            let history = self.execution_history.read().await;
            let active = history.len();

            let avg_time = if !history.is_empty() {
                history.iter().map(|r| r.actual_time_ms).sum::<f64>() / history.len() as f64
            } else {
                0.0
            };

            (active, avg_time)
        };

        Ok(ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage,
            memory_usage,
            active_queries,
            avg_query_time_ms: avg_query_time,
        })
    }

    /// Generate query fingerprint for tracking
    fn fingerprint_query(query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let normalized = query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);

        format!("query_{:x}", hasher.finish())
    }

    /// Get current strategy performance statistics
    pub async fn get_strategy_stats(&self) -> HashMap<ExecutionStrategy, StrategyStats> {
        self.strategy_performance.read().await.clone()
    }

    /// Get planner metrics
    pub fn get_metrics(&self) -> DynamicPlannerMetrics {
        DynamicPlannerMetrics {
            total_adaptations: self.adaptations.get() as usize,
            strategy_switches: self.strategy_switches.get() as usize,
            high_load_events: self.high_load_events.get() as usize,
            avg_adaptation_quality: self.avg_adaptation_quality.get(),
        }
    }
}

/// Metrics for dynamic query planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPlannerMetrics {
    pub total_adaptations: usize,
    pub strategy_switches: usize,
    pub high_load_events: usize,
    pub avg_adaptation_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_strategy_overhead() {
        assert_eq!(ExecutionStrategy::Sequential.overhead_multiplier(), 1.0);
        assert!(ExecutionStrategy::Parallel.overhead_multiplier() > 1.0);
        assert!(ExecutionStrategy::Batched.overhead_multiplier() < 1.0);
        assert!(ExecutionStrategy::Cached.overhead_multiplier() < 0.1);
    }

    #[test]
    fn test_execution_strategy_high_load_safe() {
        assert!(ExecutionStrategy::Degraded.is_high_load_safe());
        assert!(ExecutionStrategy::Cached.is_high_load_safe());
        assert!(ExecutionStrategy::Sequential.is_high_load_safe());
        assert!(!ExecutionStrategy::Parallel.is_high_load_safe());
    }

    #[test]
    fn test_resource_snapshot_high_load() {
        let config = DynamicPlannerConfig::default();

        let low_load = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.5,
            memory_usage: 0.6,
            active_queries: 10,
            avg_query_time_ms: 50.0,
        };
        assert!(!low_load.is_high_load(&config));

        let high_load = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.9,
            memory_usage: 0.9,
            active_queries: 100,
            avg_query_time_ms: 500.0,
        };
        assert!(high_load.is_high_load(&config));
    }

    #[test]
    fn test_resource_snapshot_load_severity() {
        let config = DynamicPlannerConfig::default();

        let snapshot = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.9,    // Above threshold (0.8)
            memory_usage: 0.7, // Below threshold (0.85)
            active_queries: 50,
            avg_query_time_ms: 100.0,
        };

        let severity = snapshot.load_severity(&config);
        assert!(severity > 0.8); // Should be high due to CPU
    }

    #[test]
    fn test_adaptive_plan_creation() {
        let plan = AdaptivePlan::new("query_123".to_string(), ExecutionStrategy::Parallel, 50.0);

        assert_eq!(plan.query_fingerprint, "query_123");
        assert_eq!(plan.strategy, ExecutionStrategy::Parallel);
        assert_eq!(plan.estimated_cost, 50.0);
        assert_eq!(plan.confidence, 0.5);
    }

    #[test]
    fn test_adaptive_plan_with_fallback() {
        let plan = AdaptivePlan::new("query_123".to_string(), ExecutionStrategy::Parallel, 50.0)
            .with_fallback(ExecutionStrategy::Sequential);

        assert_eq!(plan.fallback_strategy, Some(ExecutionStrategy::Sequential));
    }

    #[test]
    fn test_adaptive_plan_with_confidence() {
        let plan = AdaptivePlan::new("query_123".to_string(), ExecutionStrategy::Parallel, 50.0)
            .with_confidence(0.85);

        assert_eq!(plan.confidence, 0.85);

        // Test clamping
        let clamped =
            AdaptivePlan::new("query_456".to_string(), ExecutionStrategy::Sequential, 10.0)
                .with_confidence(1.5);

        assert_eq!(clamped.confidence, 1.0); // Should be clamped to 1.0
    }

    #[tokio::test]
    async fn test_dynamic_planner_creation() {
        let config = DynamicPlannerConfig::default();
        let planner = DynamicQueryPlanner::new(config);

        let metrics = planner.get_metrics();
        assert_eq!(metrics.total_adaptations, 0);
        assert_eq!(metrics.strategy_switches, 0);
    }

    #[tokio::test]
    async fn test_create_plan_disabled() {
        let config = DynamicPlannerConfig {
            enabled: false,
            ..Default::default()
        };
        let planner = DynamicQueryPlanner::new(config);

        let plan = planner
            .create_plan("SELECT * WHERE { ?s ?p ?o }", 10.0)
            .await
            .unwrap();

        assert_eq!(plan.strategy, ExecutionStrategy::Sequential);
    }

    #[tokio::test]
    async fn test_create_plan_simple_query() {
        // Use max thresholds to prevent real system load from triggering degraded mode
        let config = DynamicPlannerConfig {
            cpu_threshold: 1.0,    // Never trigger high load from CPU
            memory_threshold: 1.0, // Never trigger high load from memory
            ..Default::default()
        };
        let planner = DynamicQueryPlanner::new(config);

        let plan = planner
            .create_plan("SELECT ?s WHERE { ?s ?p ?o } LIMIT 10", 5.0)
            .await
            .unwrap();

        // Simple query should NOT use parallel or degraded
        assert!(
            !matches!(
                plan.strategy,
                ExecutionStrategy::Parallel | ExecutionStrategy::Degraded
            ),
            "Simple query used {:?} strategy",
            plan.strategy
        );

        // Verify fallback strategy is set
        assert!(plan.fallback_strategy.is_some());
    }

    #[tokio::test]
    async fn test_create_plan_complex_query() {
        // Use max thresholds to prevent real system load from triggering degraded mode
        let config = DynamicPlannerConfig {
            cpu_threshold: 1.0,    // Never trigger high load from CPU
            memory_threshold: 1.0, // Never trigger high load from memory
            ..Default::default()
        };
        let planner = DynamicQueryPlanner::new(config);

        let complex_query =
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s ?p2 ?o2 . ?s ?p3 ?o3 } LIMIT 1000";
        let plan = planner.create_plan(complex_query, 75.0).await.unwrap();

        // Complex query should use optimized or parallel
        assert!(matches!(
            plan.strategy,
            ExecutionStrategy::Optimized | ExecutionStrategy::Parallel
        ));
        assert!(plan.fallback_strategy.is_some());
    }

    #[tokio::test]
    async fn test_record_execution() {
        let config = DynamicPlannerConfig::default();
        let planner = DynamicQueryPlanner::new(config);

        let result = ExecutionResult {
            query_fingerprint: "query_123".to_string(),
            strategy_used: ExecutionStrategy::Parallel,
            actual_time_ms: 150.0,
            success: true,
            error_message: None,
            resource_snapshot: ResourceSnapshot {
                timestamp: Instant::now(),
                cpu_usage: 0.6,
                memory_usage: 0.7,
                active_queries: 5,
                avg_query_time_ms: 100.0,
            },
        };

        planner.record_execution(result).await.unwrap();

        // Check strategy stats were updated
        let stats = planner.get_strategy_stats().await;
        let parallel_stats = stats.get(&ExecutionStrategy::Parallel).unwrap();

        assert_eq!(parallel_stats.total_executions, 1);
        assert_eq!(parallel_stats.successful_executions, 1);
        assert_eq!(parallel_stats.avg_time_ms, 150.0);
    }

    #[tokio::test]
    async fn test_query_fingerprinting() {
        let query1 = "SELECT ?s WHERE { ?s ?p ?o }";
        let query2 = "select ?s where { ?s ?p ?o }"; // Same query, different case
        let query3 = "SELECT ?x WHERE { ?x ?p ?o }"; // Different variables

        let fp1 = DynamicQueryPlanner::fingerprint_query(query1);
        let fp2 = DynamicQueryPlanner::fingerprint_query(query2);
        let fp3 = DynamicQueryPlanner::fingerprint_query(query3);

        // Same query (case-insensitive) should have same fingerprint
        assert_eq!(fp1, fp2);

        // Different query should have different fingerprint
        assert_ne!(fp1, fp3);
    }

    #[tokio::test]
    async fn test_default_strategy_selection() {
        let config = DynamicPlannerConfig::default();
        let planner = DynamicQueryPlanner::new(config.clone());

        let low_load = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.3,
            memory_usage: 0.4,
            active_queries: 2,
            avg_query_time_ms: 50.0,
        };

        // Low complexity
        let strategy = planner.select_default_strategy(5.0, &low_load);
        assert_eq!(strategy, ExecutionStrategy::Sequential);

        // Medium complexity
        let strategy = planner.select_default_strategy(25.0, &low_load);
        assert_eq!(strategy, ExecutionStrategy::Batched);

        // High complexity
        let strategy = planner.select_default_strategy(70.0, &low_load);
        assert_eq!(strategy, ExecutionStrategy::Parallel);

        // High load - should always use degraded
        let high_load = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.95,
            memory_usage: 0.90,
            active_queries: 100,
            avg_query_time_ms: 500.0,
        };

        let strategy = planner.select_default_strategy(70.0, &high_load);
        assert_eq!(strategy, ExecutionStrategy::Degraded);
    }

    #[test]
    fn test_strategy_stats_default() {
        let stats = StrategyStats::default();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.successful_executions, 0);
        assert_eq!(stats.error_rate, 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = DynamicPlannerConfig::default();
        assert!(config.enabled);
        assert!(config.enable_ml_prediction);
        assert_eq!(config.cpu_threshold, 0.80);
        assert_eq!(config.memory_threshold, 0.85);
    }
}
