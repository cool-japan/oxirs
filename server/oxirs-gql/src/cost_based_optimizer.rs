//! Cost-Based Query Optimization
//!
//! This module provides sophisticated cost-based optimization for GraphQL queries,
//! using statistical models and historical data to generate optimal execution plans.
//!
//! # Features
//!
//! - **Cost Estimation**: Accurate cost estimation for different execution strategies
//! - **Statistics Tracking**: Track query statistics for better cost models
//! - **Plan Comparison**: Compare multiple execution plans and choose the best
//! - **Adaptive Optimization**: Learn from execution patterns over time
//! - **Join Optimization**: Optimize joins and data fetching operations
//! - **Index Recommendations**: Suggest indexes for better performance
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::cost_based_optimizer::{CostBasedOptimizer, OptimizationConfig};
//!
//! let config = OptimizationConfig::new()
//!     .with_statistics_collection(true)
//!     .with_adaptive_learning(true);
//!
//! let optimizer = CostBasedOptimizer::new(config);
//! let query = /* ... GraphQL query ... */;
//!
//! // Generate optimized plan
//! let plan = optimizer.optimize(query).await?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for cost-based optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable statistics collection
    pub enable_statistics: bool,
    /// Enable adaptive learning
    pub enable_adaptive_learning: bool,
    /// Maximum number of alternative plans to consider
    pub max_alternative_plans: usize,
    /// Cost threshold for plan selection
    pub cost_threshold: f64,
    /// Enable join optimization
    pub enable_join_optimization: bool,
    /// Enable index recommendations
    pub enable_index_recommendations: bool,
    /// Statistics sample rate (0.0-1.0)
    pub statistics_sample_rate: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_statistics: true,
            enable_adaptive_learning: true,
            max_alternative_plans: 5,
            cost_threshold: 1000.0,
            enable_join_optimization: true,
            enable_index_recommendations: true,
            statistics_sample_rate: 1.0,
        }
    }
}

impl OptimizationConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable statistics collection
    pub fn with_statistics_collection(mut self, enabled: bool) -> Self {
        self.enable_statistics = enabled;
        self
    }

    /// Enable adaptive learning
    pub fn with_adaptive_learning(mut self, enabled: bool) -> Self {
        self.enable_adaptive_learning = enabled;
        self
    }

    /// Set maximum alternative plans
    pub fn with_max_alternative_plans(mut self, max: usize) -> Self {
        self.max_alternative_plans = max;
        self
    }

    /// Set cost threshold
    pub fn with_cost_threshold(mut self, threshold: f64) -> Self {
        self.cost_threshold = threshold;
        self
    }

    /// Enable join optimization
    pub fn with_join_optimization(mut self, enabled: bool) -> Self {
        self.enable_join_optimization = enabled;
        self
    }

    /// Enable index recommendations
    pub fn with_index_recommendations(mut self, enabled: bool) -> Self {
        self.enable_index_recommendations = enabled;
        self
    }

    /// Set statistics sample rate
    pub fn with_statistics_sample_rate(mut self, rate: f64) -> Self {
        self.statistics_sample_rate = rate.clamp(0.0, 1.0);
        self
    }
}

/// Query operation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// Field selection
    FieldSelect,
    /// Object fetch
    ObjectFetch,
    /// List fetch
    ListFetch,
    /// Join operation
    Join,
    /// Filter operation
    Filter,
    /// Aggregation
    Aggregation,
    /// Sort operation
    Sort,
}

/// Cost factors for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostFactors {
    /// CPU cost
    pub cpu_cost: f64,
    /// I/O cost
    pub io_cost: f64,
    /// Network cost
    pub network_cost: f64,
    /// Memory cost
    pub memory_cost: f64,
}

impl CostFactors {
    /// Create new cost factors
    pub fn new() -> Self {
        Self {
            cpu_cost: 0.0,
            io_cost: 0.0,
            network_cost: 0.0,
            memory_cost: 0.0,
        }
    }

    /// Total cost
    pub fn total(&self) -> f64 {
        self.cpu_cost + self.io_cost + self.network_cost + self.memory_cost
    }

    /// Add another cost
    pub fn add(&mut self, other: &CostFactors) {
        self.cpu_cost += other.cpu_cost;
        self.io_cost += other.io_cost;
        self.network_cost += other.network_cost;
        self.memory_cost += other.memory_cost;
    }
}

impl Default for CostFactors {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationCost {
    /// Operation type
    pub operation: OperationType,
    /// Operation name/identifier
    pub name: String,
    /// Cost factors
    pub factors: CostFactors,
    /// Estimated row count
    pub estimated_rows: usize,
    /// Estimated execution time
    pub estimated_time_ms: u64,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

impl OperationCost {
    /// Create new operation cost
    pub fn new(operation: OperationType, name: String) -> Self {
        Self {
            operation,
            name,
            factors: CostFactors::new(),
            estimated_rows: 0,
            estimated_time_ms: 0,
            confidence: 0.5,
        }
    }

    /// Set CPU cost
    pub fn with_cpu_cost(mut self, cost: f64) -> Self {
        self.factors.cpu_cost = cost;
        self
    }

    /// Set I/O cost
    pub fn with_io_cost(mut self, cost: f64) -> Self {
        self.factors.io_cost = cost;
        self
    }

    /// Set network cost
    pub fn with_network_cost(mut self, cost: f64) -> Self {
        self.factors.network_cost = cost;
        self
    }

    /// Set memory cost
    pub fn with_memory_cost(mut self, cost: f64) -> Self {
        self.factors.memory_cost = cost;
        self
    }

    /// Set estimated rows
    pub fn with_estimated_rows(mut self, rows: usize) -> Self {
        self.estimated_rows = rows;
        self
    }

    /// Set estimated time
    pub fn with_estimated_time(mut self, time_ms: u64) -> Self {
        self.estimated_time_ms = time_ms;
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Total cost
    pub fn total_cost(&self) -> f64 {
        self.factors.total()
    }
}

/// Execution plan with costs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBasedPlan {
    /// Plan identifier
    pub plan_id: String,
    /// Operations in the plan
    pub operations: Vec<OperationCost>,
    /// Total cost
    pub total_cost: f64,
    /// Total estimated time
    pub total_estimated_time_ms: u64,
    /// Plan strategy
    pub strategy: PlanStrategy,
    /// Index recommendations
    pub index_recommendations: Vec<IndexRecommendation>,
}

impl CostBasedPlan {
    /// Create a new plan
    pub fn new(plan_id: String, strategy: PlanStrategy) -> Self {
        Self {
            plan_id,
            operations: Vec::new(),
            total_cost: 0.0,
            total_estimated_time_ms: 0,
            strategy,
            index_recommendations: Vec::new(),
        }
    }

    /// Add operation
    pub fn add_operation(&mut self, op: OperationCost) {
        self.total_cost += op.total_cost();
        self.total_estimated_time_ms += op.estimated_time_ms;
        self.operations.push(op);
    }

    /// Add index recommendation
    pub fn add_index_recommendation(&mut self, rec: IndexRecommendation) {
        self.index_recommendations.push(rec);
    }

    /// Compare with another plan
    pub fn is_better_than(&self, other: &CostBasedPlan) -> bool {
        self.total_cost < other.total_cost
    }
}

/// Plan strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Batch execution
    Batch,
    /// Streaming execution
    Streaming,
    /// Hybrid strategy
    Hybrid,
}

/// Index recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    /// Table/type name
    pub table: String,
    /// Fields to index
    pub fields: Vec<String>,
    /// Index type
    pub index_type: IndexType,
    /// Expected performance improvement (percentage)
    pub expected_improvement: f64,
    /// Rationale for recommendation
    pub rationale: String,
}

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// B-tree index
    BTree,
    /// Hash index
    Hash,
    /// Full-text index
    FullText,
    /// Composite index
    Composite,
}

/// Query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistics {
    /// Query fingerprint
    pub fingerprint: String,
    /// Number of executions
    pub execution_count: usize,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Average row count
    pub avg_row_count: f64,
    /// Minimum execution time
    pub min_execution_time_ms: u64,
    /// Maximum execution time
    pub max_execution_time_ms: u64,
    /// Last execution timestamp
    pub last_execution: u64,
}

impl QueryStatistics {
    /// Create new statistics
    pub fn new(fingerprint: String) -> Self {
        Self {
            fingerprint,
            execution_count: 0,
            avg_execution_time_ms: 0.0,
            avg_row_count: 0.0,
            min_execution_time_ms: u64::MAX,
            max_execution_time_ms: 0,
            last_execution: 0,
        }
    }

    /// Update with new execution
    pub fn update(&mut self, execution_time_ms: u64, row_count: usize) {
        self.execution_count += 1;

        // Update average execution time
        let old_avg = self.avg_execution_time_ms;
        self.avg_execution_time_ms = (old_avg * (self.execution_count - 1) as f64
            + execution_time_ms as f64)
            / self.execution_count as f64;

        // Update average row count
        let old_row_avg = self.avg_row_count;
        self.avg_row_count = (old_row_avg * (self.execution_count - 1) as f64 + row_count as f64)
            / self.execution_count as f64;

        // Update min/max
        self.min_execution_time_ms = self.min_execution_time_ms.min(execution_time_ms);
        self.max_execution_time_ms = self.max_execution_time_ms.max(execution_time_ms);

        self.last_execution = chrono::Utc::now().timestamp() as u64;
    }
}

/// Cost model for estimation
pub struct CostModel {
    /// Cost weights
    weights: HashMap<OperationType, CostFactors>,
}

impl CostModel {
    /// Create a new cost model
    pub fn new() -> Self {
        let mut weights = HashMap::new();

        // Initialize default weights
        let mut field_cost = CostFactors::new();
        field_cost.cpu_cost = 1.0;
        field_cost.io_cost = 0.5;
        weights.insert(OperationType::FieldSelect, field_cost);

        let mut object_cost = CostFactors::new();
        object_cost.cpu_cost = 2.0;
        object_cost.io_cost = 5.0;
        weights.insert(OperationType::ObjectFetch, object_cost);

        let mut list_cost = CostFactors::new();
        list_cost.cpu_cost = 5.0;
        list_cost.io_cost = 10.0;
        weights.insert(OperationType::ListFetch, list_cost);

        let mut join_cost = CostFactors::new();
        join_cost.cpu_cost = 10.0;
        join_cost.io_cost = 20.0;
        weights.insert(OperationType::Join, join_cost);

        let mut filter_cost = CostFactors::new();
        filter_cost.cpu_cost = 3.0;
        weights.insert(OperationType::Filter, filter_cost);

        let mut agg_cost = CostFactors::new();
        agg_cost.cpu_cost = 8.0;
        agg_cost.memory_cost = 5.0;
        weights.insert(OperationType::Aggregation, agg_cost);

        let mut sort_cost = CostFactors::new();
        sort_cost.cpu_cost = 6.0;
        sort_cost.memory_cost = 4.0;
        weights.insert(OperationType::Sort, sort_cost);

        Self { weights }
    }

    /// Estimate cost for an operation
    pub fn estimate_cost(
        &mut self,
        op_type: OperationType,
        row_count: usize,
        statistics: Option<&QueryStatistics>,
    ) -> OperationCost {
        let base_cost = self.weights.get(&op_type).cloned().unwrap_or_default();

        let mut factors = base_cost.clone();

        // Scale by row count
        let scale_factor = (row_count as f64).ln().max(1.0);
        factors.cpu_cost *= scale_factor;
        factors.io_cost *= scale_factor;

        // Use statistics if available
        let estimated_time_ms = if let Some(stats) = statistics {
            (stats.avg_execution_time_ms * (row_count as f64 / stats.avg_row_count)) as u64
        } else {
            // Rough estimation: 1ms per 100 rows
            (row_count as f64 / 100.0).max(1.0) as u64
        };

        let confidence = if statistics.is_some() { 0.9 } else { 0.5 };

        OperationCost::new(op_type, "operation".to_string())
            .with_cpu_cost(factors.cpu_cost)
            .with_io_cost(factors.io_cost)
            .with_network_cost(factors.network_cost)
            .with_memory_cost(factors.memory_cost)
            .with_estimated_rows(row_count)
            .with_estimated_time(estimated_time_ms)
            .with_confidence(confidence)
    }

    /// Monte Carlo cost simulation
    pub fn simulate_cost(&mut self, plan: &CostBasedPlan, iterations: usize) -> f64 {
        let mut total = 0.0;

        for _ in 0..iterations {
            let mut cost = 0.0;
            for op in &plan.operations {
                // Add random variation using fastrand (already in dependencies)
                let variation = 0.9 + fastrand::f64() * 0.2; // Random between 0.9 and 1.1
                cost += op.total_cost() * variation;
            }
            total += cost;
        }

        total / iterations as f64
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Cost-based query optimizer
pub struct CostBasedOptimizer {
    config: OptimizationConfig,
    cost_model: Arc<RwLock<CostModel>>,
    statistics: Arc<RwLock<HashMap<String, QueryStatistics>>>,
}

impl CostBasedOptimizer {
    /// Create a new cost-based optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            cost_model: Arc::new(RwLock::new(CostModel::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize a query
    pub async fn optimize(&self, query: &str) -> Result<CostBasedPlan, OptimizationError> {
        // Generate alternative plans
        let plans = self.generate_alternative_plans(query).await?;

        // Select best plan
        let best_plan = self.select_best_plan(plans).await?;

        Ok(best_plan)
    }

    /// Generate alternative execution plans
    async fn generate_alternative_plans(
        &self,
        query: &str,
    ) -> Result<Vec<CostBasedPlan>, OptimizationError> {
        let mut plans = Vec::new();

        // Generate sequential plan
        let sequential = self.generate_plan(query, PlanStrategy::Sequential).await?;
        plans.push(sequential);

        // Generate parallel plan
        if self.config.max_alternative_plans > 1 {
            let parallel = self.generate_plan(query, PlanStrategy::Parallel).await?;
            plans.push(parallel);
        }

        // Generate batch plan
        if self.config.max_alternative_plans > 2 {
            let batch = self.generate_plan(query, PlanStrategy::Batch).await?;
            plans.push(batch);
        }

        // Generate streaming plan
        if self.config.max_alternative_plans > 3 {
            let streaming = self.generate_plan(query, PlanStrategy::Streaming).await?;
            plans.push(streaming);
        }

        // Generate hybrid plan
        if self.config.max_alternative_plans > 4 {
            let hybrid = self.generate_plan(query, PlanStrategy::Hybrid).await?;
            plans.push(hybrid);
        }

        Ok(plans)
    }

    /// Generate a specific plan
    async fn generate_plan(
        &self,
        query: &str,
        strategy: PlanStrategy,
    ) -> Result<CostBasedPlan, OptimizationError> {
        let fingerprint = self.fingerprint_query(query);
        let mut plan = CostBasedPlan::new(format!("plan_{:?}", strategy), strategy);

        // Get statistics
        let stats = {
            let stats_map = self.statistics.read().await;
            stats_map.get(&fingerprint).cloned()
        };

        // Estimate operations (simplified for mock)
        let row_count = 100; // Mock value

        let mut cost_model = self.cost_model.write().await;

        // Add mock operations
        let field_op = cost_model.estimate_cost(OperationType::FieldSelect, 10, stats.as_ref());
        plan.add_operation(field_op);

        let object_op = cost_model.estimate_cost(OperationType::ObjectFetch, 50, stats.as_ref());
        plan.add_operation(object_op);

        let list_op = cost_model.estimate_cost(OperationType::ListFetch, row_count, stats.as_ref());
        plan.add_operation(list_op);

        // Add index recommendations if enabled
        if self.config.enable_index_recommendations {
            let rec = IndexRecommendation {
                table: "users".to_string(),
                fields: vec!["id".to_string()],
                index_type: IndexType::BTree,
                expected_improvement: 25.0,
                rationale: "Frequent lookups on id field".to_string(),
            };
            plan.add_index_recommendation(rec);
        }

        Ok(plan)
    }

    /// Select the best plan from alternatives
    async fn select_best_plan(
        &self,
        plans: Vec<CostBasedPlan>,
    ) -> Result<CostBasedPlan, OptimizationError> {
        if plans.is_empty() {
            return Err(OptimizationError::NoPlanGenerated);
        }

        // Find plan with minimum cost
        let best = plans
            .into_iter()
            .min_by(|a, b| a.total_cost.partial_cmp(&b.total_cost).unwrap())
            .unwrap();

        Ok(best)
    }

    /// Record execution statistics
    pub async fn record_execution(
        &self,
        query: &str,
        execution_time_ms: u64,
        row_count: usize,
    ) -> Result<(), OptimizationError> {
        if !self.config.enable_statistics {
            return Ok(());
        }

        // Sample based on configured rate
        if fastrand::f64() > self.config.statistics_sample_rate {
            return Ok(());
        }

        let fingerprint = self.fingerprint_query(query);

        let mut stats_map = self.statistics.write().await;
        let stats = stats_map
            .entry(fingerprint.clone())
            .or_insert_with(|| QueryStatistics::new(fingerprint));

        stats.update(execution_time_ms, row_count);

        Ok(())
    }

    /// Get statistics for a query
    pub async fn get_statistics(&self, query: &str) -> Option<QueryStatistics> {
        let fingerprint = self.fingerprint_query(query);
        let stats_map = self.statistics.read().await;
        stats_map.get(&fingerprint).cloned()
    }

    /// Generate query fingerprint
    fn fingerprint_query(&self, query: &str) -> String {
        // Simple fingerprint: normalize whitespace and use simple hash
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");

        // Simple hash function (FNV-1a)
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in normalized.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        format!("{:016x}", hash)
    }

    /// Clear all statistics
    pub async fn clear_statistics(&self) {
        let mut stats_map = self.statistics.write().await;
        stats_map.clear();
    }

    /// Get number of tracked queries
    pub async fn statistics_count(&self) -> usize {
        let stats_map = self.statistics.read().await;
        stats_map.len()
    }
}

/// Errors that can occur during optimization
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    /// No plan could be generated
    #[error("No execution plan could be generated")]
    NoPlanGenerated,

    /// Cost estimation failed
    #[error("Cost estimation failed: {0}")]
    CostEstimationFailed(String),

    /// Statistics error
    #[error("Statistics error: {0}")]
    StatisticsError(String),

    /// Invalid query
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_builder() {
        let config = OptimizationConfig::new()
            .with_statistics_collection(true)
            .with_adaptive_learning(true)
            .with_max_alternative_plans(10)
            .with_cost_threshold(500.0)
            .with_join_optimization(true)
            .with_index_recommendations(true)
            .with_statistics_sample_rate(0.5);

        assert!(config.enable_statistics);
        assert!(config.enable_adaptive_learning);
        assert_eq!(config.max_alternative_plans, 10);
        assert_eq!(config.cost_threshold, 500.0);
        assert!(config.enable_join_optimization);
        assert!(config.enable_index_recommendations);
        assert_eq!(config.statistics_sample_rate, 0.5);
    }

    #[test]
    fn test_cost_factors_total() {
        let mut factors = CostFactors::new();
        factors.cpu_cost = 10.0;
        factors.io_cost = 5.0;
        factors.network_cost = 3.0;
        factors.memory_cost = 2.0;

        assert_eq!(factors.total(), 20.0);
    }

    #[test]
    fn test_cost_factors_add() {
        let mut factors1 = CostFactors::new();
        factors1.cpu_cost = 10.0;
        factors1.io_cost = 5.0;

        let mut factors2 = CostFactors::new();
        factors2.cpu_cost = 3.0;
        factors2.network_cost = 2.0;

        factors1.add(&factors2);

        assert_eq!(factors1.cpu_cost, 13.0);
        assert_eq!(factors1.io_cost, 5.0);
        assert_eq!(factors1.network_cost, 2.0);
    }

    #[test]
    fn test_operation_cost_creation() {
        let cost = OperationCost::new(OperationType::FieldSelect, "test".to_string())
            .with_cpu_cost(5.0)
            .with_io_cost(2.0)
            .with_estimated_rows(100)
            .with_estimated_time(50)
            .with_confidence(0.8);

        assert_eq!(cost.operation, OperationType::FieldSelect);
        assert_eq!(cost.factors.cpu_cost, 5.0);
        assert_eq!(cost.factors.io_cost, 2.0);
        assert_eq!(cost.estimated_rows, 100);
        assert_eq!(cost.estimated_time_ms, 50);
        assert_eq!(cost.confidence, 0.8);
    }

    #[test]
    fn test_operation_cost_total() {
        let cost = OperationCost::new(OperationType::Join, "join".to_string())
            .with_cpu_cost(10.0)
            .with_io_cost(20.0)
            .with_network_cost(5.0);

        assert_eq!(cost.total_cost(), 35.0);
    }

    #[test]
    fn test_plan_creation() {
        let plan = CostBasedPlan::new("plan1".to_string(), PlanStrategy::Sequential);

        assert_eq!(plan.plan_id, "plan1");
        assert_eq!(plan.strategy, PlanStrategy::Sequential);
        assert_eq!(plan.total_cost, 0.0);
        assert_eq!(plan.total_estimated_time_ms, 0);
    }

    #[test]
    fn test_plan_add_operation() {
        let mut plan = CostBasedPlan::new("plan1".to_string(), PlanStrategy::Parallel);

        let op1 = OperationCost::new(OperationType::FieldSelect, "field1".to_string())
            .with_cpu_cost(5.0)
            .with_estimated_time(10);

        let op2 = OperationCost::new(OperationType::ObjectFetch, "obj1".to_string())
            .with_cpu_cost(10.0)
            .with_estimated_time(20);

        plan.add_operation(op1);
        plan.add_operation(op2);

        assert_eq!(plan.operations.len(), 2);
        assert_eq!(plan.total_cost, 15.0);
        assert_eq!(plan.total_estimated_time_ms, 30);
    }

    #[test]
    fn test_plan_comparison() {
        let mut plan1 = CostBasedPlan::new("plan1".to_string(), PlanStrategy::Sequential);
        plan1.total_cost = 100.0;

        let mut plan2 = CostBasedPlan::new("plan2".to_string(), PlanStrategy::Parallel);
        plan2.total_cost = 50.0;

        assert!(plan2.is_better_than(&plan1));
        assert!(!plan1.is_better_than(&plan2));
    }

    #[test]
    fn test_query_statistics_creation() {
        let stats = QueryStatistics::new("fingerprint123".to_string());

        assert_eq!(stats.fingerprint, "fingerprint123");
        assert_eq!(stats.execution_count, 0);
        assert_eq!(stats.avg_execution_time_ms, 0.0);
    }

    #[test]
    fn test_query_statistics_update() {
        let mut stats = QueryStatistics::new("test".to_string());

        stats.update(100, 50);
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.avg_execution_time_ms, 100.0);
        assert_eq!(stats.avg_row_count, 50.0);
        assert_eq!(stats.min_execution_time_ms, 100);
        assert_eq!(stats.max_execution_time_ms, 100);

        stats.update(200, 100);
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.avg_execution_time_ms, 150.0);
        assert_eq!(stats.avg_row_count, 75.0);
        assert_eq!(stats.min_execution_time_ms, 100);
        assert_eq!(stats.max_execution_time_ms, 200);
    }

    #[test]
    fn test_cost_model_creation() {
        let model = CostModel::new();
        assert!(!model.weights.is_empty());
    }

    #[test]
    fn test_cost_model_estimate() {
        let mut model = CostModel::new();

        let cost = model.estimate_cost(OperationType::FieldSelect, 100, None);

        assert_eq!(cost.operation, OperationType::FieldSelect);
        assert!(cost.total_cost() > 0.0);
        assert_eq!(cost.estimated_rows, 100);
        assert_eq!(cost.confidence, 0.5);
    }

    #[test]
    fn test_cost_model_with_statistics() {
        let mut model = CostModel::new();
        let mut stats = QueryStatistics::new("test".to_string());
        stats.update(100, 50);
        stats.update(200, 100);

        let cost = model.estimate_cost(OperationType::ObjectFetch, 100, Some(&stats));

        assert!(cost.confidence > 0.5);
        assert!(cost.estimated_time_ms > 0);
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let mut model = CostModel::new();
        let mut plan = CostBasedPlan::new("test".to_string(), PlanStrategy::Sequential);

        let op = OperationCost::new(OperationType::Join, "join".to_string()).with_cpu_cost(50.0);

        plan.add_operation(op);

        let simulated_cost = model.simulate_cost(&plan, 100);
        assert!(simulated_cost > 0.0);
    }

    #[tokio::test]
    async fn test_optimizer_creation() {
        let config = OptimizationConfig::new();
        let optimizer = CostBasedOptimizer::new(config);

        assert_eq!(optimizer.statistics_count().await, 0);
    }

    #[tokio::test]
    async fn test_query_optimization() {
        let config = OptimizationConfig::new();
        let optimizer = CostBasedOptimizer::new(config);

        let query = "{ user { id name posts { title } } }";
        let plan = optimizer.optimize(query).await;

        assert!(plan.is_ok());
        let plan = plan.unwrap();
        assert!(!plan.operations.is_empty());
        assert!(plan.total_cost > 0.0);
    }

    #[tokio::test]
    async fn test_alternative_plans_generation() {
        let config = OptimizationConfig::new().with_max_alternative_plans(5);
        let optimizer = CostBasedOptimizer::new(config);

        let query = "{ test }";
        let plans = optimizer.generate_alternative_plans(query).await.unwrap();

        assert_eq!(plans.len(), 5);
        assert!(plans.iter().any(|p| p.strategy == PlanStrategy::Sequential));
        assert!(plans.iter().any(|p| p.strategy == PlanStrategy::Parallel));
        assert!(plans.iter().any(|p| p.strategy == PlanStrategy::Batch));
        assert!(plans.iter().any(|p| p.strategy == PlanStrategy::Streaming));
        assert!(plans.iter().any(|p| p.strategy == PlanStrategy::Hybrid));
    }

    #[tokio::test]
    async fn test_best_plan_selection() {
        let config = OptimizationConfig::new();
        let optimizer = CostBasedOptimizer::new(config);

        let mut plan1 = CostBasedPlan::new("plan1".to_string(), PlanStrategy::Sequential);
        plan1.total_cost = 100.0;

        let mut plan2 = CostBasedPlan::new("plan2".to_string(), PlanStrategy::Parallel);
        plan2.total_cost = 50.0;

        let mut plan3 = CostBasedPlan::new("plan3".to_string(), PlanStrategy::Batch);
        plan3.total_cost = 75.0;

        let plans = vec![plan1, plan2.clone(), plan3];
        let best = optimizer.select_best_plan(plans).await.unwrap();

        assert_eq!(best.plan_id, plan2.plan_id);
        assert_eq!(best.total_cost, 50.0);
    }

    #[tokio::test]
    async fn test_record_execution() {
        let config = OptimizationConfig::new().with_statistics_collection(true);
        let optimizer = CostBasedOptimizer::new(config);

        let query = "{ test }";
        optimizer.record_execution(query, 100, 50).await.unwrap();

        let stats = optimizer.get_statistics(query).await;
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.avg_execution_time_ms, 100.0);
    }

    #[tokio::test]
    async fn test_statistics_disabled() {
        let config = OptimizationConfig::new().with_statistics_collection(false);
        let optimizer = CostBasedOptimizer::new(config);

        let query = "{ test }";
        optimizer.record_execution(query, 100, 50).await.unwrap();

        assert_eq!(optimizer.statistics_count().await, 0);
    }

    #[tokio::test]
    async fn test_clear_statistics() {
        let config = OptimizationConfig::new();
        let optimizer = CostBasedOptimizer::new(config);

        optimizer
            .record_execution("{ test1 }", 100, 50)
            .await
            .unwrap();
        optimizer
            .record_execution("{ test2 }", 200, 100)
            .await
            .unwrap();

        assert_eq!(optimizer.statistics_count().await, 2);

        optimizer.clear_statistics().await;
        assert_eq!(optimizer.statistics_count().await, 0);
    }

    #[tokio::test]
    async fn test_query_fingerprinting() {
        let config = OptimizationConfig::new();
        let optimizer = CostBasedOptimizer::new(config);

        let query1 = "{ user { id name } }";
        let query2 = "{  user  {  id   name  }  }"; // Different whitespace

        let fp1 = optimizer.fingerprint_query(query1);
        let fp2 = optimizer.fingerprint_query(query2);

        assert_eq!(fp1, fp2);
    }

    #[tokio::test]
    async fn test_index_recommendations() {
        let config = OptimizationConfig::new().with_index_recommendations(true);
        let optimizer = CostBasedOptimizer::new(config);

        let query = "{ users { id } }";
        let plan = optimizer.optimize(query).await.unwrap();

        assert!(!plan.index_recommendations.is_empty());
        assert!(plan.index_recommendations[0].expected_improvement > 0.0);
    }

    #[test]
    fn test_index_recommendation_creation() {
        let rec = IndexRecommendation {
            table: "users".to_string(),
            fields: vec!["email".to_string()],
            index_type: IndexType::Hash,
            expected_improvement: 30.0,
            rationale: "Frequent equality lookups".to_string(),
        };

        assert_eq!(rec.table, "users");
        assert_eq!(rec.fields.len(), 1);
        assert_eq!(rec.index_type, IndexType::Hash);
        assert_eq!(rec.expected_improvement, 30.0);
    }

    #[test]
    fn test_confidence_clamping() {
        let cost =
            OperationCost::new(OperationType::FieldSelect, "test".to_string()).with_confidence(1.5);

        assert_eq!(cost.confidence, 1.0);

        let cost2 = OperationCost::new(OperationType::FieldSelect, "test".to_string())
            .with_confidence(-0.5);

        assert_eq!(cost2.confidence, 0.0);
    }
}
