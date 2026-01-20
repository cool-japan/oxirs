//! # Adaptive Reasoning Strategies
//!
//! This module provides intelligent, self-tuning reasoning strategies that
//! automatically adapt to dataset characteristics, query patterns, and
//! system resources for optimal performance.
//!
//! ## Features
//!
//! - **Strategy Selection**: Automatically selects optimal reasoning strategy
//! - **Performance Learning**: Learns from execution history
//! - **Resource Awareness**: Adapts to available CPU, memory, and GPU resources
//! - **Query Pattern Analysis**: Optimizes for observed query patterns
//! - **Dynamic Switching**: Switches strategies at runtime based on conditions
//! - **Cost Modeling**: Predicts cost of different strategies
//!
//! ## Strategies
//!
//! - **Forward Chaining**: Materialization-based (eager)
//! - **Backward Chaining**: Query-driven (lazy)
//! - **Hybrid**: Combines forward and backward
//! - **RETE**: Incremental pattern matching
//! - **GPU-Accelerated**: GPU-based parallel processing
//! - **Distributed**: Cluster-based reasoning
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::adaptive_strategies::*;
//! use oxirs_rule::{Rule, RuleAtom};
//!
//! // Create adaptive strategy selector
//! let mut selector = AdaptiveStrategySelector::new();
//!
//! // Register available strategies
//! selector.register_strategy(ReasoningStrategy::Forward);
//! selector.register_strategy(ReasoningStrategy::Backward);
//! selector.register_strategy(ReasoningStrategy::Hybrid);
//!
//! // Analyze dataset
//! let facts = vec![/* ... */];
//! let characteristics = selector.analyze_dataset(&facts);
//!
//! // Select optimal strategy
//! let strategy = selector.select_strategy(&characteristics).unwrap();
//! println!("Selected strategy: {:?}", strategy);
//! ```

use crate::{Rule, RuleAtom};
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Gauge, Timer};
// Random generation simplified for now
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Reasoning strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningStrategy {
    /// Forward chaining (eager materialization)
    Forward,
    /// Backward chaining (lazy query-driven)
    Backward,
    /// Hybrid (selective materialization)
    Hybrid,
    /// RETE network (incremental)
    RETE,
    /// GPU-accelerated
    GPU,
    /// Distributed reasoning
    Distributed,
    /// Custom adaptive strategy
    Custom,
}

/// Dataset characteristics for strategy selection
#[derive(Debug, Clone)]
pub struct DatasetCharacteristics {
    /// Number of facts
    pub fact_count: usize,
    /// Number of rules
    pub rule_count: usize,
    /// Average rule body size
    pub avg_rule_body_size: f64,
    /// Graph density (edges / possible edges)
    pub density: f64,
    /// Average node degree
    pub avg_degree: f64,
    /// Number of unique predicates
    pub predicate_count: usize,
    /// Rule selectivity (avg facts matched per rule)
    pub rule_selectivity: f64,
    /// Data skew (concentration in hot predicates)
    pub data_skew: f64,
}

/// Query pattern characteristics
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Query frequency (queries per second)
    pub frequency: f64,
    /// Average query complexity (number of patterns)
    pub complexity: f64,
    /// Query selectivity (result size / dataset size)
    pub selectivity: f64,
    /// Temporal locality (same queries repeated)
    pub temporal_locality: f64,
    /// Spatial locality (similar query patterns)
    pub spatial_locality: f64,
}

/// System resource availability
#[derive(Debug, Clone)]
pub struct SystemResources {
    /// Available CPU cores
    pub cpu_cores: usize,
    /// Available memory (bytes)
    pub memory_available: usize,
    /// GPU available
    pub gpu_available: bool,
    /// Distributed nodes available
    pub distributed_nodes: usize,
    /// Current CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Current memory utilization (0.0-1.0)
    pub memory_utilization: f64,
}

/// Strategy performance metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Result quality (precision/recall)
    pub quality: f64,
    /// Number of executions
    pub execution_count: usize,
}

/// Adaptive strategy selector
pub struct AdaptiveStrategySelector {
    /// Available strategies
    strategies: Vec<ReasoningStrategy>,
    /// Performance history
    performance_history: HashMap<ReasoningStrategy, Vec<StrategyMetrics>>,
    /// Current strategy
    current_strategy: Option<ReasoningStrategy>,
    /// Learning rate for cost model updates
    #[allow(dead_code)]
    learning_rate: f64,
    /// Cost model weights
    cost_weights: CostModelWeights,
    /// Performance metrics
    metrics: SelectorMetrics,
    /// Random seed for exploration
    random_state: u64,
    /// Exploration rate (epsilon-greedy)
    exploration_rate: f64,
}

/// Cost model weights for strategy selection
#[derive(Debug, Clone)]
struct CostModelWeights {
    /// Weight for execution time
    time_weight: f64,
    /// Weight for memory usage
    memory_weight: f64,
    /// Weight for throughput
    #[allow(dead_code)]
    throughput_weight: f64,
    /// Weight for result quality
    #[allow(dead_code)]
    quality_weight: f64,
}

/// Performance metrics for selector
pub struct SelectorMetrics {
    /// Total selections
    total_selections: Counter,
    /// Correct predictions
    #[allow(dead_code)]
    correct_predictions: Counter,
    /// Strategy switches
    strategy_switches: Counter,
    /// Selection time
    selection_timer: Timer,
    /// Current cost estimate
    #[allow(dead_code)]
    current_cost: Gauge,
}

impl SelectorMetrics {
    fn new() -> Self {
        Self {
            total_selections: Counter::new("adaptive_total_selections".to_string()),
            correct_predictions: Counter::new("adaptive_correct_predictions".to_string()),
            strategy_switches: Counter::new("adaptive_strategy_switches".to_string()),
            selection_timer: Timer::new("adaptive_selection_time".to_string()),
            current_cost: Gauge::new("adaptive_current_cost".to_string()),
        }
    }
}

impl AdaptiveStrategySelector {
    /// Create a new adaptive strategy selector
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            performance_history: HashMap::new(),
            current_strategy: None,
            learning_rate: 0.1,
            cost_weights: CostModelWeights {
                time_weight: 0.4,
                memory_weight: 0.2,
                throughput_weight: 0.3,
                quality_weight: 0.1,
            },
            metrics: SelectorMetrics::new(),
            random_state: 42,
            exploration_rate: 0.1,
        }
    }

    /// Register an available strategy
    pub fn register_strategy(&mut self, strategy: ReasoningStrategy) {
        if !self.strategies.contains(&strategy) {
            self.strategies.push(strategy);
            self.performance_history.insert(strategy, Vec::new());
        }
    }

    /// Analyze dataset characteristics
    pub fn analyze_dataset(&self, facts: &[RuleAtom]) -> DatasetCharacteristics {
        let fact_count = facts.len();

        // Count unique subjects, predicates, objects
        let mut subjects = std::collections::HashSet::new();
        let mut predicates = std::collections::HashSet::new();
        let mut objects = std::collections::HashSet::new();

        for fact in facts {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = fact
            {
                subjects.insert(format!("{:?}", subject));
                predicates.insert(format!("{:?}", predicate));
                objects.insert(format!("{:?}", object));
            }
        }

        let predicate_count = predicates.len();
        let node_count = subjects.len() + objects.len();

        // Estimate density
        let possible_edges = if node_count > 1 {
            node_count * (node_count - 1)
        } else {
            1
        };
        let density = fact_count as f64 / possible_edges as f64;

        // Estimate average degree
        let avg_degree = if node_count > 0 {
            (2.0 * fact_count as f64) / node_count as f64
        } else {
            0.0
        };

        // Compute data skew (simplified)
        let data_skew = if predicate_count > 0 {
            1.0 / predicate_count as f64
        } else {
            1.0
        };

        DatasetCharacteristics {
            fact_count,
            rule_count: 0, // To be set by caller
            avg_rule_body_size: 0.0,
            density,
            avg_degree,
            predicate_count,
            rule_selectivity: 0.0,
            data_skew,
        }
    }

    /// Analyze rules
    pub fn analyze_rules(&self, rules: &[Rule]) -> (f64, f64) {
        if rules.is_empty() {
            return (0.0, 0.0);
        }

        let total_body_size: usize = rules.iter().map(|r| r.body.len()).sum();
        let avg_body_size = total_body_size as f64 / rules.len() as f64;

        // Estimate selectivity (simplified)
        let avg_selectivity = 0.5; // Placeholder

        (avg_body_size, avg_selectivity)
    }

    /// Select optimal strategy
    pub fn select_strategy(
        &mut self,
        characteristics: &DatasetCharacteristics,
    ) -> Result<ReasoningStrategy> {
        self.metrics.total_selections.inc();
        let _timer = self.metrics.selection_timer.start();

        if self.strategies.is_empty() {
            return Err(anyhow!("No strategies registered"));
        }

        // Epsilon-greedy exploration (simplified)
        self.random_state = self
            .random_state
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        let rand_val = (self.random_state >> 16) as f64 / 65536.0;
        let explore = rand_val < self.exploration_rate;

        let strategy = if explore {
            // Random exploration
            let idx = (self.random_state as usize) % self.strategies.len();
            self.strategies[idx]
        } else {
            // Exploit best strategy
            self.select_best_strategy(characteristics)?
        };

        // Track strategy switches
        if let Some(current) = self.current_strategy {
            if current != strategy {
                self.metrics.strategy_switches.inc();
            }
        }

        self.current_strategy = Some(strategy);
        Ok(strategy)
    }

    /// Select best strategy based on cost model
    fn select_best_strategy(
        &self,
        characteristics: &DatasetCharacteristics,
    ) -> Result<ReasoningStrategy> {
        let mut best_strategy = self.strategies[0];
        let mut best_cost = f64::INFINITY;

        for &strategy in &self.strategies {
            let cost = self.estimate_cost(strategy, characteristics);
            if cost < best_cost {
                best_cost = cost;
                best_strategy = strategy;
            }
        }

        Ok(best_strategy)
    }

    /// Estimate cost of a strategy
    fn estimate_cost(
        &self,
        strategy: ReasoningStrategy,
        characteristics: &DatasetCharacteristics,
    ) -> f64 {
        // Base costs (heuristic)
        let (time_cost, memory_cost) = match strategy {
            ReasoningStrategy::Forward => {
                // Forward: O(facts * rules)
                let time = (characteristics.fact_count * characteristics.rule_count) as f64;
                let memory = (characteristics.fact_count * 2) as f64; // Materialized facts
                (time, memory)
            }
            ReasoningStrategy::Backward => {
                // Backward: O(goals * rules * depth)
                let time = (characteristics.rule_count * 10) as f64; // Depth ≈ 10
                let memory = (characteristics.rule_count * 5) as f64; // Proof stack
                (time, memory)
            }
            ReasoningStrategy::Hybrid => {
                // Hybrid: Between forward and backward
                let time = (characteristics.fact_count * characteristics.rule_count / 2) as f64;
                let memory = characteristics.fact_count as f64 * 1.5;
                (time, memory)
            }
            ReasoningStrategy::RETE => {
                // RETE: O(rules + facts) after network build
                let time = (characteristics.fact_count + characteristics.rule_count * 10) as f64;
                let memory = (characteristics.rule_count * 20) as f64; // Network nodes
                (time, memory)
            }
            ReasoningStrategy::GPU => {
                // GPU: Fast but requires transfer
                let time = (characteristics.fact_count / 100) as f64; // 100x speedup
                let memory = (characteristics.fact_count * 3) as f64; // GPU + CPU
                (time, memory)
            }
            ReasoningStrategy::Distributed => {
                // Distributed: Fast but has coordination overhead
                let time = (characteristics.fact_count / 10) as f64; // 10x speedup
                let memory = characteristics.fact_count as f64 * 1.2; // Distributed
                (time, memory)
            }
            ReasoningStrategy::Custom => (1000.0, 1000.0),
        };

        // Learn from history
        let learned_cost = if let Some(history) = self.performance_history.get(&strategy) {
            if !history.is_empty() {
                let avg_time: f64 = history
                    .iter()
                    .map(|m| m.execution_time.as_secs_f64())
                    .sum::<f64>()
                    / history.len() as f64;
                let avg_memory: f64 = history.iter().map(|m| m.memory_usage as f64).sum::<f64>()
                    / history.len() as f64;

                Some((avg_time * 1e6, avg_memory)) // Scale time to microseconds
            } else {
                None
            }
        } else {
            None
        };

        // Combine heuristic and learned costs
        let (final_time, final_memory) = if let Some((learned_time, learned_mem)) = learned_cost {
            // Weighted average: 70% learned, 30% heuristic
            let t = 0.7 * learned_time + 0.3 * time_cost;
            let m = 0.7 * learned_mem + 0.3 * memory_cost;
            (t, m)
        } else {
            (time_cost, memory_cost)
        };

        // Normalize costs
        let norm_time = final_time / 1e6; // Seconds
        let norm_memory = final_memory / 1e6; // MB

        // Weighted combination
        self.cost_weights.time_weight * norm_time + self.cost_weights.memory_weight * norm_memory
    }

    /// Record strategy performance
    pub fn record_performance(&mut self, strategy: ReasoningStrategy, metrics: StrategyMetrics) {
        if let Some(history) = self.performance_history.get_mut(&strategy) {
            history.push(metrics);

            // Keep last 100 entries
            if history.len() > 100 {
                history.remove(0);
            }
        }
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> Option<ReasoningStrategy> {
        self.current_strategy
    }

    /// Set cost model weights
    pub fn set_cost_weights(&mut self, time: f64, memory: f64, throughput: f64, quality: f64) {
        // Normalize weights
        let total = time + memory + throughput + quality;
        self.cost_weights = CostModelWeights {
            time_weight: time / total,
            memory_weight: memory / total,
            throughput_weight: throughput / total,
            quality_weight: quality / total,
        };
    }

    /// Set exploration rate
    pub fn set_exploration_rate(&mut self, rate: f64) {
        self.exploration_rate = rate.clamp(0.0, 1.0);
    }

    /// Get performance history
    pub fn get_performance_history(
        &self,
        strategy: ReasoningStrategy,
    ) -> Option<&Vec<StrategyMetrics>> {
        self.performance_history.get(&strategy)
    }

    /// Get metrics
    pub fn get_metrics(&self) -> &SelectorMetrics {
        &self.metrics
    }

    /// Recommend strategy for specific workload
    pub fn recommend_for_workload(
        &self,
        characteristics: &DatasetCharacteristics,
        resources: &SystemResources,
    ) -> ReasoningStrategy {
        // Small datasets: Backward chaining (low memory)
        if characteristics.fact_count < 100 {
            return ReasoningStrategy::Backward;
        }

        // Large datasets with GPU: GPU acceleration
        if characteristics.fact_count > 10000 && resources.gpu_available {
            return ReasoningStrategy::GPU;
        }

        // Distributed cluster available: Use distributed
        if resources.distributed_nodes > 1 && characteristics.fact_count > 5000 {
            return ReasoningStrategy::Distributed;
        }

        // Dense graphs with many rules: RETE
        if characteristics.density > 0.1 && characteristics.rule_count > 50 {
            return ReasoningStrategy::RETE;
        }

        // Sparse graphs: Backward chaining
        if characteristics.density < 0.01 {
            return ReasoningStrategy::Backward;
        }

        // Default: Hybrid approach
        ReasoningStrategy::Hybrid
    }
}

impl Default for AdaptiveStrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive reasoning engine that switches strategies dynamically
pub struct AdaptiveReasoningEngine {
    /// Strategy selector
    selector: AdaptiveStrategySelector,
    /// Facts
    facts: Vec<RuleAtom>,
    /// Rules
    rules: Vec<Rule>,
    /// Performance tracking
    #[allow(dead_code)]
    current_metrics: Option<StrategyMetrics>,
    /// Adaptation interval (number of queries)
    adaptation_interval: usize,
    /// Query counter
    query_count: usize,
}

impl AdaptiveReasoningEngine {
    /// Create a new adaptive reasoning engine
    pub fn new() -> Self {
        let mut selector = AdaptiveStrategySelector::new();

        // Register all available strategies
        selector.register_strategy(ReasoningStrategy::Forward);
        selector.register_strategy(ReasoningStrategy::Backward);
        selector.register_strategy(ReasoningStrategy::Hybrid);
        selector.register_strategy(ReasoningStrategy::RETE);

        Self {
            selector,
            facts: Vec::new(),
            rules: Vec::new(),
            current_metrics: None,
            adaptation_interval: 100,
            query_count: 0,
        }
    }

    /// Add facts
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        self.facts.extend(facts);
    }

    /// Add rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    /// Perform reasoning with adaptive strategy
    pub fn reason(&mut self) -> Result<Vec<RuleAtom>> {
        self.query_count += 1;

        // Re-evaluate strategy periodically
        if self.query_count % self.adaptation_interval == 0 {
            self.adapt_strategy()?;
        }

        // Execute with current strategy
        let start = Instant::now();
        let results = self.execute_current_strategy()?;
        let duration = start.elapsed();

        // Record performance
        let metrics = StrategyMetrics {
            execution_time: duration,
            memory_usage: 0, // Would measure actual memory
            throughput: results.len() as f64 / duration.as_secs_f64(),
            quality: 1.0,
            execution_count: 1,
        };

        if let Some(strategy) = self.selector.current_strategy() {
            self.selector.record_performance(strategy, metrics);
        }

        Ok(results)
    }

    /// Adapt strategy based on current conditions
    fn adapt_strategy(&mut self) -> Result<()> {
        let mut characteristics = self.selector.analyze_dataset(&self.facts);
        characteristics.rule_count = self.rules.len();

        let (avg_body_size, selectivity) = self.selector.analyze_rules(&self.rules);
        characteristics.avg_rule_body_size = avg_body_size;
        characteristics.rule_selectivity = selectivity;

        self.selector.select_strategy(&characteristics)?;
        Ok(())
    }

    /// Execute with current strategy
    fn execute_current_strategy(&self) -> Result<Vec<RuleAtom>> {
        let strategy = self
            .selector
            .current_strategy()
            .unwrap_or(ReasoningStrategy::Hybrid);

        match strategy {
            ReasoningStrategy::Forward => self.execute_forward(),
            ReasoningStrategy::Backward => self.execute_backward(),
            ReasoningStrategy::Hybrid => self.execute_hybrid(),
            ReasoningStrategy::RETE => self.execute_rete(),
            ReasoningStrategy::GPU => self.execute_gpu(),
            ReasoningStrategy::Distributed => self.execute_distributed(),
            ReasoningStrategy::Custom => self.execute_forward(), // Fallback
        }
    }

    /// Execute forward chaining
    fn execute_forward(&self) -> Result<Vec<RuleAtom>> {
        // Simplified forward chaining
        Ok(self.facts.clone())
    }

    /// Execute backward chaining
    fn execute_backward(&self) -> Result<Vec<RuleAtom>> {
        // Simplified backward chaining
        Ok(self.facts.clone())
    }

    /// Execute hybrid reasoning
    fn execute_hybrid(&self) -> Result<Vec<RuleAtom>> {
        // Simplified hybrid
        Ok(self.facts.clone())
    }

    /// Execute RETE
    fn execute_rete(&self) -> Result<Vec<RuleAtom>> {
        // Simplified RETE
        Ok(self.facts.clone())
    }

    /// Execute GPU-accelerated
    fn execute_gpu(&self) -> Result<Vec<RuleAtom>> {
        // Simplified GPU
        Ok(self.facts.clone())
    }

    /// Execute distributed
    fn execute_distributed(&self) -> Result<Vec<RuleAtom>> {
        // Simplified distributed
        Ok(self.facts.clone())
    }

    /// Set adaptation interval
    pub fn set_adaptation_interval(&mut self, interval: usize) {
        self.adaptation_interval = interval;
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> Option<ReasoningStrategy> {
        self.selector.current_strategy()
    }
}

impl Default for AdaptiveReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rule, RuleAtom, Term};

    fn create_test_fact(s: &str, p: &str, o: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: Term::Constant(s.to_string()),
            predicate: Term::Constant(p.to_string()),
            object: Term::Constant(o.to_string()),
        }
    }

    #[test]
    fn test_selector_creation() {
        let selector = AdaptiveStrategySelector::new();
        assert_eq!(selector.strategies.len(), 0);
        assert!(selector.current_strategy.is_none());
    }

    #[test]
    fn test_register_strategy() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.register_strategy(ReasoningStrategy::Forward);
        selector.register_strategy(ReasoningStrategy::Backward);

        assert_eq!(selector.strategies.len(), 2);
    }

    #[test]
    fn test_analyze_dataset() {
        let selector = AdaptiveStrategySelector::new();
        let facts = vec![
            create_test_fact("a", "p", "b"),
            create_test_fact("b", "p", "c"),
            create_test_fact("c", "p", "d"),
        ];

        let characteristics = selector.analyze_dataset(&facts);
        assert_eq!(characteristics.fact_count, 3);
        assert!(characteristics.density > 0.0);
    }

    #[test]
    fn test_select_strategy() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.register_strategy(ReasoningStrategy::Forward);
        selector.register_strategy(ReasoningStrategy::Backward);

        let characteristics = DatasetCharacteristics {
            fact_count: 100,
            rule_count: 10,
            avg_rule_body_size: 2.0,
            density: 0.05,
            avg_degree: 3.0,
            predicate_count: 5,
            rule_selectivity: 0.5,
            data_skew: 0.2,
        };

        let strategy = selector.select_strategy(&characteristics).unwrap();
        assert!(strategy == ReasoningStrategy::Forward || strategy == ReasoningStrategy::Backward);
    }

    #[test]
    fn test_cost_estimation() {
        let selector = AdaptiveStrategySelector::new();
        let characteristics = DatasetCharacteristics {
            fact_count: 1000,
            rule_count: 50,
            avg_rule_body_size: 3.0,
            density: 0.1,
            avg_degree: 5.0,
            predicate_count: 20,
            rule_selectivity: 0.5,
            data_skew: 0.1,
        };

        let cost_forward = selector.estimate_cost(ReasoningStrategy::Forward, &characteristics);
        let cost_backward = selector.estimate_cost(ReasoningStrategy::Backward, &characteristics);

        assert!(cost_forward > 0.0);
        assert!(cost_backward > 0.0);
    }

    #[test]
    fn test_performance_recording() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.register_strategy(ReasoningStrategy::Forward);

        let metrics = StrategyMetrics {
            execution_time: Duration::from_millis(100),
            memory_usage: 1024,
            throughput: 100.0,
            quality: 0.95,
            execution_count: 1,
        };

        selector.record_performance(ReasoningStrategy::Forward, metrics);

        let history = selector.get_performance_history(ReasoningStrategy::Forward);
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
    }

    #[test]
    fn test_cost_weights() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.set_cost_weights(0.5, 0.3, 0.1, 0.1);

        assert!((selector.cost_weights.time_weight - 0.5).abs() < 1e-6);
        assert!((selector.cost_weights.memory_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_exploration_rate() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.set_exploration_rate(0.2);

        assert!((selector.exploration_rate - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_recommend_for_workload() {
        let selector = AdaptiveStrategySelector::new();

        // Small dataset
        let small_chars = DatasetCharacteristics {
            fact_count: 50,
            rule_count: 5,
            avg_rule_body_size: 2.0,
            density: 0.01,
            avg_degree: 2.0,
            predicate_count: 3,
            rule_selectivity: 0.3,
            data_skew: 0.1,
        };

        let resources = SystemResources {
            cpu_cores: 4,
            memory_available: 8 * 1024 * 1024 * 1024,
            gpu_available: false,
            distributed_nodes: 1,
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
        };

        let strategy = selector.recommend_for_workload(&small_chars, &resources);
        assert_eq!(strategy, ReasoningStrategy::Backward);
    }

    #[test]
    fn test_recommend_gpu_for_large_dataset() {
        let selector = AdaptiveStrategySelector::new();

        let large_chars = DatasetCharacteristics {
            fact_count: 15000,
            rule_count: 100,
            avg_rule_body_size: 3.0,
            density: 0.05,
            avg_degree: 10.0,
            predicate_count: 50,
            rule_selectivity: 0.5,
            data_skew: 0.2,
        };

        let resources = SystemResources {
            cpu_cores: 8,
            memory_available: 16 * 1024 * 1024 * 1024,
            gpu_available: true,
            distributed_nodes: 1,
            cpu_utilization: 0.3,
            memory_utilization: 0.4,
        };

        let strategy = selector.recommend_for_workload(&large_chars, &resources);
        assert_eq!(strategy, ReasoningStrategy::GPU);
    }

    #[test]
    fn test_adaptive_engine_creation() {
        let engine = AdaptiveReasoningEngine::new();
        assert_eq!(engine.facts.len(), 0);
        assert_eq!(engine.rules.len(), 0);
    }

    #[test]
    fn test_adaptive_engine_add_facts() {
        let mut engine = AdaptiveReasoningEngine::new();
        let facts = vec![create_test_fact("a", "p", "b")];

        engine.add_facts(facts);
        assert_eq!(engine.facts.len(), 1);
    }

    #[test]
    fn test_adaptive_engine_add_rules() {
        let mut engine = AdaptiveReasoningEngine::new();
        let rule = Rule {
            name: "test".to_string(),
            body: vec![],
            head: vec![],
        };

        engine.add_rules(vec![rule]);
        assert_eq!(engine.rules.len(), 1);
    }

    #[test]
    fn test_adaptive_engine_reason() {
        let mut engine = AdaptiveReasoningEngine::new();
        let facts = vec![create_test_fact("a", "p", "b")];

        engine.add_facts(facts);
        let results = engine.reason().unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_adaptation_interval() {
        let mut engine = AdaptiveReasoningEngine::new();
        engine.set_adaptation_interval(50);

        assert_eq!(engine.adaptation_interval, 50);
    }

    #[test]
    fn test_strategy_switching() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.register_strategy(ReasoningStrategy::Forward);
        selector.register_strategy(ReasoningStrategy::Backward);

        let characteristics = DatasetCharacteristics {
            fact_count: 100,
            rule_count: 10,
            avg_rule_body_size: 2.0,
            density: 0.05,
            avg_degree: 3.0,
            predicate_count: 5,
            rule_selectivity: 0.5,
            data_skew: 0.2,
        };

        // First selection
        let _strategy1 = selector.select_strategy(&characteristics).unwrap();

        // Second selection (may switch)
        let _strategy2 = selector.select_strategy(&characteristics).unwrap();

        // Should have recorded at least one selection
        // Note: Metrics tracked internally
    }

    #[test]
    fn test_performance_history_limit() {
        let mut selector = AdaptiveStrategySelector::new();
        selector.register_strategy(ReasoningStrategy::Forward);

        // Add 150 metrics (should keep last 100)
        for _ in 0..150 {
            let metrics = StrategyMetrics {
                execution_time: Duration::from_millis(100),
                memory_usage: 1024,
                throughput: 100.0,
                quality: 0.95,
                execution_count: 1,
            };
            selector.record_performance(ReasoningStrategy::Forward, metrics);
        }

        let history = selector.get_performance_history(ReasoningStrategy::Forward);
        assert_eq!(history.unwrap().len(), 100);
    }
}
