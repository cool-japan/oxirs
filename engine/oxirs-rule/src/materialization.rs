//! Materialization Strategies for Rule-Based Reasoning
//!
//! Provides different strategies for materializing inferred knowledge:
//! - **Eager**: Materialize all derivable facts immediately
//! - **Lazy**: Materialize facts only when queried
//! - **Semi-Eager**: Selectively materialize based on heuristics
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::materialization::{MaterializationStrategy, EagerStrategy};
//! use oxirs_rule::{Rule, RuleEngine};
//!
//! let mut engine = RuleEngine::new();
//! let strategy = EagerStrategy::new();
//!
//! // Strategy will materialize all facts immediately
//! ```

use crate::{RuleAtom, RuleEngine};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Materialization strategy trait
pub trait MaterializationStrategy: Send + Sync {
    /// Materialize facts using this strategy
    fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy description
    fn description(&self) -> &str;
}

/// Eager materialization strategy
///
/// Computes all derivable facts immediately and stores them.
/// Best for: Small knowledge bases, frequent queries, complete reasoning
#[derive(Debug, Clone, Default)]
pub struct EagerStrategy {
    /// Maximum iterations for fixpoint computation
    pub max_iterations: usize,
}

impl EagerStrategy {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
        }
    }

    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl MaterializationStrategy for EagerStrategy {
    fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        info!("Starting eager materialization");

        // Add all initial facts
        engine.add_facts(initial_facts.to_vec());

        // Run forward chaining to fixpoint
        let all_facts = engine.forward_chain(initial_facts)?;

        info!("Eager materialization complete: {} facts", all_facts.len());
        Ok(all_facts)
    }

    fn name(&self) -> &str {
        "eager"
    }

    fn description(&self) -> &str {
        "Materialize all derivable facts immediately"
    }
}

/// Lazy materialization strategy
///
/// Computes facts only when they are queried.
/// Best for: Large knowledge bases, infrequent queries, partial reasoning
#[derive(Debug, Clone, Default)]
pub struct LazyStrategy {
    /// Cache of materialized facts
    cache: HashMap<RuleAtom, bool>,
}

impl LazyStrategy {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Check if a specific fact can be derived
    pub fn can_derive(&mut self, engine: &mut RuleEngine, target: &RuleAtom) -> Result<bool> {
        // Check cache first
        if let Some(&result) = self.cache.get(target) {
            debug!("Cache hit for fact");
            return Ok(result);
        }

        // Use backward chaining to check if target is derivable
        let result = engine.backward_chain(target)?;

        // Cache the result
        self.cache.insert(target.clone(), result);

        Ok(result)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl MaterializationStrategy for LazyStrategy {
    fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        info!("Lazy materialization: only storing initial facts");

        // Only store initial facts, don't derive anything yet
        engine.add_facts(initial_facts.to_vec());

        Ok(initial_facts.to_vec())
    }

    fn name(&self) -> &str {
        "lazy"
    }

    fn description(&self) -> &str {
        "Materialize facts only when queried"
    }
}

/// Semi-eager materialization strategy
///
/// Selectively materializes facts based on heuristics:
/// - Materializes frequently used predicates
/// - Materializes small derivation sets
/// - Uses cost-based analysis
///
/// Best for: Medium knowledge bases, mixed query patterns
#[derive(Debug, Clone)]
pub struct SemiEagerStrategy {
    /// Predicates to eagerly materialize
    pub eager_predicates: HashSet<String>,
    /// Maximum facts to materialize eagerly
    pub max_eager_facts: usize,
    /// Threshold for predicate frequency
    pub frequency_threshold: usize,
}

impl Default for SemiEagerStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiEagerStrategy {
    pub fn new() -> Self {
        Self {
            eager_predicates: HashSet::new(),
            max_eager_facts: 10000,
            frequency_threshold: 10,
        }
    }

    pub fn with_eager_predicates(mut self, predicates: Vec<String>) -> Self {
        self.eager_predicates = predicates.into_iter().collect();
        self
    }

    pub fn with_max_eager_facts(mut self, max: usize) -> Self {
        self.max_eager_facts = max;
        self
    }

    /// Analyze predicates and decide which to materialize eagerly
    fn analyze_predicates(&self, facts: &[RuleAtom]) -> HashSet<String> {
        let mut predicate_counts = HashMap::new();

        // Count predicate frequencies
        for fact in facts {
            if let RuleAtom::Triple {
                predicate: crate::Term::Constant(pred),
                ..
            } = fact
            {
                *predicate_counts.entry(pred.clone()).or_insert(0) += 1;
            }
        }

        // Select predicates above threshold
        let mut selected = self.eager_predicates.clone();
        for (pred, count) in predicate_counts {
            if count >= self.frequency_threshold {
                selected.insert(pred);
            }
        }

        selected
    }
}

impl MaterializationStrategy for SemiEagerStrategy {
    fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        info!("Starting semi-eager materialization");

        // Add initial facts
        engine.add_facts(initial_facts.to_vec());

        // Analyze which predicates to materialize
        let eager_preds = self.analyze_predicates(initial_facts);
        debug!("Eagerly materializing {} predicates", eager_preds.len());

        // Materialize facts for selected predicates
        let all_facts = engine.forward_chain(initial_facts)?;

        // Filter to keep only eagerly materialized predicates
        let mut materialized = initial_facts.to_vec();
        for fact in all_facts {
            if materialized.len() >= self.max_eager_facts {
                break;
            }

            if let RuleAtom::Triple {
                predicate: crate::Term::Constant(pred),
                ..
            } = &fact
            {
                if eager_preds.contains(pred) {
                    materialized.push(fact);
                }
            }
        }

        info!(
            "Semi-eager materialization complete: {} facts materialized",
            materialized.len()
        );
        Ok(materialized)
    }

    fn name(&self) -> &str {
        "semi-eager"
    }

    fn description(&self) -> &str {
        "Selectively materialize based on heuristics"
    }
}

/// Adaptive materialization strategy
///
/// Dynamically switches between strategies based on workload characteristics.
/// Uses machine learning-like heuristics to optimize performance.
#[derive(Debug, Clone)]
pub struct AdaptiveStrategy {
    /// Current active strategy
    current_strategy: StrategyType,
    /// Query count since last adaptation
    query_count: usize,
    /// Threshold for switching strategies
    adaptation_threshold: usize,
    /// Performance metrics
    metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Copy)]
enum StrategyType {
    Eager,
    Lazy,
    SemiEager,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PerformanceMetrics {
    total_queries: usize,
    cache_hits: usize,
    materialization_time_ms: u128,
    query_time_ms: u128,
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveStrategy {
    pub fn new() -> Self {
        Self {
            current_strategy: StrategyType::SemiEager,
            query_count: 0,
            adaptation_threshold: 100,
            metrics: PerformanceMetrics::default(),
        }
    }

    /// Analyze performance and adapt strategy
    fn adapt(&mut self) {
        let cache_hit_rate = if self.metrics.total_queries > 0 {
            self.metrics.cache_hits as f64 / self.metrics.total_queries as f64
        } else {
            0.0
        };

        // Switch to eager if high cache hit rate
        if cache_hit_rate > 0.8 {
            debug!("Adapting to eager strategy (high cache hit rate)");
            self.current_strategy = StrategyType::Eager;
        }
        // Switch to lazy if low cache hit rate
        else if cache_hit_rate < 0.2 {
            debug!("Adapting to lazy strategy (low cache hit rate)");
            self.current_strategy = StrategyType::Lazy;
        }
        // Stay semi-eager otherwise
        else {
            self.current_strategy = StrategyType::SemiEager;
        }

        // Reset metrics
        self.query_count = 0;
    }

    /// Record a query for adaptation
    pub fn record_query(&mut self, cache_hit: bool) {
        self.metrics.total_queries += 1;
        if cache_hit {
            self.metrics.cache_hits += 1;
        }

        self.query_count += 1;
        if self.query_count >= self.adaptation_threshold {
            self.adapt();
        }
    }
}

impl MaterializationStrategy for AdaptiveStrategy {
    fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        info!(
            "Adaptive materialization using {:?} strategy",
            self.current_strategy
        );

        match self.current_strategy {
            StrategyType::Eager => EagerStrategy::new().materialize(engine, initial_facts),
            StrategyType::Lazy => LazyStrategy::new().materialize(engine, initial_facts),
            StrategyType::SemiEager => SemiEagerStrategy::new().materialize(engine, initial_facts),
        }
    }

    fn name(&self) -> &str {
        "adaptive"
    }

    fn description(&self) -> &str {
        "Dynamically adapt strategy based on workload"
    }
}

/// Materialization manager
///
/// Manages different materialization strategies and provides a unified interface
pub struct MaterializationManager {
    strategies: HashMap<String, Box<dyn MaterializationStrategy>>,
    active_strategy: String,
}

impl MaterializationManager {
    pub fn new() -> Self {
        let mut manager = Self {
            strategies: HashMap::new(),
            active_strategy: String::from("semi-eager"),
        };

        // Register default strategies
        manager.register_strategy(Box::new(EagerStrategy::new()));
        manager.register_strategy(Box::new(LazyStrategy::new()));
        manager.register_strategy(Box::new(SemiEagerStrategy::new()));
        manager.register_strategy(Box::new(AdaptiveStrategy::new()));

        manager
    }

    /// Register a materialization strategy
    pub fn register_strategy(&mut self, strategy: Box<dyn MaterializationStrategy>) {
        let name = strategy.name().to_string();
        self.strategies.insert(name, strategy);
    }

    /// Set the active strategy
    pub fn set_active_strategy(&mut self, name: &str) -> Result<()> {
        if !self.strategies.contains_key(name) {
            return Err(anyhow::anyhow!("Strategy '{}' not found", name));
        }
        self.active_strategy = name.to_string();
        info!("Active materialization strategy set to '{}'", name);
        Ok(())
    }

    /// Get the active strategy
    pub fn get_active_strategy(&self) -> Option<&dyn MaterializationStrategy> {
        self.strategies
            .get(&self.active_strategy)
            .map(|s| s.as_ref())
    }

    /// List available strategies
    pub fn list_strategies(&self) -> Vec<(&str, &str)> {
        self.strategies
            .values()
            .map(|s| (s.name(), s.description()))
            .collect()
    }

    /// Materialize using the active strategy
    pub fn materialize(
        &self,
        engine: &mut RuleEngine,
        initial_facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        let strategy = self
            .get_active_strategy()
            .ok_or_else(|| anyhow::anyhow!("No active strategy"))?;

        info!("Materializing with {} strategy", strategy.name());
        strategy.materialize(engine, initial_facts)
    }
}

impl Default for MaterializationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_eager_strategy() {
        let mut engine = RuleEngine::new();
        let strategy = EagerStrategy::new();

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let result = strategy.materialize(&mut engine, &facts).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_lazy_strategy() {
        let mut engine = RuleEngine::new();
        let strategy = LazyStrategy::new();

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let result = strategy.materialize(&mut engine, &facts).unwrap();
        // Lazy strategy only stores initial facts
        assert_eq!(result.len(), facts.len());
    }

    #[test]
    fn test_materialization_manager() {
        let mut manager = MaterializationManager::new();

        // Check default strategies are registered
        let strategies = manager.list_strategies();
        assert!(strategies.len() >= 4);

        // Set active strategy
        manager.set_active_strategy("eager").unwrap();
        assert_eq!(manager.active_strategy, "eager");
    }
}
