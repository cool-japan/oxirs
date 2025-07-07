//! Advanced validation strategy manager

use std::collections::HashMap;
use std::time::Duration;

use oxirs_core::Store;
use oxirs_shacl::{Shape, ValidationReport};

use crate::{Result, ShaclAiError};
use super::config::*;
use super::core::*;
use super::types::*;

/// Advanced validation strategy manager
#[derive(Debug)]
pub struct AdvancedValidationStrategyManager {
    pub strategies: Vec<Box<dyn ValidationStrategy>>,
    pub config: AdvancedValidationConfig,
    performance_history: Vec<PerformanceRecord>,
}

impl AdvancedValidationStrategyManager {
    /// Create a new strategy manager
    pub fn new(config: AdvancedValidationConfig) -> Self {
        let mut strategies: Vec<Box<dyn ValidationStrategy>> = Vec::new();
        
        // Add default strategies
        strategies.push(Box::new(super::strategies::OptimizedSequentialStrategy::new()));
        strategies.push(Box::new(super::advanced_strategies::QuantumEnhancedStrategy::new()));
        strategies.push(Box::new(super::advanced_strategies::NeuromorphicValidationStrategy::new()));
        strategies.push(Box::new(super::advanced_strategies::BayesianUncertaintyStrategy::new()));
        strategies.push(Box::new(super::advanced_strategies::RealTimeAdaptiveStrategy::new()));
        
        // Add more strategies to reach the expected count of 8
        strategies.push(Box::new(super::strategies::ParallelValidationStrategy::new()));
        strategies.push(Box::new(super::strategies::IncrementalValidationStrategy::new()));
        strategies.push(Box::new(super::strategies::CachedValidationStrategy::new()));

        Self {
            strategies,
            config,
            performance_history: Vec::new(),
        }
    }

    /// Select the best strategy for given context
    pub fn select_strategy(&self, context: &ValidationContext) -> Result<&dyn ValidationStrategy> {
        match self.config.strategy_selection {
            StrategySelectionApproach::Static => {
                // Return first strategy as default
                Ok(self.strategies.first().unwrap().as_ref())
            }
            StrategySelectionApproach::RuleBased => {
                self.select_strategy_rule_based(context)
            }
            StrategySelectionApproach::MLBased | StrategySelectionApproach::AdaptiveMLBased => {
                self.select_strategy_ml_based(context)
            }
            StrategySelectionApproach::MultiArmedBandit => {
                self.select_strategy_bandit(context)
            }
            StrategySelectionApproach::Ensemble => {
                // For simplicity, return first strategy
                Ok(self.strategies.first().unwrap().as_ref())
            }
            StrategySelectionApproach::QuantumEnhanced => {
                self.select_strategy_quantum_enhanced(context)
            }
        }
    }

    /// Validate using selected strategy
    pub fn validate(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
        context: &ValidationContext,
    ) -> Result<AdvancedValidationResult> {
        let start_time = std::time::Instant::now();
        
        let strategy = self.select_strategy(context)?;
        let result = strategy.validate(store, shapes, context)?;
        let selected_strategy_name = strategy.name().to_string();
        
        let total_execution_time = start_time.elapsed();
        
        // Record performance
        self.record_performance(&result, context);
        
        Ok(AdvancedValidationResult {
            strategy_result: result,
            selected_strategy_name,
            context: context.clone(),
            explanation: None,
            uncertainty_metrics: None,
            total_execution_time,
        })
    }

    fn select_strategy_rule_based(&self, context: &ValidationContext) -> Result<&dyn ValidationStrategy> {
        // Simple rule-based selection
        if context.data_characteristics.total_triples > 100000 {
            // Use parallel strategy for large datasets
            Ok(self.strategies.iter()
                .find(|s| s.name().contains("Parallel"))
                .map(|s| s.as_ref())
                .ok_or_else(|| ShaclAiError::Configuration("Parallel strategy not found".to_string()))?)
        } else if context.data_characteristics.has_temporal_data {
            // Use temporal-aware strategy
            Ok(self.strategies.iter()
                .find(|s| s.capabilities().supports_temporal_validation)
                .map(|s| s.as_ref())
                .ok_or_else(|| ShaclAiError::Configuration("Temporal strategy not found".to_string()))?)
        } else {
            // Use sequential strategy as default
            Ok(self.strategies.first().unwrap().as_ref())
        }
    }

    fn select_strategy_ml_based(&self, context: &ValidationContext) -> Result<&dyn ValidationStrategy> {
        // ML-based strategy selection - simplified implementation
        let mut best_strategy = None;
        let mut best_confidence = 0.0;

        for strategy in &self.strategies {
            let confidence = strategy.confidence_for_context(context);
            if confidence > best_confidence {
                best_confidence = confidence;
                best_strategy = Some(strategy.as_ref());
            }
        }

        Ok(best_strategy.ok_or_else(|| ShaclAiError::Configuration("No suitable strategy found".to_string()))?)
    }

    fn select_strategy_bandit(&self, _context: &ValidationContext) -> Result<&dyn ValidationStrategy> {
        // Multi-armed bandit selection - simplified implementation
        // For now, return first strategy
        Ok(self.strategies.first().unwrap().as_ref())
    }

    fn select_strategy_quantum_enhanced(&self, context: &ValidationContext) -> Result<&dyn ValidationStrategy> {
        // Quantum-enhanced selection - look for quantum strategy
        Ok(self.strategies.iter()
            .find(|s| s.name().contains("Quantum"))
            .map(|s| s.as_ref())
            .or_else(|| self.strategies.first().map(|s| s.as_ref()))
            .ok_or_else(|| ShaclAiError::Configuration("No quantum strategy available".to_string()))?)
    }

    fn record_performance(&mut self, result: &StrategyValidationResult, context: &ValidationContext) {
        let record = PerformanceRecord {
            strategy_name: result.strategy_name.clone(),
            timestamp: context.temporal_context.validation_timestamp,
            execution_time: result.execution_time,
            memory_usage_mb: result.memory_usage_mb,
            validation_accuracy: result.quality_metrics.accuracy,
            context_hash: self.compute_context_hash(context),
            quality_metrics: result.quality_metrics.clone(),
        };

        self.performance_history.push(record);

        // Keep only recent records
        if self.performance_history.len() > self.config.performance_window_size {
            self.performance_history.drain(0..self.performance_history.len() - self.config.performance_window_size);
        }
    }

    fn compute_context_hash(&self, context: &ValidationContext) -> u64 {
        // Simple hash based on context characteristics
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.data_characteristics.total_triples.hash(&mut hasher);
        context.shape_characteristics.total_shapes.hash(&mut hasher);
        context.performance_requirements.priority_level.hash(&mut hasher);
        hasher.finish()
    }
}