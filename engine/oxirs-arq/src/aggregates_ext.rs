//! Extended Aggregate Functions for SPARQL
//!
//! This module provides a framework for custom aggregate functions and extensions
//! to the standard SPARQL aggregates (COUNT, SUM, MIN, MAX, AVG, SAMPLE, GROUP_CONCAT).
//!
//! Based on Apache Jena ARQ's aggregate extension framework.

use crate::algebra::{Binding, Expression, Term};
use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Accumulator for aggregate computation
///
/// An accumulator maintains state during aggregate computation over a group.
/// Each binding in the group is processed by calling `accumulate()`, and
/// the final result is retrieved with `get_value()`.
pub trait Accumulator: Send + Sync {
    /// Process one binding from the group
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()>;

    /// Get the final aggregate value
    fn get_value(&self) -> Result<Term>;

    /// Reset accumulator state (for reuse)
    fn reset(&mut self);

    /// Clone the accumulator
    fn clone_accumulator(&self) -> Box<dyn Accumulator>;
}

/// Factory for creating accumulators
pub trait AggregateFactory: Send + Sync {
    /// Get the URI/name of this aggregate
    fn uri(&self) -> &str;

    /// Get short name (e.g., "median" instead of full URI)
    fn name(&self) -> &str {
        self.uri().rsplit('/').next().unwrap_or(self.uri())
    }

    /// Create a new accumulator instance
    fn create_accumulator(&self, distinct: bool) -> Box<dyn Accumulator>;

    /// Validate the aggregate during query planning
    fn validate(&self, expr: &Expression) -> Result<()> {
        let _ = expr;
        Ok(())
    }

    /// Get documentation for this aggregate
    fn documentation(&self) -> &str {
        "No documentation available"
    }
}

/// Registry for custom aggregate functions
pub struct AggregateRegistry {
    aggregates: DashMap<String, Arc<dyn AggregateFactory>>,
}

impl AggregateRegistry {
    /// Create a new aggregate registry
    pub fn new() -> Self {
        Self {
            aggregates: DashMap::new(),
        }
    }

    /// Register a custom aggregate factory
    pub fn register<F: AggregateFactory + 'static>(&self, factory: F) -> Result<()> {
        let uri = factory.uri().to_string();
        self.aggregates.insert(uri, Arc::new(factory));
        Ok(())
    }

    /// Get an aggregate factory by URI
    pub fn get(&self, uri: &str) -> Option<Arc<dyn AggregateFactory>> {
        self.aggregates.get(uri).map(|entry| Arc::clone(&*entry))
    }

    /// Check if an aggregate is registered
    pub fn is_registered(&self, uri: &str) -> bool {
        self.aggregates.contains_key(uri)
    }

    /// Get all registered aggregate URIs
    pub fn registered_uris(&self) -> Vec<String> {
        self.aggregates
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Create a registry with standard extended aggregates
    pub fn with_standard_aggregates() -> Result<Self> {
        let registry = Self::new();
        register_standard_aggregates(&registry)?;
        Ok(registry)
    }
}

impl Default for AggregateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Register standard extended aggregates
pub fn register_standard_aggregates(registry: &AggregateRegistry) -> Result<()> {
    registry.register(MedianAggregateFactory)?;
    registry.register(ModeAggregateFactory)?;
    registry.register(StdDevAggregateFactory)?;
    registry.register(VarianceAggregateFactory)?;
    registry.register(FirstAggregateFactory)?;
    registry.register(LastAggregateFactory)?;
    Ok(())
}

// ============================================================================
// Standard Extended Aggregates
// ============================================================================

/// MEDIAN aggregate - returns median value of numeric expressions
struct MedianAggregateFactory;

impl AggregateFactory for MedianAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#median"
    }

    fn name(&self) -> &str {
        "MEDIAN"
    }

    fn documentation(&self) -> &str {
        "Returns the median value of numeric expressions in a group"
    }

    fn create_accumulator(&self, distinct: bool) -> Box<dyn Accumulator> {
        Box::new(MedianAccumulator {
            values: Vec::new(),
            distinct,
        })
    }
}

struct MedianAccumulator {
    values: Vec<f64>,
    distinct: bool,
}

impl Accumulator for MedianAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        // For now, just accumulate if expr is a variable
        if let Expression::Variable(var) = expr {
            if let Some(Term::Literal(lit)) = binding.get(var) {
                if let Ok(num) = lit.value.parse::<f64>() {
                    self.values.push(num);
                }
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        let mut values = self.values.clone();

        if self.distinct {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup_by(|a, b| a == b);
        }

        if values.is_empty() {
            bail!("No numeric values for MEDIAN");
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        Ok(Term::Literal(crate::algebra::Literal {
            value: median.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#decimal",
            )),
        }))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self {
            values: Vec::new(),
            distinct: self.distinct,
        })
    }
}

/// MODE aggregate - returns most frequent value
struct ModeAggregateFactory;

impl AggregateFactory for ModeAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#mode"
    }

    fn name(&self) -> &str {
        "MODE"
    }

    fn documentation(&self) -> &str {
        "Returns the most frequently occurring value in a group"
    }

    fn create_accumulator(&self, _distinct: bool) -> Box<dyn Accumulator> {
        Box::new(ModeAccumulator {
            counts: HashMap::new(),
        })
    }
}

struct ModeAccumulator {
    counts: HashMap<String, usize>,
}

impl Accumulator for ModeAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        if let Expression::Variable(var) = expr {
            if let Some(term) = binding.get(var) {
                let key = term.to_string();
                *self.counts.entry(key).or_insert(0) += 1;
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        if self.counts.is_empty() {
            bail!("No values for MODE");
        }

        let (mode_value, _) = self
            .counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .ok_or_else(|| anyhow!("Cannot determine mode"))?;

        Ok(Term::Literal(crate::algebra::Literal {
            value: mode_value.clone(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#string",
            )),
        }))
    }

    fn reset(&mut self) {
        self.counts.clear();
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self {
            counts: HashMap::new(),
        })
    }
}

/// STDEV aggregate - returns standard deviation
struct StdDevAggregateFactory;

impl AggregateFactory for StdDevAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#stdev"
    }

    fn name(&self) -> &str {
        "STDEV"
    }

    fn documentation(&self) -> &str {
        "Returns the standard deviation of numeric values in a group"
    }

    fn create_accumulator(&self, distinct: bool) -> Box<dyn Accumulator> {
        Box::new(StdDevAccumulator {
            values: Vec::new(),
            distinct,
        })
    }
}

struct StdDevAccumulator {
    values: Vec<f64>,
    distinct: bool,
}

impl Accumulator for StdDevAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        if let Expression::Variable(var) = expr {
            if let Some(Term::Literal(lit)) = binding.get(var) {
                if let Ok(num) = lit.value.parse::<f64>() {
                    self.values.push(num);
                }
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        let mut values = self.values.clone();

        if self.distinct {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup_by(|a, b| a == b);
        }

        if values.is_empty() {
            bail!("No numeric values for STDEV");
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let stdev = variance.sqrt();

        Ok(Term::Literal(crate::algebra::Literal {
            value: stdev.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#decimal",
            )),
        }))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self {
            values: Vec::new(),
            distinct: self.distinct,
        })
    }
}

/// VARIANCE aggregate - returns variance
struct VarianceAggregateFactory;

impl AggregateFactory for VarianceAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#variance"
    }

    fn name(&self) -> &str {
        "VARIANCE"
    }

    fn documentation(&self) -> &str {
        "Returns the variance of numeric values in a group"
    }

    fn create_accumulator(&self, distinct: bool) -> Box<dyn Accumulator> {
        Box::new(VarianceAccumulator {
            values: Vec::new(),
            distinct,
        })
    }
}

struct VarianceAccumulator {
    values: Vec<f64>,
    distinct: bool,
}

impl Accumulator for VarianceAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        if let Expression::Variable(var) = expr {
            if let Some(Term::Literal(lit)) = binding.get(var) {
                if let Ok(num) = lit.value.parse::<f64>() {
                    self.values.push(num);
                }
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        let mut values = self.values.clone();

        if self.distinct {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup_by(|a, b| a == b);
        }

        if values.is_empty() {
            bail!("No numeric values for VARIANCE");
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        Ok(Term::Literal(crate::algebra::Literal {
            value: variance.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#decimal",
            )),
        }))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self {
            values: Vec::new(),
            distinct: self.distinct,
        })
    }
}

/// FIRST aggregate - returns first value (order-dependent)
struct FirstAggregateFactory;

impl AggregateFactory for FirstAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#first"
    }

    fn name(&self) -> &str {
        "FIRST"
    }

    fn documentation(&self) -> &str {
        "Returns the first value in a group (order-dependent)"
    }

    fn create_accumulator(&self, _distinct: bool) -> Box<dyn Accumulator> {
        Box::new(FirstAccumulator { first_value: None })
    }
}

struct FirstAccumulator {
    first_value: Option<Term>,
}

impl Accumulator for FirstAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        if self.first_value.is_none() {
            if let Expression::Variable(var) = expr {
                if let Some(term) = binding.get(var) {
                    self.first_value = Some(term.clone());
                }
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        self.first_value
            .clone()
            .ok_or_else(|| anyhow!("No values for FIRST"))
    }

    fn reset(&mut self) {
        self.first_value = None;
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self { first_value: None })
    }
}

/// LAST aggregate - returns last value (order-dependent)
struct LastAggregateFactory;

impl AggregateFactory for LastAggregateFactory {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/function#last"
    }

    fn name(&self) -> &str {
        "LAST"
    }

    fn documentation(&self) -> &str {
        "Returns the last value in a group (order-dependent)"
    }

    fn create_accumulator(&self, _distinct: bool) -> Box<dyn Accumulator> {
        Box::new(LastAccumulator { last_value: None })
    }
}

struct LastAccumulator {
    last_value: Option<Term>,
}

impl Accumulator for LastAccumulator {
    fn accumulate(&mut self, binding: &Binding, expr: &Expression) -> Result<()> {
        if let Expression::Variable(var) = expr {
            if let Some(term) = binding.get(var) {
                self.last_value = Some(term.clone());
            }
        }
        Ok(())
    }

    fn get_value(&self) -> Result<Term> {
        self.last_value
            .clone()
            .ok_or_else(|| anyhow!("No values for LAST"))
    }

    fn reset(&mut self) {
        self.last_value = None;
    }

    fn clone_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(Self { last_value: None })
    }
}

// ============================================================================
// Aggregate Optimization Hints
// ============================================================================

/// Optimization hints for aggregate execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOptimization {
    /// No special optimization
    None,
    /// Aggregate can be computed incrementally
    Incremental,
    /// Aggregate only needs to scan first/last value
    EarlyTermination,
    /// Aggregate can benefit from sorted input
    PreferSorted,
    /// Aggregate can be parallelized
    Parallelizable,
}

/// Metadata about aggregate optimization
pub struct AggregateMetadata {
    /// Optimization hints
    pub optimization: AggregateOptimization,
    /// Whether the aggregate is order-dependent
    pub order_dependent: bool,
    /// Whether the aggregate can handle null values
    pub null_handling: bool,
    /// Expected memory usage category
    pub memory_usage: MemoryUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUsage {
    Constant,
    Linear,
    Quadratic,
}

impl AggregateMetadata {
    /// Get metadata for standard aggregates
    pub fn for_aggregate_name(name: &str) -> Self {
        match name.to_uppercase().as_str() {
            "COUNT" => Self {
                optimization: AggregateOptimization::Incremental,
                order_dependent: false,
                null_handling: true,
                memory_usage: MemoryUsage::Constant,
            },
            "SUM" | "AVG" => Self {
                optimization: AggregateOptimization::Incremental,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Constant,
            },
            "MIN" | "MAX" => Self {
                optimization: AggregateOptimization::PreferSorted,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Constant,
            },
            "SAMPLE" => Self {
                optimization: AggregateOptimization::EarlyTermination,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Constant,
            },
            "FIRST" | "LAST" => Self {
                optimization: AggregateOptimization::EarlyTermination,
                order_dependent: true,
                null_handling: false,
                memory_usage: MemoryUsage::Constant,
            },
            "GROUP_CONCAT" => Self {
                optimization: AggregateOptimization::Incremental,
                order_dependent: true,
                null_handling: false,
                memory_usage: MemoryUsage::Linear,
            },
            "MEDIAN" => Self {
                optimization: AggregateOptimization::PreferSorted,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Linear,
            },
            "STDEV" | "VARIANCE" => Self {
                optimization: AggregateOptimization::None,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Linear,
            },
            _ => Self {
                optimization: AggregateOptimization::None,
                order_dependent: false,
                null_handling: false,
                memory_usage: MemoryUsage::Linear,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::Variable;

    #[test]
    fn test_aggregate_registry() {
        let registry = AggregateRegistry::new();
        registry.register(MedianAggregateFactory).unwrap();

        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#median"));
        assert!(!registry.is_registered("http://example.org/unknown"));
    }

    #[test]
    fn test_standard_aggregates_registration() {
        let registry = AggregateRegistry::with_standard_aggregates().unwrap();

        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#median"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#mode"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#stdev"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#variance"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#first"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/function#last"));

        let uris = registry.registered_uris();
        assert_eq!(uris.len(), 6);
    }

    #[test]
    fn test_median_accumulator() {
        let factory = MedianAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate values: 1, 2, 3, 4, 5
        for i in 1..=5 {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            let value: f64 = lit.value.parse().unwrap();
            assert!((value - 3.0).abs() < 0.001);
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_median_even_count() {
        let factory = MedianAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate values: 1, 2, 3, 4
        for i in 1..=4 {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            let value: f64 = lit.value.parse().unwrap();
            assert!((value - 2.5).abs() < 0.001);
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_mode_accumulator() {
        let factory = ModeAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate: "a", "b", "a", "c", "a"
        for val in &["a", "b", "a", "c", "a"] {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: val.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            // Mode should be "a" (appears 3 times)
            assert!(lit.value.contains('a'));
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_stdev_accumulator() {
        let factory = StdDevAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate values: 2, 4, 4, 4, 5, 5, 7, 9
        for i in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            let value: f64 = lit.value.parse().unwrap();
            // Expected stdev ≈ 2.0
            assert!((value - 2.0).abs() < 0.1);
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_variance_accumulator() {
        let factory = VarianceAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate values: 1, 2, 3, 4, 5
        for i in 1..=5 {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            let value: f64 = lit.value.parse().unwrap();
            // Expected variance = 2.0
            assert!((value - 2.0).abs() < 0.001);
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_first_accumulator() {
        let factory = FirstAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        for val in &["first", "second", "third"] {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: val.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            assert_eq!(lit.value, "first");
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_last_accumulator() {
        let factory = LastAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        for val in &["first", "second", "third"] {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: val.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            assert_eq!(lit.value, "third");
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_distinct_median() {
        let factory = MedianAggregateFactory;
        let mut acc = factory.create_accumulator(true);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        // Accumulate: 1, 2, 2, 3, 3, 3
        for val in &[1, 2, 2, 3, 3, 3] {
            binding.insert(
                var.clone(),
                Term::Literal(crate::algebra::Literal {
                    value: val.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            acc.accumulate(&binding, &Expression::Variable(var.clone()))
                .unwrap();
        }

        let result = acc.get_value().unwrap();
        if let Term::Literal(lit) = result {
            let value: f64 = lit.value.parse().unwrap();
            // DISTINCT gives us [1, 2, 3], median = 2
            assert!((value - 2.0).abs() < 0.001);
        } else {
            panic!("Expected literal result");
        }
    }

    #[test]
    fn test_accumulator_reset() {
        let factory = FirstAggregateFactory;
        let mut acc = factory.create_accumulator(false);

        let var = Variable::new("x").unwrap();
        let mut binding = HashMap::new();

        binding.insert(
            var.clone(),
            Term::Literal(crate::algebra::Literal {
                value: "first".to_string(),
                language: None,
                datatype: None,
            }),
        );
        acc.accumulate(&binding, &Expression::Variable(var.clone()))
            .unwrap();

        assert!(acc.get_value().is_ok());

        acc.reset();

        // After reset, should have no value
        assert!(acc.get_value().is_err());
    }

    #[test]
    fn test_aggregate_metadata() {
        let count_meta = AggregateMetadata::for_aggregate_name("COUNT");
        assert_eq!(count_meta.optimization, AggregateOptimization::Incremental);
        assert_eq!(count_meta.memory_usage, MemoryUsage::Constant);
        assert!(count_meta.null_handling);

        let median_meta = AggregateMetadata::for_aggregate_name("MEDIAN");
        assert_eq!(
            median_meta.optimization,
            AggregateOptimization::PreferSorted
        );
        assert_eq!(median_meta.memory_usage, MemoryUsage::Linear);

        let first_meta = AggregateMetadata::for_aggregate_name("FIRST");
        assert_eq!(
            first_meta.optimization,
            AggregateOptimization::EarlyTermination
        );
        assert!(first_meta.order_dependent);
    }

    #[test]
    fn test_factory_name_extraction() {
        let factory = MedianAggregateFactory;
        assert_eq!(factory.name(), "MEDIAN");

        let factory = StdDevAggregateFactory;
        assert_eq!(factory.name(), "STDEV");
    }

    #[test]
    fn test_empty_accumulator() {
        let factory = MedianAggregateFactory;
        let acc = factory.create_accumulator(false);

        let result = acc.get_value();
        assert!(result.is_err());
    }
}
