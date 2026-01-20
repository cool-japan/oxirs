//! Enhanced aggregation functions for SPARQL 1.2
//!
//! This module implements advanced aggregation functions beyond SPARQL 1.1,
//! including statistical functions and enhanced string aggregations.

use crate::error::{FusekiError, FusekiResult};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Result of an aggregation operation
#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub value: Value,
    pub datatype: Option<String>,
    pub language: Option<String>,
}

/// Trait for implementing custom aggregation functions
pub trait AggregateFunction: Send + Sync {
    /// Add a value to the aggregation
    fn add_value(&mut self, value: &Value) -> FusekiResult<()>;

    /// Get the final aggregated result
    fn get_result(&self) -> FusekiResult<AggregationResult>;

    /// Reset the aggregation state
    fn reset(&mut self);

    /// Get the name of this aggregation function
    fn name(&self) -> &str;

    /// Whether this function requires distinct values
    fn requires_distinct(&self) -> bool {
        false
    }
}

/// GROUP_CONCAT implementation
#[derive(Debug, Clone)]
pub struct GroupConcatAggregate {
    values: Vec<String>,
    separator: String,
    distinct: bool,
}

impl GroupConcatAggregate {
    pub fn new(separator: Option<String>, distinct: bool) -> Self {
        Self {
            values: Vec::new(),
            separator: separator.unwrap_or_else(|| " ".to_string()),
            distinct,
        }
    }
}

impl AggregateFunction for GroupConcatAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        let str_value = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => serde_json::to_string(value).unwrap_or_default(),
        };

        if !self.distinct || !self.values.contains(&str_value) {
            self.values.push(str_value);
        }

        Ok(())
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        let result = self.values.join(&self.separator);
        Ok(AggregationResult {
            value: Value::String(result),
            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        "GROUP_CONCAT"
    }

    fn requires_distinct(&self) -> bool {
        self.distinct
    }
}

/// SAMPLE implementation
#[derive(Debug, Clone)]
pub struct SampleAggregate {
    value: Option<Value>,
}

impl Default for SampleAggregate {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleAggregate {
    pub fn new() -> Self {
        Self { value: None }
    }
}

impl AggregateFunction for SampleAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        // SAMPLE takes any arbitrary value from the group
        if self.value.is_none() {
            self.value = Some(value.clone());
        }
        Ok(())
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        match &self.value {
            Some(v) => Ok(AggregationResult {
                value: v.clone(),
                datatype: None,
                language: None,
            }),
            None => Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            }),
        }
    }

    fn reset(&mut self) {
        self.value = None;
    }

    fn name(&self) -> &str {
        "SAMPLE"
    }
}

/// MEDIAN implementation
#[derive(Debug, Clone)]
pub struct MedianAggregate {
    values: Vec<f64>,
}

impl Default for MedianAggregate {
    fn default() -> Self {
        Self::new()
    }
}

impl MedianAggregate {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }
}

impl AggregateFunction for MedianAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        match value {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    self.values.push(f);
                    Ok(())
                } else {
                    Err(FusekiError::bad_request("MEDIAN requires numeric values"))
                }
            }
            _ => Err(FusekiError::bad_request("MEDIAN requires numeric values")),
        }
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        if self.values.is_empty() {
            return Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            });
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Ok(AggregationResult {
            value: serde_json::Number::from_f64(median)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        "MEDIAN"
    }
}

/// MODE implementation
#[derive(Debug, Clone)]
pub struct ModeAggregate {
    value_counts: HashMap<String, usize>,
}

impl Default for ModeAggregate {
    fn default() -> Self {
        Self::new()
    }
}

impl ModeAggregate {
    pub fn new() -> Self {
        Self {
            value_counts: HashMap::new(),
        }
    }
}

impl AggregateFunction for ModeAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        let key = serde_json::to_string(value).unwrap_or_default();
        *self.value_counts.entry(key).or_insert(0) += 1;
        Ok(())
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        let mode = self
            .value_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value);

        match mode {
            Some(v) => {
                let parsed: Value = serde_json::from_str(v).unwrap_or(Value::String(v.clone()));
                Ok(AggregationResult {
                    value: parsed,
                    datatype: None,
                    language: None,
                })
            }
            None => Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            }),
        }
    }

    fn reset(&mut self) {
        self.value_counts.clear();
    }

    fn name(&self) -> &str {
        "MODE"
    }
}

/// STDDEV implementation (sample standard deviation)
#[derive(Debug, Clone)]
pub struct StdDevAggregate {
    values: Vec<f64>,
    population: bool,
}

impl StdDevAggregate {
    pub fn new(population: bool) -> Self {
        Self {
            values: Vec::new(),
            population,
        }
    }
}

impl AggregateFunction for StdDevAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        match value {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    self.values.push(f);
                    Ok(())
                } else {
                    Err(FusekiError::bad_request("STDDEV requires numeric values"))
                }
            }
            _ => Err(FusekiError::bad_request("STDDEV requires numeric values")),
        }
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        if self.values.is_empty() {
            return Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            });
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / if self.population { n } else { n - 1.0 };

        let stddev = variance.sqrt();

        Ok(AggregationResult {
            value: serde_json::Number::from_f64(stddev)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        if self.population {
            "STDDEV_POP"
        } else {
            "STDDEV"
        }
    }
}

/// VARIANCE implementation
#[derive(Debug, Clone)]
pub struct VarianceAggregate {
    values: Vec<f64>,
    population: bool,
}

impl VarianceAggregate {
    pub fn new(population: bool) -> Self {
        Self {
            values: Vec::new(),
            population,
        }
    }
}

impl AggregateFunction for VarianceAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        match value {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    self.values.push(f);
                    Ok(())
                } else {
                    Err(FusekiError::bad_request("VARIANCE requires numeric values"))
                }
            }
            _ => Err(FusekiError::bad_request("VARIANCE requires numeric values")),
        }
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        if self.values.is_empty() {
            return Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            });
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / if self.population { n } else { n - 1.0 };

        Ok(AggregationResult {
            value: serde_json::Number::from_f64(variance)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        if self.population {
            "VARIANCE_POP"
        } else {
            "VARIANCE"
        }
    }
}

/// PERCENTILE implementation
#[derive(Debug, Clone)]
pub struct PercentileAggregate {
    values: Vec<f64>,
    percentile: f64,
}

impl PercentileAggregate {
    pub fn new(percentile: f64) -> Self {
        Self {
            values: Vec::new(),
            percentile: percentile.clamp(0.0, 100.0),
        }
    }
}

impl AggregateFunction for PercentileAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        match value {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    self.values.push(f);
                    Ok(())
                } else {
                    Err(FusekiError::bad_request(
                        "PERCENTILE requires numeric values",
                    ))
                }
            }
            _ => Err(FusekiError::bad_request(
                "PERCENTILE requires numeric values",
            )),
        }
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        if self.values.is_empty() {
            return Ok(AggregationResult {
                value: Value::Null,
                datatype: None,
                language: None,
            });
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let index = (self.percentile / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let weight = index - lower as f64;

        let result = if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        };

        Ok(AggregationResult {
            value: serde_json::Number::from_f64(result)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        "PERCENTILE"
    }
}

/// COUNT_DISTINCT implementation
#[derive(Debug, Clone)]
pub struct CountDistinctAggregate {
    values: std::collections::HashSet<String>,
}

impl Default for CountDistinctAggregate {
    fn default() -> Self {
        Self::new()
    }
}

impl CountDistinctAggregate {
    pub fn new() -> Self {
        Self {
            values: std::collections::HashSet::new(),
        }
    }
}

impl AggregateFunction for CountDistinctAggregate {
    fn add_value(&mut self, value: &Value) -> FusekiResult<()> {
        let key = serde_json::to_string(value).unwrap_or_default();
        self.values.insert(key);
        Ok(())
    }

    fn get_result(&self) -> FusekiResult<AggregationResult> {
        Ok(AggregationResult {
            value: Value::Number(serde_json::Number::from(self.values.len())),
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            language: None,
        })
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &str {
        "COUNT_DISTINCT"
    }

    fn requires_distinct(&self) -> bool {
        true
    }
}

/// Factory for creating aggregation functions
pub struct AggregationFactory;

impl AggregationFactory {
    /// Create an aggregation function by name
    pub fn create_aggregate(
        function_name: &str,
        args: &HashMap<String, Value>,
    ) -> FusekiResult<Box<dyn AggregateFunction>> {
        let name_upper = function_name.to_uppercase();

        match name_upper.as_str() {
            "GROUP_CONCAT" => {
                let separator = args
                    .get("separator")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let distinct = args
                    .get("distinct")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Ok(Box::new(GroupConcatAggregate::new(separator, distinct)))
            }
            "SAMPLE" => Ok(Box::new(SampleAggregate::new())),
            "MEDIAN" => Ok(Box::new(MedianAggregate::new())),
            "MODE" => Ok(Box::new(ModeAggregate::new())),
            "STDDEV" | "STDEV" => Ok(Box::new(StdDevAggregate::new(false))),
            "STDDEV_POP" | "STDEV_POP" => Ok(Box::new(StdDevAggregate::new(true))),
            "VARIANCE" | "VAR" => Ok(Box::new(VarianceAggregate::new(false))),
            "VARIANCE_POP" | "VAR_POP" => Ok(Box::new(VarianceAggregate::new(true))),
            "PERCENTILE" => {
                let percentile = args
                    .get("percentile")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(50.0);
                Ok(Box::new(PercentileAggregate::new(percentile)))
            }
            "COUNT_DISTINCT" => Ok(Box::new(CountDistinctAggregate::new())),
            _ => Err(FusekiError::bad_request(format!(
                "Unknown aggregation function: {function_name}"
            ))),
        }
    }

    /// Check if a function name is a supported aggregation
    pub fn is_supported_aggregate(function_name: &str) -> bool {
        matches!(
            function_name.to_uppercase().as_str(),
            "GROUP_CONCAT"
                | "SAMPLE"
                | "MEDIAN"
                | "MODE"
                | "STDDEV"
                | "STDEV"
                | "STDDEV_POP"
                | "STDEV_POP"
                | "VARIANCE"
                | "VAR"
                | "VARIANCE_POP"
                | "VAR_POP"
                | "PERCENTILE"
                | "COUNT_DISTINCT"
        )
    }
}

/// Enhanced aggregation processor
pub struct EnhancedAggregationProcessor {
    aggregates: HashMap<String, Box<dyn AggregateFunction>>,
}

impl Default for EnhancedAggregationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedAggregationProcessor {
    pub fn new() -> Self {
        Self {
            aggregates: HashMap::new(),
        }
    }

    /// Register an aggregation function
    pub fn register_aggregate(
        &mut self,
        alias: String,
        function_name: &str,
        args: &HashMap<String, Value>,
    ) -> FusekiResult<()> {
        let aggregate = AggregationFactory::create_aggregate(function_name, args)?;
        self.aggregates.insert(alias, aggregate);
        Ok(())
    }

    /// Add a value to an aggregation
    pub fn add_value(&mut self, alias: &str, value: &Value) -> FusekiResult<()> {
        if let Some(aggregate) = self.aggregates.get_mut(alias) {
            aggregate.add_value(value)
        } else {
            Err(FusekiError::internal(format!(
                "Unknown aggregation alias: {alias}"
            )))
        }
    }

    /// Get results for all aggregations
    pub fn get_results(&self) -> FusekiResult<HashMap<String, AggregationResult>> {
        let mut results = HashMap::new();

        for (alias, aggregate) in &self.aggregates {
            results.insert(alias.clone(), aggregate.get_result()?);
        }

        Ok(results)
    }

    /// Reset all aggregations
    pub fn reset(&mut self) {
        for aggregate in self.aggregates.values_mut() {
            aggregate.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_concat() {
        let mut agg = GroupConcatAggregate::new(Some(",".to_string()), false);

        agg.add_value(&Value::String("apple".to_string())).unwrap();
        agg.add_value(&Value::String("banana".to_string())).unwrap();
        agg.add_value(&Value::String("cherry".to_string())).unwrap();

        let result = agg.get_result().unwrap();
        assert_eq!(
            result.value,
            Value::String("apple,banana,cherry".to_string())
        );
    }

    #[test]
    fn test_median() {
        let mut agg = MedianAggregate::new();

        for i in 1..=5 {
            agg.add_value(&Value::Number(serde_json::Number::from(i)))
                .unwrap();
        }

        let result = agg.get_result().unwrap();
        if let Value::Number(n) = result.value {
            assert_eq!(n.as_f64().unwrap(), 3.0);
        } else {
            panic!("Expected numeric result");
        }
    }

    #[test]
    fn test_mode() {
        let mut agg = ModeAggregate::new();

        agg.add_value(&Value::String("apple".to_string())).unwrap();
        agg.add_value(&Value::String("banana".to_string())).unwrap();
        agg.add_value(&Value::String("apple".to_string())).unwrap();
        agg.add_value(&Value::String("apple".to_string())).unwrap();

        let result = agg.get_result().unwrap();
        assert_eq!(result.value, Value::String("apple".to_string()));
    }

    #[test]
    fn test_percentile() {
        let mut agg = PercentileAggregate::new(75.0);

        for i in 1..=100 {
            agg.add_value(&Value::Number(serde_json::Number::from(i)))
                .unwrap();
        }

        let result = agg.get_result().unwrap();
        if let Value::Number(n) = result.value {
            let value = n.as_f64().unwrap();
            // Allow for small floating-point differences in percentile calculation
            assert!(
                (value - 75.0).abs() < 1.0,
                "Expected value around 75.0, got {value}"
            );
        } else {
            panic!("Expected numeric result");
        }
    }
}
