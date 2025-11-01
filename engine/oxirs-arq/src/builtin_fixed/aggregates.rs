//! Built-in SPARQL Functions - Aggregate Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation, plus additional
//! statistical aggregates using scirs2-stats.

use crate::extensions::{AggregateState, CustomAggregate, Value};
use anyhow::{anyhow, bail, Result};

// Aggregate Functions

#[derive(Debug, Clone)]
pub(crate) struct CountAggregate;

impl CustomAggregate for CountAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#count"
    }
    fn documentation(&self) -> &str {
        "Counts the number of values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(CountState { count: 0 })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CountState {
    count: i64,
}

impl AggregateState for CountState {
    fn add(&mut self, _value: &Value) -> Result<()> {
        self.count += 1;
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::Integer(self.count))
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SumAggregate;

impl CustomAggregate for SumAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sum"
    }
    fn documentation(&self) -> &str {
        "Sums numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(SumState { sum: 0.0, count: 0 })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SumState {
    sum: f64,
    count: usize,
}

impl AggregateState for SumState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                self.sum += *i as f64;
                self.count += 1;
            }
            Value::Float(f) => {
                self.sum += f;
                self.count += 1;
            }
            _ => bail!("sum() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.count == 0 {
            Ok(Value::Integer(0))
        } else {
            Ok(Value::Float(self.sum))
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// Additional Aggregate Functions

#[derive(Debug, Clone)]
pub(crate) struct MinAggregate;

impl CustomAggregate for MinAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#min"
    }
    fn documentation(&self) -> &str {
        "Returns the minimum value"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(MinState { min: None })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MinState {
    min: Option<Value>,
}

impl AggregateState for MinState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match &self.min {
            None => self.min = Some(value.clone()),
            Some(current) => {
                if value < current {
                    self.min = Some(value.clone());
                }
            }
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        self.min
            .clone()
            .ok_or_else(|| anyhow!("No values to aggregate"))
    }

    fn reset(&mut self) {
        self.min = None;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MaxAggregate;

impl CustomAggregate for MaxAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#max"
    }
    fn documentation(&self) -> &str {
        "Returns the maximum value"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(MaxState { max: None })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MaxState {
    max: Option<Value>,
}

impl AggregateState for MaxState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match &self.max {
            None => self.max = Some(value.clone()),
            Some(current) => {
                if value > current {
                    self.max = Some(value.clone());
                }
            }
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        self.max
            .clone()
            .ok_or_else(|| anyhow!("No values to aggregate"))
    }

    fn reset(&mut self) {
        self.max = None;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AvgAggregate;

impl CustomAggregate for AvgAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#avg"
    }
    fn documentation(&self) -> &str {
        "Returns the average of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(AvgState { sum: 0.0, count: 0 })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AvgState {
    sum: f64,
    count: usize,
}

impl AggregateState for AvgState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                self.sum += *i as f64;
                self.count += 1;
            }
            Value::Float(f) => {
                self.sum += f;
                self.count += 1;
            }
            _ => bail!("avg() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.count == 0 {
            bail!("No values to average")
        } else {
            Ok(Value::Float(self.sum / self.count as f64))
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SampleAggregate;

impl CustomAggregate for SampleAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sample"
    }
    fn documentation(&self) -> &str {
        "Returns an arbitrary sample value"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(SampleState { sample: None })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SampleState {
    sample: Option<Value>,
}

impl AggregateState for SampleState {
    fn add(&mut self, value: &Value) -> Result<()> {
        if self.sample.is_none() {
            self.sample = Some(value.clone());
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        self.sample
            .clone()
            .ok_or_else(|| anyhow!("No values to sample"))
    }

    fn reset(&mut self) {
        self.sample = None;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GroupConcatAggregate;

impl CustomAggregate for GroupConcatAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#group-concat"
    }
    fn documentation(&self) -> &str {
        "Concatenates string values with separator"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(GroupConcatState {
            values: Vec::new(),
            separator: " ".to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GroupConcatState {
    values: Vec<String>,
    separator: String,
}

impl AggregateState for GroupConcatState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::String(s) => self.values.push(s.clone()),
            Value::Literal { value, .. } => self.values.push(value.clone()),
            Value::Integer(i) => self.values.push(i.to_string()),
            Value::Float(f) => self.values.push(f.to_string()),
            Value::Boolean(b) => self.values.push(b.to_string()),
            _ => bail!("group_concat() can only concatenate string-like values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::String(self.values.join(&self.separator)))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// Statistical Aggregate Functions using scirs2-stats

/// MEDIAN aggregate function - returns the median value
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct MedianAggregate;

impl CustomAggregate for MedianAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#median"
    }
    fn documentation(&self) -> &str {
        "Returns the median (50th percentile) of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(MedianState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct MedianState {
    values: Vec<f64>,
}

impl AggregateState for MedianState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("median() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute median")
        }

        // Compute median using scirs2-core
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sorted.len();
        let result = if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        };
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// STDDEV aggregate function - returns the sample standard deviation
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct StdDevAggregate;

impl CustomAggregate for StdDevAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#stddev"
    }
    fn documentation(&self) -> &str {
        "Returns the sample standard deviation of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(StdDevState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct StdDevState {
    values: Vec<f64>,
}

impl AggregateState for StdDevState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("stddev() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.len() < 2 {
            bail!("At least 2 values required for standard deviation")
        }

        // Compute sample standard deviation
        let n = self.values.len() as f64;
        let mean: f64 = self.values.iter().sum::<f64>() / n;
        let variance: f64 = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let result = variance.sqrt();
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// VARIANCE aggregate function - returns the sample variance
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VarianceAggregate;

impl CustomAggregate for VarianceAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#variance"
    }
    fn documentation(&self) -> &str {
        "Returns the sample variance of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(VarianceState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VarianceState {
    values: Vec<f64>,
}

impl AggregateState for VarianceState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("variance() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.len() < 2 {
            bail!("At least 2 values required for variance")
        }

        // Compute sample variance
        let n = self.values.len() as f64;
        let mean: f64 = self.values.iter().sum::<f64>() / n;
        let result: f64 = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// PERCENTILE aggregate function - returns the Pth percentile
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct PercentileAggregate {
    percentile: f64,
}

impl PercentileAggregate {
    #[allow(dead_code)]
    pub fn new(percentile: f64) -> Self {
        Self { percentile }
    }
}

impl CustomAggregate for PercentileAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#percentile"
    }
    fn documentation(&self) -> &str {
        "Returns the Pth percentile of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(PercentileState {
            values: Vec::new(),
            percentile: self.percentile,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct PercentileState {
    values: Vec<f64>,
    percentile: f64,
}

impl AggregateState for PercentileState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("percentile() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute percentile")
        }

        // Compute percentile
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = (self.percentile * (sorted.len() - 1) as f64).round() as usize;
        let result = sorted[index.min(sorted.len() - 1)];
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// MODE aggregate function - returns the most frequent value
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ModeAggregate;

impl CustomAggregate for ModeAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#mode"
    }
    fn documentation(&self) -> &str {
        "Returns the most frequently occurring value"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(ModeState {
            value_counts: std::collections::HashMap::new(),
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ModeState {
    value_counts: std::collections::HashMap<String, usize>,
}

impl AggregateState for ModeState {
    fn add(&mut self, value: &Value) -> Result<()> {
        let key = match value {
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Literal { value, .. } => value.clone(),
            Value::Boolean(b) => b.to_string(),
            _ => bail!("mode() cannot aggregate this value type"),
        };

        *self.value_counts.entry(key).or_insert(0) += 1;
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.value_counts.is_empty() {
            bail!("No values to compute mode")
        }

        let mode = self
            .value_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(val, _)| val.clone())
            .ok_or_else(|| anyhow!("Failed to compute mode"))?;

        // Try to parse back to appropriate type
        if let Ok(i) = mode.parse::<i64>() {
            Ok(Value::Integer(i))
        } else if let Ok(f) = mode.parse::<f64>() {
            Ok(Value::Float(f))
        } else {
            Ok(Value::String(mode))
        }
    }

    fn reset(&mut self) {
        self.value_counts.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// RANGE aggregate function - returns the range (max - min)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct RangeAggregate;

impl CustomAggregate for RangeAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#range"
    }
    fn documentation(&self) -> &str {
        "Returns the range (maximum - minimum) of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(RangeState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct RangeState {
    values: Vec<f64>,
}

impl AggregateState for RangeState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("range() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute range")
        }

        let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(Value::Float(max - min))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_aggregate() {
        let agg = MedianAggregate;
        let mut state = agg.init();

        // Add values
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(3)).unwrap();
        state.add(&Value::Integer(4)).unwrap();
        state.add(&Value::Integer(5)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!((f - 3.0).abs() < 0.001),
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_stddev_aggregate() {
        let agg = StdDevAggregate;
        let mut state = agg.init();

        // Add values with known standard deviation
        for i in 1..=10 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!(f > 0.0),
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_variance_aggregate() {
        let agg = VarianceAggregate;
        let mut state = agg.init();

        for i in 1..=10 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!(f > 0.0),
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_percentile_aggregate() {
        let agg = PercentileAggregate::new(0.75); // 75th percentile
        let mut state = agg.init();

        for i in 1..=100 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!(f > 70.0 && f < 80.0),
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_mode_aggregate() {
        let agg = ModeAggregate;
        let mut state = agg.init();

        // Most frequent value is 5
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(5)).unwrap();
        state.add(&Value::Integer(5)).unwrap();
        state.add(&Value::Integer(5)).unwrap();
        state.add(&Value::Integer(3)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Integer(i) => assert_eq!(i, 5),
            _ => panic!("Expected Integer result"),
        }
    }

    #[test]
    fn test_range_aggregate() {
        let agg = RangeAggregate;
        let mut state = agg.init();

        state.add(&Value::Integer(10)).unwrap();
        state.add(&Value::Integer(50)).unwrap();
        state.add(&Value::Integer(30)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!((f - 40.0).abs() < 0.001),
            _ => panic!("Expected Float result"),
        }
    }
}
