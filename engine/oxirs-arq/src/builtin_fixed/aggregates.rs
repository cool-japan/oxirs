//! Built-in SPARQL Functions - Aggregate Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation, plus additional
//! statistical aggregates using scirs2-stats.

use crate::extensions::{AggregateState, CustomAggregate, Value};
use anyhow::{anyhow, bail, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_stats::{kurtosis, median, skew, std};

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

        // Use scirs2-stats for median calculation
        let arr = Array1::from_vec(self.values.clone());
        let result = median(&arr.view()).map_err(|e| anyhow!("Failed to compute median: {}", e))?;
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

        // Use scirs2-stats for standard deviation (sample std, ddof=1)
        let arr = Array1::from_vec(self.values.clone());
        let result = std(&arr.view(), 1, None)
            .map_err(|e| anyhow!("Failed to compute standard deviation: {}", e))?;
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

        // Use scirs2-stats std and square it to get variance (sample variance, ddof=1)
        let arr = Array1::from_vec(self.values.clone());
        let std_val = std(&arr.view(), 1, None)
            .map_err(|e| anyhow!("Failed to compute standard deviation: {}", e))?;
        let result = std_val * std_val;
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

/// DISTINCT_COUNT aggregate function - counts distinct values
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct DistinctCountAggregate;

impl CustomAggregate for DistinctCountAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#distinct-count"
    }
    fn documentation(&self) -> &str {
        "Counts the number of distinct values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(DistinctCountState {
            unique_values: std::collections::HashSet::new(),
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct DistinctCountState {
    unique_values: std::collections::HashSet<String>,
}

impl AggregateState for DistinctCountState {
    fn add(&mut self, value: &Value) -> Result<()> {
        let key = match value {
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Literal { value, .. } => value.clone(),
            Value::Boolean(b) => b.to_string(),
            _ => bail!("distinct_count() cannot process this value type"),
        };

        self.unique_values.insert(key);
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::Integer(self.unique_values.len() as i64))
    }

    fn reset(&mut self) {
        self.unique_values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// PRODUCT aggregate function - returns the product of numeric values
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ProductAggregate;

impl CustomAggregate for ProductAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#product"
    }
    fn documentation(&self) -> &str {
        "Returns the product of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(ProductState {
            product: 1.0,
            count: 0,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ProductState {
    product: f64,
    count: usize,
}

impl AggregateState for ProductState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                self.product *= *i as f64;
                self.count += 1;
            }
            Value::Float(f) => {
                self.product *= f;
                self.count += 1;
            }
            _ => bail!("product() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.count == 0 {
            Ok(Value::Integer(1))
        } else {
            Ok(Value::Float(self.product))
        }
    }

    fn reset(&mut self) {
        self.product = 1.0;
        self.count = 0;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// GEOMETRIC_MEAN aggregate function - returns the geometric mean
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct GeometricMeanAggregate;

impl CustomAggregate for GeometricMeanAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#geometric-mean"
    }
    fn documentation(&self) -> &str {
        "Returns the geometric mean of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(GeometricMeanState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct GeometricMeanState {
    values: Vec<f64>,
}

impl AggregateState for GeometricMeanState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                if *i <= 0 {
                    bail!("geometric_mean() requires positive values");
                }
                self.values.push(*i as f64);
            }
            Value::Float(f) => {
                if *f <= 0.0 {
                    bail!("geometric_mean() requires positive values");
                }
                self.values.push(*f);
            }
            _ => bail!("geometric_mean() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute geometric mean")
        }

        // Compute geometric mean: (product of values)^(1/n)
        let n = self.values.len() as f64;
        let log_sum: f64 = self.values.iter().map(|x| x.ln()).sum();
        let result = (log_sum / n).exp();
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// HARMONIC_MEAN aggregate function - returns the harmonic mean
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct HarmonicMeanAggregate;

impl CustomAggregate for HarmonicMeanAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#harmonic-mean"
    }
    fn documentation(&self) -> &str {
        "Returns the harmonic mean of numeric values"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(HarmonicMeanState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct HarmonicMeanState {
    values: Vec<f64>,
}

impl AggregateState for HarmonicMeanState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                if *i == 0 {
                    bail!("harmonic_mean() cannot accept zero values");
                }
                self.values.push(*i as f64);
            }
            Value::Float(f) => {
                if f.abs() < f64::EPSILON {
                    bail!("harmonic_mean() cannot accept zero values");
                }
                self.values.push(*f);
            }
            _ => bail!("harmonic_mean() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute harmonic mean")
        }

        // Compute harmonic mean: n / (sum of 1/x)
        let n = self.values.len() as f64;
        let reciprocal_sum: f64 = self.values.iter().map(|x| 1.0 / x).sum();
        let result = n / reciprocal_sum;
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// SKEWNESS aggregate function - returns the skewness (third standardized moment)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct SkewnessAggregate;

impl CustomAggregate for SkewnessAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#skewness"
    }
    fn documentation(&self) -> &str {
        "Returns the skewness (asymmetry) of the distribution"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(SkewnessState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct SkewnessState {
    values: Vec<f64>,
}

impl AggregateState for SkewnessState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("skewness() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.len() < 3 {
            bail!("At least 3 values required for skewness")
        }

        // Use scirs2-stats for skewness calculation (bias=false for sample skewness)
        let arr = Array1::from_vec(self.values.clone());
        let result = skew(&arr.view(), false, None)
            .map_err(|e| anyhow!("Failed to compute skewness: {}", e))?;
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// KURTOSIS aggregate function - returns the kurtosis (fourth standardized moment)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct KurtosisAggregate;

impl CustomAggregate for KurtosisAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#kurtosis"
    }
    fn documentation(&self) -> &str {
        "Returns the kurtosis (tailedness) of the distribution"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(KurtosisState { values: Vec::new() })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct KurtosisState {
    values: Vec<f64>,
}

impl AggregateState for KurtosisState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("kurtosis() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.len() < 4 {
            bail!("At least 4 values required for kurtosis")
        }

        // Use scirs2-stats for kurtosis calculation (fisher=true for excess kurtosis, bias=false for sample)
        let arr = Array1::from_vec(self.values.clone());
        let result = kurtosis(&arr.view(), true, false, None)
            .map_err(|e| anyhow!("Failed to compute kurtosis: {}", e))?;
        Ok(Value::Float(result))
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

/// QUANTILE aggregate function - returns the quantile at a specified probability
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct QuantileAggregate {
    quantile: f64,
}

impl QuantileAggregate {
    #[allow(dead_code)]
    pub fn new(quantile: f64) -> Self {
        Self { quantile }
    }
}

impl CustomAggregate for QuantileAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#quantile"
    }
    fn documentation(&self) -> &str {
        "Returns the quantile at a specified probability (0.0 to 1.0)"
    }

    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(QuantileState {
            values: Vec::new(),
            quantile: self.quantile,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct QuantileState {
    values: Vec<f64>,
    quantile: f64,
}

impl AggregateState for QuantileState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => self.values.push(*i as f64),
            Value::Float(f) => self.values.push(*f),
            _ => bail!("quantile() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.values.is_empty() {
            bail!("No values to compute quantile")
        }

        if !(0.0..=1.0).contains(&self.quantile) {
            bail!("Quantile must be between 0.0 and 1.0")
        }

        // Compute quantile using linear interpolation
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let index = self.quantile * (n - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        let result = if lower_idx == upper_idx {
            sorted[lower_idx]
        } else {
            let weight = index - lower_idx as f64;
            sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
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

    #[test]
    fn test_distinct_count_aggregate() {
        let agg = DistinctCountAggregate;
        let mut state = agg.init();

        // Add duplicate values
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(3)).unwrap();
        state.add(&Value::Integer(2)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Integer(i) => assert_eq!(i, 3), // Only 1, 2, 3 are distinct
            _ => panic!("Expected Integer result"),
        }
    }

    #[test]
    fn test_product_aggregate() {
        let agg = ProductAggregate;
        let mut state = agg.init();

        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(3)).unwrap();
        state.add(&Value::Integer(4)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!((f - 24.0).abs() < 0.001), // 2 * 3 * 4 = 24
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_geometric_mean_aggregate() {
        let agg = GeometricMeanAggregate;
        let mut state = agg.init();

        // Geometric mean of 1, 2, 4, 8 is (1*2*4*8)^(1/4) = 2^(10/4) = 2^2.5 ≈ 2.828
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(4)).unwrap();
        state.add(&Value::Integer(8)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => {
                let expected = (1.0 * 2.0 * 4.0 * 8.0_f64).powf(0.25);
                assert!((f - expected).abs() < 0.001);
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_harmonic_mean_aggregate() {
        let agg = HarmonicMeanAggregate;
        let mut state = agg.init();

        // Harmonic mean of 1, 2, 4 is 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 ≈ 1.714
        state.add(&Value::Integer(1)).unwrap();
        state.add(&Value::Integer(2)).unwrap();
        state.add(&Value::Integer(4)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => {
                let expected = 3.0 / (1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0);
                assert!((f - expected).abs() < 0.001);
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_skewness_aggregate() {
        let agg = SkewnessAggregate;
        let mut state = agg.init();

        // Add values with known distribution
        for i in 1..=10 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(_f) => {
                // For uniform distribution, skewness should be close to 0
                // Just verify we got a result
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_kurtosis_aggregate() {
        let agg = KurtosisAggregate;
        let mut state = agg.init();

        // Add values
        for i in 1..=10 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(_f) => {
                // For uniform distribution, excess kurtosis should be negative
                // Just verify we got a result
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_quantile_aggregate() {
        let agg = QuantileAggregate::new(0.5); // Median
        let mut state = agg.init();

        for i in 1..=10 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => {
                // 0.5 quantile (median) of 1..10 should be 5.5
                assert!((f - 5.5).abs() < 0.001);
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_quantile_aggregate_first_quartile() {
        let agg = QuantileAggregate::new(0.25); // First quartile
        let mut state = agg.init();

        for i in 1..=100 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => {
                // 0.25 quantile should be around 25.75
                assert!(f > 24.0 && f < 27.0);
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_quantile_aggregate_third_quartile() {
        let agg = QuantileAggregate::new(0.75); // Third quartile
        let mut state = agg.init();

        for i in 1..=100 {
            state.add(&Value::Integer(i)).unwrap();
        }

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => {
                // 0.75 quantile should be around 75.25
                assert!(f > 74.0 && f < 77.0);
            }
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_enhanced_median_with_scirs2() {
        // Test that the enhanced median using scirs2-stats works correctly
        let agg = MedianAggregate;
        let mut state = agg.init();

        // Even number of values: median of [1.5, 2.5, 3.5, 4.5] should be (2.5 + 3.5) / 2 = 3.0
        state.add(&Value::Float(1.5)).unwrap();
        state.add(&Value::Float(2.5)).unwrap();
        state.add(&Value::Float(3.5)).unwrap();
        state.add(&Value::Float(4.5)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!((f - 3.0).abs() < 0.01), // Median should be 3.0
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_enhanced_stddev_with_scirs2() {
        // Test that the enhanced stddev using scirs2-stats works correctly
        let agg = StdDevAggregate;
        let mut state = agg.init();

        // Known values
        state.add(&Value::Float(2.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(5.0)).unwrap();
        state.add(&Value::Float(5.0)).unwrap();
        state.add(&Value::Float(7.0)).unwrap();
        state.add(&Value::Float(9.0)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!(f > 0.0 && f < 5.0), // Sample stddev should be reasonable
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_enhanced_variance_with_scirs2() {
        // Test that the enhanced variance using scirs2-stats works correctly
        let agg = VarianceAggregate;
        let mut state = agg.init();

        state.add(&Value::Float(2.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(4.0)).unwrap();
        state.add(&Value::Float(5.0)).unwrap();
        state.add(&Value::Float(5.0)).unwrap();
        state.add(&Value::Float(7.0)).unwrap();
        state.add(&Value::Float(9.0)).unwrap();

        let result = state.result().unwrap();
        match result {
            Value::Float(f) => assert!(f > 0.0), // Variance should be positive
            _ => panic!("Expected Float result"),
        }
    }
}
