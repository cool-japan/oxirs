//! Built-in SPARQL Functions - Aggregate Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{
    AggregateState, CustomAggregate, Value,
};
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
