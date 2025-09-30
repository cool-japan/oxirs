//! Aggregation functions and state management for event processing
//!
//! This module provides aggregation capabilities including:
//! - Basic aggregations (count, sum, average, min, max)
//! - Complex aggregations (distinct, custom expressions)
//! - Aggregation state management

use crate::StreamEvent;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Aggregation functions for window processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum { field: String },
    Average { field: String },
    Min { field: String },
    Max { field: String },
    First,
    Last,
    Distinct { field: String },
    Custom { name: String, expression: String },
}

/// Aggregation state for maintaining running calculations
#[derive(Debug, Clone)]
pub enum AggregationState {
    Count(u64),
    Sum(f64),
    Average { sum: f64, count: u64 },
    Min(f64),
    Max(f64),
    First(StreamEvent),
    Last(StreamEvent),
    Distinct(HashSet<String>),
}

impl AggregationState {
    /// Create new aggregation state for a function
    pub fn new(function: &AggregateFunction) -> Self {
        match function {
            AggregateFunction::Count => AggregationState::Count(0),
            AggregateFunction::Sum { .. } => AggregationState::Sum(0.0),
            AggregateFunction::Average { .. } => AggregationState::Average { sum: 0.0, count: 0 },
            AggregateFunction::Min { .. } => AggregationState::Min(f64::INFINITY),
            AggregateFunction::Max { .. } => AggregationState::Max(f64::NEG_INFINITY),
            AggregateFunction::First => AggregationState::First(StreamEvent::TripleAdded {
                subject: String::new(),
                predicate: String::new(),
                object: String::new(),
                graph: None,
                metadata: crate::event::EventMetadata::default(),
            }),
            AggregateFunction::Last => AggregationState::Last(StreamEvent::TripleAdded {
                subject: String::new(),
                predicate: String::new(),
                object: String::new(),
                graph: None,
                metadata: crate::event::EventMetadata::default(),
            }),
            AggregateFunction::Distinct { .. } => AggregationState::Distinct(HashSet::new()),
            AggregateFunction::Custom { .. } => AggregationState::Count(0), // Default for custom
        }
    }

    /// Update aggregation state with new event
    pub fn update(&mut self, event: &StreamEvent, function: &AggregateFunction) -> Result<()> {
        match (self, function) {
            (AggregationState::Count(count), AggregateFunction::Count) => {
                *count += 1;
            }
            (AggregationState::Sum(sum), AggregateFunction::Sum { field }) => {
                if let Some(value) = extract_numeric_field(event, field)? {
                    *sum += value;
                }
            }
            (AggregationState::Average { sum, count }, AggregateFunction::Average { field }) => {
                if let Some(value) = extract_numeric_field(event, field)? {
                    *sum += value;
                    *count += 1;
                }
            }
            (AggregationState::Min(min), AggregateFunction::Min { field }) => {
                if let Some(value) = extract_numeric_field(event, field)? {
                    *min = min.min(value);
                }
            }
            (AggregationState::Max(max), AggregateFunction::Max { field }) => {
                if let Some(value) = extract_numeric_field(event, field)? {
                    *max = max.max(value);
                }
            }
            (AggregationState::First(first), AggregateFunction::First) => {
                *first = event.clone();
            }
            (AggregationState::Last(last), AggregateFunction::Last) => {
                *last = event.clone();
            }
            (AggregationState::Distinct(set), AggregateFunction::Distinct { field }) => {
                if let Some(value) = extract_string_field(event, field)? {
                    set.insert(value);
                }
            }
            (AggregationState::Count(count), AggregateFunction::Custom { expression, .. }) => {
                // Custom aggregation evaluation
                if let Some(result) = evaluate_custom_expression(expression, event)? {
                    *count += result as u64;
                }
            }
            _ => return Err(anyhow!("Mismatched aggregation state and function")),
        }
        Ok(())
    }

    /// Get the current aggregation result
    pub fn result(&self) -> Result<serde_json::Value> {
        match self {
            AggregationState::Count(count) => Ok(serde_json::Value::Number((*count).into())),
            AggregationState::Sum(sum) => Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(*sum).unwrap_or(0.into()),
            )),
            AggregationState::Average { sum, count } => {
                if *count > 0 {
                    let avg = *sum / (*count as f64);
                    Ok(serde_json::Value::Number(
                        serde_json::Number::from_f64(avg).unwrap_or(0.into()),
                    ))
                } else {
                    Ok(serde_json::Value::Number(0.into()))
                }
            }
            AggregationState::Min(min) => {
                if min.is_finite() {
                    Ok(serde_json::Value::Number(
                        serde_json::Number::from_f64(*min).unwrap_or(0.into()),
                    ))
                } else {
                    Ok(serde_json::Value::Null)
                }
            }
            AggregationState::Max(max) => {
                if max.is_finite() {
                    Ok(serde_json::Value::Number(
                        serde_json::Number::from_f64(*max).unwrap_or(0.into()),
                    ))
                } else {
                    Ok(serde_json::Value::Null)
                }
            }
            AggregationState::First(event) => Ok(serde_json::to_value(event)?),
            AggregationState::Last(event) => Ok(serde_json::to_value(event)?),
            AggregationState::Distinct(set) => Ok(serde_json::Value::Number(set.len().into())),
        }
    }
}

/// Extract numeric field from event
fn extract_numeric_field(event: &StreamEvent, _field: &str) -> Result<Option<f64>> {
    // Implementation would depend on StreamEvent structure
    // This is a simplified version for the actual StreamEvent variants
    match event {
        StreamEvent::SparqlUpdate { .. } => Ok(None),
        StreamEvent::TripleAdded { .. } => Ok(None),
        StreamEvent::TripleRemoved { .. } => Ok(None),
        StreamEvent::QuadAdded { .. } => Ok(None),
        StreamEvent::QuadRemoved { .. } => Ok(None),
        StreamEvent::GraphCreated { .. } => Ok(None),
        StreamEvent::GraphCleared { .. } => Ok(None),
        StreamEvent::GraphDeleted { .. } => Ok(None),
        StreamEvent::TransactionBegin { .. } => Ok(None),
        StreamEvent::TransactionCommit { .. } => Ok(None),
        StreamEvent::TransactionAbort { .. } => Ok(None),
        _ => Ok(None),
    }
}

/// Extract string field from event
fn extract_string_field(event: &StreamEvent, field: &str) -> Result<Option<String>> {
    // Implementation would depend on StreamEvent structure
    // This is a simplified version for the actual StreamEvent variants
    match event {
        StreamEvent::TripleAdded {
            subject,
            predicate,
            object,
            ..
        } => match field {
            "subject" => Ok(Some(subject.clone())),
            "predicate" => Ok(Some(predicate.clone())),
            "object" => Ok(Some(object.clone())),
            _ => Ok(None),
        },
        StreamEvent::TripleRemoved {
            subject,
            predicate,
            object,
            ..
        } => match field {
            "subject" => Ok(Some(subject.clone())),
            "predicate" => Ok(Some(predicate.clone())),
            "object" => Ok(Some(object.clone())),
            _ => Ok(None),
        },
        StreamEvent::QuadAdded {
            subject,
            predicate,
            object,
            graph,
            ..
        } => match field {
            "subject" => Ok(Some(subject.clone())),
            "predicate" => Ok(Some(predicate.clone())),
            "object" => Ok(Some(object.clone())),
            "graph" => Ok(Some(graph.clone())),
            _ => Ok(None),
        },
        StreamEvent::QuadRemoved {
            subject,
            predicate,
            object,
            graph,
            ..
        } => match field {
            "subject" => Ok(Some(subject.clone())),
            "predicate" => Ok(Some(predicate.clone())),
            "object" => Ok(Some(object.clone())),
            "graph" => Ok(Some(graph.clone())),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}

/// Evaluate custom expression
fn evaluate_custom_expression(expression: &str, event: &StreamEvent) -> Result<Option<f64>> {
    // Parse and evaluate custom expressions
    // This is a simplified implementation
    match expression {
        expr if expr.starts_with("field:") => {
            let field = expr.strip_prefix("field:").unwrap();
            extract_numeric_field(event, field)
        }
        expr if expr.starts_with("const:") => {
            let value = expr.strip_prefix("const:").unwrap();
            match value.parse::<f64>() {
                Ok(n) => Ok(Some(n)),
                Err(_) => Ok(None),
            }
        }
        expr if expr.contains('+') => {
            let parts: Vec<&str> = expr.split('+').collect();
            if parts.len() == 2 {
                let left = evaluate_custom_expression(parts[0].trim(), event)?;
                let right = evaluate_custom_expression(parts[1].trim(), event)?;
                match (left, right) {
                    (Some(l), Some(r)) => Ok(Some(l + r)),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        expr if expr.contains('*') => {
            let parts: Vec<&str> = expr.split('*').collect();
            if parts.len() == 2 {
                let left = evaluate_custom_expression(parts[0].trim(), event)?;
                let right = evaluate_custom_expression(parts[1].trim(), event)?;
                match (left, right) {
                    (Some(l), Some(r)) => Ok(Some(l * r)),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Aggregation manager for handling multiple aggregations
pub struct AggregationManager {
    aggregations: HashMap<String, (AggregateFunction, AggregationState)>,
}

impl AggregationManager {
    /// Create new aggregation manager
    pub fn new() -> Self {
        Self {
            aggregations: HashMap::new(),
        }
    }

    /// Add aggregation function
    pub fn add_aggregation(&mut self, name: String, function: AggregateFunction) {
        let state = AggregationState::new(&function);
        self.aggregations.insert(name, (function, state));
    }

    /// Update all aggregations with new event
    pub fn update(&mut self, event: &StreamEvent) -> Result<()> {
        for (_, (function, state)) in self.aggregations.iter_mut() {
            state.update(event, function)?;
        }
        Ok(())
    }

    /// Get all aggregation results
    pub fn results(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut results = HashMap::new();
        for (name, (_, state)) in &self.aggregations {
            results.insert(name.clone(), state.result()?);
        }
        Ok(results)
    }
}

impl Default for AggregationManager {
    fn default() -> Self {
        Self::new()
    }
}
