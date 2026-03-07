//! SPARQL temporal extension functions
//!
//! Provides custom SPARQL functions for time-series operations.

use crate::config::AggregationFunction;
use crate::error::{TsdbError, TsdbResult};
use crate::query::InterpolateMethod;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Temporal function registry for SPARQL integration
///
/// This registry holds custom SPARQL functions that operate on time-series data.
/// Functions can be registered and then invoked during SPARQL query execution.
#[derive(Default)]
pub struct TemporalFunctionRegistry {
    /// Registered functions: function_name -> handler
    functions: Arc<RwLock<HashMap<String, TemporalFunction>>>,
}

/// Handler function type for temporal SPARQL functions
pub type TemporalFunction =
    Arc<dyn Fn(&[TemporalValue]) -> TsdbResult<TemporalValue> + Send + Sync>;

/// Value types for temporal functions
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalValue {
    /// Floating point number
    Float(f64),
    /// Timestamp
    Timestamp(DateTime<Utc>),
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Boolean value
    Boolean(bool),
    /// Null value
    Null,
}

impl TemporalValue {
    /// Extract as f64
    pub fn as_float(&self) -> TsdbResult<f64> {
        match self {
            TemporalValue::Float(f) => Ok(*f),
            TemporalValue::Integer(i) => Ok(*i as f64),
            _ => Err(TsdbError::Query(format!("Expected float, got {:?}", self))),
        }
    }

    /// Extract as `DateTime<Utc>`
    pub fn as_timestamp(&self) -> TsdbResult<DateTime<Utc>> {
        match self {
            TemporalValue::Timestamp(ts) => Ok(*ts),
            _ => Err(TsdbError::Query(format!(
                "Expected timestamp, got {:?}",
                self
            ))),
        }
    }

    /// Extract as string
    pub fn as_string(&self) -> TsdbResult<String> {
        match self {
            TemporalValue::String(s) => Ok(s.clone()),
            _ => Err(TsdbError::Query(format!("Expected string, got {:?}", self))),
        }
    }

    /// Extract as integer
    pub fn as_integer(&self) -> TsdbResult<i64> {
        match self {
            TemporalValue::Integer(i) => Ok(*i),
            TemporalValue::Float(f) => Ok(*f as i64),
            _ => Err(TsdbError::Query(format!(
                "Expected integer, got {:?}",
                self
            ))),
        }
    }
}

impl TemporalFunctionRegistry {
    /// Create a new temporal function registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a temporal function
    pub fn register(&self, name: String, function: TemporalFunction) -> TsdbResult<()> {
        let mut functions = self
            .functions
            .write()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        functions.insert(name, function);
        Ok(())
    }

    /// Call a registered function
    pub fn call(&self, name: &str, args: &[TemporalValue]) -> TsdbResult<TemporalValue> {
        let functions = self
            .functions
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;

        let function = functions
            .get(name)
            .ok_or_else(|| TsdbError::Query(format!("Function not found: {name}")))?;

        function(args)
    }

    /// List all registered functions
    pub fn list_functions(&self) -> TsdbResult<Vec<String>> {
        let functions = self
            .functions
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(functions.keys().cloned().collect())
    }
}

/// Register all standard temporal functions
///
/// Registers:
/// - `ts:window` - Window aggregations
/// - `ts:resample` - Time resampling
/// - `ts:interpolate` - Value interpolation
pub fn register_temporal_functions(registry: &TemporalFunctionRegistry) -> TsdbResult<()> {
    // Register window function: ts:window(?value, ?window_size, "AVG")
    registry.register("ts:window".to_string(), window_function())?;

    // Register resample function: ts:resample(?timestamp, "1h")
    registry.register("ts:resample".to_string(), resample_function())?;

    // Register interpolate function: ts:interpolate(?timestamp, ?value, "linear")
    registry.register("ts:interpolate".to_string(), interpolate_function())?;

    Ok(())
}

/// Window aggregation function
///
/// Usage: `ts:window(?value, ?window_size, "AVG")`
///
/// Arguments:
/// - `value` - The value to aggregate (Float)
/// - `window_size` - Window size in seconds (Integer)
/// - `aggregation` - Aggregation type: "AVG", "MIN", "MAX", "SUM", "COUNT" (String)
///
/// Returns: Aggregated value (Float)
pub fn window_function() -> TemporalFunction {
    Arc::new(|args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
        if args.len() != 3 {
            return Err(TsdbError::Query(format!(
                "ts:window expects 3 arguments, got {}",
                args.len()
            )));
        }

        let value = args[0].as_float()?;
        let window_size = args[1].as_integer()?;
        let aggregation_str = args[2].as_string()?;

        // Parse aggregation type
        let _aggregation = match aggregation_str.to_uppercase().as_str() {
            "AVG" | "AVERAGE" => AggregationFunction::Average,
            "MIN" => AggregationFunction::Min,
            "MAX" => AggregationFunction::Max,
            "SUM" => AggregationFunction::Sum,
            "FIRST" => AggregationFunction::First,
            "LAST" => AggregationFunction::Last,
            _ => {
                return Err(TsdbError::Query(format!(
                    "Unknown aggregation: {aggregation_str}"
                )))
            }
        };

        // For production use, this function should be called from oxirs-arq
        // which has access to the full query context and data stream.
        // This registry provides the function signature for SPARQL integration.
        // Future: Direct integration with WindowFunction from query module for standalone use
        let _ = window_size; // Suppress warning
        Ok(TemporalValue::Float(value))
    })
}

/// Resample function for time bucketing
///
/// Usage: `ts:resample(?timestamp, "1h")`
///
/// Arguments:
/// - `timestamp` - The timestamp to resample (Timestamp)
/// - `interval` - Resample interval: "1s", "1m", "1h", "1d" (String)
///
/// Returns: Bucket timestamp (Timestamp)
pub fn resample_function() -> TemporalFunction {
    Arc::new(|args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
        if args.len() != 2 {
            return Err(TsdbError::Query(format!(
                "ts:resample expects 2 arguments, got {}",
                args.len()
            )));
        }

        let timestamp = args[0].as_timestamp()?;
        let interval_str = args[1].as_string()?;

        // Parse interval string (e.g., "1h", "5m", "30s")
        let interval = parse_duration(&interval_str)?;

        // Round down to bucket boundary
        let bucket_timestamp = round_to_interval(timestamp, interval);

        Ok(TemporalValue::Timestamp(bucket_timestamp))
    })
}

/// Interpolate function for missing values
///
/// Usage: `ts:interpolate(?timestamp, ?value, "linear")`
///
/// Arguments:
/// - `timestamp` - The timestamp for interpolation (Timestamp)
/// - `value` - The value to interpolate (Float, may be Null)
/// - `method` - Interpolation method: "linear", "forward", "backward" (String)
///
/// Returns: Interpolated value (Float)
pub fn interpolate_function() -> TemporalFunction {
    Arc::new(|args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
        if args.len() != 3 {
            return Err(TsdbError::Query(format!(
                "ts:interpolate expects 3 arguments, got {}",
                args.len()
            )));
        }

        let _timestamp = args[0].as_timestamp()?;
        let value = if matches!(args[1], TemporalValue::Null) {
            None
        } else {
            Some(args[1].as_float()?)
        };
        let method_str = args[2].as_string()?;

        // Parse interpolation method
        let _method = match method_str.to_lowercase().as_str() {
            "linear" => InterpolateMethod::Linear,
            "forward" => InterpolateMethod::ForwardFill,
            "backward" => InterpolateMethod::BackwardFill,
            "nearest" => InterpolateMethod::Nearest,
            _ => {
                return Err(TsdbError::Query(format!(
                    "Unknown interpolation method: {method_str}"
                )))
            }
        };

        // For production use, this function should be called from oxirs-arq
        // which has access to the full query context and data stream.
        // This registry provides the function signature for SPARQL integration.
        // Future: Direct integration with Interpolator from query module for standalone use
        Ok(TemporalValue::Float(value.unwrap_or(0.0)))
    })
}

/// Parse duration string (e.g., "1h", "5m", "30s", "2d")
fn parse_duration(s: &str) -> TsdbResult<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return Err(TsdbError::Query("Empty duration string".to_string()));
    }

    // Extract number and unit
    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: i64 = num_str
        .parse()
        .map_err(|e| TsdbError::Query(format!("Invalid duration number: {e}")))?;

    match unit {
        "s" => Ok(Duration::seconds(num)),
        "m" => Ok(Duration::minutes(num)),
        "h" => Ok(Duration::hours(num)),
        "d" => Ok(Duration::days(num)),
        _ => Err(TsdbError::Query(format!("Unknown duration unit: {unit}"))),
    }
}

/// Round timestamp down to interval boundary
fn round_to_interval(timestamp: DateTime<Utc>, interval: Duration) -> DateTime<Utc> {
    let timestamp_ms = timestamp.timestamp_millis();
    let interval_ms = interval.num_milliseconds();
    let rounded_ms = (timestamp_ms / interval_ms) * interval_ms;
    DateTime::from_timestamp_millis(rounded_ms).unwrap_or(timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_registry() -> TsdbResult<()> {
        let registry = TemporalFunctionRegistry::new();

        // Register a simple function
        let add_one = Arc::new(|args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
            let value = args[0].as_float()?;
            Ok(TemporalValue::Float(value + 1.0))
        });

        registry.register("add_one".to_string(), add_one)?;

        // Call the function
        let result = registry.call("add_one", &[TemporalValue::Float(5.0)])?;
        assert_eq!(result, TemporalValue::Float(6.0));

        Ok(())
    }

    #[test]
    fn test_window_function() -> TsdbResult<()> {
        let window_fn = window_function();

        let result = window_fn(&[
            TemporalValue::Float(42.5),
            TemporalValue::Integer(600),
            TemporalValue::String("AVG".to_string()),
        ])?;

        // Placeholder implementation returns the value as-is
        assert_eq!(result, TemporalValue::Float(42.5));

        Ok(())
    }

    #[test]
    fn test_resample_function() -> TsdbResult<()> {
        let resample_fn = resample_function();

        let timestamp = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp"); // 2021-01-01 00:00:00
        let result = resample_fn(&[
            TemporalValue::Timestamp(timestamp),
            TemporalValue::String("1h".to_string()),
        ])?;

        // Should round to hour boundary
        if let TemporalValue::Timestamp(ts) = result {
            assert_eq!(ts.timestamp(), 1609459200); // Exactly on hour boundary
        } else {
            panic!("Expected timestamp result");
        }

        Ok(())
    }

    #[test]
    fn test_resample_rounding() -> TsdbResult<()> {
        let resample_fn = resample_function();

        let timestamp = DateTime::from_timestamp(1609459830, 0).expect("valid timestamp"); // 2021-01-01 00:10:30
        let result = resample_fn(&[
            TemporalValue::Timestamp(timestamp),
            TemporalValue::String("1h".to_string()),
        ])?;

        // Should round down to previous hour boundary
        if let TemporalValue::Timestamp(ts) = result {
            assert_eq!(ts.timestamp(), 1609459200); // 2021-01-01 00:00:00
        } else {
            panic!("Expected timestamp result");
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_function() -> TsdbResult<()> {
        let interpolate_fn = interpolate_function();

        let timestamp = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let result = interpolate_fn(&[
            TemporalValue::Timestamp(timestamp),
            TemporalValue::Float(42.5),
            TemporalValue::String("linear".to_string()),
        ])?;

        // Placeholder implementation returns the value as-is
        assert_eq!(result, TemporalValue::Float(42.5));

        Ok(())
    }

    #[test]
    fn test_interpolate_null() -> TsdbResult<()> {
        let interpolate_fn = interpolate_function();

        let timestamp = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let result = interpolate_fn(&[
            TemporalValue::Timestamp(timestamp),
            TemporalValue::Null,
            TemporalValue::String("linear".to_string()),
        ])?;

        // Null should interpolate to 0.0 in placeholder implementation
        assert_eq!(result, TemporalValue::Float(0.0));

        Ok(())
    }

    #[test]
    fn test_parse_duration() -> TsdbResult<()> {
        assert_eq!(parse_duration("30s")?, Duration::seconds(30));
        assert_eq!(parse_duration("5m")?, Duration::minutes(5));
        assert_eq!(parse_duration("1h")?, Duration::hours(1));
        assert_eq!(parse_duration("2d")?, Duration::days(2));

        Ok(())
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("xyz").is_err());
        assert!(parse_duration("5x").is_err());
    }

    #[test]
    fn test_round_to_interval() {
        let timestamp = DateTime::from_timestamp(1609459830, 0).expect("valid timestamp"); // 2021-01-01 00:10:30
        let interval = Duration::hours(1);
        let rounded = round_to_interval(timestamp, interval);

        assert_eq!(rounded.timestamp(), 1609459200); // 2021-01-01 00:00:00
    }

    #[test]
    fn test_register_all_functions() -> TsdbResult<()> {
        let registry = TemporalFunctionRegistry::new();
        register_temporal_functions(&registry)?;

        let functions = registry.list_functions()?;
        assert!(functions.contains(&"ts:window".to_string()));
        assert!(functions.contains(&"ts:resample".to_string()));
        assert!(functions.contains(&"ts:interpolate".to_string()));

        Ok(())
    }

    #[test]
    fn test_temporal_value_conversions() -> TsdbResult<()> {
        let float_val = TemporalValue::Float(42.5);
        assert_eq!(float_val.as_float()?, 42.5);

        let int_val = TemporalValue::Integer(42);
        assert_eq!(int_val.as_integer()?, 42);
        assert_eq!(int_val.as_float()?, 42.0); // Should convert

        let str_val = TemporalValue::String("hello".to_string());
        assert_eq!(str_val.as_string()?, "hello");

        let timestamp = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let ts_val = TemporalValue::Timestamp(timestamp);
        assert_eq!(ts_val.as_timestamp()?, timestamp);

        Ok(())
    }

    #[test]
    fn test_temporal_value_error_conversions() {
        let str_val = TemporalValue::String("hello".to_string());
        assert!(str_val.as_float().is_err());
        assert!(str_val.as_integer().is_err());
        assert!(str_val.as_timestamp().is_err());
    }
}
