//! SPARQL temporal extension functions
//!
//! Provides custom SPARQL functions for time-series operations.

use crate::config::AggregationFunction;
use crate::error::{TsdbError, TsdbResult};
use crate::query::aggregate::{Aggregation, Aggregator};
use crate::query::interpolate::Interpolator;
use crate::query::InterpolateMethod;
use crate::series::DataPoint;
use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

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

/// Map a parsed [`AggregationFunction`] onto the query engine's
/// [`Aggregation`] type used by [`Aggregator`].
fn to_aggregation(function: AggregationFunction) -> Aggregation {
    match function {
        AggregationFunction::Average => Aggregation::Avg,
        AggregationFunction::Min => Aggregation::Min,
        AggregationFunction::Max => Aggregation::Max,
        AggregationFunction::Sum => Aggregation::Sum,
        AggregationFunction::First => Aggregation::First,
        AggregationFunction::Last => Aggregation::Last,
    }
}

/// Window aggregation function
///
/// Usage: `ts:window(?value, ?window_size, "AVG")`
///
/// Arguments:
/// - `value` - The value to aggregate (Float)
/// - `window_size` - Number of most recent values to aggregate over (Integer)
/// - `aggregation` - Aggregation type: "AVG", "MIN", "MAX", "SUM", "FIRST", "LAST" (String)
///
/// Returns: Aggregated value (Float)
///
/// This maintains a real sliding window of the last `window_size` values
/// seen by *this registered function instance* (SPARQL evaluates rows in
/// order, so successive calls form a stream) and computes the requested
/// aggregation over that window using the crate's [`Aggregator`] -- it does
/// not simply echo the input value back unchanged.
pub fn window_function() -> TemporalFunction {
    let window: Arc<Mutex<VecDeque<f64>>> = Arc::new(Mutex::new(VecDeque::new()));

    Arc::new(move |args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
        if args.len() != 3 {
            return Err(TsdbError::Query(format!(
                "ts:window expects 3 arguments, got {}",
                args.len()
            )));
        }

        let value = args[0].as_float()?;
        let window_size = args[1].as_integer()?;
        let aggregation_str = args[2].as_string()?;

        if window_size <= 0 {
            return Err(TsdbError::Query(format!(
                "ts:window window_size must be positive, got {window_size}"
            )));
        }

        // Parse aggregation type
        let aggregation = match aggregation_str.to_uppercase().as_str() {
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

        let mut buf = window
            .lock()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        buf.push_back(value);
        while buf.len() > window_size as usize {
            buf.pop_front();
        }

        // Aggregate the current window using the crate's real aggregation
        // machinery instead of returning the raw input value.
        let now = Utc::now();
        let mut aggregator = Aggregator::new();
        for (i, v) in buf.iter().enumerate() {
            aggregator.add(&DataPoint::new(now + Duration::milliseconds(i as i64), *v));
        }
        let result = aggregator.result(to_aggregation(aggregation))?;

        Ok(TemporalValue::Float(result))
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

/// Maximum number of observed (non-null) points retained per registered
/// `ts:interpolate` function instance to interpolate future null values
/// against.
const INTERPOLATE_HISTORY_CAPACITY: usize = 256;

/// Interpolate function for missing values
///
/// Usage: `ts:interpolate(?timestamp, ?value, "linear")`
///
/// Arguments:
/// - `timestamp` - The timestamp for interpolation (Timestamp)
/// - `value` - The value to interpolate (Float, may be Null)
/// - `method` - Interpolation method: "linear", "forward", "backward", "nearest" (String)
///
/// Returns: Interpolated value (Float)
///
/// SPARQL evaluates rows in order, so this registered function instance
/// keeps a bounded history of observed (non-null) `(timestamp, value)`
/// points. A non-null call records its point and is returned unchanged. A
/// null call is interpolated against that history using the crate's real
/// [`Interpolator`] for the requested method -- it no longer collapses to a
/// hard-coded `0.0`. If no observations have been seen yet (or the method
/// cannot be satisfied, e.g. `backward` with no later observation ever
/// recorded), this returns an explicit error rather than fabricating data.
pub fn interpolate_function() -> TemporalFunction {
    let history: Arc<Mutex<VecDeque<DataPoint>>> = Arc::new(Mutex::new(VecDeque::new()));

    Arc::new(move |args: &[TemporalValue]| -> TsdbResult<TemporalValue> {
        if args.len() != 3 {
            return Err(TsdbError::Query(format!(
                "ts:interpolate expects 3 arguments, got {}",
                args.len()
            )));
        }

        let timestamp = args[0].as_timestamp()?;
        let value = if matches!(args[1], TemporalValue::Null) {
            None
        } else {
            Some(args[1].as_float()?)
        };
        let method_str = args[2].as_string()?;

        // Parse interpolation method
        let method = match method_str.to_lowercase().as_str() {
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

        let mut hist = history
            .lock()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;

        if let Some(v) = value {
            let point = DataPoint::new(timestamp, v);
            let insert_at = hist
                .iter()
                .position(|p| p.timestamp > timestamp)
                .unwrap_or(hist.len());
            hist.insert(insert_at, point);
            if hist.len() > INTERPOLATE_HISTORY_CAPACITY {
                hist.pop_front();
            }
            return Ok(TemporalValue::Float(v));
        }

        if hist.is_empty() {
            return Err(TsdbError::Query(
                "ts:interpolate: no observed values yet to interpolate from".to_string(),
            ));
        }

        let points: Vec<DataPoint> = hist.iter().copied().collect();
        let interpolated = Interpolator::new(method).interpolate_at(timestamp, &points)?;
        Ok(TemporalValue::Float(interpolated))
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

        // A single call has a window of one value, so AVG-of-one == itself.
        assert_eq!(result, TemporalValue::Float(42.5));

        Ok(())
    }

    /// Regression test: `ts:window` must maintain a real sliding window and
    /// actually aggregate, not just echo the latest input back. See P1 stub
    /// finding on `sparql/extensions.rs`.
    #[test]
    fn test_window_function_sliding_average() -> TsdbResult<()> {
        let window_fn = window_function();

        window_fn(&[
            TemporalValue::Float(10.0),
            TemporalValue::Integer(2),
            TemporalValue::String("AVG".to_string()),
        ])?;
        window_fn(&[
            TemporalValue::Float(20.0),
            TemporalValue::Integer(2),
            TemporalValue::String("AVG".to_string()),
        ])?;
        // Window size 2: after the 3rd call the window holds [20.0, 30.0].
        let result = window_fn(&[
            TemporalValue::Float(30.0),
            TemporalValue::Integer(2),
            TemporalValue::String("AVG".to_string()),
        ])?;

        assert_eq!(result, TemporalValue::Float(25.0));
        Ok(())
    }

    #[test]
    fn test_window_function_min_max_over_window() -> TsdbResult<()> {
        let min_fn = window_function();
        for v in [5.0, 1.0, 3.0] {
            let _ = min_fn(&[
                TemporalValue::Float(v),
                TemporalValue::Integer(3),
                TemporalValue::String("MIN".to_string()),
            ])?;
        }
        let min_result = min_fn(&[
            TemporalValue::Float(3.0),
            TemporalValue::Integer(3),
            TemporalValue::String("MIN".to_string()),
        ])?;
        // Window now holds the last 3 of [5,1,3,3] = [1,3,3]; MIN = 1.0
        assert_eq!(min_result, TemporalValue::Float(1.0));
        Ok(())
    }

    #[test]
    fn test_window_function_rejects_non_positive_window_size() {
        let window_fn = window_function();
        let result = window_fn(&[
            TemporalValue::Float(1.0),
            TemporalValue::Integer(0),
            TemporalValue::String("AVG".to_string()),
        ]);
        assert!(result.is_err());
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

        // A non-null observation is recorded and returned unchanged.
        assert_eq!(result, TemporalValue::Float(42.5));

        Ok(())
    }

    /// Regression test: a `Null` value with no prior observations must fail
    /// loudly instead of silently fabricating `0.0`. See P1 stub finding on
    /// `sparql/extensions.rs`.
    #[test]
    fn test_interpolate_null_without_history_errors() {
        let interpolate_fn = interpolate_function();

        let timestamp = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let result = interpolate_fn(&[
            TemporalValue::Timestamp(timestamp),
            TemporalValue::Null,
            TemporalValue::String("linear".to_string()),
        ]);

        assert!(
            result.is_err(),
            "interpolating a Null with no observed history must error, not return 0.0"
        );
    }

    /// Regression test: `ts:interpolate` must use the real `Interpolator`
    /// machinery over previously observed points instead of collapsing a
    /// missing value to `0.0`.
    #[test]
    fn test_interpolate_forward_fill_uses_last_observed_value() -> TsdbResult<()> {
        let interpolate_fn = interpolate_function();

        let t0 = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let t1 = t0 + Duration::seconds(10);

        // Observe a real value at t0.
        interpolate_fn(&[
            TemporalValue::Timestamp(t0),
            TemporalValue::Float(42.5),
            TemporalValue::String("forward".to_string()),
        ])?;

        // A missing value at t1 should be forward-filled from t0's
        // observation, not zeroed out.
        let result = interpolate_fn(&[
            TemporalValue::Timestamp(t1),
            TemporalValue::Null,
            TemporalValue::String("forward".to_string()),
        ])?;

        assert_eq!(result, TemporalValue::Float(42.5));
        Ok(())
    }

    #[test]
    fn test_interpolate_linear_between_two_observations() -> TsdbResult<()> {
        let interpolate_fn = interpolate_function();

        let t0 = DateTime::from_timestamp(1609459200, 0).expect("valid timestamp");
        let t1 = t0 + Duration::seconds(10);
        let mid = t0 + Duration::seconds(5);

        interpolate_fn(&[
            TemporalValue::Timestamp(t0),
            TemporalValue::Float(10.0),
            TemporalValue::String("linear".to_string()),
        ])?;
        interpolate_fn(&[
            TemporalValue::Timestamp(t1),
            TemporalValue::Float(20.0),
            TemporalValue::String("linear".to_string()),
        ])?;

        let result = interpolate_fn(&[
            TemporalValue::Timestamp(mid),
            TemporalValue::Null,
            TemporalValue::String("linear".to_string()),
        ])?;

        match result {
            TemporalValue::Float(v) => assert!((v - 15.0).abs() < 0.001),
            other => panic!("expected float, got {other:?}"),
        }
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
