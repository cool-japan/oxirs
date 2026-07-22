//! SPARQL Built-in Date/Time, Hash, and Aggregate Functions
//!
//! Date/time extraction (NOW, YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TIMEZONE, TZ),
//! cryptographic hash functions (MD5, SHA1, SHA256, SHA384, SHA512),
//! and SPARQL aggregate functions (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE).

use crate::extensions::{
    AggregateState, CustomAggregate, CustomFunction, ExecutionContext, Value, ValueType,
};
use anyhow::{bail, Result};
use chrono::Datelike;

// ─── Date/Time Functions ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct NowFunction;

impl CustomFunction for NowFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#current-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(0)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::DateTime
    }
    fn documentation(&self) -> &str {
        "Returns the current date and time"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("now() takes no arguments");
        }

        Ok(Value::DateTime(context.query_time))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct YearFunction;

impl CustomFunction for YearFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#year-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::DateTime]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the year from a dateTime"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("year() requires exactly 1 argument");
        }

        match &args[0] {
            Value::DateTime(dt) => Ok(Value::Integer(dt.year() as i64)),
            _ => bail!("year() requires a dateTime argument"),
        }
    }
}

// ─── Date/time component functions ──────────────────────────────────────────

use chrono::Timelike;

/// Generate a `CustomFunction` that extracts an integer component from a
/// `Value::DateTime` (e.g. month, hour). `$extract` is a closure taking the
/// `DateTime<Utc>` and returning the `i64` component value.
macro_rules! datetime_int_function {
    ($name:ident, $iri:expr_2021, $doc:expr_2021, $extract:expr_2021) => {
        #[derive(Debug, Clone)]
        pub(crate) struct $name;

        impl CustomFunction for $name {
            fn name(&self) -> &str {
                $iri
            }
            fn arity(&self) -> Option<usize> {
                Some(1)
            }
            fn parameter_types(&self) -> Vec<ValueType> {
                vec![ValueType::DateTime]
            }
            fn return_type(&self) -> ValueType {
                ValueType::Integer
            }
            fn documentation(&self) -> &str {
                $doc
            }
            fn clone_function(&self) -> Box<dyn CustomFunction> {
                Box::new(self.clone())
            }

            fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
                if args.len() != 1 {
                    bail!("{} requires exactly 1 argument", self.name());
                }
                match &args[0] {
                    Value::DateTime(dt) => {
                        let extract: fn(&chrono::DateTime<chrono::Utc>) -> i64 = $extract;
                        Ok(Value::Integer(extract(dt)))
                    }
                    _ => bail!("{} requires a dateTime argument", self.name()),
                }
            }
        }
    };
}

datetime_int_function!(
    MonthFunction,
    "http://www.w3.org/2005/xpath-functions#month-from-dateTime",
    "Extracts month from dateTime",
    |dt| dt.month() as i64
);
datetime_int_function!(
    DayFunction,
    "http://www.w3.org/2005/xpath-functions#day-from-dateTime",
    "Extracts day from dateTime",
    |dt| dt.day() as i64
);
datetime_int_function!(
    HoursFunction,
    "http://www.w3.org/2005/xpath-functions#hours-from-dateTime",
    "Extracts hours from dateTime",
    |dt| dt.hour() as i64
);
datetime_int_function!(
    MinutesFunction,
    "http://www.w3.org/2005/xpath-functions#minutes-from-dateTime",
    "Extracts minutes from dateTime",
    |dt| dt.minute() as i64
);
datetime_int_function!(
    SecondsFunction,
    "http://www.w3.org/2005/xpath-functions#seconds-from-dateTime",
    "Extracts seconds from dateTime",
    |dt| dt.second() as i64
);

/// `TIMEZONE(dt)` — the timezone of a dateTime as an xsd:dayTimeDuration.
/// `Value::DateTime` is normalized to UTC, so the timezone is always `PT0S`.
#[derive(Debug, Clone)]
pub(crate) struct TimezoneFunction;

impl CustomFunction for TimezoneFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#timezone-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::DateTime]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }
    fn documentation(&self) -> &str {
        "Extracts timezone from dateTime"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }
    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("timezone() requires exactly 1 argument");
        }
        match &args[0] {
            Value::DateTime(_) => Ok(Value::Literal {
                value: "PT0S".to_string(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#dayTimeDuration".to_string()),
            }),
            _ => bail!("timezone() requires a dateTime argument"),
        }
    }
}

/// `TZ(dt)` — the timezone of a dateTime as a simple string. `Value::DateTime`
/// is normalized to UTC, so the result is always `"Z"`.
#[derive(Debug, Clone)]
pub(crate) struct TzFunction;

impl CustomFunction for TzFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#tz"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::DateTime]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns timezone string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }
    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("tz() requires exactly 1 argument");
        }
        match &args[0] {
            Value::DateTime(_) => Ok(Value::String("Z".to_string())),
            _ => bail!("tz() requires a dateTime argument"),
        }
    }
}

// ─── Hash Functions ──────────────────────────────────────────────────────────

/// Extract the string bytes to hash from a hash function's single argument.
fn hash_input_value(args: &[Value], fn_name: &str) -> Result<String> {
    if args.len() != 1 {
        bail!("{fn_name}() requires exactly 1 argument");
    }
    match &args[0] {
        Value::String(s) => Ok(s.clone()),
        Value::Literal { value, .. } => Ok(value.clone()),
        _ => bail!("{fn_name}() requires a string argument"),
    }
}

/// Generate a `CustomFunction` computing a lowercase-hex digest via a
/// RustCrypto `Digest` implementation.
macro_rules! digest_function {
    ($name:ident, $iri:expr_2021, $doc:expr_2021, $fn_name:expr_2021, $hasher:ty) => {
        #[derive(Debug, Clone)]
        pub(crate) struct $name;

        impl CustomFunction for $name {
            fn name(&self) -> &str {
                $iri
            }
            fn arity(&self) -> Option<usize> {
                Some(1)
            }
            fn parameter_types(&self) -> Vec<ValueType> {
                vec![ValueType::String]
            }
            fn return_type(&self) -> ValueType {
                ValueType::String
            }
            fn documentation(&self) -> &str {
                $doc
            }
            fn clone_function(&self) -> Box<dyn CustomFunction> {
                Box::new(self.clone())
            }
            fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
                use sha2::Digest;
                let input = hash_input_value(args, $fn_name)?;
                let mut hasher = <$hasher>::new();
                hasher.update(input.as_bytes());
                Ok(Value::String(hex::encode(hasher.finalize())))
            }
        }
    };
}

#[derive(Debug, Clone)]
pub(crate) struct Md5Function;

impl CustomFunction for Md5Function {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#md5"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Computes MD5 hash of a string"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("md5() requires exactly 1 argument");
        }

        let s = hash_input_value(args, "md5")?;
        Ok(Value::String(format!("{:x}", md5::compute(s.as_bytes()))))
    }
}

digest_function!(
    Sha1Function,
    "http://www.w3.org/2005/xpath-functions#sha1",
    "Computes SHA1 hash",
    "sha1",
    sha1::Sha1
);
digest_function!(
    Sha256Function,
    "http://www.w3.org/2005/xpath-functions#sha256",
    "Computes SHA256 hash",
    "sha256",
    sha2::Sha256
);
digest_function!(
    Sha384Function,
    "http://www.w3.org/2005/xpath-functions#sha384",
    "Computes SHA384 hash",
    "sha384",
    sha2::Sha384
);
digest_function!(
    Sha512Function,
    "http://www.w3.org/2005/xpath-functions#sha512",
    "Computes SHA512 hash",
    "sha512",
    sha2::Sha512
);

// ─── Aggregate Functions ─────────────────────────────────────────────────────

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
struct CountState {
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
        Box::new(SumState::default())
    }
}

/// Accumulator for SPARQL `SUM`. While every operand seen so far is an
/// `xsd:integer`, the sum is tracked in an exact `i64` and the result keeps
/// `xsd:integer` typing with full precision (no `f64` rounding beyond 2^53).
/// The first non-integer operand promotes the running total to `f64`.
#[derive(Debug, Clone, Default)]
struct SumState {
    int_sum: i64,
    float_sum: f64,
    /// True once a non-integer operand has been seen (or an integer overflowed).
    promoted: bool,
    count: usize,
}

impl AggregateState for SumState {
    fn add(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Integer(i) => {
                if self.promoted {
                    self.float_sum += *i as f64;
                } else if let Some(next) = self.int_sum.checked_add(*i) {
                    self.int_sum = next;
                } else {
                    // Integer overflow: fall back to f64 accumulation.
                    self.float_sum = self.int_sum as f64 + *i as f64;
                    self.promoted = true;
                }
                self.count += 1;
            }
            Value::Float(f) => {
                if !self.promoted {
                    self.float_sum = self.int_sum as f64;
                    self.promoted = true;
                }
                self.float_sum += f;
                self.count += 1;
            }
            _ => bail!("sum() can only aggregate numeric values"),
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        if self.count == 0 {
            // SPARQL SUM of an empty group is 0 (xsd:integer).
            Ok(Value::Integer(0))
        } else if self.promoted {
            Ok(Value::Float(self.float_sum))
        } else {
            Ok(Value::Integer(self.int_sum))
        }
    }

    fn reset(&mut self) {
        *self = SumState::default();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// ─── AVG ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct AvgAggregate;

impl CustomAggregate for AvgAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#avg"
    }
    fn documentation(&self) -> &str {
        "Computes average of numeric values"
    }
    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(AvgState::default())
    }
}

#[derive(Debug, Clone, Default)]
struct AvgState {
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
            // AVG of an empty group is 0 (SPARQL 1.1).
            Ok(Value::Integer(0))
        } else {
            Ok(Value::Float(self.sum / self.count as f64))
        }
    }

    fn reset(&mut self) {
        *self = AvgState::default();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// ─── MIN / MAX ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct MinAggregate;

impl CustomAggregate for MinAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#min"
    }
    fn documentation(&self) -> &str {
        "Finds minimum value"
    }
    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(ExtremumState {
            keep_max: false,
            current: None,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MaxAggregate;

impl CustomAggregate for MaxAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#max"
    }
    fn documentation(&self) -> &str {
        "Finds maximum value"
    }
    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(ExtremumState {
            keep_max: true,
            current: None,
        })
    }
}

/// Shared MIN/MAX accumulator: keeps the extreme value seen using the total
/// ordering defined by `Value::partial_cmp`.
#[derive(Debug, Clone)]
struct ExtremumState {
    keep_max: bool,
    current: Option<Value>,
}

impl AggregateState for ExtremumState {
    fn add(&mut self, value: &Value) -> Result<()> {
        if matches!(value, Value::Null) {
            return Ok(());
        }
        match &self.current {
            None => self.current = Some(value.clone()),
            Some(existing) => {
                if let Some(ordering) = value.partial_cmp(existing) {
                    let replace = if self.keep_max {
                        ordering == std::cmp::Ordering::Greater
                    } else {
                        ordering == std::cmp::Ordering::Less
                    };
                    if replace {
                        self.current = Some(value.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(self.current.clone().unwrap_or(Value::Null))
    }

    fn reset(&mut self) {
        self.current = None;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// ─── GROUP_CONCAT ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct GroupConcatAggregate;

impl CustomAggregate for GroupConcatAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#group-concat"
    }
    fn documentation(&self) -> &str {
        "Concatenates group values"
    }
    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(GroupConcatState::default())
    }
}

/// GROUP_CONCAT accumulator: joins the string form of each value with a single
/// space separator (the SPARQL default when no `SEPARATOR` is given).
#[derive(Debug, Clone, Default)]
struct GroupConcatState {
    parts: Vec<String>,
}

impl GroupConcatState {
    fn value_to_string(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Literal { value, .. } => value.clone(),
            Value::Iri(iri) => iri.clone(),
            Value::BlankNode(id) => id.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::DateTime(dt) => dt.to_rfc3339(),
            other => format!("{other:?}"),
        }
    }
}

impl AggregateState for GroupConcatState {
    fn add(&mut self, value: &Value) -> Result<()> {
        if !matches!(value, Value::Null) {
            self.parts.push(Self::value_to_string(value));
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::String(self.parts.join(" ")))
    }

    fn reset(&mut self) {
        self.parts.clear();
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

// ─── SAMPLE ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct SampleAggregate;

impl CustomAggregate for SampleAggregate {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sample"
    }
    fn documentation(&self) -> &str {
        "Returns sample value from group"
    }
    fn init(&self) -> Box<dyn AggregateState> {
        Box::new(SampleState { sample: None })
    }
}

/// SAMPLE accumulator: keeps the first non-null value seen (any deterministic
/// choice is valid per SPARQL 1.1).
#[derive(Debug, Clone)]
struct SampleState {
    sample: Option<Value>,
}

impl AggregateState for SampleState {
    fn add(&mut self, value: &Value) -> Result<()> {
        if self.sample.is_none() && !matches!(value, Value::Null) {
            self.sample = Some(value.clone());
        }
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(self.sample.clone().unwrap_or(Value::Null))
    }

    fn reset(&mut self) {
        self.sample = None;
    }

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::{CustomAggregate, CustomFunction, ExecutionContext, Value};

    /// SUM over integers must keep xsd:integer typing and full i64 precision.
    #[test]
    fn regression_sum_integer_typing() {
        let agg = SumAggregate;
        let mut state = agg.init();
        for v in [1i64, 2, 3] {
            state.add(&Value::Integer(v)).unwrap();
        }
        match state.result().unwrap() {
            Value::Integer(6) => {}
            other => panic!("SUM of integers must be Integer(6), got {other:?}"),
        }

        // Large sums beyond 2^53 keep exact precision as i64.
        let mut big = agg.init();
        big.add(&Value::Integer(9_007_199_254_740_993)).unwrap(); // 2^53 + 1
        big.add(&Value::Integer(2)).unwrap();
        match big.result().unwrap() {
            Value::Integer(v) => assert_eq!(v, 9_007_199_254_740_995),
            other => panic!("expected exact integer, got {other:?}"),
        }

        // A float operand promotes the result to Float.
        let mut mixed = agg.init();
        mixed.add(&Value::Integer(1)).unwrap();
        mixed.add(&Value::Float(0.5)).unwrap();
        match mixed.result().unwrap() {
            Value::Float(f) => assert!((f - 1.5).abs() < 1e-9),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    /// AVG/MIN/MAX/SAMPLE/GROUP_CONCAT must compute real results, not Null.
    #[test]
    fn regression_aggregates_real() {
        let ctx = ExecutionContext::default();
        let _ = &ctx;

        let mut avg = AvgAggregate.init();
        for v in [2i64, 4] {
            avg.add(&Value::Integer(v)).unwrap();
        }
        match avg.result().unwrap() {
            Value::Float(f) => assert!((f - 3.0).abs() < 1e-9),
            other => panic!("AVG must be Float(3.0), got {other:?}"),
        }

        let mut min = MinAggregate.init();
        let mut max = MaxAggregate.init();
        for v in [5i64, 1, 3] {
            min.add(&Value::Integer(v)).unwrap();
            max.add(&Value::Integer(v)).unwrap();
        }
        assert_eq!(min.result().unwrap(), Value::Integer(1));
        assert_eq!(max.result().unwrap(), Value::Integer(5));

        let mut gc = GroupConcatState::default();
        gc.add(&Value::String("a".into())).unwrap();
        gc.add(&Value::String("b".into())).unwrap();
        assert_eq!(gc.result().unwrap(), Value::String("a b".into()));

        let mut sample = SampleAggregate.init();
        sample.add(&Value::Integer(7)).unwrap();
        sample.add(&Value::Integer(8)).unwrap();
        assert_eq!(sample.result().unwrap(), Value::Integer(7));
    }

    /// Registered SHA/MD5 functions must compute real digests.
    #[test]
    fn regression_registered_hash_functions_real() {
        let ctx = ExecutionContext::default();
        let sha256 = Sha256Function;
        let out = sha256
            .execute(&[Value::String("abc".into())], &ctx)
            .unwrap();
        assert_eq!(
            out,
            Value::String(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into()
            )
        );

        let md5 = Md5Function;
        let out = md5.execute(&[Value::String("abc".into())], &ctx).unwrap();
        assert_eq!(
            out,
            Value::String("900150983cd24fb0d6963f7d28e17f72".into())
        );
    }
}
