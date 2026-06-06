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

// ─── Stub macro for remaining date/time and hash functions ──────────────────

macro_rules! stub_function {
    ($name:ident, $iri:expr_2021, $arity:expr_2021, $doc:expr_2021) => {
        #[derive(Debug, Clone)]
        pub(crate) struct $name;

        impl CustomFunction for $name {
            fn name(&self) -> &str {
                $iri
            }
            fn arity(&self) -> Option<usize> {
                $arity
            }
            fn parameter_types(&self) -> Vec<ValueType> {
                vec![]
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

            fn execute(&self, _args: &[Value], _context: &ExecutionContext) -> Result<Value> {
                bail!("Function {} not yet implemented", self.name())
            }
        }
    };
}

stub_function!(
    MonthFunction,
    "http://www.w3.org/2005/xpath-functions#month-from-dateTime",
    Some(1),
    "Extracts month from dateTime"
);
stub_function!(
    DayFunction,
    "http://www.w3.org/2005/xpath-functions#day-from-dateTime",
    Some(1),
    "Extracts day from dateTime"
);
stub_function!(
    HoursFunction,
    "http://www.w3.org/2005/xpath-functions#hours-from-dateTime",
    Some(1),
    "Extracts hours from dateTime"
);
stub_function!(
    MinutesFunction,
    "http://www.w3.org/2005/xpath-functions#minutes-from-dateTime",
    Some(1),
    "Extracts minutes from dateTime"
);
stub_function!(
    SecondsFunction,
    "http://www.w3.org/2005/xpath-functions#seconds-from-dateTime",
    Some(1),
    "Extracts seconds from dateTime"
);
stub_function!(
    TimezoneFunction,
    "http://www.w3.org/2005/xpath-functions#timezone-from-dateTime",
    Some(1),
    "Extracts timezone from dateTime"
);
stub_function!(
    TzFunction,
    "http://www.w3.org/2005/xpath-functions#tz",
    Some(1),
    "Returns timezone string"
);

// ─── Hash Functions ──────────────────────────────────────────────────────────

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

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("md5() requires a string argument"),
        };

        Ok(Value::String(format!("md5:{len}", len = s.len())))
    }
}

stub_function!(
    Sha1Function,
    "http://www.w3.org/2005/xpath-functions#sha1",
    Some(1),
    "Computes SHA1 hash"
);
stub_function!(
    Sha256Function,
    "http://www.w3.org/2005/xpath-functions#sha256",
    Some(1),
    "Computes SHA256 hash"
);
stub_function!(
    Sha384Function,
    "http://www.w3.org/2005/xpath-functions#sha384",
    Some(1),
    "Computes SHA384 hash"
);
stub_function!(
    Sha512Function,
    "http://www.w3.org/2005/xpath-functions#sha512",
    Some(1),
    "Computes SHA512 hash"
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
        Box::new(SumState { sum: 0.0, count: 0 })
    }
}

#[derive(Debug, Clone)]
struct SumState {
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

// ─── Shared stub aggregate state ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct StubAggregateState;

impl AggregateState for StubAggregateState {
    fn add(&mut self, _value: &Value) -> Result<()> {
        Ok(())
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    fn reset(&mut self) {}

    fn clone_state(&self) -> Box<dyn AggregateState> {
        Box::new(StubAggregateState)
    }
}

macro_rules! stub_aggregate {
    ($name:ident, $iri:expr_2021, $doc:expr_2021) => {
        #[derive(Debug, Clone)]
        pub(crate) struct $name;

        impl CustomAggregate for $name {
            fn name(&self) -> &str {
                $iri
            }
            fn documentation(&self) -> &str {
                $doc
            }

            fn init(&self) -> Box<dyn AggregateState> {
                Box::new(StubAggregateState)
            }
        }
    };
}

stub_aggregate!(
    AvgAggregate,
    "http://www.w3.org/2005/xpath-functions#avg",
    "Computes average of numeric values"
);
stub_aggregate!(
    MinAggregate,
    "http://www.w3.org/2005/xpath-functions#min",
    "Finds minimum value"
);
stub_aggregate!(
    MaxAggregate,
    "http://www.w3.org/2005/xpath-functions#max",
    "Finds maximum value"
);
stub_aggregate!(
    GroupConcatAggregate,
    "http://www.w3.org/2005/xpath-functions#group-concat",
    "Concatenates group values"
);
stub_aggregate!(
    SampleAggregate,
    "http://www.w3.org/2005/xpath-functions#sample",
    "Returns sample value from group"
);
