//! Built-in SPARQL Functions - Date-Time Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{
    CustomFunction, ExecutionContext, Value,
    ValueType,
};
use anyhow::{bail, Context, Result};

use chrono::{DateTime, Datelike, Timelike, Utc};

// Date/Time Functions

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
        ValueType::Literal
    }
    fn documentation(&self) -> &str {
        "Returns the current date and time"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("now() requires no arguments");
        }

        let now = Utc::now();
        Ok(Value::Literal {
            value: now.to_rfc3339(),
            language: None,
            datatype: Some("http://www.w3.org/2001/XMLSchema#dateTime".to_string()),
        })
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
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the year from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("year() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("year() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.year() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MonthFunction;

impl CustomFunction for MonthFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#month-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the month from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("month() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("month() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.month() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DayFunction;

impl CustomFunction for DayFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#day-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the day from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("day() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("day() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.day() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HoursFunction;

impl CustomFunction for HoursFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#hours-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the hours from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("hours() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("hours() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.hour() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MinutesFunction;

impl CustomFunction for MinutesFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#minutes-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the minutes from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("minutes() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("minutes() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.minute() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SecondsFunction;

impl CustomFunction for SecondsFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#seconds-from-dateTime"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Extracts the seconds from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("seconds() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("seconds() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::Integer(datetime.second() as i64))
    }
}

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
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Extracts the timezone from a dateTime literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("timezone() requires exactly 1 argument");
        }

        let datetime_str = match &args[0] {
            Value::Literal { value, .. } => value,
            _ => bail!("timezone() requires a dateTime literal"),
        };

        let datetime =
            DateTime::parse_from_rfc3339(datetime_str).context("Invalid dateTime format")?;
        Ok(Value::String(datetime.format("%z").to_string()))
    }
}
