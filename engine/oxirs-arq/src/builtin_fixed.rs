//! Built-in SPARQL Functions (Fixed Version)
//!
//! This module implements essential SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::algebra::{Expression, Iri, Literal, Term};
use crate::extensions::{
    AggregateState, CustomAggregate, CustomFunction, ExecutionContext, ExtensionRegistry, Value,
    ValueType,
};
use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use md5;
use rand::Rng;
use regex::Regex;
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::{Digest, Sha256, Sha384, Sha512};
use std::collections::HashMap;
use urlencoding;
use uuid::Uuid;

/// Register comprehensive SPARQL 1.1 built-in functions
pub fn register_builtin_functions(registry: &ExtensionRegistry) -> Result<()> {
    // String functions
    registry.register_function(StrFunction)?;
    registry.register_function(LangFunction)?;
    registry.register_function(DatatypeFunction)?;
    registry.register_function(ConcatFunction)?;
    registry.register_function(SubstrFunction)?;
    registry.register_function(StrlenFunction)?;
    registry.register_function(UcaseFunction)?;
    registry.register_function(LcaseFunction)?;
    registry.register_function(ContainsFunction)?;
    registry.register_function(StrStartsFunction)?;
    registry.register_function(StrEndsFunction)?;
    registry.register_function(ReplaceFunction)?;

    // Numeric functions
    registry.register_function(AbsFunction)?;

    // Type checking functions
    registry.register_function(BoundFunction)?;
    registry.register_function(IsIriFunction)?;
    registry.register_function(IsLiteralFunction)?;

    // Additional string functions
    registry.register_function(RegexFunction)?;
    registry.register_function(EncodeForUriFunction)?;
    registry.register_function(SubstringBeforeFunction)?;
    registry.register_function(SubstringAfterFunction)?;

    // Additional numeric functions
    registry.register_function(CeilFunction)?;
    registry.register_function(FloorFunction)?;
    registry.register_function(RoundFunction)?;
    registry.register_function(RandFunction)?;

    // Date/time functions
    registry.register_function(NowFunction)?;
    registry.register_function(YearFunction)?;
    registry.register_function(MonthFunction)?;
    registry.register_function(DayFunction)?;
    registry.register_function(HoursFunction)?;
    registry.register_function(MinutesFunction)?;
    registry.register_function(SecondsFunction)?;
    registry.register_function(TimezoneFunction)?;

    // Hash functions
    registry.register_function(Md5Function)?;
    registry.register_function(Sha1Function)?;
    registry.register_function(Sha256Function)?;
    registry.register_function(Sha384Function)?;
    registry.register_function(Sha512Function)?;

    // Other functions
    registry.register_function(UuidFunction)?;
    registry.register_function(StruuidFunction)?;
    registry.register_function(IriFunction)?;
    registry.register_function(BnodeFunction)?;
    registry.register_function(CoalesceFunction)?;
    registry.register_function(IfFunction)?;
    registry.register_function(SametermFunction)?;
    registry.register_function(LangMatchesFunction)?;

    // Aggregate functions
    registry.register_aggregate(CountAggregate)?;
    registry.register_aggregate(SumAggregate)?;
    registry.register_aggregate(MinAggregate)?;
    registry.register_aggregate(MaxAggregate)?;
    registry.register_aggregate(AvgAggregate)?;
    registry.register_aggregate(SampleAggregate)?;
    registry.register_aggregate(GroupConcatAggregate)?;

    Ok(())
}

// String Functions

#[derive(Debug, Clone)]
struct StrFunction;

impl CustomFunction for StrFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/XMLSchema#string"
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
        "Returns the string value of a literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("str() requires exactly 1 argument");
        }

        match &args[0] {
            Value::String(s) => Ok(Value::String(s.clone())),
            Value::Literal { value, .. } => Ok(Value::String(value.clone())),
            Value::Iri(iri) => Ok(Value::String(iri.clone())),
            Value::Integer(i) => Ok(Value::String(i.to_string())),
            Value::Float(f) => Ok(Value::String(f.to_string())),
            Value::Boolean(b) => Ok(Value::String(b.to_string())),
            _ => bail!("Cannot convert {} to string", args[0].type_name()),
        }
    }
}

#[derive(Debug, Clone)]
struct LangFunction;

impl CustomFunction for LangFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#lang"
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
        "Returns the language tag of a literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("lang() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Literal {
                language: Some(lang),
                ..
            } => Ok(Value::String(lang.clone())),
            Value::Literal { language: None, .. } => Ok(Value::String("".to_string())),
            _ => bail!("lang() can only be applied to literals"),
        }
    }
}

#[derive(Debug, Clone)]
struct DatatypeFunction;

impl CustomFunction for DatatypeFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#datatype"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Iri
    }
    fn documentation(&self) -> &str {
        "Returns the datatype IRI of a literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("datatype() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Literal {
                datatype: Some(dt),
                language: None,
                ..
            } => Ok(Value::Iri(dt.clone())),
            Value::Literal {
                datatype: None,
                language: Some(_),
                ..
            } => Ok(Value::Iri(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            )),
            Value::Literal {
                datatype: None,
                language: None,
                ..
            } => Ok(Value::Iri(
                "http://www.w3.org/2001/XMLSchema#string".to_string(),
            )),
            _ => bail!("datatype() can only be applied to literals"),
        }
    }
}

#[derive(Debug, Clone)]
struct BoundFunction;

impl CustomFunction for BoundFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/sparql-results#bound"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests whether a variable is bound"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("bound() requires exactly 1 argument");
        }

        match &args[0] {
            Value::String(var_name) => {
                // Convert string to Variable type
                let variable = crate::algebra::Variable::new(var_name)?;
                let is_bound = context.variables.contains_key(&variable);
                Ok(Value::Boolean(is_bound))
            }
            _ => bail!("bound() requires a variable name as string"),
        }
    }
}

// Numeric Functions

#[derive(Debug, Clone)]
struct AbsFunction;

impl CustomFunction for AbsFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#abs"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Float]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Float
    }
    fn documentation(&self) -> &str {
        "Returns the absolute value of a number"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("abs() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Integer(i) => Ok(Value::Integer(i.abs())),
            Value::Float(f) => Ok(Value::Float(f.abs())),
            _ => bail!("abs() requires a numeric argument"),
        }
    }
}

// Type checking functions

#[derive(Debug, Clone)]
struct IsIriFunction;

impl CustomFunction for IsIriFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#isIRI"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a term is an IRI"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isIRI() requires exactly 1 argument");
        }

        let is_iri = matches!(&args[0], Value::Iri(_));
        Ok(Value::Boolean(is_iri))
    }
}

#[derive(Debug, Clone)]
struct IsLiteralFunction;

impl CustomFunction for IsLiteralFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#isLiteral"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a term is a literal"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isLiteral() requires exactly 1 argument");
        }

        let is_literal = matches!(
            &args[0],
            Value::Literal { .. }
                | Value::String(_)
                | Value::Integer(_)
                | Value::Float(_)
                | Value::Boolean(_)
        );
        Ok(Value::Boolean(is_literal))
    }
}

// Aggregate Functions

#[derive(Debug, Clone)]
struct CountAggregate;

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
struct SumAggregate;

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

// Additional String Functions

#[derive(Debug, Clone)]
struct ConcatFunction;

impl CustomFunction for ConcatFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#concat"
    }
    fn arity(&self) -> Option<usize> {
        None
    } // Variable arity
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Concatenates multiple strings"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.is_empty() {
            bail!("concat() requires at least 1 argument");
        }

        let mut result = String::new();
        for arg in args {
            match arg {
                Value::String(s) => result.push_str(s),
                Value::Literal { value, .. } => result.push_str(value),
                _ => bail!("concat() can only concatenate string values"),
            }
        }
        Ok(Value::String(result))
    }
}

#[derive(Debug, Clone)]
struct SubstrFunction;

impl CustomFunction for SubstrFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#substring"
    }
    fn arity(&self) -> Option<usize> {
        Some(3)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::Integer, ValueType::Integer]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns substring of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() < 2 || args.len() > 3 {
            bail!("substr() requires 2 or 3 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to substr() must be a string"),
        };

        let start = match &args[1] {
            Value::Integer(i) => *i as usize,
            _ => bail!("Second argument to substr() must be an integer"),
        };

        if args.len() == 3 {
            let length = match &args[2] {
                Value::Integer(i) => *i as usize,
                _ => bail!("Third argument to substr() must be an integer"),
            };

            let result = if start > 0 && start <= string.len() {
                let start_idx = start - 1; // SPARQL uses 1-based indexing
                let end_idx = std::cmp::min(start_idx + length, string.len());
                string[start_idx..end_idx].to_string()
            } else {
                String::new()
            };
            Ok(Value::String(result))
        } else {
            let result = if start > 0 && start <= string.len() {
                let start_idx = start - 1;
                string[start_idx..].to_string()
            } else {
                String::new()
            };
            Ok(Value::String(result))
        }
    }
}

#[derive(Debug, Clone)]
struct StrlenFunction;

impl CustomFunction for StrlenFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#string-length"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Integer
    }
    fn documentation(&self) -> &str {
        "Returns the length of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("strlen() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strlen() argument must be a string"),
        };

        Ok(Value::Integer(string.chars().count() as i64))
    }
}

#[derive(Debug, Clone)]
struct UcaseFunction;

impl CustomFunction for UcaseFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#upper-case"
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
        "Converts string to uppercase"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("ucase() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("ucase() argument must be a string"),
        };

        Ok(Value::String(string.to_uppercase()))
    }
}

#[derive(Debug, Clone)]
struct LcaseFunction;

impl CustomFunction for LcaseFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#lower-case"
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
        "Converts string to lowercase"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("lcase() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("lcase() argument must be a string"),
        };

        Ok(Value::String(string.to_lowercase()))
    }
}

#[derive(Debug, Clone)]
struct ContainsFunction;

impl CustomFunction for ContainsFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#contains"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a string contains another string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("contains() requires exactly 2 arguments");
        }

        let haystack = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to contains() must be a string"),
        };

        let needle = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to contains() must be a string"),
        };

        Ok(Value::Boolean(haystack.contains(needle)))
    }
}

#[derive(Debug, Clone)]
struct StrStartsFunction;

impl CustomFunction for StrStartsFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#starts-with"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a string starts with another string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("strStarts() requires exactly 2 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to strStarts() must be a string"),
        };

        let prefix = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to strStarts() must be a string"),
        };

        Ok(Value::Boolean(string.starts_with(prefix)))
    }
}

#[derive(Debug, Clone)]
struct StrEndsFunction;

impl CustomFunction for StrEndsFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#ends-with"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a string ends with another string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("strEnds() requires exactly 2 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to strEnds() must be a string"),
        };

        let suffix = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to strEnds() must be a string"),
        };

        Ok(Value::Boolean(string.ends_with(suffix)))
    }
}

#[derive(Debug, Clone)]
struct ReplaceFunction;

impl CustomFunction for ReplaceFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#replace"
    }
    fn arity(&self) -> Option<usize> {
        Some(3)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Replaces all occurrences of a pattern with replacement string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 3 {
            bail!("replace() requires exactly 3 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to replace() must be a string"),
        };

        let pattern = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to replace() must be a string"),
        };

        let replacement = match &args[2] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Third argument to replace() must be a string"),
        };

        let regex = Regex::new(pattern).context("Invalid regex pattern")?;
        Ok(Value::String(
            regex.replace_all(string, replacement).to_string(),
        ))
    }
}

// Additional String Functions

#[derive(Debug, Clone)]
struct RegexFunction;

impl CustomFunction for RegexFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#matches"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a string matches a regular expression"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("regex() requires exactly 2 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to regex() must be a string"),
        };

        let pattern = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to regex() must be a string"),
        };

        let regex = Regex::new(pattern).context("Invalid regex pattern")?;
        Ok(Value::Boolean(regex.is_match(string)))
    }
}

#[derive(Debug, Clone)]
struct EncodeForUriFunction;

impl CustomFunction for EncodeForUriFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#encode-for-uri"
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
        "Encodes a string for use in a URI"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("encode-for-uri() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("encode-for-uri() argument must be a string"),
        };

        let encoded = urlencoding::encode(string);
        Ok(Value::String(encoded.to_string()))
    }
}

#[derive(Debug, Clone)]
struct SubstringBeforeFunction;

impl CustomFunction for SubstringBeforeFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#substring-before"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns the substring before the first occurrence of a delimiter"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("substring-before() requires exactly 2 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to substring-before() must be a string"),
        };

        let delimiter = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to substring-before() must be a string"),
        };

        let result = if let Some(pos) = string.find(delimiter) {
            string[..pos].to_string()
        } else {
            String::new()
        };

        Ok(Value::String(result))
    }
}

#[derive(Debug, Clone)]
struct SubstringAfterFunction;

impl CustomFunction for SubstringAfterFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#substring-after"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns the substring after the first occurrence of a delimiter"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("substring-after() requires exactly 2 arguments");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to substring-after() must be a string"),
        };

        let delimiter = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to substring-after() must be a string"),
        };

        let result = if let Some(pos) = string.find(delimiter) {
            string[pos + delimiter.len()..].to_string()
        } else {
            String::new()
        };

        Ok(Value::String(result))
    }
}

// Additional Numeric Functions

#[derive(Debug, Clone)]
struct CeilFunction;

impl CustomFunction for CeilFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#ceiling"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Float]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Float
    }
    fn documentation(&self) -> &str {
        "Returns the ceiling (round up) of a number"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("ceil() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Integer(i) => Ok(Value::Integer(*i)),
            Value::Float(f) => Ok(Value::Float(f.ceil())),
            _ => bail!("ceil() requires a numeric argument"),
        }
    }
}

#[derive(Debug, Clone)]
struct FloorFunction;

impl CustomFunction for FloorFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#floor"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Float]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Float
    }
    fn documentation(&self) -> &str {
        "Returns the floor (round down) of a number"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("floor() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Integer(i) => Ok(Value::Integer(*i)),
            Value::Float(f) => Ok(Value::Float(f.floor())),
            _ => bail!("floor() requires a numeric argument"),
        }
    }
}

#[derive(Debug, Clone)]
struct RoundFunction;

impl CustomFunction for RoundFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#round"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Float]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Float
    }
    fn documentation(&self) -> &str {
        "Returns the rounded value of a number"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("round() requires exactly 1 argument");
        }

        match &args[0] {
            Value::Integer(i) => Ok(Value::Integer(*i)),
            Value::Float(f) => Ok(Value::Float(f.round())),
            _ => bail!("round() requires a numeric argument"),
        }
    }
}

#[derive(Debug, Clone)]
struct RandFunction;

impl CustomFunction for RandFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#random"
    }
    fn arity(&self) -> Option<usize> {
        Some(0)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Float
    }
    fn documentation(&self) -> &str {
        "Returns a random number between 0 and 1"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("rand() requires no arguments");
        }

        let mut rng = rand::thread_rng();
        Ok(Value::Float(rng.gen::<f64>()))
    }
}

// Date/Time Functions

#[derive(Debug, Clone)]
struct NowFunction;

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
struct YearFunction;

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
struct MonthFunction;

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
struct DayFunction;

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
struct HoursFunction;

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
struct MinutesFunction;

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
struct SecondsFunction;

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
struct TimezoneFunction;

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

// Hash Functions

#[derive(Debug, Clone)]
struct Md5Function;

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
        "Returns the MD5 hash of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("md5() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("md5() argument must be a string"),
        };

        let digest = md5::compute(string.as_bytes());
        Ok(Value::String(format!("{:x}", digest)))
    }
}

#[derive(Debug, Clone)]
struct Sha1Function;

impl CustomFunction for Sha1Function {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sha1"
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
        "Returns the SHA1 hash of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("sha1() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("sha1() argument must be a string"),
        };

        let mut hasher = Sha1::new();
        Sha1Digest::update(&mut hasher, string.as_bytes());
        let result = hasher.finalize();
        Ok(Value::String(format!("{:x}", result)))
    }
}

#[derive(Debug, Clone)]
struct Sha256Function;

impl CustomFunction for Sha256Function {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sha256"
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
        "Returns the SHA256 hash of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("sha256() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("sha256() argument must be a string"),
        };

        let mut hasher = Sha256::new();
        hasher.update(string.as_bytes());
        let result = hasher.finalize();
        Ok(Value::String(format!("{:x}", result)))
    }
}

#[derive(Debug, Clone)]
struct Sha384Function;

impl CustomFunction for Sha384Function {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sha384"
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
        "Returns the SHA384 hash of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("sha384() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("sha384() argument must be a string"),
        };

        let mut hasher = Sha384::new();
        hasher.update(string.as_bytes());
        let result = hasher.finalize();
        Ok(Value::String(format!("{:x}", result)))
    }
}

#[derive(Debug, Clone)]
struct Sha512Function;

impl CustomFunction for Sha512Function {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sha512"
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
        "Returns the SHA512 hash of a string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("sha512() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("sha512() argument must be a string"),
        };

        let mut hasher = Sha512::new();
        hasher.update(string.as_bytes());
        let result = hasher.finalize();
        Ok(Value::String(format!("{:x}", result)))
    }
}

// Other Functions

#[derive(Debug, Clone)]
struct UuidFunction;

impl CustomFunction for UuidFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#uuid"
    }
    fn arity(&self) -> Option<usize> {
        Some(0)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Generates a random UUID"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("uuid() requires no arguments");
        }

        let uuid = Uuid::new_v4();
        Ok(Value::String(uuid.to_string()))
    }
}

#[derive(Debug, Clone)]
struct StruuidFunction;

impl CustomFunction for StruuidFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#struuid"
    }
    fn arity(&self) -> Option<usize> {
        Some(0)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Generates a random UUID as an IRI string"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("struuid() requires no arguments");
        }

        let uuid = Uuid::new_v4();
        Ok(Value::String(format!("urn:uuid:{}", uuid)))
    }
}

#[derive(Debug, Clone)]
struct IriFunction;

impl CustomFunction for IriFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#iri"
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Iri
    }
    fn documentation(&self) -> &str {
        "Converts a string to an IRI"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("iri() requires exactly 1 argument");
        }

        let string = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("iri() argument must be a string"),
        };

        Ok(Value::Iri(string.clone()))
    }
}

#[derive(Debug, Clone)]
struct BnodeFunction;

impl CustomFunction for BnodeFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#bnode"
    }
    fn arity(&self) -> Option<usize> {
        Some(0)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::BlankNode
    }
    fn documentation(&self) -> &str {
        "Generates a new blank node"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if !args.is_empty() {
            bail!("bnode() requires no arguments");
        }

        let uuid = Uuid::new_v4();
        Ok(Value::BlankNode(format!("b{}", uuid.simple())))
    }
}

#[derive(Debug, Clone)]
struct CoalesceFunction;

impl CustomFunction for CoalesceFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#coalesce"
    }
    fn arity(&self) -> Option<usize> {
        None
    } // Variable arity
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns the first non-null value from the arguments"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.is_empty() {
            bail!("coalesce() requires at least 1 argument");
        }

        for arg in args {
            match arg {
                Value::Null => continue,
                _ => return Ok(arg.clone()),
            }
        }

        Ok(Value::Null)
    }
}

#[derive(Debug, Clone)]
struct IfFunction;

impl CustomFunction for IfFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#if"
    }
    fn arity(&self) -> Option<usize> {
        Some(3)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Boolean, ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Conditional function: if condition then expr1 else expr2"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 3 {
            bail!("if() requires exactly 3 arguments");
        }

        let condition = match &args[0] {
            Value::Boolean(b) => *b,
            _ => bail!("First argument to if() must be a boolean"),
        };

        if condition {
            Ok(args[1].clone())
        } else {
            Ok(args[2].clone())
        }
    }
}

#[derive(Debug, Clone)]
struct SametermFunction;

impl CustomFunction for SametermFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#sameTerm"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if two RDF terms are the same"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("sameTerm() requires exactly 2 arguments");
        }

        Ok(Value::Boolean(args[0] == args[1]))
    }
}

#[derive(Debug, Clone)]
struct LangMatchesFunction;

impl CustomFunction for LangMatchesFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#langMatches"
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }
    fn documentation(&self) -> &str {
        "Tests if a language tag matches a language range"
    }
    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("langMatches() requires exactly 2 arguments");
        }

        let lang_tag = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("First argument to langMatches() must be a string"),
        };

        let lang_range = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("Second argument to langMatches() must be a string"),
        };

        // Simple implementation - for full RFC 4647 compliance, use a proper library
        let matches = if lang_range == "*" {
            !lang_tag.is_empty()
        } else {
            lang_tag
                .to_lowercase()
                .starts_with(&lang_range.to_lowercase())
        };

        Ok(Value::Boolean(matches))
    }
}

// Additional Aggregate Functions

#[derive(Debug, Clone)]
struct MinAggregate;

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
struct MinState {
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
struct MaxAggregate;

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
struct MaxState {
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
struct AvgAggregate;

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
struct SampleAggregate;

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
struct SampleState {
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
struct GroupConcatAggregate;

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
struct GroupConcatState {
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
