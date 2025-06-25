//! Built-in SPARQL Functions (Fixed Version)
//!
//! This module implements essential SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{CustomFunction, Value, ValueType, ExecutionContext, ExtensionRegistry, CustomAggregate, AggregateState};
use crate::algebra::{Expression, Term, Iri, Literal};
use anyhow::{Result, anyhow, bail, Context};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Datelike, Timelike};
use regex::Regex;
use md5;
use sha1::Sha1;
use sha2::{Sha256, Sha384, Sha512, Digest};
use rand::Rng;
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
    
    // Aggregate functions
    registry.register_aggregate(CountAggregate)?;
    registry.register_aggregate(SumAggregate)?;
    
    Ok(())
}

// String Functions

#[derive(Debug, Clone)]
struct StrFunction;

impl CustomFunction for StrFunction {
    fn name(&self) -> &str { "http://www.w3.org/2001/XMLSchema#string" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Literal] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Returns the string value of a literal" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#lang" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Literal] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Returns the language tag of a literal" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("lang() requires exactly 1 argument");
        }
        
        match &args[0] {
            Value::Literal { language: Some(lang), .. } => Ok(Value::String(lang.clone())),
            Value::Literal { language: None, .. } => Ok(Value::String("".to_string())),
            _ => bail!("lang() can only be applied to literals"),
        }
    }
}

#[derive(Debug, Clone)]
struct DatatypeFunction;

impl CustomFunction for DatatypeFunction {
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#datatype" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Literal] }
    fn return_type(&self) -> ValueType { ValueType::Iri }
    fn documentation(&self) -> &str { "Returns the datatype IRI of a literal" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("datatype() requires exactly 1 argument");
        }
        
        match &args[0] {
            Value::Literal { datatype: Some(dt), language: None, .. } => Ok(Value::Iri(dt.clone())),
            Value::Literal { datatype: None, language: Some(_), .. } => {
                Ok(Value::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string()))
            }
            Value::Literal { datatype: None, language: None, .. } => {
                Ok(Value::Iri("http://www.w3.org/2001/XMLSchema#string".to_string()))
            }
            _ => bail!("datatype() can only be applied to literals"),
        }
    }
}

#[derive(Debug, Clone)]
struct BoundFunction;

impl CustomFunction for BoundFunction {
    fn name(&self) -> &str { "http://www.w3.org/2005/sparql-results#bound" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests whether a variable is bound" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("bound() requires exactly 1 argument");
        }
        
        match &args[0] {
            Value::String(var_name) => {
                let is_bound = context.variables.contains_key(var_name);
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#abs" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Float] }
    fn return_type(&self) -> ValueType { ValueType::Float }
    fn documentation(&self) -> &str { "Returns the absolute value of a number" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#isIRI" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Literal] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests if a term is an IRI" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#isLiteral" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::Literal] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests if a term is a literal" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isLiteral() requires exactly 1 argument");
        }
        
        let is_literal = matches!(&args[0], Value::Literal { .. } | Value::String(_) | Value::Integer(_) | Value::Float(_) | Value::Boolean(_));
        Ok(Value::Boolean(is_literal))
    }
}

// Aggregate Functions

#[derive(Debug, Clone)]
struct CountAggregate;

impl CustomAggregate for CountAggregate {
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#count" }
    fn documentation(&self) -> &str { "Counts the number of values" }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#sum" }
    fn documentation(&self) -> &str { "Sums numeric values" }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#concat" }
    fn arity(&self) -> Option<usize> { None } // Variable arity
    fn parameter_types(&self) -> Vec<ValueType> { vec![] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Concatenates multiple strings" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#substring" }
    fn arity(&self) -> Option<usize> { Some(3) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String, ValueType::Integer, ValueType::Integer] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Returns substring of a string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#string-length" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::Integer }
    fn documentation(&self) -> &str { "Returns the length of a string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#upper-case" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Converts string to uppercase" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#lower-case" }
    fn arity(&self) -> Option<usize> { Some(1) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Converts string to lowercase" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#contains" }
    fn arity(&self) -> Option<usize> { Some(2) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String, ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests if a string contains another string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#starts-with" }
    fn arity(&self) -> Option<usize> { Some(2) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String, ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests if a string starts with another string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#ends-with" }
    fn arity(&self) -> Option<usize> { Some(2) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String, ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::Boolean }
    fn documentation(&self) -> &str { "Tests if a string ends with another string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
    fn name(&self) -> &str { "http://www.w3.org/2005/xpath-functions#replace" }
    fn arity(&self) -> Option<usize> { Some(3) }
    fn parameter_types(&self) -> Vec<ValueType> { vec![ValueType::String, ValueType::String, ValueType::String] }
    fn return_type(&self) -> ValueType { ValueType::String }
    fn documentation(&self) -> &str { "Replaces all occurrences of a pattern with replacement string" }
    fn clone_function(&self) -> Box<dyn CustomFunction> { Box::new(self.clone()) }
    
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
        Ok(Value::String(regex.replace_all(string, replacement).to_string()))
    }
}