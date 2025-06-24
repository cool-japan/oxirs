//! Built-in SPARQL Functions (Fixed Version)
//!
//! This module implements essential SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{CustomFunction, Value, ValueType, ExecutionContext, ExtensionRegistry, CustomAggregate, AggregateState};
use crate::algebra::{Expression, Term, Iri, Literal};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Datelike};

/// Register essential built-in SPARQL functions
pub fn register_builtin_functions(registry: &ExtensionRegistry) -> Result<()> {
    // Essential string functions
    registry.register_function(StrFunction)?;
    registry.register_function(LangFunction)?;
    registry.register_function(DatatypeFunction)?;
    registry.register_function(BoundFunction)?;
    
    // Essential numeric functions
    registry.register_function(AbsFunction)?;
    
    // Essential type checking functions
    registry.register_function(IsIriFunction)?;
    registry.register_function(IsLiteralFunction)?;
    
    // Essential aggregate functions
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