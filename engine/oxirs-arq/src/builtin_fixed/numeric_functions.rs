//! Built-in SPARQL Functions - Numeric Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{
    CustomFunction, ExecutionContext, Value,
    ValueType,
};
use anyhow::{bail, Result};

use scirs2_core::random::{Random, Rng};

// Numeric Functions

#[derive(Debug, Clone)]
pub(crate) struct AbsFunction;

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

// Additional Numeric Functions

#[derive(Debug, Clone)]
pub(crate) struct CeilFunction;

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
pub(crate) struct FloorFunction;

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
pub(crate) struct RoundFunction;

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
pub(crate) struct RandFunction;

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

        let mut random = Random::default();
        Ok(Value::Float(random.random::<f64>()))
    }
}
