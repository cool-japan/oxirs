//! Built-in SPARQL Functions - Type Checking Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{
    CustomFunction, ExecutionContext, Value,
    ValueType,
};
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub(crate) struct BoundFunction;

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

// Type checking functions

#[derive(Debug, Clone)]
pub(crate) struct IsIriFunction;

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
pub(crate) struct IsLiteralFunction;

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
