//! Built-in SPARQL Functions - Other Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Result};

use uuid::Uuid;

// Other Functions

#[derive(Debug, Clone)]
pub(crate) struct UuidFunction;

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
pub(crate) struct StruuidFunction;

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
        Ok(Value::String(format!("urn:uuid:{uuid}")))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IriFunction;

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
pub(crate) struct BnodeFunction;

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
        Ok(Value::BlankNode(format!("b{uuid}", uuid = uuid.simple())))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CoalesceFunction;

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
pub(crate) struct IfFunction;

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
pub(crate) struct SametermFunction;

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
pub(crate) struct LangMatchesFunction;

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
