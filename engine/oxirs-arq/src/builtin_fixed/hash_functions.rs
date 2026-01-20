//! Built-in SPARQL Functions - Hash Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Result};

use md5;
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::{Sha256, Sha384, Sha512};

// Hash Functions

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
        Ok(Value::String(format!("{digest:x}")))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Sha1Function;

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
        Ok(Value::String(format!("{result:x}")))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Sha256Function;

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
        Ok(Value::String(format!("{result:x}")))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Sha384Function;

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
        Ok(Value::String(format!("{result:x}")))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Sha512Function;

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
        Ok(Value::String(format!("{result:x}")))
    }
}
