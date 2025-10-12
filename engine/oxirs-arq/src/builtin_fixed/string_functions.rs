//! Built-in SPARQL Functions - String Functions
//!
//! This module implements SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Context, Result};

use regex::Regex;
use urlencoding;

// String Functions

#[derive(Debug, Clone)]
pub(crate) struct StrFunction;

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
pub(crate) struct LangFunction;

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
pub(crate) struct DatatypeFunction;

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

// Additional String Functions

#[derive(Debug, Clone)]
pub(crate) struct ConcatFunction;

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
pub(crate) struct SubstrFunction;

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
pub(crate) struct StrlenFunction;

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
pub(crate) struct UcaseFunction;

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
pub(crate) struct LcaseFunction;

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
pub(crate) struct ContainsFunction;

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
pub(crate) struct StrStartsFunction;

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
pub(crate) struct StrEndsFunction;

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
pub(crate) struct ReplaceFunction;

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
pub(crate) struct RegexFunction;

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
pub(crate) struct EncodeForUriFunction;

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
pub(crate) struct SubstringBeforeFunction;

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
pub(crate) struct SubstringAfterFunction;

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
