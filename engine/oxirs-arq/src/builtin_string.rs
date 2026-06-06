//! SPARQL Built-in String Functions
//!
//! String type checking, manipulation, and RDF-specific string functions
//! as specified in the SPARQL 1.1 W3C recommendation.

use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Result};

// ─── Core RDF String/Term Predicates ──────────────────────────────────────────

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
                if let Ok(variable) = oxirs_core::Variable::new(var_name) {
                    let is_bound = context.variables.contains_key(&variable);
                    Ok(Value::Boolean(is_bound))
                } else {
                    Ok(Value::Boolean(false))
                }
            }
            _ => bail!("bound() requires a variable name as string"),
        }
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
        "Constructs an IRI from a string"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("iri() requires exactly 1 argument");
        }

        match &args[0] {
            Value::String(s) => {
                let resolved = if let Some(base) = &context.base_iri {
                    if s.starts_with("http://")
                        || s.starts_with("https://")
                        || s.starts_with("ftp://")
                    {
                        s.clone()
                    } else {
                        format!("{base}{s}")
                    }
                } else {
                    s.clone()
                };
                Ok(Value::Iri(resolved))
            }
            Value::Iri(iri) => Ok(Value::Iri(iri.clone())),
            _ => bail!("iri() requires a string argument"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UriFunction;

impl CustomFunction for UriFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#uri"
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
        "Alias for iri() function"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        IriFunction.execute(args, context)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlankFunction;

impl CustomFunction for BlankFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#bnode"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::BlankNode
    }
    fn documentation(&self) -> &str {
        "Constructs a blank node"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        match args.len() {
            0 => {
                let id = format!(
                    "_:gen{r}",
                    r = {
                        use scirs2_core::random::{rng, RngExt};
                        let mut random = rng();
                        random.random::<u32>()
                    }
                );
                Ok(Value::BlankNode(id))
            }
            1 => match &args[0] {
                Value::String(s) => Ok(Value::BlankNode(format!("_{s}"))),
                _ => bail!("bnode() requires a string argument"),
            },
            _ => bail!("bnode() takes 0 or 1 arguments"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LiteralFunction;

impl CustomFunction for LiteralFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#literal"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }
    fn documentation(&self) -> &str {
        "Constructs a literal with optional language or datatype"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        match args.len() {
            1 => match &args[0] {
                Value::String(s) => Ok(Value::Literal {
                    value: s.clone(),
                    language: None,
                    datatype: None,
                }),
                _ => bail!("literal() first argument must be a string"),
            },
            2 => match (&args[0], &args[1]) {
                (Value::String(s), Value::String(lang_or_dt)) => {
                    if lang_or_dt.starts_with("http://") {
                        Ok(Value::Literal {
                            value: s.clone(),
                            language: None,
                            datatype: Some(lang_or_dt.clone()),
                        })
                    } else {
                        Ok(Value::Literal {
                            value: s.clone(),
                            language: Some(lang_or_dt.clone()),
                            datatype: None,
                        })
                    }
                }
                _ => bail!("literal() arguments must be strings"),
            },
            _ => bail!("literal() takes 1 or 2 arguments"),
        }
    }
}

// ─── String Manipulation Functions ─────────────────────────────────────────────

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

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strlen() requires a string argument"),
        };

        Ok(Value::Integer(s.chars().count() as i64))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SubstrFunction;

impl CustomFunction for SubstrFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#substring"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Returns a substring of a string"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 && args.len() != 3 {
            bail!("substr() requires 2 or 3 arguments");
        }

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("substr() first argument must be a string"),
        };

        let start = match &args[1] {
            Value::Integer(i) => *i as usize,
            _ => bail!("substr() second argument must be an integer"),
        };

        let chars: Vec<char> = s.chars().collect();

        if start > chars.len() {
            return Ok(Value::String("".to_string()));
        }

        let result = if args.len() == 3 {
            let length = match &args[2] {
                Value::Integer(i) => *i as usize,
                _ => bail!("substr() third argument must be an integer"),
            };
            let end = std::cmp::min(start + length, chars.len());
            chars[start..end].iter().collect()
        } else {
            chars[start..].iter().collect()
        };

        Ok(Value::String(result))
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
        "Converts a string to uppercase"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("ucase() requires exactly 1 argument");
        }

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("ucase() requires a string argument"),
        };

        Ok(Value::String(s.to_uppercase()))
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
        "Converts a string to lowercase"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("lcase() requires exactly 1 argument");
        }

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("lcase() requires a string argument"),
        };

        Ok(Value::String(s.to_lowercase()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StrstartsFunction;

impl CustomFunction for StrstartsFunction {
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
            bail!("strstarts() requires exactly 2 arguments");
        }

        let s1 = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strstarts() first argument must be a string"),
        };

        let s2 = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strstarts() second argument must be a string"),
        };

        Ok(Value::Boolean(s1.starts_with(s2)))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StrendsFunction;

impl CustomFunction for StrendsFunction {
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
            bail!("strends() requires exactly 2 arguments");
        }

        let s1 = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strends() first argument must be a string"),
        };

        let s2 = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("strends() second argument must be a string"),
        };

        Ok(Value::Boolean(s1.ends_with(s2)))
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

        let s1 = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("contains() first argument must be a string"),
        };

        let s2 = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("contains() second argument must be a string"),
        };

        Ok(Value::Boolean(s1.contains(s2)))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ConcatFunction;

impl CustomFunction for ConcatFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#concat"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
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
            return Ok(Value::String("".to_string()));
        }

        let mut result = String::new();
        for arg in args {
            let s = match arg {
                Value::String(s) => s,
                Value::Literal { value, .. } => value,
                Value::Integer(i) => &i.to_string(),
                Value::Float(f) => &f.to_string(),
                Value::Boolean(b) => &b.to_string(),
                _ => bail!("concat() arguments must be convertible to strings"),
            };
            result.push_str(s);
        }

        Ok(Value::String(result))
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
            bail!("encode_for_uri() requires exactly 1 argument");
        }

        let s = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("encode_for_uri() requires a string argument"),
        };

        let encoded = s
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "-_.~".contains(c) {
                    c.to_string()
                } else {
                    format!("%{c:02X}", c = c as u8)
                }
            })
            .collect();

        Ok(Value::String(encoded))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReplaceFunction;

impl CustomFunction for ReplaceFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#replace"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::String
    }
    fn documentation(&self) -> &str {
        "Replaces occurrences of a pattern in a string"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 3 && args.len() != 4 {
            bail!("replace() requires 3 or 4 arguments");
        }

        let input = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("replace() first argument must be a string"),
        };

        let pattern = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("replace() second argument must be a string"),
        };

        let replacement = match &args[2] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("replace() third argument must be a string"),
        };

        let result = input.replace(pattern, replacement);
        Ok(Value::String(result))
    }
}

// ─── Type Checking Functions ────────────────────────────────────────────────

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
pub(crate) struct IsUriFunction;

impl CustomFunction for IsUriFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#isURI"
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
        "Alias for isIRI()"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
        IsIriFunction.execute(args, context)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IsBlankFunction;

impl CustomFunction for IsBlankFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#isBlank"
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
        "Tests if a term is a blank node"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isBlank() requires exactly 1 argument");
        }

        let is_blank = matches!(&args[0], Value::BlankNode(_));
        Ok(Value::Boolean(is_blank))
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

#[derive(Debug, Clone)]
pub(crate) struct IsNumericFunction;

impl CustomFunction for IsNumericFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#isNumeric"
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
        "Tests if a term is numeric"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isNumeric() requires exactly 1 argument");
        }

        let is_numeric = matches!(&args[0], Value::Integer(_) | Value::Float(_));
        Ok(Value::Boolean(is_numeric))
    }
}

// ─── Logical Functions ──────────────────────────────────────────────────────

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
        vec![ValueType::Boolean, ValueType::Literal, ValueType::Literal]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }
    fn documentation(&self) -> &str {
        "Conditional expression"
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
            _ => bail!("if() first argument must be boolean"),
        };

        if condition {
            Ok(args[1].clone())
        } else {
            Ok(args[2].clone())
        }
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
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
    }
    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }
    fn documentation(&self) -> &str {
        "Returns the first non-null argument"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.is_empty() {
            bail!("coalesce() requires at least 1 argument");
        }

        for arg in args {
            if !matches!(arg, Value::Null) {
                return Ok(arg.clone());
            }
        }

        Ok(Value::Null)
    }
}

// ─── Regex Function ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct RegexFunction;

impl CustomFunction for RegexFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2005/xpath-functions#matches"
    }
    fn arity(&self) -> Option<usize> {
        None
    }
    fn parameter_types(&self) -> Vec<ValueType> {
        vec![]
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
        use regex::Regex;

        if args.len() != 2 && args.len() != 3 {
            bail!("regex() requires 2 or 3 arguments");
        }

        let text = match &args[0] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("regex() first argument must be a string"),
        };

        let pattern = match &args[1] {
            Value::String(s) => s,
            Value::Literal { value, .. } => value,
            _ => bail!("regex() second argument must be a string"),
        };

        let _flags = if args.len() == 3 {
            match &args[2] {
                Value::String(s) => s,
                Value::Literal { value, .. } => value,
                _ => bail!("regex() third argument must be a string"),
            }
        } else {
            ""
        };

        match Regex::new(pattern) {
            Ok(re) => Ok(Value::Boolean(re.is_match(text))),
            Err(_) => bail!("Invalid regular expression: {}", pattern),
        }
    }
}
