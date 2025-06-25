//! SPARQL 1.2 built-in functions and extensions
//!
//! This module implements the extended function library for SPARQL 1.2,
//! including new string functions, math functions, and advanced features.

use crate::model::{Term, Literal, NamedNode};
use crate::query::algebra::Expression;
use crate::OxirsError;
use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration, Datelike, Timelike};
use regex::Regex;

/// SPARQL function registry
pub struct FunctionRegistry {
    /// Built-in functions
    functions: HashMap<String, FunctionImpl>,
    /// Custom extension functions
    extensions: HashMap<String, Arc<dyn CustomFunction>>,
}

/// Function implementation
pub enum FunctionImpl {
    /// Native Rust implementation
    Native(NativeFunction),
    /// JavaScript implementation (for extensibility)
    JavaScript(String),
    /// WASM module
    Wasm(Vec<u8>),
}

/// Native function pointer
pub type NativeFunction = Arc<dyn Fn(&[Term]) -> Result<Term, OxirsError> + Send + Sync>;

/// Custom function trait
pub trait CustomFunction: Send + Sync {
    /// Execute the function
    fn execute(&self, args: &[Term]) -> Result<Term, OxirsError>;
    
    /// Get function metadata
    fn metadata(&self) -> FunctionMetadata;
}

/// Function metadata
#[derive(Debug, Clone)]
pub struct FunctionMetadata {
    /// Function name
    pub name: String,
    /// Description
    pub description: String,
    /// Minimum arguments
    pub min_args: usize,
    /// Maximum arguments (None = unlimited)
    pub max_args: Option<usize>,
    /// Argument types
    pub arg_types: Vec<ArgumentType>,
    /// Return type
    pub return_type: ReturnType,
}

/// Argument type specification
#[derive(Debug, Clone)]
pub enum ArgumentType {
    /// Any RDF term
    Any,
    /// String literal
    String,
    /// Numeric literal
    Numeric,
    /// Boolean literal
    Boolean,
    /// Date/time literal
    DateTime,
    /// IRI
    IRI,
    /// Specific datatype
    Datatype(String),
}

/// Return type specification
#[derive(Debug, Clone)]
pub enum ReturnType {
    /// Same as input
    SameAsInput,
    /// Specific type
    Fixed(ArgumentType),
    /// Dynamic based on input
    Dynamic,
}

impl FunctionRegistry {
    /// Create new function registry with SPARQL 1.2 built-ins
    pub fn new() -> Self {
        let mut registry = FunctionRegistry {
            functions: HashMap::new(),
            extensions: HashMap::new(),
        };
        
        registry.register_sparql_12_functions();
        registry
    }
    
    /// Register all SPARQL 1.2 built-in functions
    fn register_sparql_12_functions(&mut self) {
        // String functions
        self.register_native("CONCAT", Arc::new(fn_concat));
        self.register_native("STRLEN", Arc::new(fn_strlen));
        self.register_native("SUBSTR", Arc::new(fn_substr));
        self.register_native("REPLACE", Arc::new(fn_replace));
        self.register_native("REGEX", Arc::new(fn_regex));
        self.register_native("STRAFTER", Arc::new(fn_strafter));
        self.register_native("STRBEFORE", Arc::new(fn_strbefore));
        self.register_native("STRSTARTS", Arc::new(fn_strstarts));
        self.register_native("STRENDS", Arc::new(fn_strends));
        self.register_native("CONTAINS", Arc::new(fn_contains));
        self.register_native("ENCODE_FOR_URI", Arc::new(fn_encode_for_uri));
        
        // Case functions
        self.register_native("UCASE", Arc::new(fn_ucase));
        self.register_native("LCASE", Arc::new(fn_lcase));
        
        // Numeric functions
        self.register_native("ABS", Arc::new(fn_abs));
        self.register_native("CEIL", Arc::new(fn_ceil));
        self.register_native("FLOOR", Arc::new(fn_floor));
        self.register_native("ROUND", Arc::new(fn_round));
        self.register_native("RAND", Arc::new(fn_rand));
        
        // Math functions (SPARQL 1.2 additions)
        self.register_native("SQRT", Arc::new(fn_sqrt));
        self.register_native("SIN", Arc::new(fn_sin));
        self.register_native("COS", Arc::new(fn_cos));
        self.register_native("TAN", Arc::new(fn_tan));
        self.register_native("ASIN", Arc::new(fn_asin));
        self.register_native("ACOS", Arc::new(fn_acos));
        self.register_native("ATAN", Arc::new(fn_atan));
        self.register_native("ATAN2", Arc::new(fn_atan2));
        self.register_native("EXP", Arc::new(fn_exp));
        self.register_native("LOG", Arc::new(fn_log));
        self.register_native("LOG10", Arc::new(fn_log10));
        self.register_native("POW", Arc::new(fn_pow));
        
        // Date/time functions
        self.register_native("NOW", Arc::new(fn_now));
        self.register_native("YEAR", Arc::new(fn_year));
        self.register_native("MONTH", Arc::new(fn_month));
        self.register_native("DAY", Arc::new(fn_day));
        self.register_native("HOURS", Arc::new(fn_hours));
        self.register_native("MINUTES", Arc::new(fn_minutes));
        self.register_native("SECONDS", Arc::new(fn_seconds));
        self.register_native("TIMEZONE", Arc::new(fn_timezone));
        self.register_native("TZ", Arc::new(fn_tz));
        
        // Hash functions (SPARQL 1.2)
        self.register_native("SHA1", Arc::new(fn_sha1));
        self.register_native("SHA256", Arc::new(fn_sha256));
        self.register_native("SHA384", Arc::new(fn_sha384));
        self.register_native("SHA512", Arc::new(fn_sha512));
        self.register_native("MD5", Arc::new(fn_md5));
        
        // Type functions
        self.register_native("STR", Arc::new(fn_str));
        self.register_native("LANG", Arc::new(fn_lang));
        self.register_native("DATATYPE", Arc::new(fn_datatype));
        self.register_native("IRI", Arc::new(fn_iri));
        self.register_native("URI", Arc::new(fn_iri)); // Alias
        self.register_native("BNODE", Arc::new(fn_bnode));
        self.register_native("STRDT", Arc::new(fn_strdt));
        self.register_native("STRLANG", Arc::new(fn_strlang));
        self.register_native("UUID", Arc::new(fn_uuid));
        self.register_native("STRUUID", Arc::new(fn_struuid));
        
        // Aggregate functions
        self.register_native("COUNT", Arc::new(fn_count));
        self.register_native("SUM", Arc::new(fn_sum));
        self.register_native("AVG", Arc::new(fn_avg));
        self.register_native("MIN", Arc::new(fn_min));
        self.register_native("MAX", Arc::new(fn_max));
        self.register_native("GROUP_CONCAT", Arc::new(fn_group_concat));
        self.register_native("SAMPLE", Arc::new(fn_sample));
        
        // Boolean functions
        self.register_native("NOT", Arc::new(fn_not));
        self.register_native("EXISTS", Arc::new(fn_exists));
        self.register_native("NOT_EXISTS", Arc::new(fn_not_exists));
        self.register_native("BOUND", Arc::new(fn_bound));
        self.register_native("COALESCE", Arc::new(fn_coalesce));
        self.register_native("IF", Arc::new(fn_if));
        
        // List functions (SPARQL 1.2)
        self.register_native("IN", Arc::new(fn_in));
        self.register_native("NOT_IN", Arc::new(fn_not_in));
    }
    
    /// Register a native function
    fn register_native(&mut self, name: &str, func: NativeFunction) {
        self.functions.insert(
            name.to_string(),
            FunctionImpl::Native(func)
        );
    }
    
    /// Register a custom function
    pub fn register_custom(&mut self, func: Arc<dyn CustomFunction>) {
        let metadata = func.metadata();
        self.extensions.insert(metadata.name.clone(), func);
    }
    
    /// Execute a function
    pub fn execute(&self, name: &str, args: &[Term]) -> Result<Term, OxirsError> {
        // Check built-in functions
        if let Some(func) = self.functions.get(name) {
            match func {
                FunctionImpl::Native(f) => f(args),
                FunctionImpl::JavaScript(_) => {
                    Err(OxirsError::Query("JavaScript functions not yet implemented".to_string()))
                }
                FunctionImpl::Wasm(_) => {
                    Err(OxirsError::Query("WASM functions not yet implemented".to_string()))
                }
            }
        }
        // Check custom functions
        else if let Some(func) = self.extensions.get(name) {
            func.execute(args)
        }
        else {
            Err(OxirsError::Query(format!("Unknown function: {}", name)))
        }
    }
}

// String functions implementation

fn fn_concat(args: &[Term]) -> Result<Term, OxirsError> {
    let mut result = String::new();
    
    for arg in args {
        match arg {
            Term::Literal(lit) => result.push_str(lit.value()),
            Term::NamedNode(nn) => result.push_str(nn.as_str()),
            _ => return Err(OxirsError::Query("CONCAT requires string arguments".to_string())),
        }
    }
    
    Ok(Term::Literal(Literal::new(&result)))
}

fn fn_strlen(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("STRLEN requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let len = lit.value().chars().count() as i64;
            Ok(Term::Literal(Literal::new_typed(
                &len.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("STRLEN requires string literal".to_string())),
    }
}

fn fn_substr(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query("SUBSTR requires 2 or 3 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(start_lit)) => {
            let string = str_lit.value();
            let start = start_lit.value().parse::<usize>()
                .map_err(|_| OxirsError::Query("Invalid start position".to_string()))?;
            
            let result = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(len_lit) => {
                        let len = len_lit.value().parse::<usize>()
                            .map_err(|_| OxirsError::Query("Invalid length".to_string()))?;
                        string.chars().skip(start - 1).take(len).collect::<String>()
                    }
                    _ => return Err(OxirsError::Query("Length must be numeric".to_string())),
                }
            } else {
                string.chars().skip(start - 1).collect::<String>()
            };
            
            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query("SUBSTR requires string and numeric arguments".to_string())),
    }
}

fn fn_replace(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 3 || args.len() > 4 {
        return Err(OxirsError::Query("REPLACE requires 3 or 4 arguments".to_string()));
    }
    
    match (&args[0], &args[1], &args[2]) {
        (Term::Literal(text), Term::Literal(pattern), Term::Literal(replacement)) => {
            let flags = if args.len() == 4 {
                match &args[3] {
                    Term::Literal(f) => f.value(),
                    _ => return Err(OxirsError::Query("Flags must be string".to_string())),
                }
            } else {
                ""
            };
            
            // Build regex with flags
            let regex_str = if flags.contains('i') {
                format!("(?i){}", pattern.value())
            } else {
                pattern.value().to_string()
            };
            
            let regex = Regex::new(&regex_str)
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {}", e)))?;
            
            let result = regex.replace_all(text.value(), replacement.value());
            Ok(Term::Literal(Literal::new(result.as_ref())))
        }
        _ => Err(OxirsError::Query("REPLACE requires string arguments".to_string())),
    }
}

fn fn_regex(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query("REGEX requires 2 or 3 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(pattern)) => {
            let flags = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(f) => f.value(),
                    _ => return Err(OxirsError::Query("Flags must be string".to_string())),
                }
            } else {
                ""
            };
            
            let regex_str = if flags.contains('i') {
                format!("(?i){}", pattern.value())
            } else {
                pattern.value().to_string()
            };
            
            let regex = Regex::new(&regex_str)
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {}", e)))?;
            
            let matches = regex.is_match(text.value());
            Ok(Term::Literal(Literal::new_typed(
                if matches { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("REGEX requires string arguments".to_string())),
    }
}

// String manipulation functions

fn fn_strafter(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRAFTER requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(after_lit)) => {
            let string = str_lit.value();
            let after = after_lit.value();
            
            if let Some(pos) = string.find(after) {
                let result = &string[pos + after.len()..];
                Ok(Term::Literal(Literal::new(result)))
            } else {
                Ok(Term::Literal(Literal::new("")))
            }
        }
        _ => Err(OxirsError::Query("STRAFTER requires string arguments".to_string())),
    }
}

fn fn_strbefore(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRBEFORE requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(before_lit)) => {
            let string = str_lit.value();
            let before = before_lit.value();
            
            if let Some(pos) = string.find(before) {
                let result = &string[..pos];
                Ok(Term::Literal(Literal::new(result)))
            } else {
                Ok(Term::Literal(Literal::new("")))
            }
        }
        _ => Err(OxirsError::Query("STRBEFORE requires string arguments".to_string())),
    }
}

fn fn_strstarts(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRSTARTS requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(prefix_lit)) => {
            let result = str_lit.value().starts_with(prefix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("STRSTARTS requires string arguments".to_string())),
    }
}

fn fn_strends(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRENDS requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(suffix_lit)) => {
            let result = str_lit.value().ends_with(suffix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("STRENDS requires string arguments".to_string())),
    }
}

fn fn_contains(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("CONTAINS requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(substr_lit)) => {
            let result = str_lit.value().contains(substr_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("CONTAINS requires string arguments".to_string())),
    }
}

fn fn_encode_for_uri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ENCODE_FOR_URI requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let encoded = urlencoding::encode(lit.value());
            Ok(Term::Literal(Literal::new(encoded.as_ref())))
        }
        _ => Err(OxirsError::Query("ENCODE_FOR_URI requires string argument".to_string())),
    }
}

// Case functions

fn fn_ucase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("UCASE requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            Ok(Term::Literal(Literal::new(&lit.value().to_uppercase())))
        }
        _ => Err(OxirsError::Query("UCASE requires string argument".to_string())),
    }
}

fn fn_lcase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("LCASE requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            Ok(Term::Literal(Literal::new(&lit.value().to_lowercase())))
        }
        _ => Err(OxirsError::Query("LCASE requires string argument".to_string())),
    }
}

// Numeric functions

fn fn_abs(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ABS requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ABS requires numeric argument".to_string()))?;
            let result = value.abs();
            
            // Preserve datatype
            let dt = lit.datatype();
            if dt.as_str() == "http://www.w3.org/2001/XMLSchema#integer" {
                Ok(Term::Literal(Literal::new_typed(
                    &(result as i64).to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
                )))
            } else {
                Ok(Term::Literal(Literal::new_typed(
                    &result.to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
                )))
            }
        }
        _ => Err(OxirsError::Query("ABS requires numeric literal".to_string())),
    }
}

fn fn_ceil(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("CEIL requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("CEIL requires numeric argument".to_string()))?;
            let result = value.ceil() as i64;
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("CEIL requires numeric literal".to_string())),
    }
}

fn fn_floor(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("FLOOR requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("FLOOR requires numeric argument".to_string()))?;
            let result = value.floor() as i64;
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("FLOOR requires numeric literal".to_string())),
    }
}

fn fn_round(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ROUND requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ROUND requires numeric argument".to_string()))?;
            let result = value.round() as i64;
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("ROUND requires numeric literal".to_string())),
    }
}

fn fn_rand(_args: &[Term]) -> Result<Term, OxirsError> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let value: f64 = rng.gen();
    Ok(Term::Literal(Literal::new_typed(
        &value.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
    )))
}

// Math functions (SPARQL 1.2)

fn fn_sqrt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SQRT requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("SQRT requires numeric argument".to_string()))?;
            if value < 0.0 {
                return Err(OxirsError::Query("SQRT of negative number".to_string()));
            }
            let result = value.sqrt();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("SQRT requires numeric literal".to_string())),
    }
}

fn fn_sin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SIN requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("SIN requires numeric argument".to_string()))?;
            let result = value.sin();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("SIN requires numeric literal".to_string())),
    }
}

fn fn_cos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("COS requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("COS requires numeric argument".to_string()))?;
            let result = value.cos();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("COS requires numeric literal".to_string())),
    }
}

fn fn_tan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("TAN requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("TAN requires numeric argument".to_string()))?;
            let result = value.tan();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("TAN requires numeric literal".to_string())),
    }
}

fn fn_asin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ASIN requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ASIN requires numeric argument".to_string()))?;
            if value < -1.0 || value > 1.0 {
                return Err(OxirsError::Query("ASIN argument must be between -1 and 1".to_string()));
            }
            let result = value.asin();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("ASIN requires numeric literal".to_string())),
    }
}

fn fn_acos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ACOS requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ACOS requires numeric argument".to_string()))?;
            if value < -1.0 || value > 1.0 {
                return Err(OxirsError::Query("ACOS argument must be between -1 and 1".to_string()));
            }
            let result = value.acos();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("ACOS requires numeric literal".to_string())),
    }
}

fn fn_atan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("ATAN requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN requires numeric argument".to_string()))?;
            let result = value.atan();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("ATAN requires numeric literal".to_string())),
    }
}

fn fn_atan2(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("ATAN2 requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(y_lit), Term::Literal(x_lit)) => {
            let y = y_lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let x = x_lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let result = y.atan2(x);
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("ATAN2 requires numeric literals".to_string())),
    }
}

fn fn_exp(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("EXP requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("EXP requires numeric argument".to_string()))?;
            let result = value.exp();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("EXP requires numeric literal".to_string())),
    }
}

fn fn_log(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("LOG requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query("LOG of non-positive number".to_string()));
            }
            let result = value.ln();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("LOG requires numeric literal".to_string())),
    }
}

fn fn_log10(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("LOG10 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG10 requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query("LOG10 of non-positive number".to_string()));
            }
            let result = value.log10();
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("LOG10 requires numeric literal".to_string())),
    }
}

fn fn_pow(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("POW requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(base_lit), Term::Literal(exp_lit)) => {
            let base = base_lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let exp = exp_lit.value().parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let result = base.powf(exp);
            Ok(Term::Literal(Literal::new_typed(
                &result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("POW requires numeric literals".to_string())),
    }
}

// Date/time functions

fn fn_now(_args: &[Term]) -> Result<Term, OxirsError> {
    let now = Utc::now();
    Ok(Term::Literal(Literal::new_typed(
        &now.to_rfc3339(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#dateTime").unwrap()
    )))
}

fn fn_year(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("YEAR requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &dt.year().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("YEAR requires dateTime literal".to_string())),
    }
}

fn fn_month(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("MONTH requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &dt.month().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("MONTH requires dateTime literal".to_string())),
    }
}

fn fn_day(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("DAY requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &dt.day().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("DAY requires dateTime literal".to_string())),
    }
}

fn fn_hours(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("HOURS requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &dt.hour().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("HOURS requires dateTime literal".to_string())),
    }
}

fn fn_minutes(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("MINUTES requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &dt.minute().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("MINUTES requires dateTime literal".to_string())),
    }
}

fn fn_seconds(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SECONDS requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                &format!("{}.{}", dt.second(), dt.nanosecond() / 1_000_000_000),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("SECONDS requires dateTime literal".to_string())),
    }
}

fn fn_timezone(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("TIMEZONE requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            let offset = dt.offset();
            let hours = offset.local_minus_utc() / 3600;
            let minutes = (offset.local_minus_utc() % 3600) / 60;
            
            let duration = if hours == 0 && minutes == 0 {
                "PT0S".to_string()
            } else {
                format!("PT{}H{}M", hours.abs(), minutes.abs())
            };
            
            Ok(Term::Literal(Literal::new_typed(
                &duration,
                NamedNode::new("http://www.w3.org/2001/XMLSchema#dayTimeDuration").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("TIMEZONE requires dateTime literal".to_string())),
    }
}

fn fn_tz(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("TZ requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            let offset = dt.offset();
            let hours = offset.local_minus_utc() / 3600;
            let minutes = (offset.local_minus_utc() % 3600) / 60;
            
            let tz_string = if hours == 0 && minutes == 0 {
                "Z".to_string()
            } else {
                format!("{:+03}:{:02}", hours, minutes.abs())
            };
            
            Ok(Term::Literal(Literal::new(&tz_string)))
        }
        _ => Err(OxirsError::Query("TZ requires dateTime literal".to_string())),
    }
}

// Hash functions

fn fn_sha1(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SHA1 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            use sha1::{Sha1, Digest};
            let mut hasher = Sha1::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{:x}", result);
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("SHA1 requires string literal".to_string())),
    }
}

fn fn_sha256(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SHA256 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{:x}", result);
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("SHA256 requires string literal".to_string())),
    }
}

fn fn_sha384(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SHA384 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Sha384, Digest};
            let mut hasher = Sha384::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{:x}", result);
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("SHA384 requires string literal".to_string())),
    }
}

fn fn_sha512(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("SHA512 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Sha512, Digest};
            let mut hasher = Sha512::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{:x}", result);
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("SHA512 requires string literal".to_string())),
    }
}

fn fn_md5(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("MD5 requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let mut hasher = md5::Context::new();
            hasher.consume(lit.value().as_bytes());
            let result = hasher.compute();
            let hex = format!("{:x}", result);
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("MD5 requires string literal".to_string())),
    }
}

// Type conversion functions

fn fn_str(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("STR requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value()))),
        Term::NamedNode(nn) => Ok(Term::Literal(Literal::new(nn.as_str()))),
        _ => Err(OxirsError::Query("STR requires literal or IRI".to_string())),
    }
}

fn fn_lang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("LANG requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let lang = lit.language().unwrap_or("");
            Ok(Term::Literal(Literal::new(lang)))
        }
        _ => Err(OxirsError::Query("LANG requires literal".to_string())),
    }
}

fn fn_datatype(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("DATATYPE requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let dt = lit.datatype();
            Ok(Term::NamedNode(NamedNode::new(dt.as_str()).unwrap()))
        }
        _ => Err(OxirsError::Query("DATATYPE requires literal".to_string())),
    }
}

fn fn_iri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("IRI requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let iri = NamedNode::new(lit.value())?;
            Ok(Term::NamedNode(iri))
        }
        Term::NamedNode(nn) => Ok(Term::NamedNode(nn.clone())),
        _ => Err(OxirsError::Query("IRI requires string literal or IRI".to_string())),
    }
}

fn fn_bnode(args: &[Term]) -> Result<Term, OxirsError> {
    use crate::model::BlankNode;
    
    if args.is_empty() {
        Ok(Term::BlankNode(BlankNode::new_unique()))
    } else if args.len() == 1 {
        match &args[0] {
            Term::Literal(lit) => {
                let bnode = BlankNode::new(lit.value())?;
                Ok(Term::BlankNode(bnode))
            }
            _ => Err(OxirsError::Query("BNODE requires string literal or no arguments".to_string())),
        }
    } else {
        Err(OxirsError::Query("BNODE requires 0 or 1 arguments".to_string()))
    }
}

fn fn_strdt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRDT requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::NamedNode(datatype)) => {
            Ok(Term::Literal(Literal::new_typed(value_lit.value(), datatype.clone())))
        }
        _ => Err(OxirsError::Query("STRDT requires string literal and IRI".to_string())),
    }
}

fn fn_strlang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query("STRLANG requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::Literal(lang_lit)) => {
            Ok(Term::Literal(Literal::new_lang(value_lit.value(), lang_lit.value())?))
        }
        _ => Err(OxirsError::Query("STRLANG requires two string literals".to_string())),
    }
}

fn fn_uuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    let iri = NamedNode::new(format!("urn:uuid:{}", uuid))?;
    Ok(Term::NamedNode(iri))
}

fn fn_struuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    Ok(Term::Literal(Literal::new(&uuid.to_string())))
}

// Aggregate functions (placeholder implementations)

fn fn_count(_args: &[Term]) -> Result<Term, OxirsError> {
    // Aggregate functions need special handling in query evaluation
    Err(OxirsError::Query("COUNT is an aggregate function".to_string()))
}

fn fn_sum(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("SUM is an aggregate function".to_string()))
}

fn fn_avg(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("AVG is an aggregate function".to_string()))
}

fn fn_min(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("MIN is an aggregate function".to_string()))
}

fn fn_max(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("MAX is an aggregate function".to_string()))
}

fn fn_group_concat(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("GROUP_CONCAT is an aggregate function".to_string()))
}

fn fn_sample(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query("SAMPLE is an aggregate function".to_string()))
}

// Boolean functions

fn fn_not(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("NOT requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => Err(OxirsError::Query("NOT requires boolean literal".to_string())),
    }
}

fn fn_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // EXISTS needs special handling in query evaluation
    Err(OxirsError::Query("EXISTS requires graph pattern context".to_string()))
}

fn fn_not_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // NOT EXISTS needs special handling in query evaluation
    Err(OxirsError::Query("NOT EXISTS requires graph pattern context".to_string()))
}

fn fn_bound(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query("BOUND requires exactly 1 argument".to_string()));
    }
    
    let is_bound = !matches!(&args[0], Term::Variable(_));
    Ok(Term::Literal(Literal::new_typed(
        if is_bound { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
    )))
}

fn fn_coalesce(args: &[Term]) -> Result<Term, OxirsError> {
    for arg in args {
        if !matches!(arg, Term::Variable(_)) {
            return Ok(arg.clone());
        }
    }
    Err(OxirsError::Query("COALESCE: all arguments are unbound".to_string()))
}

fn fn_if(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 3 {
        return Err(OxirsError::Query("IF requires exactly 3 arguments".to_string()));
    }
    
    match &args[0] {
        Term::Literal(condition) => {
            let is_true = condition.value() == "true";
            Ok(if is_true { args[1].clone() } else { args[2].clone() })
        }
        _ => Err(OxirsError::Query("IF condition must be boolean".to_string())),
    }
}

// List functions

fn fn_in(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 {
        return Err(OxirsError::Query("IN requires at least 2 arguments".to_string()));
    }
    
    let value = &args[0];
    for item in &args[1..] {
        if value == item {
            return Ok(Term::Literal(Literal::new_typed(
                "true",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )));
        }
    }
    
    Ok(Term::Literal(Literal::new_typed(
        "false",
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
    )))
}

fn fn_not_in(args: &[Term]) -> Result<Term, OxirsError> {
    match fn_in(args)? {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap()
            )))
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_functions() {
        let registry = FunctionRegistry::new();
        
        // Test CONCAT
        let args = vec![
            Term::Literal(Literal::new("Hello")),
            Term::Literal(Literal::new(" ")),
            Term::Literal(Literal::new("World")),
        ];
        let result = registry.execute("CONCAT", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "Hello World"),
            _ => panic!("Expected literal"),
        }
        
        // Test STRLEN
        let args = vec![Term::Literal(Literal::new("Hello"))];
        let result = registry.execute("STRLEN", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "5"),
            _ => panic!("Expected literal"),
        }
        
        // Test UCASE
        let args = vec![Term::Literal(Literal::new("hello"))];
        let result = registry.execute("UCASE", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "HELLO"),
            _ => panic!("Expected literal"),
        }
    }
    
    #[test]
    fn test_numeric_functions() {
        let registry = FunctionRegistry::new();
        
        // Test ABS
        let args = vec![Term::Literal(Literal::new_typed("-42", NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()))];
        let result = registry.execute("ABS", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "42"),
            _ => panic!("Expected literal"),
        }
        
        // Test SQRT
        let args = vec![Term::Literal(Literal::new_typed("9", NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()))];
        let result = registry.execute("SQRT", &args).unwrap();
        match result {
            Term::Literal(lit) => {
                let value: f64 = lit.value().parse().unwrap();
                assert!((value - 3.0).abs() < 0.0001);
            }
            _ => panic!("Expected literal"),
        }
    }
    
    #[test]
    fn test_hash_functions() {
        let registry = FunctionRegistry::new();
        
        // Test SHA256
        let args = vec![Term::Literal(Literal::new("test"))];
        let result = registry.execute("SHA256", &args).unwrap();
        match result {
            Term::Literal(lit) => {
                assert_eq!(lit.value(), "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");
            }
            _ => panic!("Expected literal"),
        }
    }
}