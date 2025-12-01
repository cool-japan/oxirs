//! SPARQL 1.2 built-in functions and extensions
//!
//! This module implements the extended function library for SPARQL 1.2,
//! including new string functions, math functions, and advanced features.

use crate::model::{Literal, NamedNode, Term};
// use crate::query::algebra::Expression; // For future expression evaluation
use crate::OxirsError;
use chrono::{DateTime, Datelike, Timelike, Utc};
use regex::Regex;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use std::collections::HashMap;
use std::sync::Arc;

/// SPARQL function registry with built-in performance monitoring
pub struct FunctionRegistry {
    /// Built-in functions
    functions: HashMap<String, FunctionImpl>,
    /// Custom extension functions
    extensions: HashMap<String, Arc<dyn CustomFunction>>,
    /// Function execution counter (tracks calls per function)
    execution_counter: Arc<Counter>,
    /// Function execution timer (tracks execution time)
    execution_timer: Arc<Timer>,
    /// Function error counter
    error_counter: Arc<Counter>,
    /// Function execution time histogram
    execution_histogram: Arc<Histogram>,
    /// Metrics registry for global tracking
    metrics_registry: Arc<MetricsRegistry>,
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

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionRegistry {
    /// Create new function registry with SPARQL 1.2 built-ins and performance monitoring
    pub fn new() -> Self {
        let metrics_registry = Arc::new(MetricsRegistry::new());

        let execution_counter = Arc::new(Counter::new("function_executions".to_string()));
        let execution_timer = Arc::new(Timer::new("function_duration".to_string()));
        let error_counter = Arc::new(Counter::new("function_errors".to_string()));
        let execution_histogram = Arc::new(Histogram::new("function_duration_dist".to_string()));

        let mut registry = FunctionRegistry {
            functions: HashMap::new(),
            extensions: HashMap::new(),
            execution_counter,
            execution_timer,
            error_counter,
            execution_histogram,
            metrics_registry,
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

        // SPARQL 1.2 additional string functions
        self.register_native("CONCAT_WS", Arc::new(fn_concat_ws));
        self.register_native("SPLIT", Arc::new(fn_split));
        self.register_native("LPAD", Arc::new(fn_lpad));
        self.register_native("RPAD", Arc::new(fn_rpad));

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
        self.register_native("ADJUST", Arc::new(fn_adjust));

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

        // Type checking functions (SPARQL 1.1)
        self.register_native("isIRI", Arc::new(fn_is_iri));
        self.register_native("isURI", Arc::new(fn_is_iri));
        self.register_native("isBLANK", Arc::new(fn_is_blank));
        self.register_native("isLITERAL", Arc::new(fn_is_literal));
        self.register_native("isNUMERIC", Arc::new(fn_is_numeric));
        self.register_native("sameTerm", Arc::new(fn_same_term));
        self.register_native("LANGMATCHES", Arc::new(fn_langmatches));

        // List functions (SPARQL 1.2)
        self.register_native("IN", Arc::new(fn_in));
        self.register_native("NOT_IN", Arc::new(fn_not_in));
    }

    /// Register a native function
    fn register_native(&mut self, name: &str, func: NativeFunction) {
        self.functions
            .insert(name.to_string(), FunctionImpl::Native(func));
    }

    /// Register a custom function
    pub fn register_custom(&mut self, func: Arc<dyn CustomFunction>) {
        let metadata = func.metadata();
        self.extensions.insert(metadata.name.clone(), func);
    }

    /// Execute a function with automatic performance monitoring
    pub fn execute(&self, name: &str, args: &[Term]) -> Result<Term, OxirsError> {
        // Start timing
        let start = std::time::Instant::now();

        // Increment execution counter
        self.execution_counter.inc();

        // Execute function
        let result = if let Some(func) = self.functions.get(name) {
            match func {
                FunctionImpl::Native(f) => f(args),
                FunctionImpl::JavaScript(_) => Err(OxirsError::Query(
                    "JavaScript functions not yet implemented".to_string(),
                )),
                FunctionImpl::Wasm(_) => Err(OxirsError::Query(
                    "WASM functions not yet implemented".to_string(),
                )),
            }
        }
        // Check custom functions
        else if let Some(func) = self.extensions.get(name) {
            func.execute(args)
        } else {
            Err(OxirsError::Query(format!("Unknown function: {name}")))
        };

        // Record execution time
        let duration = start.elapsed();
        self.execution_timer.observe(duration);
        self.execution_histogram
            .observe(duration.as_micros() as f64);

        // Track errors
        if result.is_err() {
            self.error_counter.inc();
        }

        result
    }

    /// Get function execution statistics
    pub fn get_statistics(&self) -> FunctionStatistics {
        let timer_stats = self.execution_timer.get_stats();

        FunctionStatistics {
            total_executions: self.execution_counter.get(),
            total_errors: self.error_counter.get(),
            average_duration_micros: timer_stats.mean * 1_000_000.0, // Convert to microseconds
            // Note: SCIRS2 Histogram doesn't currently expose percentiles
            p95_duration_micros: timer_stats.mean * 1_000_000.0, // Use mean as approximation
            p99_duration_micros: timer_stats.mean * 1_000_000.0, // Use mean as approximation
        }
    }

    /// Get metrics registry for external monitoring systems
    pub fn metrics_registry(&self) -> &Arc<MetricsRegistry> {
        &self.metrics_registry
    }
}

/// Function execution statistics
#[derive(Debug, Clone)]
pub struct FunctionStatistics {
    /// Total number of function executions
    pub total_executions: u64,
    /// Total number of function errors
    pub total_errors: u64,
    /// Average execution duration in microseconds
    pub average_duration_micros: f64,
    /// 95th percentile execution duration in microseconds
    pub p95_duration_micros: f64,
    /// 99th percentile execution duration in microseconds
    pub p99_duration_micros: f64,
}

impl std::fmt::Display for FunctionStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FunctionStats {{ executions: {}, errors: {}, avg: {:.2}μs, p95: {:.2}μs, p99: {:.2}μs, error_rate: {:.2}% }}",
            self.total_executions,
            self.total_errors,
            self.average_duration_micros,
            self.p95_duration_micros,
            self.p99_duration_micros,
            if self.total_executions > 0 {
                (self.total_errors as f64 / self.total_executions as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

// String functions implementation

fn fn_concat(args: &[Term]) -> Result<Term, OxirsError> {
    let mut result = String::new();

    for arg in args {
        match arg {
            Term::Literal(lit) => result.push_str(lit.value()),
            Term::NamedNode(nn) => result.push_str(nn.as_str()),
            _ => {
                return Err(OxirsError::Query(
                    "CONCAT requires string arguments".to_string(),
                ))
            }
        }
    }

    Ok(Term::Literal(Literal::new(&result)))
}

fn fn_strlen(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "STRLEN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let len = lit.value().chars().count() as i64;
            Ok(Term::Literal(Literal::new_typed(
                len.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRLEN requires string literal".to_string(),
        )),
    }
}

fn fn_substr(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "SUBSTR requires 2 or 3 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(start_lit)) => {
            let string = str_lit.value();
            let start = start_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("Invalid start position".to_string()))?;

            let result = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(len_lit) => {
                        let len = len_lit
                            .value()
                            .parse::<usize>()
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
        _ => Err(OxirsError::Query(
            "SUBSTR requires string and numeric arguments".to_string(),
        )),
    }
}

fn fn_replace(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 3 || args.len() > 4 {
        return Err(OxirsError::Query(
            "REPLACE requires 3 or 4 arguments".to_string(),
        ));
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
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {e}")))?;

            let result = regex.replace_all(text.value(), replacement.value());
            Ok(Term::Literal(Literal::new(result.as_ref())))
        }
        _ => Err(OxirsError::Query(
            "REPLACE requires string arguments".to_string(),
        )),
    }
}

fn fn_regex(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "REGEX requires 2 or 3 arguments".to_string(),
        ));
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
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {e}")))?;

            let matches = regex.is_match(text.value());
            Ok(Term::Literal(Literal::new_typed(
                if matches { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "REGEX requires string arguments".to_string(),
        )),
    }
}

// String manipulation functions

fn fn_strafter(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRAFTER requires exactly 2 arguments".to_string(),
        ));
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
        _ => Err(OxirsError::Query(
            "STRAFTER requires string arguments".to_string(),
        )),
    }
}

fn fn_strbefore(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRBEFORE requires exactly 2 arguments".to_string(),
        ));
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
        _ => Err(OxirsError::Query(
            "STRBEFORE requires string arguments".to_string(),
        )),
    }
}

fn fn_strstarts(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRSTARTS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(prefix_lit)) => {
            let result = str_lit.value().starts_with(prefix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRSTARTS requires string arguments".to_string(),
        )),
    }
}

fn fn_strends(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRENDS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(suffix_lit)) => {
            let result = str_lit.value().ends_with(suffix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRENDS requires string arguments".to_string(),
        )),
    }
}

fn fn_contains(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "CONTAINS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(substr_lit)) => {
            let result = str_lit.value().contains(substr_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "CONTAINS requires string arguments".to_string(),
        )),
    }
}

fn fn_encode_for_uri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ENCODE_FOR_URI requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let encoded = urlencoding::encode(lit.value());
            Ok(Term::Literal(Literal::new(encoded.as_ref())))
        }
        _ => Err(OxirsError::Query(
            "ENCODE_FOR_URI requires string argument".to_string(),
        )),
    }
}

// Case functions

fn fn_ucase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "UCASE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value().to_uppercase()))),
        _ => Err(OxirsError::Query(
            "UCASE requires string argument".to_string(),
        )),
    }
}

fn fn_lcase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LCASE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value().to_lowercase()))),
        _ => Err(OxirsError::Query(
            "LCASE requires string argument".to_string(),
        )),
    }
}

// SPARQL 1.2 Additional String Functions

/// CONCAT_WS - Concatenate with separator
/// CONCAT_WS(separator, str1, str2, ...) joins strings with a separator
fn fn_concat_ws(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 {
        return Err(OxirsError::Query(
            "CONCAT_WS requires at least 2 arguments (separator and at least one string)"
                .to_string(),
        ));
    }

    // First argument is the separator
    let separator = match &args[0] {
        Term::Literal(lit) => lit.value(),
        _ => {
            return Err(OxirsError::Query(
                "CONCAT_WS separator must be a string literal".to_string(),
            ))
        }
    };

    // Remaining arguments are strings to concatenate
    let strings: Result<Vec<&str>, OxirsError> = args[1..]
        .iter()
        .map(|arg| match arg {
            Term::Literal(lit) => Ok(lit.value()),
            Term::NamedNode(nn) => Ok(nn.as_str()),
            _ => Err(OxirsError::Query(
                "CONCAT_WS requires string arguments".to_string(),
            )),
        })
        .collect();

    let result = strings?.join(separator);
    Ok(Term::Literal(Literal::new(&result)))
}

/// SPLIT - Split string by delimiter into multiple strings
/// Returns a concatenated result with | separator for now (SPARQL doesn't have native arrays)
fn fn_split(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "SPLIT requires exactly 2 arguments (string and delimiter)".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(delimiter)) => {
            let parts: Vec<&str> = text.value().split(delimiter.value()).collect();
            // Since SPARQL doesn't have native array return type, we return JSON array as string
            let result = format!(
                "[{}]",
                parts
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "SPLIT requires string arguments".to_string(),
        )),
    }
}

/// LPAD - Left pad string to specified length
/// LPAD(string, length, padString) pads the left side of string
fn fn_lpad(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "LPAD requires 2 or 3 arguments (string, length, [padString])".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(length_lit)) => {
            let target_length = length_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("LPAD length must be numeric".to_string()))?;

            let pad_string = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(pad) => pad.value(),
                    _ => {
                        return Err(OxirsError::Query(
                            "LPAD pad string must be a string literal".to_string(),
                        ))
                    }
                }
            } else {
                " " // Default to space
            };

            let text_value = text.value();
            let current_length = text_value.chars().count();

            let result = if current_length >= target_length {
                text_value.to_string()
            } else {
                let pad_length = target_length - current_length;
                let pad_chars: Vec<char> = pad_string.chars().collect();
                if pad_chars.is_empty() {
                    return Err(OxirsError::Query(
                        "LPAD pad string cannot be empty".to_string(),
                    ));
                }

                let mut padding = String::new();
                for i in 0..pad_length {
                    padding.push(pad_chars[i % pad_chars.len()]);
                }
                format!("{}{}", padding, text_value)
            };

            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "LPAD requires string and numeric arguments".to_string(),
        )),
    }
}

/// RPAD - Right pad string to specified length
/// RPAD(string, length, padString) pads the right side of string
fn fn_rpad(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "RPAD requires 2 or 3 arguments (string, length, [padString])".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(length_lit)) => {
            let target_length = length_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("RPAD length must be numeric".to_string()))?;

            let pad_string = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(pad) => pad.value(),
                    _ => {
                        return Err(OxirsError::Query(
                            "RPAD pad string must be a string literal".to_string(),
                        ))
                    }
                }
            } else {
                " " // Default to space
            };

            let text_value = text.value();
            let current_length = text_value.chars().count();

            let result = if current_length >= target_length {
                text_value.to_string()
            } else {
                let pad_length = target_length - current_length;
                let pad_chars: Vec<char> = pad_string.chars().collect();
                if pad_chars.is_empty() {
                    return Err(OxirsError::Query(
                        "RPAD pad string cannot be empty".to_string(),
                    ));
                }

                let mut padding = String::new();
                for i in 0..pad_length {
                    padding.push(pad_chars[i % pad_chars.len()]);
                }
                format!("{}{}", text_value, padding)
            };

            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "RPAD requires string and numeric arguments".to_string(),
        )),
    }
}

// Numeric functions

fn fn_abs(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ABS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ABS requires numeric argument".to_string()))?;
            let result = value.abs();

            // Preserve datatype
            let dt = lit.datatype();
            if dt.as_str() == "http://www.w3.org/2001/XMLSchema#integer" {
                Ok(Term::Literal(Literal::new_typed(
                    (result as i64).to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
                )))
            } else {
                Ok(Term::Literal(Literal::new_typed(
                    result.to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
                )))
            }
        }
        _ => Err(OxirsError::Query(
            "ABS requires numeric literal".to_string(),
        )),
    }
}

fn fn_ceil(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "CEIL requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("CEIL requires numeric argument".to_string()))?;
            let result = value.ceil() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "CEIL requires numeric literal".to_string(),
        )),
    }
}

fn fn_floor(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "FLOOR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("FLOOR requires numeric argument".to_string()))?;
            let result = value.floor() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "FLOOR requires numeric literal".to_string(),
        )),
    }
}

fn fn_round(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ROUND requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ROUND requires numeric argument".to_string()))?;
            let result = value.round() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "ROUND requires numeric literal".to_string(),
        )),
    }
}

fn fn_rand(_args: &[Term]) -> Result<Term, OxirsError> {
    use scirs2_core::random::{Random, Rng};
    let mut random = Random::default();
    let value: f64 = random.random();
    Ok(Term::Literal(Literal::new_typed(
        value.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
    )))
}

// Math functions (SPARQL 1.2)

fn fn_sqrt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SQRT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SQRT requires numeric argument".to_string()))?;
            if value < 0.0 {
                return Err(OxirsError::Query("SQRT of negative number".to_string()));
            }
            let result = value.sqrt();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "SQRT requires numeric literal".to_string(),
        )),
    }
}

fn fn_sin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SIN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SIN requires numeric argument".to_string()))?;
            let result = value.sin();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "SIN requires numeric literal".to_string(),
        )),
    }
}

fn fn_cos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "COS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("COS requires numeric argument".to_string()))?;
            let result = value.cos();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "COS requires numeric literal".to_string(),
        )),
    }
}

fn fn_tan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TAN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("TAN requires numeric argument".to_string()))?;
            let result = value.tan();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "TAN requires numeric literal".to_string(),
        )),
    }
}

fn fn_asin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ASIN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ASIN requires numeric argument".to_string()))?;
            if !(-1.0..=1.0).contains(&value) {
                return Err(OxirsError::Query(
                    "ASIN argument must be between -1 and 1".to_string(),
                ));
            }
            let result = value.asin();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "ASIN requires numeric literal".to_string(),
        )),
    }
}

fn fn_acos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ACOS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ACOS requires numeric argument".to_string()))?;
            if !(-1.0..=1.0).contains(&value) {
                return Err(OxirsError::Query(
                    "ACOS argument must be between -1 and 1".to_string(),
                ));
            }
            let result = value.acos();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "ACOS requires numeric literal".to_string(),
        )),
    }
}

fn fn_atan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ATAN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN requires numeric argument".to_string()))?;
            let result = value.atan();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "ATAN requires numeric literal".to_string(),
        )),
    }
}

fn fn_atan2(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "ATAN2 requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(y_lit), Term::Literal(x_lit)) => {
            let y = y_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let x = x_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let result = y.atan2(x);
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "ATAN2 requires numeric literals".to_string(),
        )),
    }
}

fn fn_exp(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "EXP requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("EXP requires numeric argument".to_string()))?;
            let result = value.exp();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "EXP requires numeric literal".to_string(),
        )),
    }
}

fn fn_log(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LOG requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query("LOG of non-positive number".to_string()));
            }
            let result = value.ln();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "LOG requires numeric literal".to_string(),
        )),
    }
}

fn fn_log10(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LOG10 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG10 requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query(
                    "LOG10 of non-positive number".to_string(),
                ));
            }
            let result = value.log10();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "LOG10 requires numeric literal".to_string(),
        )),
    }
}

fn fn_pow(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "POW requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(base_lit), Term::Literal(exp_lit)) => {
            let base = base_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let exp = exp_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let result = base.powf(exp);
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "POW requires numeric literals".to_string(),
        )),
    }
}

// Date/time functions

fn fn_now(_args: &[Term]) -> Result<Term, OxirsError> {
    let now = Utc::now();
    Ok(Term::Literal(Literal::new_typed(
        now.to_rfc3339(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#dateTime").unwrap(),
    )))
}

fn fn_year(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "YEAR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.year().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "YEAR requires dateTime literal".to_string(),
        )),
    }
}

fn fn_month(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MONTH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.month().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "MONTH requires dateTime literal".to_string(),
        )),
    }
}

fn fn_day(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "DAY requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.day().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "DAY requires dateTime literal".to_string(),
        )),
    }
}

fn fn_hours(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "HOURS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.hour().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "HOURS requires dateTime literal".to_string(),
        )),
    }
}

fn fn_minutes(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MINUTES requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.minute().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "MINUTES requires dateTime literal".to_string(),
        )),
    }
}

fn fn_seconds(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SECONDS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                format!("{}.{:09}", dt.second(), dt.nanosecond()),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "SECONDS requires dateTime literal".to_string(),
        )),
    }
}

fn fn_timezone(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TIMEZONE requires exactly 1 argument".to_string(),
        ));
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
                NamedNode::new("http://www.w3.org/2001/XMLSchema#dayTimeDuration").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "TIMEZONE requires dateTime literal".to_string(),
        )),
    }
}

fn fn_tz(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TZ requires exactly 1 argument".to_string(),
        ));
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
        _ => Err(OxirsError::Query(
            "TZ requires dateTime literal".to_string(),
        )),
    }
}

// Hash functions

fn fn_sha1(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA1 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha1::{Digest, Sha1};
            let mut hasher = Sha1::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA1 requires string literal".to_string(),
        )),
    }
}

fn fn_sha256(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA256 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA256 requires string literal".to_string(),
        )),
    }
}

fn fn_sha384(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA384 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha384};
            let mut hasher = Sha384::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA384 requires string literal".to_string(),
        )),
    }
}

fn fn_sha512(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA512 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha512};
            let mut hasher = Sha512::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA512 requires string literal".to_string(),
        )),
    }
}

fn fn_md5(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MD5 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let mut hasher = md5::Context::new();
            hasher.consume(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("MD5 requires string literal".to_string())),
    }
}

// Type conversion functions

fn fn_str(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "STR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value()))),
        Term::NamedNode(nn) => Ok(Term::Literal(Literal::new(nn.as_str()))),
        _ => Err(OxirsError::Query("STR requires literal or IRI".to_string())),
    }
}

fn fn_lang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LANG requires exactly 1 argument".to_string(),
        ));
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
        return Err(OxirsError::Query(
            "DATATYPE requires exactly 1 argument".to_string(),
        ));
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
        return Err(OxirsError::Query(
            "IRI requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let iri = NamedNode::new(lit.value())?;
            Ok(Term::NamedNode(iri))
        }
        Term::NamedNode(nn) => Ok(Term::NamedNode(nn.clone())),
        _ => Err(OxirsError::Query(
            "IRI requires string literal or IRI".to_string(),
        )),
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
            _ => Err(OxirsError::Query(
                "BNODE requires string literal or no arguments".to_string(),
            )),
        }
    } else {
        Err(OxirsError::Query(
            "BNODE requires 0 or 1 arguments".to_string(),
        ))
    }
}

fn fn_strdt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRDT requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::NamedNode(datatype)) => Ok(Term::Literal(
            Literal::new_typed(value_lit.value(), datatype.clone()),
        )),
        _ => Err(OxirsError::Query(
            "STRDT requires string literal and IRI".to_string(),
        )),
    }
}

fn fn_strlang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRLANG requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::Literal(lang_lit)) => Ok(Term::Literal(
            Literal::new_lang(value_lit.value(), lang_lit.value())?,
        )),
        _ => Err(OxirsError::Query(
            "STRLANG requires two string literals".to_string(),
        )),
    }
}

fn fn_uuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    let iri = NamedNode::new(format!("urn:uuid:{uuid}"))?;
    Ok(Term::NamedNode(iri))
}

fn fn_struuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    Ok(Term::Literal(Literal::new(uuid.to_string())))
}

// Aggregate functions (placeholder implementations)

fn fn_count(_args: &[Term]) -> Result<Term, OxirsError> {
    // Aggregate functions need special handling in query evaluation
    Err(OxirsError::Query(
        "COUNT is an aggregate function".to_string(),
    ))
}

fn fn_sum(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "SUM is an aggregate function".to_string(),
    ))
}

fn fn_avg(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "AVG is an aggregate function".to_string(),
    ))
}

fn fn_min(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "MIN is an aggregate function".to_string(),
    ))
}

fn fn_max(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "MAX is an aggregate function".to_string(),
    ))
}

fn fn_group_concat(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "GROUP_CONCAT is an aggregate function".to_string(),
    ))
}

fn fn_sample(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "SAMPLE is an aggregate function".to_string(),
    ))
}

// Boolean functions

fn fn_not(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "NOT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "NOT requires boolean literal".to_string(),
        )),
    }
}

fn fn_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // EXISTS needs special handling in query evaluation
    Err(OxirsError::Query(
        "EXISTS requires graph pattern context".to_string(),
    ))
}

fn fn_not_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // NOT EXISTS needs special handling in query evaluation
    Err(OxirsError::Query(
        "NOT EXISTS requires graph pattern context".to_string(),
    ))
}

fn fn_bound(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "BOUND requires exactly 1 argument".to_string(),
        ));
    }

    let is_bound = !matches!(&args[0], Term::Variable(_));
    Ok(Term::Literal(Literal::new_typed(
        if is_bound { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

fn fn_coalesce(args: &[Term]) -> Result<Term, OxirsError> {
    for arg in args {
        if !matches!(arg, Term::Variable(_)) {
            return Ok(arg.clone());
        }
    }
    Err(OxirsError::Query(
        "COALESCE: all arguments are unbound".to_string(),
    ))
}

fn fn_if(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 3 {
        return Err(OxirsError::Query(
            "IF requires exactly 3 arguments".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(condition) => {
            let is_true = condition.value() == "true";
            Ok(if is_true {
                args[1].clone()
            } else {
                args[2].clone()
            })
        }
        _ => Err(OxirsError::Query(
            "IF condition must be boolean".to_string(),
        )),
    }
}

// Type checking functions

#[allow(dead_code)]
fn fn_is_iri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isIRI requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::NamedNode(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

#[allow(dead_code)]
fn fn_is_blank(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isBLANK requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::BlankNode(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

#[allow(dead_code)]
fn fn_is_literal(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isLITERAL requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::Literal(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

#[allow(dead_code)]
fn fn_is_numeric(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isNUMERIC requires exactly 1 argument".to_string(),
        ));
    }

    let result = match &args[0] {
        Term::Literal(lit) => {
            let dt = lit.datatype();
            dt.as_str() == "http://www.w3.org/2001/XMLSchema#integer"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#float"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#double"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#negativeInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#long"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#int"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#short"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#byte"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedLong"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedInt"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedShort"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedByte"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#positiveInteger"
        }
        _ => false,
    };

    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

#[allow(dead_code)]
fn fn_same_term(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "sameTerm requires exactly 2 arguments".to_string(),
        ));
    }

    let result = args[0] == args[1];
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

#[allow(dead_code)]
fn fn_langmatches(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "LANGMATCHES requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lang_tag), Term::Literal(lang_range)) => {
            let tag = lang_tag.value().to_lowercase();
            let range = lang_range.value().to_lowercase();

            let result = if range == "*" {
                !tag.is_empty()
            } else {
                tag == range || tag.starts_with(&format!("{}-", range))
            };

            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "LANGMATCHES requires two string literals".to_string(),
        )),
    }
}

// Date/time adjustment function

#[allow(dead_code)]
fn fn_adjust(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "ADJUST requires exactly 2 arguments".to_string(),
        ));
    }

    // For now, just return the first argument (datetime value)
    // A full implementation would apply timezone adjustment
    Ok(args[0].clone())
}

// List functions

fn fn_in(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 {
        return Err(OxirsError::Query(
            "IN requires at least 2 arguments".to_string(),
        ));
    }

    let value = &args[0];
    for item in &args[1..] {
        if value == item {
            return Ok(Term::Literal(Literal::new_typed(
                "true",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )));
        }
    }

    Ok(Term::Literal(Literal::new_typed(
        "false",
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

fn fn_not_in(args: &[Term]) -> Result<Term, OxirsError> {
    match fn_in(args)? {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
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
        let args = vec![Term::Literal(Literal::new_typed(
            "-42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        ))];
        let result = registry.execute("ABS", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "42"),
            _ => panic!("Expected literal"),
        }

        // Test SQRT
        let args = vec![Term::Literal(Literal::new_typed(
            "9",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap(),
        ))];
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
                assert_eq!(
                    lit.value(),
                    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
                );
            }
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_concat_ws_function() {
        let registry = FunctionRegistry::new();

        // Test CONCAT_WS with comma separator
        let args = vec![
            Term::Literal(Literal::new(",")),
            Term::Literal(Literal::new("apple")),
            Term::Literal(Literal::new("banana")),
            Term::Literal(Literal::new("cherry")),
        ];
        let result = registry.execute("CONCAT_WS", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "apple,banana,cherry"),
            _ => panic!("Expected literal"),
        }

        // Test CONCAT_WS with space separator
        let args = vec![
            Term::Literal(Literal::new(" ")),
            Term::Literal(Literal::new("Hello")),
            Term::Literal(Literal::new("SPARQL")),
            Term::Literal(Literal::new("World")),
        ];
        let result = registry.execute("CONCAT_WS", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "Hello SPARQL World"),
            _ => panic!("Expected literal"),
        }

        // Test CONCAT_WS with single value
        let args = vec![
            Term::Literal(Literal::new(",")),
            Term::Literal(Literal::new("single")),
        ];
        let result = registry.execute("CONCAT_WS", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "single"),
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_split_function() {
        let registry = FunctionRegistry::new();

        // Test SPLIT with comma delimiter
        let args = vec![
            Term::Literal(Literal::new("apple,banana,cherry")),
            Term::Literal(Literal::new(",")),
        ];
        let result = registry.execute("SPLIT", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "[\"apple\",\"banana\",\"cherry\"]"),
            _ => panic!("Expected literal"),
        }

        // Test SPLIT with space delimiter
        let args = vec![
            Term::Literal(Literal::new("Hello SPARQL World")),
            Term::Literal(Literal::new(" ")),
        ];
        let result = registry.execute("SPLIT", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "[\"Hello\",\"SPARQL\",\"World\"]"),
            _ => panic!("Expected literal"),
        }

        // Test SPLIT with no delimiter found
        let args = vec![
            Term::Literal(Literal::new("nodeLimiter")),
            Term::Literal(Literal::new(",")),
        ];
        let result = registry.execute("SPLIT", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "[\"nodeLimiter\"]"),
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_lpad_function() {
        let registry = FunctionRegistry::new();

        // Test LPAD with default space padding
        let args = vec![
            Term::Literal(Literal::new("123")),
            Term::Literal(Literal::new_typed(
                "5",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ];
        let result = registry.execute("LPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "  123"),
            _ => panic!("Expected literal"),
        }

        // Test LPAD with custom padding
        let args = vec![
            Term::Literal(Literal::new("test")),
            Term::Literal(Literal::new_typed(
                "10",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
            Term::Literal(Literal::new("*")),
        ];
        let result = registry.execute("LPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "******test"),
            _ => panic!("Expected literal"),
        }

        // Test LPAD with repeating pattern
        let args = vec![
            Term::Literal(Literal::new("X")),
            Term::Literal(Literal::new_typed(
                "5",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
            Term::Literal(Literal::new("ab")),
        ];
        let result = registry.execute("LPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "ababX"),
            _ => panic!("Expected literal"),
        }

        // Test LPAD when string is already longer
        let args = vec![
            Term::Literal(Literal::new("toolong")),
            Term::Literal(Literal::new_typed(
                "3",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ];
        let result = registry.execute("LPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "toolong"),
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_function_metrics() {
        let registry = FunctionRegistry::new();

        // Execute some functions
        let args = vec![Term::Literal(Literal::new("test"))];
        let _ = registry.execute("STRLEN", &args);
        let _ = registry.execute("UCASE", &args);
        let _ = registry.execute("STRLEN", &args);

        // Try an error case
        let bad_args = vec![Term::Literal(Literal::new("bad"))];
        let _ = registry.execute("UNKNOWN_FUNCTION", &bad_args);

        // Get statistics
        let stats = registry.get_statistics();

        // Verify metrics
        assert_eq!(stats.total_executions, 4); // 3 successful + 1 error
        assert_eq!(stats.total_errors, 1); // 1 unknown function error
        assert!(stats.average_duration_micros >= 0.0);
        assert!(stats.p95_duration_micros >= 0.0);
        assert!(stats.p99_duration_micros >= 0.0);

        // Verify display
        let display = format!("{}", stats);
        assert!(display.contains("executions: 4"));
        assert!(display.contains("errors: 1"));

        // Verify metrics registry exists
        let _metrics = registry.metrics_registry();
        // Basic smoke test - metrics registry is created
        assert!(stats.total_executions > 0);
    }

    #[test]
    fn test_rpad_function() {
        let registry = FunctionRegistry::new();

        // Test RPAD with default space padding
        let args = vec![
            Term::Literal(Literal::new("123")),
            Term::Literal(Literal::new_typed(
                "5",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ];
        let result = registry.execute("RPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "123  "),
            _ => panic!("Expected literal"),
        }

        // Test RPAD with custom padding
        let args = vec![
            Term::Literal(Literal::new("test")),
            Term::Literal(Literal::new_typed(
                "10",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
            Term::Literal(Literal::new("*")),
        ];
        let result = registry.execute("RPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "test******"),
            _ => panic!("Expected literal"),
        }

        // Test RPAD with repeating pattern
        let args = vec![
            Term::Literal(Literal::new("X")),
            Term::Literal(Literal::new_typed(
                "5",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
            Term::Literal(Literal::new("ab")),
        ];
        let result = registry.execute("RPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "Xabab"),
            _ => panic!("Expected literal"),
        }

        // Test RPAD when string is already longer
        let args = vec![
            Term::Literal(Literal::new("toolong")),
            Term::Literal(Literal::new_typed(
                "3",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ];
        let result = registry.execute("RPAD", &args).unwrap();
        match result {
            Term::Literal(lit) => assert_eq!(lit.value(), "toolong"),
            _ => panic!("Expected literal"),
        }
    }
}
