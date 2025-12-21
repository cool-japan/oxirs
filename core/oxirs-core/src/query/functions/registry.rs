//! Function registry and core types for SPARQL functions

use crate::model::Term;
use crate::OxirsError;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use std::collections::HashMap;
use std::sync::Arc;

// Import all function implementations
use super::aggregate::*;
use super::bitwise::*;
use super::datetime::*;
use super::hash::*;
use super::numeric::*;
use super::string::*;
use super::type_check::*;

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

        // Advanced string utility functions
        self.register_native("TRIM", Arc::new(fn_trim));
        self.register_native("LTRIM", Arc::new(fn_ltrim));
        self.register_native("RTRIM", Arc::new(fn_rtrim));
        self.register_native("REVERSE", Arc::new(fn_reverse));
        self.register_native("REPEAT", Arc::new(fn_repeat));

        // String inspection functions (SPARQL Extension)
        self.register_native("CAPITALIZE", Arc::new(fn_capitalize));
        self.register_native("ISALPHA", Arc::new(fn_isalpha));
        self.register_native("ISDIGIT", Arc::new(fn_isdigit));
        self.register_native("ISALNUM", Arc::new(fn_isalnum));
        self.register_native("ISWHITESPACE", Arc::new(fn_iswhitespace));

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

        // Hyperbolic functions (SPARQL Extension)
        self.register_native("SINH", Arc::new(fn_sinh));
        self.register_native("COSH", Arc::new(fn_cosh));
        self.register_native("TANH", Arc::new(fn_tanh));
        self.register_native("ASINH", Arc::new(fn_asinh));
        self.register_native("ACOSH", Arc::new(fn_acosh));
        self.register_native("ATANH", Arc::new(fn_atanh));

        // Mathematical constants (SPARQL Extension)
        self.register_native("PI", Arc::new(fn_pi));
        self.register_native("E", Arc::new(fn_e));
        self.register_native("TAU", Arc::new(fn_tau));

        // Advanced numeric utility functions
        self.register_native("SIGN", Arc::new(fn_sign));
        self.register_native("MOD", Arc::new(fn_mod));
        self.register_native("TRUNC", Arc::new(fn_trunc));
        self.register_native("GCD", Arc::new(fn_gcd));
        self.register_native("LCM", Arc::new(fn_lcm));

        // Bitwise operations (SPARQL Extension)
        self.register_native("BITAND", Arc::new(fn_bitand));
        self.register_native("BITOR", Arc::new(fn_bitor));
        self.register_native("BITXOR", Arc::new(fn_bitxor));
        self.register_native("BITNOT", Arc::new(fn_bitnot));
        self.register_native("LSHIFT", Arc::new(fn_lshift));
        self.register_native("RSHIFT", Arc::new(fn_rshift));

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
