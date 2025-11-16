//! SWRL (Semantic Web Rule Language) - SWRL Engine
//!
//! This module implements SWRL rule components.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::builtins::*;
use super::stats::SwrlStats;
use super::types::*;
use super::vocabulary;

pub struct SwrlEngine {
    /// SWRL rules
    rules: Vec<SwrlRule>,
    /// Built-in functions registry
    builtins: HashMap<String, BuiltinFunction>,
    /// Core rule engine for basic reasoning
    rule_engine: RuleEngine,
}

impl Default for SwrlEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SwrlEngine {
    /// Create a new SWRL engine
    pub fn new() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            builtins: HashMap::new(),
            rule_engine: RuleEngine::new(),
        };

        engine.register_core_builtins();
        engine
    }

    /// Register core SWRL built-in functions
    fn register_core_builtins(&mut self) {
        // Comparison built-ins
        self.register_builtin(BuiltinFunction {
            name: "equal".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_equal,
        });

        self.register_builtin(BuiltinFunction {
            name: "notEqual".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_not_equal,
        });

        self.register_builtin(BuiltinFunction {
            name: "lessThan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_less_than,
        });

        self.register_builtin(BuiltinFunction {
            name: "greaterThan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_greater_than,
        });

        // Math built-ins
        self.register_builtin(BuiltinFunction {
            name: "add".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_add,
        });

        self.register_builtin(BuiltinFunction {
            name: "subtract".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_subtract,
        });

        self.register_builtin(BuiltinFunction {
            name: "multiply".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_multiply,
        });

        // String built-ins
        self.register_builtin(BuiltinFunction {
            name: "stringConcat".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: None,
            implementation: builtin_string_concat,
        });

        self.register_builtin(BuiltinFunction {
            name: "stringLength".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_string_length,
        });

        // Boolean built-ins
        self.register_builtin(BuiltinFunction {
            name: "booleanValue".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_boolean_value,
        });

        // Advanced mathematical built-ins
        self.register_builtin(BuiltinFunction {
            name: "mod".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_mod,
        });

        self.register_builtin(BuiltinFunction {
            name: "pow".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_pow,
        });

        self.register_builtin(BuiltinFunction {
            name: "sqrt".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_sqrt,
        });

        self.register_builtin(BuiltinFunction {
            name: "sin".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_sin,
        });

        self.register_builtin(BuiltinFunction {
            name: "cos".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_cos,
        });

        // Date and time built-ins
        self.register_builtin(BuiltinFunction {
            name: "dayTimeDuration".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_day_time_duration,
        });

        self.register_builtin(BuiltinFunction {
            name: "yearMonthDuration".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_year_month_duration,
        });

        self.register_builtin(BuiltinFunction {
            name: "dateTime".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_date_time,
        });

        // List operations
        self.register_builtin(BuiltinFunction {
            name: "listConcat".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_list_concat,
        });

        self.register_builtin(BuiltinFunction {
            name: "listLength".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_length,
        });

        self.register_builtin(BuiltinFunction {
            name: "member".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_member,
        });

        // Enhanced string operations
        self.register_builtin(BuiltinFunction {
            name: "stringMatches".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(3),
            implementation: builtin_string_matches,
        });

        self.register_builtin(BuiltinFunction {
            name: "substring".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(4),
            implementation: builtin_substring,
        });

        self.register_builtin(BuiltinFunction {
            name: "upperCase".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_upper_case,
        });

        self.register_builtin(BuiltinFunction {
            name: "lowerCase".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_lower_case,
        });

        // Geographic operations
        self.register_builtin(BuiltinFunction {
            name: "distance".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_distance,
        });

        self.register_builtin(BuiltinFunction {
            name: "within".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_within,
        });

        // Advanced mathematical functions
        self.register_builtin(BuiltinFunction {
            name: "tan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_tan,
        });

        self.register_builtin(BuiltinFunction {
            name: "asin".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_asin,
        });

        self.register_builtin(BuiltinFunction {
            name: "acos".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_acos,
        });

        self.register_builtin(BuiltinFunction {
            name: "atan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_atan,
        });

        self.register_builtin(BuiltinFunction {
            name: "log".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_log,
        });

        self.register_builtin(BuiltinFunction {
            name: "exp".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_exp,
        });

        // Temporal operations
        self.register_builtin(BuiltinFunction {
            name: "dateAdd".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_date_add,
        });

        self.register_builtin(BuiltinFunction {
            name: "dateDiff".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_date_diff,
        });

        self.register_builtin(BuiltinFunction {
            name: "now".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_now,
        });

        // SWRL-X Temporal Extensions
        self.register_builtin(BuiltinFunction {
            name: "before".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_temporal_before,
        });

        self.register_builtin(BuiltinFunction {
            name: "after".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_temporal_after,
        });

        self.register_builtin(BuiltinFunction {
            name: "during".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_temporal_during,
        });

        self.register_builtin(BuiltinFunction {
            name: "overlaps".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_temporal_overlaps,
        });

        self.register_builtin(BuiltinFunction {
            name: "meets".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_temporal_meets,
        });

        self.register_builtin(BuiltinFunction {
            name: "intervalDuration".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_interval_duration,
        });

        // Additional mathematical functions
        self.register_builtin(BuiltinFunction {
            name: "abs".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_abs,
        });

        self.register_builtin(BuiltinFunction {
            name: "floor".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_floor,
        });

        self.register_builtin(BuiltinFunction {
            name: "ceil".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_ceil,
        });

        self.register_builtin(BuiltinFunction {
            name: "round".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_round,
        });

        // Additional list operations
        self.register_builtin(BuiltinFunction {
            name: "first".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_first,
        });

        self.register_builtin(BuiltinFunction {
            name: "rest".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_rest,
        });

        self.register_builtin(BuiltinFunction {
            name: "nth".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_nth,
        });

        self.register_builtin(BuiltinFunction {
            name: "append".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_append,
        });

        // Enhanced string operations with regex
        self.register_builtin(BuiltinFunction {
            name: "stringMatchesRegex".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(3),
            implementation: builtin_string_matches_regex,
        });

        // Additional geographic operations
        self.register_builtin(BuiltinFunction {
            name: "contains".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 9,
            max_args: Some(9),
            implementation: builtin_geo_contains,
        });

        self.register_builtin(BuiltinFunction {
            name: "intersects".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 8,
            max_args: Some(8),
            implementation: builtin_geo_intersects,
        });

        self.register_builtin(BuiltinFunction {
            name: "area".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_geo_area,
        });

        // ====== EXPANDED SWRL BUILT-INS ======

        // Division and Integer Operations
        self.register_builtin(BuiltinFunction {
            name: "divide".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_divide,
        });

        self.register_builtin(BuiltinFunction {
            name: "integerDivide".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_integer_divide,
        });

        self.register_builtin(BuiltinFunction {
            name: "unaryMinus".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_unary_minus,
        });

        self.register_builtin(BuiltinFunction {
            name: "unaryPlus".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_unary_plus,
        });

        // Advanced Mathematical Functions
        self.register_builtin(BuiltinFunction {
            name: "min".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_min,
        });

        self.register_builtin(BuiltinFunction {
            name: "max".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_max,
        });

        self.register_builtin(BuiltinFunction {
            name: "avg".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_avg,
        });

        self.register_builtin(BuiltinFunction {
            name: "sum".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_sum,
        });

        // Advanced Comparison Operations
        self.register_builtin(BuiltinFunction {
            name: "lessThanOrEqual".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_less_than_or_equal,
        });

        self.register_builtin(BuiltinFunction {
            name: "greaterThanOrEqual".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_greater_than_or_equal,
        });

        self.register_builtin(BuiltinFunction {
            name: "between".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_between,
        });

        // Type Checking Operations
        self.register_builtin(BuiltinFunction {
            name: "isInteger".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_integer,
        });

        self.register_builtin(BuiltinFunction {
            name: "isFloat".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_float,
        });

        self.register_builtin(BuiltinFunction {
            name: "isString".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_string,
        });

        self.register_builtin(BuiltinFunction {
            name: "isBoolean".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_boolean,
        });

        self.register_builtin(BuiltinFunction {
            name: "isURI".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_uri,
        });

        // Type Conversion Operations
        self.register_builtin(BuiltinFunction {
            name: "intValue".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_int_value,
        });

        self.register_builtin(BuiltinFunction {
            name: "floatValue".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_float_value,
        });

        self.register_builtin(BuiltinFunction {
            name: "stringValue".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_string_value,
        });

        // Enhanced String Operations
        self.register_builtin(BuiltinFunction {
            name: "stringContains".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_string_contains,
        });

        self.register_builtin(BuiltinFunction {
            name: "startsWith".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_starts_with,
        });

        self.register_builtin(BuiltinFunction {
            name: "endsWith".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_ends_with,
        });

        self.register_builtin(BuiltinFunction {
            name: "replace".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_replace,
        });

        self.register_builtin(BuiltinFunction {
            name: "trim".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_trim,
        });

        self.register_builtin(BuiltinFunction {
            name: "split".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_split,
        });

        self.register_builtin(BuiltinFunction {
            name: "indexOf".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_index_of,
        });

        self.register_builtin(BuiltinFunction {
            name: "lastIndexOf".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_last_index_of,
        });

        self.register_builtin(BuiltinFunction {
            name: "normalizeSpace".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_normalize_space,
        });

        // Enhanced Date and Time Operations
        self.register_builtin(BuiltinFunction {
            name: "date".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_date,
        });

        self.register_builtin(BuiltinFunction {
            name: "time".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_time,
        });

        self.register_builtin(BuiltinFunction {
            name: "year".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_year,
        });

        self.register_builtin(BuiltinFunction {
            name: "month".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_month,
        });

        self.register_builtin(BuiltinFunction {
            name: "day".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_day,
        });

        self.register_builtin(BuiltinFunction {
            name: "hour".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_hour,
        });

        self.register_builtin(BuiltinFunction {
            name: "minute".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_minute,
        });

        self.register_builtin(BuiltinFunction {
            name: "second".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_second,
        });

        // Hash and Cryptographic Operations
        self.register_builtin(BuiltinFunction {
            name: "hash".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_hash,
        });

        self.register_builtin(BuiltinFunction {
            name: "base64Encode".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_base64_encode,
        });

        self.register_builtin(BuiltinFunction {
            name: "base64Decode".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_base64_decode,
        });

        // Statistical Operations
        self.register_builtin(BuiltinFunction {
            name: "mean".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_mean,
        });

        self.register_builtin(BuiltinFunction {
            name: "median".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_median,
        });

        self.register_builtin(BuiltinFunction {
            name: "variance".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_variance,
        });

        self.register_builtin(BuiltinFunction {
            name: "stddev".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_stddev,
        });

        // RDF-Specific Operations
        self.register_builtin(BuiltinFunction {
            name: "langMatches".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_lang_matches,
        });

        self.register_builtin(BuiltinFunction {
            name: "str".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_str,
        });

        self.register_builtin(BuiltinFunction {
            name: "isLiteral".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_literal,
        });

        self.register_builtin(BuiltinFunction {
            name: "isBlank".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_blank,
        });

        self.register_builtin(BuiltinFunction {
            name: "isIRI".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_is_iri,
        });

        // URI Operations
        self.register_builtin(BuiltinFunction {
            name: "resolveURI".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_resolve_uri,
        });

        self.register_builtin(BuiltinFunction {
            name: "encodeURI".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_encode_uri,
        });

        self.register_builtin(BuiltinFunction {
            name: "decodeURI".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_decode_uri,
        });

        // Advanced Collection Operations
        self.register_builtin(BuiltinFunction {
            name: "makeList".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: None,
            implementation: builtin_make_list,
        });

        self.register_builtin(BuiltinFunction {
            name: "listInsert".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_list_insert,
        });

        self.register_builtin(BuiltinFunction {
            name: "listRemove".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_remove,
        });

        self.register_builtin(BuiltinFunction {
            name: "listReverse".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_reverse,
        });

        self.register_builtin(BuiltinFunction {
            name: "listSort".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_sort,
        });

        self.register_builtin(BuiltinFunction {
            name: "listUnion".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_union,
        });

        self.register_builtin(BuiltinFunction {
            name: "listIntersection".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_intersection,
        });

        info!(
            "Registered {} core SWRL built-in functions",
            self.builtins.len()
        );
    }

    /// Register a built-in function
    pub fn register_builtin(&mut self, builtin: BuiltinFunction) {
        let full_name = format!("{}{}", builtin.namespace, builtin.name);
        debug!("Registering SWRL built-in: {}", full_name);
        self.builtins.insert(full_name, builtin);
    }

    /// Register a custom built-in function with validation
    pub fn register_custom_builtin(
        &mut self,
        name: String,
        namespace: String,
        min_args: usize,
        max_args: Option<usize>,
        implementation: fn(&[SwrlArgument]) -> Result<bool>,
    ) -> Result<(), String> {
        // Validate namespace
        if !namespace.starts_with("http://") && !namespace.starts_with("https://") {
            return Err(format!(
                "Invalid namespace '{namespace}': must be a valid IRI"
            ));
        }

        // Validate argument constraints
        if let Some(max) = max_args {
            if min_args > max {
                return Err(format!(
                    "Invalid argument constraints: min_args ({min_args}) > max_args ({max})"
                ));
            }
        }

        let builtin = BuiltinFunction {
            name: name.clone(),
            namespace: namespace.clone(),
            min_args,
            max_args,
            implementation,
        };

        self.register_builtin(builtin);

        info!("Registered custom SWRL built-in: {}{}", namespace, name);
        Ok(())
    }

    /// List all registered built-in functions
    pub fn list_builtins(&self) -> Vec<String> {
        self.builtins.keys().cloned().collect()
    }

    /// Check if a built-in function is registered
    pub fn has_builtin(&self, name: &str) -> bool {
        self.builtins.contains_key(name)
    }

    /// Add a SWRL rule
    pub fn add_rule(&mut self, rule: SwrlRule) -> Result<()> {
        debug!("Adding SWRL rule: {}", rule.id);

        // Convert SWRL rule to internal Rule format
        let internal_rule = self.convert_swrl_to_rule(&rule)?;
        self.rule_engine.add_rule(internal_rule);

        self.rules.push(rule);
        Ok(())
    }

    /// Convert SWRL rule to internal Rule format
    fn convert_swrl_to_rule(&self, swrl_rule: &SwrlRule) -> Result<Rule> {
        let mut body_atoms = Vec::new();
        let mut head_atoms = Vec::new();

        // Convert body atoms
        for atom in &swrl_rule.body {
            body_atoms.push(self.convert_swrl_atom_to_rule_atom(atom)?);
        }

        // Convert head atoms
        for atom in &swrl_rule.head {
            head_atoms.push(self.convert_swrl_atom_to_rule_atom(atom)?);
        }

        Ok(Rule {
            name: swrl_rule.id.clone(),
            body: body_atoms,
            head: head_atoms,
        })
    }

    /// Convert SWRL atom to RuleAtom
    fn convert_swrl_atom_to_rule_atom(&self, swrl_atom: &SwrlAtom) -> Result<RuleAtom> {
        match swrl_atom {
            SwrlAtom::Class {
                class_predicate,
                argument,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant(class_predicate.clone()),
            }),
            SwrlAtom::IndividualProperty {
                property_predicate,
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(property_predicate.clone()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::DatavalueProperty {
                property_predicate,
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(property_predicate.clone()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::Builtin {
                builtin_predicate,
                arguments,
            } => {
                let builtin_args: Vec<Term> = arguments
                    .iter()
                    .map(|arg| self.convert_swrl_argument_to_term(arg))
                    .collect();

                Ok(RuleAtom::Builtin {
                    name: builtin_predicate.clone(),
                    args: builtin_args,
                })
            }
            SwrlAtom::SameIndividual {
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant("http://www.w3.org/2002/07/owl#sameAs".to_string()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::DifferentIndividuals {
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(
                    "http://www.w3.org/2002/07/owl#differentFrom".to_string(),
                ),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
        }
    }

    /// Convert SWRL argument to Term
    fn convert_swrl_argument_to_term(&self, argument: &SwrlArgument) -> Term {
        match argument {
            SwrlArgument::Variable(name) => Term::Variable(name.clone()),
            SwrlArgument::Individual(name) => Term::Constant(name.clone()),
            SwrlArgument::Literal(value) => Term::Literal(value.clone()),
        }
    }

    /// Execute SWRL rules on a set of facts
    pub fn execute(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        info!("Executing SWRL rules on {} facts", facts.len());

        // Use the internal rule engine for basic reasoning
        let inferred_facts = self.rule_engine.forward_chain(facts)?;

        // Apply SWRL-specific reasoning
        let mut all_facts = inferred_facts;
        let mut new_facts_added = true;
        let mut iteration = 0;

        while new_facts_added && iteration < 100 {
            new_facts_added = false;
            iteration += 1;

            debug!("SWRL execution iteration {}", iteration);

            // Apply each SWRL rule
            for rule in &self.rules.clone() {
                let rule_facts = self.apply_swrl_rule(rule, &all_facts)?;
                for fact in rule_facts {
                    if !all_facts.contains(&fact) {
                        all_facts.push(fact);
                        new_facts_added = true;
                    }
                }
            }
        }

        info!(
            "SWRL execution completed after {} iterations, {} facts total",
            iteration,
            all_facts.len()
        );
        Ok(all_facts)
    }

    /// Apply a single SWRL rule
    fn apply_swrl_rule(&self, rule: &SwrlRule, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut new_facts = Vec::new();

        // Find all variable bindings that satisfy the rule body
        let bindings = self.find_rule_bindings(&rule.body, facts)?;

        // Apply each binding to generate head facts
        for binding in bindings {
            for head_atom in &rule.head {
                let instantiated_fact = self.instantiate_swrl_atom(head_atom, &binding)?;
                if let Some(fact) = instantiated_fact {
                    new_facts.push(fact);
                }
            }
        }

        if !new_facts.is_empty() {
            debug!(
                "SWRL rule '{}' generated {} new facts",
                rule.id,
                new_facts.len()
            );
        }

        Ok(new_facts)
    }

    /// Find variable bindings that satisfy rule body
    fn find_rule_bindings(
        &self,
        body: &[SwrlAtom],
        facts: &[RuleAtom],
    ) -> Result<Vec<HashMap<String, SwrlArgument>>> {
        let mut bindings = vec![HashMap::new()];

        for atom in body {
            let mut new_bindings = Vec::new();

            for current_binding in bindings {
                let atom_bindings = self.match_swrl_atom(atom, facts, &current_binding)?;
                new_bindings.extend(atom_bindings);
            }

            bindings = new_bindings;
        }

        Ok(bindings)
    }

    /// Match a SWRL atom against facts
    fn match_swrl_atom(
        &self,
        atom: &SwrlAtom,
        facts: &[RuleAtom],
        current_binding: &HashMap<String, SwrlArgument>,
    ) -> Result<Vec<HashMap<String, SwrlArgument>>> {
        let mut bindings = Vec::new();

        match atom {
            SwrlAtom::Builtin {
                builtin_predicate,
                arguments,
            } => {
                // Evaluate built-in predicate
                if self.evaluate_builtin(builtin_predicate, arguments, current_binding)? {
                    bindings.push(current_binding.clone());
                }
            }
            _ => {
                // Convert to RuleAtom and match against facts
                let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;

                for fact in facts {
                    if let Some(new_binding) =
                        self.unify_rule_atoms(&rule_atom, fact, current_binding.clone())?
                    {
                        bindings.push(new_binding);
                    }
                }
            }
        }

        Ok(bindings)
    }

    /// Evaluate a built-in predicate
    fn evaluate_builtin(
        &self,
        predicate: &str,
        arguments: &[SwrlArgument],
        binding: &HashMap<String, SwrlArgument>,
    ) -> Result<bool> {
        // Resolve arguments with current bindings
        let resolved_args: Vec<SwrlArgument> = arguments
            .iter()
            .map(|arg| self.resolve_argument(arg, binding))
            .collect();

        // Look up built-in function
        if let Some(builtin) = self.builtins.get(predicate) {
            // Check argument count
            if resolved_args.len() < builtin.min_args {
                return Err(anyhow::anyhow!(
                    "Built-in {} requires at least {} arguments, got {}",
                    predicate,
                    builtin.min_args,
                    resolved_args.len()
                ));
            }

            if let Some(max_args) = builtin.max_args {
                if resolved_args.len() > max_args {
                    return Err(anyhow::anyhow!(
                        "Built-in {} requires at most {} arguments, got {}",
                        predicate,
                        max_args,
                        resolved_args.len()
                    ));
                }
            }

            // Execute built-in
            (builtin.implementation)(&resolved_args)
        } else {
            warn!("Unknown SWRL built-in: {}", predicate);
            Ok(false)
        }
    }

    /// Resolve an argument with current bindings
    fn resolve_argument(
        &self,
        argument: &SwrlArgument,
        binding: &HashMap<String, SwrlArgument>,
    ) -> SwrlArgument {
        match argument {
            SwrlArgument::Variable(name) => binding
                .get(name)
                .cloned()
                .unwrap_or_else(|| argument.clone()),
            _ => argument.clone(),
        }
    }

    /// Unify two rule atoms
    fn unify_rule_atoms(
        &self,
        pattern: &RuleAtom,
        fact: &RuleAtom,
        mut binding: HashMap<String, SwrlArgument>,
    ) -> Result<Option<HashMap<String, SwrlArgument>>> {
        match (pattern, fact) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if self.unify_terms(s1, s2, &mut binding)?
                    && self.unify_terms(p1, p2, &mut binding)?
                    && self.unify_terms(o1, o2, &mut binding)?
                {
                    Ok(Some(binding))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Unify two terms
    fn unify_terms(
        &self,
        term1: &Term,
        term2: &Term,
        binding: &mut HashMap<String, SwrlArgument>,
    ) -> Result<bool> {
        match (term1, term2) {
            (Term::Variable(var), term) => {
                let swrl_arg = self.term_to_swrl_argument(term);
                if let Some(existing) = binding.get(var) {
                    Ok(existing == &swrl_arg)
                } else {
                    binding.insert(var.clone(), swrl_arg);
                    Ok(true)
                }
            }
            (term, Term::Variable(var)) => {
                let swrl_arg = self.term_to_swrl_argument(term);
                if let Some(existing) = binding.get(var) {
                    Ok(existing == &swrl_arg)
                } else {
                    binding.insert(var.clone(), swrl_arg);
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            _ => Ok(false),
        }
    }

    /// Convert Term to SwrlArgument
    fn term_to_swrl_argument(&self, term: &Term) -> SwrlArgument {
        match term {
            Term::Variable(name) => SwrlArgument::Variable(name.clone()),
            Term::Constant(name) => SwrlArgument::Individual(name.clone()),
            Term::Literal(value) => SwrlArgument::Literal(value.clone()),
            Term::Function { name, args } => {
                // For functions, convert to a complex literal representation
                let arg_strs: Vec<String> = args
                    .iter()
                    .map(|arg| match arg {
                        Term::Variable(v) => format!("?{v}"),
                        Term::Constant(c) => c.clone(),
                        Term::Literal(l) => l.clone(),
                        Term::Function {
                            name: fn_name,
                            args: fn_args,
                        } => {
                            format!("{}({} args)", fn_name, fn_args.len())
                        }
                    })
                    .collect();
                SwrlArgument::Literal(format!("{}({})", name, arg_strs.join(",")))
            }
        }
    }

    /// Instantiate a SWRL atom with variable bindings
    fn instantiate_swrl_atom(
        &self,
        atom: &SwrlAtom,
        binding: &HashMap<String, SwrlArgument>,
    ) -> Result<Option<RuleAtom>> {
        let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;

        let instantiated = RuleAtom::Triple {
            subject: self.instantiate_term_with_bindings(&rule_atom, binding, 0),
            predicate: self.instantiate_term_with_bindings(&rule_atom, binding, 1),
            object: self.instantiate_term_with_bindings(&rule_atom, binding, 2),
        };

        Ok(Some(instantiated))
    }

    /// Instantiate a term with bindings (helper method)
    fn instantiate_term_with_bindings(
        &self,
        rule_atom: &RuleAtom,
        binding: &HashMap<String, SwrlArgument>,
        position: usize,
    ) -> Term {
        let term = match (rule_atom, position) {
            (RuleAtom::Triple { subject, .. }, 0) => subject,
            (RuleAtom::Triple { predicate, .. }, 1) => predicate,
            (RuleAtom::Triple { object, .. }, 2) => object,
            _ => return Term::Constant("error".to_string()),
        };

        match term {
            Term::Variable(var) => {
                if let Some(bound_value) = binding.get(var) {
                    self.convert_swrl_argument_to_term(bound_value)
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }

    /// Get rule statistics
    pub fn get_stats(&self) -> SwrlStats {
        SwrlStats {
            total_rules: self.rules.len(),
            total_builtins: self.builtins.len(),
        }
    }
}
