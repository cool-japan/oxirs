//! Query Hints System for SPARQL Query Optimization
//!
//! This module provides a comprehensive query hints system that allows users to guide
//! the query optimizer with specific optimization directives. Similar to SQL query hints
//! in PostgreSQL, MySQL, and Oracle.
//!
//! # Features
//!
//! - **Join Hints**: Specify join algorithms (HASH_JOIN, MERGE_JOIN, NESTED_LOOP)
//! - **Index Hints**: Force or avoid specific indexes
//! - **Cardinality Hints**: Override cardinality estimates
//! - **Parallelism Hints**: Control parallel execution
//! - **Materialization Hints**: Control intermediate result materialization
//! - **Timeout Hints**: Set query-specific timeouts
//! - **Memory Hints**: Control memory allocation for query processing
//!
//! # Usage
//!
//! ```sparql
//! # Hint comments are embedded in SPARQL queries
//! # /*+ HASH_JOIN(?s, ?o) PARALLEL(4) TIMEOUT(30s) */
//! SELECT ?s ?p ?o WHERE { ?s ?p ?o }
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_arq::query_hints::{QueryHints, JoinHint, HintParser};
//!
//! let query = r#"
//!     # /*+ HASH_JOIN(?person, ?name) CARDINALITY(?person, 1000) */
//!     SELECT ?person ?name WHERE {
//!         ?person <http://xmlns.com/foaf/0.1/name> ?name .
//!     }
//! "#;
//!
//! let hints = HintParser::parse(query).unwrap();
//! println!("Parsed hints: {:?}", hints);
//! ```

use crate::algebra::Variable;
use anyhow::{anyhow, Result};
use regex::Regex;
// Metrics imports available for future use
#[allow(unused_imports)]
use scirs2_core::metrics::{Counter, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Query hints for optimizer guidance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryHints {
    /// Join algorithm hints
    pub join_hints: Vec<JoinHint>,
    /// Index usage hints
    pub index_hints: Vec<IndexHint>,
    /// Cardinality override hints
    pub cardinality_hints: Vec<CardinalityHint>,
    /// Parallelism configuration hints
    pub parallelism_hints: Option<ParallelismHint>,
    /// Materialization strategy hints
    pub materialization_hints: Vec<MaterializationHint>,
    /// Query timeout hint
    pub timeout_hint: Option<Duration>,
    /// Memory limit hint
    pub memory_hint: Option<MemoryHint>,
    /// Cache control hints
    pub cache_hints: Option<CacheHint>,
    /// Ordering hints for join execution
    pub join_order_hint: Option<JoinOrderHint>,
    /// Filter pushdown hints
    pub filter_hints: Vec<FilterHint>,
    /// Custom optimizer directives
    pub custom_directives: HashMap<String, String>,
}

impl QueryHints {
    /// Create new empty hints
    pub fn new() -> Self {
        Self::default()
    }

    /// Create hints builder
    pub fn builder() -> QueryHintsBuilder {
        QueryHintsBuilder::new()
    }

    /// Check if any hints are specified
    pub fn is_empty(&self) -> bool {
        self.join_hints.is_empty()
            && self.index_hints.is_empty()
            && self.cardinality_hints.is_empty()
            && self.parallelism_hints.is_none()
            && self.materialization_hints.is_empty()
            && self.timeout_hint.is_none()
            && self.memory_hint.is_none()
            && self.cache_hints.is_none()
            && self.join_order_hint.is_none()
            && self.filter_hints.is_empty()
            && self.custom_directives.is_empty()
    }

    /// Get hint count
    pub fn hint_count(&self) -> usize {
        let mut count = self.join_hints.len()
            + self.index_hints.len()
            + self.cardinality_hints.len()
            + self.materialization_hints.len()
            + self.filter_hints.len()
            + self.custom_directives.len();

        if self.parallelism_hints.is_some() {
            count += 1;
        }
        if self.timeout_hint.is_some() {
            count += 1;
        }
        if self.memory_hint.is_some() {
            count += 1;
        }
        if self.cache_hints.is_some() {
            count += 1;
        }
        if self.join_order_hint.is_some() {
            count += 1;
        }
        count
    }

    /// Merge hints from another QueryHints instance (other takes precedence)
    pub fn merge(&mut self, other: QueryHints) {
        self.join_hints.extend(other.join_hints);
        self.index_hints.extend(other.index_hints);
        self.cardinality_hints.extend(other.cardinality_hints);
        self.materialization_hints
            .extend(other.materialization_hints);
        self.filter_hints.extend(other.filter_hints);
        self.custom_directives.extend(other.custom_directives);

        if other.parallelism_hints.is_some() {
            self.parallelism_hints = other.parallelism_hints;
        }
        if other.timeout_hint.is_some() {
            self.timeout_hint = other.timeout_hint;
        }
        if other.memory_hint.is_some() {
            self.memory_hint = other.memory_hint;
        }
        if other.cache_hints.is_some() {
            self.cache_hints = other.cache_hints;
        }
        if other.join_order_hint.is_some() {
            self.join_order_hint = other.join_order_hint;
        }
    }

    /// Get join hint for specific variables
    pub fn get_join_hint(&self, vars: &[Variable]) -> Option<&JoinHint> {
        self.join_hints.iter().find(|hint| {
            vars.iter()
                .all(|v| hint.variables.iter().any(|hv| hv == v.name()))
        })
    }

    /// Get cardinality hint for a variable
    pub fn get_cardinality_hint(&self, var: &Variable) -> Option<u64> {
        self.cardinality_hints
            .iter()
            .find(|hint| hint.variable == var.name())
            .map(|hint| hint.cardinality)
    }

    /// Get index hint for a pattern
    pub fn get_index_hint(&self, pattern_id: &str) -> Option<&IndexHint> {
        self.index_hints
            .iter()
            .find(|hint| hint.pattern_id == pattern_id)
    }
}

/// Builder for QueryHints
#[derive(Debug, Default)]
pub struct QueryHintsBuilder {
    hints: QueryHints,
}

impl QueryHintsBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a join hint
    pub fn with_join_hint(mut self, hint: JoinHint) -> Self {
        self.hints.join_hints.push(hint);
        self
    }

    /// Add hash join hint for variables
    pub fn hash_join(self, variables: Vec<&str>) -> Self {
        self.with_join_hint(JoinHint::new(
            variables.into_iter().map(String::from).collect(),
            JoinAlgorithmHint::HashJoin,
        ))
    }

    /// Add merge join hint for variables
    pub fn merge_join(self, variables: Vec<&str>) -> Self {
        self.with_join_hint(JoinHint::new(
            variables.into_iter().map(String::from).collect(),
            JoinAlgorithmHint::MergeJoin,
        ))
    }

    /// Add nested loop join hint for variables
    pub fn nested_loop_join(self, variables: Vec<&str>) -> Self {
        self.with_join_hint(JoinHint::new(
            variables.into_iter().map(String::from).collect(),
            JoinAlgorithmHint::NestedLoop,
        ))
    }

    /// Add an index hint
    pub fn with_index_hint(mut self, hint: IndexHint) -> Self {
        self.hints.index_hints.push(hint);
        self
    }

    /// Force use of a specific index
    pub fn use_index(self, pattern_id: &str, index_name: &str) -> Self {
        self.with_index_hint(IndexHint::use_index(pattern_id, index_name))
    }

    /// Ignore a specific index
    pub fn ignore_index(self, pattern_id: &str, index_name: &str) -> Self {
        self.with_index_hint(IndexHint::ignore_index(pattern_id, index_name))
    }

    /// Add a cardinality hint
    pub fn with_cardinality_hint(mut self, hint: CardinalityHint) -> Self {
        self.hints.cardinality_hints.push(hint);
        self
    }

    /// Override cardinality for a variable
    pub fn cardinality(self, variable: &str, cardinality: u64) -> Self {
        self.with_cardinality_hint(CardinalityHint::new(variable, cardinality))
    }

    /// Set parallelism hint
    pub fn with_parallelism(mut self, hint: ParallelismHint) -> Self {
        self.hints.parallelism_hints = Some(hint);
        self
    }

    /// Enable parallel execution with specified threads
    pub fn parallel(self, threads: usize) -> Self {
        self.with_parallelism(ParallelismHint::new(threads))
    }

    /// Disable parallel execution
    pub fn no_parallel(self) -> Self {
        self.with_parallelism(ParallelismHint::disabled())
    }

    /// Add materialization hint
    pub fn with_materialization_hint(mut self, hint: MaterializationHint) -> Self {
        self.hints.materialization_hints.push(hint);
        self
    }

    /// Set query timeout
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.hints.timeout_hint = Some(duration);
        self
    }

    /// Set timeout in seconds
    pub fn timeout_secs(self, secs: u64) -> Self {
        self.timeout(Duration::from_secs(secs))
    }

    /// Set memory hint
    pub fn with_memory_hint(mut self, hint: MemoryHint) -> Self {
        self.hints.memory_hint = Some(hint);
        self
    }

    /// Set memory limit in bytes
    pub fn memory_limit(self, bytes: usize) -> Self {
        self.with_memory_hint(MemoryHint {
            max_memory: bytes,
            prefer_streaming: false,
            spill_to_disk: true,
        })
    }

    /// Prefer streaming execution
    pub fn prefer_streaming(self) -> Self {
        self.with_memory_hint(MemoryHint {
            max_memory: usize::MAX,
            prefer_streaming: true,
            spill_to_disk: false,
        })
    }

    /// Set cache hints
    pub fn with_cache_hint(mut self, hint: CacheHint) -> Self {
        self.hints.cache_hints = Some(hint);
        self
    }

    /// Bypass query cache
    pub fn no_cache(self) -> Self {
        self.with_cache_hint(CacheHint {
            use_cache: false,
            update_cache: false,
            cache_ttl: None,
        })
    }

    /// Set join order hint
    pub fn with_join_order(mut self, hint: JoinOrderHint) -> Self {
        self.hints.join_order_hint = Some(hint);
        self
    }

    /// Force specific join order
    pub fn fixed_join_order(self, order: Vec<&str>) -> Self {
        self.with_join_order(JoinOrderHint {
            strategy: JoinOrderStrategy::Fixed,
            order: order.into_iter().map(String::from).collect(),
        })
    }

    /// Add filter hint
    pub fn with_filter_hint(mut self, hint: FilterHint) -> Self {
        self.hints.filter_hints.push(hint);
        self
    }

    /// Add custom directive
    pub fn directive(mut self, key: &str, value: &str) -> Self {
        self.hints
            .custom_directives
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Build the QueryHints
    pub fn build(self) -> QueryHints {
        self.hints
    }
}

/// Join algorithm hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinHint {
    /// Variables involved in the join
    pub variables: Vec<String>,
    /// Preferred join algorithm
    pub algorithm: JoinAlgorithmHint,
    /// Optional build side for hash join
    pub build_side: Option<JoinBuildSide>,
    /// Priority (higher = more important)
    pub priority: u8,
}

impl JoinHint {
    /// Create new join hint
    pub fn new(variables: Vec<String>, algorithm: JoinAlgorithmHint) -> Self {
        Self {
            variables,
            algorithm,
            build_side: None,
            priority: 1,
        }
    }

    /// Set build side for hash join
    pub fn with_build_side(mut self, side: JoinBuildSide) -> Self {
        self.build_side = Some(side);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Supported join algorithms for hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JoinAlgorithmHint {
    /// Hash join (good for large datasets)
    HashJoin,
    /// Sort-merge join (good for pre-sorted data)
    MergeJoin,
    /// Nested loop join (good for small datasets or selective filters)
    NestedLoop,
    /// Index-based join
    IndexJoin,
    /// Let optimizer decide
    Auto,
}

impl std::fmt::Display for JoinAlgorithmHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinAlgorithmHint::HashJoin => write!(f, "HASH_JOIN"),
            JoinAlgorithmHint::MergeJoin => write!(f, "MERGE_JOIN"),
            JoinAlgorithmHint::NestedLoop => write!(f, "NESTED_LOOP"),
            JoinAlgorithmHint::IndexJoin => write!(f, "INDEX_JOIN"),
            JoinAlgorithmHint::Auto => write!(f, "AUTO"),
        }
    }
}

/// Build side for hash join
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinBuildSide {
    /// Use left side as build side
    Left,
    /// Use right side as build side
    Right,
    /// Let optimizer choose
    Auto,
}

/// Index usage hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHint {
    /// Pattern or variable this hint applies to
    pub pattern_id: String,
    /// Index directive
    pub directive: IndexDirective,
    /// Optional specific index names
    pub index_names: Vec<String>,
}

impl IndexHint {
    /// Create hint to use a specific index
    pub fn use_index(pattern_id: &str, index_name: &str) -> Self {
        Self {
            pattern_id: pattern_id.to_string(),
            directive: IndexDirective::Use,
            index_names: vec![index_name.to_string()],
        }
    }

    /// Create hint to ignore a specific index
    pub fn ignore_index(pattern_id: &str, index_name: &str) -> Self {
        Self {
            pattern_id: pattern_id.to_string(),
            directive: IndexDirective::Ignore,
            index_names: vec![index_name.to_string()],
        }
    }

    /// Create hint to force index scan
    pub fn force_index(pattern_id: &str) -> Self {
        Self {
            pattern_id: pattern_id.to_string(),
            directive: IndexDirective::Force,
            index_names: Vec::new(),
        }
    }

    /// Create hint to prevent index usage
    pub fn no_index(pattern_id: &str) -> Self {
        Self {
            pattern_id: pattern_id.to_string(),
            directive: IndexDirective::NoIndex,
            index_names: Vec::new(),
        }
    }
}

/// Index directive types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexDirective {
    /// Prefer using specified indexes
    Use,
    /// Ignore specified indexes
    Ignore,
    /// Force index usage (any available)
    Force,
    /// Disable index usage (full scan)
    NoIndex,
}

/// Cardinality override hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardinalityHint {
    /// Variable name
    pub variable: String,
    /// Override cardinality value
    pub cardinality: u64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
}

impl CardinalityHint {
    /// Create new cardinality hint
    pub fn new(variable: &str, cardinality: u64) -> Self {
        Self {
            variable: variable.to_string(),
            cardinality,
            confidence: 1.0,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Parallelism configuration hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismHint {
    /// Enable parallel execution
    pub enabled: bool,
    /// Number of threads (None = auto)
    pub threads: Option<usize>,
    /// Minimum batch size for parallel execution
    pub min_batch_size: usize,
    /// Enable work stealing
    pub work_stealing: bool,
}

impl ParallelismHint {
    /// Create new parallelism hint with specified threads
    pub fn new(threads: usize) -> Self {
        Self {
            enabled: true,
            threads: Some(threads),
            min_batch_size: 1000,
            work_stealing: true,
        }
    }

    /// Create disabled parallelism hint
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            threads: None,
            min_batch_size: 0,
            work_stealing: false,
        }
    }

    /// Create auto-parallelism hint
    pub fn auto() -> Self {
        Self {
            enabled: true,
            threads: None,
            min_batch_size: 1000,
            work_stealing: true,
        }
    }
}

/// Materialization strategy hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationHint {
    /// Pattern or subquery to materialize
    pub target: String,
    /// Materialization strategy
    pub strategy: MaterializationStrategy,
}

impl MaterializationHint {
    /// Create hint to materialize a pattern
    pub fn materialize(target: &str) -> Self {
        Self {
            target: target.to_string(),
            strategy: MaterializationStrategy::Eager,
        }
    }

    /// Create hint for lazy evaluation
    pub fn lazy(target: &str) -> Self {
        Self {
            target: target.to_string(),
            strategy: MaterializationStrategy::Lazy,
        }
    }

    /// Create hint for streaming execution
    pub fn streaming(target: &str) -> Self {
        Self {
            target: target.to_string(),
            strategy: MaterializationStrategy::Streaming,
        }
    }
}

/// Materialization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationStrategy {
    /// Materialize immediately
    Eager,
    /// Lazy evaluation (materialize on demand)
    Lazy,
    /// Streaming (no materialization, pipeline)
    Streaming,
    /// Partial materialization with threshold
    Partial,
}

/// Memory configuration hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHint {
    /// Maximum memory to use (bytes)
    pub max_memory: usize,
    /// Prefer streaming execution for memory efficiency
    pub prefer_streaming: bool,
    /// Allow spilling to disk
    pub spill_to_disk: bool,
}

/// Cache control hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHint {
    /// Use cached results if available
    pub use_cache: bool,
    /// Update cache with new results
    pub update_cache: bool,
    /// Cache TTL override
    pub cache_ttl: Option<Duration>,
}

/// Join order hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderHint {
    /// Join ordering strategy
    pub strategy: JoinOrderStrategy,
    /// Explicit join order (pattern/variable names)
    pub order: Vec<String>,
}

/// Join ordering strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinOrderStrategy {
    /// Use optimizer's choice
    Auto,
    /// Fixed order as specified
    Fixed,
    /// Process in left-to-right order as written
    LeftToRight,
    /// Start with smallest estimates
    SmallestFirst,
    /// Start with most selective
    MostSelectiveFirst,
}

/// Filter pushdown hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterHint {
    /// Filter expression identifier
    pub filter_id: String,
    /// Pushdown directive
    pub directive: FilterPushdownDirective,
    /// Target pattern for pushdown
    pub target_pattern: Option<String>,
}

/// Filter pushdown directives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterPushdownDirective {
    /// Push filter down (default behavior)
    Push,
    /// Keep filter at current level
    NoPush,
    /// Push to specific pattern
    PushTo,
}

/// Parser for query hints embedded in SPARQL comments
pub struct HintParser {
    /// Statistics counter
    hints_parsed: Arc<AtomicU64>,
    /// Parsing timer
    parse_timer: Arc<AtomicU64>,
}

impl Default for HintParser {
    fn default() -> Self {
        Self::new()
    }
}

impl HintParser {
    /// Create new hint parser
    pub fn new() -> Self {
        Self {
            hints_parsed: Arc::new(AtomicU64::new(0)),
            parse_timer: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Parse hints from a SPARQL query string
    pub fn parse(query: &str) -> Result<QueryHints> {
        let parser = Self::new();
        parser.parse_query(query)
    }

    /// Parse hints from query
    pub fn parse_query(&self, query: &str) -> Result<QueryHints> {
        let start = std::time::Instant::now();
        let mut hints = QueryHints::new();

        // First, find all line-comment style hints: # /*+ ... */
        // and collect their positions to avoid double-matching
        let line_hint_pattern = line_hint_regex();
        let mut line_hint_positions: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for cap in line_hint_pattern.captures_iter(query) {
            if let Some(hint_text) = cap.get(1) {
                let parsed = self.parse_hint_block(hint_text.as_str())?;
                hints.merge(parsed);
                // Record the position of the /*+ start to avoid double-matching
                if let Some(m) = cap.get(0) {
                    // The regular hint /*+ starts somewhere within this match
                    // Find where /*+ starts in the original query
                    let match_start = m.start();
                    let match_str = m.as_str();
                    if let Some(inner_pos) = match_str.find("/*+") {
                        line_hint_positions.insert(match_start + inner_pos);
                    }
                }
            }
        }

        // Look for regular hint comments: /*+ HINT1 HINT2 ... */
        // Skip any that are already matched as line comments
        let hint_pattern = regex();

        for cap in hint_pattern.captures_iter(query) {
            if let Some(m) = cap.get(0) {
                // Skip if this was already matched as a line comment
                if line_hint_positions.contains(&m.start()) {
                    continue;
                }
            }
            if let Some(hint_text) = cap.get(1) {
                let parsed = self.parse_hint_block(hint_text.as_str())?;
                hints.merge(parsed);
            }
        }

        self.hints_parsed
            .fetch_add(hints.hint_count() as u64, Ordering::Relaxed);
        self.parse_timer
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        Ok(hints)
    }

    /// Parse a hint block (the text inside /*+ ... */)
    fn parse_hint_block(&self, text: &str) -> Result<QueryHints> {
        let mut hints = QueryHints::new();

        // Split by whitespace, keeping parenthesized groups together
        let tokens = self.tokenize_hints(text);
        let mut i = 0;

        while i < tokens.len() {
            let token = &tokens[i];
            let hint_upper = token.to_uppercase();

            match hint_upper.as_str() {
                "HASH_JOIN" => {
                    if i + 1 < tokens.len() {
                        let vars = self.parse_variable_list(&tokens[i + 1])?;
                        hints
                            .join_hints
                            .push(JoinHint::new(vars, JoinAlgorithmHint::HashJoin));
                        i += 1;
                    }
                }
                "MERGE_JOIN" => {
                    if i + 1 < tokens.len() {
                        let vars = self.parse_variable_list(&tokens[i + 1])?;
                        hints
                            .join_hints
                            .push(JoinHint::new(vars, JoinAlgorithmHint::MergeJoin));
                        i += 1;
                    }
                }
                "NESTED_LOOP" | "NL_JOIN" => {
                    if i + 1 < tokens.len() {
                        let vars = self.parse_variable_list(&tokens[i + 1])?;
                        hints
                            .join_hints
                            .push(JoinHint::new(vars, JoinAlgorithmHint::NestedLoop));
                        i += 1;
                    }
                }
                "INDEX_JOIN" => {
                    if i + 1 < tokens.len() {
                        let vars = self.parse_variable_list(&tokens[i + 1])?;
                        hints
                            .join_hints
                            .push(JoinHint::new(vars, JoinAlgorithmHint::IndexJoin));
                        i += 1;
                    }
                }
                "CARDINALITY" => {
                    if i + 1 < tokens.len() {
                        let (var, card) = self.parse_cardinality_hint(&tokens[i + 1])?;
                        hints
                            .cardinality_hints
                            .push(CardinalityHint::new(&var, card));
                        i += 1;
                    }
                }
                "PARALLEL" => {
                    if i + 1 < tokens.len() {
                        let threads = self.parse_single_value(&tokens[i + 1])?;
                        hints.parallelism_hints = Some(ParallelismHint::new(threads as usize));
                        i += 1;
                    } else {
                        hints.parallelism_hints = Some(ParallelismHint::auto());
                    }
                }
                "NO_PARALLEL" | "NOPARALLEL" => {
                    hints.parallelism_hints = Some(ParallelismHint::disabled());
                }
                "TIMEOUT" => {
                    if i + 1 < tokens.len() {
                        let timeout = self.parse_duration(&tokens[i + 1])?;
                        hints.timeout_hint = Some(timeout);
                        i += 1;
                    }
                }
                "MEMORY_LIMIT" | "MAX_MEMORY" => {
                    if i + 1 < tokens.len() {
                        let bytes = self.parse_memory_size(&tokens[i + 1])?;
                        hints.memory_hint = Some(MemoryHint {
                            max_memory: bytes,
                            prefer_streaming: false,
                            spill_to_disk: true,
                        });
                        i += 1;
                    }
                }
                "STREAMING" => {
                    hints.memory_hint = Some(MemoryHint {
                        max_memory: usize::MAX,
                        prefer_streaming: true,
                        spill_to_disk: false,
                    });
                }
                "NO_CACHE" | "NOCACHE" => {
                    hints.cache_hints = Some(CacheHint {
                        use_cache: false,
                        update_cache: false,
                        cache_ttl: None,
                    });
                }
                "CACHE" => {
                    hints.cache_hints = Some(CacheHint {
                        use_cache: true,
                        update_cache: true,
                        cache_ttl: None,
                    });
                }
                "USE_INDEX" | "FORCE_INDEX" => {
                    if i + 1 < tokens.len() {
                        let (pattern, index) = self.parse_index_hint(&tokens[i + 1])?;
                        hints
                            .index_hints
                            .push(IndexHint::use_index(&pattern, &index));
                        i += 1;
                    }
                }
                "IGNORE_INDEX" | "NO_INDEX" => {
                    if i + 1 < tokens.len() {
                        let (pattern, index) = self.parse_index_hint(&tokens[i + 1])?;
                        hints
                            .index_hints
                            .push(IndexHint::ignore_index(&pattern, &index));
                        i += 1;
                    }
                }
                "ORDERED" | "FIXED_ORDER" => {
                    hints.join_order_hint = Some(JoinOrderHint {
                        strategy: JoinOrderStrategy::LeftToRight,
                        order: Vec::new(),
                    });
                }
                "LEADING" => {
                    if i + 1 < tokens.len() {
                        let order = self.parse_variable_list(&tokens[i + 1])?;
                        hints.join_order_hint = Some(JoinOrderHint {
                            strategy: JoinOrderStrategy::Fixed,
                            order,
                        });
                        i += 1;
                    }
                }
                "MATERIALIZE" => {
                    if i + 1 < tokens.len() {
                        let target = self.parse_single_string(&tokens[i + 1])?;
                        hints
                            .materialization_hints
                            .push(MaterializationHint::materialize(&target));
                        i += 1;
                    }
                }
                _ => {
                    // Unknown hint - store as custom directive
                    if i + 1 < tokens.len() && tokens[i + 1].starts_with('(') {
                        hints.custom_directives.insert(
                            hint_upper,
                            tokens[i + 1]
                                .trim_matches(|c| c == '(' || c == ')')
                                .to_string(),
                        );
                        i += 1;
                    }
                }
            }
            i += 1;
        }

        Ok(hints)
    }

    /// Tokenize hint text preserving parenthesized groups
    /// Splits hint names from their arguments: HASH_JOIN(?s, ?o) -> ["HASH_JOIN", "(?s, ?o)"]
    fn tokenize_hints(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut paren_depth = 0;

        for ch in text.chars() {
            match ch {
                '(' if paren_depth == 0 => {
                    // Starting a parenthesized group
                    // If there's a current token (hint name), push it first
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    paren_depth += 1;
                    current.push(ch);
                }
                '(' => {
                    // Nested parens
                    paren_depth += 1;
                    current.push(ch);
                }
                ')' => {
                    paren_depth -= 1;
                    current.push(ch);
                    if paren_depth == 0 && !current.is_empty() {
                        // End of parenthesized group, push it as a token
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                ' ' | '\t' | '\n' | '\r' if paren_depth == 0 => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Parse variable list from parenthesized string: (?var1, ?var2)
    fn parse_variable_list(&self, text: &str) -> Result<Vec<String>> {
        let inner = text.trim_matches(|c| c == '(' || c == ')');
        let vars: Vec<String> = inner
            .split(',')
            .map(|s| s.trim().trim_start_matches('?').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if vars.is_empty() {
            return Err(anyhow!("Empty variable list"));
        }
        Ok(vars)
    }

    /// Parse cardinality hint: (?var, 1000)
    fn parse_cardinality_hint(&self, text: &str) -> Result<(String, u64)> {
        let inner = text.trim_matches(|c| c == '(' || c == ')');
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid cardinality hint format: expected (?var, number)"
            ));
        }

        let var = parts[0].trim_start_matches('?').to_string();
        let card: u64 = parts[1]
            .parse()
            .map_err(|_| anyhow!("Invalid cardinality value: {}", parts[1]))?;

        Ok((var, card))
    }

    /// Parse single numeric value: (4)
    fn parse_single_value(&self, text: &str) -> Result<u64> {
        let inner = text.trim_matches(|c| c == '(' || c == ')');
        inner
            .parse()
            .map_err(|_| anyhow!("Invalid numeric value: {}", inner))
    }

    /// Parse single string value: (name)
    fn parse_single_string(&self, text: &str) -> Result<String> {
        let inner = text.trim_matches(|c| c == '(' || c == ')');
        Ok(inner.trim().to_string())
    }

    /// Parse duration: (30s), (5m), (1h)
    fn parse_duration(&self, text: &str) -> Result<Duration> {
        let inner = text.trim_matches(|c| c == '(' || c == ')').to_lowercase();

        if let Some(secs) = inner.strip_suffix('s') {
            let val: u64 = secs
                .parse()
                .map_err(|_| anyhow!("Invalid duration: {}", text))?;
            return Ok(Duration::from_secs(val));
        }
        if let Some(mins) = inner.strip_suffix('m') {
            let val: u64 = mins
                .parse()
                .map_err(|_| anyhow!("Invalid duration: {}", text))?;
            return Ok(Duration::from_secs(val * 60));
        }
        if let Some(hours) = inner.strip_suffix('h') {
            let val: u64 = hours
                .parse()
                .map_err(|_| anyhow!("Invalid duration: {}", text))?;
            return Ok(Duration::from_secs(val * 3600));
        }

        // Assume milliseconds if no suffix
        let val: u64 = inner
            .parse()
            .map_err(|_| anyhow!("Invalid duration: {}", text))?;
        Ok(Duration::from_millis(val))
    }

    /// Parse memory size: (1GB), (512MB), (1024KB)
    fn parse_memory_size(&self, text: &str) -> Result<usize> {
        let inner = text.trim_matches(|c| c == '(' || c == ')').to_uppercase();

        if let Some(gb) = inner.strip_suffix("GB") {
            let val: usize = gb
                .parse()
                .map_err(|_| anyhow!("Invalid memory size: {}", text))?;
            return Ok(val * 1024 * 1024 * 1024);
        }
        if let Some(mb) = inner.strip_suffix("MB") {
            let val: usize = mb
                .parse()
                .map_err(|_| anyhow!("Invalid memory size: {}", text))?;
            return Ok(val * 1024 * 1024);
        }
        if let Some(kb) = inner.strip_suffix("KB") {
            let val: usize = kb
                .parse()
                .map_err(|_| anyhow!("Invalid memory size: {}", text))?;
            return Ok(val * 1024);
        }

        // Assume bytes if no suffix
        inner
            .parse()
            .map_err(|_| anyhow!("Invalid memory size: {}", text))
    }

    /// Parse index hint: (pattern, index_name)
    fn parse_index_hint(&self, text: &str) -> Result<(String, String)> {
        let inner = text.trim_matches(|c| c == '(' || c == ')');
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid index hint format: expected (pattern, index_name)"
            ));
        }

        Ok((parts[0].to_string(), parts[1].to_string()))
    }

    /// Get parsing statistics
    pub fn statistics(&self) -> HintParserStats {
        HintParserStats {
            hints_parsed: self.hints_parsed.load(Ordering::Relaxed),
            total_parse_time_ns: self.parse_timer.load(Ordering::Relaxed),
        }
    }
}

/// Hint parser statistics
#[derive(Debug, Clone)]
pub struct HintParserStats {
    /// Total hints parsed
    pub hints_parsed: u64,
    /// Total parse time in nanoseconds
    pub total_parse_time_ns: u64,
}

/// Regex for hint comments
fn regex() -> &'static Regex {
    static HINT_REGEX: OnceLock<Regex> = OnceLock::new();
    HINT_REGEX.get_or_init(|| {
        // Match /*+ hint content */ - capture everything between /*+ and */
        // Simple pattern: non-greedy match of any chars between markers
        Regex::new(r"/\*\+\s*(.+?)\s*\*/").expect("Invalid regex")
    })
}

/// Regex for line hint comments
fn line_hint_regex() -> &'static Regex {
    static LINE_HINT_REGEX: OnceLock<Regex> = OnceLock::new();
    LINE_HINT_REGEX.get_or_init(|| {
        // Match # /*+ hint content */
        Regex::new(r"#\s*/\*\+\s*(.+?)\s*\*/").expect("Invalid regex")
    })
}

/// Hint application result
#[derive(Debug, Clone)]
pub struct HintApplicationResult {
    /// Hints that were applied
    pub applied: Vec<String>,
    /// Hints that were ignored (e.g., not applicable)
    pub ignored: Vec<String>,
    /// Hints that conflicted and were resolved
    pub conflicts: Vec<String>,
}

impl HintApplicationResult {
    /// Create new result
    pub fn new() -> Self {
        Self {
            applied: Vec::new(),
            ignored: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    /// Record an applied hint
    pub fn applied(&mut self, hint: &str) {
        self.applied.push(hint.to_string());
    }

    /// Record an ignored hint
    pub fn ignored(&mut self, hint: &str) {
        self.ignored.push(hint.to_string());
    }

    /// Record a conflict resolution
    pub fn conflict(&mut self, hint: &str) {
        self.conflicts.push(hint.to_string());
    }
}

impl Default for HintApplicationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Hint validator for checking hint consistency
pub struct HintValidator;

impl HintValidator {
    /// Validate hints for consistency and conflicts
    pub fn validate(hints: &QueryHints) -> Vec<HintValidationWarning> {
        let mut warnings = Vec::new();

        // Check for conflicting join hints on same variables
        let mut seen_vars: HashMap<String, Vec<JoinAlgorithmHint>> = HashMap::new();
        for hint in &hints.join_hints {
            for var in &hint.variables {
                seen_vars
                    .entry(var.clone())
                    .or_default()
                    .push(hint.algorithm);
            }
        }
        for (var, algorithms) in seen_vars {
            if algorithms.len() > 1 {
                let unique: std::collections::HashSet<_> = algorithms.iter().collect();
                if unique.len() > 1 {
                    warnings.push(HintValidationWarning {
                        severity: WarningSeverity::Warning,
                        message: format!(
                            "Conflicting join hints for variable '{}': {:?}",
                            var, algorithms
                        ),
                    });
                }
            }
        }

        // Check for streaming + parallelism conflict
        if let Some(ref mem) = hints.memory_hint {
            if mem.prefer_streaming {
                if let Some(ref par) = hints.parallelism_hints {
                    if par.enabled && par.threads.is_some_and(|t| t > 1) {
                        warnings.push(HintValidationWarning {
                            severity: WarningSeverity::Info,
                            message:
                                "Streaming mode with parallelism may reduce streaming benefits"
                                    .to_string(),
                        });
                    }
                }
            }
        }

        // Check for cardinality hints with very high values
        for hint in &hints.cardinality_hints {
            if hint.cardinality > 1_000_000_000 {
                warnings.push(HintValidationWarning {
                    severity: WarningSeverity::Warning,
                    message: format!(
                        "Very high cardinality hint for '{}': {} (may affect optimization)",
                        hint.variable, hint.cardinality
                    ),
                });
            }
        }

        // Check timeout is reasonable
        if let Some(timeout) = hints.timeout_hint {
            if timeout < Duration::from_millis(100) {
                warnings.push(HintValidationWarning {
                    severity: WarningSeverity::Warning,
                    message: format!(
                        "Very short timeout: {:?} (query may timeout immediately)",
                        timeout
                    ),
                });
            }
            if timeout > Duration::from_secs(3600) {
                warnings.push(HintValidationWarning {
                    severity: WarningSeverity::Info,
                    message: format!(
                        "Very long timeout: {:?} (consider using async execution)",
                        timeout
                    ),
                });
            }
        }

        warnings
    }
}

/// Hint validation warning
#[derive(Debug, Clone)]
pub struct HintValidationWarning {
    /// Warning severity
    pub severity: WarningSeverity,
    /// Warning message
    pub message: String,
}

/// Warning severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    /// Informational
    Info,
    /// Warning (may affect performance)
    Warning,
    /// Error (hint will be ignored)
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_matching() {
        let re = regex();
        let query = "/*+ HASH_JOIN(?s, ?o) */ SELECT ?s ?o WHERE { ?s ?p ?o }";

        let caps: Vec<_> = re.captures_iter(query).collect();
        println!("Number of captures: {}", caps.len());
        for cap in &caps {
            println!("Full match: {:?}", cap.get(0).map(|m| m.as_str()));
            println!("Group 1: {:?}", cap.get(1).map(|m| m.as_str()));
        }

        assert!(!caps.is_empty(), "Regex should match the hint comment");
    }

    #[test]
    fn test_parse_hash_join_hint() {
        let query = "/*+ HASH_JOIN(?s, ?o) */ SELECT ?s ?o WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.join_hints.len(), 1);
        assert_eq!(hints.join_hints[0].algorithm, JoinAlgorithmHint::HashJoin);
        assert_eq!(hints.join_hints[0].variables, vec!["s", "o"]);
    }

    #[test]
    fn test_parse_cardinality_hint() {
        let query = "/*+ CARDINALITY(?person, 1000) */ SELECT ?person WHERE { ?person a <Person> }";
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.cardinality_hints.len(), 1);
        assert_eq!(hints.cardinality_hints[0].variable, "person");
        assert_eq!(hints.cardinality_hints[0].cardinality, 1000);
    }

    #[test]
    fn test_parse_parallel_hint() {
        let query = "/*+ PARALLEL(4) */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert!(hints.parallelism_hints.is_some());
        let par = hints.parallelism_hints.unwrap();
        assert!(par.enabled);
        assert_eq!(par.threads, Some(4));
    }

    #[test]
    fn test_parse_no_parallel_hint() {
        let query = "/*+ NO_PARALLEL */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert!(hints.parallelism_hints.is_some());
        assert!(!hints.parallelism_hints.unwrap().enabled);
    }

    #[test]
    fn test_parse_timeout_hint() {
        let query = "/*+ TIMEOUT(30s) */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.timeout_hint, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_memory_limit_hint() {
        let query = "/*+ MEMORY_LIMIT(1GB) */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert!(hints.memory_hint.is_some());
        assert_eq!(hints.memory_hint.unwrap().max_memory, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_no_cache_hint() {
        let query = "/*+ NO_CACHE */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert!(hints.cache_hints.is_some());
        assert!(!hints.cache_hints.unwrap().use_cache);
    }

    #[test]
    fn test_parse_multiple_hints() {
        let query =
            "/*+ HASH_JOIN(?s, ?o) PARALLEL(8) TIMEOUT(60s) */ SELECT ?s ?o WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.join_hints.len(), 1);
        assert!(hints.parallelism_hints.is_some());
        assert_eq!(hints.timeout_hint, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_parse_line_comment_hint() {
        let query = r#"
            # /*+ CARDINALITY(?x, 5000) PARALLEL(2) */
            SELECT ?x WHERE { ?x a <Thing> }
        "#;
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.cardinality_hints.len(), 1);
        assert!(hints.parallelism_hints.is_some());
    }

    #[test]
    fn test_hints_builder() {
        let hints = QueryHints::builder()
            .hash_join(vec!["s", "o"])
            .cardinality("person", 1000)
            .parallel(4)
            .timeout_secs(30)
            .no_cache()
            .build();

        assert_eq!(hints.join_hints.len(), 1);
        assert_eq!(hints.cardinality_hints.len(), 1);
        assert!(hints.parallelism_hints.is_some());
        assert_eq!(hints.timeout_hint, Some(Duration::from_secs(30)));
        assert!(hints.cache_hints.is_some());
    }

    #[test]
    fn test_hint_validator() {
        let hints = QueryHints::builder()
            .hash_join(vec!["x", "y"])
            .merge_join(vec!["x", "z"]) // Conflicting hint on ?x
            .timeout(Duration::from_millis(50)) // Very short timeout
            .build();

        let warnings = HintValidator::validate(&hints);

        assert!(warnings.iter().any(|w| w.message.contains("Conflicting")));
        assert!(warnings.iter().any(|w| w.message.contains("short timeout")));
    }

    #[test]
    fn test_hint_merge() {
        let mut hints1 = QueryHints::builder().hash_join(vec!["a", "b"]).build();

        let hints2 = QueryHints::builder()
            .cardinality("c", 500)
            .parallel(4)
            .build();

        hints1.merge(hints2);

        assert_eq!(hints1.join_hints.len(), 1);
        assert_eq!(hints1.cardinality_hints.len(), 1);
        assert!(hints1.parallelism_hints.is_some());
    }

    #[test]
    fn test_empty_hints() {
        let hints = QueryHints::new();
        assert!(hints.is_empty());
        assert_eq!(hints.hint_count(), 0);
    }

    #[test]
    fn test_get_join_hint() {
        let hints = QueryHints::builder().hash_join(vec!["s", "o"]).build();

        let var_s = Variable::new("s").unwrap();
        let var_o = Variable::new("o").unwrap();

        let hint = hints.get_join_hint(&[var_s, var_o]);
        assert!(hint.is_some());
        assert_eq!(hint.unwrap().algorithm, JoinAlgorithmHint::HashJoin);
    }

    #[test]
    fn test_get_cardinality_hint() {
        let hints = QueryHints::builder().cardinality("person", 1000).build();

        let var = Variable::new("person").unwrap();
        let card = hints.get_cardinality_hint(&var);

        assert_eq!(card, Some(1000));
    }

    #[test]
    fn test_parse_index_hint() {
        let query = "/*+ USE_INDEX(pattern1, idx_subject) */ SELECT * WHERE { ?s ?p ?o }";
        let hints = HintParser::parse(query).unwrap();

        assert_eq!(hints.index_hints.len(), 1);
        assert_eq!(hints.index_hints[0].pattern_id, "pattern1");
        assert_eq!(hints.index_hints[0].directive, IndexDirective::Use);
    }

    #[test]
    fn test_parse_leading_hint() {
        let query = "/*+ LEADING(?a, ?b, ?c) */ SELECT * WHERE { ?a ?p ?b . ?b ?q ?c }";
        let hints = HintParser::parse(query).unwrap();

        assert!(hints.join_order_hint.is_some());
        let order = hints.join_order_hint.unwrap();
        assert_eq!(order.strategy, JoinOrderStrategy::Fixed);
        assert_eq!(order.order, vec!["a", "b", "c"]);
    }
}
