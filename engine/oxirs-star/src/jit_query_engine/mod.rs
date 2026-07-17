//! # JIT-Compiled SPARQL-star Query Engine
//!
//! Adaptive execution engine for SPARQL-star queries: a real interpreted
//! execution path today, with hot-path detection and a native-code
//! compilation pipeline that is architecturally wired but **not yet
//! functional end-to-end** (see the "Compiled Mode" caveat below).
//!
//! This module provides:
//! - **Interpreted Execution**: Real BGP-based SPARQL-star execution
//!   (parses the query text and runs it against the store's indices).
//! - **Query Plan Caching**: Cache query plans with smart invalidation.
//! - **Hot Path Detection**: Track per-query execution counts and flag hot
//!   queries as compilation candidates.
//! - **JIT Compilation Pipeline**: Parse → IR → `scirs2_core::jit`
//!   kernel compilation (kernel IDs are produced and cached).
//!
//! ## Overview
//!
//! The JIT query engine operates in two functional modes plus one pending
//! mode:
//! 1. **Interpreted Mode**: Real execution via the crate's BGP query
//!    executor. This is the mode every query actually runs in today.
//! 2. **Warm-up Mode**: Profile collection and hot path detection (feeds
//!    into background compilation attempts).
//! 3. **Compiled Mode (not yet functional)**: Hot queries are compiled to a
//!    cached kernel ID via `scirs2_core::jit`, but there is currently no
//!    integration with `scirs2_core::jit::execute_kernel` to actually run
//!    that kernel — [`compiler::SparqlJitCompiler::execute_compiled`]
//!    always returns a typed error, and [`JitQueryEngine::execute`]
//!    transparently falls back to interpreted execution when this happens.
//!    Do not rely on this mode for a throughput improvement until the
//!    kernel-execution integration lands.
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_star::jit_query_engine::{JitQueryEngine, CompilationStrategy};
//! use oxirs_star::StarStore;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = JitQueryEngine::new();
//! let store = StarStore::new();
//!
//! // Set adaptive compilation threshold
//! engine.set_compilation_threshold(10); // Compile after 10 executions
//!
//! // Execute query (real interpreted BGP execution).
//! let query = "SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}";
//! let results1 = engine.execute(query, &store).await?;
//!
//! // Execute again (still interpreted, building profile).
//! for _ in 0..10 {
//!     engine.execute(query, &store).await?;
//! }
//!
//! // A background compilation attempt may be triggered once the query is
//! // "hot", but execution itself still runs interpreted today: compiled
//! // kernel execution is not yet wired up (see module docs above).
//! let results_still_interpreted = engine.execute(query, &store).await?;
//!
//! # Ok(())
//! # }
//! ```

// Sub-modules
pub mod compiler;
pub mod ir;

use crate::{StarResult, StarStore, StarTriple};
use compiler::SparqlJitCompiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

/// JIT query engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitConfig {
    /// Compilation threshold (number of executions before JIT)
    pub compilation_threshold: usize,
    /// Enable background compilation
    pub enable_background_compilation: bool,
    /// Maximum number of cached plans
    pub max_cached_plans: usize,
    /// Plan cache TTL in seconds
    pub plan_cache_ttl_secs: u64,
    /// Enable hot path detection
    pub enable_hot_path_detection: bool,
    /// Profiling sample rate (0.0-1.0)
    pub profiling_sample_rate: f64,
    /// Enable query plan optimization
    pub enable_plan_optimization: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            compilation_threshold: 10,
            enable_background_compilation: true,
            max_cached_plans: 1000,
            plan_cache_ttl_secs: 3600,
            enable_hot_path_detection: true,
            profiling_sample_rate: 0.1,
            enable_plan_optimization: true,
        }
    }
}

/// Compilation strategy for queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilationStrategy {
    /// Always interpret (no JIT)
    AlwaysInterpret,
    /// Always compile (eager JIT)
    AlwaysCompile,
    /// Adaptive: interpret then compile hot paths
    Adaptive,
    /// Profile-guided: collect profile then compile
    ProfileGuided,
}

/// Query execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Interpreted execution
    Interpreted,
    /// JIT-compiled execution
    Compiled,
    /// Profiling mode (collecting metrics)
    Profiling,
}

/// Query plan representation
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Original query string
    pub query: String,
    /// Query hash for caching
    pub hash: u64,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Execution mode
    pub mode: ExecutionMode,
    /// Compilation timestamp
    pub compiled_at: Option<Instant>,
    /// Execution count
    pub execution_count: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new(query: String) -> Self {
        let hash = Self::compute_hash(&query);
        Self {
            query,
            hash,
            estimated_cost: 0.0,
            mode: ExecutionMode::Interpreted,
            compiled_at: None,
            execution_count: 0,
            total_execution_time: Duration::ZERO,
            avg_execution_time: Duration::ZERO,
        }
    }

    /// Compute query hash
    fn compute_hash(query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Update execution statistics
    pub fn update_stats(&mut self, execution_time: Duration) {
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.avg_execution_time = self.total_execution_time / self.execution_count as u32;
    }

    /// Check if query is hot (should be compiled)
    pub fn is_hot(&self, threshold: usize) -> bool {
        self.execution_count >= threshold && self.mode == ExecutionMode::Interpreted
    }
}

/// Compiled query representation
#[derive(Debug, Clone)]
pub struct CompiledQuery {
    /// Query plan
    pub plan: QueryPlan,
    /// Compiled function pointer (placeholder)
    pub compiled_code: Option<Arc<Vec<u8>>>,
    /// Compilation time
    pub compilation_time: Duration,
    /// Compilation timestamp
    pub compiled_at: Instant,
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// Interpreted executions
    pub interpreted_count: u64,
    /// Compiled executions
    pub compiled_count: u64,
    /// Total compilation time
    pub total_compilation_time: Duration,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average interpreted time
    pub avg_interpreted_time: Duration,
    /// Average compiled time
    pub avg_compiled_time: Duration,
}

/// JIT query engine
pub struct JitQueryEngine {
    /// Configuration
    config: JitConfig,
    /// Query plan cache
    plan_cache: Arc<RwLock<HashMap<u64, QueryPlan>>>,
    /// Compiled query cache
    compiled_cache: Arc<RwLock<HashMap<u64, CompiledQuery>>>,
    /// Execution statistics
    stats: Arc<RwLock<QueryStats>>,
    /// Compilation strategy
    strategy: CompilationStrategy,
    /// Actual JIT compiler (scirs2_core::jit)
    jit_compiler: Arc<RwLock<SparqlJitCompiler>>,
}

impl JitQueryEngine {
    /// Create a new JIT query engine
    pub fn new() -> Self {
        Self::with_config(JitConfig::default())
    }

    /// Create a JIT query engine with custom configuration
    pub fn with_config(config: JitConfig) -> Self {
        // Initialize the actual JIT compiler
        let jit_compiler = SparqlJitCompiler::new().unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to initialise LLVM JIT backend ({}); \
                 falling back to interpreter backend — \
                 execute_compiled() will surface typed errors until LLVM is available",
                e
            );
            SparqlJitCompiler::interpreter_fallback().unwrap_or_default()
        });

        Self {
            config,
            plan_cache: Arc::new(RwLock::new(HashMap::new())),
            compiled_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(QueryStats::default())),
            strategy: CompilationStrategy::Adaptive,
            jit_compiler: Arc::new(RwLock::new(jit_compiler)),
        }
    }

    /// Set compilation strategy
    pub fn set_strategy(&mut self, strategy: CompilationStrategy) {
        self.strategy = strategy;
    }

    /// Set compilation threshold
    pub fn set_compilation_threshold(&mut self, threshold: usize) {
        self.config.compilation_threshold = threshold;
    }

    /// Execute a SPARQL-star query
    #[instrument(skip(self, store))]
    pub async fn execute(&self, query: &str, store: &StarStore) -> StarResult<Vec<StarTriple>> {
        let start = Instant::now();
        let hash = QueryPlan::compute_hash(query);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_queries += 1;
        }

        // Check compiled cache first
        let compiled = self.compiled_cache.read().await;
        if let Some(compiled_query) = compiled.get(&hash) {
            debug!("Cache hit: executing compiled query (hash: {})", hash);

            // Clone the plan before releasing the lock
            let plan = compiled_query.plan.clone();
            drop(compiled); // Release read lock

            match self.execute_compiled(&plan, store).await {
                Ok(result) => {
                    let execution_time = start.elapsed();

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.compiled_count += 1;
                    stats.cache_hits += 1;
                    stats.avg_compiled_time = if stats.compiled_count == 1 {
                        execution_time
                    } else {
                        (stats.avg_compiled_time * (stats.compiled_count - 1) as u32
                            + execution_time)
                            / stats.compiled_count as u32
                    };

                    return Ok(result);
                }
                Err(e) => {
                    // Compiled kernel execution is not yet integrated with
                    // scirs2_core::jit::execute_kernel (see module docs): fall
                    // back to the real interpreted executor below instead of
                    // permanently failing every future execution of a query
                    // that merely happened to become "hot" enough to compile.
                    debug!(
                        "Compiled execution unavailable for query hash {hash} ({e}); \
                         falling back to interpreted mode"
                    );
                }
            }
        } else {
            drop(compiled);
        }

        // Check plan cache
        let mut plan_cache = self.plan_cache.write().await;
        let plan = plan_cache
            .entry(hash)
            .or_insert_with(|| QueryPlan::new(query.to_string()));

        // Update execution stats
        let execution_time = start.elapsed();
        plan.update_stats(execution_time);

        // Check if we should compile
        let should_compile = match self.strategy {
            CompilationStrategy::AlwaysInterpret => false,
            CompilationStrategy::AlwaysCompile => true,
            CompilationStrategy::Adaptive => plan.is_hot(self.config.compilation_threshold),
            CompilationStrategy::ProfileGuided => {
                plan.is_hot(self.config.compilation_threshold)
                    && self.config.enable_hot_path_detection
            }
        };

        if should_compile {
            info!(
                "Query is hot (executed {} times), triggering JIT compilation",
                plan.execution_count
            );

            // Clone plan for background compilation
            let plan_for_compilation = plan.clone();
            drop(plan_cache); // Release write lock

            // Compile in background if enabled
            if self.config.enable_background_compilation {
                let compiled_cache = self.compiled_cache.clone();
                let stats = self.stats.clone();
                let jit_compiler = self.jit_compiler.clone();
                tokio::spawn(async move {
                    if let Ok(compiled) =
                        Self::compile_query_internal(plan_for_compilation, jit_compiler).await
                    {
                        let mut cache = compiled_cache.write().await;
                        let mut stats_guard = stats.write().await;
                        stats_guard.total_compilation_time += compiled.compilation_time;
                        cache.insert(hash, compiled);
                        info!("Background compilation complete for query hash {}", hash);
                    }
                });
            } else {
                // Synchronous compilation
                let compiled =
                    Self::compile_query_internal(plan_for_compilation, self.jit_compiler.clone())
                        .await?;
                let mut cache = self.compiled_cache.write().await;
                let mut stats = self.stats.write().await;
                stats.total_compilation_time += compiled.compilation_time;
                cache.insert(hash, compiled);
            }
        }

        // Execute in interpreted mode for now
        let result = self.execute_interpreted(query, store).await?;

        // Update interpreted stats
        let mut stats = self.stats.write().await;
        stats.interpreted_count += 1;
        stats.cache_misses += 1;
        stats.avg_interpreted_time = if stats.interpreted_count == 1 {
            execution_time
        } else {
            (stats.avg_interpreted_time * (stats.interpreted_count - 1) as u32 + execution_time)
                / stats.interpreted_count as u32
        };

        Ok(result)
    }

    /// Execute query in interpreted mode.
    ///
    /// Parses `query` as a SPARQL-star `SELECT ... WHERE { ... }` text
    /// (via [`crate::query::QueryParser`]) into a basic graph pattern, then
    /// runs it through the crate's real BGP-based query executor
    /// ([`crate::query::QueryExecutor`]) against `store`, returning the
    /// concrete triples that satisfy the WHERE-clause patterns for each
    /// variable binding found. This replaces the previous placeholder that
    /// ignored `query` entirely and returned the whole store.
    async fn execute_interpreted(
        &self,
        query: &str,
        store: &StarStore,
    ) -> StarResult<Vec<StarTriple>> {
        debug!("Executing query in interpreted mode: {}", query);

        let (_select_vars, bgp) = crate::query::QueryParser::parse_simple_select(query)?;

        if bgp.patterns().is_empty() {
            return Err(crate::StarError::query_error(format!(
                "Interpreted JIT execution could not extract any WHERE-clause \
                 triple pattern from query: {query}"
            )));
        }

        let mut executor = crate::query::QueryExecutor::new(store.clone());
        let construct_patterns = bgp.patterns().to_vec();
        let constructed = executor.execute_construct(&bgp, &construct_patterns)?;

        Ok(constructed.triples().to_vec())
    }

    /// Execute compiled query using JIT-compiled native code
    async fn execute_compiled(
        &self,
        plan: &QueryPlan,
        store: &StarStore,
    ) -> StarResult<Vec<StarTriple>> {
        debug!(
            "Executing JIT-compiled query (executions: {})",
            plan.execution_count
        );

        // Get kernel ID from compiled cache
        let hash = plan.hash;
        let compiled = self.compiled_cache.read().await;

        if let Some(compiled_query) = compiled.get(&hash) {
            if let Some(compiled_code) = &compiled_query.compiled_code {
                // Convert bytes back to kernel ID
                let kernel_id = String::from_utf8_lossy(compiled_code).to_string();
                drop(compiled); // Release lock

                // Execute compiled kernel
                let compiler = self.jit_compiler.read().await;
                return compiler.execute_compiled(&kernel_id, store);
            }
        }

        drop(compiled);

        // Fallback to interpreted if compilation not ready
        debug!("Compiled code not available, falling back to interpreted mode");
        self.execute_interpreted(&plan.query, store).await
    }

    /// Compile a query to native code using scirs2_core::jit
    async fn compile_query_internal(
        plan: QueryPlan,
        jit_compiler: Arc<RwLock<SparqlJitCompiler>>,
    ) -> StarResult<CompiledQuery> {
        let start = Instant::now();

        debug!("Compiling query with scirs2_core::jit: {}", plan.query);

        // Parse query to IR
        let mut compiler = jit_compiler.write().await;
        let ir_plan =
            compiler
                .parse_to_ir(&plan.query)
                .map_err(|e| crate::StarError::QueryError {
                    message: format!("IR parsing failed: {}", e),
                    query_fragment: Some(plan.query.clone()),
                    position: None,
                    suggestion: None,
                })?;

        // Compile IR to native code
        let kernel_id =
            compiler
                .compile_ir(&ir_plan)
                .map_err(|e| crate::StarError::QueryError {
                    message: format!("JIT compilation failed: {}", e),
                    query_fragment: Some(plan.query.clone()),
                    position: None,
                    suggestion: Some("Check query syntax or disable JIT compilation".to_string()),
                })?;

        info!("Successfully compiled query to kernel: {}", kernel_id);

        let compilation_time = start.elapsed();

        // Store kernel ID as compiled code
        let compiled_code = Some(Arc::new(kernel_id.into_bytes()));

        Ok(CompiledQuery {
            plan: QueryPlan {
                mode: ExecutionMode::Compiled,
                compiled_at: Some(Instant::now()),
                ..plan
            },
            compiled_code,
            compilation_time,
            compiled_at: Instant::now(),
        })
    }

    /// Get query execution statistics
    pub async fn stats(&self) -> QueryStats {
        self.stats.read().await.clone()
    }

    /// Clear all caches
    pub async fn clear_caches(&self) {
        let mut plan_cache = self.plan_cache.write().await;
        let mut compiled_cache = self.compiled_cache.write().await;
        plan_cache.clear();
        compiled_cache.clear();
        info!("Caches cleared");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let plan_cache = self.plan_cache.read().await;
        let compiled_cache = self.compiled_cache.read().await;
        (plan_cache.len(), compiled_cache.len())
    }

    /// Invalidate a specific query from cache
    pub async fn invalidate_query(&self, query: &str) {
        let hash = QueryPlan::compute_hash(query);
        let mut plan_cache = self.plan_cache.write().await;
        let mut compiled_cache = self.compiled_cache.write().await;
        plan_cache.remove(&hash);
        compiled_cache.remove(&hash);
        debug!("Invalidated query from cache (hash: {})", hash);
    }

    /// Get hottest queries (most frequently executed)
    pub async fn hot_queries(&self, limit: usize) -> Vec<QueryPlan> {
        let plan_cache = self.plan_cache.read().await;
        let mut plans: Vec<_> = plan_cache.values().cloned().collect();
        plans.sort_by_key(|b| std::cmp::Reverse(b.execution_count));
        plans.into_iter().take(limit).collect()
    }

    /// Get compilation candidates (queries ready to be compiled)
    pub async fn compilation_candidates(&self) -> Vec<QueryPlan> {
        let plan_cache = self.plan_cache.read().await;
        plan_cache
            .values()
            .filter(|p| p.is_hot(self.config.compilation_threshold))
            .cloned()
            .collect()
    }

    /// Precompile a query (eager compilation)
    pub async fn precompile(&self, query: &str) -> StarResult<()> {
        let hash = QueryPlan::compute_hash(query);
        let plan = QueryPlan::new(query.to_string());

        let compiled = Self::compile_query_internal(plan, self.jit_compiler.clone()).await?;

        let mut cache = self.compiled_cache.write().await;
        let mut stats = self.stats.write().await;
        stats.total_compilation_time += compiled.compilation_time;
        cache.insert(hash, compiled);

        info!("Precompiled query (hash: {})", hash);
        Ok(())
    }
}

impl Default for JitQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert_eq!(config.compilation_threshold, 10);
        assert!(config.enable_background_compilation);
        assert_eq!(config.max_cached_plans, 1000);
    }

    #[test]
    fn test_query_plan_creation() {
        let plan = QueryPlan::new("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}".to_string());
        assert_eq!(plan.execution_count, 0);
        assert_eq!(plan.mode, ExecutionMode::Interpreted);
        assert!(plan.compiled_at.is_none());
    }

    #[test]
    fn test_query_plan_stats_update() {
        let mut plan = QueryPlan::new("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}".to_string());
        plan.update_stats(Duration::from_millis(100));
        assert_eq!(plan.execution_count, 1);
        assert_eq!(plan.avg_execution_time, Duration::from_millis(100));

        plan.update_stats(Duration::from_millis(200));
        assert_eq!(plan.execution_count, 2);
        assert_eq!(plan.avg_execution_time, Duration::from_millis(150));
    }

    #[test]
    fn test_query_plan_is_hot() {
        let mut plan = QueryPlan::new("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}".to_string());
        assert!(!plan.is_hot(10));

        for _ in 0..10 {
            plan.update_stats(Duration::from_millis(100));
        }

        assert!(plan.is_hot(10));
    }

    #[test]
    fn test_jit_engine_creation() {
        let engine = JitQueryEngine::new();
        assert_eq!(engine.strategy, CompilationStrategy::Adaptive);
    }

    #[test]
    fn test_jit_engine_strategy() {
        let mut engine = JitQueryEngine::new();
        engine.set_strategy(CompilationStrategy::AlwaysCompile);
        assert_eq!(engine.strategy, CompilationStrategy::AlwaysCompile);
    }

    #[tokio::test]
    async fn test_jit_engine_execute_interpreted() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        let result = engine
            .execute("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}", &store)
            .await
            .unwrap();

        // Empty store, so the (now real) WHERE-clause match is empty too.
        assert!(result.is_empty());

        let stats = engine.stats().await;
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.interpreted_count, 1);
    }

    /// Regression test for the P1 stub fix: interpreted execution must
    /// actually honor the query text (via the real BGP executor) instead of
    /// ignoring it and returning the entire store.
    #[tokio::test]
    async fn test_jit_engine_execute_interpreted_uses_query_pattern() {
        use crate::model::{StarTerm, StarTriple};

        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        let matching = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        let non_matching = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/likes").unwrap(),
            StarTerm::iri("http://example.org/cake").unwrap(),
        );
        store.insert(&matching).unwrap();
        store.insert(&non_matching).unwrap();

        // Query bound to the "knows" predicate only.
        let query = "SELECT ?s ?o\nWHERE {\n?s <http://example.org/knows> ?o .\n}";
        let result = engine.execute(query, &store).await.unwrap();

        assert_eq!(
            result,
            vec![matching],
            "interpreted execution must return only triples matching the query's WHERE clause"
        );
    }

    /// Regression test: a query whose WHERE clause cannot be parsed into any
    /// triple pattern must fail loudly instead of silently returning
    /// unrelated data.
    #[tokio::test]
    async fn test_jit_engine_execute_interpreted_unparseable_query_errors() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        let result = engine.execute("not a sparql query at all", &store).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jit_engine_cache_stats() {
        let engine = JitQueryEngine::new();
        let (plan_count, compiled_count) = engine.cache_stats().await;
        assert_eq!(plan_count, 0);
        assert_eq!(compiled_count, 0);
    }

    #[tokio::test]
    async fn test_jit_engine_clear_caches() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        // Execute a query to populate cache
        engine
            .execute("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}", &store)
            .await
            .unwrap();

        let (plan_count, _) = engine.cache_stats().await;
        assert_eq!(plan_count, 1);

        // Clear caches
        engine.clear_caches().await;

        let (plan_count, compiled_count) = engine.cache_stats().await;
        assert_eq!(plan_count, 0);
        assert_eq!(compiled_count, 0);
    }

    #[tokio::test]
    async fn test_jit_engine_hot_queries() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        // Execute same query multiple times
        for _ in 0..15 {
            engine
                .execute("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}", &store)
                .await
                .unwrap();
        }

        let hot = engine.hot_queries(5).await;
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].execution_count, 15);
    }

    #[tokio::test]
    async fn test_jit_engine_compilation_candidates() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        // Execute query below threshold
        for _ in 0..9 {
            engine
                .execute("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}", &store)
                .await
                .unwrap();
        }

        let candidates = engine.compilation_candidates().await;
        assert_eq!(candidates.len(), 0); // Not hot yet

        // Execute one more time to cross threshold
        engine
            .execute("SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}", &store)
            .await
            .unwrap();

        let candidates = engine.compilation_candidates().await;
        assert_eq!(candidates.len(), 1); // Now hot
    }

    #[tokio::test]
    async fn test_jit_engine_invalidate_query() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();

        let query = "SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}";
        engine.execute(query, &store).await.unwrap();

        let (plan_count, _) = engine.cache_stats().await;
        assert_eq!(plan_count, 1);

        engine.invalidate_query(query).await;

        let (plan_count, _) = engine.cache_stats().await;
        assert_eq!(plan_count, 0);
    }

    /// Regression test: once a query has been "compiled" (a kernel ID is
    /// cached), `execute()` must not permanently fail for that query just
    /// because compiled kernel execution is not yet implemented — it must
    /// fall back to the real interpreted executor instead.
    #[tokio::test]
    async fn test_jit_engine_execute_falls_back_when_compiled_execution_unavailable() {
        let engine = JitQueryEngine::new();
        let store = StarStore::new();
        let query = "SELECT ?s ?p ?o\nWHERE {\n?s ?p ?o .\n}";

        // Force the query into the compiled cache directly, without waiting
        // for the background hot-path trigger.
        engine.precompile(query).await.unwrap();
        let (_plan_count, compiled_count) = engine.cache_stats().await;
        assert_eq!(compiled_count, 1);

        // Compiled kernel execution (SparqlJitCompiler::execute_compiled)
        // always errors today; execute() must still succeed by falling back
        // to interpreted execution instead of surfacing that error.
        let result = engine.execute(query, &store).await;
        assert!(
            result.is_ok(),
            "execute() should fall back to interpreted mode instead of erroring: {result:?}"
        );
    }
}
