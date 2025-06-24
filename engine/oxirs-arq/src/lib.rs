//! # OxiRS ARQ
//!
//! Jena-style SPARQL algebra with extension points and query optimization.
//!
//! This crate provides advanced SPARQL query processing capabilities including
//! query algebra, optimization, and extension points for custom functions.

use anyhow::Result;

pub mod algebra;
pub mod optimizer;
pub mod executor;
pub mod extensions;
pub mod path;
pub mod builtin_fixed as builtin;
pub mod query;

// Re-export main types for convenience
pub use algebra::*;
pub use optimizer::*;
pub use executor::*;
pub use extensions::*;
pub use path::*;
pub use query::*;

/// SPARQL Query Engine - High-level interface
pub struct SparqlEngine {
    executor: QueryExecutor,
    optimizer: QueryOptimizer,
    extensions: ExtensionRegistry,
    parser: query::QueryParser,
}

impl SparqlEngine {
    /// Create a new SPARQL engine with default configuration
    pub fn new() -> Result<Self> {
        let mut extensions = ExtensionRegistry::new();
        builtin::register_builtin_functions(&extensions)?;
        
        Ok(Self {
            executor: QueryExecutor::new(),
            optimizer: QueryOptimizer::new(),
            extensions,
            parser: query::QueryParser::new(),
        })
    }
    
    /// Create a new SPARQL engine with custom configuration
    pub fn with_config(executor_config: executor::ExecutionContext, optimizer_config: optimizer::OptimizerConfig) -> Result<Self> {
        let mut extensions = ExtensionRegistry::new();
        builtin::register_builtin_functions(&extensions)?;
        
        Ok(Self {
            executor: QueryExecutor::with_context(executor_config),
            optimizer: QueryOptimizer::with_config(optimizer_config),
            extensions,
            parser: query::QueryParser::new(),
        })
    }
    
    /// Parse and execute a SPARQL query
    pub fn execute_query(&mut self, query_str: &str, dataset: &dyn executor::Dataset) -> Result<(algebra::Solution, executor::ExecutionStats)> {
        // Parse query
        let query = self.parser.parse(query_str)?;
        
        // Convert to algebra
        let algebra = self.convert_query_to_algebra(query)?;
        
        // Optimize algebra
        let optimized_algebra = self.optimizer.optimize(algebra)?;
        
        // Execute
        self.executor.execute(&optimized_algebra, dataset)
    }
    
    /// Register a custom function
    pub fn register_function<F>(&self, function: F) -> Result<()>
    where
        F: extensions::CustomFunction + 'static,
    {
        self.extensions.register_function(function)
    }
    
    /// Register a custom aggregate
    pub fn register_aggregate<A>(&self, aggregate: A) -> Result<()>
    where
        A: extensions::CustomAggregate + 'static,
    {
        self.extensions.register_aggregate(aggregate)
    }
    
    /// Convert parsed query to algebra expression
    fn convert_query_to_algebra(&self, query: query::Query) -> Result<Algebra> {
        // TODO: Implement proper query-to-algebra conversion
        // For now, return the where clause
        Ok(query.where_clause)
    }
}

impl Default for SparqlEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default SPARQL engine")
    }
}