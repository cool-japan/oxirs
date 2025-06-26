//! # OxiRS ARQ
//!
//! Jena-style SPARQL algebra with extension points and query optimization.
//!
//! This crate provides advanced SPARQL query processing capabilities including
//! query algebra, optimization, and extension points for custom functions.

use anyhow::Result;

pub mod algebra;
pub mod builtin_fixed;
pub mod executor;
pub mod extensions;
pub mod optimizer;
pub mod path;
pub use builtin_fixed as builtin;
pub mod query;
pub mod term;
pub mod expression;

// Re-export main types for convenience
pub use algebra::*;
pub use executor::*;
pub use extensions::*;
pub use optimizer::*;
pub use path::*;
pub use query::*;
pub use term::*;
pub use expression::*;

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
    pub fn with_config(
        executor_config: executor::ExecutionContext,
        optimizer_config: optimizer::OptimizerConfig,
    ) -> Result<Self> {
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
    pub fn execute_query(
        &mut self,
        query_str: &str,
        dataset: &dyn executor::Dataset,
    ) -> Result<(algebra::Solution, executor::ExecutionStats)> {
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
        let mut algebra = query.where_clause;

        // Apply query modifiers in reverse order of precedence

        // 1. Apply GROUP BY and aggregates if present
        if !query.group_by.is_empty() {
            algebra = Algebra::Group {
                pattern: Box::new(algebra),
                variables: query.group_by,
                aggregates: Vec::new(), // TODO: Extract from select variables
            };
        }

        // 2. Apply HAVING clause if present
        if let Some(having_condition) = query.having {
            algebra = Algebra::Having {
                pattern: Box::new(algebra),
                condition: having_condition,
            };
        }

        // 3. Apply projection (SELECT variables)
        match query.query_type {
            query::QueryType::Select => {
                if !query.select_variables.is_empty() {
                    algebra = Algebra::Project {
                        pattern: Box::new(algebra),
                        variables: query.select_variables,
                    };
                }

                // Apply DISTINCT or REDUCED
                if query.distinct {
                    algebra = Algebra::Distinct {
                        pattern: Box::new(algebra),
                    };
                } else if query.reduced {
                    algebra = Algebra::Reduced {
                        pattern: Box::new(algebra),
                    };
                }
            }
            query::QueryType::Construct => {
                // For CONSTRUCT queries, we need to handle the construct template
                // This is a simplified implementation
                if query.distinct {
                    algebra = Algebra::Distinct {
                        pattern: Box::new(algebra),
                    };
                }
            }
            query::QueryType::Ask | query::QueryType::Describe => {
                // ASK and DESCRIBE don't need projection modifications
            }
        }

        // 4. Apply ORDER BY if present
        if !query.order_by.is_empty() {
            algebra = Algebra::OrderBy {
                pattern: Box::new(algebra),
                conditions: query.order_by,
            };
        }

        // 5. Apply SLICE (LIMIT and OFFSET) if present
        if query.limit.is_some() || query.offset.is_some() {
            algebra = Algebra::Slice {
                pattern: Box::new(algebra),
                offset: query.offset,
                limit: query.limit,
            };
        }

        Ok(algebra)
    }
}

impl Default for SparqlEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default SPARQL engine")
    }
}
