//! SPARQL query processing module

pub mod algebra;
pub mod binding_optimizer;
pub mod distributed;
pub mod exec;
pub mod functions;
pub mod gpu;
pub mod jit;
pub mod optimizer;
pub mod parser;
pub mod pattern_optimizer;
pub mod pattern_unification;
pub mod plan;
pub mod property_paths;
pub mod sparql_algebra;
pub mod sparql_query;
pub mod streaming_results;
pub mod update;
pub mod wasm;

// Re-export the enhanced SPARQL algebra and query types from sparql_algebra
pub use crate::{GraphName, Triple};
pub use sparql_algebra::{
    Expression as SparqlExpression, GraphPattern as SparqlGraphPattern, NamedNodePattern,
    PropertyPathExpression, TermPattern as SparqlTermPattern, TriplePattern as SparqlTriplePattern,
};
pub use sparql_query::*;

// Re-export execution plan types
pub use plan::ExecutionPlan;

// Re-export algebra types (without Query to avoid conflict)
// Use explicit aliases to avoid conflicts
pub use algebra::{
    AlgebraTriplePattern, Expression as AlgebraExpression, GraphPattern as AlgebraGraphPattern,
    PropertyPath, Query as AlgebraQuery, TermPattern as AlgebraTermPattern,
};
pub use binding_optimizer::{BindingIterator, BindingOptimizer, BindingSet, Constraint, TermType};
pub use distributed::{DistributedConfig, DistributedQueryEngine, FederatedEndpoint};
pub use gpu::{GpuBackend, GpuQueryExecutor};
pub use jit::{JitCompiler, JitConfig};
pub use optimizer::{AIQueryOptimizer, MultiQueryOptimizer};
pub use parser::*;
pub use pattern_optimizer::{IndexType, OptimizedPatternPlan, PatternExecutor, PatternOptimizer};
pub use pattern_unification::{
    PatternConverter, PatternOptimizer as UnifiedPatternOptimizer, UnifiedTermPattern,
    UnifiedTriplePattern,
};
pub use streaming_results::{
    ConstructResults, SelectResults, Solution as StreamingSolution, SolutionMetadata,
    StreamingConfig, StreamingProgress, StreamingQueryResults, StreamingResultBuilder,
};
pub use update::{UpdateExecutor, UpdateParser};
pub use wasm::{OptimizationLevel, WasmQueryCompiler, WasmTarget};

// TODO: Temporary compatibility layer for SHACL module
pub use exec::{QueryExecutor, QueryResults, Solution};

use crate::model::{Object, Predicate, Subject, Term, Variable};
use crate::OxirsError;
use crate::Store;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

// Import TermPattern for internal usage
use algebra::TermPattern;

/// Type alias for federated query execution future
type FederatedQueryFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<HashMap<String, Term>>, OxirsError>> + 'a>>;

/// Simplified QueryResult for SHACL compatibility
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// SELECT query results
    Select {
        variables: Vec<String>,
        bindings: Vec<HashMap<String, Term>>,
    },
    /// ASK query results
    Ask(bool),
    /// CONSTRUCT query results
    Construct(Vec<crate::model::Triple>),
}

/// Simplified QueryEngine for SHACL compatibility
pub struct QueryEngine {
    /// Query parser for converting SPARQL strings to Query objects
    parser: parser::SparqlParser,
    /// Query executor for executing plans
    executor_config: QueryExecutorConfig,
    /// Federation executor for SERVICE clause support
    federation_executor: Option<crate::federation::FederationExecutor>,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for query execution
#[derive(Debug, Clone)]
pub struct QueryExecutorConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Enable query optimization
    pub optimize: bool,
}

impl Default for QueryExecutorConfig {
    fn default() -> Self {
        Self {
            max_results: 10000,
            timeout_ms: Some(30000),
            optimize: true,
        }
    }
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {
            parser: parser::SparqlParser::new(),
            executor_config: QueryExecutorConfig::default(),
            federation_executor: crate::federation::FederationExecutor::new().ok(),
        }
    }

    /// Create a new query engine with custom configuration
    pub fn with_config(config: QueryExecutorConfig) -> Self {
        Self {
            parser: parser::SparqlParser::new(),
            executor_config: config,
            federation_executor: crate::federation::FederationExecutor::new().ok(),
        }
    }

    /// Enable federation support
    pub fn with_federation(mut self) -> Self {
        self.federation_executor = crate::federation::FederationExecutor::new().ok();
        self
    }

    /// Disable federation support
    pub fn without_federation(mut self) -> Self {
        self.federation_executor = None;
        self
    }

    /// Execute a SPARQL query string against a store
    pub fn query(&self, query_str: &str, store: &dyn Store) -> Result<QueryResult, OxirsError> {
        // Parse the query string
        let parsed_query = self.parser.parse(query_str)?;

        // Execute the parsed query
        self.execute_query(&parsed_query, store)
    }

    /// Execute a parsed Query object against a store
    pub fn execute_query(
        &self,
        query: &sparql_query::Query,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        match query {
            sparql_query::Query::Select {
                pattern, dataset, ..
            } => self.execute_select_query(pattern, dataset.as_ref(), store),
            sparql_query::Query::Ask {
                pattern, dataset, ..
            } => self.execute_ask_query(pattern, dataset.as_ref(), store),
            sparql_query::Query::Construct {
                template,
                pattern,
                dataset,
                ..
            } => self.execute_construct_query(template, pattern, dataset.as_ref(), store),
            sparql_query::Query::Describe {
                pattern, dataset, ..
            } => self.execute_describe_query(pattern, dataset.as_ref(), store),
        }
    }

    /// Execute a SELECT query
    fn execute_select_query(
        &self,
        pattern: &SparqlGraphPattern,
        _dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        let executor = QueryExecutor::new(store);

        // Convert graph pattern to execution plan
        let plan = self.pattern_to_plan(pattern)?;

        // Execute the plan
        let solutions = executor.execute(&plan)?;

        // Extract variable names and convert solutions
        let variables = self.extract_variables(pattern);
        let bindings: Vec<HashMap<String, Term>> = solutions
            .into_iter()
            .take(self.executor_config.max_results)
            .map(|sol| {
                let mut binding = HashMap::new();
                for var in &variables {
                    if let Some(term) = sol.get(var) {
                        binding.insert(var.name().to_string(), term.clone());
                    }
                }
                binding
            })
            .collect();

        Ok(QueryResult::Select {
            variables: variables
                .into_iter()
                .map(|v| v.name().to_string())
                .collect(),
            bindings,
        })
    }

    /// Execute an ASK query
    fn execute_ask_query(
        &self,
        pattern: &SparqlGraphPattern,
        _dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        let executor = QueryExecutor::new(store);

        // Convert graph pattern to execution plan
        let plan = self.pattern_to_plan(pattern)?;

        // Execute the plan
        let solutions = executor.execute(&plan)?;

        // ASK query returns true if there are any solutions
        Ok(QueryResult::Ask(!solutions.is_empty()))
    }

    /// Execute a CONSTRUCT query
    fn execute_construct_query(
        &self,
        template: &[SparqlTriplePattern],
        pattern: &SparqlGraphPattern,
        _dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        let executor = QueryExecutor::new(store);

        // Convert graph pattern to execution plan
        let plan = self.pattern_to_plan(pattern)?;

        // Execute the plan
        let solutions = executor.execute(&plan)?;

        // Construct triples from template and solutions
        let mut triples = Vec::new();
        for solution in solutions.into_iter().take(self.executor_config.max_results) {
            for triple_pattern in template {
                if let Some(triple) = self.instantiate_triple_pattern(triple_pattern, &solution)? {
                    triples.push(triple);
                }
            }
        }

        Ok(QueryResult::Construct(triples))
    }

    /// Execute a DESCRIBE query
    fn execute_describe_query(
        &self,
        pattern: &SparqlGraphPattern,
        _dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        // For now, treat DESCRIBE like CONSTRUCT *
        // This is a simplified implementation
        let executor = QueryExecutor::new(store);

        // Convert graph pattern to execution plan
        let plan = self.pattern_to_plan(pattern)?;

        // Execute the plan
        let solutions = executor.execute(&plan)?;

        // Get all triples involving the found entities
        let mut triples = Vec::new();
        for solution in solutions.into_iter().take(self.executor_config.max_results) {
            // For each bound entity, get all triples where it appears
            for (_, term) in solution.iter() {
                if let Ok(store_quads) =
                    store.find_quads(None, None, None, Some(&GraphName::DefaultGraph))
                {
                    for quad in store_quads {
                        let triple = Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        );
                        if self.triple_involves_term(&triple, term) {
                            triples.push(triple);
                        }
                    }
                }
            }
        }

        triples.dedup();
        Ok(QueryResult::Construct(triples))
    }

    /// Convert a graph pattern to an execution plan
    fn pattern_to_plan(&self, pattern: &SparqlGraphPattern) -> Result<ExecutionPlan, OxirsError> {
        match pattern {
            SparqlGraphPattern::Bgp { patterns } => {
                if patterns.len() == 1 {
                    // Single triple pattern
                    Ok(ExecutionPlan::TripleScan {
                        pattern: self.convert_sparql_triple_pattern(&patterns[0])?,
                    })
                } else {
                    // Multiple patterns - join them
                    let mut plan = ExecutionPlan::TripleScan {
                        pattern: self.convert_sparql_triple_pattern(&patterns[0])?,
                    };

                    for triple_pattern in &patterns[1..] {
                        let right_plan = ExecutionPlan::TripleScan {
                            pattern: self.convert_sparql_triple_pattern(triple_pattern)?,
                        };

                        // Find join variables
                        let join_vars = self.find_join_variables(&plan, &right_plan);

                        plan = ExecutionPlan::HashJoin {
                            left: Box::new(plan),
                            right: Box::new(right_plan),
                            join_vars,
                        };
                    }

                    Ok(plan)
                }
            }
            SparqlGraphPattern::Join { left, right } => {
                let left_plan = self.pattern_to_plan(left)?;
                let right_plan = self.pattern_to_plan(right)?;
                let join_vars = self.find_join_variables(&left_plan, &right_plan);

                Ok(ExecutionPlan::HashJoin {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    join_vars,
                })
            }
            SparqlGraphPattern::Filter { expr, inner } => {
                let input_plan = self.pattern_to_plan(inner)?;
                // Convert sparql_algebra::Expression to algebra::Expression
                let condition = self.convert_expression(expr.clone())?;
                Ok(ExecutionPlan::Filter {
                    input: Box::new(input_plan),
                    condition,
                })
            }
            SparqlGraphPattern::Union { left, right } => {
                let left_plan = self.pattern_to_plan(left)?;
                let right_plan = self.pattern_to_plan(right)?;

                Ok(ExecutionPlan::Union {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                })
            }
            SparqlGraphPattern::Project { inner, variables } => {
                let input_plan = self.pattern_to_plan(inner)?;
                Ok(ExecutionPlan::Project {
                    input: Box::new(input_plan),
                    vars: variables.clone(),
                })
            }
            SparqlGraphPattern::Distinct { inner } => {
                let input_plan = self.pattern_to_plan(inner)?;
                Ok(ExecutionPlan::Distinct {
                    input: Box::new(input_plan),
                })
            }
            SparqlGraphPattern::Slice {
                inner,
                start,
                length,
            } => {
                let input_plan = self.pattern_to_plan(inner)?;
                Ok(ExecutionPlan::Limit {
                    input: Box::new(input_plan),
                    limit: length.unwrap_or(usize::MAX),
                    offset: *start,
                })
            }
            _ => {
                // For unsupported patterns, return an error for now
                Err(OxirsError::Query(format!(
                    "Unsupported graph pattern type: {pattern:?}"
                )))
            }
        }
    }

    /// Convert a SPARQL triple pattern to a model triple pattern
    fn convert_sparql_triple_pattern(
        &self,
        pattern: &SparqlTriplePattern,
    ) -> Result<crate::model::pattern::TriplePattern, OxirsError> {
        use crate::model::pattern::*;

        let subject = match &pattern.subject {
            SparqlTermPattern::Variable(v) => Some(SubjectPattern::Variable(v.clone())),
            SparqlTermPattern::NamedNode(n) => Some(SubjectPattern::NamedNode(n.clone())),
            SparqlTermPattern::BlankNode(b) => Some(SubjectPattern::BlankNode(b.clone())),
            _ => None,
        };

        let predicate = match &pattern.predicate {
            SparqlTermPattern::Variable(v) => Some(PredicatePattern::Variable(v.clone())),
            SparqlTermPattern::NamedNode(n) => Some(PredicatePattern::NamedNode(n.clone())),
            _ => None,
        };

        let object = match &pattern.object {
            SparqlTermPattern::Variable(v) => Some(ObjectPattern::Variable(v.clone())),
            SparqlTermPattern::NamedNode(n) => Some(ObjectPattern::NamedNode(n.clone())),
            SparqlTermPattern::BlankNode(b) => Some(ObjectPattern::BlankNode(b.clone())),
            SparqlTermPattern::Literal(l) => Some(ObjectPattern::Literal(l.clone())),
            #[cfg(feature = "sparql-12")]
            SparqlTermPattern::Triple(_) => {
                // Triple patterns in object position not yet fully supported
                None
            }
        };

        Ok(crate::model::pattern::TriplePattern {
            subject,
            predicate,
            object,
        })
    }

    /// Convert a SPARQL algebra triple pattern to a model triple pattern
    #[allow(dead_code)]
    fn convert_triple_pattern(
        &self,
        pattern: &AlgebraTriplePattern,
    ) -> Result<crate::model::pattern::TriplePattern, OxirsError> {
        use crate::model::pattern::*;

        let subject = match &pattern.subject {
            TermPattern::Variable(v) => Some(SubjectPattern::Variable(v.clone())),
            TermPattern::NamedNode(n) => Some(SubjectPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Some(SubjectPattern::BlankNode(b.clone())),
            _ => None,
        };

        let predicate = match &pattern.predicate {
            TermPattern::Variable(v) => Some(PredicatePattern::Variable(v.clone())),
            TermPattern::NamedNode(n) => Some(PredicatePattern::NamedNode(n.clone())),
            _ => None,
        };

        let object = match &pattern.object {
            TermPattern::Variable(v) => Some(ObjectPattern::Variable(v.clone())),
            TermPattern::NamedNode(n) => Some(ObjectPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Some(ObjectPattern::BlankNode(b.clone())),
            TermPattern::Literal(l) => Some(ObjectPattern::Literal(l.clone())),
        };

        Ok(crate::model::pattern::TriplePattern {
            subject,
            predicate,
            object,
        })
    }

    /// Find variables that appear in both execution plans
    fn find_join_variables(&self, _left: &ExecutionPlan, _right: &ExecutionPlan) -> Vec<Variable> {
        // Simplified implementation - would need to analyze the plans
        Vec::new()
    }

    /// Convert sparql_algebra::Expression to algebra::Expression
    #[allow(clippy::only_used_in_recursion)]
    fn convert_expression(
        &self,
        expr: sparql_algebra::Expression,
    ) -> Result<AlgebraExpression, OxirsError> {
        use sparql_algebra::Expression as SparqlExpr;
        use AlgebraExpression as AlgebraExpr;

        match expr {
            SparqlExpr::NamedNode(n) => Ok(AlgebraExpr::Term(crate::model::Term::NamedNode(n))),
            SparqlExpr::Literal(l) => Ok(AlgebraExpr::Term(crate::model::Term::Literal(l))),
            SparqlExpr::Variable(v) => Ok(AlgebraExpr::Variable(v)),
            SparqlExpr::Or(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::Or(Box::new(left_expr), Box::new(right_expr)))
            }
            SparqlExpr::And(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::And(Box::new(left_expr), Box::new(right_expr)))
            }
            SparqlExpr::Equal(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::Equal(
                    Box::new(left_expr),
                    Box::new(right_expr),
                ))
            }
            SparqlExpr::SameTerm(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::Equal(
                    Box::new(left_expr),
                    Box::new(right_expr),
                )) // Map SameTerm to Equal for now
            }
            SparqlExpr::Greater(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::Greater(
                    Box::new(left_expr),
                    Box::new(right_expr),
                ))
            }
            SparqlExpr::GreaterOrEqual(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::GreaterOrEqual(
                    Box::new(left_expr),
                    Box::new(right_expr),
                ))
            }
            SparqlExpr::Less(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::Less(Box::new(left_expr), Box::new(right_expr)))
            }
            SparqlExpr::LessOrEqual(left, right) => {
                let left_expr = self.convert_expression(*left)?;
                let right_expr = self.convert_expression(*right)?;
                Ok(AlgebraExpr::LessOrEqual(
                    Box::new(left_expr),
                    Box::new(right_expr),
                ))
            }
            SparqlExpr::Not(inner) => {
                let inner_expr = self.convert_expression(*inner)?;
                Ok(AlgebraExpr::Not(Box::new(inner_expr)))
            }
            _ => {
                // For expressions not yet supported, create a placeholder
                Err(OxirsError::Query(format!(
                    "Expression type not yet supported in conversion: {expr:?}"
                )))
            }
        }
    }

    /// Extract all variables from a graph pattern
    fn extract_variables(&self, pattern: &SparqlGraphPattern) -> Vec<Variable> {
        let mut variables = Vec::new();
        self.collect_variables_from_pattern(pattern, &mut variables);
        variables.sort_by_key(|v: &Variable| v.name().to_owned());
        variables.dedup();
        variables
    }

    /// Recursively collect variables from a graph pattern
    fn collect_variables_from_pattern(
        &self,
        pattern: &SparqlGraphPattern,
        variables: &mut Vec<Variable>,
    ) {
        match pattern {
            SparqlGraphPattern::Bgp { patterns } => {
                for triple_pattern in patterns {
                    self.collect_variables_from_triple_pattern(triple_pattern, variables);
                }
            }
            SparqlGraphPattern::Join { left, right } => {
                self.collect_variables_from_pattern(left, variables);
                self.collect_variables_from_pattern(right, variables);
            }
            SparqlGraphPattern::Filter { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables);
            }
            SparqlGraphPattern::Union { left, right } => {
                self.collect_variables_from_pattern(left, variables);
                self.collect_variables_from_pattern(right, variables);
            }
            SparqlGraphPattern::Project {
                inner,
                variables: proj_vars,
            } => {
                self.collect_variables_from_pattern(inner, variables);
                variables.extend(proj_vars.iter().cloned());
            }
            SparqlGraphPattern::Distinct { inner } => {
                self.collect_variables_from_pattern(inner, variables);
            }
            SparqlGraphPattern::Slice { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables);
            }
            _ => {
                // Handle other pattern types as needed
            }
        }
    }

    /// Collect variables from a triple pattern
    fn collect_variables_from_triple_pattern(
        &self,
        pattern: &SparqlTriplePattern,
        variables: &mut Vec<Variable>,
    ) {
        if let SparqlTermPattern::Variable(v) = &pattern.subject {
            variables.push(v.clone());
        }
        if let SparqlTermPattern::Variable(v) = &pattern.predicate {
            variables.push(v.clone());
        }
        if let SparqlTermPattern::Variable(v) = &pattern.object {
            variables.push(v.clone());
        }
    }

    /// Instantiate a triple pattern with a solution
    fn instantiate_triple_pattern(
        &self,
        pattern: &SparqlTriplePattern,
        solution: &Solution,
    ) -> Result<Option<crate::model::Triple>, OxirsError> {
        use crate::model::*;

        let subject = match &pattern.subject {
            SparqlTermPattern::Variable(v) => {
                if let Some(term) = solution.get(v) {
                    match term {
                        Term::NamedNode(n) => Subject::NamedNode(n.clone()),
                        Term::BlankNode(b) => Subject::BlankNode(b.clone()),
                        _ => return Ok(None), // Invalid subject
                    }
                } else {
                    return Ok(None); // Unbound variable
                }
            }
            SparqlTermPattern::NamedNode(n) => Subject::NamedNode(n.clone()),
            SparqlTermPattern::BlankNode(b) => Subject::BlankNode(b.clone()),
            _ => return Ok(None), // Invalid subject pattern
        };

        let predicate = match &pattern.predicate {
            SparqlTermPattern::Variable(v) => {
                if let Some(Term::NamedNode(n)) = solution.get(v) {
                    Predicate::NamedNode(n.clone())
                } else {
                    return Ok(None); // Unbound or invalid predicate
                }
            }
            SparqlTermPattern::NamedNode(n) => Predicate::NamedNode(n.clone()),
            _ => return Ok(None), // Invalid predicate pattern
        };

        let object = match &pattern.object {
            SparqlTermPattern::Variable(v) => {
                if let Some(term) = solution.get(v) {
                    match term {
                        Term::NamedNode(n) => Object::NamedNode(n.clone()),
                        Term::BlankNode(b) => Object::BlankNode(b.clone()),
                        Term::Literal(l) => Object::Literal(l.clone()),
                        _ => return Ok(None), // Invalid object
                    }
                } else {
                    return Ok(None); // Unbound variable
                }
            }
            SparqlTermPattern::NamedNode(n) => Object::NamedNode(n.clone()),
            SparqlTermPattern::BlankNode(b) => Object::BlankNode(b.clone()),
            SparqlTermPattern::Literal(l) => Object::Literal(l.clone()),
            #[cfg(feature = "sparql-12")]
            SparqlTermPattern::Triple(_) => {
                // Triple patterns in object position not yet fully supported
                return Ok(None);
            }
        };

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    /// Execute a SPARQL query string against a store (async version with federation support)
    pub async fn query_async(
        &self,
        query_str: &str,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        // Parse the query string
        let parsed_query = self.parser.parse(query_str)?;

        // Execute the parsed query
        self.execute_query_async(&parsed_query, store).await
    }

    /// Execute a parsed Query object against a store (async version)
    pub async fn execute_query_async(
        &self,
        query: &sparql_query::Query,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        match query {
            sparql_query::Query::Select {
                pattern, dataset, ..
            } => {
                self.execute_select_query_async(pattern, dataset.as_ref(), store)
                    .await
            }
            sparql_query::Query::Ask {
                pattern, dataset, ..
            } => {
                self.execute_ask_query_async(pattern, dataset.as_ref(), store)
                    .await
            }
            sparql_query::Query::Construct {
                template,
                pattern,
                dataset,
                ..
            } => {
                self.execute_construct_query_async(template, pattern, dataset.as_ref(), store)
                    .await
            }
            sparql_query::Query::Describe {
                pattern, dataset, ..
            } => {
                self.execute_describe_query_async(pattern, dataset.as_ref(), store)
                    .await
            }
        }
    }

    /// Execute a SELECT query (async version with federation support)
    async fn execute_select_query_async(
        &self,
        pattern: &SparqlGraphPattern,
        _dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        // Check if pattern contains SERVICE clause
        if self.contains_service_clause(pattern) {
            // Use federation executor
            return self.execute_federated_select(pattern, store).await;
        }

        // Fall back to regular execution
        self.execute_select_query(pattern, _dataset, store)
    }

    /// Execute an ASK query (async version)
    async fn execute_ask_query_async(
        &self,
        pattern: &SparqlGraphPattern,
        dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        if self.contains_service_clause(pattern) {
            let result = self.execute_federated_select(pattern, store).await?;
            if let QueryResult::Select { bindings, .. } = result {
                return Ok(QueryResult::Ask(!bindings.is_empty()));
            }
        }
        self.execute_ask_query(pattern, dataset, store)
    }

    /// Execute a CONSTRUCT query (async version)
    async fn execute_construct_query_async(
        &self,
        template: &[SparqlTriplePattern],
        pattern: &SparqlGraphPattern,
        dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        if self.contains_service_clause(pattern) {
            return Err(OxirsError::Federation(
                "CONSTRUCT with SERVICE is not yet fully supported".to_string(),
            ));
        }
        self.execute_construct_query(template, pattern, dataset, store)
    }

    /// Execute a DESCRIBE query (async version)
    async fn execute_describe_query_async(
        &self,
        pattern: &SparqlGraphPattern,
        dataset: Option<&QueryDataset>,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        if self.contains_service_clause(pattern) {
            return Err(OxirsError::Federation(
                "DESCRIBE with SERVICE is not yet fully supported".to_string(),
            ));
        }
        self.execute_describe_query(pattern, dataset, store)
    }

    /// Check if a pattern contains a SERVICE clause
    fn contains_service_clause(&self, pattern: &SparqlGraphPattern) -> bool {
        // Check for SERVICE patterns using recursive pattern traversal
        matches!(pattern, SparqlGraphPattern::Service { .. })
            || Self::pattern_contains_service_recursive(pattern)
    }

    /// Recursively check for SERVICE in nested patterns
    fn pattern_contains_service_recursive(pattern: &SparqlGraphPattern) -> bool {
        match pattern {
            SparqlGraphPattern::Service { .. } => true,
            SparqlGraphPattern::Join { left, right }
            | SparqlGraphPattern::Union { left, right } => {
                Self::pattern_contains_service_recursive(left)
                    || Self::pattern_contains_service_recursive(right)
            }
            SparqlGraphPattern::Filter { inner, .. }
            | SparqlGraphPattern::Distinct { inner }
            | SparqlGraphPattern::Reduced { inner }
            | SparqlGraphPattern::Project { inner, .. } => {
                Self::pattern_contains_service_recursive(inner)
            }
            SparqlGraphPattern::LeftJoin { left, right, .. } => {
                Self::pattern_contains_service_recursive(left)
                    || Self::pattern_contains_service_recursive(right)
            }
            _ => false,
        }
    }

    /// Execute a federated SELECT query
    async fn execute_federated_select(
        &self,
        pattern: &SparqlGraphPattern,
        store: &dyn Store,
    ) -> Result<QueryResult, OxirsError> {
        let federation_executor = self.federation_executor.as_ref().ok_or_else(|| {
            OxirsError::Federation("Federation executor not available".to_string())
        })?;

        // Start with empty bindings from local store
        let local_bindings = Vec::new();

        // Execute the federated pattern
        let bindings = self
            .execute_pattern_with_federation(pattern, local_bindings, federation_executor, store)
            .await?;

        // Extract variable names
        let variables = self
            .extract_variables(pattern)
            .into_iter()
            .map(|v| v.name().to_string())
            .collect();

        Ok(QueryResult::Select {
            variables,
            bindings,
        })
    }

    /// Execute a pattern with federation support
    fn execute_pattern_with_federation<'a>(
        &'a self,
        pattern: &'a SparqlGraphPattern,
        current_bindings: Vec<HashMap<String, Term>>,
        federation_executor: &'a crate::federation::FederationExecutor,
        store: &'a dyn Store,
    ) -> FederatedQueryFuture<'a> {
        Box::pin(async move {
            let current_bindings = current_bindings;
            match pattern {
                SparqlGraphPattern::Service {
                    name,
                    inner,
                    silent,
                } => {
                    // Execute SERVICE clause
                    let remote_bindings = federation_executor
                        .execute_service(name, inner, *silent, &current_bindings)
                        .await?;

                    // Merge local and remote bindings
                    if current_bindings.is_empty() {
                        Ok(remote_bindings)
                    } else {
                        Ok(federation_executor.merge_bindings(current_bindings, remote_bindings))
                    }
                }
                SparqlGraphPattern::Join { left, right } => {
                    // Execute left side first
                    let left_bindings = self
                        .execute_pattern_with_federation(
                            left,
                            current_bindings,
                            federation_executor,
                            store,
                        )
                        .await?;

                    // Then execute right side with left bindings
                    self.execute_pattern_with_federation(
                        right,
                        left_bindings,
                        federation_executor,
                        store,
                    )
                    .await
                }
                _ => {
                    // For non-federated patterns, use regular execution
                    let executor = QueryExecutor::new(store);
                    let plan = self.pattern_to_plan(pattern)?;
                    let solutions = executor.execute(&plan)?;

                    let bindings: Vec<HashMap<String, Term>> = solutions
                        .into_iter()
                        .take(self.executor_config.max_results)
                        .map(|sol| {
                            sol.iter()
                                .map(|(var, term)| (var.name().to_string(), term.clone()))
                                .collect()
                        })
                        .collect();

                    if current_bindings.is_empty() {
                        Ok(bindings)
                    } else {
                        // Merge current and new bindings
                        Ok(federation_executor.merge_bindings(current_bindings, bindings))
                    }
                }
            }
        })
    }

    /// Check if a triple involves a specific term
    fn triple_involves_term(&self, triple: &crate::model::Triple, term: &Term) -> bool {
        match term {
            Term::NamedNode(n) => {
                matches!(triple.subject(), Subject::NamedNode(sn) if sn == n)
                    || matches!(triple.predicate(), Predicate::NamedNode(pn) if pn == n)
                    || matches!(triple.object(), Object::NamedNode(on) if on == n)
            }
            Term::BlankNode(b) => {
                matches!(triple.subject(), Subject::BlankNode(sb) if sb == b)
                    || matches!(triple.object(), Object::BlankNode(ob) if ob == b)
            }
            Term::Literal(l) => {
                matches!(triple.object(), Object::Literal(ol) if ol == l)
            }
            _ => false,
        }
    }
}
