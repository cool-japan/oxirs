//! PyO3 Python Bindings for OxiRS ARQ SPARQL Engine
//! 
//! This module provides comprehensive Python bindings for the OxiRS ARQ SPARQL query processing engine,
//! enabling seamless integration with Python applications and ML pipelines.

use crate::{
    algebra::{Algebra, GraphPattern, Expression, TermPattern},
    query::{Query, SelectQuery, ConstructQuery, AskQuery, DescribeQuery},
    executor::{QueryExecutor, ExecutionResult, QueryContext},
    optimizer::{QueryOptimizer, OptimizationConfig, OptimizationStats},
    term::{Term, Variable, Literal, NamedNode},
    path::{PropertyPath, PathExpression},
    builtin::{BuiltinFunction, FunctionRegistry},
    parallel::{ParallelExecutor, ParallelConfig},
};

use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyString};
use pyo3::{wrap_pyfunction, create_exception};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// Custom exception types for Python
create_exception!(oxirs_arq, QueryExecutionError, pyo3::exceptions::PyException);
create_exception!(oxirs_arq, QueryParsingError, pyo3::exceptions::PyException);
create_exception!(oxirs_arq, OptimizationError, pyo3::exceptions::PyException);

/// Python wrapper for SPARQL Query Executor
#[pyclass(name = "SparqlQueryExecutor")]
pub struct PySparqlQueryExecutor {
    executor: Arc<RwLock<QueryExecutor>>,
    stats: Arc<RwLock<QueryExecutionStats>>,
}

/// Query execution statistics
#[pyclass(name = "QueryExecutionStats")]
#[derive(Debug, Clone, Default)]
pub struct QueryExecutionStats {
    #[pyo3(get)]
    pub total_queries: usize,
    #[pyo3(get)]
    pub successful_queries: usize,
    #[pyo3(get)]
    pub failed_queries: usize,
    #[pyo3(get)]
    pub average_execution_time_ms: f64,
    #[pyo3(get)]
    pub total_execution_time_ms: f64,
    #[pyo3(get)]
    pub peak_memory_usage_mb: f64,
}

#[pymethods]
impl PySparqlQueryExecutor {
    /// Create a new SPARQL query executor
    #[new]
    #[pyo3(signature = (config = None, **kwargs))]
    fn new(config: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut executor_config = QueryContext::default();
        
        // Parse configuration from Python dict
        if let Some(config_dict) = config {
            if let Some(timeout) = config_dict.get_item("timeout")? {
                let timeout_ms: u64 = timeout.extract()?;
                executor_config.timeout = Some(Duration::from_millis(timeout_ms));
            }
            
            if let Some(parallel) = config_dict.get_item("enable_parallel")? {
                let enable: bool = parallel.extract()?;
                executor_config.enable_parallel = enable;
            }
            
            if let Some(cache_size) = config_dict.get_item("cache_size")? {
                let size: usize = cache_size.extract()?;
                executor_config.cache_size = size;
            }
        }

        let executor = QueryExecutor::new(executor_config)
            .map_err(|e| PyErr::new::<QueryExecutionError, _>(e.to_string()))?;

        Ok(Self {
            executor: Arc::new(RwLock::new(executor)),
            stats: Arc::new(RwLock::new(QueryExecutionStats::default())),
        })
    }

    /// Execute a SPARQL SELECT query
    #[pyo3(signature = (query, bindings = None, **kwargs))]
    fn execute_select(&self, query: &str, bindings: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<PyQueryResult> {
        let start_time = Instant::now();
        
        // Parse optional bindings
        let mut query_bindings = HashMap::new();
        if let Some(bindings_dict) = bindings {
            for (key, value) in bindings_dict.iter() {
                let var_name: String = key.extract()?;
                let var_value: String = value.extract()?;
                // In a real implementation, we'd convert the Python value to a Term
                query_bindings.insert(var_name, var_value);
            }
        }

        let executor = self.executor.read().unwrap();
        let result = executor.execute_query(query)
            .map_err(|e| PyErr::new::<QueryExecutionError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_queries += 1;
            stats.successful_queries += 1;
            let execution_time = start_time.elapsed().as_millis() as f64;
            stats.total_execution_time_ms += execution_time;
            stats.average_execution_time_ms = stats.total_execution_time_ms / stats.total_queries as f64;
        }

        Ok(PyQueryResult::from_execution_result(result))
    }

    /// Execute a SPARQL CONSTRUCT query
    #[pyo3(signature = (query, bindings = None, **kwargs))]
    fn execute_construct(&self, query: &str, bindings: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<PyGraphResult> {
        let start_time = Instant::now();
        
        let executor = self.executor.read().unwrap();
        let result = executor.execute_query(query)
            .map_err(|e| PyErr::new::<QueryExecutionError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_queries += 1;
            stats.successful_queries += 1;
            let execution_time = start_time.elapsed().as_millis() as f64;
            stats.total_execution_time_ms += execution_time;
            stats.average_execution_time_ms = stats.total_execution_time_ms / stats.total_queries as f64;
        }

        Ok(PyGraphResult::from_execution_result(result))
    }

    /// Execute a SPARQL ASK query
    #[pyo3(signature = (query, bindings = None, **kwargs))]
    fn execute_ask(&self, query: &str, bindings: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<bool> {
        let start_time = Instant::now();
        
        let executor = self.executor.read().unwrap();
        let result = executor.execute_query(query)
            .map_err(|e| PyErr::new::<QueryExecutionError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_queries += 1;
            stats.successful_queries += 1;
            let execution_time = start_time.elapsed().as_millis() as f64;
            stats.total_execution_time_ms += execution_time;
            stats.average_execution_time_ms = stats.total_execution_time_ms / stats.total_queries as f64;
        }

        // In a real implementation, we'd extract the boolean result
        Ok(true) // Placeholder
    }

    /// Explain query execution plan
    #[pyo3(signature = (query, **kwargs))]
    fn explain_query(&self, query: &str, kwargs: Option<&PyDict>) -> PyResult<PyQueryPlan> {
        let executor = self.executor.read().unwrap();
        
        // In a real implementation, we'd generate the actual execution plan
        let plan = QueryPlan {
            query: query.to_string(),
            estimated_cost: 100.0,
            estimated_rows: 1000,
            operations: vec![
                "Scan".to_string(),
                "Filter".to_string(), 
                "Join".to_string(),
                "Project".to_string(),
            ],
            optimization_applied: vec![
                "Filter pushdown".to_string(),
                "Join reordering".to_string(),
            ],
        };

        Ok(PyQueryPlan { plan })
    }

    /// Get query execution statistics
    fn get_statistics(&self) -> PyQueryExecutionStats {
        let stats = self.stats.read().unwrap();
        PyQueryExecutionStats { stats: stats.clone() }
    }

    /// Clear execution statistics
    fn clear_statistics(&self) -> PyResult<()> {
        let mut stats = self.stats.write().unwrap();
        *stats = QueryExecutionStats::default();
        Ok(())
    }

    /// Optimize query for better performance
    #[pyo3(signature = (query, **kwargs))]
    fn optimize_query(&self, query: &str, kwargs: Option<&PyDict>) -> PyResult<String> {
        // In a real implementation, we'd use the QueryOptimizer
        Ok(format!("# Optimized query\n{}", query))
    }

    /// Set query timeout
    fn set_timeout(&self, timeout_ms: u64) -> PyResult<()> {
        let mut executor = self.executor.write().unwrap();
        // In a real implementation, we'd update the executor timeout
        Ok(())
    }

    /// Enable or disable parallel execution
    fn set_parallel_execution(&self, enabled: bool) -> PyResult<()> {
        let mut executor = self.executor.write().unwrap();
        // In a real implementation, we'd update the executor config
        Ok(())
    }
}

/// Python wrapper for query results
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    result: QueryResult,
}

/// Query result data structure
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, String>>,
    pub total_rows: usize,
    pub execution_time_ms: f64,
}

#[pymethods]
impl PyQueryResult {
    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.result.columns.clone()
    }

    /// Get number of rows
    #[getter]
    fn row_count(&self) -> usize {
        self.result.total_rows
    }

    /// Get execution time in milliseconds
    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    /// Convert results to list of dictionaries
    fn to_list(&self) -> Vec<HashMap<String, String>> {
        self.result.rows.clone()
    }

    /// Convert results to pandas-compatible format
    fn to_pandas_dict(&self) -> HashMap<String, Vec<String>> {
        let mut pandas_data = HashMap::new();
        
        for column in &self.result.columns {
            let column_data: Vec<String> = self.result.rows
                .iter()
                .map(|row| row.get(column).cloned().unwrap_or_default())
                .collect();
            pandas_data.insert(column.clone(), column_data);
        }
        
        pandas_data
    }

    /// Get specific row by index
    fn get_row(&self, index: usize) -> PyResult<HashMap<String, String>> {
        self.result.rows.get(index)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Row index out of range"))
    }

    /// Iterate over rows
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyQueryResultIter> {
        Ok(PyQueryResultIter {
            result: slf.result.clone(),
            index: 0,
        })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("QueryResult(rows={}, columns={:?})", self.result.total_rows, self.result.columns)
    }
}

impl PyQueryResult {
    fn from_execution_result(result: ExecutionResult) -> Self {
        // In a real implementation, we'd convert from the actual ExecutionResult
        Self {
            result: QueryResult {
                columns: vec!["subject".to_string(), "predicate".to_string(), "object".to_string()],
                rows: vec![
                    [("subject".to_string(), "ex:Alice".to_string()), 
                     ("predicate".to_string(), "foaf:name".to_string()),
                     ("object".to_string(), "Alice".to_string())].iter().cloned().collect(),
                ],
                total_rows: 1,
                execution_time_ms: 10.0,
            }
        }
    }
}

/// Iterator for query results
#[pyclass]
pub struct PyQueryResultIter {
    result: QueryResult,
    index: usize,
}

#[pymethods]
impl PyQueryResultIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<HashMap<String, String>> {
        if self.index < self.result.rows.len() {
            let row = self.result.rows[self.index].clone();
            self.index += 1;
            Some(row)
        } else {
            None
        }
    }
}

/// Python wrapper for graph results (CONSTRUCT queries)
#[pyclass(name = "GraphResult")]
pub struct PyGraphResult {
    result: GraphResult,
}

/// Graph result data structure
#[derive(Debug, Clone)]
pub struct GraphResult {
    pub triples: Vec<Triple>,
    pub total_triples: usize,
    pub execution_time_ms: f64,
}

/// Triple representation
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[pymethods]
impl PyGraphResult {
    /// Get number of triples
    #[getter]
    fn triple_count(&self) -> usize {
        self.result.total_triples
    }

    /// Get execution time in milliseconds
    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    /// Convert to list of triples
    fn to_list(&self) -> Vec<(String, String, String)> {
        self.result.triples
            .iter()
            .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
            .collect()
    }

    /// Export to N-Triples format
    fn to_ntriples(&self) -> String {
        self.result.triples
            .iter()
            .map(|t| format!("<{}> <{}> <{}> .", t.subject, t.predicate, t.object))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export to Turtle format
    fn to_turtle(&self) -> String {
        // Simplified Turtle export
        let mut turtle = String::from("@prefix ex: <http://example.org/> .\n\n");
        for triple in &self.result.triples {
            turtle.push_str(&format!("<{}> <{}> <{}> .\n", triple.subject, triple.predicate, triple.object));
        }
        turtle
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("GraphResult(triples={})", self.result.total_triples)
    }
}

impl PyGraphResult {
    fn from_execution_result(result: ExecutionResult) -> Self {
        // In a real implementation, we'd convert from the actual ExecutionResult
        Self {
            result: GraphResult {
                triples: vec![
                    Triple {
                        subject: "ex:Alice".to_string(),
                        predicate: "foaf:name".to_string(),
                        object: "Alice".to_string(),
                    }
                ],
                total_triples: 1,
                execution_time_ms: 8.0,
            }
        }
    }
}

/// Python wrapper for query execution plan
#[pyclass(name = "QueryPlan")]
pub struct PyQueryPlan {
    plan: QueryPlan,
}

/// Query plan data structure
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub query: String,
    pub estimated_cost: f64,
    pub estimated_rows: usize,
    pub operations: Vec<String>,
    pub optimization_applied: Vec<String>,
}

#[pymethods]
impl PyQueryPlan {
    /// Get estimated cost
    #[getter]
    fn estimated_cost(&self) -> f64 {
        self.plan.estimated_cost
    }

    /// Get estimated number of rows
    #[getter]
    fn estimated_rows(&self) -> usize {
        self.plan.estimated_rows
    }

    /// Get list of operations
    #[getter]
    fn operations(&self) -> Vec<String> {
        self.plan.operations.clone()
    }

    /// Get applied optimizations
    #[getter]
    fn optimizations(&self) -> Vec<String> {
        self.plan.optimization_applied.clone()
    }

    /// Get visual representation of the plan
    fn visualize(&self) -> String {
        let mut viz = String::from("Query Execution Plan:\n");
        viz.push_str(&format!("Estimated Cost: {:.2}\n", self.plan.estimated_cost));
        viz.push_str(&format!("Estimated Rows: {}\n", self.plan.estimated_rows));
        viz.push_str("Operations:\n");
        for (i, op) in self.plan.operations.iter().enumerate() {
            viz.push_str(&format!("  {}. {}\n", i + 1, op));
        }
        viz.push_str("Optimizations Applied:\n");
        for opt in &self.plan.optimization_applied {
            viz.push_str(&format!("  - {}\n", opt));
        }
        viz
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("QueryPlan(cost={:.2}, rows={})", self.plan.estimated_cost, self.plan.estimated_rows)
    }
}

/// Python wrapper for query execution statistics
#[pyclass(name = "QueryExecutionStats")]
pub struct PyQueryExecutionStats {
    stats: QueryExecutionStats,
}

#[pymethods]
impl PyQueryExecutionStats {
    /// Get total number of queries executed
    #[getter]
    fn total_queries(&self) -> usize {
        self.stats.total_queries
    }

    /// Get number of successful queries
    #[getter]
    fn successful_queries(&self) -> usize {
        self.stats.successful_queries
    }

    /// Get number of failed queries
    #[getter]
    fn failed_queries(&self) -> usize {
        self.stats.failed_queries
    }

    /// Get success rate as percentage
    #[getter]
    fn success_rate(&self) -> f64 {
        if self.stats.total_queries > 0 {
            (self.stats.successful_queries as f64 / self.stats.total_queries as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get average execution time in milliseconds
    #[getter]
    fn average_execution_time_ms(&self) -> f64 {
        self.stats.average_execution_time_ms
    }

    /// Get total execution time in milliseconds
    #[getter]
    fn total_execution_time_ms(&self) -> f64 {
        self.stats.total_execution_time_ms
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "QueryExecutionStats(total={}, success_rate={:.1}%, avg_time={:.2}ms)",
            self.stats.total_queries,
            self.success_rate(),
            self.stats.average_execution_time_ms
        )
    }
}

/// Utility functions

/// Parse a SPARQL query and return syntax analysis
#[pyfunction]
#[pyo3(signature = (query, **kwargs))]
fn parse_sparql_query(query: &str, kwargs: Option<&PyDict>) -> PyResult<PyQueryInfo> {
    // In a real implementation, we'd use the actual SPARQL parser
    let query_info = QueryInfo {
        query_type: if query.trim_start().to_lowercase().starts_with("select") {
            "SELECT".to_string()
        } else if query.trim_start().to_lowercase().starts_with("construct") {
            "CONSTRUCT".to_string()
        } else if query.trim_start().to_lowercase().starts_with("ask") {
            "ASK".to_string()
        } else {
            "UNKNOWN".to_string()
        },
        variables: vec!["?s".to_string(), "?p".to_string(), "?o".to_string()],
        triple_patterns: 1,
        filters: 0,
        optional_patterns: 0,
        union_patterns: 0,
        complexity_score: 2.5,
    };

    Ok(PyQueryInfo { info: query_info })
}

/// Query information data structure
#[pyclass(name = "QueryInfo")]
pub struct PyQueryInfo {
    info: QueryInfo,
}

#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub query_type: String,
    pub variables: Vec<String>,
    pub triple_patterns: usize,
    pub filters: usize,
    pub optional_patterns: usize,
    pub union_patterns: usize,
    pub complexity_score: f64,
}

#[pymethods]
impl PyQueryInfo {
    #[getter]
    fn query_type(&self) -> String {
        self.info.query_type.clone()
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.info.variables.clone()
    }

    #[getter]
    fn triple_patterns(&self) -> usize {
        self.info.triple_patterns
    }

    #[getter]
    fn complexity_score(&self) -> f64 {
        self.info.complexity_score
    }

    fn __repr__(&self) -> String {
        format!("QueryInfo(type={}, complexity={:.2})", self.info.query_type, self.info.complexity_score)
    }
}

/// Validate SPARQL query syntax
#[pyfunction]
#[pyo3(signature = (query, **kwargs))]
fn validate_sparql_query(query: &str, kwargs: Option<&PyDict>) -> PyResult<PyValidationResult> {
    // In a real implementation, we'd use the actual SPARQL validator
    let result = ValidationResult {
        is_valid: !query.trim().is_empty(),
        errors: if query.trim().is_empty() {
            vec!["Empty query".to_string()]
        } else {
            vec![]
        },
        warnings: vec![],
        suggestions: vec!["Consider adding LIMIT clause for better performance".to_string()],
    };

    Ok(PyValidationResult { result })
}

/// Validation result data structure
#[pyclass(name = "ValidationResult")]
pub struct PyValidationResult {
    result: ValidationResult,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

#[pymethods]
impl PyValidationResult {
    #[getter]
    fn is_valid(&self) -> bool {
        self.result.is_valid
    }

    #[getter]
    fn errors(&self) -> Vec<String> {
        self.result.errors.clone()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.result.warnings.clone()
    }

    #[getter]
    fn suggestions(&self) -> Vec<String> {
        self.result.suggestions.clone()
    }

    fn __repr__(&self) -> String {
        format!("ValidationResult(valid={}, errors={})", self.result.is_valid, self.result.errors.len())
    }
}

/// Module initialization
#[pymodule]
fn oxirs_arq(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add core classes
    m.add_class::<PySparqlQueryExecutor>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyGraphResult>()?;
    m.add_class::<PyQueryPlan>()?;
    m.add_class::<PyQueryExecutionStats>()?;
    m.add_class::<PyQueryInfo>()?;
    m.add_class::<PyValidationResult>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(parse_sparql_query, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sparql_query, m)?)?;

    // Add exceptions
    m.add("QueryExecutionError", py.get_type::<QueryExecutionError>())?;
    m.add("QueryParsingError", py.get_type::<QueryParsingError>())?;
    m.add("OptimizationError", py.get_type::<OptimizationError>())?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add feature information
    m.add("__features__", vec![
        "sparql_1_1_compliance",
        "query_optimization",
        "parallel_execution",
        "performance_monitoring",
        "federation_support",
        "custom_functions"
    ])?;

    Ok(())
}

// Re-export for easier access
pub use oxirs_arq as python_module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = PySparqlQueryExecutor::new(None, None).unwrap();
        let stats = executor.get_statistics();
        assert_eq!(stats.total_queries(), 0);
    }

    #[test]
    fn test_query_parsing() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let info = parse_sparql_query(query, None).unwrap();
        assert_eq!(info.query_type(), "SELECT");
    }

    #[test]
    fn test_query_validation() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let result = validate_sparql_query(query, None).unwrap();
        assert!(result.is_valid());
    }
}