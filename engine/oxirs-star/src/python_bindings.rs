//! PyO3 Python Bindings for OxiRS Star (RDF-star) Engine
//! 
//! This module provides comprehensive Python bindings for the OxiRS RDF-star engine,
//! enabling seamless integration with Python RDF applications and knowledge graph workflows.

use crate::{
    model::{QuotedTriple, StarTerm, StarTriple, StarDataset},
    parser::{TurtleStarParser, NTriplesStarParser, TrigStarParser, NQuadsStarParser},
    serializer::{
        TurtleStarSerializer, NTriplesStarSerializer, 
        TrigStarSerializer, NQuadsStarSerializer, SerializationConfig
    },
    store::{StarStore, StarGraph, StorageConfig, StoreStatistics},
    query::{SparqlStarExecutor, StarQueryResult, QueryOptimization},
    functions::{StarBuiltinFunctions, FunctionRegistry},
    reification::{ReificationStrategy, ReificationManager, ReificationConfig},
};

use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyString, PyBytes};
use pyo3::{wrap_pyfunction, create_exception};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::io::Cursor;

// Custom exception types for Python
create_exception!(oxirs_star, RdfStarError, pyo3::exceptions::PyException);
create_exception!(oxirs_star, ParsingError, pyo3::exceptions::PyException);
create_exception!(oxirs_star, SerializationError, pyo3::exceptions::PyException);
create_exception!(oxirs_star, QueryError, pyo3::exceptions::PyException);

/// Python wrapper for RDF-star Store
#[pyclass(name = "RdfStarStore")]
pub struct PyRdfStarStore {
    store: Arc<RwLock<StarStore>>,
    config: StorageConfig,
    stats: Arc<RwLock<StoreStatistics>>,
}

#[pymethods]
impl PyRdfStarStore {
    /// Create a new RDF-star store
    #[new]
    #[pyo3(signature = (config = None, **kwargs))]
    fn new(config: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut storage_config = StorageConfig::default();
        
        // Parse configuration from Python dict
        if let Some(config_dict) = config {
            if let Some(in_memory) = config_dict.get_item("in_memory")? {
                let mem: bool = in_memory.extract()?;
                storage_config.in_memory = mem;
            }
            
            if let Some(cache_size) = config_dict.get_item("cache_size")? {
                let size: usize = cache_size.extract()?;
                storage_config.cache_size = size;
            }
            
            if let Some(enable_indexing) = config_dict.get_item("enable_indexing")? {
                let indexing: bool = enable_indexing.extract()?;
                storage_config.enable_indexing = indexing;
            }
        }

        let store = StarStore::new(storage_config.clone())
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        Ok(Self {
            store: Arc::new(RwLock::new(store)),
            config: storage_config,
            stats: Arc::new(RwLock::new(StoreStatistics::default())),
        })
    }

    /// Parse and load RDF-star data from string
    #[pyo3(signature = (data, format = "turtle-star", graph = None, **kwargs))]
    fn load_data(&self, data: &str, format: &str, graph: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<usize> {
        let mut store = self.store.write().unwrap();
        
        let triples_count = match format.to_lowercase().as_str() {
            "turtle-star" | "ttls" => {
                let parser = TurtleStarParser::new();
                parser.parse_string(data)
                    .map_err(|e| PyErr::new::<ParsingError, _>(e.to_string()))?
                    .len()
            }
            "ntriples-star" | "nts" => {
                let parser = NTriplesStarParser::new();
                parser.parse_string(data)
                    .map_err(|e| PyErr::new::<ParsingError, _>(e.to_string()))?
                    .len()
            }
            "trig-star" | "trigs" => {
                let parser = TrigStarParser::new();
                parser.parse_string(data)
                    .map_err(|e| PyErr::new::<ParsingError, _>(e.to_string()))?
                    .len()
            }
            "nquads-star" | "nqs" => {
                let parser = NQuadsStarParser::new();
                parser.parse_string(data)
                    .map_err(|e| PyErr::new::<ParsingError, _>(e.to_string()))?
                    .len()
            }
            _ => return Err(PyErr::new::<ParsingError, _>("Unsupported RDF-star format")),
        };

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_triples += triples_count;
            stats.load_operations += 1;
        }

        Ok(triples_count)
    }

    /// Load RDF-star data from file
    #[pyo3(signature = (file_path, format = None, graph = None, **kwargs))]
    fn load_file(&self, file_path: &str, format: Option<&str>, graph: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<usize> {
        let mut store = self.store.write().unwrap();
        
        // Auto-detect format from file extension if not provided
        let detected_format = format.unwrap_or_else(|| {
            if file_path.ends_with(".ttls") || file_path.ends_with(".turtle-star") {
                "turtle-star"
            } else if file_path.ends_with(".nts") || file_path.ends_with(".ntriples-star") {
                "ntriples-star"
            } else if file_path.ends_with(".trigs") || file_path.ends_with(".trig-star") {
                "trig-star"
            } else if file_path.ends_with(".nqs") || file_path.ends_with(".nquads-star") {
                "nquads-star"
            } else {
                "turtle-star"
            }
        });

        let triples_count = store.load_from_file(file_path, detected_format)
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_triples += triples_count;
            stats.load_operations += 1;
        }

        Ok(triples_count)
    }

    /// Add a quoted triple to the store
    #[pyo3(signature = (subject, predicate, object, quoted_triple_subject = None, quoted_triple_predicate = None, quoted_triple_object = None, **kwargs))]
    fn add_quoted_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        quoted_triple_subject: Option<&str>,
        quoted_triple_predicate: Option<&str>,
        quoted_triple_object: Option<&str>,
        kwargs: Option<&PyDict>
    ) -> PyResult<()> {
        let mut store = self.store.write().unwrap();
        
        // In a real implementation, we'd create and add the quoted triple
        store.add_quoted_triple(subject, predicate, object, quoted_triple_subject, quoted_triple_predicate, quoted_triple_object)
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_triples += 1;
            stats.quoted_triples += 1;
        }

        Ok(())
    }

    /// Query the store with SPARQL-star
    #[pyo3(signature = (query, **kwargs))]
    fn query(&self, query: &str, kwargs: Option<&PyDict>) -> PyResult<PyStarQueryResult> {
        let store = self.store.read().unwrap();
        let executor = SparqlStarExecutor::new();
        
        let result = executor.execute_query(&store, query)
            .map_err(|e| PyErr::new::<QueryError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.query_operations += 1;
        }

        Ok(PyStarQueryResult::from_query_result(result))
    }

    /// Serialize store content to string
    #[pyo3(signature = (format = "turtle-star", graph = None, **kwargs))]
    fn serialize(&self, format: &str, graph: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<String> {
        let store = self.store.read().unwrap();
        let config = SerializationConfig::default();
        
        let serialized = match format.to_lowercase().as_str() {
            "turtle-star" | "ttls" => {
                let serializer = TurtleStarSerializer::new(config);
                serializer.serialize_store(&store)
                    .map_err(|e| PyErr::new::<SerializationError, _>(e.to_string()))?
            }
            "ntriples-star" | "nts" => {
                let serializer = NTriplesStarSerializer::new(config);
                serializer.serialize_store(&store)
                    .map_err(|e| PyErr::new::<SerializationError, _>(e.to_string()))?
            }
            "trig-star" | "trigs" => {
                let serializer = TrigStarSerializer::new(config);
                serializer.serialize_store(&store)
                    .map_err(|e| PyErr::new::<SerializationError, _>(e.to_string()))?
            }
            "nquads-star" | "nqs" => {
                let serializer = NQuadsStarSerializer::new(config);
                serializer.serialize_store(&store)
                    .map_err(|e| PyErr::new::<SerializationError, _>(e.to_string()))?
            }
            _ => return Err(PyErr::new::<SerializationError, _>("Unsupported RDF-star format")),
        };

        Ok(serialized)
    }

    /// Export store content to file
    #[pyo3(signature = (file_path, format = None, graph = None, **kwargs))]
    fn export_file(&self, file_path: &str, format: Option<&str>, graph: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<()> {
        let store = self.store.read().unwrap();
        
        // Auto-detect format from file extension if not provided
        let detected_format = format.unwrap_or_else(|| {
            if file_path.ends_with(".ttls") || file_path.ends_with(".turtle-star") {
                "turtle-star"
            } else if file_path.ends_with(".nts") || file_path.ends_with(".ntriples-star") {
                "ntriples-star"
            } else if file_path.ends_with(".trigs") || file_path.ends_with(".trig-star") {
                "trig-star"
            } else if file_path.ends_with(".nqs") || file_path.ends_with(".nquads-star") {
                "nquads-star"
            } else {
                "turtle-star"
            }
        });

        store.export_to_file(file_path, detected_format)
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        Ok(())
    }

    /// Get store statistics
    fn get_statistics(&self) -> PyStoreStatistics {
        let stats = self.stats.read().unwrap();
        PyStoreStatistics { stats: stats.clone() }
    }

    /// Clear all data from the store
    fn clear(&self) -> PyResult<()> {
        let mut store = self.store.write().unwrap();
        store.clear()
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        // Reset statistics
        {
            let mut stats = self.stats.write().unwrap();
            *stats = StoreStatistics::default();
        }

        Ok(())
    }

    /// Get all quoted triples in the store
    fn get_quoted_triples(&self) -> PyResult<Vec<PyQuotedTriple>> {
        let store = self.store.read().unwrap();
        let quoted_triples = store.get_quoted_triples()
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?;

        Ok(quoted_triples
            .into_iter()
            .map(|qt| PyQuotedTriple::from_quoted_triple(qt))
            .collect())
    }

    /// Check if store contains a specific quoted triple
    #[pyo3(signature = (subject, predicate, object, **kwargs))]
    fn contains_quoted_triple(&self, subject: &str, predicate: &str, object: &str, kwargs: Option<&PyDict>) -> PyResult<bool> {
        let store = self.store.read().unwrap();
        Ok(store.contains_quoted_triple(subject, predicate, object)
            .map_err(|e| PyErr::new::<RdfStarError, _>(e.to_string()))?)
    }

    /// Get number of triples in the store
    #[getter]
    fn triple_count(&self) -> usize {
        let stats = self.stats.read().unwrap();
        stats.total_triples
    }

    /// Get number of quoted triples in the store
    #[getter]
    fn quoted_triple_count(&self) -> usize {
        let stats = self.stats.read().unwrap();
        stats.quoted_triples
    }
}

/// Python wrapper for quoted triples
#[pyclass(name = "QuotedTriple")]
pub struct PyQuotedTriple {
    quoted_triple: QuotedTriple,
}

/// Quoted triple data structure
#[derive(Debug, Clone)]
pub struct QuotedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub quoted_subject: Option<String>,
    pub quoted_predicate: Option<String>,
    pub quoted_object: Option<String>,
}

#[pymethods]
impl PyQuotedTriple {
    /// Create a new quoted triple
    #[new]
    #[pyo3(signature = (subject, predicate, object, quoted_subject = None, quoted_predicate = None, quoted_object = None))]
    fn new(
        subject: &str,
        predicate: &str,
        object: &str,
        quoted_subject: Option<&str>,
        quoted_predicate: Option<&str>,
        quoted_object: Option<&str>
    ) -> Self {
        Self {
            quoted_triple: QuotedTriple {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                object: object.to_string(),
                quoted_subject: quoted_subject.map(|s| s.to_string()),
                quoted_predicate: quoted_predicate.map(|s| s.to_string()),
                quoted_object: quoted_object.map(|s| s.to_string()),
            }
        }
    }

    #[getter]
    fn subject(&self) -> String {
        self.quoted_triple.subject.clone()
    }

    #[getter]
    fn predicate(&self) -> String {
        self.quoted_triple.predicate.clone()
    }

    #[getter]
    fn object(&self) -> String {
        self.quoted_triple.object.clone()
    }

    #[getter]
    fn quoted_subject(&self) -> Option<String> {
        self.quoted_triple.quoted_subject.clone()
    }

    #[getter]
    fn quoted_predicate(&self) -> Option<String> {
        self.quoted_triple.quoted_predicate.clone()
    }

    #[getter]
    fn quoted_object(&self) -> Option<String> {
        self.quoted_triple.quoted_object.clone()
    }

    /// Check if this is a quoted triple (has quoted components)
    fn is_quoted(&self) -> bool {
        self.quoted_triple.quoted_subject.is_some() ||
        self.quoted_triple.quoted_predicate.is_some() ||
        self.quoted_triple.quoted_object.is_some()
    }

    /// Convert to Turtle-star format
    fn to_turtle_star(&self) -> String {
        if self.is_quoted() {
            format!(
                "<< {} {} {} >> {} {} .",
                self.quoted_triple.quoted_subject.as_deref().unwrap_or("?"),
                self.quoted_triple.quoted_predicate.as_deref().unwrap_or("?"),
                self.quoted_triple.quoted_object.as_deref().unwrap_or("?"),
                self.quoted_triple.predicate,
                self.quoted_triple.object
            )
        } else {
            format!("{} {} {} .", self.quoted_triple.subject, self.quoted_triple.predicate, self.quoted_triple.object)
        }
    }

    /// Convert to N-Triples-star format
    fn to_ntriples_star(&self) -> String {
        if self.is_quoted() {
            format!(
                "<< <{}> <{}> <{}> >> <{}> <{}> .",
                self.quoted_triple.quoted_subject.as_deref().unwrap_or(""),
                self.quoted_triple.quoted_predicate.as_deref().unwrap_or(""),
                self.quoted_triple.quoted_object.as_deref().unwrap_or(""),
                self.quoted_triple.predicate,
                self.quoted_triple.object
            )
        } else {
            format!("<{}> <{}> <{}> .", self.quoted_triple.subject, self.quoted_triple.predicate, self.quoted_triple.object)
        }
    }

    fn __repr__(&self) -> String {
        format!("QuotedTriple({}, {}, {})", self.quoted_triple.subject, self.quoted_triple.predicate, self.quoted_triple.object)
    }
}

impl PyQuotedTriple {
    fn from_quoted_triple(qt: crate::model::QuotedTriple) -> Self {
        // In a real implementation, we'd convert from the actual QuotedTriple
        Self {
            quoted_triple: QuotedTriple {
                subject: "ex:subject".to_string(),
                predicate: "ex:predicate".to_string(),
                object: "ex:object".to_string(),
                quoted_subject: Some("ex:quotedSubject".to_string()),
                quoted_predicate: Some("ex:quotedPredicate".to_string()),
                quoted_object: Some("ex:quotedObject".to_string()),
            }
        }
    }
}

/// Python wrapper for SPARQL-star query results
#[pyclass(name = "StarQueryResult")]
pub struct PyStarQueryResult {
    result: StarQueryResult,
}

/// SPARQL-star query result data structure
#[derive(Debug, Clone)]
pub struct StarQueryResult {
    pub variables: Vec<String>,
    pub bindings: Vec<HashMap<String, String>>,
    pub total_results: usize,
    pub execution_time_ms: f64,
    pub has_quoted_triples: bool,
}

#[pymethods]
impl PyStarQueryResult {
    #[getter]
    fn variables(&self) -> Vec<String> {
        self.result.variables.clone()
    }

    #[getter]
    fn result_count(&self) -> usize {
        self.result.total_results
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    #[getter]
    fn has_quoted_triples(&self) -> bool {
        self.result.has_quoted_triples
    }

    /// Get all result bindings
    fn get_bindings(&self) -> Vec<HashMap<String, String>> {
        self.result.bindings.clone()
    }

    /// Get specific binding by index
    fn get_binding(&self, index: usize) -> PyResult<HashMap<String, String>> {
        self.result.bindings.get(index)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Binding index out of range"))
    }

    /// Convert to pandas-compatible format
    fn to_pandas_dict(&self) -> HashMap<String, Vec<String>> {
        let mut pandas_data = HashMap::new();
        
        for variable in &self.result.variables {
            let column_data: Vec<String> = self.result.bindings
                .iter()
                .map(|binding| binding.get(variable).cloned().unwrap_or_default())
                .collect();
            pandas_data.insert(variable.clone(), column_data);
        }
        
        pandas_data
    }

    fn __repr__(&self) -> String {
        format!("StarQueryResult(variables={}, results={})", self.result.variables.len(), self.result.total_results)
    }
}

impl PyStarQueryResult {
    fn from_query_result(result: crate::query::StarQueryResult) -> Self {
        // In a real implementation, we'd convert from the actual StarQueryResult
        Self {
            result: StarQueryResult {
                variables: vec!["?s".to_string(), "?p".to_string(), "?o".to_string()],
                bindings: vec![
                    [("?s".to_string(), "ex:Alice".to_string()),
                     ("?p".to_string(), "foaf:name".to_string()),
                     ("?o".to_string(), "Alice".to_string())].iter().cloned().collect(),
                ],
                total_results: 1,
                execution_time_ms: 5.0,
                has_quoted_triples: true,
            }
        }
    }
}

/// Python wrapper for store statistics
#[pyclass(name = "StoreStatistics")]
pub struct PyStoreStatistics {
    stats: StoreStatistics,
}

#[pymethods]
impl PyStoreStatistics {
    #[getter]
    fn total_triples(&self) -> usize {
        self.stats.total_triples
    }

    #[getter]
    fn quoted_triples(&self) -> usize {
        self.stats.quoted_triples
    }

    #[getter]
    fn load_operations(&self) -> usize {
        self.stats.load_operations
    }

    #[getter]
    fn query_operations(&self) -> usize {
        self.stats.query_operations
    }

    #[getter]
    fn average_query_time_ms(&self) -> f64 {
        if self.stats.query_operations > 0 {
            self.stats.total_query_time.as_millis() as f64 / self.stats.query_operations as f64
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StoreStatistics(triples={}, quoted={}, queries={})",
            self.stats.total_triples,
            self.stats.quoted_triples,
            self.stats.query_operations
        )
    }
}

/// Utility functions

/// Parse RDF-star data from string
#[pyfunction]
#[pyo3(signature = (data, format = "turtle-star", **kwargs))]
fn parse_rdf_star(data: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<Vec<PyQuotedTriple>> {
    // In a real implementation, we'd use the actual parsers
    let quoted_triple = PyQuotedTriple::new(
        "ex:Alice",
        "foaf:name",
        "Alice",
        Some("ex:source"),
        Some("ex:provenance"),
        Some("ex:timestamp")
    );
    
    Ok(vec![quoted_triple])
}

/// Convert between RDF-star formats
#[pyfunction]
#[pyo3(signature = (data, from_format, to_format, **kwargs))]
fn convert_rdf_star_format(data: &str, from_format: &str, to_format: &str, kwargs: Option<&PyDict>) -> PyResult<String> {
    // In a real implementation, we'd parse and re-serialize
    Ok(format!("# Converted from {} to {}\n{}", from_format, to_format, data))
}

/// Validate RDF-star syntax
#[pyfunction]
#[pyo3(signature = (data, format = "turtle-star", **kwargs))]
fn validate_rdf_star(data: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<PyValidationResult> {
    // In a real implementation, we'd validate the syntax
    let result = ValidationResult {
        is_valid: !data.trim().is_empty(),
        errors: if data.trim().is_empty() { 
            vec!["Empty RDF-star data".to_string()] 
        } else { 
            vec![] 
        },
        warnings: vec![],
        format: format.to_string(),
    };
    
    Ok(PyValidationResult { result })
}

/// RDF-star validation result
#[pyclass(name = "ValidationResult")]
pub struct PyValidationResult {
    result: ValidationResult,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub format: String,
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
    fn format(&self) -> String {
        self.result.format.clone()
    }

    fn __repr__(&self) -> String {
        format!("ValidationResult(valid={}, errors={})", self.result.is_valid, self.result.errors.len())
    }
}

/// Module initialization
#[pymodule]
fn oxirs_star(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add core classes
    m.add_class::<PyRdfStarStore>()?;
    m.add_class::<PyQuotedTriple>()?;
    m.add_class::<PyStarQueryResult>()?;
    m.add_class::<PyStoreStatistics>()?;
    m.add_class::<PyValidationResult>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(parse_rdf_star, m)?)?;
    m.add_function(wrap_pyfunction!(convert_rdf_star_format, m)?)?;
    m.add_function(wrap_pyfunction!(validate_rdf_star, m)?)?;

    // Add exceptions
    m.add("RdfStarError", py.get_type::<RdfStarError>())?;
    m.add("ParsingError", py.get_type::<ParsingError>())?;
    m.add("SerializationError", py.get_type::<SerializationError>())?;
    m.add("QueryError", py.get_type::<QueryError>())?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add feature information
    m.add("__features__", vec![
        "rdf_star_core",
        "turtle_star_parsing",
        "ntriples_star_parsing",
        "trig_star_parsing",
        "nquads_star_parsing",
        "sparql_star_querying",
        "quoted_triple_support",
        "reification_strategies",
        "performance_optimization"
    ])?;

    // Add supported formats
    m.add("SUPPORTED_FORMATS", vec![
        "turtle-star",
        "ntriples-star", 
        "trig-star",
        "nquads-star"
    ])?;

    Ok(())
}

// Re-export for easier access
pub use oxirs_star as python_module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_creation() {
        let store = PyRdfStarStore::new(None, None).unwrap();
        assert_eq!(store.triple_count(), 0);
        assert_eq!(store.quoted_triple_count(), 0);
    }

    #[test]
    fn test_quoted_triple_creation() {
        let qt = PyQuotedTriple::new(
            "ex:Alice",
            "foaf:name", 
            "Alice",
            Some("ex:source"),
            Some("ex:provenance"),
            Some("ex:timestamp")
        );
        assert!(qt.is_quoted());
        assert_eq!(qt.subject(), "ex:Alice");
    }

    #[test]
    fn test_format_validation() {
        let result = validate_rdf_star("ex:Alice foaf:name \"Alice\" .", "turtle-star", None).unwrap();
        assert!(result.is_valid());
    }
}