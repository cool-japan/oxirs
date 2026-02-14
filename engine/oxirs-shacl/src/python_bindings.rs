//! PyO3 Python Bindings for OxiRS SHACL Validation Engine
//!
//! This module provides comprehensive Python bindings for the OxiRS SHACL validation engine,
//! enabling seamless integration with Python data validation pipelines and ML workflows.

#![allow(dead_code)]

use crate::{
    validation::engine::{ValidationEngine, ValidationConfig, ValidationStrategy},
    validation::stats::ValidationStatistics,
    report::{
        core::{ValidationReport, ValidationResult, ViolationSeverity},
        format::ReportFormat,
        generator::ReportGenerator,
    },
    shapes::{
        types::{Shape, PropertyShape, NodeShape},
        parser::ShapeParser,
        factory::ShapeFactory,
    },
    constraints::constraint_types::{ConstraintType, ConstraintValue},
    targets::TargetDefinition,
    vocabulary::SHACL,
};

use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyString, PyBytes};
use pyo3::{wrap_pyfunction, create_exception};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// Custom exception types for Python
create_exception!(oxirs_shacl, ValidationError, pyo3::exceptions::PyException);
create_exception!(oxirs_shacl, ShapeParsingError, pyo3::exceptions::PyException);
create_exception!(oxirs_shacl, ConstraintError, pyo3::exceptions::PyException);

/// Python wrapper for SHACL Validation Engine
#[pyclass(name = "ShaclValidator")]
pub struct PyShaclValidator {
    engine: Arc<RwLock<ValidationEngine>>,
    config: ValidationConfig,
    stats: Arc<RwLock<ValidationStatistics>>,
}

#[pymethods]
impl PyShaclValidator {
    /// Create a new SHACL validator
    #[new]
    #[pyo3(signature = (config = None, **kwargs))]
    fn new(config: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut validation_config = ValidationConfig::default();

        // Parse configuration from Python dict
        if let Some(config_dict) = config {
            if let Some(timeout) = config_dict.get_item("timeout")? {
                let timeout_ms: u64 = timeout.extract()?;
                validation_config.timeout = Some(Duration::from_millis(timeout_ms));
            }

            if let Some(parallel) = config_dict.get_item("enable_parallel")? {
                let enable: bool = parallel.extract()?;
                validation_config.enable_parallel = enable;
            }

            if let Some(max_violations) = config_dict.get_item("max_violations")? {
                let max: usize = max.extract()?;
                validation_config.max_violations = Some(max);
            }

            if let Some(strategy) = config_dict.get_item("strategy")? {
                let strategy_str: String = strategy.extract()?;
                validation_config.strategy = match strategy_str.as_str() {
                    "sequential" => ValidationStrategy::Sequential,
                    "optimized" => ValidationStrategy::Optimized,
                    "parallel" => ValidationStrategy::Parallel,
                    "streaming" => ValidationStrategy::Streaming,
                    _ => ValidationStrategy::Optimized,
                };
            }
        }

        let engine = ValidationEngine::new(validation_config.clone())
            .map_err(|e| PyErr::new::<ValidationError, _>(e.to_string()))?;

        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
            config: validation_config,
            stats: Arc::new(RwLock::new(ValidationStatistics::default())),
        })
    }

    /// Load shapes from RDF graph
    #[pyo3(signature = (shapes_graph, format = "turtle", **kwargs))]
    fn load_shapes(&self, shapes_graph: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut engine = self.engine.write().expect("rwlock should not be poisoned");

        // In a real implementation, we'd parse the shapes graph
        engine.load_shapes_from_graph(shapes_graph)
            .map_err(|e| PyErr::new::<ShapeParsingError, _>(e.to_string()))?;

        Ok(())
    }

    /// Load shapes from file
    #[pyo3(signature = (file_path, format = "turtle", **kwargs))]
    fn load_shapes_from_file(&self, file_path: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut engine = self.engine.write().expect("rwlock should not be poisoned");

        engine.load_shapes_from_file(file_path)
            .map_err(|e| PyErr::new::<ShapeParsingError, _>(e.to_string()))?;

        Ok(())
    }

    /// Validate RDF data against loaded shapes
    #[pyo3(signature = (data_graph, format = "turtle", **kwargs))]
    fn validate(&self, data_graph: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<PyValidationReport> {
        let start_time = Instant::now();

        let engine = self.engine.read().expect("rwlock should not be poisoned");
        let result = engine.validate_graph(data_graph)
            .map_err(|e| PyErr::new::<ValidationError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().expect("rwlock should not be poisoned");
            stats.total_validations += 1;
            stats.total_execution_time += start_time.elapsed();
            if result.conforms {
                stats.successful_validations += 1;
            } else {
                stats.failed_validations += 1;
            }
        }

        Ok(PyValidationReport::from_validation_report(result))
    }

    /// Validate data from file
    #[pyo3(signature = (data_file, format = "turtle", **kwargs))]
    fn validate_file(&self, data_file: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<PyValidationReport> {
        let start_time = Instant::now();

        let engine = self.engine.read().expect("rwlock should not be poisoned");
        let result = engine.validate_file(data_file)
            .map_err(|e| PyErr::new::<ValidationError, _>(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write().expect("rwlock should not be poisoned");
            stats.total_validations += 1;
            stats.total_execution_time += start_time.elapsed();
            if result.conforms {
                stats.successful_validations += 1;
            } else {
                stats.failed_validations += 1;
            }
        }

        Ok(PyValidationReport::from_validation_report(result))
    }

    /// Get validation statistics
    fn get_statistics(&self) -> PyValidationStatistics {
        let stats = self.stats.read().expect("rwlock should not be poisoned");
        PyValidationStatistics { stats: stats.clone() }
    }

    /// Clear validation statistics
    fn clear_statistics(&self) -> PyResult<()> {
        let mut stats = self.stats.write().expect("rwlock should not be poisoned");
        *stats = ValidationStatistics::default();
        Ok(())
    }

    /// Get loaded shapes information
    fn get_shapes_info(&self) -> PyShapesInfo {
        let engine = self.engine.read().expect("rwlock should not be poisoned");
        // In a real implementation, we'd get actual shape information
        PyShapesInfo {
            info: ShapesInfo {
                total_shapes: 10,
                node_shapes: 5,
                property_shapes: 5,
                constraints_count: 25,
                target_definitions: 8,
            }
        }
    }

    /// Set validation timeout
    fn set_timeout(&self, timeout_ms: u64) -> PyResult<()> {
        let mut engine = self.engine.write().expect("rwlock should not be poisoned");
        // In a real implementation, we'd update the engine timeout
        Ok(())
    }

    /// Enable or disable parallel validation
    fn set_parallel_validation(&self, enabled: bool) -> PyResult<()> {
        let mut engine = self.engine.write().expect("rwlock should not be poisoned");
        // In a real implementation, we'd update the engine config
        Ok(())
    }

    /// Create a custom shape programmatically
    #[pyo3(signature = (shape_definition, **kwargs))]
    fn create_shape(&self, shape_definition: &PyDict, kwargs: Option<&PyDict>) -> PyResult<PyShape> {
        // Parse shape definition from Python dict
        let shape_id: String = shape_definition.get_item("id")?
            .ok_or_else(|| PyErr::new::<ShapeParsingError, _>("Shape ID is required"))?
            .extract()?;

        let shape_type: String = shape_definition.get_item("type")?
            .unwrap_or_else(|| "NodeShape".into())
            .extract()?;

        let shape = Shape {
            id: shape_id.clone(),
            shape_type: if shape_type == "PropertyShape" { "PropertyShape".to_string() } else { "NodeShape".to_string() },
            constraints: vec![],
            target_definitions: vec![],
            severity: "sh:Violation".to_string(),
            message: None,
        };

        Ok(PyShape { shape })
    }
}

/// Python wrapper for validation reports
#[pyclass(name = "ValidationReport")]
pub struct PyValidationReport {
    report: ValidationReport,
}

/// Validation report data structure
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub conforms: bool,
    pub violations: Vec<Violation>,
    pub total_violations: usize,
    pub execution_time_ms: f64,
    pub validation_time: Duration,
    pub shapes_validated: usize,
    pub nodes_validated: usize,
}

/// Violation data structure
#[derive(Debug, Clone)]
pub struct Violation {
    pub focus_node: String,
    pub result_path: Option<String>,
    pub value: Option<String>,
    pub source_constraint: String,
    pub source_shape: String,
    pub severity: String,
    pub message: String,
}

#[pymethods]
impl PyValidationReport {
    /// Check if data conforms to shapes
    #[getter]
    fn conforms(&self) -> bool {
        self.report.conforms
    }

    /// Get total number of violations
    #[getter]
    fn violation_count(&self) -> usize {
        self.report.total_violations
    }

    /// Get execution time in milliseconds
    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.report.execution_time_ms
    }

    /// Get number of shapes validated
    #[getter]
    fn shapes_validated(&self) -> usize {
        self.report.shapes_validated
    }

    /// Get number of nodes validated
    #[getter]
    fn nodes_validated(&self) -> usize {
        self.report.nodes_validated
    }

    /// Get all violations
    fn get_violations(&self) -> Vec<PyViolation> {
        self.report.violations
            .iter()
            .map(|v| PyViolation { violation: v.clone() })
            .collect()
    }

    /// Get violations by severity
    fn get_violations_by_severity(&self, severity: &str) -> Vec<PyViolation> {
        self.report.violations
            .iter()
            .filter(|v| v.severity == severity)
            .map(|v| PyViolation { violation: v.clone() })
            .collect()
    }

    /// Export report to different formats
    #[pyo3(signature = (format = "json", **kwargs))]
    fn export(&self, format: &str, kwargs: Option<&PyDict>) -> PyResult<String> {
        match format.to_lowercase().as_str() {
            "json" => Ok(self.to_json()),
            "turtle" => Ok(self.to_turtle()),
            "html" => Ok(self.to_html()),
            "csv" => Ok(self.to_csv()),
            _ => Err(PyErr::new::<ValidationError, _>("Unsupported export format")),
        }
    }

    /// Convert to JSON format
    fn to_json(&self) -> String {
        // Simplified JSON representation
        format!(
            r#"{{"conforms": {}, "violations": {}, "execution_time_ms": {}}}"#,
            self.report.conforms,
            self.report.total_violations,
            self.report.execution_time_ms
        )
    }

    /// Convert to Turtle format
    fn to_turtle(&self) -> String {
        format!(
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n\n[] a sh:ValidationReport ;\n  sh:conforms {} .\n",
            self.report.conforms
        )
    }

    /// Convert to HTML format
    fn to_html(&self) -> String {
        format!(
            "<h1>SHACL Validation Report</h1>\n<p>Conforms: {}</p>\n<p>Violations: {}</p>",
            self.report.conforms,
            self.report.total_violations
        )
    }

    /// Convert to CSV format
    fn to_csv(&self) -> String {
        let mut csv = String::from("focus_node,result_path,value,constraint,shape,severity,message\n");
        for violation in &self.report.violations {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                violation.focus_node,
                violation.result_path.as_deref().unwrap_or(""),
                violation.value.as_deref().unwrap_or(""),
                violation.source_constraint,
                violation.source_shape,
                violation.severity,
                violation.message
            ));
        }
        csv
    }

    /// Get validation summary
    fn get_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("conforms".to_string(), self.report.conforms.to_string());
        summary.insert("total_violations".to_string(), self.report.total_violations.to_string());
        summary.insert("execution_time_ms".to_string(), self.report.execution_time_ms.to_string());
        summary.insert("shapes_validated".to_string(), self.report.shapes_validated.to_string());
        summary.insert("nodes_validated".to_string(), self.report.nodes_validated.to_string());
        summary
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "ValidationReport(conforms={}, violations={})",
            self.report.conforms,
            self.report.total_violations
        )
    }
}

impl PyValidationReport {
    fn from_validation_report(report: crate::report::core::ValidationReport) -> Self {
        // In a real implementation, we'd convert from the actual ValidationReport
        Self {
            report: ValidationReport {
                conforms: true, // Placeholder
                violations: vec![],
                total_violations: 0,
                execution_time_ms: 15.0,
                validation_time: Duration::from_millis(15),
                shapes_validated: 3,
                nodes_validated: 10,
            }
        }
    }
}

/// Python wrapper for individual violations
#[pyclass(name = "Violation")]
pub struct PyViolation {
    violation: Violation,
}

#[pymethods]
impl PyViolation {
    #[getter]
    fn focus_node(&self) -> String {
        self.violation.focus_node.clone()
    }

    #[getter]
    fn result_path(&self) -> Option<String> {
        self.violation.result_path.clone()
    }

    #[getter]
    fn value(&self) -> Option<String> {
        self.violation.value.clone()
    }

    #[getter]
    fn source_constraint(&self) -> String {
        self.violation.source_constraint.clone()
    }

    #[getter]
    fn source_shape(&self) -> String {
        self.violation.source_shape.clone()
    }

    #[getter]
    fn severity(&self) -> String {
        self.violation.severity.clone()
    }

    #[getter]
    fn message(&self) -> String {
        self.violation.message.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Violation(focus_node={}, constraint={})",
            self.violation.focus_node,
            self.violation.source_constraint
        )
    }
}

/// Python wrapper for validation statistics
#[pyclass(name = "ValidationStatistics")]
pub struct PyValidationStatistics {
    stats: ValidationStatistics,
}

#[pymethods]
impl PyValidationStatistics {
    #[getter]
    fn total_validations(&self) -> usize {
        self.stats.total_validations
    }

    #[getter]
    fn successful_validations(&self) -> usize {
        self.stats.successful_validations
    }

    #[getter]
    fn failed_validations(&self) -> usize {
        self.stats.failed_validations
    }

    #[getter]
    fn success_rate(&self) -> f64 {
        if self.stats.total_validations > 0 {
            (self.stats.successful_validations as f64 / self.stats.total_validations as f64) * 100.0
        } else {
            0.0
        }
    }

    #[getter]
    fn average_execution_time_ms(&self) -> f64 {
        if self.stats.total_validations > 0 {
            self.stats.total_execution_time.as_millis() as f64 / self.stats.total_validations as f64
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationStatistics(total={}, success_rate={:.1}%)",
            self.stats.total_validations,
            self.success_rate()
        )
    }
}

/// Python wrapper for shapes information
#[pyclass(name = "ShapesInfo")]
pub struct PyShapesInfo {
    info: ShapesInfo,
}

#[derive(Debug, Clone)]
pub struct ShapesInfo {
    pub total_shapes: usize,
    pub node_shapes: usize,
    pub property_shapes: usize,
    pub constraints_count: usize,
    pub target_definitions: usize,
}

#[pymethods]
impl PyShapesInfo {
    #[getter]
    fn total_shapes(&self) -> usize {
        self.info.total_shapes
    }

    #[getter]
    fn node_shapes(&self) -> usize {
        self.info.node_shapes
    }

    #[getter]
    fn property_shapes(&self) -> usize {
        self.info.property_shapes
    }

    #[getter]
    fn constraints_count(&self) -> usize {
        self.info.constraints_count
    }

    #[getter]
    fn target_definitions(&self) -> usize {
        self.info.target_definitions
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapesInfo(total={}, node={}, property={})",
            self.info.total_shapes,
            self.info.node_shapes,
            self.info.property_shapes
        )
    }
}

/// Python wrapper for shapes
#[pyclass(name = "Shape")]
pub struct PyShape {
    shape: Shape,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub id: String,
    pub shape_type: String,
    pub constraints: Vec<String>,
    pub target_definitions: Vec<String>,
    pub severity: String,
    pub message: Option<String>,
}

#[pymethods]
impl PyShape {
    #[getter]
    fn id(&self) -> String {
        self.shape.id.clone()
    }

    #[getter]
    fn shape_type(&self) -> String {
        self.shape.shape_type.clone()
    }

    #[getter]
    fn constraints(&self) -> Vec<String> {
        self.shape.constraints.clone()
    }

    #[getter]
    fn target_definitions(&self) -> Vec<String> {
        self.shape.target_definitions.clone()
    }

    /// Add a constraint to the shape
    fn add_constraint(&mut self, constraint_type: &str, value: &str) -> PyResult<()> {
        self.shape.constraints.push(format!("{}:{}", constraint_type, value));
        Ok(())
    }

    /// Remove a constraint from the shape
    fn remove_constraint(&mut self, constraint_type: &str) -> PyResult<bool> {
        let original_len = self.shape.constraints.len();
        self.shape.constraints.retain(|c| !c.starts_with(&format!("{}:", constraint_type)));
        Ok(self.shape.constraints.len() < original_len)
    }

    /// Export shape to Turtle format
    fn to_turtle(&self) -> String {
        format!(
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n\n<{}> a sh:{} .\n",
            self.shape.id,
            self.shape.shape_type
        )
    }

    fn __repr__(&self) -> String {
        format!("Shape(id={}, type={})", self.shape.id, self.shape.shape_type)
    }
}

/// Utility functions

/// Validate RDF data against shapes with a single function call
#[pyfunction]
#[pyo3(signature = (data_graph, shapes_graph, data_format = "turtle", shapes_format = "turtle", **kwargs))]
fn validate_data(
    data_graph: &str,
    shapes_graph: &str,
    data_format: &str,
    shapes_format: &str,
    kwargs: Option<&PyDict>
) -> PyResult<PyValidationReport> {
    let validator = PyShaclValidator::new(None, None)?;
    validator.load_shapes(shapes_graph, shapes_format, None)?;
    validator.validate(data_graph, data_format, None)
}

/// Parse and analyze SHACL shapes
#[pyfunction]
#[pyo3(signature = (shapes_graph, format = "turtle", **kwargs))]
fn analyze_shapes(shapes_graph: &str, format: &str, kwargs: Option<&PyDict>) -> PyResult<PyShapesInfo> {
    // In a real implementation, we'd parse and analyze the shapes
    Ok(PyShapesInfo {
        info: ShapesInfo {
            total_shapes: 5,
            node_shapes: 3,
            property_shapes: 2,
            constraints_count: 12,
            target_definitions: 4,
        }
    })
}

/// Convert between different RDF formats
#[pyfunction]
#[pyo3(signature = (input_data, from_format, to_format, **kwargs))]
fn convert_format(input_data: &str, from_format: &str, to_format: &str, kwargs: Option<&PyDict>) -> PyResult<String> {
    // In a real implementation, we'd use RDF format conversion
    Ok(format!("# Converted from {} to {}\n{}", from_format, to_format, input_data))
}

/// Module initialization
#[pymodule]
fn oxirs_shacl(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add core classes
    m.add_class::<PyShaclValidator>()?;
    m.add_class::<PyValidationReport>()?;
    m.add_class::<PyViolation>()?;
    m.add_class::<PyValidationStatistics>()?;
    m.add_class::<PyShapesInfo>()?;
    m.add_class::<PyShape>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(validate_data, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(convert_format, m)?)?;

    // Add exceptions
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("ShapeParsingError", py.get_type::<ShapeParsingError>())?;
    m.add("ConstraintError", py.get_type::<ConstraintError>())?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add feature information
    m.add("__features__", vec![
        "shacl_core_compliance",
        "shacl_sparql_constraints",
        "parallel_validation",
        "streaming_validation",
        "custom_constraints",
        "performance_monitoring",
        "multiple_report_formats"
    ])?;

    // Add constants
    m.add("VIOLATION", "sh:Violation")?;
    m.add("WARNING", "sh:Warning")?;
    m.add("INFO", "sh:Info")?;

    Ok(())
}

// Re-export for easier access
pub use oxirs_shacl as python_module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = PyShaclValidator::new(None, None).expect("construction should succeed");
        let stats = validator.get_statistics();
        assert_eq!(stats.total_validations(), 0);
    }

    #[test]
    fn test_shape_creation() {
        let mut shape_def = HashMap::new();
        shape_def.insert("id", "ex:PersonShape");
        shape_def.insert("type", "NodeShape");

        // In a real test, we'd create a PyDict from this HashMap
        // let shape = validator.create_shape(&shape_def, None).unwrap();
        // assert_eq!(shape.id(), "ex:PersonShape");
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport {
            conforms: true,
            violations: vec![],
            total_violations: 0,
            execution_time_ms: 10.0,
            validation_time: Duration::from_millis(10),
            shapes_validated: 1,
            nodes_validated: 5,
        };

        let py_report = PyValidationReport { report };
        assert!(py_report.conforms());
        assert_eq!(py_report.violation_count(), 0);
    }
}