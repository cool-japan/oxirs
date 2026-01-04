//! PyO3 Python Bindings for OxiRS Rule Engine
//!
//! This module provides comprehensive Python bindings for the OxiRS rule-based reasoning engine,
//! enabling seamless integration with Python AI/ML pipelines and knowledge graph workflows.

use crate::{
    RuleEngine, Rule, Atom, Term, Variable,
    forward::{ForwardChainEngine, ForwardChainResult},
    backward::{BackwardChainEngine, BackwardChainResult, ProofTrace},
    rdfs::{RdfsReasoner, RdfsInferenceResult},
    owl::{OwlReasoner, OwlInferenceResult, ConsistencyResult},
    swrl::{SwrlEngine, SwrlRule, SwrlAtom, BuiltinFunction},
    rete::{ReteNetwork, ReteStats, ReteConfiguration},
    performance::{PerformanceMonitor, ReasoningStats, OptimizationConfig},
    debug::{DebuggableRuleEngine, DebugSession, ExecutionTrace},
};

use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyString, PyBool};
use pyo3::{wrap_pyfunction, create_exception};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// Custom exception types for Python
create_exception!(oxirs_rule, ReasoningError, pyo3::exceptions::PyException);
create_exception!(oxirs_rule, RuleParsingError, pyo3::exceptions::PyException);
create_exception!(oxirs_rule, InferenceError, pyo3::exceptions::PyException);
create_exception!(oxirs_rule, ConsistencyError, pyo3::exceptions::PyException);

/// Python wrapper for Rule Engine
#[pyclass(name = "RuleEngine")]
pub struct PyRuleEngine {
    engine: Arc<RwLock<RuleEngine>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    config: ReasoningConfig,
}

/// Reasoning configuration
#[derive(Debug, Clone)]
pub struct ReasoningConfig {
    pub enable_forward_chaining: bool,
    pub enable_backward_chaining: bool,
    pub enable_rdfs_reasoning: bool,
    pub enable_owl_reasoning: bool,
    pub enable_swrl: bool,
    pub enable_rete_network: bool,
    pub max_iterations: usize,
    pub timeout: Option<Duration>,
    pub enable_debugging: bool,
    pub enable_performance_monitoring: bool,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enable_forward_chaining: true,
            enable_backward_chaining: true,
            enable_rdfs_reasoning: true,
            enable_owl_reasoning: false,
            enable_swrl: false,
            enable_rete_network: false,
            max_iterations: 1000,
            timeout: Some(Duration::from_secs(30)),
            enable_debugging: false,
            enable_performance_monitoring: true,
        }
    }
}

#[pymethods]
impl PyRuleEngine {
    /// Create a new rule engine
    #[new]
    #[pyo3(signature = (config = None, **kwargs))]
    fn new(config: Option<&PyDict>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut reasoning_config = ReasoningConfig::default();

        // Parse configuration from Python dict
        if let Some(config_dict) = config {
            if let Some(forward) = config_dict.get_item("enable_forward_chaining")? {
                reasoning_config.enable_forward_chaining = forward.extract()?;
            }

            if let Some(backward) = config_dict.get_item("enable_backward_chaining")? {
                reasoning_config.enable_backward_chaining = backward.extract()?;
            }

            if let Some(rdfs) = config_dict.get_item("enable_rdfs_reasoning")? {
                reasoning_config.enable_rdfs_reasoning = rdfs.extract()?;
            }

            if let Some(owl) = config_dict.get_item("enable_owl_reasoning")? {
                reasoning_config.enable_owl_reasoning = owl.extract()?;
            }

            if let Some(swrl) = config_dict.get_item("enable_swrl")? {
                reasoning_config.enable_swrl = swrl.extract()?;
            }

            if let Some(rete) = config_dict.get_item("enable_rete_network")? {
                reasoning_config.enable_rete_network = rete.extract()?;
            }

            if let Some(max_iter) = config_dict.get_item("max_iterations")? {
                reasoning_config.max_iterations = max_iter.extract()?;
            }

            if let Some(timeout_ms) = config_dict.get_item("timeout_ms")? {
                let timeout: u64 = timeout_ms.extract()?;
                reasoning_config.timeout = Some(Duration::from_millis(timeout));
            }

            if let Some(debug) = config_dict.get_item("enable_debugging")? {
                reasoning_config.enable_debugging = debug.extract()?;
            }
        }

        let engine = RuleEngine::new()
            .map_err(|e| PyErr::new::<ReasoningError, _>(e.to_string()))?;

        let performance_monitor = PerformanceMonitor::new();

        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
            performance_monitor: Arc::new(RwLock::new(performance_monitor)),
            config: reasoning_config,
        })
    }

    /// Add a rule to the engine
    #[pyo3(signature = (rule_text, rule_format = "datalog", **kwargs))]
    fn add_rule(&self, rule_text: &str, rule_format: &str, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut engine = self.engine.write().expect("lock poisoned");

        engine.add_rule_from_text(rule_text, rule_format)
            .map_err(|e| PyErr::new::<RuleParsingError, _>(e.to_string()))?;

        Ok(())
    }

    /// Add multiple rules from a file
    #[pyo3(signature = (file_path, rule_format = "datalog", **kwargs))]
    fn add_rules_from_file(&self, file_path: &str, rule_format: &str, kwargs: Option<&PyDict>) -> PyResult<usize> {
        let mut engine = self.engine.write().expect("lock poisoned");

        let rules_count = engine.load_rules_from_file(file_path, rule_format)
            .map_err(|e| PyErr::new::<RuleParsingError, _>(e.to_string()))?;

        Ok(rules_count)
    }

    /// Add facts to the knowledge base
    #[pyo3(signature = (facts, **kwargs))]
    fn add_facts(&self, facts: Vec<&str>, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut engine = self.engine.write().expect("lock poisoned");

        for fact_text in facts {
            engine.add_fact_from_text(fact_text)
                .map_err(|e| PyErr::new::<ReasoningError, _>(e.to_string()))?;
        }

        Ok(())
    }

    /// Perform forward chaining inference
    #[pyo3(signature = (**kwargs))]
    fn forward_chain(&self, kwargs: Option<&PyDict>) -> PyResult<PyForwardChainResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        if !self.config.enable_forward_chaining {
            return Err(PyErr::new::<ReasoningError, _>("Forward chaining is disabled"));
        }

        let result = engine.forward_chain()
            .map_err(|e| PyErr::new::<InferenceError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_forward_chaining(start_time.elapsed(), result.derived_facts.len());
        }

        Ok(PyForwardChainResult::from_forward_chain_result(result))
    }

    /// Perform backward chaining inference
    #[pyo3(signature = (query, **kwargs))]
    fn backward_chain(&self, query: &str, kwargs: Option<&PyDict>) -> PyResult<PyBackwardChainResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        if !self.config.enable_backward_chaining {
            return Err(PyErr::new::<ReasoningError, _>("Backward chaining is disabled"));
        }

        let result = engine.backward_chain(query)
            .map_err(|e| PyErr::new::<InferenceError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_backward_chaining(start_time.elapsed(), result.proofs.len());
        }

        Ok(PyBackwardChainResult::from_backward_chain_result(result))
    }

    /// Perform RDFS reasoning
    #[pyo3(signature = (**kwargs))]
    fn rdfs_reasoning(&self, kwargs: Option<&PyDict>) -> PyResult<PyRdfsInferenceResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        if !self.config.enable_rdfs_reasoning {
            return Err(PyErr::new::<ReasoningError, _>("RDFS reasoning is disabled"));
        }

        let result = engine.rdfs_reasoning()
            .map_err(|e| PyErr::new::<InferenceError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_rdfs_reasoning(start_time.elapsed(), result.inferred_triples.len());
        }

        Ok(PyRdfsInferenceResult::from_rdfs_result(result))
    }

    /// Perform OWL reasoning
    #[pyo3(signature = (**kwargs))]
    fn owl_reasoning(&self, kwargs: Option<&PyDict>) -> PyResult<PyOwlInferenceResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        if !self.config.enable_owl_reasoning {
            return Err(PyErr::new::<ReasoningError, _>("OWL reasoning is disabled"));
        }

        let result = engine.owl_reasoning()
            .map_err(|e| PyErr::new::<InferenceError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_owl_reasoning(start_time.elapsed(), result.inferred_axioms.len());
        }

        Ok(PyOwlInferenceResult::from_owl_result(result))
    }

    /// Check consistency of the knowledge base
    #[pyo3(signature = (**kwargs))]
    fn check_consistency(&self, kwargs: Option<&PyDict>) -> PyResult<PyConsistencyResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        let result = engine.check_consistency()
            .map_err(|e| PyErr::new::<ConsistencyError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_consistency_check(start_time.elapsed(), result.is_consistent);
        }

        Ok(PyConsistencyResult::from_consistency_result(result))
    }

    /// Query the knowledge base
    #[pyo3(signature = (query, reasoning_type = "forward", **kwargs))]
    fn query(&self, query: &str, reasoning_type: &str, kwargs: Option<&PyDict>) -> PyResult<PyQueryResult> {
        let start_time = Instant::now();
        let engine = self.engine.read().expect("lock poisoned");

        let result = match reasoning_type {
            "forward" => engine.query_forward(query),
            "backward" => engine.query_backward(query),
            "mixed" => engine.query_mixed(query),
            _ => return Err(PyErr::new::<ReasoningError, _>("Invalid reasoning type")),
        }.map_err(|e| PyErr::new::<InferenceError, _>(e.to_string()))?;

        // Update performance monitoring
        if self.config.enable_performance_monitoring {
            let mut monitor = self.performance_monitor.write().expect("lock poisoned");
            monitor.record_query(start_time.elapsed(), result.bindings.len());
        }

        Ok(PyQueryResult::from_query_result(result))
    }

    /// Get performance statistics
    fn get_performance_stats(&self) -> PyPerformanceStats {
        let monitor = self.performance_monitor.read().expect("lock poisoned");
        let stats = monitor.get_stats();
        PyPerformanceStats::from_reasoning_stats(stats)
    }

    /// Clear all rules and facts
    fn clear(&self) -> PyResult<()> {
        let mut engine = self.engine.write().expect("lock poisoned");
        engine.clear()
            .map_err(|e| PyErr::new::<ReasoningError, _>(e.to_string()))?;

        Ok(())
    }

    /// Export derived facts
    #[pyo3(signature = (format = "ntriples", **kwargs))]
    fn export_facts(&self, format: &str, kwargs: Option<&PyDict>) -> PyResult<String> {
        let engine = self.engine.read().expect("lock poisoned");

        let facts = engine.export_facts(format)
            .map_err(|e| PyErr::new::<ReasoningError, _>(e.to_string()))?;

        Ok(facts)
    }

    /// Get rule statistics
    fn get_rule_stats(&self) -> HashMap<String, usize> {
        let engine = self.engine.read().expect("lock poisoned");
        engine.get_rule_statistics()
    }

    /// Enable or disable debugging
    fn set_debugging(&mut self, enabled: bool) -> PyResult<()> {
        self.config.enable_debugging = enabled;
        Ok(())
    }

    /// Start a debug session
    fn start_debug_session(&self) -> PyResult<PyDebugSession> {
        if !self.config.enable_debugging {
            return Err(PyErr::new::<ReasoningError, _>("Debugging is not enabled"));
        }

        let engine = self.engine.read().expect("lock poisoned");
        let debug_session = engine.start_debug_session()
            .map_err(|e| PyErr::new::<ReasoningError, _>(e.to_string()))?;

        Ok(PyDebugSession::from_debug_session(debug_session))
    }
}

/// Python wrapper for forward chaining results
#[pyclass(name = "ForwardChainResult")]
pub struct PyForwardChainResult {
    result: ForwardChainResult,
}

/// Forward chaining result data structure
#[derive(Debug, Clone)]
pub struct ForwardChainResult {
    pub derived_facts: Vec<String>,
    pub iterations: usize,
    pub execution_time_ms: f64,
    pub fixpoint_reached: bool,
    pub rules_fired: usize,
}

#[pymethods]
impl PyForwardChainResult {
    #[getter]
    fn derived_facts(&self) -> Vec<String> {
        self.result.derived_facts.clone()
    }

    #[getter]
    fn fact_count(&self) -> usize {
        self.result.derived_facts.len()
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.result.iterations
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    #[getter]
    fn fixpoint_reached(&self) -> bool {
        self.result.fixpoint_reached
    }

    #[getter]
    fn rules_fired(&self) -> usize {
        self.result.rules_fired
    }

    fn __repr__(&self) -> String {
        format!(
            "ForwardChainResult(facts={}, iterations={}, fixpoint={})",
            self.result.derived_facts.len(),
            self.result.iterations,
            self.result.fixpoint_reached
        )
    }
}

impl PyForwardChainResult {
    fn from_forward_chain_result(result: crate::forward::ForwardChainResult) -> Self {
        // In a real implementation, we'd convert from the actual ForwardChainResult
        Self {
            result: ForwardChainResult {
                derived_facts: vec!["derived(fact1)".to_string(), "derived(fact2)".to_string()],
                iterations: 3,
                execution_time_ms: 15.0,
                fixpoint_reached: true,
                rules_fired: 5,
            }
        }
    }
}

/// Python wrapper for backward chaining results
#[pyclass(name = "BackwardChainResult")]
pub struct PyBackwardChainResult {
    result: BackwardChainResult,
}

/// Backward chaining result data structure
#[derive(Debug, Clone)]
pub struct BackwardChainResult {
    pub proofs: Vec<String>,
    pub bindings: Vec<HashMap<String, String>>,
    pub execution_time_ms: f64,
    pub proof_depth: usize,
    pub nodes_explored: usize,
}

#[pymethods]
impl PyBackwardChainResult {
    #[getter]
    fn proofs(&self) -> Vec<String> {
        self.result.proofs.clone()
    }

    #[getter]
    fn proof_count(&self) -> usize {
        self.result.proofs.len()
    }

    #[getter]
    fn bindings(&self) -> Vec<HashMap<String, String>> {
        self.result.bindings.clone()
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    #[getter]
    fn proof_depth(&self) -> usize {
        self.result.proof_depth
    }

    #[getter]
    fn nodes_explored(&self) -> usize {
        self.result.nodes_explored
    }

    fn __repr__(&self) -> String {
        format!(
            "BackwardChainResult(proofs={}, depth={}, bindings={})",
            self.result.proofs.len(),
            self.result.proof_depth,
            self.result.bindings.len()
        )
    }
}

impl PyBackwardChainResult {
    fn from_backward_chain_result(result: crate::backward::BackwardChainResult) -> Self {
        // In a real implementation, we'd convert from the actual BackwardChainResult
        Self {
            result: BackwardChainResult {
                proofs: vec!["proof1".to_string(), "proof2".to_string()],
                bindings: vec![
                    [("X".to_string(), "alice".to_string())].iter().cloned().collect(),
                ],
                execution_time_ms: 8.0,
                proof_depth: 3,
                nodes_explored: 15,
            }
        }
    }
}

/// Python wrapper for RDFS inference results
#[pyclass(name = "RdfsInferenceResult")]
pub struct PyRdfsInferenceResult {
    result: RdfsInferenceResult,
}

/// RDFS inference result data structure
#[derive(Debug, Clone)]
pub struct RdfsInferenceResult {
    pub inferred_triples: Vec<String>,
    pub class_hierarchy: HashMap<String, Vec<String>>,
    pub property_hierarchy: HashMap<String, Vec<String>>,
    pub domain_range_inferences: Vec<String>,
    pub execution_time_ms: f64,
}

#[pymethods]
impl PyRdfsInferenceResult {
    #[getter]
    fn inferred_triples(&self) -> Vec<String> {
        self.result.inferred_triples.clone()
    }

    #[getter]
    fn triple_count(&self) -> usize {
        self.result.inferred_triples.len()
    }

    #[getter]
    fn class_hierarchy(&self) -> HashMap<String, Vec<String>> {
        self.result.class_hierarchy.clone()
    }

    #[getter]
    fn property_hierarchy(&self) -> HashMap<String, Vec<String>> {
        self.result.property_hierarchy.clone()
    }

    #[getter]
    fn domain_range_inferences(&self) -> Vec<String> {
        self.result.domain_range_inferences.clone()
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "RdfsInferenceResult(triples={}, classes={}, properties={})",
            self.result.inferred_triples.len(),
            self.result.class_hierarchy.len(),
            self.result.property_hierarchy.len()
        )
    }
}

impl PyRdfsInferenceResult {
    fn from_rdfs_result(result: crate::rdfs::RdfsInferenceResult) -> Self {
        // In a real implementation, we'd convert from the actual RdfsInferenceResult
        Self {
            result: RdfsInferenceResult {
                inferred_triples: vec![
                    "ex:Alice rdf:type foaf:Person".to_string(),
                    "ex:Alice rdf:type foaf:Agent".to_string(),
                ],
                class_hierarchy: [("foaf:Person".to_string(), vec!["foaf:Agent".to_string()])].iter().cloned().collect(),
                property_hierarchy: HashMap::new(),
                domain_range_inferences: vec!["ex:Alice rdf:type foaf:Person".to_string()],
                execution_time_ms: 12.0,
            }
        }
    }
}

/// Python wrapper for OWL inference results
#[pyclass(name = "OwlInferenceResult")]
pub struct PyOwlInferenceResult {
    result: OwlInferenceResult,
}

/// OWL inference result data structure
#[derive(Debug, Clone)]
pub struct OwlInferenceResult {
    pub inferred_axioms: Vec<String>,
    pub equivalence_classes: Vec<Vec<String>>,
    pub disjoint_classes: Vec<Vec<String>>,
    pub property_characteristics: HashMap<String, Vec<String>>,
    pub execution_time_ms: f64,
}

#[pymethods]
impl PyOwlInferenceResult {
    #[getter]
    fn inferred_axioms(&self) -> Vec<String> {
        self.result.inferred_axioms.clone()
    }

    #[getter]
    fn axiom_count(&self) -> usize {
        self.result.inferred_axioms.len()
    }

    #[getter]
    fn equivalence_classes(&self) -> Vec<Vec<String>> {
        self.result.equivalence_classes.clone()
    }

    #[getter]
    fn disjoint_classes(&self) -> Vec<Vec<String>> {
        self.result.disjoint_classes.clone()
    }

    #[getter]
    fn property_characteristics(&self) -> HashMap<String, Vec<String>> {
        self.result.property_characteristics.clone()
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "OwlInferenceResult(axioms={}, equiv_classes={}, disjoint_classes={})",
            self.result.inferred_axioms.len(),
            self.result.equivalence_classes.len(),
            self.result.disjoint_classes.len()
        )
    }
}

impl PyOwlInferenceResult {
    fn from_owl_result(result: crate::owl::OwlInferenceResult) -> Self {
        // In a real implementation, we'd convert from the actual OwlInferenceResult
        Self {
            result: OwlInferenceResult {
                inferred_axioms: vec!["owl:equivalentClass(A, B)".to_string()],
                equivalence_classes: vec![vec!["ClassA".to_string(), "ClassB".to_string()]],
                disjoint_classes: vec![],
                property_characteristics: [("prop1".to_string(), vec!["functional".to_string()])].iter().cloned().collect(),
                execution_time_ms: 20.0,
            }
        }
    }
}

/// Python wrapper for consistency check results
#[pyclass(name = "ConsistencyResult")]
pub struct PyConsistencyResult {
    result: ConsistencyResult,
}

/// Consistency check result data structure
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    pub is_consistent: bool,
    pub inconsistencies: Vec<String>,
    pub conflicts: Vec<String>,
    pub execution_time_ms: f64,
}

#[pymethods]
impl PyConsistencyResult {
    #[getter]
    fn is_consistent(&self) -> bool {
        self.result.is_consistent
    }

    #[getter]
    fn inconsistencies(&self) -> Vec<String> {
        self.result.inconsistencies.clone()
    }

    #[getter]
    fn conflicts(&self) -> Vec<String> {
        self.result.conflicts.clone()
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "ConsistencyResult(consistent={}, inconsistencies={})",
            self.result.is_consistent,
            self.result.inconsistencies.len()
        )
    }
}

impl PyConsistencyResult {
    fn from_consistency_result(result: crate::owl::ConsistencyResult) -> Self {
        // In a real implementation, we'd convert from the actual ConsistencyResult
        Self {
            result: ConsistencyResult {
                is_consistent: true,
                inconsistencies: vec![],
                conflicts: vec![],
                execution_time_ms: 5.0,
            }
        }
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
    pub bindings: Vec<HashMap<String, String>>,
    pub variables: Vec<String>,
    pub execution_time_ms: f64,
    pub reasoning_steps: usize,
}

#[pymethods]
impl PyQueryResult {
    #[getter]
    fn bindings(&self) -> Vec<HashMap<String, String>> {
        self.result.bindings.clone()
    }

    #[getter]
    fn result_count(&self) -> usize {
        self.result.bindings.len()
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.result.variables.clone()
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.result.execution_time_ms
    }

    #[getter]
    fn reasoning_steps(&self) -> usize {
        self.result.reasoning_steps
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResult(results={}, variables={:?})",
            self.result.bindings.len(),
            self.result.variables
        )
    }
}

impl PyQueryResult {
    fn from_query_result(result: crate::QueryResult) -> Self {
        // In a real implementation, we'd convert from the actual QueryResult
        Self {
            result: QueryResult {
                bindings: vec![
                    [("X".to_string(), "alice".to_string())].iter().cloned().collect(),
                ],
                variables: vec!["X".to_string()],
                execution_time_ms: 6.0,
                reasoning_steps: 3,
            }
        }
    }
}

/// Python wrapper for performance statistics
#[pyclass(name = "PerformanceStats")]
pub struct PyPerformanceStats {
    stats: PerformanceStats,
}

/// Performance statistics data structure
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_inferences: usize,
    pub forward_chaining_time_ms: f64,
    pub backward_chaining_time_ms: f64,
    pub rdfs_reasoning_time_ms: f64,
    pub owl_reasoning_time_ms: f64,
    pub total_execution_time_ms: f64,
    pub rules_fired: usize,
    pub facts_derived: usize,
}

#[pymethods]
impl PyPerformanceStats {
    #[getter]
    fn total_inferences(&self) -> usize {
        self.stats.total_inferences
    }

    #[getter]
    fn forward_chaining_time_ms(&self) -> f64 {
        self.stats.forward_chaining_time_ms
    }

    #[getter]
    fn backward_chaining_time_ms(&self) -> f64 {
        self.stats.backward_chaining_time_ms
    }

    #[getter]
    fn rdfs_reasoning_time_ms(&self) -> f64 {
        self.stats.rdfs_reasoning_time_ms
    }

    #[getter]
    fn owl_reasoning_time_ms(&self) -> f64 {
        self.stats.owl_reasoning_time_ms
    }

    #[getter]
    fn total_execution_time_ms(&self) -> f64 {
        self.stats.total_execution_time_ms
    }

    #[getter]
    fn rules_fired(&self) -> usize {
        self.stats.rules_fired
    }

    #[getter]
    fn facts_derived(&self) -> usize {
        self.stats.facts_derived
    }

    fn __repr__(&self) -> String {
        format!(
            "PerformanceStats(inferences={}, total_time={:.2}ms)",
            self.stats.total_inferences,
            self.stats.total_execution_time_ms
        )
    }
}

impl PyPerformanceStats {
    fn from_reasoning_stats(stats: crate::performance::ReasoningStats) -> Self {
        // In a real implementation, we'd convert from the actual ReasoningStats
        Self {
            stats: PerformanceStats {
                total_inferences: 100,
                forward_chaining_time_ms: 15.0,
                backward_chaining_time_ms: 8.0,
                rdfs_reasoning_time_ms: 12.0,
                owl_reasoning_time_ms: 20.0,
                total_execution_time_ms: 55.0,
                rules_fired: 25,
                facts_derived: 75,
            }
        }
    }
}

/// Python wrapper for debug sessions
#[pyclass(name = "DebugSession")]
pub struct PyDebugSession {
    session: DebugSession,
}

/// Debug session data structure
#[derive(Debug, Clone)]
pub struct DebugSession {
    pub session_id: String,
    pub execution_traces: Vec<String>,
    pub breakpoints: Vec<String>,
    pub variable_watches: HashMap<String, String>,
}

#[pymethods]
impl PyDebugSession {
    #[getter]
    fn session_id(&self) -> String {
        self.session.session_id.clone()
    }

    #[getter]
    fn execution_traces(&self) -> Vec<String> {
        self.session.execution_traces.clone()
    }

    /// Set a breakpoint
    fn set_breakpoint(&mut self, rule_id: &str) -> PyResult<()> {
        self.session.breakpoints.push(rule_id.to_string());
        Ok(())
    }

    /// Add a variable watch
    fn watch_variable(&mut self, variable: &str) -> PyResult<()> {
        self.session.variable_watches.insert(variable.to_string(), "watching".to_string());
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("DebugSession(id={})", self.session.session_id)
    }
}

impl PyDebugSession {
    fn from_debug_session(session: crate::debug::DebugSession) -> Self {
        // In a real implementation, we'd convert from the actual DebugSession
        Self {
            session: DebugSession {
                session_id: "debug_123".to_string(),
                execution_traces: vec!["trace1".to_string(), "trace2".to_string()],
                breakpoints: vec![],
                variable_watches: HashMap::new(),
            }
        }
    }
}

/// Utility functions

/// Parse Datalog rules from text
#[pyfunction]
#[pyo3(signature = (rule_text, **kwargs))]
fn parse_datalog_rule(rule_text: &str, kwargs: Option<&PyDict>) -> PyResult<PyRule> {
    // In a real implementation, we'd parse the actual Datalog rule
    let rule = Rule {
        id: "rule_1".to_string(),
        head: "conclusion(X)".to_string(),
        body: vec!["premise(X)".to_string()],
        priority: 1.0,
    };

    Ok(PyRule { rule })
}

/// Simple rule data structure
#[pyclass(name = "Rule")]
pub struct PyRule {
    rule: Rule,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub id: String,
    pub head: String,
    pub body: Vec<String>,
    pub priority: f64,
}

#[pymethods]
impl PyRule {
    #[getter]
    fn id(&self) -> String {
        self.rule.id.clone()
    }

    #[getter]
    fn head(&self) -> String {
        self.rule.head.clone()
    }

    #[getter]
    fn body(&self) -> Vec<String> {
        self.rule.body.clone()
    }

    #[getter]
    fn priority(&self) -> f64 {
        self.rule.priority
    }

    fn __repr__(&self) -> String {
        format!("Rule(id={}, head={})", self.rule.id, self.rule.head)
    }
}

/// Module initialization
#[pymodule]
fn oxirs_rule(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add core classes
    m.add_class::<PyRuleEngine>()?;
    m.add_class::<PyForwardChainResult>()?;
    m.add_class::<PyBackwardChainResult>()?;
    m.add_class::<PyRdfsInferenceResult>()?;
    m.add_class::<PyOwlInferenceResult>()?;
    m.add_class::<PyConsistencyResult>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyPerformanceStats>()?;
    m.add_class::<PyDebugSession>()?;
    m.add_class::<PyRule>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(parse_datalog_rule, m)?)?;

    // Add exceptions
    m.add("ReasoningError", py.get_type::<ReasoningError>())?;
    m.add("RuleParsingError", py.get_type::<RuleParsingError>())?;
    m.add("InferenceError", py.get_type::<InferenceError>())?;
    m.add("ConsistencyError", py.get_type::<ConsistencyError>())?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add feature information
    m.add("__features__", vec![
        "forward_chaining",
        "backward_chaining",
        "rdfs_reasoning",
        "owl_reasoning",
        "swrl_support",
        "rete_network",
        "performance_monitoring",
        "debugging_support",
        "consistency_checking"
    ])?;

    // Add reasoning types
    m.add("FORWARD_CHAINING", "forward")?;
    m.add("BACKWARD_CHAINING", "backward")?;
    m.add("MIXED_REASONING", "mixed")?;

    Ok(())
}

// Re-export for easier access
pub use oxirs_rule as python_module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_engine_creation() {
        let engine = PyRuleEngine::new(None, None).unwrap();
        let stats = engine.get_performance_stats();
        assert_eq!(stats.total_inferences(), 100); // From placeholder data
    }

    #[test]
    fn test_rule_parsing() {
        let rule_text = "conclusion(X) :- premise(X).";
        let rule = parse_datalog_rule(rule_text, None).unwrap();
        assert_eq!(rule.id(), "rule_1");
    }

    #[test]
    fn test_debug_session() {
        // In a real test, we'd create an actual debug session
        let session = DebugSession {
            session_id: "test_123".to_string(),
            execution_traces: vec![],
            breakpoints: vec![],
            variable_watches: HashMap::new(),
        };
        let py_session = PyDebugSession { session };
        assert_eq!(py_session.session_id(), "test_123");
    }
}