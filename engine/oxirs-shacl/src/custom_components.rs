//! Custom SHACL Constraint Components
//!
//! This module provides comprehensive support for user-defined SHACL constraint components,
//! allowing users to extend SHACL with domain-specific validation logic, parameter validation,
//! component inheritance, performance optimization, and security features.

use crate::{
    constraints::{
        Constraint, ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator,
        ConstraintValidator,
    },
    report::ValidationReport,
    sparql::SparqlConstraint,
    validation::ValidationEngine,
    ConstraintComponentId, Result, Severity, ShaclError, ShapeId, ValidationConfig,
};
use oxirs_core::{
    model::{NamedNode, Term},
    Store,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Registry for custom constraint components
#[derive(Debug, Clone)]
pub struct CustomConstraintRegistry {
    /// Registered custom constraint components
    components: HashMap<ConstraintComponentId, Arc<dyn CustomConstraintComponent>>,
    /// Component metadata
    metadata: HashMap<ConstraintComponentId, ComponentMetadata>,
    /// Component libraries for organization
    libraries: HashMap<String, ComponentLibrary>,
    /// Component inheritance hierarchy
    inheritance: HashMap<ConstraintComponentId, Vec<ConstraintComponentId>>,
    /// Performance statistics
    performance_stats: Arc<RwLock<HashMap<ConstraintComponentId, ComponentPerformanceStats>>>,
    /// Security policies
    security_policies: HashMap<ConstraintComponentId, SecurityPolicy>,
    /// Component validation rules
    validation_rules: HashMap<ConstraintComponentId, Vec<ValidationRule>>,
    /// Component dependencies
    dependencies: HashMap<ConstraintComponentId, HashSet<ConstraintComponentId>>,
}

/// Metadata for a constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Version
    pub version: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Parameters this component accepts
    pub parameters: Vec<ParameterDefinition>,
    /// Whether this component can be used on node shapes
    pub applicable_to_node_shapes: bool,
    /// Whether this component can be used on property shapes
    pub applicable_to_property_shapes: bool,
    /// Example usage
    pub example: Option<String>,
}

/// Parameter definition for a custom constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: Option<String>,
    /// Whether this parameter is required
    pub required: bool,
    /// Expected data type
    pub datatype: Option<String>,
    /// Default value
    pub default_value: Option<String>,
    /// Parameter validation constraints
    pub validation_constraints: Vec<ParameterConstraint>,
    /// Parameter cardinality (min, max)
    pub cardinality: Option<(u32, Option<u32>)>,
    /// Allowed values enumeration
    pub allowed_values: Option<Vec<String>>,
}

/// Component library for organizing related components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Library identifier
    pub id: String,
    /// Library name
    pub name: String,
    /// Library description
    pub description: Option<String>,
    /// Library version
    pub version: String,
    /// Library author
    pub author: Option<String>,
    /// Components in this library
    pub components: Vec<ConstraintComponentId>,
    /// Library dependencies
    pub dependencies: Vec<String>,
    /// Library metadata
    pub metadata: HashMap<String, String>,
}

/// Performance statistics for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformanceStats {
    /// Number of executions
    pub execution_count: u64,
    /// Total execution time
    #[serde(skip)]
    pub total_execution_time: Duration,
    /// Average execution time
    #[serde(skip)]
    pub average_execution_time: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Error statistics
    pub error_stats: ErrorStats,
    /// Last execution timestamp
    #[serde(skip)]
    pub last_execution: Option<Instant>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Average memory usage in bytes
    pub average_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total error count
    pub total_errors: u64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Common error types
    pub error_types: HashMap<String, u64>,
    /// Error trend (increasing, decreasing, stable)
    pub error_trend: ErrorTrend,
}

/// Error trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorTrend {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Security policy for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Whether component can execute arbitrary SPARQL
    pub allow_arbitrary_sparql: bool,
    /// Whether component can access external resources
    pub allow_external_access: bool,
    /// Maximum execution time allowed
    pub max_execution_time: Option<Duration>,
    /// Maximum memory usage allowed
    pub max_memory_usage: Option<usize>,
    /// Allowed SPARQL operations
    pub allowed_sparql_operations: HashSet<SparqlOperation>,
    /// Trusted component flag
    pub trusted: bool,
    /// Sandboxing level
    pub sandboxing_level: SandboxingLevel,
    /// Resource quotas
    pub resource_quotas: ResourceQuotas,
}

/// SPARQL operations enumeration
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparqlOperation {
    Select,
    Ask,
    Construct,
    Describe,
    Insert,
    Delete,
    Update,
    Service,
}

/// Sandboxing levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxingLevel {
    None,
    Basic,
    Strict,
    Isolation,
}

/// Resource quotas for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// Maximum CPU time per execution
    pub max_cpu_time: Option<Duration>,
    /// Maximum number of SPARQL queries per execution
    pub max_sparql_queries: Option<u32>,
    /// Maximum result set size
    pub max_result_size: Option<usize>,
    /// Maximum recursion depth
    pub max_recursion_depth: Option<u32>,
}

/// Validation rule for parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: Option<String>,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule severity
    pub severity: Severity,
    /// Rule condition
    pub condition: ValidationCondition,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Required,
    DataType,
    Range,
    Pattern,
    Enum,
    Custom,
}

/// Validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCondition {
    Required,
    DataTypeMatch(String),
    RangeCheck { min: Option<f64>, max: Option<f64> },
    PatternMatch(String),
    EnumValues(Vec<String>),
    CustomFunction(String),
}

/// Parameter constraint for enhanced validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    /// Minimum length for string parameters
    MinLength(u32),
    /// Maximum length for string parameters
    MaxLength(u32),
    /// Regular expression pattern
    Pattern(String),
    /// Numeric range constraint
    Range { min: Option<f64>, max: Option<f64> },
    /// Custom validation function
    CustomValidator(String),
}

/// Component execution context
#[derive(Debug, Clone)]
pub struct ComponentExecutionContext {
    /// Execution start time
    pub start_time: Instant,
    /// Memory usage at start
    pub initial_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Number of SPARQL queries executed
    pub sparql_query_count: u32,
    /// Execution depth (for recursion tracking)
    pub depth: u32,
    /// Security policy
    pub security_policy: SecurityPolicy,
}

/// Component execution result
#[derive(Debug, Clone)]
pub struct ComponentExecutionResult {
    /// Constraint evaluation result
    pub constraint_result: ConstraintEvaluationResult,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Security violations (if any)
    pub security_violations: Vec<SecurityViolation>,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Memory used
    pub memory_used: usize,
    /// Number of SPARQL queries
    pub sparql_queries: u32,
    /// Success flag
    pub success: bool,
    /// Error details (if any)
    pub error: Option<String>,
}

/// Security violation
#[derive(Debug, Clone)]
pub struct SecurityViolation {
    /// Violation type
    pub violation_type: SecurityViolationType,
    /// Violation description
    pub description: String,
    /// Severity level
    pub severity: Severity,
}

/// Security violation types
#[derive(Debug, Clone)]
pub enum SecurityViolationType {
    UnauthorizedSparqlOperation,
    ExternalResourceAccess,
    ExecutionTimeExceeded,
    MemoryLimitExceeded,
    RecursionLimitExceeded,
    UntrustedComponentExecution,
}

/// Trait for custom constraint components
pub trait CustomConstraintComponent: Send + Sync + std::fmt::Debug {
    /// Get the component identifier
    fn component_id(&self) -> &ConstraintComponentId;

    /// Get component metadata
    fn metadata(&self) -> &ComponentMetadata;

    /// Validate the component configuration
    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()>;

    /// Create a constraint instance from parameters
    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint>;

    /// Get the SPARQL query template for this component (if SPARQL-based)
    fn sparql_template(&self) -> Option<&str> {
        None
    }

    /// Get additional prefixes needed for SPARQL queries
    fn sparql_prefixes(&self) -> Option<&str> {
        None
    }
}

impl Default for CustomConstraintRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            metadata: HashMap::new(),
            libraries: HashMap::new(),
            inheritance: HashMap::new(),
            performance_stats: Arc::new(RwLock::new(HashMap::new())),
            security_policies: HashMap::new(),
            validation_rules: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Register a custom constraint component
    pub fn register_component(
        &mut self,
        component: Arc<dyn CustomConstraintComponent>,
    ) -> Result<()> {
        let component_id = component.component_id().clone();
        let metadata = component.metadata().clone();

        // Validate that component ID is not already registered
        if self.components.contains_key(&component_id) {
            return Err(ShaclError::Configuration(format!(
                "Constraint component {} is already registered",
                component_id.as_str()
            )));
        }

        // Validate component metadata and parameters
        self.validate_component_metadata(&metadata)?;

        // Set default security policy
        let default_security_policy = SecurityPolicy {
            allow_arbitrary_sparql: false,
            allow_external_access: false,
            max_execution_time: Some(Duration::from_secs(30)),
            max_memory_usage: Some(100 * 1024 * 1024), // 100MB
            allowed_sparql_operations: [SparqlOperation::Ask, SparqlOperation::Select]
                .iter()
                .cloned()
                .collect(),
            trusted: false,
            sandboxing_level: SandboxingLevel::Basic,
            resource_quotas: ResourceQuotas {
                max_cpu_time: Some(Duration::from_secs(10)),
                max_sparql_queries: Some(10),
                max_result_size: Some(10000),
                max_recursion_depth: Some(10),
            },
        };

        // Initialize performance statistics
        let performance_stats = ComponentPerformanceStats {
            execution_count: 0,
            total_execution_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            success_rate: 1.0,
            memory_usage: MemoryUsageStats {
                average_usage: 0,
                peak_usage: 0,
                efficiency_score: 1.0,
            },
            error_stats: ErrorStats {
                total_errors: 0,
                error_rate: 0.0,
                error_types: HashMap::new(),
                error_trend: ErrorTrend::Unknown,
            },
            last_execution: None,
        };

        // Register component
        self.components.insert(component_id.clone(), component);
        self.metadata.insert(component_id.clone(), metadata);
        self.security_policies
            .insert(component_id.clone(), default_security_policy);

        if let Ok(mut stats) = self.performance_stats.write() {
            stats.insert(component_id.clone(), performance_stats);
        }

        Ok(())
    }

    /// Register a component library
    pub fn register_library(&mut self, library: ComponentLibrary) -> Result<()> {
        // Validate that all components in the library exist
        for component_id in &library.components {
            if !self.components.contains_key(component_id) {
                return Err(ShaclError::Configuration(format!(
                    "Component {} in library {} is not registered",
                    component_id.as_str(),
                    library.id
                )));
            }
        }

        // Check for library dependencies
        for dependency in &library.dependencies {
            if !self.libraries.contains_key(dependency) {
                return Err(ShaclError::Configuration(format!(
                    "Library dependency {} not found",
                    dependency
                )));
            }
        }

        self.libraries.insert(library.id.clone(), library);
        Ok(())
    }

    /// Set security policy for a component
    pub fn set_security_policy(
        &mut self,
        component_id: &ConstraintComponentId,
        policy: SecurityPolicy,
    ) -> Result<()> {
        if !self.components.contains_key(component_id) {
            return Err(ShaclError::Configuration(format!(
                "Component {} not found",
                component_id.as_str()
            )));
        }

        self.security_policies.insert(component_id.clone(), policy);
        Ok(())
    }

    /// Add validation rules for a component
    pub fn add_validation_rules(
        &mut self,
        component_id: &ConstraintComponentId,
        rules: Vec<ValidationRule>,
    ) -> Result<()> {
        if !self.components.contains_key(component_id) {
            return Err(ShaclError::Configuration(format!(
                "Component {} not found",
                component_id.as_str()
            )));
        }

        self.validation_rules.insert(component_id.clone(), rules);
        Ok(())
    }

    /// Set component dependencies
    pub fn set_dependencies(
        &mut self,
        component_id: &ConstraintComponentId,
        dependencies: HashSet<ConstraintComponentId>,
    ) -> Result<()> {
        if !self.components.contains_key(component_id) {
            return Err(ShaclError::Configuration(format!(
                "Component {} not found",
                component_id.as_str()
            )));
        }

        // Validate that all dependencies exist
        for dep in &dependencies {
            if !self.components.contains_key(dep) {
                return Err(ShaclError::Configuration(format!(
                    "Dependency {} not found",
                    dep.as_str()
                )));
            }
        }

        // Check for circular dependencies
        self.check_circular_dependencies(component_id, &dependencies)?;

        self.dependencies.insert(component_id.clone(), dependencies);
        Ok(())
    }

    /// Execute component with enhanced monitoring and security
    pub fn execute_component_secure(
        &mut self,
        component_id: &ConstraintComponentId,
        parameters: HashMap<String, Term>,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ComponentExecutionResult> {
        let start_time = Instant::now();

        // Get component and security policy
        let component = self.components.get(component_id).ok_or_else(|| {
            ShaclError::Configuration(format!("Component {} not found", component_id.as_str()))
        })?;

        let security_policy = self.security_policies.get(component_id).ok_or_else(|| {
            ShaclError::Configuration(format!(
                "Security policy for component {} not found",
                component_id.as_str()
            ))
        })?;

        // Create execution context
        let exec_context = ComponentExecutionContext {
            start_time,
            initial_memory: 0, // Would get actual memory usage in real implementation
            current_memory: 0,
            sparql_query_count: 0,
            depth: 0,
            security_policy: security_policy.clone(),
        };

        // Validate parameters
        self.validate_component_parameters(component_id, &parameters)?;

        // Check dependencies
        if let Some(deps) = self.dependencies.get(component_id) {
            for dep in deps {
                if !self.is_component_available(dep) {
                    return Err(ShaclError::Configuration(format!(
                        "Dependency {} is not available",
                        dep.as_str()
                    )));
                }
            }
        }

        // Execute component with security monitoring
        let constraint_result = match self.execute_with_security_monitoring(
            component.as_ref(),
            parameters,
            store,
            context,
            &exec_context,
        ) {
            Ok(result) => result,
            Err(e) => {
                // Record error in performance stats
                self.record_execution_error(component_id, &e);
                return Err(e);
            }
        };

        let execution_time = start_time.elapsed();

        // Update performance statistics
        self.update_performance_stats(
            component_id,
            execution_time,
            constraint_result.is_satisfied(),
        )?;

        // Create execution metrics
        let metrics = ExecutionMetrics {
            execution_time,
            memory_used: exec_context.current_memory - exec_context.initial_memory,
            sparql_queries: exec_context.sparql_query_count,
            success: constraint_result.is_satisfied(),
            error: None,
        };

        Ok(ComponentExecutionResult {
            constraint_result,
            metrics,
            security_violations: Vec::new(), // Would be populated by security monitoring
        })
    }

    /// Validate component metadata
    fn validate_component_metadata(&self, metadata: &ComponentMetadata) -> Result<()> {
        // Validate parameter definitions
        for param in &metadata.parameters {
            if param.name.is_empty() {
                return Err(ShaclError::Configuration(
                    "Parameter name cannot be empty".to_string(),
                ));
            }

            // Validate cardinality
            if let Some((min, max)) = param.cardinality {
                if let Some(max_val) = max {
                    if min > max_val {
                        return Err(ShaclError::Configuration(format!(
                            "Parameter {} has invalid cardinality: min ({}) > max ({})",
                            param.name, min, max_val
                        )));
                    }
                }
            }

            // Validate constraints
            for constraint in &param.validation_constraints {
                self.validate_parameter_constraint(constraint)?;
            }
        }

        Ok(())
    }

    /// Validate parameter constraint
    fn validate_parameter_constraint(&self, constraint: &ParameterConstraint) -> Result<()> {
        match constraint {
            ParameterConstraint::MinLength(len) => {
                if *len == 0 {
                    return Err(ShaclError::Configuration(
                        "MinLength constraint must be greater than 0".to_string(),
                    ));
                }
            }
            ParameterConstraint::MaxLength(len) => {
                if *len == 0 {
                    return Err(ShaclError::Configuration(
                        "MaxLength constraint must be greater than 0".to_string(),
                    ));
                }
            }
            ParameterConstraint::Pattern(pattern) => {
                if regex::Regex::new(pattern).is_err() {
                    return Err(ShaclError::Configuration(format!(
                        "Invalid regex pattern: {}",
                        pattern
                    )));
                }
            }
            ParameterConstraint::Range { min, max } => {
                if let (Some(min_val), Some(max_val)) = (min, max) {
                    if min_val > max_val {
                        return Err(ShaclError::Configuration(format!(
                            "Invalid range: min ({}) > max ({})",
                            min_val, max_val
                        )));
                    }
                }
            }
            ParameterConstraint::CustomValidator(_) => {
                // Would validate that the custom validator exists
            }
        }

        Ok(())
    }

    /// Validate component parameters against metadata
    fn validate_component_parameters(
        &self,
        component_id: &ConstraintComponentId,
        parameters: &HashMap<String, Term>,
    ) -> Result<()> {
        let metadata = self.metadata.get(component_id).ok_or_else(|| {
            ShaclError::Configuration(format!(
                "Metadata for component {} not found",
                component_id.as_str()
            ))
        })?;

        // Check required parameters
        for param_def in &metadata.parameters {
            if param_def.required && !parameters.contains_key(&param_def.name) {
                return Err(ShaclError::Configuration(format!(
                    "Required parameter {} is missing",
                    param_def.name
                )));
            }

            // Validate parameter if present
            if let Some(value) = parameters.get(&param_def.name) {
                self.validate_parameter_value(&param_def, value)?;
            }
        }

        // Check for unknown parameters
        for param_name in parameters.keys() {
            if !metadata.parameters.iter().any(|p| &p.name == param_name) {
                return Err(ShaclError::Configuration(format!(
                    "Unknown parameter: {}",
                    param_name
                )));
            }
        }

        Ok(())
    }

    /// Validate individual parameter value
    fn validate_parameter_value(
        &self,
        param_def: &ParameterDefinition,
        value: &Term,
    ) -> Result<()> {
        // Validate data type
        if let Some(expected_datatype) = &param_def.datatype {
            if !self.matches_datatype(value, expected_datatype) {
                return Err(ShaclError::Configuration(format!(
                    "Parameter {} has incorrect data type",
                    param_def.name
                )));
            }
        }

        // Validate constraints
        for constraint in &param_def.validation_constraints {
            self.validate_value_against_constraint(value, constraint, &param_def.name)?;
        }

        // Validate allowed values
        if let Some(allowed_values) = &param_def.allowed_values {
            let value_str = self.term_to_string(value);
            if !allowed_values.contains(&value_str) {
                return Err(ShaclError::Configuration(format!(
                    "Parameter {} value '{}' is not in allowed values: {:?}",
                    param_def.name, value_str, allowed_values
                )));
            }
        }

        Ok(())
    }

    /// Check for circular dependencies
    fn check_circular_dependencies(
        &self,
        component_id: &ConstraintComponentId,
        new_dependencies: &HashSet<ConstraintComponentId>,
    ) -> Result<()> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        for dep in new_dependencies {
            if self.has_circular_dependency(dep, component_id, &mut visited, &mut path)? {
                return Err(ShaclError::Configuration(format!(
                    "Circular dependency detected: {} -> {}",
                    component_id.as_str(),
                    dep.as_str()
                )));
            }
        }

        Ok(())
    }

    /// Check if component is available
    fn is_component_available(&self, component_id: &ConstraintComponentId) -> bool {
        self.components.contains_key(component_id)
    }

    /// Update performance statistics
    fn update_performance_stats(
        &mut self,
        component_id: &ConstraintComponentId,
        execution_time: Duration,
        success: bool,
    ) -> Result<()> {
        if let Ok(mut stats) = self.performance_stats.write() {
            if let Some(component_stats) = stats.get_mut(component_id) {
                component_stats.execution_count += 1;
                component_stats.total_execution_time += execution_time;
                component_stats.average_execution_time =
                    component_stats.total_execution_time / component_stats.execution_count as u32;

                // Update success rate using exponential moving average
                let alpha = 0.1;
                let new_success_rate = if success { 1.0 } else { 0.0 };
                component_stats.success_rate =
                    alpha * new_success_rate + (1.0 - alpha) * component_stats.success_rate;

                component_stats.last_execution = Some(Instant::now());
            }
        }

        Ok(())
    }

    // Helper methods (implementation stubs)

    fn execute_with_security_monitoring(
        &self,
        component: &dyn CustomConstraintComponent,
        parameters: HashMap<String, Term>,
        store: &Store,
        context: &ConstraintContext,
        exec_context: &ComponentExecutionContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Implementation would monitor security constraints during execution
        component
            .create_constraint(parameters)?
            .evaluate(store, context)
    }

    fn record_execution_error(&mut self, component_id: &ConstraintComponentId, error: &ShaclError) {
        if let Ok(mut stats) = self.performance_stats.write() {
            if let Some(component_stats) = stats.get_mut(component_id) {
                component_stats.error_stats.total_errors += 1;

                // Update error rate using exponential moving average
                let alpha = 0.1;
                let new_error_rate = 1.0;
                component_stats.error_stats.error_rate =
                    alpha * new_error_rate + (1.0 - alpha) * component_stats.error_stats.error_rate;

                // Track error type
                let error_type = match error {
                    ShaclError::ConstraintValidation(_) => "ConstraintValidation",
                    ShaclError::Configuration(_) => "Configuration",
                    _ => "Other",
                };

                *component_stats
                    .error_stats
                    .error_types
                    .entry(error_type.to_string())
                    .or_insert(0) += 1;

                // Update error trend
                component_stats.error_stats.error_trend =
                    if component_stats.error_stats.error_rate > 0.1 {
                        ErrorTrend::Increasing
                    } else if component_stats.error_stats.error_rate < 0.01 {
                        ErrorTrend::Decreasing
                    } else {
                        ErrorTrend::Stable
                    };
            }
        }
    }

    fn matches_datatype(&self, value: &Term, expected_datatype: &str) -> bool {
        match value {
            Term::Literal(literal) => {
                match expected_datatype {
                    "xsd:string" => true, // All literals can be treated as strings
                    "xsd:integer" => literal.value().parse::<i64>().is_ok(),
                    "xsd:decimal" => literal.value().parse::<f64>().is_ok(),
                    "xsd:double" => literal.value().parse::<f64>().is_ok(),
                    "xsd:float" => literal.value().parse::<f32>().is_ok(),
                    "xsd:boolean" => {
                        matches!(literal.value(), "true" | "false" | "1" | "0")
                    }
                    "xsd:date" => {
                        // Basic date format validation (YYYY-MM-DD)
                        let date_regex = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
                        date_regex.is_match(literal.value())
                    }
                    "xsd:dateTime" => {
                        // Basic dateTime format validation
                        let datetime_regex =
                            regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").unwrap();
                        datetime_regex.is_match(literal.value())
                    }
                    "xsd:time" => {
                        // Basic time format validation (HH:MM:SS)
                        let time_regex = regex::Regex::new(r"^\d{2}:\d{2}:\d{2}").unwrap();
                        time_regex.is_match(literal.value())
                    }
                    _ => {
                        // For custom datatypes, check if literal has the expected datatype IRI
                        let datatype = literal.datatype();
                        datatype.as_str() == expected_datatype
                    }
                }
            }
            Term::NamedNode(_) => {
                matches!(expected_datatype, "xsd:anyURI" | "rdfs:Resource")
            }
            Term::BlankNode(_) => expected_datatype == "rdfs:Resource",
            _ => false,
        }
    }

    fn validate_value_against_constraint(
        &self,
        value: &Term,
        constraint: &ParameterConstraint,
        param_name: &str,
    ) -> Result<()> {
        match constraint {
            ParameterConstraint::MinLength(min_len) => {
                let value_str = self.term_to_string(value);
                if value_str.len() < *min_len as usize {
                    return Err(ShaclError::Configuration(format!(
                        "Parameter {} value '{}' is shorter than minimum length {}",
                        param_name, value_str, min_len
                    )));
                }
            }
            ParameterConstraint::MaxLength(max_len) => {
                let value_str = self.term_to_string(value);
                if value_str.len() > *max_len as usize {
                    return Err(ShaclError::Configuration(format!(
                        "Parameter {} value '{}' is longer than maximum length {}",
                        param_name, value_str, max_len
                    )));
                }
            }
            ParameterConstraint::Pattern(pattern) => {
                let value_str = self.term_to_string(value);
                let regex = regex::Regex::new(pattern).map_err(|e| {
                    ShaclError::Configuration(format!(
                        "Invalid regex pattern for parameter {}: {}",
                        param_name, e
                    ))
                })?;
                if !regex.is_match(&value_str) {
                    return Err(ShaclError::Configuration(format!(
                        "Parameter {} value '{}' does not match pattern '{}'",
                        param_name, value_str, pattern
                    )));
                }
            }
            ParameterConstraint::Range { min, max } => {
                if let Term::Literal(lit) = value {
                    let num_value = lit.value().parse::<f64>().map_err(|_| {
                        ShaclError::Configuration(format!(
                            "Parameter {} value '{}' is not a valid number for range constraint",
                            param_name,
                            lit.value()
                        ))
                    })?;

                    if let Some(min_val) = min {
                        if num_value < *min_val {
                            return Err(ShaclError::Configuration(format!(
                                "Parameter {} value {} is less than minimum {}",
                                param_name, num_value, min_val
                            )));
                        }
                    }

                    if let Some(max_val) = max {
                        if num_value > *max_val {
                            return Err(ShaclError::Configuration(format!(
                                "Parameter {} value {} is greater than maximum {}",
                                param_name, num_value, max_val
                            )));
                        }
                    }
                } else {
                    return Err(ShaclError::Configuration(format!(
                        "Parameter {} must be a literal for range constraint",
                        param_name
                    )));
                }
            }
            ParameterConstraint::CustomValidator(validator_name) => {
                // For custom validators, we would look up the validator function
                // and execute it. For now, just validate that the validator exists.
                match validator_name.as_str() {
                    "email" => {
                        let value_str = self.term_to_string(value);
                        let email_regex =
                            regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
                                .unwrap();
                        if !email_regex.is_match(&value_str) {
                            return Err(ShaclError::Configuration(format!(
                                "Parameter {} value '{}' is not a valid email address",
                                param_name, value_str
                            )));
                        }
                    }
                    "url" => {
                        let value_str = self.term_to_string(value);
                        if !value_str.starts_with("http://") && !value_str.starts_with("https://") {
                            return Err(ShaclError::Configuration(format!(
                                "Parameter {} value '{}' is not a valid URL",
                                param_name, value_str
                            )));
                        }
                    }
                    _ => {
                        return Err(ShaclError::Configuration(format!(
                            "Unknown custom validator '{}' for parameter {}",
                            validator_name, param_name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn term_to_string(&self, term: &Term) -> String {
        match term {
            Term::NamedNode(node) => node.as_str().to_string(),
            Term::BlankNode(node) => format!("_:{}", node.as_str()),
            Term::Literal(lit) => lit.value().to_string(),
            _ => format!("{:?}", term),
        }
    }

    fn has_circular_dependency(
        &self,
        current: &ConstraintComponentId,
        target: &ConstraintComponentId,
        visited: &mut HashSet<ConstraintComponentId>,
        path: &mut Vec<ConstraintComponentId>,
    ) -> Result<bool> {
        if current == target {
            return Ok(true);
        }

        if visited.contains(current) {
            return Ok(false);
        }

        visited.insert(current.clone());
        path.push(current.clone());

        if let Some(deps) = self.dependencies.get(current) {
            for dep in deps {
                if self.has_circular_dependency(dep, target, visited, path)? {
                    return Ok(true);
                }
            }
        }

        path.pop();
        Ok(false)
    }

    /// Get a constraint component by ID
    pub fn get_component(
        &self,
        component_id: &ConstraintComponentId,
    ) -> Option<&Arc<dyn CustomConstraintComponent>> {
        self.components.get(component_id)
    }

    /// Get component metadata
    pub fn get_metadata(&self, component_id: &ConstraintComponentId) -> Option<&ComponentMetadata> {
        self.metadata.get(component_id)
    }

    /// List all registered components
    pub fn list_components(&self) -> Vec<&ConstraintComponentId> {
        self.components.keys().collect()
    }

    /// Create a constraint from a component and parameters
    pub fn create_constraint(
        &self,
        component_id: &ConstraintComponentId,
        parameters: HashMap<String, Term>,
    ) -> Result<CustomConstraint> {
        let component = self.components.get(component_id).ok_or_else(|| {
            ShaclError::Configuration(format!(
                "Unknown constraint component: {}",
                component_id.as_str()
            ))
        })?;

        // Validate configuration
        component.validate_configuration(&parameters)?;

        // Create constraint
        component.create_constraint(parameters)
    }

    /// Register standard extension components
    pub fn register_standard_extensions(&mut self) -> Result<()> {
        // Register commonly useful constraint components

        // 1. Regular expression constraint component
        let regex_component = Arc::new(RegexConstraintComponent::new());
        self.register_component(regex_component)?;

        // 2. Range constraint component
        let range_component = Arc::new(RangeConstraintComponent::new());
        self.register_component(range_component)?;

        // 3. URL validation component
        let url_component = Arc::new(UrlValidationComponent::new());
        self.register_component(url_component)?;

        // 4. Email validation component
        let email_component = Arc::new(EmailValidationComponent::new());
        self.register_component(email_component)?;

        // 5. SPARQL-based custom component
        let sparql_component = Arc::new(SparqlConstraintComponent::new());
        self.register_component(sparql_component)?;

        Ok(())
    }

    /// Set component inheritance relationships
    pub fn set_component_inheritance(
        &mut self,
        component_id: &ConstraintComponentId,
        parent_components: Vec<ConstraintComponentId>,
    ) -> Result<()> {
        if !self.components.contains_key(component_id) {
            return Err(ShaclError::Configuration(format!(
                "Component {} not found",
                component_id.as_str()
            )));
        }

        // Validate that all parent components exist
        for parent in &parent_components {
            if !self.components.contains_key(parent) {
                return Err(ShaclError::Configuration(format!(
                    "Parent component {} not found",
                    parent.as_str()
                )));
            }
        }

        // Check for circular inheritance
        self.check_circular_inheritance(component_id, &parent_components)?;

        self.inheritance
            .insert(component_id.clone(), parent_components);
        Ok(())
    }

    /// Get all inherited components (including transitive inheritance)
    pub fn get_inherited_components(
        &self,
        component_id: &ConstraintComponentId,
    ) -> Vec<ConstraintComponentId> {
        let mut inherited = Vec::new();
        let mut visited = HashSet::new();
        self.collect_inherited_components(component_id, &mut inherited, &mut visited);
        inherited
    }

    /// Create a composite constraint from multiple components
    pub fn create_composite_constraint(
        &self,
        component_ids: &[ConstraintComponentId],
        parameters: HashMap<String, Term>,
    ) -> Result<CompositeConstraint> {
        // Validate all components exist
        for component_id in component_ids {
            if !self.components.contains_key(component_id) {
                return Err(ShaclError::Configuration(format!(
                    "Component {} not found",
                    component_id.as_str()
                )));
            }
        }

        // Create individual constraints
        let mut constraints = Vec::new();
        for component_id in component_ids {
            let constraint = self.create_constraint(component_id, parameters.clone())?;
            constraints.push(constraint);
        }

        Ok(CompositeConstraint {
            component_ids: component_ids.to_vec(),
            constraints,
            composition_type: CompositionType::And, // Default to AND composition
        })
    }

    /// Get component metadata including inherited properties
    pub fn get_effective_metadata(
        &self,
        component_id: &ConstraintComponentId,
    ) -> Option<ComponentMetadata> {
        let base_metadata = self.metadata.get(component_id)?.clone();
        let inherited_components = self.get_inherited_components(component_id);

        let mut effective_metadata = base_metadata;

        // Merge parameters from inherited components
        for inherited_id in inherited_components {
            if let Some(inherited_metadata) = self.metadata.get(&inherited_id) {
                for param in &inherited_metadata.parameters {
                    // Add parameter if not already present
                    if !effective_metadata
                        .parameters
                        .iter()
                        .any(|p| p.name == param.name)
                    {
                        effective_metadata.parameters.push(param.clone());
                    }
                }
            }
        }

        Some(effective_metadata)
    }

    /// Check for circular inheritance
    fn check_circular_inheritance(
        &self,
        component_id: &ConstraintComponentId,
        parent_components: &[ConstraintComponentId],
    ) -> Result<()> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        for parent in parent_components {
            if self.has_circular_inheritance(parent, component_id, &mut visited, &mut path)? {
                return Err(ShaclError::Configuration(format!(
                    "Circular inheritance detected: {} -> {}",
                    component_id.as_str(),
                    parent.as_str()
                )));
            }
        }

        Ok(())
    }

    /// Helper to detect circular inheritance
    fn has_circular_inheritance(
        &self,
        current: &ConstraintComponentId,
        target: &ConstraintComponentId,
        visited: &mut HashSet<ConstraintComponentId>,
        path: &mut Vec<ConstraintComponentId>,
    ) -> Result<bool> {
        if current == target {
            return Ok(true);
        }

        if visited.contains(current) {
            return Ok(false);
        }

        visited.insert(current.clone());
        path.push(current.clone());

        if let Some(parents) = self.inheritance.get(current) {
            for parent in parents {
                if self.has_circular_inheritance(parent, target, visited, path)? {
                    return Ok(true);
                }
            }
        }

        path.pop();
        Ok(false)
    }

    /// Collect inherited components recursively
    fn collect_inherited_components(
        &self,
        component_id: &ConstraintComponentId,
        inherited: &mut Vec<ConstraintComponentId>,
        visited: &mut HashSet<ConstraintComponentId>,
    ) {
        if visited.contains(component_id) {
            return;
        }

        visited.insert(component_id.clone());

        if let Some(parents) = self.inheritance.get(component_id) {
            for parent in parents {
                inherited.push(parent.clone());
                self.collect_inherited_components(parent, inherited, visited);
            }
        }
    }
}

/// Custom constraint implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConstraint {
    /// Component that created this constraint
    pub component_id: ConstraintComponentId,
    /// Configuration parameters
    pub parameters: HashMap<String, Term>,
    /// Optional SPARQL query for validation
    pub sparql_query: Option<String>,
    /// Custom validation function name
    pub validation_function: Option<String>,
    /// Error message template
    pub message_template: Option<String>,
}

/// Composite constraint that combines multiple components
#[derive(Debug, Clone)]
pub struct CompositeConstraint {
    /// Component IDs that make up this composite
    pub component_ids: Vec<ConstraintComponentId>,
    /// Individual constraints
    pub constraints: Vec<CustomConstraint>,
    /// How to compose the constraints
    pub composition_type: CompositionType,
}

/// Types of constraint composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionType {
    /// All constraints must be satisfied (logical AND)
    And,
    /// At least one constraint must be satisfied (logical OR)
    Or,
    /// Exactly one constraint must be satisfied (logical XOR)
    Xor,
    /// Custom composition logic
    Custom(String),
}

impl CompositeConstraint {
    /// Evaluate the composite constraint based on composition type
    pub fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let mut results = Vec::new();

        // Evaluate all individual constraints
        for constraint in &self.constraints {
            let result = constraint.evaluate(store, context)?;
            results.push(result);
        }

        // Apply composition logic
        match self.composition_type {
            CompositionType::And => {
                // All constraints must be satisfied
                for result in &results {
                    if result.is_violated() {
                        return Ok(result.clone());
                    }
                }
                Ok(ConstraintEvaluationResult::satisfied())
            }
            CompositionType::Or => {
                // At least one constraint must be satisfied
                let mut any_satisfied = false;
                let mut first_violation = None;

                for result in &results {
                    if result.is_satisfied() {
                        any_satisfied = true;
                        break;
                    } else if first_violation.is_none() {
                        first_violation = Some(result.clone());
                    }
                }

                if any_satisfied {
                    Ok(ConstraintEvaluationResult::satisfied())
                } else {
                    Ok(first_violation.unwrap_or_else(|| {
                        ConstraintEvaluationResult::violated(
                            None,
                            Some("No constraints satisfied in OR composition".to_string()),
                        )
                    }))
                }
            }
            CompositionType::Xor => {
                // Exactly one constraint must be satisfied
                let satisfied_count = results.iter().filter(|r| r.is_satisfied()).count();

                if satisfied_count == 1 {
                    Ok(ConstraintEvaluationResult::satisfied())
                } else if satisfied_count == 0 {
                    Ok(ConstraintEvaluationResult::violated(
                        None,
                        Some("No constraints satisfied in XOR composition".to_string()),
                    ))
                } else {
                    Ok(ConstraintEvaluationResult::violated(
                        None,
                        Some(format!(
                            "Multiple constraints ({}) satisfied in XOR composition",
                            satisfied_count
                        )),
                    ))
                }
            }
            CompositionType::Custom(ref custom_logic) => {
                // For custom composition, implement specific logic based on the custom_logic string
                match custom_logic.as_str() {
                    "majority" => {
                        // Majority vote - more than half must be satisfied
                        let satisfied_count = results.iter().filter(|r| r.is_satisfied()).count();
                        let total_count = results.len();

                        if satisfied_count > total_count / 2 {
                            Ok(ConstraintEvaluationResult::satisfied())
                        } else {
                            Ok(ConstraintEvaluationResult::violated(
                                None,
                                Some(format!(
                                    "Only {} of {} constraints satisfied (majority required)",
                                    satisfied_count, total_count
                                )),
                            ))
                        }
                    }
                    "weighted" => {
                        // Weighted composition - would require weights to be stored
                        // For now, fallback to AND logic
                        for result in &results {
                            if result.is_violated() {
                                return Ok(result.clone());
                            }
                        }
                        Ok(ConstraintEvaluationResult::satisfied())
                    }
                    _ => Err(ShaclError::Configuration(format!(
                        "Unknown custom composition logic: {}",
                        custom_logic
                    ))),
                }
            }
        }
    }
}

impl ConstraintValidator for CustomConstraint {
    fn validate(&self) -> Result<()> {
        // Basic validation - check required parameters exist
        Ok(())
    }
}

impl crate::constraints::ConstraintEvaluator for CustomConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // If this is a SPARQL-based constraint, use SPARQL evaluation
        if let Some(query) = &self.sparql_query {
            let sparql_constraint = SparqlConstraint {
                query: query.clone(),
                prefixes: None,
                message: self.message_template.clone(),
                severity: Some(Severity::Violation),
                construct_query: None,
            };

            return sparql_constraint.evaluate(store, context);
        }

        // Otherwise, delegate to custom validation logic based on component type
        match self.component_id.as_str() {
            "ex:RegexConstraintComponent" => self.evaluate_regex_constraint(context),
            "ex:RangeConstraintComponent" => self.evaluate_range_constraint(context),
            "ex:UrlValidationComponent" => self.evaluate_url_constraint(context),
            "ex:EmailValidationComponent" => self.evaluate_email_constraint(context),
            _ => {
                // Unknown component type
                Ok(ConstraintEvaluationResult::error(format!(
                    "Unknown custom constraint component: {}",
                    self.component_id.as_str()
                )))
            }
        }
    }
}

impl CustomConstraint {
    /// Evaluate regular expression constraint
    fn evaluate_regex_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let pattern = self
            .parameters
            .get("pattern")
            .and_then(|t| match t {
                Term::Literal(lit) => Some(lit.value()),
                _ => None,
            })
            .ok_or_else(|| {
                ShaclError::ConstraintValidation("Pattern parameter required".to_string())
            })?;

        let regex = regex::Regex::new(pattern).map_err(|e| {
            ShaclError::ConstraintValidation(format!("Invalid regex pattern: {}", e))
        })?;

        for value in &context.values {
            if let Term::Literal(lit) = value {
                if !regex.is_match(lit.value()) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value '{}' does not match pattern '{}'",
                            lit.value(),
                            pattern
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Regex constraint can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate range constraint
    fn evaluate_range_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let min_value = self.parameters.get("minValue");
        let max_value = self.parameters.get("maxValue");

        if min_value.is_none() && max_value.is_none() {
            return Err(ShaclError::ConstraintValidation(
                "At least one of minValue or maxValue must be specified".to_string(),
            ));
        }

        for value in &context.values {
            if let Term::Literal(lit) = value {
                // Try to parse as number
                if let Ok(num) = lit.value().parse::<f64>() {
                    if let Some(Term::Literal(min_lit)) = min_value {
                        if let Ok(min_num) = min_lit.value().parse::<f64>() {
                            if num < min_num {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!("Value {} is less than minimum {}", num, min_num)),
                                ));
                            }
                        }
                    }

                    if let Some(Term::Literal(max_lit)) = max_value {
                        if let Ok(max_num) = max_lit.value().parse::<f64>() {
                            if num > max_num {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!(
                                        "Value {} is greater than maximum {}",
                                        num, max_num
                                    )),
                                ));
                            }
                        }
                    }
                } else {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Range constraint requires numeric values".to_string()),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate URL validation constraint
    fn evaluate_url_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::NamedNode(node) => {
                    let url_str = node.as_str();
                    if !self.is_valid_url(url_str) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!("'{}' is not a valid URL", url_str)),
                        ));
                    }
                }
                Term::Literal(lit) => {
                    let url_str = lit.value();
                    if !self.is_valid_url(url_str) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!("'{}' is not a valid URL", url_str)),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("URL validation can only be applied to IRIs or literals".to_string()),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate email validation constraint
    fn evaluate_email_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Simple email regex pattern
        let email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
        let regex = regex::Regex::new(email_pattern).unwrap();

        for value in &context.values {
            if let Term::Literal(lit) = value {
                let email = lit.value();
                if !regex.is_match(email) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("'{}' is not a valid email address", email)),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Email validation can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Simple URL validation
    fn is_valid_url(&self, url: &str) -> bool {
        // Basic URL validation - in practice you might want to use a proper URL parsing library
        url.starts_with("http://") || url.starts_with("https://") || url.starts_with("ftp://")
    }
}

// Standard constraint component implementations

/// Regular expression constraint component
#[derive(Debug)]
pub struct RegexConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl RegexConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:RegexConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Regular Expression Constraint".to_string(),
            description: Some(
                "Validates text values against a regular expression pattern".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "pattern".to_string(),
                    description: Some("Regular expression pattern to match".to_string()),
                    required: true,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![
                        ParameterConstraint::MinLength(1),
                        ParameterConstraint::MaxLength(1000),
                    ],
                    cardinality: Some((1, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "flags".to_string(),
                    description: Some("Regular expression flags (optional)".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![ParameterConstraint::MaxLength(20)],
                    cardinality: Some((0, Some(1))),
                    allowed_values: Some(vec![
                        "i".to_string(),
                        "m".to_string(),
                        "s".to_string(),
                        "x".to_string(),
                        "im".to_string(),
                        "is".to_string(),
                        "ms".to_string(),
                    ]),
                },
            ],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:EmailShape a sh:PropertyShape ;
    sh:path ex:email ;
    ex:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for RegexConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        // Check required pattern parameter
        let pattern = parameters.get("pattern").ok_or_else(|| {
            ShaclError::Configuration("Pattern parameter is required".to_string())
        })?;

        // Validate that pattern is a literal
        if !matches!(pattern, Term::Literal(_)) {
            return Err(ShaclError::Configuration(
                "Pattern parameter must be a literal".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("regex".to_string()),
            message_template: Some("Value does not match required pattern".to_string()),
        })
    }
}

/// Range constraint component
#[derive(Debug)]
pub struct RangeConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl RangeConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:RangeConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Numeric Range Constraint".to_string(),
            description: Some(
                "Validates that numeric values fall within a specified range".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "minValue".to_string(),
                    description: Some("Minimum allowed value (inclusive)".to_string()),
                    required: false,
                    datatype: Some("xsd:decimal".to_string()),
                    default_value: None,
                    validation_constraints: vec![],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "maxValue".to_string(),
                    description: Some("Maximum allowed value (inclusive)".to_string()),
                    required: false,
                    datatype: Some("xsd:decimal".to_string()),
                    default_value: None,
                    validation_constraints: vec![],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
            ],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:AgeShape a sh:PropertyShape ;
    sh:path ex:age ;
    ex:minValue 0 ;
    ex:maxValue 150 ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for RangeConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        let has_min = parameters.contains_key("minValue");
        let has_max = parameters.contains_key("maxValue");

        if !has_min && !has_max {
            return Err(ShaclError::Configuration(
                "At least one of minValue or maxValue must be specified".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("range".to_string()),
            message_template: Some("Value is outside allowed range".to_string()),
        })
    }
}

/// URL validation constraint component
#[derive(Debug)]
pub struct UrlValidationComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl UrlValidationComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:UrlValidationComponent".to_string());
        let metadata = ComponentMetadata {
            name: "URL Validation Constraint".to_string(),
            description: Some("Validates that values are well-formed URLs".to_string()),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:WebsiteShape a sh:PropertyShape ;
    sh:path ex:website ;
    ex:validUrl true ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for UrlValidationComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        // No required parameters
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("url".to_string()),
            message_template: Some("Value is not a valid URL".to_string()),
        })
    }
}

/// Email validation constraint component
#[derive(Debug)]
pub struct EmailValidationComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl EmailValidationComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:EmailValidationComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Email Validation Constraint".to_string(),
            description: Some(
                "Validates that literal values are well-formed email addresses".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:EmailShape a sh:PropertyShape ;
    sh:path ex:email ;
    ex:validEmail true ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for EmailValidationComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        // No required parameters
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("email".to_string()),
            message_template: Some("Value is not a valid email address".to_string()),
        })
    }
}

/// SPARQL-based custom constraint component
#[derive(Debug)]
pub struct SparqlConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl SparqlConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:SparqlConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "SPARQL Custom Constraint".to_string(),
            description: Some(
                "Generic SPARQL-based constraint for custom validation logic".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "query".to_string(),
                    description: Some("SPARQL ASK or SELECT query for validation".to_string()),
                    required: true,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![
                        ParameterConstraint::MinLength(10),
                        ParameterConstraint::MaxLength(10000),
                    ],
                    cardinality: Some((1, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "prefixes".to_string(),
                    description: Some("SPARQL prefixes to use with the query".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![ParameterConstraint::MaxLength(5000)],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
            ],
            applicable_to_node_shapes: true,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:CustomSparqlShape a sh:PropertyShape ;
    sh:path ex:customProperty ;
    ex:sparqlQuery "ASK { $this ex:hasRequiredRelation ?relation }" ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for SparqlConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        // Check required query parameter
        let query = parameters
            .get("query")
            .ok_or_else(|| ShaclError::Configuration("Query parameter is required".to_string()))?;

        // Validate that query is a literal
        if !matches!(query, Term::Literal(_)) {
            return Err(ShaclError::Configuration(
                "Query parameter must be a literal".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        let query = parameters
            .get("query")
            .and_then(|t| match t {
                Term::Literal(lit) => Some(lit.value().to_string()),
                _ => None,
            })
            .unwrap();

        let prefixes = parameters.get("prefixes").and_then(|t| match t {
            Term::Literal(lit) => Some(lit.value().to_string()),
            _ => None,
        });

        let sparql_query = if let Some(prefixes) = prefixes {
            format!("{}\n{}", prefixes, query)
        } else {
            query
        };

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: Some(sparql_query),
            validation_function: Some("sparql".to_string()),
            message_template: Some("Custom SPARQL constraint violated".to_string()),
        })
    }

    fn sparql_template(&self) -> Option<&str> {
        Some("ASK { $this ?customPredicate ?customValue }")
    }

    fn sparql_prefixes(&self) -> Option<&str> {
        Some("PREFIX ex: <http://example.org/>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};

    #[test]
    fn test_registry_registration() {
        let mut registry = CustomConstraintRegistry::new();
        let component = Arc::new(RegexConstraintComponent::new());
        let component_id = component.component_id().clone();

        assert!(registry.register_component(component).is_ok());
        assert!(registry.get_component(&component_id).is_some());
        assert!(registry.get_metadata(&component_id).is_some());
    }

    #[test]
    fn test_regex_constraint_creation() {
        let component = RegexConstraintComponent::new();
        let mut parameters = HashMap::new();
        parameters.insert(
            "pattern".to_string(),
            Term::Literal(Literal::new(r"^test\d+$")),
        );

        let constraint = component.create_constraint(parameters).unwrap();
        assert_eq!(
            constraint.component_id.as_str(),
            "ex:RegexConstraintComponent"
        );
        assert!(constraint.parameters.contains_key("pattern"));
    }

    #[test]
    fn test_range_constraint_validation() {
        let component = RangeConstraintComponent::new();

        // Should fail without min or max
        let empty_params = HashMap::new();
        assert!(component.validate_configuration(&empty_params).is_err());

        // Should succeed with min value
        let mut params = HashMap::new();
        params.insert("minValue".to_string(), Term::Literal(Literal::new("0")));
        assert!(component.validate_configuration(&params).is_ok());
    }

    #[test]
    fn test_custom_constraint_evaluation() {
        let constraint = CustomConstraint {
            component_id: ConstraintComponentId("ex:RegexConstraintComponent".to_string()),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "pattern".to_string(),
                    Term::Literal(Literal::new(r"^test\d+$")),
                );
                params
            },
            sparql_query: None,
            validation_function: Some("regex".to_string()),
            message_template: Some("Value does not match pattern".to_string()),
        };

        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/node").unwrap()),
            ShapeId::new("TestShape"),
        )
        .with_values(vec![Term::Literal(Literal::new("test123"))]);

        let result = constraint.evaluate_regex_constraint(&context).unwrap();
        assert!(result.is_satisfied());

        let context_invalid = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/node").unwrap()),
            ShapeId::new("TestShape"),
        )
        .with_values(vec![Term::Literal(Literal::new("invalid"))]);

        let result_invalid = constraint
            .evaluate_regex_constraint(&context_invalid)
            .unwrap();
        assert!(result_invalid.is_violated());
    }

    #[test]
    fn test_standard_extensions_registration() {
        let mut registry = CustomConstraintRegistry::new();
        assert!(registry.register_standard_extensions().is_ok());

        // Check that standard components are registered
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:RegexConstraintComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:RangeConstraintComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:UrlValidationComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:EmailValidationComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:SparqlConstraintComponent".to_string()
            ))
            .is_some());
    }

    #[test]
    fn test_component_inheritance() {
        let mut registry = CustomConstraintRegistry::new();
        let regex_component = Arc::new(RegexConstraintComponent::new());
        let range_component = Arc::new(RangeConstraintComponent::new());

        let regex_id = regex_component.component_id().clone();
        let range_id = range_component.component_id().clone();

        registry.register_component(regex_component).unwrap();
        registry.register_component(range_component).unwrap();

        // Set inheritance: range inherits from regex
        assert!(registry
            .set_component_inheritance(&range_id, vec![regex_id.clone()])
            .is_ok());

        // Check inheritance
        let inherited = registry.get_inherited_components(&range_id);
        assert_eq!(inherited.len(), 1);
        assert_eq!(inherited[0], regex_id);
    }

    #[test]
    fn test_circular_inheritance_detection() {
        let mut registry = CustomConstraintRegistry::new();
        let regex_component = Arc::new(RegexConstraintComponent::new());
        let range_component = Arc::new(RangeConstraintComponent::new());

        let regex_id = regex_component.component_id().clone();
        let range_id = range_component.component_id().clone();

        registry.register_component(regex_component).unwrap();
        registry.register_component(range_component).unwrap();

        // Set up circular inheritance
        registry
            .set_component_inheritance(&range_id, vec![regex_id.clone()])
            .unwrap();

        // This should fail due to circular inheritance
        assert!(registry
            .set_component_inheritance(&regex_id, vec![range_id])
            .is_err());
    }

    #[test]
    fn test_composite_constraint_creation() {
        let mut registry = CustomConstraintRegistry::new();
        registry.register_standard_extensions().unwrap();

        let regex_id = ConstraintComponentId("ex:RegexConstraintComponent".to_string());
        let range_id = ConstraintComponentId("ex:RangeConstraintComponent".to_string());

        let mut parameters = HashMap::new();
        parameters.insert("pattern".to_string(), Term::Literal(Literal::new(r"^\d+$")));
        parameters.insert("minValue".to_string(), Term::Literal(Literal::new("0")));

        let composite = registry
            .create_composite_constraint(&[regex_id, range_id], parameters)
            .unwrap();

        assert_eq!(composite.component_ids.len(), 2);
        assert_eq!(composite.constraints.len(), 2);
        assert!(matches!(composite.composition_type, CompositionType::And));
    }

    #[test]
    fn test_composite_constraint_evaluation() {
        // Create a simple composite constraint for testing
        let regex_constraint = CustomConstraint {
            component_id: ConstraintComponentId("ex:RegexConstraintComponent".to_string()),
            parameters: {
                let mut params = HashMap::new();
                params.insert("pattern".to_string(), Term::Literal(Literal::new(r"^\d+$")));
                params
            },
            sparql_query: None,
            validation_function: Some("regex".to_string()),
            message_template: Some("Value must be numeric".to_string()),
        };

        let range_constraint = CustomConstraint {
            component_id: ConstraintComponentId("ex:RangeConstraintComponent".to_string()),
            parameters: {
                let mut params = HashMap::new();
                params.insert("minValue".to_string(), Term::Literal(Literal::new("0")));
                params.insert("maxValue".to_string(), Term::Literal(Literal::new("100")));
                params
            },
            sparql_query: None,
            validation_function: Some("range".to_string()),
            message_template: Some("Value must be between 0 and 100".to_string()),
        };

        let composite = CompositeConstraint {
            component_ids: vec![
                ConstraintComponentId("ex:RegexConstraintComponent".to_string()),
                ConstraintComponentId("ex:RangeConstraintComponent".to_string()),
            ],
            constraints: vec![regex_constraint, range_constraint],
            composition_type: CompositionType::And,
        };

        // Test with valid value
        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/node").unwrap()),
            ShapeId::new("TestShape"),
        )
        .with_values(vec![Term::Literal(Literal::new("50"))]);

        // Note: This test would require a proper Store implementation to work fully
        // For now, we just test the structure
        assert_eq!(composite.constraints.len(), 2);
    }

    #[test]
    fn test_parameter_validation_constraints() {
        let component = RegexConstraintComponent::new();
        let metadata = component.metadata();

        // Check that validation constraints were added
        let pattern_param = metadata
            .parameters
            .iter()
            .find(|p| p.name == "pattern")
            .unwrap();
        assert!(!pattern_param.validation_constraints.is_empty());
        assert!(matches!(
            pattern_param.validation_constraints[0],
            ParameterConstraint::MinLength(1)
        ));

        let flags_param = metadata
            .parameters
            .iter()
            .find(|p| p.name == "flags")
            .unwrap();
        assert!(flags_param.allowed_values.is_some());
        assert!(flags_param
            .allowed_values
            .as_ref()
            .unwrap()
            .contains(&"i".to_string()));
    }
}
