//! SHACL Advanced Features - Parameterized Constraints
//!
//! Support for parameterized constraint components that can be reused
//! with different parameter values.
//!
//! Allows defining constraint templates that can be instantiated with
//! specific parameters for different use cases.

use crate::{PropertyPath, Result, ShaclError};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameter definition for constraint components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintParameter {
    /// Parameter name (IRI)
    pub name: String,
    /// Parameter description
    pub description: Option<String>,
    /// Whether this parameter is required
    pub required: bool,
    /// Default value if parameter is optional
    pub default_value: Option<ParameterValue>,
    /// Parameter type constraints
    pub param_type: ParameterTypeConstraint,
    /// Allowed values (if restricted)
    pub allowed_values: Option<Vec<ParameterValue>>,
}

impl ConstraintParameter {
    /// Create a new required parameter
    pub fn required(name: impl Into<String>, param_type: ParameterTypeConstraint) -> Self {
        Self {
            name: name.into(),
            description: None,
            required: true,
            default_value: None,
            param_type,
            allowed_values: None,
        }
    }

    /// Create a new optional parameter
    pub fn optional(
        name: impl Into<String>,
        param_type: ParameterTypeConstraint,
        default_value: ParameterValue,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            required: false,
            default_value: Some(default_value),
            param_type,
            allowed_values: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Restrict to specific allowed values
    pub fn with_allowed_values(mut self, values: Vec<ParameterValue>) -> Self {
        self.allowed_values = Some(values);
        self
    }

    /// Validate a parameter value
    pub fn validate_value(&self, value: &ParameterValue) -> Result<()> {
        // Check type constraint
        if !self.param_type.accepts(value) {
            return Err(ShaclError::ConstraintValidation(format!(
                "Parameter '{}' has invalid type: expected {:?}, got {:?}",
                self.name, self.param_type, value
            )));
        }

        // Check allowed values
        if let Some(ref allowed) = self.allowed_values {
            if !allowed.contains(value) {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Parameter '{}' has invalid value: not in allowed set",
                    self.name
                )));
            }
        }

        Ok(())
    }
}

/// Type constraint for parameters
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterTypeConstraint {
    /// Any RDF term
    Any,
    /// IRI/URI
    Iri,
    /// Literal value
    Literal,
    /// Integer literal
    Integer,
    /// Decimal/float literal
    Decimal,
    /// Boolean literal
    Boolean,
    /// String literal
    String,
    /// Property path
    Path,
    /// List of values
    List(Box<ParameterTypeConstraint>),
    /// One of multiple types
    Union(Vec<ParameterTypeConstraint>),
}

impl ParameterTypeConstraint {
    /// Check if this type constraint accepts a value
    pub fn accepts(&self, value: &ParameterValue) -> bool {
        match (self, value) {
            (ParameterTypeConstraint::Any, _) => true,
            (ParameterTypeConstraint::Iri, ParameterValue::Iri(_)) => true,
            (ParameterTypeConstraint::Literal, ParameterValue::Literal(_)) => true,
            (ParameterTypeConstraint::Integer, ParameterValue::Integer(_)) => true,
            (ParameterTypeConstraint::Decimal, ParameterValue::Decimal(_)) => true,
            (ParameterTypeConstraint::Boolean, ParameterValue::Boolean(_)) => true,
            (ParameterTypeConstraint::String, ParameterValue::String(_)) => true,
            (ParameterTypeConstraint::Path, ParameterValue::Path(_)) => true,
            (ParameterTypeConstraint::List(inner), ParameterValue::List(values)) => {
                values.iter().all(|v| inner.accepts(v))
            }
            (ParameterTypeConstraint::Union(types), value) => {
                types.iter().any(|t| t.accepts(value))
            }
            _ => false,
        }
    }
}

/// Parameter value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    /// IRI value
    Iri(String),
    /// Literal value
    Literal(String),
    /// Integer value
    Integer(i64),
    /// Decimal value
    Decimal(f64),
    /// Boolean value
    Boolean(bool),
    /// String value
    String(String),
    /// Property path value
    Path(PropertyPath),
    /// List of values
    List(Vec<ParameterValue>),
    /// RDF term
    Term(Term),
}

impl ParameterValue {
    /// Convert to RDF term if possible
    pub fn to_term(&self) -> Option<Term> {
        match self {
            ParameterValue::Term(t) => Some(t.clone()),
            // TODO: Convert other types to terms
            _ => None,
        }
    }
}

/// Parameterized constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterizedConstraintComponent {
    /// Component identifier
    pub id: String,
    /// Component name
    pub name: String,
    /// Component description
    pub description: Option<String>,
    /// Parameter definitions
    pub parameters: Vec<ConstraintParameter>,
    /// Constraint implementation
    pub implementation: ConstraintImplementation,
}

/// Constraint implementation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintImplementation {
    /// SPARQL-based constraint using ASK query
    SparqlAsk {
        /// SPARQL ASK query template with placeholders
        query_template: String,
    },
    /// SPARQL-based constraint using SELECT query
    SparqlSelect {
        /// SPARQL SELECT query template
        query_template: String,
        /// Variable name for results
        result_variable: String,
    },
    /// JavaScript/WASM-based constraint
    Script {
        /// Script source code
        source: String,
        /// Script language
        language: ScriptLanguage,
    },
    /// Built-in constraint validator
    BuiltIn {
        /// Built-in validator name
        validator_name: String,
    },
}

/// Script language for custom validators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScriptLanguage {
    JavaScript,
    Wasm,
    Python,
}

impl ParameterizedConstraintComponent {
    /// Create a new parameterized constraint component
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        parameters: Vec<ConstraintParameter>,
        implementation: ConstraintImplementation,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            parameters,
            implementation,
        }
    }

    /// Add description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Instantiate this constraint with parameter values
    pub fn instantiate(
        &self,
        parameter_values: HashMap<String, ParameterValue>,
    ) -> Result<ConstraintInstance> {
        // Validate all parameters
        let mut resolved_params = HashMap::new();

        for param_def in &self.parameters {
            // Get parameter value
            let value = if let Some(value) = parameter_values.get(&param_def.name) {
                value.clone()
            } else if param_def.required {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Required parameter '{}' not provided",
                    param_def.name
                )));
            } else if let Some(ref default) = param_def.default_value {
                default.clone()
            } else {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Parameter '{}' has no value and no default",
                    param_def.name
                )));
            };

            // Validate value
            param_def.validate_value(&value)?;

            resolved_params.insert(param_def.name.clone(), value);
        }

        Ok(ConstraintInstance {
            component: self.clone(),
            parameters: resolved_params,
        })
    }
}

/// Instance of a parameterized constraint with specific parameter values
#[derive(Debug, Clone)]
pub struct ConstraintInstance {
    /// The constraint component definition
    pub component: ParameterizedConstraintComponent,
    /// Resolved parameter values
    pub parameters: HashMap<String, ParameterValue>,
}

impl ConstraintInstance {
    /// Execute this constraint instance
    pub fn execute(
        &self,
        focus_node: &Term,
        value_nodes: &[Term],
        store: &dyn Store,
    ) -> Result<ConstraintExecutionResult> {
        match &self.component.implementation {
            ConstraintImplementation::SparqlAsk { query_template } => {
                self.execute_sparql_ask(query_template, focus_node, value_nodes, store)
            }
            ConstraintImplementation::SparqlSelect {
                query_template,
                result_variable,
            } => self.execute_sparql_select(
                query_template,
                result_variable,
                focus_node,
                value_nodes,
                store,
            ),
            ConstraintImplementation::Script { source, language } => {
                self.execute_script(source, language, focus_node, value_nodes, store)
            }
            ConstraintImplementation::BuiltIn { validator_name } => {
                self.execute_builtin(validator_name, focus_node, value_nodes, store)
            }
        }
    }

    /// Execute SPARQL ASK constraint
    fn execute_sparql_ask(
        &self,
        query_template: &str,
        _focus_node: &Term,
        _value_nodes: &[Term],
        _store: &dyn Store,
    ) -> Result<ConstraintExecutionResult> {
        // TODO: Implement SPARQL ASK execution with parameter substitution
        tracing::debug!("Executing SPARQL ASK constraint: {}", query_template);
        Ok(ConstraintExecutionResult::conforms())
    }

    /// Execute SPARQL SELECT constraint
    fn execute_sparql_select(
        &self,
        query_template: &str,
        _result_variable: &str,
        _focus_node: &Term,
        _value_nodes: &[Term],
        _store: &dyn Store,
    ) -> Result<ConstraintExecutionResult> {
        // TODO: Implement SPARQL SELECT execution with parameter substitution
        tracing::debug!("Executing SPARQL SELECT constraint: {}", query_template);
        Ok(ConstraintExecutionResult::conforms())
    }

    /// Execute script-based constraint
    fn execute_script(
        &self,
        source: &str,
        language: &ScriptLanguage,
        _focus_node: &Term,
        _value_nodes: &[Term],
        _store: &dyn Store,
    ) -> Result<ConstraintExecutionResult> {
        // TODO: Implement script execution
        tracing::debug!("Executing {:?} script constraint", language);
        Ok(ConstraintExecutionResult::conforms())
    }

    /// Execute built-in constraint
    fn execute_builtin(
        &self,
        validator_name: &str,
        _focus_node: &Term,
        _value_nodes: &[Term],
        _store: &dyn Store,
    ) -> Result<ConstraintExecutionResult> {
        // TODO: Implement built-in validator dispatch
        tracing::debug!("Executing built-in constraint: {}", validator_name);
        Ok(ConstraintExecutionResult::conforms())
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(name)
    }
}

/// Result of constraint execution
#[derive(Debug, Clone)]
pub struct ConstraintExecutionResult {
    /// Whether constraint was satisfied
    pub conforms: bool,
    /// Violation message if constraint failed
    pub message: Option<String>,
    /// Additional result data
    pub data: HashMap<String, ParameterValue>,
}

impl ConstraintExecutionResult {
    /// Create a conforming result
    pub fn conforms() -> Self {
        Self {
            conforms: true,
            message: None,
            data: HashMap::new(),
        }
    }

    /// Create a violation result
    pub fn violation(message: impl Into<String>) -> Self {
        Self {
            conforms: false,
            message: Some(message.into()),
            data: HashMap::new(),
        }
    }

    /// Add result data
    pub fn with_data(mut self, key: impl Into<String>, value: ParameterValue) -> Self {
        self.data.insert(key.into(), value);
        self
    }
}

/// Registry for parameterized constraint components
#[derive(Debug, Default)]
pub struct ParameterizedConstraintRegistry {
    /// Registered components
    components: HashMap<String, ParameterizedConstraintComponent>,
}

impl ParameterizedConstraintRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a constraint component
    pub fn register(&mut self, component: ParameterizedConstraintComponent) {
        self.components.insert(component.id.clone(), component);
    }

    /// Get a component by ID
    pub fn get(&self, id: &str) -> Option<&ParameterizedConstraintComponent> {
        self.components.get(id)
    }

    /// List all registered components
    pub fn list(&self) -> Vec<&ParameterizedConstraintComponent> {
        self.components.values().collect()
    }

    /// Instantiate a component with parameters
    pub fn instantiate(
        &self,
        component_id: &str,
        parameters: HashMap<String, ParameterValue>,
    ) -> Result<ConstraintInstance> {
        let component = self.get(component_id).ok_or_else(|| {
            ShaclError::ConstraintValidation(format!(
                "Constraint component not found: {}",
                component_id
            ))
        })?;

        component.instantiate(parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_creation() {
        let param = ConstraintParameter::required("minLength", ParameterTypeConstraint::Integer);
        assert!(param.required);
        assert_eq!(param.name, "minLength");
    }

    #[test]
    fn test_parameter_validation() {
        let param = ConstraintParameter::required("minLength", ParameterTypeConstraint::Integer);
        let value = ParameterValue::Integer(5);
        assert!(param.validate_value(&value).is_ok());

        let invalid_value = ParameterValue::String("not an integer".to_string());
        assert!(param.validate_value(&invalid_value).is_err());
    }

    #[test]
    fn test_constraint_component_creation() {
        let param = ConstraintParameter::required("minLength", ParameterTypeConstraint::Integer);
        let component = ParameterizedConstraintComponent::new(
            "test:MinLengthConstraint",
            "MinLength",
            vec![param],
            ConstraintImplementation::BuiltIn {
                validator_name: "minLength".to_string(),
            },
        );

        assert_eq!(component.id, "test:MinLengthConstraint");
        assert_eq!(component.parameters.len(), 1);
    }

    #[test]
    fn test_constraint_instantiation() {
        let param = ConstraintParameter::required("minLength", ParameterTypeConstraint::Integer);
        let component = ParameterizedConstraintComponent::new(
            "test:MinLengthConstraint",
            "MinLength",
            vec![param],
            ConstraintImplementation::BuiltIn {
                validator_name: "minLength".to_string(),
            },
        );

        let mut params = HashMap::new();
        params.insert("minLength".to_string(), ParameterValue::Integer(5));

        let instance = component.instantiate(params);
        assert!(instance.is_ok());
    }

    #[test]
    fn test_registry() {
        let mut registry = ParameterizedConstraintRegistry::new();
        let param = ConstraintParameter::required("minLength", ParameterTypeConstraint::Integer);
        let component = ParameterizedConstraintComponent::new(
            "test:MinLengthConstraint",
            "MinLength",
            vec![param],
            ConstraintImplementation::BuiltIn {
                validator_name: "minLength".to_string(),
            },
        );

        registry.register(component);
        assert!(registry.get("test:MinLengthConstraint").is_some());
    }
}
