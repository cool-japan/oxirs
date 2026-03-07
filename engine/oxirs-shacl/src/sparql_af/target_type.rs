//! `sh:SPARQLTargetType` — parameterised SPARQL target type templates
//!
//! A `sh:SPARQLTargetType` is a reusable target definition template that accepts
//! named parameters.  A shape can declare a target of that type and supply
//! parameter values, which are substituted into the SELECT query template
//! before execution.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#SPARQLTargetType>

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::sparql_target::SparqlTargetEvaluator;
use super::{AfResult, PrefixMap, SparqlAfError, SubstitutionContext};

/// Type metadata for a declared `sh:SPARQLTargetType` parameter
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlTargetParameter {
    /// Parameter name (used as `$paramName` in the query template)
    pub name: String,
    /// Optional short description
    pub description: Option<String>,
    /// Whether this parameter is optional (if true, missing values are silently ignored)
    pub optional: bool,
    /// Optional default value (SPARQL term string)
    pub default_value: Option<String>,
}

impl SparqlTargetParameter {
    /// Create a required parameter
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            optional: false,
            default_value: None,
        }
    }

    /// Create an optional parameter with a default value
    pub fn optional(name: impl Into<String>, default_value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            optional: true,
            default_value: Some(default_value.into()),
        }
    }

    /// Set a description (builder pattern)
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A `sh:SPARQLTargetType` definition
///
/// Encapsulates a reusable SPARQL SELECT query template with named parameters.
/// Instances of this type are created by supplying concrete parameter bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlTargetType {
    /// Unique IRI identifying this target type
    pub type_iri: String,
    /// Human-readable label
    pub label: Option<String>,
    /// The SPARQL SELECT query template (uses `$paramName` placeholders)
    pub select_template: String,
    /// Namespace prefixes to include with queries
    pub prefixes: PrefixMap,
    /// Declared parameters
    pub parameters: Vec<SparqlTargetParameter>,
}

impl SparqlTargetType {
    /// Create a new target type from a query template
    pub fn new(type_iri: impl Into<String>, select_template: impl Into<String>) -> Self {
        Self {
            type_iri: type_iri.into(),
            label: None,
            select_template: select_template.into(),
            prefixes: PrefixMap::new(),
            parameters: Vec::new(),
        }
    }

    /// Set a label (builder pattern)
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a namespace prefix (builder pattern)
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.0.insert(prefix.into(), iri.into());
        self
    }

    /// Declare a required parameter (builder pattern)
    pub fn require_param(mut self, name: impl Into<String>) -> Self {
        self.parameters.push(SparqlTargetParameter::required(name));
        self
    }

    /// Declare an optional parameter with default (builder pattern)
    pub fn optional_param(mut self, name: impl Into<String>, default: impl Into<String>) -> Self {
        self.parameters
            .push(SparqlTargetParameter::optional(name, default));
        self
    }

    /// Instantiate this target type with concrete parameter bindings
    pub fn instantiate(
        &self,
        bindings: HashMap<String, String>,
    ) -> AfResult<SparqlTargetTypeInstance> {
        // Validate and complete parameter bindings
        let mut resolved: HashMap<String, String> = HashMap::new();

        for param in &self.parameters {
            if let Some(value) = bindings.get(&param.name) {
                resolved.insert(param.name.clone(), value.clone());
            } else if param.optional {
                if let Some(default) = &param.default_value {
                    resolved.insert(param.name.clone(), default.clone());
                }
                // If optional and no default, the placeholder stays (may produce empty result)
            } else {
                return Err(SparqlAfError::MissingParameter(param.name.clone()));
            }
        }

        Ok(SparqlTargetTypeInstance {
            target_type_iri: self.type_iri.clone(),
            select_template: self.select_template.clone(),
            prefixes: self.prefixes.clone(),
            resolved_bindings: resolved,
        })
    }
}

/// An instantiated `sh:SPARQLTargetType` with all parameters resolved
#[derive(Debug, Clone)]
pub struct SparqlTargetTypeInstance {
    /// IRI of the target type this was instantiated from
    pub target_type_iri: String,
    /// Query template (still contains `$placeholders` until build)
    select_template: String,
    /// Namespace prefixes
    prefixes: PrefixMap,
    /// Resolved parameter bindings
    resolved_bindings: HashMap<String, String>,
}

impl SparqlTargetTypeInstance {
    /// Build the complete SPARQL query by substituting all parameters
    pub fn build_query(&self) -> String {
        let mut ctx = SubstitutionContext::new();
        for (name, value) in &self.resolved_bindings {
            ctx = ctx.bind(name, value);
        }

        let query_body = ctx.apply(&self.select_template);
        let prefix_block = self.prefixes.render_declarations();

        if !prefix_block.is_empty() {
            format!("{prefix_block}\n{query_body}")
        } else {
            query_body
        }
    }

    /// Evaluate this instance using the provided SPARQL evaluator.
    ///
    /// Returns the list of focus node term strings from `?this` bindings.
    pub fn evaluate(&self, evaluator: &dyn SparqlTargetEvaluator) -> AfResult<Vec<String>> {
        let query = self.build_query();
        let rows = evaluator.execute_select(&query)?;

        let nodes: Vec<String> = rows
            .into_iter()
            .filter_map(|row| row.get("this").cloned())
            .collect();

        Ok(nodes)
    }
}

/// Registry for `sh:SPARQLTargetType` definitions
///
/// Stores named target type templates and allows instantiating them by IRI.
#[derive(Debug, Default)]
pub struct SparqlTargetTypeRegistry {
    types: HashMap<String, SparqlTargetType>,
}

impl SparqlTargetTypeRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a target type definition
    pub fn register(&mut self, target_type: SparqlTargetType) {
        self.types.insert(target_type.type_iri.clone(), target_type);
    }

    /// Look up a target type by IRI
    pub fn get(&self, type_iri: &str) -> Option<&SparqlTargetType> {
        self.types.get(type_iri)
    }

    /// Instantiate a target type with specific parameter bindings
    pub fn instantiate(
        &self,
        type_iri: &str,
        bindings: HashMap<String, String>,
    ) -> AfResult<SparqlTargetTypeInstance> {
        let target_type = self
            .types
            .get(type_iri)
            .ok_or_else(|| SparqlAfError::TargetTypeNotFound(type_iri.to_string()))?;
        target_type.instantiate(bindings)
    }

    /// Returns the number of registered target types
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns `true` if the registry contains no target types
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Returns an iterator over all registered type IRIs
    pub fn type_iris(&self) -> impl Iterator<Item = &str> {
        self.types.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::super::sparql_target::{this_row, SparqlTargetMock};
    use super::*;

    const CLASS_TYPE_IRI: &str = "http://example.org/target/ClassInstances";

    fn class_instances_type() -> SparqlTargetType {
        SparqlTargetType::new(
            CLASS_TYPE_IRI,
            "SELECT ?this WHERE { ?this a $targetClass }",
        )
        .with_label("Class instances target type")
        .with_prefix("ex", "http://example.org/")
        .require_param("targetClass")
    }

    // ---- SparqlTargetParameter tests ----

    #[test]
    fn test_required_parameter() {
        let param = SparqlTargetParameter::required("class");
        assert_eq!(param.name, "class");
        assert!(!param.optional);
        assert!(param.default_value.is_none());
    }

    #[test]
    fn test_optional_parameter_with_default() {
        let param = SparqlTargetParameter::optional("limit", "100");
        assert!(param.optional);
        assert_eq!(param.default_value, Some("100".to_string()));
    }

    // ---- SparqlTargetType tests ----

    #[test]
    fn test_target_type_construction() {
        let tt = class_instances_type();
        assert_eq!(tt.type_iri, CLASS_TYPE_IRI);
        assert_eq!(tt.label, Some("Class instances target type".to_string()));
        assert_eq!(tt.parameters.len(), 1);
        assert!(tt.prefixes.0.contains_key("ex"));
    }

    #[test]
    fn test_instantiate_with_required_param() {
        let tt = class_instances_type();
        let mut bindings = HashMap::new();
        bindings.insert(
            "targetClass".to_string(),
            "<http://example.org/Person>".to_string(),
        );

        let instance = tt
            .instantiate(bindings)
            .expect("instantiation should succeed");
        assert_eq!(instance.target_type_iri, CLASS_TYPE_IRI);
    }

    #[test]
    fn test_instantiate_missing_required_param_fails() {
        let tt = class_instances_type();
        let bindings = HashMap::new(); // no bindings
        let result = tt.instantiate(bindings);
        assert!(result.is_err());
        match result {
            Err(SparqlAfError::MissingParameter(name)) => {
                assert_eq!(name, "targetClass");
            }
            _ => panic!("Expected MissingParameter error"),
        }
    }

    #[test]
    fn test_instantiate_optional_param_uses_default() {
        let tt = SparqlTargetType::new(
            "http://example.org/tt/LimitedTarget",
            "SELECT ?this WHERE { ?this a ex:Thing } LIMIT $limit",
        )
        .optional_param("limit", "50");

        let instance = tt
            .instantiate(HashMap::new())
            .expect("instantiation with defaults should succeed");

        let query = instance.build_query();
        assert!(query.contains("50")); // default used
    }

    // ---- SparqlTargetTypeInstance tests ----

    #[test]
    fn test_instance_build_query_substitutes_params() {
        let tt = class_instances_type();
        let mut bindings = HashMap::new();
        bindings.insert(
            "targetClass".to_string(),
            "<http://example.org/Employee>".to_string(),
        );

        let instance = tt.instantiate(bindings).expect("should succeed");
        let query = instance.build_query();

        assert!(query.contains("<http://example.org/Employee>"));
        assert!(!query.contains("$targetClass"));
        assert!(query.contains("PREFIX ex: <http://example.org/>"));
    }

    #[test]
    fn test_instance_evaluate() {
        let tt = class_instances_type();
        let mut bindings = HashMap::new();
        bindings.insert(
            "targetClass".to_string(),
            "<http://example.org/Person>".to_string(),
        );

        let instance = tt.instantiate(bindings).expect("should succeed");

        let evaluator = SparqlTargetMock::new().with_response(
            "Person",
            vec![
                this_row("http://example.org/Alice"),
                this_row("http://example.org/Bob"),
            ],
        );

        let nodes = instance
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&"http://example.org/Alice".to_string()));
        assert!(nodes.contains(&"http://example.org/Bob".to_string()));
    }

    // ---- SparqlTargetTypeRegistry tests ----

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = SparqlTargetTypeRegistry::new();
        registry.register(class_instances_type());

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.get(CLASS_TYPE_IRI).is_some());
    }

    #[test]
    fn test_registry_get_nonexistent_returns_none() {
        let registry = SparqlTargetTypeRegistry::new();
        assert!(registry.get("http://example.org/NonExistent").is_none());
    }

    #[test]
    fn test_registry_instantiate_unknown_type_fails() {
        let registry = SparqlTargetTypeRegistry::new();
        let result = registry.instantiate("http://example.org/Unknown", HashMap::new());
        assert!(result.is_err());
        match result {
            Err(SparqlAfError::TargetTypeNotFound(iri)) => {
                assert_eq!(iri, "http://example.org/Unknown");
            }
            _ => panic!("Expected TargetTypeNotFound error"),
        }
    }

    #[test]
    fn test_registry_instantiate_and_evaluate() {
        let mut registry = SparqlTargetTypeRegistry::new();
        registry.register(class_instances_type());

        let mut bindings = HashMap::new();
        bindings.insert(
            "targetClass".to_string(),
            "<http://example.org/Vehicle>".to_string(),
        );

        let instance = registry
            .instantiate(CLASS_TYPE_IRI, bindings)
            .expect("instantiation should succeed");

        let evaluator = SparqlTargetMock::new().with_response(
            "Vehicle",
            vec![
                this_row("http://example.org/Car1"),
                this_row("http://example.org/Truck2"),
            ],
        );

        let nodes = instance
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_registry_type_iris_iterator() {
        let mut registry = SparqlTargetTypeRegistry::new();
        registry.register(class_instances_type());
        registry.register(SparqlTargetType::new(
            "http://example.org/OtherType",
            "SELECT ?this WHERE { ?this ?p ?o }",
        ));

        let iris: Vec<&str> = registry.type_iris().collect();
        assert_eq!(iris.len(), 2);
    }
}
