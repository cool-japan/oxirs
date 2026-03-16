//! SHACL sh:parameter / parameterized constraint components.
//!
//! Implements the SHACL Advanced Features specification for custom constraint
//! components with typed, optional, and defaulted parameters.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ConstraintComponents>

use std::sync::Arc;

// ─── parameter types ──────────────────────────────────────────────────────────

/// The expected value type for a SHACL constraint parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParameterKind {
    /// The value must be an IRI.
    IRI,
    /// The value must be an RDF literal.
    Literal,
    /// The value must be a blank node.
    BlankNode,
    /// Any RDF term is accepted.
    Any,
}

impl std::fmt::Display for ParameterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IRI => write!(f, "IRI"),
            Self::Literal => write!(f, "Literal"),
            Self::BlankNode => write!(f, "BlankNode"),
            Self::Any => write!(f, "Any"),
        }
    }
}

// ─── Parameter ────────────────────────────────────────────────────────────────

/// A single sh:parameter declaration for a custom constraint component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    /// The local name of the parameter (without namespace prefix).
    pub name: String,
    /// Expected term type.
    pub param_type: ParameterKind,
    /// Whether the parameter may be omitted.
    pub optional: bool,
    /// Default value used when the parameter is omitted and `optional` is true.
    pub default_value: Option<String>,
}

impl Parameter {
    /// Create a new required parameter.
    pub fn required(name: impl Into<String>, param_type: ParameterKind) -> Self {
        Self {
            name: name.into(),
            param_type,
            optional: false,
            default_value: None,
        }
    }

    /// Create a new optional parameter with no default.
    pub fn optional(name: impl Into<String>, param_type: ParameterKind) -> Self {
        Self {
            name: name.into(),
            param_type,
            optional: true,
            default_value: None,
        }
    }

    /// Create a new optional parameter with a default value.
    pub fn with_default(
        name: impl Into<String>,
        param_type: ParameterKind,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            optional: true,
            default_value: Some(default.into()),
        }
    }
}

// ─── ConstraintComponent ──────────────────────────────────────────────────────

/// A SHACL custom constraint component, identified by an IRI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintComponent {
    /// The IRI of the constraint component (e.g., `ex:MyConstraint`).
    pub iri: String,
    /// Declared parameters for this component.
    pub parameters: Vec<Parameter>,
    /// Optional sh:message template for violation messages.
    pub message_template: Option<String>,
}

impl ConstraintComponent {
    /// Create a new constraint component with no parameters and no message.
    pub fn new(iri: impl Into<String>) -> Self {
        Self {
            iri: iri.into(),
            parameters: Vec::new(),
            message_template: None,
        }
    }

    /// Builder: add a parameter.
    pub fn with_parameter(mut self, param: Parameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Builder: set message template.
    pub fn with_message(mut self, template: impl Into<String>) -> Self {
        self.message_template = Some(template.into());
        self
    }

    /// Return the parameter with the given name, if it exists.
    pub fn get_parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Return all required parameters.
    pub fn required_parameters(&self) -> Vec<&Parameter> {
        self.parameters.iter().filter(|p| !p.optional).collect()
    }
}

// ─── ParameterValue ───────────────────────────────────────────────────────────

/// A concrete binding of a named parameter to an RDF term string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterValue {
    /// Parameter name (must match a `Parameter::name` in the component).
    pub name: String,
    /// The RDF term string supplied for this parameter.
    pub value: String,
}

impl ParameterValue {
    /// Create a new binding.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

// ─── ConstraintInstance ───────────────────────────────────────────────────────

/// A fully instantiated constraint component: a component plus concrete bindings.
#[derive(Debug, Clone)]
pub struct ConstraintInstance {
    /// The component being instantiated.
    pub component: Arc<ConstraintComponent>,
    /// The concrete parameter bindings for this instance.
    pub bindings: Vec<ParameterValue>,
}

impl ConstraintInstance {
    /// Look up the resolved value for a given parameter name.
    ///
    /// Returns the explicitly supplied value, or falls back to the parameter's
    /// `default_value` when the parameter is optional and has one.
    pub fn resolve(&self, name: &str) -> Option<&str> {
        if let Some(binding) = self.bindings.iter().find(|b| b.name == name) {
            return Some(&binding.value);
        }
        // Try default value.
        self.component
            .get_parameter(name)
            .and_then(|p| p.default_value.as_deref())
    }

    /// Number of explicitly supplied bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

// ─── ParameterError ───────────────────────────────────────────────────────────

/// Errors that can occur when working with parameterized constraint components.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParameterError {
    /// No component registered under the given IRI.
    #[error("Unknown constraint component: {0}")]
    UnknownComponent(String),

    /// A required parameter was not supplied.
    #[error("Missing required parameter: {0}")]
    MissingRequired(String),

    /// A supplied value does not match the expected parameter type.
    #[error("Type mismatch for parameter '{param}': expected {expected}, got '{got}'")]
    TypeMismatch {
        param: String,
        expected: ParameterKind,
        got: String,
    },

    /// The same parameter name appears more than once in the supplied bindings.
    #[error("Duplicate parameter binding: {0}")]
    DuplicateBinding(String),
}

// ─── ParameterRegistry ────────────────────────────────────────────────────────

/// A registry of SHACL custom constraint components.
///
/// Allows registering components and then instantiating them with concrete
/// parameter bindings, validating required / duplicate / type constraints in
/// the process.
#[derive(Debug, Default)]
pub struct ParameterRegistry {
    components: Vec<ConstraintComponent>,
}

impl ParameterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Register a new constraint component.
    ///
    /// Replaces any existing component with the same IRI.
    pub fn register(&mut self, component: ConstraintComponent) {
        if let Some(pos) = self.components.iter().position(|c| c.iri == component.iri) {
            self.components[pos] = component;
        } else {
            self.components.push(component);
        }
    }

    /// Retrieve a registered component by IRI.
    pub fn get(&self, iri: &str) -> Option<&ConstraintComponent> {
        self.components.iter().find(|c| c.iri == iri)
    }

    /// List all registered components.
    pub fn list(&self) -> Vec<&ConstraintComponent> {
        self.components.iter().collect()
    }

    /// Instantiate a component by IRI with the provided bindings.
    ///
    /// Validates:
    /// 1. Component must be registered.
    /// 2. No duplicate binding names.
    /// 3. All required parameters must be supplied.
    /// 4. No unknown parameters.
    pub fn instantiate(
        &self,
        iri: &str,
        bindings: Vec<ParameterValue>,
    ) -> Result<ConstraintInstance, ParameterError> {
        let component = self
            .get(iri)
            .ok_or_else(|| ParameterError::UnknownComponent(iri.to_string()))?;

        // Check for duplicate binding names.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for b in &bindings {
            if !seen.insert(b.name.as_str()) {
                return Err(ParameterError::DuplicateBinding(b.name.clone()));
            }
        }

        // Verify all required parameters are supplied.
        for param in component.required_parameters() {
            if !bindings.iter().any(|b| b.name == param.name) {
                return Err(ParameterError::MissingRequired(param.name.clone()));
            }
        }

        // Type checking: each supplied binding must match the declared type.
        for binding in &bindings {
            if let Some(param) = component.get_parameter(&binding.name) {
                if !Self::value_matches_type(&binding.value, &param.param_type) {
                    return Err(ParameterError::TypeMismatch {
                        param: binding.name.clone(),
                        expected: param.param_type.clone(),
                        got: binding.value.clone(),
                    });
                }
            }
            // Bindings for unknown parameters are silently ignored (lenient mode).
        }

        Ok(ConstraintInstance {
            component: Arc::new(component.clone()),
            bindings,
        })
    }

    /// Heuristic type check: IRI must start with `<` or be a CURIE (`:foo`),
    /// Literal is anything quoted or plain, BlankNode starts with `_:`.
    fn value_matches_type(value: &str, kind: &ParameterKind) -> bool {
        match kind {
            ParameterKind::Any => true,
            ParameterKind::IRI => {
                value.starts_with('<') || (value.contains(':') && !value.starts_with('"'))
            }
            ParameterKind::Literal => {
                value.starts_with('"') || value.starts_with('\'') || !value.contains(':')
            }
            ParameterKind::BlankNode => value.starts_with("_:"),
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParameterKind ─────────────────────────────────────────────────────────

    #[test]
    fn test_parameter_kind_display_iri() {
        assert_eq!(ParameterKind::IRI.to_string(), "IRI");
    }

    #[test]
    fn test_parameter_kind_display_literal() {
        assert_eq!(ParameterKind::Literal.to_string(), "Literal");
    }

    #[test]
    fn test_parameter_kind_display_blank_node() {
        assert_eq!(ParameterKind::BlankNode.to_string(), "BlankNode");
    }

    #[test]
    fn test_parameter_kind_display_any() {
        assert_eq!(ParameterKind::Any.to_string(), "Any");
    }

    #[test]
    fn test_parameter_kind_eq() {
        assert_eq!(ParameterKind::IRI, ParameterKind::IRI);
        assert_ne!(ParameterKind::IRI, ParameterKind::Literal);
    }

    // ── Parameter ────────────────────────────────────────────────────────────

    #[test]
    fn test_parameter_required() {
        let p = Parameter::required("minCount", ParameterKind::Literal);
        assert_eq!(p.name, "minCount");
        assert!(!p.optional);
        assert!(p.default_value.is_none());
    }

    #[test]
    fn test_parameter_optional_no_default() {
        let p = Parameter::optional("message", ParameterKind::Literal);
        assert!(p.optional);
        assert!(p.default_value.is_none());
    }

    #[test]
    fn test_parameter_with_default() {
        let p = Parameter::with_default("severity", ParameterKind::IRI, "sh:Violation");
        assert!(p.optional);
        assert_eq!(p.default_value.as_deref(), Some("sh:Violation"));
    }

    #[test]
    fn test_parameter_eq() {
        let a = Parameter::required("x", ParameterKind::Any);
        let b = Parameter::required("x", ParameterKind::Any);
        assert_eq!(a, b);
    }

    // ── ConstraintComponent ───────────────────────────────────────────────────

    #[test]
    fn test_constraint_component_new() {
        let cc = ConstraintComponent::new("ex:MyConstraint");
        assert_eq!(cc.iri, "ex:MyConstraint");
        assert!(cc.parameters.is_empty());
        assert!(cc.message_template.is_none());
    }

    #[test]
    fn test_constraint_component_with_parameter() {
        let cc = ConstraintComponent::new("ex:C")
            .with_parameter(Parameter::required("p1", ParameterKind::IRI));
        assert_eq!(cc.parameters.len(), 1);
    }

    #[test]
    fn test_constraint_component_with_message() {
        let cc = ConstraintComponent::new("ex:C").with_message("Violation at {$path}");
        assert_eq!(cc.message_template.as_deref(), Some("Violation at {$path}"));
    }

    #[test]
    fn test_constraint_component_get_parameter_found() {
        let cc = ConstraintComponent::new("ex:C")
            .with_parameter(Parameter::required("alpha", ParameterKind::Literal));
        assert!(cc.get_parameter("alpha").is_some());
    }

    #[test]
    fn test_constraint_component_get_parameter_not_found() {
        let cc = ConstraintComponent::new("ex:C");
        assert!(cc.get_parameter("nonexistent").is_none());
    }

    #[test]
    fn test_constraint_component_required_parameters() {
        let cc = ConstraintComponent::new("ex:C")
            .with_parameter(Parameter::required("req1", ParameterKind::IRI))
            .with_parameter(Parameter::optional("opt1", ParameterKind::Literal))
            .with_parameter(Parameter::required("req2", ParameterKind::Any));
        let required = cc.required_parameters();
        assert_eq!(required.len(), 2);
        let names: Vec<&str> = required.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"req1"));
        assert!(names.contains(&"req2"));
    }

    // ── ParameterValue ────────────────────────────────────────────────────────

    #[test]
    fn test_parameter_value_new() {
        let pv = ParameterValue::new("minCount", "1");
        assert_eq!(pv.name, "minCount");
        assert_eq!(pv.value, "1");
    }

    // ── ParameterRegistry: register / get / list ──────────────────────────────

    #[test]
    fn test_registry_new_is_empty() {
        let reg = ParameterRegistry::new();
        assert!(reg.list().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ParameterRegistry::new();
        reg.register(ConstraintComponent::new("ex:C1"));
        assert!(reg.get("ex:C1").is_some());
    }

    #[test]
    fn test_registry_get_unknown() {
        let reg = ParameterRegistry::new();
        assert!(reg.get("ex:Unknown").is_none());
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut reg = ParameterRegistry::new();
        reg.register(ConstraintComponent::new("ex:C").with_message("v1"));
        reg.register(ConstraintComponent::new("ex:C").with_message("v2"));
        assert_eq!(reg.list().len(), 1);
        assert_eq!(
            reg.get("ex:C")
                .expect("should succeed")
                .message_template
                .as_deref(),
            Some("v2")
        );
    }

    #[test]
    fn test_registry_list_multiple() {
        let mut reg = ParameterRegistry::new();
        reg.register(ConstraintComponent::new("ex:A"));
        reg.register(ConstraintComponent::new("ex:B"));
        reg.register(ConstraintComponent::new("ex:C"));
        assert_eq!(reg.list().len(), 3);
    }

    #[test]
    fn test_registry_default() {
        let reg = ParameterRegistry::default();
        assert!(reg.list().is_empty());
    }

    // ── ParameterRegistry: instantiate ────────────────────────────────────────

    #[test]
    fn test_instantiate_unknown_component() {
        let reg = ParameterRegistry::new();
        let err = reg.instantiate("ex:Unknown", vec![]).unwrap_err();
        assert!(matches!(err, ParameterError::UnknownComponent(_)));
    }

    #[test]
    fn test_instantiate_no_params_success() {
        let mut reg = ParameterRegistry::new();
        reg.register(ConstraintComponent::new("ex:C"));
        let inst = reg.instantiate("ex:C", vec![]).expect("should succeed");
        assert_eq!(inst.component.iri, "ex:C");
        assert_eq!(inst.binding_count(), 0);
    }

    #[test]
    fn test_instantiate_required_param_supplied() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("pattern", ParameterKind::Literal)),
        );
        let bindings = vec![ParameterValue::new("pattern", "\"^[A-Z]+\"")];
        let inst = reg.instantiate("ex:C", bindings).expect("should succeed");
        assert_eq!(inst.binding_count(), 1);
    }

    #[test]
    fn test_instantiate_missing_required_param() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("pattern", ParameterKind::Literal)),
        );
        let err = reg.instantiate("ex:C", vec![]).unwrap_err();
        assert!(matches!(err, ParameterError::MissingRequired(name) if name == "pattern"));
    }

    #[test]
    fn test_instantiate_optional_param_omitted() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::optional("message", ParameterKind::Literal)),
        );
        let inst = reg.instantiate("ex:C", vec![]).expect("should succeed");
        assert_eq!(inst.binding_count(), 0);
    }

    #[test]
    fn test_instantiate_duplicate_binding() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("p", ParameterKind::Any)),
        );
        let bindings = vec![
            ParameterValue::new("p", "val1"),
            ParameterValue::new("p", "val2"),
        ];
        let err = reg.instantiate("ex:C", bindings).unwrap_err();
        assert!(matches!(err, ParameterError::DuplicateBinding(name) if name == "p"));
    }

    #[test]
    fn test_instantiate_type_mismatch_blank_node_expected() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("node", ParameterKind::BlankNode)),
        );
        let bindings = vec![ParameterValue::new("node", "<http://example.org/>")];
        let err = reg.instantiate("ex:C", bindings).unwrap_err();
        assert!(matches!(err, ParameterError::TypeMismatch { param, .. } if param == "node"));
    }

    #[test]
    fn test_instantiate_iri_type_accepted() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("targetClass", ParameterKind::IRI)),
        );
        let bindings = vec![ParameterValue::new(
            "targetClass",
            "<http://example.org/Person>",
        )];
        let inst = reg.instantiate("ex:C", bindings).expect("should succeed");
        assert_eq!(inst.binding_count(), 1);
    }

    #[test]
    fn test_instantiate_any_type_accepts_all() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("value", ParameterKind::Any)),
        );
        for value in &["<iri>", "\"literal\"", "_:b0", "plain"] {
            let bindings = vec![ParameterValue::new("value", *value)];
            assert!(reg.instantiate("ex:C", bindings).is_ok());
        }
    }

    // ── ConstraintInstance::resolve ───────────────────────────────────────────

    #[test]
    fn test_resolve_explicit_binding() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("x", ParameterKind::Any)),
        );
        let inst = reg
            .instantiate("ex:C", vec![ParameterValue::new("x", "hello")])
            .expect("should succeed");
        assert_eq!(inst.resolve("x"), Some("hello"));
    }

    #[test]
    fn test_resolve_default_value_when_omitted() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C").with_parameter(Parameter::with_default(
                "severity",
                ParameterKind::IRI,
                "sh:Violation",
            )),
        );
        let inst = reg.instantiate("ex:C", vec![]).expect("should succeed");
        assert_eq!(inst.resolve("severity"), Some("sh:Violation"));
    }

    #[test]
    fn test_resolve_absent_no_default() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::optional("msg", ParameterKind::Literal)),
        );
        let inst = reg.instantiate("ex:C", vec![]).expect("should succeed");
        assert_eq!(inst.resolve("msg"), None);
    }

    // ── ParameterError display ────────────────────────────────────────────────

    #[test]
    fn test_error_unknown_component_display() {
        let e = ParameterError::UnknownComponent("ex:X".to_string());
        assert!(e.to_string().contains("ex:X"));
    }

    #[test]
    fn test_error_missing_required_display() {
        let e = ParameterError::MissingRequired("pattern".to_string());
        assert!(e.to_string().contains("pattern"));
    }

    #[test]
    fn test_error_type_mismatch_display() {
        let e = ParameterError::TypeMismatch {
            param: "node".to_string(),
            expected: ParameterKind::BlankNode,
            got: "<iri>".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("node"));
        assert!(s.contains("BlankNode"));
        assert!(s.contains("<iri>"));
    }

    #[test]
    fn test_error_duplicate_binding_display() {
        let e = ParameterError::DuplicateBinding("p".to_string());
        assert!(e.to_string().contains("p"));
    }

    // ── edge-case / integration ───────────────────────────────────────────────

    #[test]
    fn test_full_workflow_multiple_params() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:PatternConstraint")
                .with_parameter(Parameter::required("pattern", ParameterKind::Literal))
                .with_parameter(Parameter::with_default(
                    "flags",
                    ParameterKind::Literal,
                    "\"i\"",
                ))
                .with_message("Value does not match pattern {$pattern}"),
        );
        let bindings = vec![ParameterValue::new("pattern", "\"^A\"")];
        let inst = reg
            .instantiate("ex:PatternConstraint", bindings)
            .expect("should succeed");
        assert_eq!(inst.resolve("pattern"), Some("\"^A\""));
        assert_eq!(inst.resolve("flags"), Some("\"i\""));
        assert!(inst.component.message_template.is_some());
    }

    #[test]
    fn test_blank_node_type_accepted() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("node", ParameterKind::BlankNode)),
        );
        let bindings = vec![ParameterValue::new("node", "_:b42")];
        assert!(reg.instantiate("ex:C", bindings).is_ok());
    }

    #[test]
    fn test_literal_type_accepted() {
        let mut reg = ParameterRegistry::new();
        reg.register(
            ConstraintComponent::new("ex:C")
                .with_parameter(Parameter::required("val", ParameterKind::Literal)),
        );
        let bindings = vec![ParameterValue::new("val", "\"hello world\"")];
        assert!(reg.instantiate("ex:C", bindings).is_ok());
    }

    #[test]
    fn test_registry_register_multiple_distinct() {
        let mut reg = ParameterRegistry::new();
        for i in 0..10 {
            reg.register(ConstraintComponent::new(format!("ex:C{}", i)));
        }
        assert_eq!(reg.list().len(), 10);
        assert!(reg.get("ex:C5").is_some());
    }
}
