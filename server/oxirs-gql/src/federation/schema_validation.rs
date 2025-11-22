//! Federation Schema Validation
//!
//! This module provides comprehensive validation for federated GraphQL schemas,
//! ensuring consistency, correctness, and adherence to Apollo Federation specifications.
//!
//! ## Features
//!
//! - **Schema Composition Validation**: Validates schema stitching and composition
//! - **Entity Validation**: Ensures @key directives and entity resolution
//! - **Field Conflict Detection**: Identifies conflicting field definitions
//! - **Type Compatibility**: Validates type consistency across subgraphs
//! - **Directive Validation**: Checks Federation-specific directives
//! - **Circular Reference Detection**: Prevents circular dependencies

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Validation severity level
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical error that prevents federation
    Error,
    /// Warning that should be addressed
    Warning,
    /// Informational notice
    Info,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity level
    pub severity: ValidationSeverity,
    /// Issue code (e.g., "FIELD_CONFLICT", "MISSING_KEY")
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Location in schema (subgraph name, type, field)
    pub location: ValidationLocation,
    /// Suggested fix (if any)
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    pub fn error(code: String, message: String, location: ValidationLocation) -> Self {
        Self {
            severity: ValidationSeverity::Error,
            code,
            message,
            location,
            suggestion: None,
        }
    }

    pub fn warning(code: String, message: String, location: ValidationLocation) -> Self {
        Self {
            severity: ValidationSeverity::Warning,
            code,
            message,
            location,
            suggestion: None,
        }
    }

    pub fn info(code: String, message: String, location: ValidationLocation) -> Self {
        Self {
            severity: ValidationSeverity::Info,
            code,
            message,
            location,
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }
}

/// Location of validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationLocation {
    /// Subgraph name
    pub subgraph: String,
    /// Type name
    pub type_name: Option<String>,
    /// Field name
    pub field_name: Option<String>,
    /// Directive name
    pub directive_name: Option<String>,
}

impl ValidationLocation {
    pub fn subgraph(name: String) -> Self {
        Self {
            subgraph: name,
            type_name: None,
            field_name: None,
            directive_name: None,
        }
    }

    pub fn type_location(subgraph: String, type_name: String) -> Self {
        Self {
            subgraph,
            type_name: Some(type_name),
            field_name: None,
            directive_name: None,
        }
    }

    pub fn field_location(subgraph: String, type_name: String, field_name: String) -> Self {
        Self {
            subgraph,
            type_name: Some(type_name),
            field_name: Some(field_name),
            directive_name: None,
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// All validation issues
    pub issues: Vec<ValidationIssue>,
    /// Whether validation passed (no errors)
    pub is_valid: bool,
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            is_valid: true,
            error_count: 0,
            warning_count: 0,
        }
    }

    pub fn add_issue(&mut self, issue: ValidationIssue) {
        match issue.severity {
            ValidationSeverity::Error => {
                self.error_count += 1;
                self.is_valid = false;
            }
            ValidationSeverity::Warning => {
                self.warning_count += 1;
            }
            ValidationSeverity::Info => {}
        }
        self.issues.push(issue);
    }

    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    pub fn has_warnings(&self) -> bool {
        self.warning_count > 0
    }

    pub fn merge(&mut self, other: ValidationResult) {
        for issue in other.issues {
            self.add_issue(issue);
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphQL type definition (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    pub name: String,
    pub kind: TypeKind,
    pub fields: Vec<FieldDefinition>,
    pub directives: Vec<DirectiveDefinition>,
    pub interfaces: Vec<String>,
}

/// Type kind
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum TypeKind {
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    Scalar,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub type_name: String,
    pub is_list: bool,
    pub is_non_null: bool,
    pub arguments: Vec<ArgumentDefinition>,
    pub directives: Vec<DirectiveDefinition>,
}

/// Argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentDefinition {
    pub name: String,
    pub type_name: String,
    pub is_non_null: bool,
    pub default_value: Option<String>,
}

/// Directive definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveDefinition {
    pub name: String,
    pub arguments: HashMap<String, String>,
}

/// Subgraph schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphSchema {
    pub name: String,
    pub types: HashMap<String, TypeDefinition>,
}

/// Federation schema validator
pub struct FederationSchemaValidator {
    /// Federation-specific directives
    federation_directives: HashSet<String>,
}

impl FederationSchemaValidator {
    pub fn new() -> Self {
        let mut federation_directives = HashSet::new();
        federation_directives.insert("key".to_string());
        federation_directives.insert("requires".to_string());
        federation_directives.insert("provides".to_string());
        federation_directives.insert("external".to_string());
        federation_directives.insert("extends".to_string());
        federation_directives.insert("shareable".to_string());
        federation_directives.insert("inaccessible".to_string());
        federation_directives.insert("override".to_string());
        federation_directives.insert("tag".to_string());

        Self {
            federation_directives,
        }
    }

    /// Validate multiple subgraph schemas
    pub fn validate(&self, subgraphs: &[SubgraphSchema]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate individual subgraphs
        for subgraph in subgraphs {
            result.merge(self.validate_subgraph(subgraph));
        }

        // Cross-subgraph validation
        result.merge(self.validate_entity_keys(subgraphs));
        result.merge(self.validate_field_conflicts(subgraphs));
        result.merge(self.validate_type_consistency(subgraphs));
        result.merge(self.validate_circular_references(subgraphs));

        result
    }

    /// Validate a single subgraph
    fn validate_subgraph(&self, subgraph: &SubgraphSchema) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check for reserved type names
        for type_name in subgraph.types.keys() {
            if self.is_reserved_type_name(type_name) {
                result.add_issue(
                    ValidationIssue::error(
                        "RESERVED_TYPE_NAME".to_string(),
                        format!("Type name '{}' is reserved", type_name),
                        ValidationLocation::type_location(subgraph.name.clone(), type_name.clone()),
                    )
                    .with_suggestion(
                        "Use a different type name that doesn't conflict with GraphQL built-ins"
                            .to_string(),
                    ),
                );
            }
        }

        // Validate each type
        for (type_name, type_def) in &subgraph.types {
            result.merge(self.validate_type(subgraph, type_name, type_def));
        }

        result
    }

    /// Validate a type definition
    fn validate_type(
        &self,
        subgraph: &SubgraphSchema,
        type_name: &str,
        type_def: &TypeDefinition,
    ) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate fields
        for field in &type_def.fields {
            result.merge(self.validate_field(subgraph, type_name, field));
        }

        // Validate directives
        for directive in &type_def.directives {
            if !self.is_valid_directive(&directive.name) {
                result.add_issue(ValidationIssue::warning(
                    "UNKNOWN_DIRECTIVE".to_string(),
                    format!("Unknown directive '@{}'", directive.name),
                    ValidationLocation::type_location(subgraph.name.clone(), type_name.to_string()),
                ));
            }
        }

        result
    }

    /// Validate a field definition
    fn validate_field(
        &self,
        subgraph: &SubgraphSchema,
        type_name: &str,
        field: &FieldDefinition,
    ) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check if field type exists (if not a built-in)
        if !self.is_builtin_type(&field.type_name) && !subgraph.types.contains_key(&field.type_name)
        {
            result.add_issue(
                ValidationIssue::error(
                    "UNKNOWN_TYPE".to_string(),
                    format!(
                        "Field '{}' references unknown type '{}'",
                        field.name, field.type_name
                    ),
                    ValidationLocation::field_location(
                        subgraph.name.clone(),
                        type_name.to_string(),
                        field.name.clone(),
                    ),
                )
                .with_suggestion(format!(
                    "Define type '{}' or import it from another subgraph",
                    field.type_name
                )),
            );
        }

        // Validate field directives
        for directive in &field.directives {
            if directive.name == "external" {
                // @external fields should not have resolvers in this subgraph
                result.add_issue(ValidationIssue::info(
                    "EXTERNAL_FIELD".to_string(),
                    format!("Field '{}' is marked as @external", field.name),
                    ValidationLocation::field_location(
                        subgraph.name.clone(),
                        type_name.to_string(),
                        field.name.clone(),
                    ),
                ));
            }
        }

        result
    }

    /// Validate entity @key directives
    fn validate_entity_keys(&self, subgraphs: &[SubgraphSchema]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Find all entities (types with @key directive)
        for subgraph in subgraphs {
            for (type_name, type_def) in &subgraph.types {
                let has_key = type_def.directives.iter().any(|d| d.name == "key");

                if has_key {
                    // Validate @key fields exist
                    for directive in &type_def.directives {
                        if directive.name == "key" {
                            if let Some(fields) = directive.arguments.get("fields") {
                                result.merge(
                                    self.validate_key_fields(subgraph, type_name, type_def, fields),
                                );
                            } else {
                                result.add_issue(ValidationIssue::error(
                                    "MISSING_KEY_FIELDS".to_string(),
                                    "@key directive must specify fields".to_string(),
                                    ValidationLocation::type_location(
                                        subgraph.name.clone(),
                                        type_name.clone(),
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Validate @key fields
    fn validate_key_fields(
        &self,
        subgraph: &SubgraphSchema,
        type_name: &str,
        type_def: &TypeDefinition,
        key_fields: &str,
    ) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Parse key fields (simplified - in production, use proper parsing)
        let fields: Vec<&str> = key_fields.split_whitespace().collect();

        for field_name in fields {
            // Check if field exists
            let field_exists = type_def.fields.iter().any(|f| f.name == field_name);

            if !field_exists {
                result.add_issue(
                    ValidationIssue::error(
                        "KEY_FIELD_NOT_FOUND".to_string(),
                        format!(
                            "@key references non-existent field '{}' on type '{}'",
                            field_name, type_name
                        ),
                        ValidationLocation::type_location(
                            subgraph.name.clone(),
                            type_name.to_string(),
                        ),
                    )
                    .with_suggestion(format!(
                        "Add field '{}' to type '{}'",
                        field_name, type_name
                    )),
                );
            }
        }

        result
    }

    /// Validate field conflicts across subgraphs
    fn validate_field_conflicts(&self, subgraphs: &[SubgraphSchema]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Build a map of types to their fields across subgraphs
        let mut type_fields: HashMap<String, Vec<(String, FieldDefinition)>> = HashMap::new();

        for subgraph in subgraphs {
            for (type_name, type_def) in &subgraph.types {
                for field in &type_def.fields {
                    type_fields
                        .entry(type_name.clone())
                        .or_default()
                        .push((subgraph.name.clone(), field.clone()));
                }
            }
        }

        // Check for conflicts
        for (type_name, fields) in type_fields {
            let mut field_map: HashMap<String, Vec<(String, FieldDefinition)>> = HashMap::new();

            for (subgraph_name, field) in fields {
                field_map
                    .entry(field.name.clone())
                    .or_default()
                    .push((subgraph_name, field));
            }

            for (field_name, definitions) in field_map {
                if definitions.len() > 1 {
                    // Check if definitions are compatible
                    let first = &definitions[0].1;

                    for (subgraph_name, def) in &definitions[1..] {
                        if !self.are_fields_compatible(first, def) {
                            result.add_issue(ValidationIssue::error(
                                "FIELD_CONFLICT".to_string(),
                                format!(
                                    "Field '{}' on type '{}' has conflicting definitions in subgraphs '{}' and '{}'",
                                    field_name, type_name, definitions[0].0, subgraph_name
                                ),
                                ValidationLocation::field_location(
                                    subgraph_name.clone(),
                                    type_name.clone(),
                                    field_name.clone(),
                                ),
                            ).with_suggestion("Use @shareable directive or make field types consistent".to_string()));
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if two field definitions are compatible
    fn are_fields_compatible(&self, field1: &FieldDefinition, field2: &FieldDefinition) -> bool {
        field1.type_name == field2.type_name
            && field1.is_list == field2.is_list
            && field1.is_non_null == field2.is_non_null
    }

    /// Validate type consistency
    fn validate_type_consistency(&self, subgraphs: &[SubgraphSchema]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check that extended types exist
        for subgraph in subgraphs {
            for (type_name, type_def) in &subgraph.types {
                let extends = type_def.directives.iter().any(|d| d.name == "extends");

                if extends {
                    // Check if type is defined in another subgraph
                    let defined_elsewhere = subgraphs
                        .iter()
                        .any(|s| s.name != subgraph.name && s.types.contains_key(type_name));

                    if !defined_elsewhere {
                        result.add_issue(ValidationIssue::error(
                            "EXTENDS_UNDEFINED_TYPE".to_string(),
                            format!(
                                "Type '{}' extends a type that is not defined in any subgraph",
                                type_name
                            ),
                            ValidationLocation::type_location(
                                subgraph.name.clone(),
                                type_name.clone(),
                            ),
                        ));
                    }
                }
            }
        }

        result
    }

    /// Validate circular references
    fn validate_circular_references(&self, subgraphs: &[SubgraphSchema]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Build dependency graph
        let mut dependencies: HashMap<String, HashSet<String>> = HashMap::new();

        for subgraph in subgraphs {
            for (type_name, type_def) in &subgraph.types {
                let mut deps = HashSet::new();

                for field in &type_def.fields {
                    deps.insert(field.type_name.clone());
                }

                dependencies.insert(type_name.clone(), deps);
            }
        }

        // Check for cycles using DFS
        for type_name in dependencies.keys() {
            if Self::has_circular_dependency(type_name, &dependencies, &mut HashSet::new()) {
                result.add_issue(
                    ValidationIssue::warning(
                        "CIRCULAR_REFERENCE".to_string(),
                        format!("Type '{}' has circular dependencies", type_name),
                        ValidationLocation::type_location("*".to_string(), type_name.clone()),
                    )
                    .with_suggestion(
                        "Circular references are allowed but may impact performance".to_string(),
                    ),
                );
            }
        }

        result
    }

    /// Check for circular dependency using DFS
    fn has_circular_dependency(
        type_name: &str,
        dependencies: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
    ) -> bool {
        if visited.contains(type_name) {
            return true;
        }

        visited.insert(type_name.to_string());

        if let Some(deps) = dependencies.get(type_name) {
            for dep in deps {
                if Self::has_circular_dependency(dep, dependencies, &mut visited.clone()) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if type name is reserved
    fn is_reserved_type_name(&self, name: &str) -> bool {
        matches!(
            name,
            "Query"
                | "Mutation"
                | "Subscription"
                | "__Schema"
                | "__Type"
                | "__Field"
                | "__InputValue"
                | "__EnumValue"
                | "__Directive"
        )
    }

    /// Check if type is a built-in GraphQL type
    fn is_builtin_type(&self, name: &str) -> bool {
        matches!(
            name,
            "String" | "Int" | "Float" | "Boolean" | "ID" | "DateTime"
        )
    }

    /// Check if directive is valid
    fn is_valid_directive(&self, name: &str) -> bool {
        self.federation_directives.contains(name)
            || matches!(name, "deprecated" | "skip" | "include" | "specifiedBy")
    }
}

impl Default for FederationSchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_type(name: &str) -> TypeDefinition {
        TypeDefinition {
            name: name.to_string(),
            kind: TypeKind::Object,
            fields: vec![],
            directives: vec![],
            interfaces: vec![],
        }
    }

    #[test]
    fn test_validation_issue_creation() {
        let location = ValidationLocation::subgraph("test_subgraph".to_string());

        let error = ValidationIssue::error(
            "TEST_ERROR".to_string(),
            "Test error message".to_string(),
            location.clone(),
        );

        assert_eq!(error.severity, ValidationSeverity::Error);
        assert_eq!(error.code, "TEST_ERROR");
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid);
        assert_eq!(result.error_count, 0);

        let error = ValidationIssue::error(
            "ERROR".to_string(),
            "Error".to_string(),
            ValidationLocation::subgraph("test".to_string()),
        );

        result.add_issue(error);
        assert!(!result.is_valid);
        assert_eq!(result.error_count, 1);
    }

    #[test]
    fn test_validator_creation() {
        let validator = FederationSchemaValidator::new();
        assert!(validator.federation_directives.contains("key"));
        assert!(validator.federation_directives.contains("external"));
    }

    #[test]
    fn test_is_reserved_type_name() {
        let validator = FederationSchemaValidator::new();

        assert!(validator.is_reserved_type_name("Query"));
        assert!(validator.is_reserved_type_name("Mutation"));
        assert!(!validator.is_reserved_type_name("User"));
    }

    #[test]
    fn test_is_builtin_type() {
        let validator = FederationSchemaValidator::new();

        assert!(validator.is_builtin_type("String"));
        assert!(validator.is_builtin_type("Int"));
        assert!(!validator.is_builtin_type("User"));
    }

    #[test]
    fn test_validate_empty_subgraphs() {
        let validator = FederationSchemaValidator::new();
        let result = validator.validate(&[]);

        assert!(result.is_valid);
        assert_eq!(result.issues.len(), 0);
    }

    #[test]
    fn test_validate_single_subgraph() {
        let validator = FederationSchemaValidator::new();

        let mut types = HashMap::new();
        types.insert("User".to_string(), create_test_type("User"));

        let subgraph = SubgraphSchema {
            name: "users".to_string(),
            types,
        };

        let result = validator.validate(&[subgraph]);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_reserved_type_name() {
        let validator = FederationSchemaValidator::new();

        let mut types = HashMap::new();
        types.insert("Query".to_string(), create_test_type("Query"));

        let subgraph = SubgraphSchema {
            name: "test".to_string(),
            types,
        };

        let result = validator.validate(&[subgraph]);
        assert!(!result.is_valid);
        assert_eq!(result.error_count, 1);
        assert_eq!(result.issues[0].code, "RESERVED_TYPE_NAME");
    }

    #[test]
    fn test_field_compatibility() {
        let validator = FederationSchemaValidator::new();

        let field1 = FieldDefinition {
            name: "id".to_string(),
            type_name: "ID".to_string(),
            is_list: false,
            is_non_null: true,
            arguments: vec![],
            directives: vec![],
        };

        let field2 = FieldDefinition {
            name: "id".to_string(),
            type_name: "ID".to_string(),
            is_list: false,
            is_non_null: true,
            arguments: vec![],
            directives: vec![],
        };

        assert!(validator.are_fields_compatible(&field1, &field2));

        let field3 = FieldDefinition {
            name: "id".to_string(),
            type_name: "String".to_string(),
            is_list: false,
            is_non_null: true,
            arguments: vec![],
            directives: vec![],
        };

        assert!(!validator.are_fields_compatible(&field1, &field3));
    }

    #[test]
    fn test_validation_severity() {
        assert_eq!(ValidationSeverity::Error, ValidationSeverity::Error);
        assert_ne!(ValidationSeverity::Error, ValidationSeverity::Warning);
    }

    #[test]
    fn test_type_kind_variants() {
        assert_eq!(TypeKind::Object, TypeKind::Object);
        assert_ne!(TypeKind::Object, TypeKind::Interface);
    }
}
