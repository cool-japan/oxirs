//! GraphQL query validation and security features
//!
//! This module provides comprehensive validation for GraphQL queries, including
//! security features like depth limiting, complexity analysis, and schema validation.

use crate::ast::{
    Definition, Document, Field, OperationDefinition, Selection, SelectionSet, Value,
    VariableDefinition,
};
use crate::types::{GraphQLType, Schema};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Configuration for query validation and security
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum allowed query depth
    pub max_depth: usize,
    /// Maximum allowed query complexity score
    pub max_complexity: usize,
    /// Maximum number of aliases allowed
    pub max_aliases: usize,
    /// Maximum number of root fields
    pub max_root_fields: usize,
    /// Enable query timeout
    pub query_timeout: Option<Duration>,
    /// Disabled introspection queries
    pub disable_introspection: bool,
    /// Maximum number of fragments
    pub max_fragments: usize,
    /// Whitelist of allowed operation names
    pub allowed_operations: Option<HashSet<String>>,
    /// Blacklist of forbidden field names
    pub forbidden_fields: HashSet<String>,
    /// Enable cost analysis
    pub enable_cost_analysis: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_complexity: 1000,
            max_aliases: 50,
            max_root_fields: 20,
            query_timeout: Some(Duration::from_secs(30)),
            disable_introspection: false,
            max_fragments: 50,
            allowed_operations: None,
            forbidden_fields: HashSet::new(),
            enable_cost_analysis: true,
        }
    }
}

/// Result of query validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub complexity_score: usize,
    pub depth: usize,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            complexity_score: 0,
            depth: 0,
        }
    }

    pub fn with_error(mut self, error: ValidationError) -> Self {
        self.is_valid = false;
        self.errors.push(error);
        self
    }

    pub fn with_warning(mut self, warning: ValidationWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    pub fn with_complexity(mut self, complexity: usize) -> Self {
        self.complexity_score = complexity;
        self
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

/// Validation error details
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub path: Vec<String>,
    pub rule: ValidationRule,
}

impl ValidationError {
    pub fn new(message: String, rule: ValidationRule) -> Self {
        Self {
            message,
            path: Vec::new(),
            rule,
        }
    }

    pub fn with_path(mut self, path: Vec<String>) -> Self {
        self.path = path;
        self
    }
}

/// Validation warning details
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
    pub suggestion: Option<String>,
}

impl ValidationWarning {
    pub fn new(message: String) -> Self {
        Self {
            message,
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }
}

/// Types of validation rules
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationRule {
    MaxDepth,
    MaxComplexity,
    MaxAliases,
    MaxRootFields,
    MaxFragments,
    FieldValidation,
    TypeValidation,
    VariableValidation,
    FragmentValidation,
    IntrospectionDisabled,
    OperationNotAllowed,
    ForbiddenField,
    CostAnalysis,
}

/// Main query validator
pub struct QueryValidator {
    config: ValidationConfig,
    schema: Schema,
}

impl QueryValidator {
    pub fn new(config: ValidationConfig, schema: Schema) -> Self {
        Self { config, schema }
    }

    /// Validate a GraphQL document
    pub fn validate(&self, document: &Document) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();
        let mut validation_context = ValidationContext::new(&self.schema);

        // Collect fragments first
        for definition in &document.definitions {
            if let Definition::Fragment(fragment) = definition {
                validation_context.add_fragment(fragment.name.clone(), fragment.clone());
            }
        }

        // Validate each operation
        for definition in &document.definitions {
            if let Definition::Operation(operation) = definition {
                result = self.validate_operation(operation, &validation_context, result)?;
            }
        }

        // Global validations
        result = self.validate_fragments(&validation_context, result)?;
        result = self.validate_global_limits(document, result)?;

        Ok(result)
    }

    fn validate_operation(
        &self,
        operation: &OperationDefinition,
        context: &ValidationContext,
        mut result: ValidationResult,
    ) -> Result<ValidationResult> {
        // Check operation name whitelist
        if let Some(ref allowed_ops) = self.config.allowed_operations {
            if let Some(ref op_name) = operation.name {
                if !allowed_ops.contains(op_name) {
                    return Ok(result.with_error(ValidationError::new(
                        format!("Operation '{}' is not allowed", op_name),
                        ValidationRule::OperationNotAllowed,
                    )));
                }
            }
        }

        // Validate variables
        result = self.validate_variables(&operation.variable_definitions, context, result)?;

        // Get root type based on operation type
        let root_type_name = match operation.operation_type {
            crate::ast::OperationType::Query => self
                .schema
                .query_type
                .as_ref()
                .ok_or_else(|| anyhow!("Schema has no query type"))?,
            crate::ast::OperationType::Mutation => self
                .schema
                .mutation_type
                .as_ref()
                .ok_or_else(|| anyhow!("Schema has no mutation type"))?,
            crate::ast::OperationType::Subscription => self
                .schema
                .subscription_type
                .as_ref()
                .ok_or_else(|| anyhow!("Schema has no subscription type"))?,
        };

        // Validate selection set
        let (depth, complexity) = self.validate_selection_set(
            &operation.selection_set,
            root_type_name,
            context,
            0,
            Vec::new(),
        )?;

        result.depth = depth.max(result.depth);
        result.complexity_score += complexity;

        // Check depth limit
        if result.depth > self.config.max_depth {
            let current_depth = result.depth;
            result = result.with_error(ValidationError::new(
                format!(
                    "Query depth {} exceeds maximum allowed depth {}",
                    current_depth, self.config.max_depth
                ),
                ValidationRule::MaxDepth,
            ));
        }

        // Check complexity limit
        if result.complexity_score > self.config.max_complexity {
            let current_complexity = result.complexity_score;
            result = result.with_error(ValidationError::new(
                format!(
                    "Query complexity {} exceeds maximum allowed complexity {}",
                    current_complexity, self.config.max_complexity
                ),
                ValidationRule::MaxComplexity,
            ));
        }

        Ok(result)
    }

    fn validate_selection_set(
        &self,
        selection_set: &SelectionSet,
        parent_type_name: &str,
        context: &ValidationContext,
        current_depth: usize,
        path: Vec<String>,
    ) -> Result<(usize, usize)> {
        let mut max_depth = current_depth;
        let mut total_complexity = 0;
        let mut alias_count = 0;

        let parent_type = self
            .schema
            .get_type(parent_type_name)
            .ok_or_else(|| anyhow!("Type '{}' not found in schema", parent_type_name))?;

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    // Check for aliases
                    if field.alias.is_some() {
                        alias_count += 1;
                    }

                    // Check forbidden fields
                    if self.config.forbidden_fields.contains(&field.name) {
                        return Err(anyhow!("Field '{}' is forbidden", field.name));
                    }

                    // Check introspection fields
                    if self.config.disable_introspection && field.name.starts_with("__") {
                        return Err(anyhow!("Introspection is disabled"));
                    }

                    // Validate field exists on type
                    let field_type = self.get_field_type(parent_type, &field.name)?;

                    let mut field_path = path.clone();
                    field_path.push(field.alias.as_ref().unwrap_or(&field.name).clone());

                    // Calculate field complexity
                    let field_complexity = self.calculate_field_complexity(field);
                    total_complexity += field_complexity;

                    // Recurse into nested selection sets
                    if let Some(ref nested_selection_set) = field.selection_set {
                        let inner_type_name = self.get_inner_type_name(field_type);
                        let (nested_depth, nested_complexity) = self.validate_selection_set(
                            nested_selection_set,
                            &inner_type_name,
                            context,
                            current_depth + 1,
                            field_path,
                        )?;
                        max_depth = max_depth.max(nested_depth);
                        total_complexity += nested_complexity;
                    }
                }
                Selection::InlineFragment(inline_fragment) => {
                    let fragment_type =
                        if let Some(ref type_condition) = inline_fragment.type_condition {
                            type_condition
                        } else {
                            parent_type_name
                        };

                    let (nested_depth, nested_complexity) = self.validate_selection_set(
                        &inline_fragment.selection_set,
                        fragment_type,
                        context,
                        current_depth,
                        path.clone(),
                    )?;
                    max_depth = max_depth.max(nested_depth);
                    total_complexity += nested_complexity;
                }
                Selection::FragmentSpread(fragment_spread) => {
                    if let Some(fragment_def) = context.get_fragment(&fragment_spread.fragment_name)
                    {
                        let (nested_depth, nested_complexity) = self.validate_selection_set(
                            &fragment_def.selection_set,
                            &fragment_def.type_condition,
                            context,
                            current_depth,
                            path.clone(),
                        )?;
                        max_depth = max_depth.max(nested_depth);
                        total_complexity += nested_complexity;
                    } else {
                        return Err(anyhow!(
                            "Fragment '{}' not found",
                            fragment_spread.fragment_name
                        ));
                    }
                }
            }
        }

        // Check alias limit
        if alias_count > self.config.max_aliases {
            return Err(anyhow!(
                "Too many aliases: {} exceeds limit {}",
                alias_count,
                self.config.max_aliases
            ));
        }

        Ok((max_depth, total_complexity))
    }

    fn validate_variables(
        &self,
        variable_definitions: &[VariableDefinition],
        _context: &ValidationContext,
        mut result: ValidationResult,
    ) -> Result<ValidationResult> {
        for var_def in variable_definitions {
            // Validate variable type exists in schema
            if !self.type_exists(&var_def.type_) {
                result = result.with_error(ValidationError::new(
                    format!(
                        "Variable type '{}' not found in schema",
                        var_def.type_.name()
                    ),
                    ValidationRule::VariableValidation,
                ));
            }

            // Validate default value compatibility
            if let Some(ref default_value) = var_def.default_value {
                if !self.is_value_compatible_with_type(default_value, &var_def.type_) {
                    result = result.with_error(ValidationError::new(
                        format!(
                            "Default value for variable '{}' is not compatible with type '{}'",
                            var_def.variable.name,
                            var_def.type_.name()
                        ),
                        ValidationRule::VariableValidation,
                    ));
                }
            }
        }

        Ok(result)
    }

    fn validate_fragments(
        &self,
        context: &ValidationContext,
        mut result: ValidationResult,
    ) -> Result<ValidationResult> {
        if context.fragments.len() > self.config.max_fragments {
            result = result.with_error(ValidationError::new(
                format!(
                    "Too many fragments: {} exceeds limit {}",
                    context.fragments.len(),
                    self.config.max_fragments
                ),
                ValidationRule::MaxFragments,
            ));
        }

        // Validate fragment type conditions
        for (fragment_name, fragment) in &context.fragments {
            if !self.schema.types.contains_key(&fragment.type_condition) {
                result = result.with_error(ValidationError::new(
                    format!(
                        "Fragment '{}' has unknown type condition '{}'",
                        fragment_name, fragment.type_condition
                    ),
                    ValidationRule::FragmentValidation,
                ));
            }
        }

        Ok(result)
    }

    fn validate_global_limits(
        &self,
        document: &Document,
        mut result: ValidationResult,
    ) -> Result<ValidationResult> {
        let mut root_field_count = 0;

        for definition in &document.definitions {
            if let Definition::Operation(operation) = definition {
                root_field_count += operation.selection_set.selections.len();
            }
        }

        if root_field_count > self.config.max_root_fields {
            result = result.with_error(ValidationError::new(
                format!(
                    "Too many root fields: {} exceeds limit {}",
                    root_field_count, self.config.max_root_fields
                ),
                ValidationRule::MaxRootFields,
            ));
        }

        Ok(result)
    }

    fn get_field_type<'a>(
        &self,
        parent_type: &'a GraphQLType,
        field_name: &str,
    ) -> Result<&'a GraphQLType> {
        match parent_type {
            GraphQLType::Object(obj) => obj
                .fields
                .get(field_name)
                .map(|field| &field.field_type)
                .ok_or_else(|| {
                    anyhow!(
                        "Field '{}' not found on object type '{}'",
                        field_name,
                        obj.name
                    )
                }),
            GraphQLType::Interface(iface) => iface
                .fields
                .get(field_name)
                .map(|field| &field.field_type)
                .ok_or_else(|| {
                    anyhow!(
                        "Field '{}' not found on interface type '{}'",
                        field_name,
                        iface.name
                    )
                }),
            _ => Err(anyhow!(
                "Cannot select field '{}' on non-composite type",
                field_name
            )),
        }
    }

    fn get_inner_type_name(&self, graphql_type: &GraphQLType) -> String {
        match graphql_type {
            GraphQLType::NonNull(inner) => self.get_inner_type_name(inner),
            GraphQLType::List(inner) => self.get_inner_type_name(inner),
            _ => graphql_type.name().to_string(),
        }
    }

    fn calculate_field_complexity(&self, field: &Field) -> usize {
        if !self.config.enable_cost_analysis {
            return 1;
        }

        let mut complexity = 1;

        // Add complexity for arguments
        complexity += field.arguments.len();

        // Add complexity for nested selections
        if let Some(ref selection_set) = field.selection_set {
            complexity += selection_set.selections.len();
        }

        // Special cases for expensive operations
        match field.name.as_str() {
            "sparql" => complexity *= 10, // Raw SPARQL queries are expensive
            name if name.contains("search") => complexity *= 3,
            name if name.contains("aggregate") => complexity *= 5,
            _ => {}
        }

        complexity
    }

    fn type_exists(&self, ast_type: &crate::ast::Type) -> bool {
        match ast_type {
            crate::ast::Type::NamedType(name) => {
                self.schema.types.contains_key(name)
                    || matches!(name.as_str(), "String" | "Int" | "Float" | "Boolean" | "ID")
            }
            crate::ast::Type::ListType(inner) => self.type_exists(inner),
            crate::ast::Type::NonNullType(inner) => self.type_exists(inner),
        }
    }

    fn is_value_compatible_with_type(&self, value: &Value, ast_type: &crate::ast::Type) -> bool {
        match (value, ast_type) {
            (Value::NullValue, crate::ast::Type::NonNullType(_)) => false,
            (Value::NullValue, _) => true,
            (Value::StringValue(_), crate::ast::Type::NamedType(name)) => {
                matches!(name.as_str(), "String" | "ID")
            }
            (Value::IntValue(_), crate::ast::Type::NamedType(name)) => {
                matches!(name.as_str(), "Int" | "ID")
            }
            (Value::FloatValue(_), crate::ast::Type::NamedType(name)) => {
                matches!(name.as_str(), "Float")
            }
            (Value::BooleanValue(_), crate::ast::Type::NamedType(name)) => {
                matches!(name.as_str(), "Boolean")
            }
            (Value::ListValue(list), crate::ast::Type::ListType(inner_type)) => list
                .iter()
                .all(|item| self.is_value_compatible_with_type(item, inner_type)),
            (_, crate::ast::Type::NonNullType(inner)) => {
                self.is_value_compatible_with_type(value, inner)
            }
            _ => false,
        }
    }
}

/// Context for validation operations
struct ValidationContext {
    schema: Schema,
    fragments: HashMap<String, crate::ast::FragmentDefinition>,
}

impl ValidationContext {
    fn new(schema: &Schema) -> Self {
        Self {
            schema: schema.clone(),
            fragments: HashMap::new(),
        }
    }

    fn add_fragment(&mut self, name: String, fragment: crate::ast::FragmentDefinition) {
        self.fragments.insert(name, fragment);
    }

    fn get_fragment(&self, name: &str) -> Option<&crate::ast::FragmentDefinition> {
        self.fragments.get(name)
    }
}

/// Rate limiting for query validation
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum queries per minute per client
    pub max_queries_per_minute: usize,
    /// Maximum complexity per minute per client
    pub max_complexity_per_minute: usize,
    /// Time window for rate limiting
    pub window_duration: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_queries_per_minute: 60,
            max_complexity_per_minute: 10000,
            window_duration: Duration::from_secs(60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BuiltinScalars, FieldType, ObjectType};

    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();

        let query_type = ObjectType::new("Query".to_string())
            .with_field(
                "hello".to_string(),
                FieldType::new(
                    "hello".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            )
            .with_field(
                "__schema".to_string(),
                FieldType::new(
                    "__schema".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            );

        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        schema
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.max_complexity, 1000);
        assert!(!config.disable_introspection);
    }

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::new()
            .with_error(ValidationError::new(
                "Test error".to_string(),
                ValidationRule::MaxDepth,
            ))
            .with_warning(ValidationWarning::new("Test warning".to_string()))
            .with_complexity(100)
            .with_depth(5);

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.complexity_score, 100);
        assert_eq!(result.depth, 5);
    }

    #[test]
    fn test_query_validator_creation() {
        let config = ValidationConfig::default();
        let schema = create_test_schema();
        let validator = QueryValidator::new(config, schema);

        // Validator should be created successfully
        assert_eq!(validator.config.max_depth, 10);
    }

    #[test]
    fn test_validation_error_with_path() {
        let error = ValidationError::new("Test error".to_string(), ValidationRule::FieldValidation)
            .with_path(vec![
                "query".to_string(),
                "user".to_string(),
                "name".to_string(),
            ]);

        assert_eq!(error.message, "Test error");
        assert_eq!(error.path, vec!["query", "user", "name"]);
        assert_eq!(error.rule, ValidationRule::FieldValidation);
    }

    #[test]
    fn test_validation_warning_with_suggestion() {
        let warning = ValidationWarning::new("Performance warning".to_string())
            .with_suggestion("Consider using pagination".to_string());

        assert_eq!(warning.message, "Performance warning");
        assert_eq!(
            warning.suggestion,
            Some("Consider using pagination".to_string())
        );
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_queries_per_minute, 60);
        assert_eq!(config.max_complexity_per_minute, 10000);
        assert_eq!(config.window_duration, Duration::from_secs(60));
    }

    #[test]
    fn test_validation_context() {
        let schema = create_test_schema();
        let mut context = ValidationContext::new(&schema);

        let fragment = crate::ast::FragmentDefinition {
            name: "TestFragment".to_string(),
            type_condition: "Query".to_string(),
            selection_set: crate::ast::SelectionSet { selections: vec![] },
            directives: vec![],
        };

        context.add_fragment("TestFragment".to_string(), fragment.clone());

        let retrieved = context.get_fragment("TestFragment");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "TestFragment");
    }
}
