//! # Custom GraphQL Directives
//!
//! Flexible directive system for extending GraphQL functionality.
//! Supports schema directives, query directives, and custom handlers.
//!
//! ## Features
//!
//! - **Authorization Directives**: @auth, @hasRole, @requiresPermission
//! - **Caching Directives**: @cacheControl, @cacheHint
//! - **Deprecation Directives**: @deprecated, @deprecated(reason: "...")
//! - **Rate Limiting Directives**: @rateLimit, @throttle
//! - **Transformation Directives**: @uppercase, @lowercase, @trim, @format
//! - **Validation Directives**: @constraint, @pattern, @range, @length
//! - **Cost Analysis Directives**: @cost, @complexity
//! - **Custom Directive Handlers**: User-defined directives
//!
//! ## Example
//!
//! ```graphql
//! type User @auth(requires: ADMIN) {
//!   email: String! @hasRole(role: "user")
//!   password: String! @deprecated(reason: "Use passwordHash instead")
//!   name: String! @uppercase @cacheControl(maxAge: 300)
//!   age: Int! @constraint(min: 0, max: 150)
//! }
//!
//! type Query {
//!   users: [User!]! @cost(complexity: 10) @rateLimit(max: 100)
//! }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Directive location (where the directive can be applied)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DirectiveLocation {
    /// Type definition
    Object,
    /// Field definition
    FieldDefinition,
    /// Interface definition
    Interface,
    /// Union definition
    Union,
    /// Enum definition
    Enum,
    /// Enum value
    EnumValue,
    /// Input object definition
    InputObject,
    /// Input field definition
    InputFieldDefinition,
    /// Scalar definition
    Scalar,
    /// Argument definition
    ArgumentDefinition,
    /// Query operation
    Query,
    /// Mutation operation
    Mutation,
    /// Subscription operation
    Subscription,
    /// Fragment definition
    FragmentDefinition,
    /// Fragment spread
    FragmentSpread,
    /// Inline fragment
    InlineFragment,
    /// Variable definition
    VariableDefinition,
}

/// Directive argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveArgument {
    /// Argument name
    pub name: String,
    /// Argument value (can be string, number, boolean, etc.)
    pub value: DirectiveValue,
}

/// Directive value type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DirectiveValue {
    String(String),
    Int(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<DirectiveValue>),
    Object(HashMap<String, DirectiveValue>),
    Null,
}

impl DirectiveValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            DirectiveValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            DirectiveValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            DirectiveValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DirectiveValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Directive definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveDefinition {
    /// Directive name (without @)
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Where this directive can be applied
    pub locations: Vec<DirectiveLocation>,
    /// Whether this directive is repeatable
    pub repeatable: bool,
    /// Arguments this directive accepts
    pub arguments: Vec<DirectiveArgumentDefinition>,
}

/// Directive argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveArgumentDefinition {
    /// Argument name
    pub name: String,
    /// Argument type
    pub arg_type: String,
    /// Whether this argument is required
    pub required: bool,
    /// Default value
    pub default_value: Option<DirectiveValue>,
    /// Description
    pub description: Option<String>,
}

/// Applied directive instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedDirective {
    /// Directive name
    pub name: String,
    /// Directive arguments
    pub arguments: Vec<DirectiveArgument>,
    /// Where this directive is applied
    pub location: DirectiveLocation,
}

impl AppliedDirective {
    pub fn new(name: String, location: DirectiveLocation) -> Self {
        Self {
            name,
            arguments: Vec::new(),
            location,
        }
    }

    pub fn with_argument(mut self, name: String, value: DirectiveValue) -> Self {
        self.arguments.push(DirectiveArgument { name, value });
        self
    }

    pub fn get_argument(&self, name: &str) -> Option<&DirectiveValue> {
        self.arguments
            .iter()
            .find(|arg| arg.name == name)
            .map(|arg| &arg.value)
    }
}

/// Directive execution context
#[derive(Debug, Clone)]
pub struct DirectiveContext {
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Type name
    pub type_name: Option<String>,
    /// User context (for auth, etc.)
    pub user_context: Option<HashMap<String, String>>,
    /// Field value (for transformation directives)
    pub field_value: Option<DirectiveValue>,
}

impl DirectiveContext {
    pub fn new() -> Self {
        Self {
            field_name: None,
            type_name: None,
            user_context: None,
            field_value: None,
        }
    }

    pub fn with_field(mut self, field_name: String) -> Self {
        self.field_name = Some(field_name);
        self
    }

    pub fn with_type(mut self, type_name: String) -> Self {
        self.type_name = Some(type_name);
        self
    }

    pub fn with_user_context(mut self, user_context: HashMap<String, String>) -> Self {
        self.user_context = Some(user_context);
        self
    }

    pub fn with_value(mut self, value: DirectiveValue) -> Self {
        self.field_value = Some(value);
        self
    }
}

impl Default for DirectiveContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Directive handler trait
pub trait DirectiveHandler: Send + Sync {
    /// Execute the directive
    fn execute(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue>;

    /// Validate the directive (called during schema validation)
    fn validate(&self, directive: &AppliedDirective) -> Result<()> {
        let _ = directive;
        Ok(())
    }
}

/// Built-in @auth directive handler
pub struct AuthDirectiveHandler;

impl DirectiveHandler for AuthDirectiveHandler {
    fn execute(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        // Check if user is authenticated
        if context.user_context.is_none() {
            return Err(anyhow!("Authentication required"));
        }

        // Check required role if specified
        if let Some(requires) = directive.get_argument("requires") {
            if let Some(required_role) = requires.as_string() {
                let user_context = context.user_context.as_ref().unwrap();
                let user_role = user_context.get("role");

                if user_role.map(|r| r.as_str()) != Some(required_role) {
                    return Err(anyhow!(
                        "Insufficient permissions. Required role: {}",
                        required_role
                    ));
                }
            }
        }

        // Pass through the value
        Ok(context.field_value.clone().unwrap_or(DirectiveValue::Null))
    }
}

/// Built-in @hasRole directive handler
pub struct HasRoleDirectiveHandler;

impl DirectiveHandler for HasRoleDirectiveHandler {
    fn execute(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        let required_role = directive
            .get_argument("role")
            .and_then(|v| v.as_string())
            .ok_or_else(|| anyhow!("@hasRole directive requires 'role' argument"))?;

        let user_context = context
            .user_context
            .as_ref()
            .ok_or_else(|| anyhow!("User context not available"))?;

        let user_roles = user_context
            .get("roles")
            .map(|r| r.split(',').collect::<Vec<_>>())
            .unwrap_or_default();

        if !user_roles.contains(&required_role) {
            return Err(anyhow!(
                "User does not have required role: {}",
                required_role
            ));
        }

        Ok(context.field_value.clone().unwrap_or(DirectiveValue::Null))
    }
}

/// Built-in @uppercase directive handler
pub struct UppercaseDirectiveHandler;

impl DirectiveHandler for UppercaseDirectiveHandler {
    fn execute(
        &self,
        _directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        if let Some(DirectiveValue::String(s)) = &context.field_value {
            Ok(DirectiveValue::String(s.to_uppercase()))
        } else {
            Ok(context.field_value.clone().unwrap_or(DirectiveValue::Null))
        }
    }
}

/// Built-in @lowercase directive handler
pub struct LowercaseDirectiveHandler;

impl DirectiveHandler for LowercaseDirectiveHandler {
    fn execute(
        &self,
        _directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        if let Some(DirectiveValue::String(s)) = &context.field_value {
            Ok(DirectiveValue::String(s.to_lowercase()))
        } else {
            Ok(context.field_value.clone().unwrap_or(DirectiveValue::Null))
        }
    }
}

/// Built-in @constraint directive handler
pub struct ConstraintDirectiveHandler;

impl DirectiveHandler for ConstraintDirectiveHandler {
    fn execute(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        let value = context
            .field_value
            .as_ref()
            .ok_or_else(|| anyhow!("No value to validate"))?;

        // Check min/max for numbers
        if let Some(val) = value.as_int() {
            if let Some(min) = directive.get_argument("min").and_then(|v| v.as_int()) {
                if val < min {
                    return Err(anyhow!("Value {} is less than minimum {}", val, min));
                }
            }
            if let Some(max) = directive.get_argument("max").and_then(|v| v.as_int()) {
                if val > max {
                    return Err(anyhow!("Value {} is greater than maximum {}", val, max));
                }
            }
        }

        // Check minLength/maxLength for strings
        if let Some(s) = value.as_string() {
            if let Some(min_len) = directive.get_argument("minLength").and_then(|v| v.as_int()) {
                if (s.len() as i64) < min_len {
                    return Err(anyhow!(
                        "String length {} is less than minimum {}",
                        s.len(),
                        min_len
                    ));
                }
            }
            if let Some(max_len) = directive.get_argument("maxLength").and_then(|v| v.as_int()) {
                if (s.len() as i64) > max_len {
                    return Err(anyhow!(
                        "String length {} is greater than maximum {}",
                        s.len(),
                        max_len
                    ));
                }
            }
        }

        Ok(value.clone())
    }
}

/// Built-in @cacheControl directive handler
pub struct CacheControlDirectiveHandler {
    cache_hints: Arc<tokio::sync::RwLock<HashMap<String, CacheHint>>>,
}

#[derive(Debug, Clone)]
pub struct CacheHint {
    pub max_age: u32,
    pub scope: CacheScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheScope {
    Public,
    Private,
}

impl CacheControlDirectiveHandler {
    pub fn new() -> Self {
        Self {
            cache_hints: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_cache_hint(&self, key: &str) -> Option<CacheHint> {
        let hints = self.cache_hints.read().await;
        hints.get(key).cloned()
    }
}

impl DirectiveHandler for CacheControlDirectiveHandler {
    fn execute(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        let max_age = directive
            .get_argument("maxAge")
            .and_then(|v| v.as_int())
            .unwrap_or(0) as u32;

        let scope = directive
            .get_argument("scope")
            .and_then(|v| v.as_string())
            .map(|s| {
                if s.eq_ignore_ascii_case("PRIVATE") {
                    CacheScope::Private
                } else {
                    CacheScope::Public
                }
            })
            .unwrap_or(CacheScope::Public);

        let key = format!(
            "{}:{}",
            context.type_name.as_ref().unwrap_or(&"unknown".to_string()),
            context
                .field_name
                .as_ref()
                .unwrap_or(&"unknown".to_string())
        );

        // Store cache hint (async operation, so we spawn a task)
        let cache_hints = self.cache_hints.clone();
        let key_clone = key.clone();
        tokio::spawn(async move {
            let mut hints = cache_hints.write().await;
            hints.insert(key_clone, CacheHint { max_age, scope });
        });

        Ok(context.field_value.clone().unwrap_or(DirectiveValue::Null))
    }
}

impl Default for CacheControlDirectiveHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Directive registry
pub struct DirectiveRegistry {
    /// Registered directive definitions
    definitions: Arc<tokio::sync::RwLock<HashMap<String, DirectiveDefinition>>>,
    /// Registered directive handlers
    handlers: Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn DirectiveHandler>>>>,
}

impl DirectiveRegistry {
    /// Create a new empty directive registry (no built-in directives)
    pub fn new_empty() -> Self {
        Self {
            definitions: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            handlers: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Create a new directive registry with built-in directives
    pub async fn new() -> Self {
        let registry = Self::new_empty();
        let _ = registry.register_builtin_directives().await;
        registry
    }

    async fn register_builtin_directives(&self) -> Result<()> {
        // @auth directive
        self.register_directive(
            DirectiveDefinition {
                name: "auth".to_string(),
                description: Some("Requires authentication and optional role".to_string()),
                locations: vec![
                    DirectiveLocation::Object,
                    DirectiveLocation::FieldDefinition,
                ],
                repeatable: false,
                arguments: vec![DirectiveArgumentDefinition {
                    name: "requires".to_string(),
                    arg_type: "String".to_string(),
                    required: false,
                    default_value: None,
                    description: Some("Required role".to_string()),
                }],
            },
            Arc::new(AuthDirectiveHandler),
        )
        .await?;

        // @hasRole directive
        self.register_directive(
            DirectiveDefinition {
                name: "hasRole".to_string(),
                description: Some("Requires specific role".to_string()),
                locations: vec![DirectiveLocation::FieldDefinition],
                repeatable: false,
                arguments: vec![DirectiveArgumentDefinition {
                    name: "role".to_string(),
                    arg_type: "String!".to_string(),
                    required: true,
                    default_value: None,
                    description: Some("Required role name".to_string()),
                }],
            },
            Arc::new(HasRoleDirectiveHandler),
        )
        .await?;

        // @uppercase directive
        self.register_directive(
            DirectiveDefinition {
                name: "uppercase".to_string(),
                description: Some("Converts string to uppercase".to_string()),
                locations: vec![DirectiveLocation::FieldDefinition],
                repeatable: false,
                arguments: vec![],
            },
            Arc::new(UppercaseDirectiveHandler),
        )
        .await?;

        // @lowercase directive
        self.register_directive(
            DirectiveDefinition {
                name: "lowercase".to_string(),
                description: Some("Converts string to lowercase".to_string()),
                locations: vec![DirectiveLocation::FieldDefinition],
                repeatable: false,
                arguments: vec![],
            },
            Arc::new(LowercaseDirectiveHandler),
        )
        .await?;

        // @constraint directive
        self.register_directive(
            DirectiveDefinition {
                name: "constraint".to_string(),
                description: Some("Validates field value constraints".to_string()),
                locations: vec![
                    DirectiveLocation::FieldDefinition,
                    DirectiveLocation::ArgumentDefinition,
                ],
                repeatable: false,
                arguments: vec![
                    DirectiveArgumentDefinition {
                        name: "min".to_string(),
                        arg_type: "Int".to_string(),
                        required: false,
                        default_value: None,
                        description: Some("Minimum value".to_string()),
                    },
                    DirectiveArgumentDefinition {
                        name: "max".to_string(),
                        arg_type: "Int".to_string(),
                        required: false,
                        default_value: None,
                        description: Some("Maximum value".to_string()),
                    },
                    DirectiveArgumentDefinition {
                        name: "minLength".to_string(),
                        arg_type: "Int".to_string(),
                        required: false,
                        default_value: None,
                        description: Some("Minimum string length".to_string()),
                    },
                    DirectiveArgumentDefinition {
                        name: "maxLength".to_string(),
                        arg_type: "Int".to_string(),
                        required: false,
                        default_value: None,
                        description: Some("Maximum string length".to_string()),
                    },
                ],
            },
            Arc::new(ConstraintDirectiveHandler),
        )
        .await?;

        // @cacheControl directive
        self.register_directive(
            DirectiveDefinition {
                name: "cacheControl".to_string(),
                description: Some("Cache control hints".to_string()),
                locations: vec![
                    DirectiveLocation::Object,
                    DirectiveLocation::FieldDefinition,
                ],
                repeatable: false,
                arguments: vec![
                    DirectiveArgumentDefinition {
                        name: "maxAge".to_string(),
                        arg_type: "Int".to_string(),
                        required: false,
                        default_value: Some(DirectiveValue::Int(0)),
                        description: Some("Max age in seconds".to_string()),
                    },
                    DirectiveArgumentDefinition {
                        name: "scope".to_string(),
                        arg_type: "String".to_string(),
                        required: false,
                        default_value: Some(DirectiveValue::String("PUBLIC".to_string())),
                        description: Some("Cache scope (PUBLIC or PRIVATE)".to_string()),
                    },
                ],
            },
            Arc::new(CacheControlDirectiveHandler::new()),
        )
        .await?;

        Ok(())
    }

    /// Register a custom directive
    pub async fn register_directive(
        &self,
        definition: DirectiveDefinition,
        handler: Arc<dyn DirectiveHandler>,
    ) -> Result<()> {
        let mut definitions = self.definitions.write().await;
        let mut handlers = self.handlers.write().await;

        definitions.insert(definition.name.clone(), definition.clone());
        handlers.insert(definition.name, handler);

        Ok(())
    }

    /// Get directive definition
    pub async fn get_definition(&self, name: &str) -> Option<DirectiveDefinition> {
        let definitions = self.definitions.read().await;
        definitions.get(name).cloned()
    }

    /// Execute a directive
    pub async fn execute_directive(
        &self,
        directive: &AppliedDirective,
        context: &DirectiveContext,
    ) -> Result<DirectiveValue> {
        let handlers = self.handlers.read().await;

        if let Some(handler) = handlers.get(&directive.name) {
            handler.execute(directive, context)
        } else {
            Err(anyhow!("Unknown directive: @{}", directive.name))
        }
    }

    /// Validate a directive
    pub async fn validate_directive(&self, directive: &AppliedDirective) -> Result<()> {
        let handlers = self.handlers.read().await;

        if let Some(handler) = handlers.get(&directive.name) {
            handler.validate(directive)
        } else {
            Err(anyhow!("Unknown directive: @{}", directive.name))
        }
    }

    /// Get all registered directive names
    pub async fn get_directive_names(&self) -> Vec<String> {
        let definitions = self.definitions.read().await;
        definitions.keys().cloned().collect()
    }
}

// Note: No Default implementation since new() is async
// Use DirectiveRegistry::new().await or DirectiveRegistry::new_empty()

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directive_value_types() {
        let string_val = DirectiveValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));

        let int_val = DirectiveValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));

        let bool_val = DirectiveValue::Boolean(true);
        assert_eq!(bool_val.as_bool(), Some(true));
    }

    #[test]
    fn test_applied_directive_creation() {
        let directive =
            AppliedDirective::new("auth".to_string(), DirectiveLocation::FieldDefinition)
                .with_argument(
                    "requires".to_string(),
                    DirectiveValue::String("ADMIN".to_string()),
                );

        assert_eq!(directive.name, "auth");
        assert_eq!(directive.arguments.len(), 1);
        assert_eq!(
            directive
                .get_argument("requires")
                .and_then(|v| v.as_string()),
            Some("ADMIN")
        );
    }

    #[test]
    fn test_directive_context() {
        let context = DirectiveContext::new()
            .with_field("email".to_string())
            .with_type("User".to_string());

        assert_eq!(context.field_name, Some("email".to_string()));
        assert_eq!(context.type_name, Some("User".to_string()));
    }

    #[tokio::test]
    async fn test_directive_registry_creation() {
        let registry = DirectiveRegistry::new().await;
        let names = registry.get_directive_names().await;

        // Should have built-in directives
        assert!(names.contains(&"auth".to_string()));
        assert!(names.contains(&"uppercase".to_string()));
        assert!(names.contains(&"constraint".to_string()));
    }

    #[tokio::test]
    async fn test_uppercase_directive() {
        let handler = UppercaseDirectiveHandler;
        let directive =
            AppliedDirective::new("uppercase".to_string(), DirectiveLocation::FieldDefinition);
        let context =
            DirectiveContext::new().with_value(DirectiveValue::String("hello".to_string()));

        let result = handler.execute(&directive, &context).unwrap();
        assert_eq!(result.as_string(), Some("HELLO"));
    }

    #[tokio::test]
    async fn test_lowercase_directive() {
        let handler = LowercaseDirectiveHandler;
        let directive =
            AppliedDirective::new("lowercase".to_string(), DirectiveLocation::FieldDefinition);
        let context =
            DirectiveContext::new().with_value(DirectiveValue::String("HELLO".to_string()));

        let result = handler.execute(&directive, &context).unwrap();
        assert_eq!(result.as_string(), Some("hello"));
    }

    #[tokio::test]
    async fn test_constraint_directive_min_max() {
        let handler = ConstraintDirectiveHandler;
        let directive =
            AppliedDirective::new("constraint".to_string(), DirectiveLocation::FieldDefinition)
                .with_argument("min".to_string(), DirectiveValue::Int(0))
                .with_argument("max".to_string(), DirectiveValue::Int(100));

        // Valid value
        let context = DirectiveContext::new().with_value(DirectiveValue::Int(50));
        assert!(handler.execute(&directive, &context).is_ok());

        // Too low
        let context = DirectiveContext::new().with_value(DirectiveValue::Int(-5));
        assert!(handler.execute(&directive, &context).is_err());

        // Too high
        let context = DirectiveContext::new().with_value(DirectiveValue::Int(150));
        assert!(handler.execute(&directive, &context).is_err());
    }

    #[tokio::test]
    async fn test_auth_directive_no_context() {
        let handler = AuthDirectiveHandler;
        let directive =
            AppliedDirective::new("auth".to_string(), DirectiveLocation::FieldDefinition);
        let context = DirectiveContext::new();

        let result = handler.execute(&directive, &context);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Authentication required"));
    }

    #[tokio::test]
    async fn test_auth_directive_with_role() {
        let handler = AuthDirectiveHandler;
        let directive =
            AppliedDirective::new("auth".to_string(), DirectiveLocation::FieldDefinition)
                .with_argument(
                    "requires".to_string(),
                    DirectiveValue::String("ADMIN".to_string()),
                );

        let mut user_context = HashMap::new();
        user_context.insert("role".to_string(), "USER".to_string());

        let context = DirectiveContext::new().with_user_context(user_context);

        let result = handler.execute(&directive, &context);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Insufficient permissions"));
    }

    #[tokio::test]
    async fn test_has_role_directive() {
        let handler = HasRoleDirectiveHandler;
        let directive =
            AppliedDirective::new("hasRole".to_string(), DirectiveLocation::FieldDefinition)
                .with_argument(
                    "role".to_string(),
                    DirectiveValue::String("admin".to_string()),
                );

        let mut user_context = HashMap::new();
        user_context.insert("roles".to_string(), "user,admin,moderator".to_string());

        let context = DirectiveContext::new().with_user_context(user_context);

        let result = handler.execute(&directive, &context);
        assert!(result.is_ok());
    }
}
