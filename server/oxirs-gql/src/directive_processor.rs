//! Custom GraphQL directive processing engine.
//!
//! Handles built-in directives (`@skip`, `@include`, `@deprecated`) and custom
//! directives (`@auth`, `@cache`) with registration, validation, and ordered
//! execution.

use std::collections::HashMap;
use std::fmt;

/// Errors that can occur during directive processing.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveError {
    /// The directive is not registered.
    UnknownDirective(String),
    /// A required argument is missing.
    MissingArgument { directive: String, argument: String },
    /// An argument has the wrong type.
    InvalidArgument {
        directive: String,
        argument: String,
        reason: String,
    },
    /// Two directives on the same field are contradictory.
    Conflict {
        directive_a: String,
        directive_b: String,
        reason: String,
    },
    /// A general processing error.
    ProcessingError(String),
}

impl fmt::Display for DirectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirectiveError::UnknownDirective(name) => write!(f, "Unknown directive: @{name}"),
            DirectiveError::MissingArgument {
                directive,
                argument,
            } => {
                write!(f, "@{directive} requires argument '{argument}'")
            }
            DirectiveError::InvalidArgument {
                directive,
                argument,
                reason,
            } => {
                write!(f, "@{directive} argument '{argument}': {reason}")
            }
            DirectiveError::Conflict {
                directive_a,
                directive_b,
                reason,
            } => {
                write!(
                    f,
                    "Conflict between @{directive_a} and @{directive_b}: {reason}"
                )
            }
            DirectiveError::ProcessingError(msg) => write!(f, "Directive error: {msg}"),
        }
    }
}

impl std::error::Error for DirectiveError {}

/// A typed argument value for directives.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentValue {
    Boolean(bool),
    String(String),
    Int(i64),
    Float(f64),
    List(Vec<ArgumentValue>),
    Null,
}

impl ArgumentValue {
    /// Try to extract a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ArgumentValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ArgumentValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract an integer value.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ArgumentValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to extract a list of strings.
    pub fn as_string_list(&self) -> Option<Vec<String>> {
        match self {
            ArgumentValue::List(items) => {
                let mut result = Vec::new();
                for item in items {
                    match item {
                        ArgumentValue::String(s) => result.push(s.clone()),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }
}

impl fmt::Display for ArgumentValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgumentValue::Boolean(b) => write!(f, "{b}"),
            ArgumentValue::String(s) => write!(f, "\"{s}\""),
            ArgumentValue::Int(i) => write!(f, "{i}"),
            ArgumentValue::Float(v) => write!(f, "{v}"),
            ArgumentValue::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            ArgumentValue::Null => write!(f, "null"),
        }
    }
}

/// When a directive handler should run relative to field resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DirectivePhase {
    /// Execute before the field resolver.
    PreField,
    /// Execute after the field resolver.
    PostField,
}

/// Schema for a single directive argument.
#[derive(Debug, Clone)]
pub struct ArgumentSchema {
    pub name: String,
    pub required: bool,
    pub expected_type: ArgumentType,
    pub default_value: Option<ArgumentValue>,
}

/// Expected type of a directive argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgumentType {
    Boolean,
    String,
    Int,
    Float,
    StringList,
}

impl ArgumentType {
    fn matches(&self, value: &ArgumentValue) -> bool {
        match (self, value) {
            (ArgumentType::Boolean, ArgumentValue::Boolean(_)) => true,
            (ArgumentType::String, ArgumentValue::String(_)) => true,
            (ArgumentType::Int, ArgumentValue::Int(_)) => true,
            (ArgumentType::Float, ArgumentValue::Float(_)) => true,
            (ArgumentType::StringList, ArgumentValue::List(_)) => value.as_string_list().is_some(),
            _ => false,
        }
    }
}

/// Result of evaluating a directive on a field.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveOutcome {
    /// Continue normal field resolution.
    Continue,
    /// Skip field resolution entirely (e.g., `@skip(if: true)`).
    SkipField,
    /// Return a cached value instead of resolving.
    CachedValue(String),
    /// Deny access to the field.
    AccessDenied(String),
    /// Add a deprecation warning to the response.
    DeprecationWarning(String),
}

/// A registered directive with its argument schema and phase.
#[derive(Debug, Clone)]
pub struct DirectiveDefinition {
    pub name: String,
    pub description: String,
    pub arguments: Vec<ArgumentSchema>,
    pub phase: DirectivePhase,
    /// Whether this directive can coexist with others.
    pub conflicts_with: Vec<String>,
}

/// An applied directive instance on a specific field.
#[derive(Debug, Clone)]
pub struct AppliedDirective {
    pub name: String,
    pub arguments: HashMap<String, ArgumentValue>,
}

impl AppliedDirective {
    /// Create a new applied directive with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: HashMap::new(),
        }
    }

    /// Add an argument to this applied directive.
    pub fn with_arg(mut self, name: impl Into<String>, value: ArgumentValue) -> Self {
        self.arguments.insert(name.into(), value);
        self
    }

    /// Get an argument value by name.
    pub fn get_arg(&self, name: &str) -> Option<&ArgumentValue> {
        self.arguments.get(name)
    }
}

/// Cache entry for the `@cache` directive.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub value: String,
    pub created_at_ms: u64,
    pub ttl_ms: u64,
}

impl CacheEntry {
    /// Check if this cache entry is expired.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_at_ms) > self.ttl_ms
    }
}

/// Deprecation record for the `@deprecated` directive.
#[derive(Debug, Clone, PartialEq)]
pub struct DeprecationInfo {
    pub type_name: String,
    pub field_name: String,
    pub reason: String,
}

/// The main directive processor engine.
pub struct DirectiveProcessor {
    /// Registered directive definitions.
    definitions: HashMap<String, DirectiveDefinition>,
    /// In-memory cache for `@cache` results.
    cache: HashMap<String, CacheEntry>,
    /// Tracked deprecations.
    deprecations: Vec<DeprecationInfo>,
}

impl DirectiveProcessor {
    /// Create a new processor with the built-in directives pre-registered.
    pub fn new() -> Self {
        let mut processor = Self {
            definitions: HashMap::new(),
            cache: HashMap::new(),
            deprecations: Vec::new(),
        };
        processor.register_builtins();
        processor
    }

    /// Create an empty processor with no built-in directives.
    pub fn empty() -> Self {
        Self {
            definitions: HashMap::new(),
            cache: HashMap::new(),
            deprecations: Vec::new(),
        }
    }

    // -- Registration --

    fn register_builtins(&mut self) {
        // @skip(if: Boolean!)
        self.register(DirectiveDefinition {
            name: "skip".into(),
            description: "Skip this field if the argument is true".into(),
            arguments: vec![ArgumentSchema {
                name: "if".into(),
                required: true,
                expected_type: ArgumentType::Boolean,
                default_value: None,
            }],
            phase: DirectivePhase::PreField,
            conflicts_with: vec!["include".into()],
        });

        // @include(if: Boolean!)
        self.register(DirectiveDefinition {
            name: "include".into(),
            description: "Include this field only if the argument is true".into(),
            arguments: vec![ArgumentSchema {
                name: "if".into(),
                required: true,
                expected_type: ArgumentType::Boolean,
                default_value: None,
            }],
            phase: DirectivePhase::PreField,
            conflicts_with: vec!["skip".into()],
        });

        // @deprecated(reason: String)
        self.register(DirectiveDefinition {
            name: "deprecated".into(),
            description: "Mark a field as deprecated".into(),
            arguments: vec![ArgumentSchema {
                name: "reason".into(),
                required: false,
                expected_type: ArgumentType::String,
                default_value: Some(ArgumentValue::String("No longer supported".into())),
            }],
            phase: DirectivePhase::PostField,
            conflicts_with: vec![],
        });

        // @auth(roles: [String!])
        self.register(DirectiveDefinition {
            name: "auth".into(),
            description: "Restrict field access by role".into(),
            arguments: vec![
                ArgumentSchema {
                    name: "roles".into(),
                    required: false,
                    expected_type: ArgumentType::StringList,
                    default_value: None,
                },
                ArgumentSchema {
                    name: "deny".into(),
                    required: false,
                    expected_type: ArgumentType::StringList,
                    default_value: None,
                },
            ],
            phase: DirectivePhase::PreField,
            conflicts_with: vec![],
        });

        // @cache(ttl: Int)
        self.register(DirectiveDefinition {
            name: "cache".into(),
            description: "Cache the field result with a TTL".into(),
            arguments: vec![ArgumentSchema {
                name: "ttl".into(),
                required: false,
                expected_type: ArgumentType::Int,
                default_value: Some(ArgumentValue::Int(60)),
            }],
            phase: DirectivePhase::PreField,
            conflicts_with: vec![],
        });
    }

    /// Register a custom directive definition.
    pub fn register(&mut self, definition: DirectiveDefinition) {
        self.definitions.insert(definition.name.clone(), definition);
    }

    /// Check if a directive is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    /// Get a directive definition by name.
    pub fn get_definition(&self, name: &str) -> Option<&DirectiveDefinition> {
        self.definitions.get(name)
    }

    /// Return the names of all registered directives.
    pub fn registered_names(&self) -> Vec<String> {
        self.definitions.keys().cloned().collect()
    }

    // -- Validation --

    /// Validate that an applied directive's arguments match its schema.
    pub fn validate(&self, applied: &AppliedDirective) -> Result<(), DirectiveError> {
        let def = self
            .definitions
            .get(&applied.name)
            .ok_or_else(|| DirectiveError::UnknownDirective(applied.name.clone()))?;

        for schema in &def.arguments {
            match applied.arguments.get(&schema.name) {
                Some(value) => {
                    if !schema.expected_type.matches(value) {
                        return Err(DirectiveError::InvalidArgument {
                            directive: applied.name.clone(),
                            argument: schema.name.clone(),
                            reason: format!("expected {:?} but got {value}", schema.expected_type),
                        });
                    }
                }
                None => {
                    if schema.required && schema.default_value.is_none() {
                        return Err(DirectiveError::MissingArgument {
                            directive: applied.name.clone(),
                            argument: schema.name.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Detect conflicts among a set of directives applied to the same field.
    pub fn detect_conflicts(&self, directives: &[AppliedDirective]) -> Result<(), DirectiveError> {
        for (i, da) in directives.iter().enumerate() {
            if let Some(def_a) = self.definitions.get(&da.name) {
                for db in directives.iter().skip(i + 1) {
                    if def_a.conflicts_with.contains(&db.name) {
                        return Err(DirectiveError::Conflict {
                            directive_a: da.name.clone(),
                            directive_b: db.name.clone(),
                            reason: "mutually exclusive directives on the same field".into(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    // -- Execution --

    /// Execute the `@skip` directive.
    pub fn execute_skip(
        &self,
        applied: &AppliedDirective,
    ) -> Result<DirectiveOutcome, DirectiveError> {
        let condition = applied
            .get_arg("if")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| DirectiveError::MissingArgument {
                directive: "skip".into(),
                argument: "if".into(),
            })?;
        if condition {
            Ok(DirectiveOutcome::SkipField)
        } else {
            Ok(DirectiveOutcome::Continue)
        }
    }

    /// Execute the `@include` directive.
    pub fn execute_include(
        &self,
        applied: &AppliedDirective,
    ) -> Result<DirectiveOutcome, DirectiveError> {
        let condition = applied
            .get_arg("if")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| DirectiveError::MissingArgument {
                directive: "include".into(),
                argument: "if".into(),
            })?;
        if condition {
            Ok(DirectiveOutcome::Continue)
        } else {
            Ok(DirectiveOutcome::SkipField)
        }
    }

    /// Execute the `@deprecated` directive and record the deprecation.
    pub fn execute_deprecated(
        &mut self,
        applied: &AppliedDirective,
        type_name: &str,
        field_name: &str,
    ) -> Result<DirectiveOutcome, DirectiveError> {
        let reason = applied
            .get_arg("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("No longer supported");

        let info = DeprecationInfo {
            type_name: type_name.to_string(),
            field_name: field_name.to_string(),
            reason: reason.to_string(),
        };
        if !self.deprecations.contains(&info) {
            self.deprecations.push(info);
        }
        Ok(DirectiveOutcome::DeprecationWarning(reason.to_string()))
    }

    /// Execute the `@auth` directive with the user's roles.
    pub fn execute_auth(
        &self,
        applied: &AppliedDirective,
        user_roles: &[String],
    ) -> Result<DirectiveOutcome, DirectiveError> {
        // Check deny list first
        if let Some(deny) = applied.get_arg("deny").and_then(|v| v.as_string_list()) {
            for role in user_roles {
                if deny.contains(role) {
                    return Ok(DirectiveOutcome::AccessDenied(format!(
                        "Role '{role}' is denied access"
                    )));
                }
            }
        }

        // Check required roles
        if let Some(required) = applied.get_arg("roles").and_then(|v| v.as_string_list()) {
            if required.is_empty() {
                return Ok(DirectiveOutcome::Continue);
            }
            let has_any = user_roles.iter().any(|r| required.contains(r));
            if !has_any {
                return Ok(DirectiveOutcome::AccessDenied(format!(
                    "Requires one of: {:?}",
                    required
                )));
            }
        }

        Ok(DirectiveOutcome::Continue)
    }

    /// Execute the `@cache` directive.
    pub fn execute_cache(
        &mut self,
        _applied: &AppliedDirective,
        cache_key: &str,
        current_time_ms: u64,
    ) -> Result<DirectiveOutcome, DirectiveError> {
        // Check if we have a valid cache entry
        if let Some(entry) = self.cache.get(cache_key) {
            if !entry.is_expired(current_time_ms) {
                return Ok(DirectiveOutcome::CachedValue(entry.value.clone()));
            }
        }
        Ok(DirectiveOutcome::Continue)
    }

    /// Store a value in the cache for the `@cache` directive.
    pub fn cache_store(
        &mut self,
        applied: &AppliedDirective,
        cache_key: &str,
        value: &str,
        current_time_ms: u64,
    ) {
        let ttl = applied
            .get_arg("ttl")
            .and_then(|v| v.as_int())
            .unwrap_or(60);
        let ttl_ms = (ttl as u64) * 1000;

        self.cache.insert(
            cache_key.to_string(),
            CacheEntry {
                key: cache_key.to_string(),
                value: value.to_string(),
                created_at_ms: current_time_ms,
                ttl_ms,
            },
        );
    }

    /// Generate a cache key for a field + arguments combination.
    pub fn generate_cache_key(
        type_name: &str,
        field_name: &str,
        args: &HashMap<String, ArgumentValue>,
    ) -> String {
        let mut key = format!("{type_name}:{field_name}");
        let mut arg_keys: Vec<&String> = args.keys().collect();
        arg_keys.sort();
        for k in arg_keys {
            if let Some(v) = args.get(k) {
                key.push_str(&format!(":{k}={v}"));
            }
        }
        key
    }

    /// Process a list of directives for a field in the correct phase order.
    #[allow(clippy::too_many_arguments)]
    pub fn process_directives(
        &mut self,
        directives: &[AppliedDirective],
        phase: DirectivePhase,
        user_roles: &[String],
        cache_key: &str,
        current_time_ms: u64,
        type_name: &str,
        field_name: &str,
    ) -> Result<DirectiveOutcome, DirectiveError> {
        for applied in directives {
            let def = match self.definitions.get(&applied.name) {
                Some(d) => d,
                None => continue,
            };
            if def.phase != phase {
                continue;
            }
            let outcome = match applied.name.as_str() {
                "skip" => self.execute_skip(applied)?,
                "include" => self.execute_include(applied)?,
                "deprecated" => self.execute_deprecated(applied, type_name, field_name)?,
                "auth" => self.execute_auth(applied, user_roles)?,
                "cache" => self.execute_cache(applied, cache_key, current_time_ms)?,
                _ => DirectiveOutcome::Continue,
            };
            match &outcome {
                DirectiveOutcome::Continue | DirectiveOutcome::DeprecationWarning(_) => continue,
                _ => return Ok(outcome),
            }
        }
        Ok(DirectiveOutcome::Continue)
    }

    // -- Introspection --

    /// Return all tracked deprecations.
    pub fn deprecations(&self) -> &[DeprecationInfo] {
        &self.deprecations
    }

    /// Return the number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the directive cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Evict expired entries from the cache.
    pub fn evict_expired(&mut self, current_time_ms: u64) {
        self.cache
            .retain(|_, entry| !entry.is_expired(current_time_ms));
    }
}

impl Default for DirectiveProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Built-in registration --

    #[test]
    fn test_builtins_registered() {
        let proc = DirectiveProcessor::new();
        assert!(proc.is_registered("skip"));
        assert!(proc.is_registered("include"));
        assert!(proc.is_registered("deprecated"));
        assert!(proc.is_registered("auth"));
        assert!(proc.is_registered("cache"));
    }

    #[test]
    fn test_empty_processor_no_builtins() {
        let proc = DirectiveProcessor::empty();
        assert!(!proc.is_registered("skip"));
    }

    #[test]
    fn test_custom_directive_registration() {
        let mut proc = DirectiveProcessor::new();
        proc.register(DirectiveDefinition {
            name: "rateLimit".into(),
            description: "Rate-limit a field".into(),
            arguments: vec![ArgumentSchema {
                name: "max".into(),
                required: true,
                expected_type: ArgumentType::Int,
                default_value: None,
            }],
            phase: DirectivePhase::PreField,
            conflicts_with: vec![],
        });
        assert!(proc.is_registered("rateLimit"));
    }

    #[test]
    fn test_get_definition() {
        let proc = DirectiveProcessor::new();
        let def = proc.get_definition("skip").expect("skip definition");
        assert_eq!(def.name, "skip");
        assert_eq!(def.arguments.len(), 1);
    }

    #[test]
    fn test_registered_names() {
        let proc = DirectiveProcessor::new();
        let names = proc.registered_names();
        assert!(names.contains(&"skip".to_string()));
        assert!(names.contains(&"include".to_string()));
    }

    // -- @skip directive --

    #[test]
    fn test_skip_true_skips() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("skip").with_arg("if", ArgumentValue::Boolean(true));
        let result = proc.execute_skip(&applied).expect("exec");
        assert_eq!(result, DirectiveOutcome::SkipField);
    }

    #[test]
    fn test_skip_false_continues() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("skip").with_arg("if", ArgumentValue::Boolean(false));
        let result = proc.execute_skip(&applied).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_skip_missing_if_error() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("skip");
        let err = proc.execute_skip(&applied).expect_err("missing");
        assert!(matches!(err, DirectiveError::MissingArgument { .. }));
    }

    // -- @include directive --

    #[test]
    fn test_include_true_continues() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("include").with_arg("if", ArgumentValue::Boolean(true));
        let result = proc.execute_include(&applied).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_include_false_skips() {
        let proc = DirectiveProcessor::new();
        let applied =
            AppliedDirective::new("include").with_arg("if", ArgumentValue::Boolean(false));
        let result = proc.execute_include(&applied).expect("exec");
        assert_eq!(result, DirectiveOutcome::SkipField);
    }

    // -- @deprecated directive --

    #[test]
    fn test_deprecated_with_reason() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("deprecated").with_arg(
            "reason",
            ArgumentValue::String("Use newField instead".into()),
        );
        let result = proc
            .execute_deprecated(&applied, "Query", "oldField")
            .expect("exec");
        assert!(matches!(result, DirectiveOutcome::DeprecationWarning(_)));
        assert_eq!(proc.deprecations().len(), 1);
        assert_eq!(proc.deprecations()[0].field_name, "oldField");
        assert_eq!(proc.deprecations()[0].reason, "Use newField instead");
    }

    #[test]
    fn test_deprecated_default_reason() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("deprecated");
        let result = proc
            .execute_deprecated(&applied, "Query", "old")
            .expect("exec");
        match result {
            DirectiveOutcome::DeprecationWarning(reason) => {
                assert_eq!(reason, "No longer supported");
            }
            other => panic!("Expected deprecation warning, got {other:?}"),
        }
    }

    #[test]
    fn test_deprecated_no_duplicates() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("deprecated")
            .with_arg("reason", ArgumentValue::String("old".into()));
        proc.execute_deprecated(&applied, "T", "f").expect("exec");
        proc.execute_deprecated(&applied, "T", "f").expect("exec");
        assert_eq!(proc.deprecations().len(), 1);
    }

    // -- @auth directive --

    #[test]
    fn test_auth_allowed_role() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth").with_arg(
            "roles",
            ArgumentValue::List(vec![
                ArgumentValue::String("admin".into()),
                ArgumentValue::String("editor".into()),
            ]),
        );
        let roles = vec!["editor".to_string()];
        let result = proc.execute_auth(&applied, &roles).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_auth_denied_no_matching_role() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth").with_arg(
            "roles",
            ArgumentValue::List(vec![ArgumentValue::String("admin".into())]),
        );
        let roles = vec!["viewer".to_string()];
        let result = proc.execute_auth(&applied, &roles).expect("exec");
        assert!(matches!(result, DirectiveOutcome::AccessDenied(_)));
    }

    #[test]
    fn test_auth_deny_list_blocks() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth").with_arg(
            "deny",
            ArgumentValue::List(vec![ArgumentValue::String("banned".into())]),
        );
        let roles = vec!["banned".to_string()];
        let result = proc.execute_auth(&applied, &roles).expect("exec");
        assert!(matches!(result, DirectiveOutcome::AccessDenied(_)));
    }

    #[test]
    fn test_auth_deny_does_not_block_other_roles() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth").with_arg(
            "deny",
            ArgumentValue::List(vec![ArgumentValue::String("banned".into())]),
        );
        let roles = vec!["admin".to_string()];
        let result = proc.execute_auth(&applied, &roles).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_auth_no_args_continues() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth");
        let roles = vec!["admin".to_string()];
        let result = proc.execute_auth(&applied, &roles).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_auth_empty_roles_list_continues() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("auth").with_arg("roles", ArgumentValue::List(vec![]));
        let result = proc.execute_auth(&applied, &[]).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    // -- @cache directive --

    #[test]
    fn test_cache_miss() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(60));
        let result = proc
            .execute_cache(&applied, "Query:field", 1000)
            .expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_cache_hit() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(60));
        proc.cache_store(&applied, "Query:f", "cached_value", 1000);
        let result = proc.execute_cache(&applied, "Query:f", 2000).expect("exec");
        assert_eq!(result, DirectiveOutcome::CachedValue("cached_value".into()));
    }

    #[test]
    fn test_cache_expired() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(1)); // 1 second TTL
        proc.cache_store(&applied, "k", "v", 1000);
        // 2 seconds later => expired
        let result = proc.execute_cache(&applied, "k", 3000).expect("exec");
        assert_eq!(result, DirectiveOutcome::Continue);
    }

    #[test]
    fn test_cache_evict_expired() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(1));
        proc.cache_store(&applied, "k1", "v1", 1000);
        proc.cache_store(&applied, "k2", "v2", 5000);
        assert_eq!(proc.cache_size(), 2);
        proc.evict_expired(3000);
        assert_eq!(proc.cache_size(), 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(60));
        proc.cache_store(&applied, "k", "v", 0);
        assert_eq!(proc.cache_size(), 1);
        proc.clear_cache();
        assert_eq!(proc.cache_size(), 0);
    }

    #[test]
    fn test_generate_cache_key() {
        let mut args = HashMap::new();
        args.insert("id".into(), ArgumentValue::Int(42));
        let key = DirectiveProcessor::generate_cache_key("Query", "user", &args);
        assert!(key.starts_with("Query:user"));
        assert!(key.contains("id=42"));
    }

    #[test]
    fn test_generate_cache_key_stable_ordering() {
        let mut args = HashMap::new();
        args.insert("b".into(), ArgumentValue::String("2".into()));
        args.insert("a".into(), ArgumentValue::String("1".into()));
        let key = DirectiveProcessor::generate_cache_key("T", "f", &args);
        // Arguments sorted alphabetically
        let a_pos = key.find(":a=").expect("a present");
        let b_pos = key.find(":b=").expect("b present");
        assert!(a_pos < b_pos);
    }

    // -- Validation --

    #[test]
    fn test_validate_valid_skip() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("skip").with_arg("if", ArgumentValue::Boolean(true));
        assert!(proc.validate(&applied).is_ok());
    }

    #[test]
    fn test_validate_missing_required_arg() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("skip"); // missing 'if'
        let err = proc.validate(&applied).expect_err("validate");
        assert!(matches!(err, DirectiveError::MissingArgument { .. }));
    }

    #[test]
    fn test_validate_wrong_arg_type() {
        let proc = DirectiveProcessor::new();
        let applied =
            AppliedDirective::new("skip").with_arg("if", ArgumentValue::String("yes".into()));
        let err = proc.validate(&applied).expect_err("validate");
        assert!(matches!(err, DirectiveError::InvalidArgument { .. }));
    }

    #[test]
    fn test_validate_unknown_directive() {
        let proc = DirectiveProcessor::new();
        let applied = AppliedDirective::new("unknown");
        let err = proc.validate(&applied).expect_err("validate");
        assert!(matches!(err, DirectiveError::UnknownDirective(_)));
    }

    // -- Conflict detection --

    #[test]
    fn test_conflict_skip_include() {
        let proc = DirectiveProcessor::new();
        let skip = AppliedDirective::new("skip").with_arg("if", ArgumentValue::Boolean(true));
        let include =
            AppliedDirective::new("include").with_arg("if", ArgumentValue::Boolean(false));
        let err = proc
            .detect_conflicts(&[skip, include])
            .expect_err("conflict");
        assert!(matches!(err, DirectiveError::Conflict { .. }));
    }

    #[test]
    fn test_no_conflict_independent() {
        let proc = DirectiveProcessor::new();
        let auth = AppliedDirective::new("auth");
        let cache = AppliedDirective::new("cache").with_arg("ttl", ArgumentValue::Int(30));
        assert!(proc.detect_conflicts(&[auth, cache]).is_ok());
    }

    // -- Process directives --

    #[test]
    fn test_process_skip_in_prefield() {
        let mut proc = DirectiveProcessor::new();
        let directives =
            vec![AppliedDirective::new("skip").with_arg("if", ArgumentValue::Boolean(true))];
        let result = proc
            .process_directives(&directives, DirectivePhase::PreField, &[], "", 0, "Q", "f")
            .expect("process");
        assert_eq!(result, DirectiveOutcome::SkipField);
    }

    #[test]
    fn test_process_deprecated_in_postfield() {
        let mut proc = DirectiveProcessor::new();
        let directives = vec![AppliedDirective::new("deprecated")
            .with_arg("reason", ArgumentValue::String("old".into()))];
        let result = proc
            .process_directives(&directives, DirectivePhase::PostField, &[], "", 0, "Q", "f")
            .expect("process");
        // DeprecationWarning is non-blocking, so process continues
        assert_eq!(result, DirectiveOutcome::Continue);
        assert_eq!(proc.deprecations().len(), 1);
    }

    #[test]
    fn test_process_auth_denies_access() {
        let mut proc = DirectiveProcessor::new();
        let directives = vec![AppliedDirective::new("auth").with_arg(
            "roles",
            ArgumentValue::List(vec![ArgumentValue::String("admin".into())]),
        )];
        let result = proc
            .process_directives(
                &directives,
                DirectivePhase::PreField,
                &["viewer".to_string()],
                "",
                0,
                "Q",
                "f",
            )
            .expect("process");
        assert!(matches!(result, DirectiveOutcome::AccessDenied(_)));
    }

    // -- ArgumentValue --

    #[test]
    fn test_argument_value_as_bool() {
        assert_eq!(ArgumentValue::Boolean(true).as_bool(), Some(true));
        assert_eq!(ArgumentValue::String("x".into()).as_bool(), None);
    }

    #[test]
    fn test_argument_value_as_str() {
        assert_eq!(ArgumentValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(ArgumentValue::Int(1).as_str(), None);
    }

    #[test]
    fn test_argument_value_as_int() {
        assert_eq!(ArgumentValue::Int(42).as_int(), Some(42));
        assert_eq!(ArgumentValue::Boolean(true).as_int(), None);
    }

    #[test]
    fn test_argument_value_display() {
        assert_eq!(format!("{}", ArgumentValue::Boolean(true)), "true");
        assert_eq!(format!("{}", ArgumentValue::String("x".into())), "\"x\"");
        assert_eq!(format!("{}", ArgumentValue::Null), "null");
        assert_eq!(format!("{}", ArgumentValue::Int(5)), "5");
    }

    #[test]
    fn test_argument_value_list_display() {
        let list = ArgumentValue::List(vec![
            ArgumentValue::String("a".into()),
            ArgumentValue::String("b".into()),
        ]);
        assert_eq!(format!("{list}"), "[\"a\", \"b\"]");
    }

    #[test]
    fn test_as_string_list() {
        let list = ArgumentValue::List(vec![
            ArgumentValue::String("x".into()),
            ArgumentValue::String("y".into()),
        ]);
        assert_eq!(
            list.as_string_list(),
            Some(vec!["x".to_string(), "y".to_string()])
        );
    }

    #[test]
    fn test_as_string_list_mixed_types_none() {
        let list = ArgumentValue::List(vec![
            ArgumentValue::String("x".into()),
            ArgumentValue::Int(1),
        ]);
        assert_eq!(list.as_string_list(), None);
    }

    // -- Error display --

    #[test]
    fn test_directive_error_display() {
        let err = DirectiveError::UnknownDirective("foo".into());
        assert!(format!("{err}").contains("@foo"));
        let err2 = DirectiveError::MissingArgument {
            directive: "skip".into(),
            argument: "if".into(),
        };
        assert!(format!("{err2}").contains("@skip"));
        let err3 = DirectiveError::Conflict {
            directive_a: "skip".into(),
            directive_b: "include".into(),
            reason: "test".into(),
        };
        assert!(format!("{err3}").contains("@skip"));
    }

    // -- Default trait --

    #[test]
    fn test_default_processor() {
        let proc = DirectiveProcessor::default();
        assert!(proc.is_registered("skip"));
        assert!(proc.is_registered("include"));
    }

    // -- CacheEntry --

    #[test]
    fn test_cache_entry_expiry() {
        let entry = CacheEntry {
            key: "k".into(),
            value: "v".into(),
            created_at_ms: 1000,
            ttl_ms: 5000,
        };
        assert!(!entry.is_expired(3000));
        assert!(entry.is_expired(7000));
    }
}
