//! Tool registry for LLM function calling.
//!
//! Provides a type-safe registry of callable tools, parameter validation,
//! and JSON schema generation compatible with OpenAI function-calling format.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ToolParameter
// ---------------------------------------------------------------------------

/// Specification of a single parameter in a tool definition.
#[derive(Debug, Clone)]
pub struct ToolParameter {
    /// Parameter name.
    pub name: String,
    /// JSON Schema type (e.g. `"string"`, `"integer"`, `"boolean"`).
    pub param_type: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this parameter must be supplied by the caller.
    pub required: bool,
    /// If present, the value must be one of these strings.
    pub enum_values: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// ToolDefinition
// ---------------------------------------------------------------------------

/// Metadata describing a tool exposed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Unique tool name.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// List of accepted parameters.
    pub parameters: Vec<ToolParameter>,
}

// ---------------------------------------------------------------------------
// ToolCall / ToolResult
// ---------------------------------------------------------------------------

/// A tool invocation produced by the LLM.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Name of the tool to invoke.
    pub tool_name: String,
    /// Argument bindings (`{ param_name → value }`).
    pub arguments: HashMap<String, String>,
}

/// The outcome of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Name of the tool that was executed.
    pub tool_name: String,
    /// Serialised output from the tool.
    pub output: String,
    /// Non-`None` when execution failed.
    pub error: Option<String>,
    /// Wall-clock execution time in milliseconds.
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// ToolError
// ---------------------------------------------------------------------------

/// Errors returned by [`ToolRegistry`] operations.
#[derive(Debug, PartialEq, Eq)]
pub enum ToolError {
    /// No tool with the given name is registered.
    NotFound(String),
    /// One or more required parameters were missing or invalid.
    ValidationFailed(Vec<String>),
    /// The tool handler returned an error during execution.
    ExecutionFailed(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(n) => write!(f, "tool not found: {n}"),
            Self::ValidationFailed(errs) => write!(f, "validation failed: {}", errs.join("; ")),
            Self::ExecutionFailed(msg) => write!(f, "execution failed: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

// ---------------------------------------------------------------------------
// ToolHandler trait
// ---------------------------------------------------------------------------

/// A callable tool handler.
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given call and return a result.
    fn execute(&self, call: &ToolCall) -> ToolResult;
    /// Name this handler is registered under (must match `ToolDefinition::name`).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// Registry mapping tool names to their definitions and handlers.
pub struct ToolRegistry {
    tools: HashMap<String, (ToolDefinition, Box<dyn ToolHandler>)>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    ///
    /// Returns `Err(ToolError::ValidationFailed)` if a tool with the same name
    /// is already registered.
    pub fn register(
        &mut self,
        def: ToolDefinition,
        handler: Box<dyn ToolHandler>,
    ) -> Result<(), ToolError> {
        if self.tools.contains_key(&def.name) {
            return Err(ToolError::ValidationFailed(vec![format!(
                "tool '{}' is already registered",
                def.name
            )]));
        }
        self.tools.insert(def.name.clone(), (def, handler));
        Ok(())
    }

    /// Remove a tool by name.  Returns `true` if it existed.
    pub fn deregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Validate and execute a tool call.
    pub fn call(&self, tool_call: &ToolCall) -> Result<ToolResult, ToolError> {
        let (_, handler) = self
            .tools
            .get(&tool_call.tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_call.tool_name.clone()))?;

        let errors = self.validate_call(tool_call);
        if !errors.is_empty() {
            return Err(ToolError::ValidationFailed(errors));
        }

        Ok(handler.execute(tool_call))
    }

    /// Validate a tool call and return a list of validation error strings.
    ///
    /// An empty list means the call is valid.
    pub fn validate_call(&self, tool_call: &ToolCall) -> Vec<String> {
        let (def, _) = match self.tools.get(&tool_call.tool_name) {
            Some(entry) => entry,
            None => return vec![format!("tool '{}' not found", tool_call.tool_name)],
        };

        let mut errors = Vec::new();

        // Check that all required parameters are supplied.
        for param in &def.parameters {
            if param.required && !tool_call.arguments.contains_key(&param.name) {
                errors.push(format!("missing required parameter '{}'", param.name));
            }
        }

        // Check enum constraints.
        for param in &def.parameters {
            if let Some(enum_vals) = &param.enum_values {
                if let Some(value) = tool_call.arguments.get(&param.name) {
                    if !enum_vals.contains(value) {
                        errors.push(format!(
                            "parameter '{}' value '{}' is not in allowed values {:?}",
                            param.name, value, enum_vals
                        ));
                    }
                }
            }
        }

        errors
    }

    /// Retrieve the definition of a registered tool.
    pub fn get_definition(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name).map(|(def, _)| def)
    }

    /// Return a sorted list of all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Emit all registered tools as a JSON string following the OpenAI
    /// function-calling schema format.
    pub fn to_json_schema(&self) -> String {
        let mut parts = Vec::new();
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();

        for name in names {
            if let Some((def, _)) = self.tools.get(name) {
                let mut prop_parts = Vec::new();
                let mut required_params: Vec<String> = Vec::new();

                for param in &def.parameters {
                    let mut prop = format!(
                        r#""{}":{{"type":"{}","description":"{}"}}"#,
                        param.name, param.param_type, param.description
                    );
                    if let Some(enum_vals) = &param.enum_values {
                        let enum_json: Vec<String> =
                            enum_vals.iter().map(|v| format!(r#""{v}""#)).collect();
                        prop = format!(
                            r#""{}":{{"type":"{}","description":"{}","enum":[{}]}}"#,
                            param.name,
                            param.param_type,
                            param.description,
                            enum_json.join(",")
                        );
                    }
                    prop_parts.push(prop);
                    if param.required {
                        required_params.push(format!(r#""{}""#, param.name));
                    }
                }

                let required_json = format!("[{}]", required_params.join(","));
                let props_json = format!("{{{}}}", prop_parts.join(","));

                let entry = format!(
                    r#"{{"name":"{}","description":"{}","parameters":{{"type":"object","properties":{},"required":{}}}}}"#,
                    def.name, def.description, props_json, required_json
                );
                parts.push(entry);
            }
        }

        format!("[{}]", parts.join(","))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // --- Test handler implementations ---

    struct EchoHandler {
        tool_name: String,
    }

    impl ToolHandler for EchoHandler {
        fn execute(&self, call: &ToolCall) -> ToolResult {
            let start = Instant::now();
            let output = call
                .arguments
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            ToolResult {
                tool_name: call.tool_name.clone(),
                output,
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
            }
        }

        fn name(&self) -> &str {
            &self.tool_name
        }
    }

    struct ErrorHandler {
        tool_name: String,
    }

    impl ToolHandler for ErrorHandler {
        fn execute(&self, call: &ToolCall) -> ToolResult {
            ToolResult {
                tool_name: call.tool_name.clone(),
                output: String::new(),
                error: Some("intentional error".into()),
                duration_ms: 0,
            }
        }

        fn name(&self) -> &str {
            &self.tool_name
        }
    }

    // --- Helpers ---

    fn simple_def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("A test tool named {name}"),
            parameters: vec![ToolParameter {
                name: "query".into(),
                param_type: "string".into(),
                description: "Search query".into(),
                required: true,
                enum_values: None,
            }],
        }
    }

    fn echo_handler(name: &str) -> Box<dyn ToolHandler> {
        Box::new(EchoHandler {
            tool_name: name.to_string(),
        })
    }

    fn make_call(tool_name: &str, args: &[(&str, &str)]) -> ToolCall {
        ToolCall {
            tool_name: tool_name.to_string(),
            arguments: args
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    // --- register ---

    #[test]
    fn test_register_success() {
        let mut reg = ToolRegistry::new();
        assert!(reg
            .register(simple_def("search"), echo_handler("search"))
            .is_ok());
        assert_eq!(reg.tool_count(), 1);
    }

    #[test]
    fn test_register_duplicate_returns_error() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("first register");
        let err = reg
            .register(simple_def("search"), echo_handler("search"))
            .unwrap_err();
        match err {
            ToolError::ValidationFailed(msgs) => assert!(!msgs.is_empty()),
            _ => panic!("expected ValidationFailed"),
        }
    }

    // --- deregister ---

    #[test]
    fn test_deregister_existing_returns_true() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        assert!(reg.deregister("search"));
        assert_eq!(reg.tool_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent_returns_false() {
        let mut reg = ToolRegistry::new();
        assert!(!reg.deregister("ghost"));
    }

    // --- call ---

    #[test]
    fn test_call_success() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let call = make_call("search", &[("query", "hello")]);
        let result = reg.call(&call).expect("call should succeed");
        assert_eq!(result.tool_name, "search");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_call_not_found_error() {
        let reg = ToolRegistry::new();
        let call = make_call("ghost", &[]);
        let err = reg.call(&call).unwrap_err();
        assert_eq!(err, ToolError::NotFound("ghost".into()));
    }

    #[test]
    fn test_call_missing_required_param_error() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let call = make_call("search", &[]); // "query" is required but missing
        let err = reg.call(&call).unwrap_err();
        match err {
            ToolError::ValidationFailed(msgs) => {
                assert!(msgs.iter().any(|m| m.contains("query")));
            }
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn test_call_with_error_handler() {
        let mut reg = ToolRegistry::new();
        let def = ToolDefinition {
            name: "err_tool".into(),
            description: "always fails".into(),
            parameters: Vec::new(),
        };
        reg.register(
            def,
            Box::new(ErrorHandler {
                tool_name: "err_tool".into(),
            }),
        )
        .expect("register");
        let call = make_call("err_tool", &[]);
        let result = reg.call(&call).expect("call itself should not error");
        assert!(result.error.is_some());
    }

    // --- validate_call ---

    #[test]
    fn test_validate_call_no_errors_for_valid_call() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let call = make_call("search", &[("query", "test")]);
        let errors = reg.validate_call(&call);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_call_missing_required() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let call = make_call("search", &[]); // missing "query"
        let errors = reg.validate_call(&call);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validate_call_invalid_enum_value() {
        let mut reg = ToolRegistry::new();
        let def = ToolDefinition {
            name: "greet".into(),
            description: "Greet in a language".into(),
            parameters: vec![ToolParameter {
                name: "lang".into(),
                param_type: "string".into(),
                description: "Language".into(),
                required: false,
                enum_values: Some(vec!["en".into(), "ja".into()]),
            }],
        };
        reg.register(def, echo_handler("greet")).expect("register");
        let call = make_call("greet", &[("lang", "fr")]); // "fr" not in enum
        let errors = reg.validate_call(&call);
        assert!(!errors.is_empty());
    }

    // --- get_definition ---

    #[test]
    fn test_get_definition_existing() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let def = reg.get_definition("search").expect("def should exist");
        assert_eq!(def.name, "search");
    }

    #[test]
    fn test_get_definition_missing_returns_none() {
        let reg = ToolRegistry::new();
        assert!(reg.get_definition("ghost").is_none());
    }

    // --- tool_names ---

    #[test]
    fn test_tool_names_returns_all() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("b_tool"), echo_handler("b_tool"))
            .expect("register");
        reg.register(simple_def("a_tool"), echo_handler("a_tool"))
            .expect("register");
        let names = reg.tool_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_tool_names_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("z_tool"), echo_handler("z_tool"))
            .expect("register");
        reg.register(simple_def("a_tool"), echo_handler("a_tool"))
            .expect("register");
        let names = reg.tool_names();
        assert_eq!(names[0], "a_tool");
        assert_eq!(names[1], "z_tool");
    }

    // --- tool_count ---

    #[test]
    fn test_tool_count_empty() {
        let reg = ToolRegistry::new();
        assert_eq!(reg.tool_count(), 0);
    }

    #[test]
    fn test_tool_count_after_register() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("t1"), echo_handler("t1"))
            .expect("register");
        reg.register(simple_def("t2"), echo_handler("t2"))
            .expect("register");
        assert_eq!(reg.tool_count(), 2);
    }

    // --- to_json_schema ---

    #[test]
    fn test_to_json_schema_contains_tool_names() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let schema = reg.to_json_schema();
        assert!(schema.contains("search"));
    }

    #[test]
    fn test_to_json_schema_empty_registry() {
        let reg = ToolRegistry::new();
        assert_eq!(reg.to_json_schema(), "[]");
    }

    #[test]
    fn test_to_json_schema_multiple_tools() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("tool_a"), echo_handler("tool_a"))
            .expect("register");
        reg.register(simple_def("tool_b"), echo_handler("tool_b"))
            .expect("register");
        let schema = reg.to_json_schema();
        assert!(schema.contains("tool_a"));
        assert!(schema.contains("tool_b"));
    }

    #[test]
    fn test_to_json_schema_contains_description() {
        let mut reg = ToolRegistry::new();
        reg.register(simple_def("search"), echo_handler("search"))
            .expect("register");
        let schema = reg.to_json_schema();
        assert!(schema.contains("A test tool named search"));
    }

    // --- ToolResult with/without error ---

    #[test]
    fn test_tool_result_without_error() {
        let result = ToolResult {
            tool_name: "t".into(),
            output: "ok".into(),
            error: None,
            duration_ms: 5,
        };
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_with_error() {
        let result = ToolResult {
            tool_name: "t".into(),
            output: String::new(),
            error: Some("boom".into()),
            duration_ms: 0,
        };
        assert_eq!(result.error.as_deref(), Some("boom"));
    }

    // --- ToolDefinition parameter types ---

    #[test]
    fn test_tool_definition_parameter_types() {
        let def = ToolDefinition {
            name: "calc".into(),
            description: "calculator".into(),
            parameters: vec![
                ToolParameter {
                    name: "x".into(),
                    param_type: "integer".into(),
                    description: "first operand".into(),
                    required: true,
                    enum_values: None,
                },
                ToolParameter {
                    name: "op".into(),
                    param_type: "string".into(),
                    description: "operator".into(),
                    required: true,
                    enum_values: Some(vec!["+".into(), "-".into(), "*".into()]),
                },
            ],
        };
        assert_eq!(def.parameters[0].param_type, "integer");
        assert_eq!(def.parameters[1].param_type, "string");
        assert_eq!(
            def.parameters[1]
                .enum_values
                .as_ref()
                .expect("should succeed")
                .len(),
            3
        );
    }

    // --- ToolError display ---

    #[test]
    fn test_tool_error_display_not_found() {
        let err = ToolError::NotFound("my_tool".into());
        assert!(err.to_string().contains("my_tool"));
    }

    #[test]
    fn test_tool_error_display_validation_failed() {
        let err = ToolError::ValidationFailed(vec!["missing x".into()]);
        assert!(err.to_string().contains("missing x"));
    }

    #[test]
    fn test_tool_error_display_execution_failed() {
        let err = ToolError::ExecutionFailed("crash".into());
        assert!(err.to_string().contains("crash"));
    }

    // --- default() ---

    #[test]
    fn test_default_is_empty() {
        let reg = ToolRegistry::default();
        assert_eq!(reg.tool_count(), 0);
    }
}
