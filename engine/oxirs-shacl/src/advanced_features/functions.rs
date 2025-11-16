//! SHACL Advanced Features - Functions
//!
//! Implementation of SHACL Functions for custom operations and transformations.
//! Provides built-in functions and a registry for custom functions.
//!
//! Based on the W3C SHACL Advanced Features specification.

use crate::{Result, ShaclError};
use oxirs_core::{
    model::{Literal, NamedNode, Term},
    Store,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Parameter type for SHACL functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    /// IRI/URI parameter
    Iri,
    /// Literal value parameter
    Literal,
    /// Any RDF term (IRI, literal, or blank node)
    RdfTerm,
    /// Boolean parameter
    Boolean,
    /// Integer parameter
    Integer,
    /// Decimal parameter
    Decimal,
    /// String parameter
    String,
    /// Custom type parameter
    Custom(String),
    /// List of values
    List(Box<ParameterType>),
}

/// Function parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Whether this parameter is optional
    pub optional: bool,
    /// Parameter order/position
    pub order: usize,
    /// Default value if parameter is optional
    pub default_value: Option<String>,
    /// Parameter description
    pub description: Option<String>,
}

impl FunctionParameter {
    /// Create a new required parameter
    pub fn required(name: impl Into<String>, param_type: ParameterType, order: usize) -> Self {
        Self {
            name: name.into(),
            param_type,
            optional: false,
            order,
            default_value: None,
            description: None,
        }
    }

    /// Create a new optional parameter
    pub fn optional(
        name: impl Into<String>,
        param_type: ParameterType,
        order: usize,
        default_value: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            optional: true,
            order,
            default_value,
            description: None,
        }
    }

    /// Add description to parameter
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Return type for SHACL functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnType {
    /// Returns a single value
    Single(ParameterType),
    /// Returns a list of values
    List(ParameterType),
    /// Returns multiple different types
    Multiple(Vec<ParameterType>),
    /// Returns void/nothing
    Void,
}

/// Function metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionMetadata {
    /// Function author
    pub author: Option<String>,
    /// Function version
    pub version: Option<String>,
    /// Function description
    pub description: Option<String>,
    /// Function documentation URL
    pub documentation: Option<String>,
    /// Function tags
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// SHACL Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclFunction {
    /// Unique function identifier (IRI)
    pub id: String,
    /// Function name
    pub name: String,
    /// Function parameters
    pub parameters: Vec<FunctionParameter>,
    /// Return type
    pub return_type: ReturnType,
    /// Function metadata
    pub metadata: FunctionMetadata,
    /// Whether this function has side effects
    pub has_side_effects: bool,
    /// Whether this function is deterministic
    pub deterministic: bool,
}

impl ShaclFunction {
    /// Create a new function
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        parameters: Vec<FunctionParameter>,
        return_type: ReturnType,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            parameters,
            return_type,
            metadata: FunctionMetadata::default(),
            has_side_effects: false,
            deterministic: true,
        }
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: FunctionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Mark as having side effects
    pub fn with_side_effects(mut self) -> Self {
        self.has_side_effects = true;
        self
    }

    /// Mark as non-deterministic
    pub fn non_deterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// Validate arguments against parameter definitions
    pub fn validate_arguments(&self, arguments: &HashMap<String, Term>) -> Result<()> {
        // Check required parameters
        for param in &self.parameters {
            if !param.optional && !arguments.contains_key(&param.name) {
                return Err(ShaclError::ValidationEngine(format!(
                    "Missing required parameter: {}",
                    param.name
                )));
            }
        }

        // Check for unknown parameters
        for arg_name in arguments.keys() {
            if !self.parameters.iter().any(|p| &p.name == arg_name) {
                return Err(ShaclError::ValidationEngine(format!(
                    "Unknown parameter: {}",
                    arg_name
                )));
            }
        }

        Ok(())
    }
}

/// Function invocation
#[derive(Debug, Clone)]
pub struct FunctionInvocation {
    /// Function identifier
    pub function_id: String,
    /// Arguments passed to the function
    pub arguments: HashMap<String, Term>,
    /// Context for execution
    pub context: FunctionContext,
}

/// Function execution context
#[derive(Debug, Clone, Default)]
pub struct FunctionContext {
    /// Variables available in the context
    pub variables: HashMap<String, Term>,
    /// Execution depth (for recursion detection)
    pub depth: usize,
    /// Maximum allowed depth
    pub max_depth: usize,
}

impl FunctionContext {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            depth: 0,
            max_depth: 100,
        }
    }

    /// Increment depth
    pub fn increment_depth(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ShaclError::RecursionLimit(format!(
                "Maximum function call depth exceeded: {}",
                self.max_depth
            )));
        }
        Ok(())
    }

    /// Decrement depth
    pub fn decrement_depth(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }
}

/// Function execution result
#[derive(Debug, Clone)]
pub enum FunctionResult {
    /// Single value result
    Single(Option<Term>),
    /// Multiple values result
    Multiple(Vec<Term>),
    /// Error during execution
    Error(String),
}

impl FunctionResult {
    /// Create a success result with a single value
    pub fn value(term: Term) -> Self {
        FunctionResult::Single(Some(term))
    }

    /// Create a success result with no value
    pub fn none() -> Self {
        FunctionResult::Single(None)
    }

    /// Create a success result with multiple values
    pub fn values(terms: Vec<Term>) -> Self {
        FunctionResult::Multiple(terms)
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        FunctionResult::Error(message.into())
    }

    /// Check if result is an error
    pub fn is_error(&self) -> bool {
        matches!(self, FunctionResult::Error(_))
    }

    /// Get single value if available
    pub fn as_single(&self) -> Option<&Term> {
        match self {
            FunctionResult::Single(Some(term)) => Some(term),
            _ => None,
        }
    }

    /// Get multiple values if available
    pub fn as_multiple(&self) -> Option<&[Term]> {
        match self {
            FunctionResult::Multiple(terms) => Some(terms),
            _ => None,
        }
    }
}

/// Function executor trait
pub trait FunctionExecutor: Send + Sync {
    /// Execute a function
    fn execute(
        &self,
        function: &ShaclFunction,
        invocation: &FunctionInvocation,
        store: &dyn Store,
    ) -> Result<FunctionResult>;

    /// Get executor name
    fn name(&self) -> &str;

    /// Check if executor can handle a function
    fn can_execute(&self, function: &ShaclFunction) -> bool;
}

/// Built-in function executor
pub struct BuiltInFunctionExecutor;

impl BuiltInFunctionExecutor {
    /// Create a new built-in function executor
    pub fn new() -> Self {
        Self
    }

    /// Execute string concatenation
    fn concat(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        let mut result = String::new();
        let mut i = 0;
        loop {
            let key = format!("arg{}", i);
            if let Some(term) = args.get(&key) {
                match term {
                    Term::Literal(lit) => result.push_str(lit.value()),
                    _ => result.push_str(&term.to_string()),
                }
                i += 1;
            } else {
                break;
            }
        }
        Ok(FunctionResult::value(Term::Literal(
            Literal::new_simple_literal(result),
        )))
    }

    /// Execute string uppercase
    fn upper_case(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("input") {
            let upper = lit.value().to_uppercase();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_simple_literal(upper),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "upperCase requires a string literal argument".to_string(),
            ))
        }
    }

    /// Execute string lowercase
    fn lower_case(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("input") {
            let lower = lit.value().to_lowercase();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_simple_literal(lower),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "lowerCase requires a string literal argument".to_string(),
            ))
        }
    }

    /// Execute substring
    fn substring(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("input") {
            let start = match args.get("start") {
                Some(Term::Literal(l)) => l
                    .value()
                    .parse::<usize>()
                    .map_err(|_| ShaclError::ValidationEngine("Invalid start index".to_string()))?,
                _ => {
                    return Err(ShaclError::ValidationEngine(
                        "substring requires start index".to_string(),
                    ))
                }
            };

            let value = lit.value();
            let result: String = if let Some(Term::Literal(l)) = args.get("length") {
                let length = l
                    .value()
                    .parse::<usize>()
                    .map_err(|_| ShaclError::ValidationEngine("Invalid length".to_string()))?;
                value.chars().skip(start).take(length).collect()
            } else {
                value.chars().skip(start).collect()
            };

            Ok(FunctionResult::value(Term::Literal(
                Literal::new_simple_literal(result),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "substring requires a string literal input".to_string(),
            ))
        }
    }

    /// Execute string length
    fn str_length(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("input") {
            let length = lit.value().chars().count();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    length.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "strLength requires a string literal argument".to_string(),
            ))
        }
    }

    /// Execute absolute value
    fn abs(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("value") {
            let value: f64 = lit.value().parse().map_err(|_| {
                ShaclError::ValidationEngine("abs requires a numeric argument".to_string())
            })?;
            let result = value.abs();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "abs requires a numeric literal argument".to_string(),
            ))
        }
    }

    /// Execute ceiling
    fn ceil(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("value") {
            let value: f64 = lit.value().parse().map_err(|_| {
                ShaclError::ValidationEngine("ceil requires a numeric argument".to_string())
            })?;
            let result = value.ceil();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "ceil requires a numeric literal argument".to_string(),
            ))
        }
    }

    /// Execute floor
    fn floor(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("value") {
            let value: f64 = lit.value().parse().map_err(|_| {
                ShaclError::ValidationEngine("floor requires a numeric argument".to_string())
            })?;
            let result = value.floor();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "floor requires a numeric literal argument".to_string(),
            ))
        }
    }

    /// Execute round
    fn round(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let Some(Term::Literal(lit)) = args.get("value") {
            let value: f64 = lit.value().parse().map_err(|_| {
                ShaclError::ValidationEngine("round requires a numeric argument".to_string())
            })?;
            let result = value.round();
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "round requires a numeric literal argument".to_string(),
            ))
        }
    }

    /// Execute contains (string contains substring)
    fn contains(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let (Some(Term::Literal(haystack)), Some(Term::Literal(needle))) =
            (args.get("string"), args.get("substring"))
        {
            let result = haystack.value().contains(needle.value());
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "contains requires string and substring arguments".to_string(),
            ))
        }
    }

    /// Execute starts-with
    fn starts_with(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let (Some(Term::Literal(string)), Some(Term::Literal(prefix))) =
            (args.get("string"), args.get("prefix"))
        {
            let result = string.value().starts_with(prefix.value());
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "startsWith requires string and prefix arguments".to_string(),
            ))
        }
    }

    /// Execute ends-with
    fn ends_with(&self, args: &HashMap<String, Term>) -> Result<FunctionResult> {
        if let (Some(Term::Literal(string)), Some(Term::Literal(suffix))) =
            (args.get("string"), args.get("suffix"))
        {
            let result = string.value().ends_with(suffix.value());
            Ok(FunctionResult::value(Term::Literal(
                Literal::new_typed_literal(
                    result.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean"),
                ),
            )))
        } else {
            Err(ShaclError::ValidationEngine(
                "endsWith requires string and suffix arguments".to_string(),
            ))
        }
    }
}

impl Default for BuiltInFunctionExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionExecutor for BuiltInFunctionExecutor {
    fn execute(
        &self,
        function: &ShaclFunction,
        invocation: &FunctionInvocation,
        _store: &dyn Store,
    ) -> Result<FunctionResult> {
        // Validate arguments
        function.validate_arguments(&invocation.arguments)?;

        // Execute based on function name
        match function.name.as_str() {
            // String functions
            "concat" => self.concat(&invocation.arguments),
            "upperCase" => self.upper_case(&invocation.arguments),
            "lowerCase" => self.lower_case(&invocation.arguments),
            "substring" => self.substring(&invocation.arguments),
            "strLength" => self.str_length(&invocation.arguments),
            "contains" => self.contains(&invocation.arguments),
            "startsWith" => self.starts_with(&invocation.arguments),
            "endsWith" => self.ends_with(&invocation.arguments),
            // Mathematical functions
            "abs" => self.abs(&invocation.arguments),
            "ceil" => self.ceil(&invocation.arguments),
            "floor" => self.floor(&invocation.arguments),
            "round" => self.round(&invocation.arguments),
            _ => Err(ShaclError::UnsupportedOperation(format!(
                "Unknown built-in function: {}",
                function.name
            ))),
        }
    }

    fn name(&self) -> &str {
        "BuiltInFunctionExecutor"
    }

    fn can_execute(&self, function: &ShaclFunction) -> bool {
        matches!(
            function.name.as_str(),
            "concat"
                | "upperCase"
                | "lowerCase"
                | "substring"
                | "strLength"
                | "contains"
                | "startsWith"
                | "endsWith"
                | "abs"
                | "ceil"
                | "floor"
                | "round"
        )
    }
}

/// Function registry for managing SHACL functions
pub struct FunctionRegistry {
    /// Registered functions
    functions: HashMap<String, ShaclFunction>,
    /// Function executors
    executors: Vec<Arc<dyn FunctionExecutor>>,
}

impl FunctionRegistry {
    /// Create a new function registry
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
            executors: Vec::new(),
        };

        // Register built-in executor
        registry.add_executor(Arc::new(BuiltInFunctionExecutor::new()));

        // Register built-in functions
        registry.register_built_in_functions();

        registry
    }

    /// Register built-in SHACL functions
    fn register_built_in_functions(&mut self) {
        // String concatenation
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#concat",
            "concat",
            vec![],
            ReturnType::Single(ParameterType::String),
        ))
        .ok();

        // String uppercase
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#upperCase",
            "upperCase",
            vec![FunctionParameter::required(
                "input",
                ParameterType::String,
                0,
            )],
            ReturnType::Single(ParameterType::String),
        ))
        .ok();

        // String lowercase
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#lowerCase",
            "lowerCase",
            vec![FunctionParameter::required(
                "input",
                ParameterType::String,
                0,
            )],
            ReturnType::Single(ParameterType::String),
        ))
        .ok();

        // Substring
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#substring",
            "substring",
            vec![
                FunctionParameter::required("input", ParameterType::String, 0),
                FunctionParameter::required("start", ParameterType::Integer, 1),
                FunctionParameter::optional("length", ParameterType::Integer, 2, None),
            ],
            ReturnType::Single(ParameterType::String),
        ))
        .ok();

        // String length
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#strLength",
            "strLength",
            vec![FunctionParameter::required(
                "input",
                ParameterType::String,
                0,
            )],
            ReturnType::Single(ParameterType::Integer),
        ))
        .ok();

        // Contains
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#contains",
            "contains",
            vec![
                FunctionParameter::required("string", ParameterType::String, 0),
                FunctionParameter::required("substring", ParameterType::String, 1),
            ],
            ReturnType::Single(ParameterType::Boolean),
        ))
        .ok();

        // Starts with
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#startsWith",
            "startsWith",
            vec![
                FunctionParameter::required("string", ParameterType::String, 0),
                FunctionParameter::required("prefix", ParameterType::String, 1),
            ],
            ReturnType::Single(ParameterType::Boolean),
        ))
        .ok();

        // Ends with
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#endsWith",
            "endsWith",
            vec![
                FunctionParameter::required("string", ParameterType::String, 0),
                FunctionParameter::required("suffix", ParameterType::String, 1),
            ],
            ReturnType::Single(ParameterType::Boolean),
        ))
        .ok();

        // Mathematical functions

        // Absolute value
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#abs",
            "abs",
            vec![FunctionParameter::required(
                "value",
                ParameterType::Decimal,
                0,
            )],
            ReturnType::Single(ParameterType::Decimal),
        ))
        .ok();

        // Ceiling
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#ceil",
            "ceil",
            vec![FunctionParameter::required(
                "value",
                ParameterType::Decimal,
                0,
            )],
            ReturnType::Single(ParameterType::Integer),
        ))
        .ok();

        // Floor
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#floor",
            "floor",
            vec![FunctionParameter::required(
                "value",
                ParameterType::Decimal,
                0,
            )],
            ReturnType::Single(ParameterType::Integer),
        ))
        .ok();

        // Round
        self.register_function(ShaclFunction::new(
            "http://www.w3.org/ns/shacl#round",
            "round",
            vec![FunctionParameter::required(
                "value",
                ParameterType::Decimal,
                0,
            )],
            ReturnType::Single(ParameterType::Integer),
        ))
        .ok();
    }

    /// Register a function
    pub fn register_function(&mut self, function: ShaclFunction) -> Result<()> {
        self.functions.insert(function.id.clone(), function);
        Ok(())
    }

    /// Get a function by ID
    pub fn get_function(&self, id: &str) -> Option<&ShaclFunction> {
        self.functions.get(id)
    }

    /// Add a function executor
    pub fn add_executor(&mut self, executor: Arc<dyn FunctionExecutor>) {
        self.executors.push(executor);
    }

    /// Execute a function
    pub fn execute(
        &self,
        invocation: &FunctionInvocation,
        store: &dyn Store,
    ) -> Result<FunctionResult> {
        // Get function definition
        let function = self.get_function(&invocation.function_id).ok_or_else(|| {
            ShaclError::ValidationEngine(format!("Function not found: {}", invocation.function_id))
        })?;

        // Find executor
        for executor in &self.executors {
            if executor.can_execute(function) {
                return executor.execute(function, invocation, store);
            }
        }

        Err(ShaclError::ValidationEngine(format!(
            "No executor found for function: {}",
            invocation.function_id
        )))
    }

    /// List all registered functions
    pub fn list_functions(&self) -> Vec<&ShaclFunction> {
        self.functions.values().collect()
    }

    /// Get function count
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_parameter_creation() {
        let param = FunctionParameter::required("test", ParameterType::String, 0);
        assert_eq!(param.name, "test");
        assert!(!param.optional);
        assert_eq!(param.order, 0);
    }

    #[test]
    fn test_function_registry_creation() {
        let registry = FunctionRegistry::new();
        assert!(registry.function_count() > 0);
    }

    #[test]
    fn test_built_in_functions_registered() {
        let registry = FunctionRegistry::new();
        assert!(registry
            .get_function("http://www.w3.org/ns/shacl#concat")
            .is_some());
        assert!(registry
            .get_function("http://www.w3.org/ns/shacl#upperCase")
            .is_some());
    }

    #[test]
    fn test_function_validation() {
        let function = ShaclFunction::new(
            "test:func",
            "testFunc",
            vec![FunctionParameter::required(
                "arg1",
                ParameterType::String,
                0,
            )],
            ReturnType::Single(ParameterType::String),
        );

        let mut args = HashMap::new();
        args.insert(
            "arg1".to_string(),
            Term::Literal(Literal::new_simple_literal("test")),
        );

        assert!(function.validate_arguments(&args).is_ok());
    }

    #[test]
    fn test_function_validation_missing_required() {
        let function = ShaclFunction::new(
            "test:func",
            "testFunc",
            vec![FunctionParameter::required(
                "arg1",
                ParameterType::String,
                0,
            )],
            ReturnType::Single(ParameterType::String),
        );

        let args = HashMap::new();
        assert!(function.validate_arguments(&args).is_err());
    }
}
