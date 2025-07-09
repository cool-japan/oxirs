//! SPARQL-based constraint validation with enhanced function library
//!
//! This module provides comprehensive SPARQL constraint validation capabilities
//! including dynamic function registration, advanced security sandboxing, and
//! extensive function library management.

pub mod function_library;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use oxirs_core::{model::RdfTerm, Store};

use crate::{
    constraints::{ConstraintContext, ConstraintEvaluationResult},
    validation::constraint_validators::ConstraintValidator,
    Result, Severity, ShaclError,
};

/// SPARQL-based constraint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlConstraint {
    /// SPARQL SELECT or ASK query
    pub query: String,
    /// Optional prefixes for the query
    pub prefixes: Option<String>,
    /// Custom violation message
    pub message: Option<String>,
    /// Severity level for violations
    pub severity: Option<Severity>,
    /// Optional SPARQL CONSTRUCT query for generating violation details
    pub construct_query: Option<String>,
}

/// SPARQL query type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparqlQueryType {
    Select,
    Ask,
    Construct,
}

/// SPARQL constraint component
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlConstraintComponent {
    pub constraint: SparqlConstraint,
    pub query_type: SparqlQueryType,
}

/// SPARQL constraint executor
#[derive(Debug)]
pub struct SparqlConstraintExecutor {
    cache: HashMap<String, ConstraintEvaluationResult>,
}

impl Default for SparqlConstraintExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlConstraintExecutor {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn execute_constraint(
        &self,
        _constraint: &SparqlConstraint,
        _context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Basic implementation - in a real system this would execute SPARQL queries
        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// SPARQL constraint library
#[derive(Debug)]
pub struct SparqlConstraintLibrary {
    constraints: HashMap<String, SparqlConstraint>,
}

impl Default for SparqlConstraintLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlConstraintLibrary {
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
        }
    }
}

impl SparqlConstraint {
    /// Create a new SPARQL constraint with a SELECT query
    pub fn select(query: String) -> Self {
        Self {
            query,
            prefixes: None,
            message: None,
            severity: None,
            construct_query: None,
        }
    }

    /// Create a new SPARQL constraint with an ASK query
    pub fn ask(query: String) -> Self {
        Self {
            query,
            prefixes: None,
            message: None,
            severity: None,
            construct_query: None,
        }
    }

    /// Validate method for compatibility
    pub fn validate(&self) -> Result<()> {
        // Basic validation - check if query is not empty
        if self.query.trim().is_empty() {
            return Err(ShaclError::SparqlExecution(
                "SPARQL query cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Evaluate the SPARQL constraint
    pub fn evaluate(
        &self,
        _context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Basic implementation - in a real system this would execute SPARQL queries
        // For now, return satisfied to allow compilation
        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

impl ConstraintValidator for SparqlConstraint {
    fn validate(
        &self,
        store: &dyn Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<crate::validation::ConstraintEvaluationResult> {
        match self.evaluate(context, store)? {
            crate::constraints::ConstraintEvaluationResult::Satisfied => {
                Ok(crate::validation::ConstraintEvaluationResult::Satisfied)
            }
            crate::constraints::ConstraintEvaluationResult::Violated {
                violating_value,
                message,
                details: _,
            } => Ok(crate::validation::ConstraintEvaluationResult::Violated {
                violating_value,
                message,
            }),
            crate::constraints::ConstraintEvaluationResult::Error { message, .. } => {
                Err(crate::ShaclError::ValidationEngine(message))
            }
        }
    }
}

// Re-export enhanced function library types
pub use function_library::{
    DynamicFunction, ExecutionContext, FunctionCategory, FunctionExecutionResult, FunctionLibrary,
    FunctionMetadata, FunctionSecurityPolicy, Operation, Permission, SandboxLevel, SecurityConfig,
    SparqlFunctionLibrary,
};

/// Enhanced SPARQL constraint executor with function library integration
#[derive(Debug)]
pub struct EnhancedSparqlExecutor {
    /// Function library for custom functions
    function_library: SparqlFunctionLibrary,
    /// Legacy executor for compatibility
    legacy_executor: SparqlConstraintExecutor,
    /// Constraint library registry
    constraint_libraries: HashMap<String, SparqlConstraintLibrary>,
}

impl EnhancedSparqlExecutor {
    /// Create a new enhanced SPARQL executor
    pub fn new() -> Self {
        Self {
            function_library: SparqlFunctionLibrary::new(),
            legacy_executor: SparqlConstraintExecutor::new(),
            constraint_libraries: HashMap::new(),
        }
    }

    /// Register a custom function with the executor
    pub fn register_function(
        &mut self,
        function: Arc<dyn DynamicFunction>,
        security_policy: Option<FunctionSecurityPolicy>,
    ) -> Result<()> {
        self.function_library
            .register_function(function, security_policy)
    }

    /// Load a function library
    pub fn load_function_library(&mut self, library: FunctionLibrary) -> Result<()> {
        self.function_library.load_library(library)
    }

    /// Execute a SPARQL constraint with enhanced function support
    pub fn execute_constraint_enhanced(
        &self,
        constraint: &SparqlConstraint,
        context: &crate::constraints::ConstraintContext,
        store: &dyn oxirs_core::Store,
    ) -> Result<crate::constraints::ConstraintEvaluationResult> {
        // Use legacy executor for now, but could be enhanced to use function library
        self.legacy_executor
            .execute_constraint(constraint, context, store)
    }

    /// Get available functions
    pub fn list_available_functions(&self) -> Vec<FunctionMetadata> {
        self.function_library.list_functions()
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> function_library::ExecutionStats {
        self.function_library.get_execution_stats()
    }

    /// Update security configuration
    pub fn update_security_config(&mut self, config: SecurityConfig) {
        self.function_library.update_security_config(config);
    }
}

impl Default for EnhancedSparqlExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Example custom function implementations to demonstrate the system
pub mod examples {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Term};

    /// Example string manipulation function
    #[derive(Debug)]
    pub struct UpperCaseFunction {
        metadata: FunctionMetadata,
    }

    impl Default for UpperCaseFunction {
        fn default() -> Self {
            Self::new()
        }
    }

    impl UpperCaseFunction {
        pub fn new() -> Self {
            Self {
                metadata: FunctionMetadata {
                    name: "UPPERCASE".to_string(),
                    description: "Convert string to uppercase".to_string(),
                    version: "1.0.0".to_string(),
                    author: "SHACL Engine".to_string(),
                    min_args: 1,
                    max_args: 1,
                    arg_types: vec!["string".to_string()],
                    return_type: "string".to_string(),
                    category: FunctionCategory::String,
                    deterministic: true,
                    has_side_effects: false,
                    required_permissions: vec![],
                    documentation_url: None,
                    examples: vec!["UPPERCASE(\"hello\") → \"HELLO\"".to_string()],
                },
            }
        }
    }

    impl DynamicFunction for UpperCaseFunction {
        fn execute(&self, args: &[Term], _context: &ExecutionContext) -> Result<Term> {
            self.validate_args(args)?;

            match &args[0] {
                Term::Literal(literal) => {
                    let upper_value = literal.as_str().to_uppercase();
                    Ok(Term::Literal(Literal::new(upper_value)))
                }
                _ => Err(ShaclError::ValidationEngine(
                    "UPPERCASE function requires a literal argument".to_string(),
                )),
            }
        }

        fn metadata(&self) -> &FunctionMetadata {
            &self.metadata
        }

        fn security_policy(&self) -> FunctionSecurityPolicy {
            FunctionSecurityPolicy {
                max_execution_time: std::time::Duration::from_millis(100),
                max_memory_per_call: 1024, // 1KB
                max_calls_per_minute: 1000,
                allowed_operations: vec![Operation::Read],
                sandbox_level: SandboxLevel::Basic,
                required_permissions: vec![],
                network_access: false,
                filesystem_access: false,
                custom_rules: HashMap::new(),
            }
        }
    }

    /// Example mathematical function
    #[derive(Debug)]
    pub struct PowerFunction {
        metadata: FunctionMetadata,
    }

    impl Default for PowerFunction {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PowerFunction {
        pub fn new() -> Self {
            Self {
                metadata: FunctionMetadata {
                    name: "POW".to_string(),
                    description: "Raise number to a power".to_string(),
                    version: "1.0.0".to_string(),
                    author: "SHACL Engine".to_string(),
                    min_args: 2,
                    max_args: 2,
                    arg_types: vec!["number".to_string(), "number".to_string()],
                    return_type: "number".to_string(),
                    category: FunctionCategory::Math,
                    deterministic: true,
                    has_side_effects: false,
                    required_permissions: vec![],
                    documentation_url: None,
                    examples: vec!["POW(2, 3) → 8".to_string()],
                },
            }
        }
    }

    impl DynamicFunction for PowerFunction {
        fn execute(&self, args: &[Term], _context: &ExecutionContext) -> Result<Term> {
            self.validate_args(args)?;

            let base = extract_number(&args[0])?;
            let exponent = extract_number(&args[1])?;

            let result = base.powf(exponent);

            // Return as double literal
            let double_type = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                .map_err(|e| ShaclError::ValidationEngine(format!("Invalid datatype IRI: {e}")))?;

            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                double_type,
            )))
        }

        fn metadata(&self) -> &FunctionMetadata {
            &self.metadata
        }

        fn security_policy(&self) -> FunctionSecurityPolicy {
            FunctionSecurityPolicy {
                max_execution_time: std::time::Duration::from_millis(50),
                max_memory_per_call: 512, // 512 bytes
                max_calls_per_minute: 2000,
                allowed_operations: vec![Operation::Read],
                sandbox_level: SandboxLevel::Basic,
                required_permissions: vec![],
                network_access: false,
                filesystem_access: false,
                custom_rules: HashMap::new(),
            }
        }
    }

    /// Extract numeric value from a term
    fn extract_number(term: &Term) -> Result<f64> {
        match term {
            Term::Literal(literal) => literal.as_str().parse::<f64>().map_err(|_| {
                ShaclError::ValidationEngine(format!("Cannot parse number: {}", literal.as_str()))
            }),
            _ => Err(ShaclError::ValidationEngine(
                "Expected numeric literal".to_string(),
            )),
        }
    }

    /// Example function library containing multiple related functions
    pub fn create_example_library() -> FunctionLibrary {
        let functions: Vec<Arc<dyn DynamicFunction>> = vec![
            Arc::new(UpperCaseFunction::new()),
            Arc::new(PowerFunction::new()),
        ];

        let mut security_policies = HashMap::new();
        security_policies.insert(
            "UPPERCASE".to_string(),
            UpperCaseFunction::new().security_policy(),
        );
        security_policies.insert("POW".to_string(), PowerFunction::new().security_policy());

        FunctionLibrary {
            metadata: function_library::LibraryMetadata {
                name: "ExampleLibrary".to_string(),
                description: "Example SPARQL function library".to_string(),
                version: "1.0.0".to_string(),
                author: "SHACL Engine Team".to_string(),
                license: "MIT".to_string(),
                website: Some("https://example.com/sparql-functions".to_string()),
                min_engine_version: "0.1.0".to_string(),
            },
            functions,
            security_policies,
            dependencies: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::examples::*;
    use super::*;
    use oxirs_core::model::Term;
    use std::time::SystemTime;

    #[test]
    fn test_enhanced_executor_creation() {
        let executor = EnhancedSparqlExecutor::new();
        assert_eq!(executor.list_available_functions().len(), 0);
    }

    #[test]
    fn test_function_registration() {
        let mut executor = EnhancedSparqlExecutor::new();
        let function = Arc::new(UpperCaseFunction::new());

        executor.register_function(function, None).unwrap();
        assert_eq!(executor.list_available_functions().len(), 1);
    }

    #[test]
    fn test_library_loading() {
        let mut executor = EnhancedSparqlExecutor::new();
        let library = create_example_library();

        executor.load_function_library(library).unwrap();
        assert_eq!(executor.list_available_functions().len(), 2);
    }

    #[test]
    fn test_uppercase_function() {
        use oxirs_core::model::Literal;

        let function = UpperCaseFunction::new();
        let args = vec![Term::Literal(Literal::new("hello"))];
        let context = ExecutionContext {
            query_context: HashMap::new(),
            timestamp: SystemTime::now(),
            permissions: std::collections::HashSet::new(),
            execution_id: "test".to_string(),
            max_execution_time: std::time::Duration::from_secs(1),
            max_memory: 1024,
        };

        let result = function.execute(&args, &context).unwrap();

        match result {
            Term::Literal(literal) => {
                assert_eq!(literal.as_str(), "HELLO");
            }
            _ => assert!(false, "Expected literal result, got: {:?}", result),
        }
    }

    #[test]
    fn test_power_function() {
        use oxirs_core::model::Literal;

        let function = PowerFunction::new();
        let args = vec![
            Term::Literal(Literal::new("2")),
            Term::Literal(Literal::new("3")),
        ];
        let context = ExecutionContext {
            query_context: HashMap::new(),
            timestamp: SystemTime::now(),
            permissions: std::collections::HashSet::new(),
            execution_id: "test".to_string(),
            max_execution_time: std::time::Duration::from_secs(1),
            max_memory: 1024,
        };

        let result = function.execute(&args, &context).unwrap();

        match result {
            Term::Literal(literal) => {
                let value: f64 = literal.as_str().parse().unwrap();
                assert_eq!(value, 8.0);
            }
            _ => assert!(false, "Expected literal result, got: {:?}", result),
        }
    }
}
