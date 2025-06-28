//! Extension Framework for Custom Functions and Operators
//!
//! This module provides a comprehensive extension framework for adding custom
//! SPARQL functions, operators, and other query processing capabilities.

use crate::algebra::{Algebra, BinaryOperator, Binding, Expression, Term, UnaryOperator, Variable};
use anyhow::{anyhow, bail, Result};
use oxirs_core::model::NamedNode;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Extension registry for managing custom functions and operators
#[derive(Debug)]
pub struct ExtensionRegistry {
    /// Custom function registry
    pub functions: Arc<RwLock<HashMap<String, Box<dyn CustomFunction>>>>,
    /// Custom operator registry
    pub operators: Arc<RwLock<HashMap<String, Box<dyn CustomOperator>>>>,
    /// Custom aggregate function registry
    pub aggregates: Arc<RwLock<HashMap<String, Box<dyn CustomAggregate>>>>,
    /// Extension plugins
    pub plugins: Arc<RwLock<Vec<Box<dyn ExtensionPlugin>>>>,
    /// Type conversion registry
    pub type_converters: Arc<RwLock<HashMap<String, Box<dyn TypeConverter>>>>,
}

/// Trait for custom SPARQL functions
pub trait CustomFunction: Send + Sync + Debug {
    /// Function name (IRI)
    fn name(&self) -> &str;

    /// Function arity (number of parameters), None for variadic
    fn arity(&self) -> Option<usize>;

    /// Function parameter types
    fn parameter_types(&self) -> Vec<ValueType>;

    /// Function return type
    fn return_type(&self) -> ValueType;

    /// Function documentation
    fn documentation(&self) -> &str;

    /// Execute the function
    fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value>;

    /// Clone this function (for registry operations)
    fn clone_function(&self) -> Box<dyn CustomFunction>;

    /// Validate function call at compile time
    fn validate(&self, args: &[Expression]) -> Result<()> {
        if let Some(expected_arity) = self.arity() {
            if args.len() != expected_arity {
                bail!(
                    "Function {} expects {} arguments, got {}",
                    self.name(),
                    expected_arity,
                    args.len()
                );
            }
        }
        Ok(())
    }

    /// Estimate execution cost
    fn cost_estimate(&self, args: &[Expression]) -> f64 {
        // Default implementation - can be overridden
        100.0 + args.len() as f64 * 10.0
    }

    /// Check if function is deterministic
    fn is_deterministic(&self) -> bool {
        true
    }

    /// Check if function can be pushed down
    fn can_pushdown(&self) -> bool {
        self.is_deterministic()
    }
}

/// Trait for custom operators
pub trait CustomOperator: Send + Sync + Debug {
    /// Operator symbol
    fn symbol(&self) -> &str;

    /// Operator precedence
    fn precedence(&self) -> i32;

    /// Operator associativity
    fn associativity(&self) -> Associativity;

    /// Operator type (binary, unary, etc.)
    fn operator_type(&self) -> OperatorType;

    /// Execute the operator
    fn execute(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
        context: &ExecutionContext,
    ) -> Result<Value>;

    /// Type checking for operator
    fn type_check(
        &self,
        left_type: Option<ValueType>,
        right_type: Option<ValueType>,
    ) -> Result<ValueType>;
}

/// Trait for custom aggregate functions
pub trait CustomAggregate: Send + Sync + Debug {
    /// Aggregate function name
    fn name(&self) -> &str;

    /// Initialize aggregate state
    fn init(&self) -> Box<dyn AggregateState>;

    /// Check if supports DISTINCT
    fn supports_distinct(&self) -> bool {
        true
    }

    /// Documentation
    fn documentation(&self) -> &str;
}

/// State for aggregate functions
pub trait AggregateState: Send + Sync + Debug {
    /// Add value to aggregate
    fn add(&mut self, value: &Value) -> Result<()>;

    /// Get final result
    fn result(&self) -> Result<Value>;

    /// Reset state
    fn reset(&mut self);

    /// Clone state
    fn clone_state(&self) -> Box<dyn AggregateState>;
}

/// Extension plugin trait for complex extensions
pub trait ExtensionPlugin: Send + Sync + Debug {
    /// Plugin name
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Plugin dependencies
    fn dependencies(&self) -> Vec<String>;

    /// Initialize plugin
    fn initialize(&mut self, registry: &mut ExtensionRegistry) -> Result<()>;

    /// Shutdown plugin
    fn shutdown(&mut self) -> Result<()>;

    /// Plugin metadata
    fn metadata(&self) -> PluginMetadata;
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub license: String,
    pub homepage: Option<String>,
    pub repository: Option<String>,
}

/// Type converter trait for custom types
pub trait TypeConverter: Send + Sync + Debug {
    /// Source type
    fn from_type(&self) -> &str;

    /// Target type
    fn to_type(&self) -> &str;

    /// Convert value
    fn convert(&self, value: &Value) -> Result<Value>;

    /// Check if conversion is possible
    fn can_convert(&self, value: &Value) -> bool;
}

/// Value types in the extension system
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Duration,
    Iri,
    BlankNode,
    Literal,
    Custom(String),
    List(Box<ValueType>),
    Optional(Box<ValueType>),
    Union(Vec<ValueType>),
}

/// Runtime values in the extension system
#[derive(Debug)]
pub enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    DateTime(chrono::DateTime<chrono::Utc>),
    Duration(chrono::Duration),
    Iri(String),
    BlankNode(String),
    Literal {
        value: String,
        language: Option<String>,
        datatype: Option<String>,
    },
    List(Vec<Value>),
    Null,
    Custom {
        type_name: String,
        data: Box<dyn Any + Send + Sync>,
    },
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::String(s) => Value::String(s.clone()),
            Value::Integer(i) => Value::Integer(*i),
            Value::Float(f) => Value::Float(*f),
            Value::Boolean(b) => Value::Boolean(*b),
            Value::DateTime(dt) => Value::DateTime(*dt),
            Value::Duration(d) => Value::Duration(*d),
            Value::Iri(iri) => Value::Iri(iri.clone()),
            Value::BlankNode(id) => Value::BlankNode(id.clone()),
            Value::Literal {
                value,
                language,
                datatype,
            } => Value::Literal {
                value: value.clone(),
                language: language.clone(),
                datatype: datatype.clone(),
            },
            Value::List(list) => Value::List(list.clone()),
            Value::Null => Value::Null,
            Value::Custom { type_name, .. } => {
                // Cannot clone arbitrary Any types, return a placeholder
                Value::String(format!("Custom({})", type_name))
            }
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::DateTime(a), Value::DateTime(b)) => a == b,
            (Value::Duration(a), Value::Duration(b)) => a == b,
            (Value::Iri(a), Value::Iri(b)) => a == b,
            (Value::BlankNode(a), Value::BlankNode(b)) => a == b,
            (
                Value::Literal {
                    value: v1,
                    language: l1,
                    datatype: d1,
                },
                Value::Literal {
                    value: v2,
                    language: l2,
                    datatype: d2,
                },
            ) => v1 == v2 && l1 == l2 && d1 == d2,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Null, Value::Null) => true,
            (Value::Custom { type_name: t1, .. }, Value::Custom { type_name: t2, .. }) => t1 == t2,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            (Value::Integer(a), Value::Integer(b)) => a.partial_cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.partial_cmp(b),
            (Value::DateTime(a), Value::DateTime(b)) => a.partial_cmp(b),
            (Value::Duration(a), Value::Duration(b)) => a.partial_cmp(b),
            (Value::Iri(a), Value::Iri(b)) => a.partial_cmp(b),
            (Value::BlankNode(a), Value::BlankNode(b)) => a.partial_cmp(b),
            (
                Value::Literal {
                    value: v1,
                    language: l1,
                    datatype: d1,
                },
                Value::Literal {
                    value: v2,
                    language: l2,
                    datatype: d2,
                },
            ) => match v1.partial_cmp(v2) {
                Some(Ordering::Equal) => match l1.partial_cmp(l2) {
                    Some(Ordering::Equal) => d1.partial_cmp(d2),
                    other => other,
                },
                other => other,
            },
            (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) => Some(Ordering::Less),
            (_, Value::Null) => Some(Ordering::Greater),
            _ => None, // Incomparable types
        }
    }
}

/// Operator associativity
#[derive(Debug, Clone, PartialEq)]
pub enum Associativity {
    Left,
    Right,
    None,
}

/// Operator types
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorType {
    Binary,
    Unary,
    Ternary,
}

/// Execution context for extensions
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub variables: HashMap<Variable, Term>,
    pub namespaces: HashMap<String, String>,
    pub base_iri: Option<String>,
    pub dataset_context: Option<String>,
    pub query_time: chrono::DateTime<chrono::Utc>,
    pub optimization_level: OptimizationLevel,
    pub memory_limit: Option<usize>,
    pub time_limit: Option<std::time::Duration>,
}

/// Optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

impl ExtensionRegistry {
    pub fn new() -> Self {
        Self {
            functions: Arc::new(RwLock::new(HashMap::new())),
            operators: Arc::new(RwLock::new(HashMap::new())),
            aggregates: Arc::new(RwLock::new(HashMap::new())),
            plugins: Arc::new(RwLock::new(Vec::new())),
            type_converters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a custom function
    pub fn register_function<F>(&self, function: F) -> Result<()>
    where
        F: CustomFunction + 'static,
    {
        let name = function.name().to_string();
        let mut functions = self
            .functions
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on functions"))?;
        functions.insert(name, Box::new(function));
        Ok(())
    }

    /// Register a custom operator
    pub fn register_operator<O>(&self, operator: O) -> Result<()>
    where
        O: CustomOperator + 'static,
    {
        let symbol = operator.symbol().to_string();
        let mut operators = self
            .operators
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on operators"))?;
        operators.insert(symbol, Box::new(operator));
        Ok(())
    }

    /// Register a custom aggregate function
    pub fn register_aggregate<A>(&self, aggregate: A) -> Result<()>
    where
        A: CustomAggregate + 'static,
    {
        let name = aggregate.name().to_string();
        let mut aggregates = self
            .aggregates
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on aggregates"))?;
        aggregates.insert(name, Box::new(aggregate));
        Ok(())
    }

    /// Register an extension plugin
    pub fn register_plugin<P>(&mut self, mut plugin: P) -> Result<()>
    where
        P: ExtensionPlugin + 'static,
    {
        // Initialize plugin
        plugin.initialize(self)?;

        let mut plugins = self
            .plugins
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on plugins"))?;
        plugins.push(Box::new(plugin));
        Ok(())
    }

    /// Register a type converter
    pub fn register_type_converter<T>(&self, converter: T) -> Result<()>
    where
        T: TypeConverter + 'static,
    {
        let key = format!("{}:{}", converter.from_type(), converter.to_type());
        let mut converters = self
            .type_converters
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on type converters"))?;
        converters.insert(key, Box::new(converter));
        Ok(())
    }

    /// Get function by name
    pub fn get_function(&self, name: &str) -> Result<Option<Box<dyn CustomFunction>>> {
        let functions = self
            .functions
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on functions"))?;
        Ok(functions.get(name).map(|f| f.clone_function()))
    }

    /// Check if function exists
    pub fn has_function(&self, name: &str) -> Result<bool> {
        let functions = self
            .functions
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on functions"))?;
        Ok(functions.contains_key(name))
    }

    /// Check if operator exists
    pub fn has_operator(&self, symbol: &str) -> Result<bool> {
        let operators = self
            .operators
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on operators"))?;
        Ok(operators.contains_key(symbol))
    }

    /// Check if aggregate exists
    pub fn has_aggregate(&self, name: &str) -> Result<bool> {
        let aggregates = self
            .aggregates
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on aggregates"))?;
        Ok(aggregates.contains_key(name))
    }

    /// Execute a function by name
    pub fn execute_function(
        &self,
        name: &str,
        args: &[Value],
        context: &ExecutionContext,
    ) -> Result<Value> {
        let functions = self
            .functions
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on functions"))?;

        if let Some(func) = functions.get(name) {
            func.execute(args, context)
        } else {
            Err(anyhow!("Function '{}' not found", name))
        }
    }

    /// Execute an operator by symbol
    pub fn execute_operator(
        &self,
        symbol: &str,
        left: Option<&Value>,
        right: Option<&Value>,
        context: &ExecutionContext,
    ) -> Result<Value> {
        let operators = self
            .operators
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on operators"))?;

        if let Some(op) = operators.get(symbol) {
            op.execute(left, right, context)
        } else {
            Err(anyhow!("Operator '{}' not found", symbol))
        }
    }

    /// Create aggregate state by name
    pub fn create_aggregate_state(&self, name: &str) -> Result<Box<dyn AggregateState>> {
        let aggregates = self
            .aggregates
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on aggregates"))?;

        if let Some(agg) = aggregates.get(name) {
            Ok(agg.init())
        } else {
            Err(anyhow!("Aggregate '{}' not found", name))
        }
    }

    /// Convert value from one type to another
    pub fn convert_value(&self, value: &Value, target_type: &str) -> Result<Value> {
        let source_type = value.type_name();
        let key = format!("{}:{}", source_type, target_type);

        let converters = self
            .type_converters
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on type converters"))?;

        if let Some(converter) = converters.get(&key) {
            converter.convert(value)
        } else {
            // Try built-in conversions
            self.builtin_convert(value, target_type)
        }
    }

    /// Built-in type conversions
    fn builtin_convert(&self, value: &Value, target_type: &str) -> Result<Value> {
        match (value, target_type) {
            (Value::String(s), "integer") => s
                .parse::<i64>()
                .map(Value::Integer)
                .map_err(|_| anyhow!("Cannot convert '{}' to integer", s)),
            (Value::String(s), "float") => s
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| anyhow!("Cannot convert '{}' to float", s)),
            (Value::Integer(i), "string") => Ok(Value::String(i.to_string())),
            (Value::Float(f), "string") => Ok(Value::String(f.to_string())),
            (Value::Boolean(b), "string") => Ok(Value::String(b.to_string())),
            _ => bail!(
                "No conversion available from {} to {}",
                value.type_name(),
                target_type
            ),
        }
    }

    /// List all registered functions
    pub fn list_functions(&self) -> Result<Vec<String>> {
        let functions = self
            .functions
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on functions"))?;
        Ok(functions.keys().cloned().collect())
    }

    /// List all registered operators
    pub fn list_operators(&self) -> Result<Vec<String>> {
        let operators = self
            .operators
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on operators"))?;
        Ok(operators.keys().cloned().collect())
    }

    /// Validate extension compatibility
    pub fn validate_extensions(&self) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        // Check plugin dependencies
        let plugins = self
            .plugins
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on plugins"))?;

        for plugin in plugins.iter() {
            for dep in plugin.dependencies() {
                let found = plugins.iter().any(|p| p.name() == dep);
                if !found {
                    errors.push(format!(
                        "Plugin '{}' missing dependency '{}'",
                        plugin.name(),
                        dep
                    ));
                }
            }
        }

        Ok(errors)
    }
}

impl Default for ExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Value {
    /// Get type name of value
    pub fn type_name(&self) -> &str {
        match self {
            Value::String(_) => "string",
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Boolean(_) => "boolean",
            Value::DateTime(_) => "datetime",
            Value::Duration(_) => "duration",
            Value::Iri(_) => "iri",
            Value::BlankNode(_) => "bnode",
            Value::Literal { .. } => "literal",
            Value::List(_) => "list",
            Value::Null => "null",
            Value::Custom { type_name, .. } => type_name,
        }
    }

    /// Convert to Term
    pub fn to_term(&self) -> Result<Term> {
        match self {
            Value::String(s) => Ok(Term::Literal(crate::algebra::Literal {
                value: s.clone(),
                language: None,
                datatype: None,
            })),
            Value::Iri(iri) => Ok(Term::Iri(NamedNode::new_unchecked(iri.clone()))),
            Value::BlankNode(id) => Ok(Term::BlankNode(id.clone())),
            Value::Literal {
                value,
                language,
                datatype,
            } => Ok(Term::Literal(crate::algebra::Literal {
                value: value.clone(),
                language: language.clone(),
                datatype: datatype
                    .as_ref()
                    .map(|dt| NamedNode::new_unchecked(dt.clone())),
            })),
            _ => bail!("Cannot convert {} to Term", self.type_name()),
        }
    }

    /// Create from Term
    pub fn from_term(term: &Term) -> Self {
        match term {
            Term::Iri(iri) => Value::Iri(iri.as_str().to_string()),
            Term::BlankNode(id) => Value::BlankNode(id.clone()),
            Term::Literal(lit) => Value::Literal {
                value: lit.value.clone(),
                language: lit.language.clone(),
                datatype: lit.datatype.as_ref().map(|dt| dt.as_str().to_string()),
            },
            Term::Variable(var) => Value::String(format!("?{}", var)),
        }
    }
}

/// Macro for easy function registration
#[macro_export]
macro_rules! register_function {
    ($registry:expr, $name:expr, $params:expr, $return_type:expr, $body:expr) => {{
        #[derive(Debug, Clone)]
        struct GeneratedFunction {
            name: String,
            params: Vec<ValueType>,
            return_type: ValueType,
            body: fn(&[Value], &ExecutionContext) -> Result<Value>,
        }

        impl CustomFunction for GeneratedFunction {
            fn name(&self) -> &str {
                &self.name
            }
            fn arity(&self) -> Option<usize> {
                Some(self.params.len())
            }
            fn parameter_types(&self) -> Vec<ValueType> {
                self.params.clone()
            }
            fn return_type(&self) -> ValueType {
                self.return_type.clone()
            }
            fn documentation(&self) -> &str {
                "Generated function"
            }
            fn clone_function(&self) -> Box<dyn CustomFunction> {
                Box::new(self.clone())
            }

            fn execute(&self, args: &[Value], context: &ExecutionContext) -> Result<Value> {
                (self.body)(args, context)
            }
        }

        let func = GeneratedFunction {
            name: $name.to_string(),
            params: $params,
            return_type: $return_type,
            body: $body,
        };

        $registry.register_function(func)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestFunction;

    impl CustomFunction for TestFunction {
        fn name(&self) -> &str {
            "http://example.org/test"
        }
        fn arity(&self) -> Option<usize> {
            Some(2)
        }
        fn parameter_types(&self) -> Vec<ValueType> {
            vec![ValueType::Integer, ValueType::Integer]
        }
        fn return_type(&self) -> ValueType {
            ValueType::Integer
        }
        fn documentation(&self) -> &str {
            "Test function that adds two integers"
        }
        fn clone_function(&self) -> Box<dyn CustomFunction> {
            Box::new(self.clone())
        }

        fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
            if args.len() != 2 {
                bail!("Expected 2 arguments, got {}", args.len());
            }

            match (&args[0], &args[1]) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
                _ => bail!("Expected integer arguments"),
            }
        }
    }

    #[test]
    fn test_function_registration() {
        let registry = ExtensionRegistry::new();
        let func = TestFunction;

        assert!(registry.register_function(func).is_ok());
        assert!(registry
            .get_function("http://example.org/test")
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_function_execution() {
        let func = TestFunction;
        let args = vec![Value::Integer(5), Value::Integer(3)];
        let context = ExecutionContext {
            variables: HashMap::new(),
            namespaces: HashMap::new(),
            base_iri: None,
            dataset_context: None,
            query_time: chrono::Utc::now(),
            optimization_level: OptimizationLevel::Basic,
            memory_limit: None,
            time_limit: None,
        };

        let result = func.execute(&args, &context).unwrap();
        assert_eq!(result, Value::Integer(8));
    }
}
