//! Stored Procedures for SPARQL
//!
//! This module implements a procedure framework for SPARQL query extensions,
//! allowing custom stored procedures to be registered and executed.
//!
//! Based on Apache Jena ARQ's procedure framework.

use crate::algebra::{Solution, Term, Variable};
use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Arguments to a procedure call
#[derive(Debug, Clone)]
pub struct ProcedureArgs {
    /// Positional arguments
    pub args: Vec<Term>,
    /// Named arguments (optional)
    pub named_args: HashMap<String, Term>,
}

impl ProcedureArgs {
    /// Create with positional arguments only
    pub fn new(args: Vec<Term>) -> Self {
        Self {
            args,
            named_args: HashMap::new(),
        }
    }

    /// Create with named arguments
    pub fn with_named_args(args: Vec<Term>, named_args: HashMap<String, Term>) -> Self {
        Self { args, named_args }
    }

    /// Get positional argument by index
    pub fn get(&self, index: usize) -> Option<&Term> {
        self.args.get(index)
    }

    /// Get named argument
    pub fn get_named(&self, name: &str) -> Option<&Term> {
        self.named_args.get(name)
    }

    /// Number of positional arguments
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Check if arguments are empty
    pub fn is_empty(&self) -> bool {
        self.args.is_empty() && self.named_args.is_empty()
    }
}

/// Execution context for procedures
#[derive(Clone)]
pub struct ProcedureContext {
    /// Current variable bindings
    pub bindings: HashMap<Variable, Term>,
    /// Procedure-specific metadata
    pub metadata: HashMap<String, String>,
}

impl ProcedureContext {
    /// Create new context
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create with existing bindings
    pub fn with_bindings(bindings: HashMap<Variable, Term>) -> Self {
        Self {
            bindings,
            metadata: HashMap::new(),
        }
    }

    /// Get binding for a variable
    pub fn get_binding(&self, var: &Variable) -> Option<&Term> {
        self.bindings.get(var)
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

impl Default for ProcedureContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of procedure execution
#[derive(Debug, Clone)]
pub enum ProcedureResult {
    /// Single solution (binding)
    Single(Solution),
    /// Multiple solutions
    Multiple(Vec<Solution>),
    /// No results (side effects only)
    Empty,
    /// Status code with optional message
    Status { code: i32, message: Option<String> },
}

impl ProcedureResult {
    /// Convert to iterator of solutions
    pub fn into_solutions(self) -> Vec<Solution> {
        match self {
            Self::Single(solution) => vec![solution],
            Self::Multiple(solutions) => solutions,
            Self::Empty | Self::Status { .. } => Vec::new(),
        }
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Multiple(v) => v.is_empty(),
            _ => false,
        }
    }

    /// Check if result is successful
    pub fn is_success(&self) -> bool {
        match self {
            Self::Status { code, .. } => *code == 0,
            Self::Empty => true,
            Self::Single(_) | Self::Multiple(_) => true,
        }
    }
}

/// Stored procedure trait
pub trait Procedure: Send + Sync {
    /// Get the URI of this procedure
    fn uri(&self) -> &str;

    /// Get the name of this procedure (short name)
    fn name(&self) -> &str {
        // Default: extract last segment of URI
        self.uri().rsplit('/').next().unwrap_or(self.uri())
    }

    /// Get documentation for this procedure
    fn documentation(&self) -> &str {
        "No documentation available"
    }

    /// Validate procedure call during query planning
    ///
    /// Called once during query planning to validate arguments
    fn build(
        &self,
        _proc_id: &str,
        _args: &ProcedureArgs,
        _context: &ProcedureContext,
    ) -> Result<()> {
        // Default: no validation
        Ok(())
    }

    /// Execute the procedure
    ///
    /// Takes input bindings and returns output bindings/solutions
    fn execute(
        &self,
        proc_id: &str,
        args: &ProcedureArgs,
        context: &ProcedureContext,
    ) -> Result<ProcedureResult>;

    /// Check if this procedure has side effects
    fn has_side_effects(&self) -> bool {
        false
    }

    /// Check if this procedure is deterministic
    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Factory for creating procedure instances
pub trait ProcedureFactory: Send + Sync {
    /// Create a new instance of the procedure
    fn create(&self) -> Result<Box<dyn Procedure>>;

    /// Get the URI this factory handles
    fn uri(&self) -> &str;
}

/// Simple procedure factory that creates new instances
pub struct SimpleProcedureFactory<P: Procedure + Default + 'static> {
    uri: String,
    _phantom: std::marker::PhantomData<P>,
}

impl<P: Procedure + Default + 'static> SimpleProcedureFactory<P> {
    pub fn new(uri: String) -> Self {
        Self {
            uri,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P: Procedure + Default + 'static> ProcedureFactory for SimpleProcedureFactory<P> {
    fn create(&self) -> Result<Box<dyn Procedure>> {
        Ok(Box::new(P::default()))
    }

    fn uri(&self) -> &str {
        &self.uri
    }
}

/// Registry for stored procedures
pub struct ProcedureRegistry {
    procedures: DashMap<String, Arc<dyn ProcedureFactory>>,
}

impl ProcedureRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            procedures: DashMap::new(),
        }
    }

    /// Register a procedure
    pub fn register<P: Procedure + 'static>(&self, procedure: P) -> Result<()> {
        let uri = procedure.uri().to_string();
        let factory = Arc::new(InstanceFactory {
            instance: Arc::new(procedure),
        });
        self.procedures.insert(uri, factory);
        Ok(())
    }

    /// Register a procedure factory
    pub fn register_factory<F: ProcedureFactory + 'static>(&self, factory: F) -> Result<()> {
        let uri = factory.uri().to_string();
        self.procedures.insert(uri, Arc::new(factory));
        Ok(())
    }

    /// Get a procedure by URI
    pub fn get(&self, uri: &str) -> Option<Result<Box<dyn Procedure>>> {
        self.procedures.get(uri).map(|factory| factory.create())
    }

    /// Check if a URI is registered
    pub fn is_registered(&self, uri: &str) -> bool {
        self.procedures.contains_key(uri)
    }

    /// Get all registered URIs
    pub fn registered_uris(&self) -> Vec<String> {
        self.procedures
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Create a registry with standard procedures
    pub fn with_standard_procedures() -> Result<Self> {
        let registry = Self::new();
        register_standard_procedures(&registry)?;
        Ok(registry)
    }
}

impl Default for ProcedureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory that returns the same instance (for stateless procedures)
struct InstanceFactory {
    instance: Arc<dyn Procedure>,
}

impl ProcedureFactory for InstanceFactory {
    fn create(&self) -> Result<Box<dyn Procedure>> {
        // Clone the Arc and wrap in a new box
        Ok(Box::new(ArcProcedure {
            inner: Arc::clone(&self.instance),
        }))
    }

    fn uri(&self) -> &str {
        self.instance.uri()
    }
}

/// Wrapper to make Arc<dyn Procedure> implement Procedure
struct ArcProcedure {
    inner: Arc<dyn Procedure>,
}

impl Procedure for ArcProcedure {
    fn uri(&self) -> &str {
        self.inner.uri()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn documentation(&self) -> &str {
        self.inner.documentation()
    }

    fn build(&self, proc_id: &str, args: &ProcedureArgs, context: &ProcedureContext) -> Result<()> {
        self.inner.build(proc_id, args, context)
    }

    fn execute(
        &self,
        proc_id: &str,
        args: &ProcedureArgs,
        context: &ProcedureContext,
    ) -> Result<ProcedureResult> {
        self.inner.execute(proc_id, args, context)
    }

    fn has_side_effects(&self) -> bool {
        self.inner.has_side_effects()
    }

    fn is_deterministic(&self) -> bool {
        self.inner.is_deterministic()
    }
}

/// Register standard procedures
pub fn register_standard_procedures(registry: &ProcedureRegistry) -> Result<()> {
    // Example procedures
    registry.register(NoOpProcedure)?;
    registry.register(EchoProcedure)?;
    Ok(())
}

// ============================================================================
// Standard Procedures
// ============================================================================

/// No-op procedure (for testing)
#[derive(Debug, Clone, Default)]
struct NoOpProcedure;

impl Procedure for NoOpProcedure {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/procedure#noop"
    }

    fn name(&self) -> &str {
        "noop"
    }

    fn documentation(&self) -> &str {
        "No-operation procedure that returns empty result"
    }

    fn execute(
        &self,
        _proc_id: &str,
        _args: &ProcedureArgs,
        _context: &ProcedureContext,
    ) -> Result<ProcedureResult> {
        Ok(ProcedureResult::Empty)
    }
}

/// Echo procedure (returns input as output)
#[derive(Debug, Clone, Default)]
struct EchoProcedure;

impl Procedure for EchoProcedure {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/procedure#echo"
    }

    fn name(&self) -> &str {
        "echo"
    }

    fn documentation(&self) -> &str {
        "Echo procedure that returns input arguments as a solution"
    }

    fn build(
        &self,
        _proc_id: &str,
        args: &ProcedureArgs,
        _context: &ProcedureContext,
    ) -> Result<()> {
        if args.is_empty() {
            bail!("echo procedure requires at least one argument");
        }
        Ok(())
    }

    fn execute(
        &self,
        _proc_id: &str,
        args: &ProcedureArgs,
        _context: &ProcedureContext,
    ) -> Result<ProcedureResult> {
        // Create a binding with echoed values
        let mut binding = HashMap::new();

        for (i, term) in args.args.iter().enumerate() {
            let var_name = format!("arg{}", i);
            let var = Variable::new(&var_name)
                .map_err(|e| anyhow!("Failed to create variable: {}", e))?;
            binding.insert(var, term.clone());
        }

        // Solution is Vec<Binding>, so wrap the binding in a Vec
        let solution = vec![binding];

        Ok(ProcedureResult::Single(solution))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procedure_args_basic() {
        use oxirs_core::model::NamedNode;

        let iri1 = Term::Iri(NamedNode::new("http://example.org/1").unwrap());
        let iri2 = Term::Iri(NamedNode::new("http://example.org/2").unwrap());

        let args = ProcedureArgs::new(vec![iri1.clone(), iri2.clone()]);

        assert_eq!(args.len(), 2);
        assert!(!args.is_empty());
        assert_eq!(args.get(0), Some(&iri1));
        assert_eq!(args.get(1), Some(&iri2));
        assert_eq!(args.get(2), None);
    }

    #[test]
    fn test_procedure_args_named() {
        use oxirs_core::model::NamedNode;

        let iri = Term::Iri(NamedNode::new("http://example.org/test").unwrap());
        let mut named = HashMap::new();
        named.insert("key1".to_string(), iri.clone());

        let args = ProcedureArgs::with_named_args(vec![], named);

        assert_eq!(args.len(), 0);
        assert_eq!(args.get_named("key1"), Some(&iri));
        assert_eq!(args.get_named("key2"), None);
    }

    #[test]
    fn test_procedure_context() {
        let mut context = ProcedureContext::new();

        assert!(context.metadata.is_empty());

        context.set_metadata("key".to_string(), "value".to_string());
        assert_eq!(context.get_metadata("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_procedure_result_into_solutions() {
        // Solution is Vec<Binding>, so create an empty solution
        let solution: Solution = Vec::new();
        let result = ProcedureResult::Single(solution.clone());
        assert_eq!(result.into_solutions().len(), 1);

        let multiple = ProcedureResult::Multiple(vec![solution.clone(), solution]);
        assert_eq!(multiple.into_solutions().len(), 2);

        let empty = ProcedureResult::Empty;
        assert_eq!(empty.into_solutions().len(), 0);
    }

    #[test]
    fn test_procedure_result_is_success() {
        assert!(ProcedureResult::Empty.is_success());
        assert!(ProcedureResult::Single(Vec::new()).is_success());
        assert!(ProcedureResult::Status {
            code: 0,
            message: None
        }
        .is_success());
        assert!(!ProcedureResult::Status {
            code: 1,
            message: Some("error".to_string())
        }
        .is_success());
    }

    #[test]
    fn test_registry_registration() {
        let registry = ProcedureRegistry::new();
        registry.register(NoOpProcedure).unwrap();

        assert!(registry.is_registered("http://jena.apache.org/ARQ/procedure#noop"));
        assert!(!registry.is_registered("http://example.org/unknown"));
    }

    #[test]
    fn test_registry_with_standard_procedures() {
        let registry = ProcedureRegistry::with_standard_procedures().unwrap();

        assert!(registry.is_registered("http://jena.apache.org/ARQ/procedure#noop"));
        assert!(registry.is_registered("http://jena.apache.org/ARQ/procedure#echo"));

        let uris = registry.registered_uris();
        assert_eq!(uris.len(), 2);
    }

    #[test]
    fn test_noop_procedure() {
        let proc = NoOpProcedure;
        let args = ProcedureArgs::new(vec![]);
        let context = ProcedureContext::new();

        let result = proc.execute(proc.uri(), &args, &context).unwrap();
        assert!(result.is_empty());
        assert!(result.is_success());
    }

    #[test]
    fn test_echo_procedure() {
        use oxirs_core::model::NamedNode;

        let proc = EchoProcedure;
        let iri = Term::Iri(NamedNode::new("http://example.org/test").unwrap());
        let args = ProcedureArgs::new(vec![iri.clone()]);
        let context = ProcedureContext::new();

        let result = proc.execute(proc.uri(), &args, &context).unwrap();

        match result {
            ProcedureResult::Single(solution) => {
                // Solution is Vec<Binding>, check first binding
                assert_eq!(solution.len(), 1);
                assert_eq!(solution[0].len(), 1);
            }
            _ => panic!("Expected Single result"),
        }
    }

    #[test]
    fn test_echo_procedure_validation() {
        let proc = EchoProcedure;
        let empty_args = ProcedureArgs::new(vec![]);
        let context = ProcedureContext::new();

        let result = proc.build(proc.uri(), &empty_args, &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_procedure_name_extraction() {
        let proc = NoOpProcedure;
        assert_eq!(proc.name(), "noop");

        let echo = EchoProcedure;
        assert_eq!(echo.name(), "echo");
    }

    #[test]
    fn test_registry_get_procedure() {
        let registry = ProcedureRegistry::with_standard_procedures().unwrap();

        let proc_result = registry.get("http://jena.apache.org/ARQ/procedure#noop");
        assert!(proc_result.is_some());

        let proc = proc_result.unwrap().unwrap();
        assert_eq!(proc.uri(), "http://jena.apache.org/ARQ/procedure#noop");
    }
}
