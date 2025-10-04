//! Property Functions (Magic Predicates) for SPARQL
//!
//! Property functions are special predicates that are evaluated procedurally
//! rather than by triple pattern matching. They enable powerful query patterns
//! like list operations, container access, and custom evaluation logic.
//!
//! Based on Apache Jena ARQ's property function framework.

use crate::algebra::{Term as AlgebraTerm, Variable};
use crate::executor::Dataset;
use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Argument to a property function (subject or object position)
#[derive(Debug, Clone, PartialEq)]
pub enum PropFuncArg {
    /// Single node argument
    Node(AlgebraTerm),
    /// List of nodes
    List(Vec<AlgebraTerm>),
}

impl PropFuncArg {
    /// Create from a single node
    pub fn node(term: AlgebraTerm) -> Self {
        Self::Node(term)
    }

    /// Create from a list of nodes
    pub fn list(terms: Vec<AlgebraTerm>) -> Self {
        Self::List(terms)
    }

    /// Check if this is a single node
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Node(_))
    }

    /// Check if this is a list
    pub fn is_list(&self) -> bool {
        matches!(self, Self::List(_))
    }

    /// Get the node (if single node argument)
    pub fn as_node(&self) -> Option<&AlgebraTerm> {
        match self {
            Self::Node(term) => Some(term),
            _ => None,
        }
    }

    /// Get the list (if list argument)
    pub fn as_list(&self) -> Option<&[AlgebraTerm]> {
        match self {
            Self::List(terms) => Some(terms),
            _ => None,
        }
    }

    /// Get list size (returns -1 if not a list)
    pub fn list_size(&self) -> isize {
        match self {
            Self::List(terms) => terms.len() as isize,
            _ => -1,
        }
    }

    /// Check if the argument contains a variable
    pub fn has_variable(&self) -> bool {
        match self {
            Self::Node(term) => matches!(term, AlgebraTerm::Variable(_)),
            Self::List(terms) => terms.iter().any(|t| matches!(t, AlgebraTerm::Variable(_))),
        }
    }

    /// Get all variables in this argument
    pub fn variables(&self) -> Vec<Variable> {
        match self {
            Self::Node(AlgebraTerm::Variable(v)) => vec![v.clone()],
            Self::List(terms) => terms
                .iter()
                .filter_map(|t| match t {
                    AlgebraTerm::Variable(v) => Some(v.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Execution context for property functions
#[derive(Clone)]
pub struct PropertyFunctionContext {
    /// Current variable bindings
    pub bindings: HashMap<Variable, AlgebraTerm>,
    /// Reference to the dataset
    pub dataset: Arc<dyn Dataset>,
}

impl PropertyFunctionContext {
    /// Create new context
    pub fn new(dataset: Arc<dyn Dataset>) -> Self {
        Self {
            bindings: HashMap::new(),
            dataset,
        }
    }

    /// Create with existing bindings
    pub fn with_bindings(
        dataset: Arc<dyn Dataset>,
        bindings: HashMap<Variable, AlgebraTerm>,
    ) -> Self {
        Self { bindings, dataset }
    }

    /// Get binding for a variable
    pub fn get_binding(&self, var: &Variable) -> Option<&AlgebraTerm> {
        self.bindings.get(var)
    }

    /// Substitute variables in an argument with current bindings
    pub fn substitute(&self, arg: &PropFuncArg) -> PropFuncArg {
        match arg {
            PropFuncArg::Node(term) => match term {
                AlgebraTerm::Variable(v) => {
                    if let Some(bound) = self.get_binding(v) {
                        PropFuncArg::Node(bound.clone())
                    } else {
                        arg.clone()
                    }
                }
                _ => arg.clone(),
            },
            PropFuncArg::List(terms) => {
                let substituted: Vec<AlgebraTerm> = terms
                    .iter()
                    .map(|term| match term {
                        AlgebraTerm::Variable(v) => {
                            self.get_binding(v).cloned().unwrap_or_else(|| term.clone())
                        }
                        _ => term.clone(),
                    })
                    .collect();
                PropFuncArg::List(substituted)
            }
        }
    }
}

/// Result of property function execution
#[derive(Debug, Clone)]
pub enum PropertyFunctionResult {
    /// Single solution (binding extension)
    Single(HashMap<Variable, AlgebraTerm>),
    /// Multiple solutions
    Multiple(Vec<HashMap<Variable, AlgebraTerm>>),
    /// Boolean result (for testing)
    Boolean(bool),
    /// No results
    Empty,
}

impl PropertyFunctionResult {
    /// Convert to iterator of solutions
    pub fn into_solutions(self) -> Vec<HashMap<Variable, AlgebraTerm>> {
        match self {
            Self::Single(bindings) => vec![bindings],
            Self::Multiple(solutions) => solutions,
            Self::Boolean(true) => vec![HashMap::new()],
            Self::Boolean(false) | Self::Empty => Vec::new(),
        }
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Empty | Self::Boolean(false) => true,
            Self::Multiple(v) => v.is_empty(),
            _ => false,
        }
    }
}

/// Property function trait
pub trait PropertyFunction: Send + Sync {
    /// Get the URI of this property function
    fn uri(&self) -> &str;

    /// Build/validate the property function call
    ///
    /// Called during query planning to validate arguments
    #[allow(unused_variables)]
    fn build(
        &self,
        subject: &PropFuncArg,
        predicate: &str,
        object: &PropFuncArg,
        _context: &PropertyFunctionContext,
    ) -> Result<()> {
        // Default implementation: basic validation
        if predicate != self.uri() {
            bail!(
                "Predicate mismatch: expected {}, got {}",
                self.uri(),
                predicate
            );
        }
        Ok(())
    }

    /// Execute the property function
    ///
    /// Returns bindings for unbound variables or verification result
    fn execute(
        &self,
        subject: &PropFuncArg,
        predicate: &str,
        object: &PropFuncArg,
        context: &PropertyFunctionContext,
    ) -> Result<PropertyFunctionResult>;
}

/// Registry for property functions
pub struct PropertyFunctionRegistry {
    functions: DashMap<String, Arc<dyn PropertyFunction>>,
}

impl PropertyFunctionRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            functions: DashMap::new(),
        }
    }

    /// Register a property function
    pub fn register<F: PropertyFunction + 'static>(&self, function: F) -> Result<()> {
        let uri = function.uri().to_string();
        self.functions.insert(uri, Arc::new(function));
        Ok(())
    }

    /// Get a property function by URI
    pub fn get(&self, uri: &str) -> Option<Arc<dyn PropertyFunction>> {
        self.functions.get(uri).map(|entry| Arc::clone(&*entry))
    }

    /// Check if a URI is a registered property function
    pub fn is_property_function(&self, uri: &str) -> bool {
        self.functions.contains_key(uri)
    }

    /// Get all registered URIs
    pub fn registered_uris(&self) -> Vec<String> {
        self.functions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Create a registry with standard property functions
    pub fn with_standard_functions() -> Result<Self> {
        let registry = Self::new();
        register_standard_functions(&registry)?;
        Ok(registry)
    }
}

impl Default for PropertyFunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Register all standard property functions
pub fn register_standard_functions(registry: &PropertyFunctionRegistry) -> Result<()> {
    // List operations
    registry.register(ListMemberFunction)?;
    registry.register(ListIndexFunction)?;
    registry.register(ListLengthFunction)?;

    Ok(())
}

// ============================================================================
// Standard Property Functions
// ============================================================================

/// list:member - List membership testing
///
/// Usage:
/// - (?list list:member ?member) - Generate all members from a list
/// - (<list> list:member <member>) - Test if member is in list
/// - (?list list:member <member>) - Find lists containing member
#[derive(Debug, Clone)]
struct ListMemberFunction;

impl PropertyFunction for ListMemberFunction {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/list#member"
    }

    fn build(
        &self,
        _subject: &PropFuncArg,
        predicate: &str,
        object: &PropFuncArg,
        _context: &PropertyFunctionContext,
    ) -> Result<()> {
        if predicate != self.uri() {
            bail!("Predicate mismatch");
        }
        if object.is_list() && object.list_size() > 0 {
            bail!("list:member does not accept list arguments in object position");
        }
        Ok(())
    }

    fn execute(
        &self,
        subject: &PropFuncArg,
        _predicate: &str,
        object: &PropFuncArg,
        context: &PropertyFunctionContext,
    ) -> Result<PropertyFunctionResult> {
        let subject = context.substitute(subject);
        let object = context.substitute(object);

        // Get the list node
        let list_node = subject
            .as_node()
            .ok_or_else(|| anyhow!("list:member requires single node in subject position"))?;

        // Check if subject is RDF list (for now, we simulate with hardcoded lists)
        // In a full implementation, this would traverse rdf:first/rdf:rest
        let members = self.get_list_members(list_node, context)?;

        if let Some(member_node) = object.as_node() {
            // Object is bound - test membership
            if let AlgebraTerm::Variable(v) = member_node {
                // Generate all members
                let solutions: Vec<HashMap<Variable, AlgebraTerm>> = members
                    .into_iter()
                    .map(|m| {
                        let mut bindings = HashMap::new();
                        bindings.insert(v.clone(), m);
                        bindings
                    })
                    .collect();
                Ok(PropertyFunctionResult::Multiple(solutions))
            } else {
                // Test if member is in list
                let is_member = members.contains(member_node);
                Ok(PropertyFunctionResult::Boolean(is_member))
            }
        } else {
            bail!("list:member requires node in object position");
        }
    }
}

impl ListMemberFunction {
    /// Get members from an RDF list
    ///
    /// This is a simplified implementation. A full implementation would
    /// traverse rdf:first/rdf:rest chains in the graph.
    fn get_list_members(
        &self,
        _list_node: &AlgebraTerm,
        _context: &PropertyFunctionContext,
    ) -> Result<Vec<AlgebraTerm>> {
        // TODO: Implement full RDF list traversal
        // For now, return empty list
        Ok(Vec::new())
    }
}

/// list:index - List index access
///
/// Usage:
/// - (?list list:index (?index ?member)) - Get member at index
/// - (<list> list:index (<index> ?member)) - Get member at specific index
#[derive(Debug, Clone)]
struct ListIndexFunction;

impl PropertyFunction for ListIndexFunction {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/list#index"
    }

    fn build(
        &self,
        _subject: &PropFuncArg,
        predicate: &str,
        object: &PropFuncArg,
        _context: &PropertyFunctionContext,
    ) -> Result<()> {
        if predicate != self.uri() {
            bail!("Predicate mismatch");
        }
        if !object.is_list() || object.list_size() != 2 {
            bail!("list:index requires exactly 2 arguments in object position: (index member)");
        }
        Ok(())
    }

    fn execute(
        &self,
        subject: &PropFuncArg,
        _predicate: &str,
        object: &PropFuncArg,
        context: &PropertyFunctionContext,
    ) -> Result<PropertyFunctionResult> {
        let subject = context.substitute(subject);
        let object = context.substitute(object);

        let list_node = subject
            .as_node()
            .ok_or_else(|| anyhow!("list:index requires single node in subject position"))?;

        let obj_list = object
            .as_list()
            .ok_or_else(|| anyhow!("list:index requires list in object position"))?;

        if obj_list.len() != 2 {
            bail!("list:index requires exactly 2 arguments: (index member)");
        }

        let index_term = &obj_list[0];
        let member_term = &obj_list[1];

        let members = self.get_list_members(list_node, context)?;

        if let AlgebraTerm::Literal(lit) = index_term {
            // Index is bound
            let index: usize = lit
                .value
                .parse()
                .map_err(|_| anyhow!("Invalid index: {}", lit.value))?;

            if let Some(member) = members.get(index) {
                if let AlgebraTerm::Variable(v) = member_term {
                    // Bind member variable
                    let mut bindings = HashMap::new();
                    bindings.insert(v.clone(), member.clone());
                    Ok(PropertyFunctionResult::Single(bindings))
                } else {
                    // Test equality
                    Ok(PropertyFunctionResult::Boolean(member == member_term))
                }
            } else {
                Ok(PropertyFunctionResult::Empty)
            }
        } else if let AlgebraTerm::Variable(v_idx) = index_term {
            // Index is unbound - generate all (index, member) pairs
            use oxirs_core::model::NamedNode;
            let xsd_integer = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                .map_err(|e| anyhow!("Failed to create xsd:integer IRI: {}", e))?;

            let solutions: Vec<HashMap<Variable, AlgebraTerm>> = members
                .into_iter()
                .enumerate()
                .map(|(idx, member)| {
                    let mut bindings = HashMap::new();
                    bindings.insert(
                        v_idx.clone(),
                        AlgebraTerm::Literal(crate::algebra::Literal {
                            value: idx.to_string(),
                            language: None,
                            datatype: Some(xsd_integer.clone()),
                        }),
                    );
                    if let AlgebraTerm::Variable(v_member) = member_term {
                        bindings.insert(v_member.clone(), member);
                    }
                    bindings
                })
                .collect();
            Ok(PropertyFunctionResult::Multiple(solutions))
        } else {
            bail!("list:index requires index to be integer or variable");
        }
    }
}

impl ListIndexFunction {
    fn get_list_members(
        &self,
        _list_node: &AlgebraTerm,
        _context: &PropertyFunctionContext,
    ) -> Result<Vec<AlgebraTerm>> {
        // TODO: Implement full RDF list traversal
        Ok(Vec::new())
    }
}

/// list:length - Get length of RDF list
///
/// Usage:
/// - (?list list:length ?length) - Get length of list
/// - (<list> list:length <n>) - Test if list has length n
#[derive(Debug, Clone)]
struct ListLengthFunction;

impl PropertyFunction for ListLengthFunction {
    fn uri(&self) -> &str {
        "http://jena.apache.org/ARQ/list#length"
    }

    fn execute(
        &self,
        subject: &PropFuncArg,
        _predicate: &str,
        object: &PropFuncArg,
        context: &PropertyFunctionContext,
    ) -> Result<PropertyFunctionResult> {
        let subject = context.substitute(subject);
        let object = context.substitute(object);

        let list_node = subject
            .as_node()
            .ok_or_else(|| anyhow!("list:length requires single node in subject position"))?;

        let length_node = object
            .as_node()
            .ok_or_else(|| anyhow!("list:length requires single node in object position"))?;

        let members = self.get_list_members(list_node, context)?;
        let length = members.len();

        if let AlgebraTerm::Variable(v) = length_node {
            // Bind length variable
            use oxirs_core::model::NamedNode;
            let xsd_integer = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                .map_err(|e| anyhow!("Failed to create xsd:integer IRI: {}", e))?;

            let mut bindings = HashMap::new();
            bindings.insert(
                v.clone(),
                AlgebraTerm::Literal(crate::algebra::Literal {
                    value: length.to_string(),
                    language: None,
                    datatype: Some(xsd_integer),
                }),
            );
            Ok(PropertyFunctionResult::Single(bindings))
        } else if let AlgebraTerm::Literal(lit) = length_node {
            // Test length
            let expected: usize = lit
                .value
                .parse()
                .map_err(|_| anyhow!("Invalid length: {}", lit.value))?;
            Ok(PropertyFunctionResult::Boolean(length == expected))
        } else {
            bail!("list:length requires integer or variable in object position");
        }
    }
}

impl ListLengthFunction {
    fn get_list_members(
        &self,
        _list_node: &AlgebraTerm,
        _context: &PropertyFunctionContext,
    ) -> Result<Vec<AlgebraTerm>> {
        // TODO: Implement full RDF list traversal
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_func_arg_node() {
        use oxirs_core::model::Variable;

        let var = Variable::new("x").unwrap();
        let term = AlgebraTerm::Variable(var);
        let arg = PropFuncArg::node(term.clone());

        assert!(arg.is_node());
        assert!(!arg.is_list());
        assert_eq!(arg.as_node(), Some(&term));
        assert_eq!(arg.list_size(), -1);
    }

    #[test]
    fn test_prop_func_arg_list() {
        use oxirs_core::model::Variable;

        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        let terms = vec![AlgebraTerm::Variable(var_x), AlgebraTerm::Variable(var_y)];
        let arg = PropFuncArg::list(terms.clone());

        assert!(arg.is_list());
        assert!(!arg.is_node());
        assert_eq!(arg.as_list(), Some(terms.as_slice()));
        assert_eq!(arg.list_size(), 2);
    }

    #[test]
    fn test_prop_func_arg_has_variable() {
        use oxirs_core::model::Variable;

        let var = Variable::new("x").unwrap();
        let arg_var = PropFuncArg::node(AlgebraTerm::Variable(var));
        assert!(arg_var.has_variable());

        let arg_iri = PropFuncArg::node(AlgebraTerm::Iri(
            crate::algebra::Iri::new("http://example.org/").unwrap(),
        ));
        assert!(!arg_iri.has_variable());

        let var_x = Variable::new("x").unwrap();
        let arg_list = PropFuncArg::list(vec![
            AlgebraTerm::Variable(var_x),
            AlgebraTerm::Iri(crate::algebra::Iri::new("http://example.org/").unwrap()),
        ]);
        assert!(arg_list.has_variable());
    }

    #[test]
    fn test_prop_func_arg_variables() {
        use oxirs_core::model::Variable;

        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        let arg = PropFuncArg::list(vec![
            AlgebraTerm::Variable(var_x.clone()),
            AlgebraTerm::Variable(var_y.clone()),
            AlgebraTerm::Iri(crate::algebra::Iri::new("http://example.org/").unwrap()),
        ]);

        let vars = arg.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.iter().any(|v| v.as_str() == "x"));
        assert!(vars.iter().any(|v| v.as_str() == "y"));
    }

    #[test]
    fn test_registry_registration() {
        let registry = PropertyFunctionRegistry::new();
        registry.register(ListMemberFunction).unwrap();

        assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#member"));
        assert!(!registry.is_property_function("http://example.org/unknown"));
    }

    #[test]
    fn test_registry_with_standard_functions() {
        let registry = PropertyFunctionRegistry::with_standard_functions().unwrap();

        assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#member"));
        assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#index"));
        assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#length"));

        let uris = registry.registered_uris();
        assert_eq!(uris.len(), 3);
    }

    #[test]
    fn test_property_function_result_into_solutions() {
        use oxirs_core::model::Variable;

        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        let single = PropertyFunctionResult::Single({
            let mut map = HashMap::new();
            map.insert(var_x.clone(), AlgebraTerm::Variable(var_y.clone()));
            map
        });
        assert_eq!(single.into_solutions().len(), 1);

        let multiple = PropertyFunctionResult::Multiple(vec![HashMap::new(), HashMap::new()]);
        assert_eq!(multiple.into_solutions().len(), 2);

        let boolean_true = PropertyFunctionResult::Boolean(true);
        assert_eq!(boolean_true.into_solutions().len(), 1);

        let boolean_false = PropertyFunctionResult::Boolean(false);
        assert_eq!(boolean_false.into_solutions().len(), 0);

        let empty = PropertyFunctionResult::Empty;
        assert_eq!(empty.into_solutions().len(), 0);
    }

    #[test]
    fn test_context_substitute() {
        use crate::executor::InMemoryDataset;
        use oxirs_core::model::Variable;

        let dataset: Arc<dyn crate::executor::Dataset> = Arc::new(InMemoryDataset::new());
        let mut bindings = HashMap::new();

        let var_x = Variable::new("x").unwrap();
        bindings.insert(
            var_x.clone(),
            AlgebraTerm::Iri(crate::algebra::Iri::new("http://example.org/bound").unwrap()),
        );

        let context = PropertyFunctionContext::with_bindings(dataset, bindings);

        let arg = PropFuncArg::node(AlgebraTerm::Variable(var_x));
        let substituted = context.substitute(&arg);

        match substituted {
            PropFuncArg::Node(AlgebraTerm::Iri(iri)) => {
                assert_eq!(iri.as_str(), "http://example.org/bound");
            }
            _ => panic!("Expected IRI after substitution"),
        }
    }
}
