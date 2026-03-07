//! # SPARQL 1.2 Property Function Registry
//!
//! Implements an extensible property function framework analogous to Jena's
//! `PropertyFunctionFactory`. Property functions are magic predicates that,
//! when encountered as the predicate of a triple pattern, trigger custom
//! evaluation logic instead of a normal triple-match.
//!
//! ## Overview
//!
//! Standard SPARQL triple patterns like `?s :p ?o` are matched against the
//! stored graph. A **property function** overrides that: when the predicate
//! IRI is registered in the `PropertyFunctionRegistry`, the engine delegates
//! evaluation to the registered `PropertyFunction` implementation.
//!
//! ### Use Cases
//!
//! - Full-text search: `?s text:search ("rust programming" 10)`
//! - Spatial queries: `?s geo:nearby (51.5 -0.12 1000)`
//! - List operations: `?list list:member ?item`
//! - Custom aggregations: `?s custom:topK (?score 5)`
//!
//! ## Architecture
//!
//! ```text
//! PropertyFunctionRegistry
//!   ├─ register(iri, factory)
//!   ├─ lookup(iri) -> Option<PropertyFunction>
//!   └─ built-in property functions
//!        ├─ list:member
//!        ├─ list:index
//!        ├─ list:length
//!        ├─ text:search
//!        ├─ apf:splitIRI
//!        └─ apf:str (string decomposition)
//! ```

use crate::model::Term;
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Represents the subject side of a property function triple pattern.
/// Can be a single variable/term or a list of arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyFunctionArg {
    /// A single RDF term
    Term(Term),
    /// A variable name (without the ? prefix)
    Variable(String),
    /// An argument list (for multi-arg property functions)
    List(Vec<PropertyFunctionArg>),
}

impl fmt::Display for PropertyFunctionArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyFunctionArg::Term(t) => write!(f, "{t:?}"),
            PropertyFunctionArg::Variable(v) => write!(f, "?{v}"),
            PropertyFunctionArg::List(args) => {
                write!(f, "(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A single binding row produced by a property function evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyFunctionBinding {
    /// Variable-to-term bindings
    bindings: HashMap<String, Term>,
}

impl PropertyFunctionBinding {
    /// Create a new empty binding
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Add a variable binding
    pub fn bind(mut self, var: impl Into<String>, term: Term) -> Self {
        self.bindings.insert(var.into(), term);
        self
    }

    /// Get a binding value
    pub fn get(&self, var: &str) -> Option<&Term> {
        self.bindings.get(var)
    }

    /// Get all bindings
    pub fn bindings(&self) -> &HashMap<String, Term> {
        &self.bindings
    }

    /// Number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether there are no bindings
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl Default for PropertyFunctionBinding {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of evaluating a property function: zero or more binding rows.
#[derive(Debug, Clone)]
pub struct PropertyFunctionResult {
    /// Rows of bindings
    rows: Vec<PropertyFunctionBinding>,
}

impl PropertyFunctionResult {
    /// Create result with no rows (empty result set)
    pub fn empty() -> Self {
        Self { rows: Vec::new() }
    }

    /// Create result with given rows
    pub fn from_rows(rows: Vec<PropertyFunctionBinding>) -> Self {
        Self { rows }
    }

    /// Create result with a single row
    pub fn single(binding: PropertyFunctionBinding) -> Self {
        Self {
            rows: vec![binding],
        }
    }

    /// Get all result rows
    pub fn rows(&self) -> &[PropertyFunctionBinding] {
        &self.rows
    }

    /// Number of result rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no result rows
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterate over result rows
    pub fn iter(&self) -> impl Iterator<Item = &PropertyFunctionBinding> {
        self.rows.iter()
    }
}

/// Metadata describing a property function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFunctionMetadata {
    /// The IRI of the property function
    pub iri: String,
    /// Human-readable name
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// Whether the subject must be bound
    pub subject_must_be_bound: bool,
    /// Whether the object must be bound
    pub object_must_be_bound: bool,
    /// Minimum number of subject arguments
    pub min_subject_args: usize,
    /// Maximum number of subject arguments (None = unlimited)
    pub max_subject_args: Option<usize>,
    /// Minimum number of object arguments
    pub min_object_args: usize,
    /// Maximum number of object arguments (None = unlimited)
    pub max_object_args: Option<usize>,
    /// Category (e.g., "list", "text", "spatial")
    pub category: String,
}

/// Core trait for property function implementations.
///
/// A property function is invoked when the SPARQL engine encounters
/// a triple pattern whose predicate matches a registered IRI.
pub trait PropertyFunction: Send + Sync + fmt::Debug {
    /// Return metadata describing this property function.
    fn metadata(&self) -> PropertyFunctionMetadata;

    /// Evaluate the property function given the subject and object arguments.
    ///
    /// # Arguments
    /// * `subject` - The subject side of the triple pattern
    /// * `object` - The object side of the triple pattern
    ///
    /// # Returns
    /// A set of bindings produced by the evaluation.
    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError>;

    /// Validate arguments before evaluation (optional override).
    fn validate(
        &self,
        subject: &PropertyFunctionArg,
        object: &PropertyFunctionArg,
    ) -> Result<(), OxirsError> {
        let meta = self.metadata();

        // Validate subject argument count for lists
        if let PropertyFunctionArg::List(args) = subject {
            if args.len() < meta.min_subject_args {
                return Err(OxirsError::Query(format!(
                    "Property function {} requires at least {} subject arguments, got {}",
                    meta.iri,
                    meta.min_subject_args,
                    args.len()
                )));
            }
            if let Some(max) = meta.max_subject_args {
                if args.len() > max {
                    return Err(OxirsError::Query(format!(
                        "Property function {} accepts at most {} subject arguments, got {}",
                        meta.iri,
                        max,
                        args.len()
                    )));
                }
            }
        }

        // Validate object argument count for lists
        if let PropertyFunctionArg::List(args) = object {
            if args.len() < meta.min_object_args {
                return Err(OxirsError::Query(format!(
                    "Property function {} requires at least {} object arguments, got {}",
                    meta.iri,
                    meta.min_object_args,
                    args.len()
                )));
            }
            if let Some(max) = meta.max_object_args {
                if args.len() > max {
                    return Err(OxirsError::Query(format!(
                        "Property function {} accepts at most {} object arguments, got {}",
                        meta.iri,
                        max,
                        args.len()
                    )));
                }
            }
        }

        Ok(())
    }

    /// Estimated cardinality of results (for query planning).
    /// Returns None if unknown. Default is None.
    fn estimated_cardinality(
        &self,
        _subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        None
    }
}

/// Factory for creating property function instances (Jena-compatible pattern).
pub trait PropertyFunctionFactory: Send + Sync {
    /// Create a new property function instance for the given IRI.
    fn create(&self, iri: &str) -> Result<Box<dyn PropertyFunction>, OxirsError>;
}

/// Registry of property functions, keyed by predicate IRI.
pub struct PropertyFunctionRegistry {
    /// Registered property function instances
    functions: HashMap<String, Arc<dyn PropertyFunction>>,
    /// Registered factories (lazy creation)
    factories: HashMap<String, Arc<dyn PropertyFunctionFactory>>,
    /// Execution statistics
    stats: PropertyFunctionStats,
}

/// Execution statistics for property functions.
#[derive(Debug, Clone, Default)]
pub struct PropertyFunctionStats {
    /// Total evaluations
    pub total_evaluations: u64,
    /// Total rows produced
    pub total_rows_produced: u64,
    /// Total errors
    pub total_errors: u64,
    /// Per-function evaluation counts
    pub per_function_counts: HashMap<String, u64>,
}

impl Default for PropertyFunctionRegistry {
    fn default() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
            factories: HashMap::new(),
            stats: PropertyFunctionStats::default(),
        };
        registry.register_builtins();
        registry
    }
}

impl PropertyFunctionRegistry {
    /// Create a new registry with built-in property functions.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty registry without built-ins (for testing).
    pub fn empty() -> Self {
        Self {
            functions: HashMap::new(),
            factories: HashMap::new(),
            stats: PropertyFunctionStats::default(),
        }
    }

    /// Register a property function instance for a given IRI.
    pub fn register(&mut self, iri: impl Into<String>, func: Arc<dyn PropertyFunction>) {
        self.functions.insert(iri.into(), func);
    }

    /// Register a factory for lazy creation of property functions.
    pub fn register_factory(
        &mut self,
        iri: impl Into<String>,
        factory: Arc<dyn PropertyFunctionFactory>,
    ) {
        self.factories.insert(iri.into(), factory);
    }

    /// Unregister a property function.
    pub fn unregister(&mut self, iri: &str) -> bool {
        let removed_func = self.functions.remove(iri).is_some();
        let removed_factory = self.factories.remove(iri).is_some();
        removed_func || removed_factory
    }

    /// Check if an IRI is registered as a property function.
    pub fn is_property_function(&self, iri: &str) -> bool {
        self.functions.contains_key(iri) || self.factories.contains_key(iri)
    }

    /// Look up a property function by IRI.
    pub fn lookup(&mut self, iri: &str) -> Option<Arc<dyn PropertyFunction>> {
        // First, check direct registrations
        if let Some(func) = self.functions.get(iri) {
            return Some(Arc::clone(func));
        }

        // Try factories (lazy creation)
        if let Some(factory) = self.factories.get(iri) {
            match factory.create(iri) {
                Ok(func) => {
                    let func: Arc<dyn PropertyFunction> = Arc::from(func);
                    self.functions.insert(iri.to_string(), Arc::clone(&func));
                    return Some(func);
                }
                Err(_) => return None,
            }
        }

        None
    }

    /// Evaluate a property function by IRI.
    pub fn evaluate(
        &mut self,
        iri: &str,
        subject: &PropertyFunctionArg,
        object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let func = self
            .lookup(iri)
            .ok_or_else(|| OxirsError::Query(format!("Unknown property function: {iri}")))?;

        // Validate first
        func.validate(subject, object)?;

        // Evaluate
        let result = func.evaluate(subject, object);

        // Update statistics
        self.stats.total_evaluations += 1;
        *self
            .stats
            .per_function_counts
            .entry(iri.to_string())
            .or_insert(0) += 1;

        match &result {
            Ok(r) => {
                self.stats.total_rows_produced += r.len() as u64;
            }
            Err(_) => {
                self.stats.total_errors += 1;
            }
        }

        result
    }

    /// Get all registered IRIs.
    pub fn registered_iris(&self) -> Vec<String> {
        let mut iris: Vec<String> = self.functions.keys().cloned().collect();
        for iri in self.factories.keys() {
            if !iris.contains(iri) {
                iris.push(iri.clone());
            }
        }
        iris.sort();
        iris
    }

    /// Get metadata for all registered property functions.
    pub fn all_metadata(&self) -> Vec<PropertyFunctionMetadata> {
        self.functions.values().map(|f| f.metadata()).collect()
    }

    /// Get execution statistics.
    pub fn statistics(&self) -> &PropertyFunctionStats {
        &self.stats
    }

    /// Reset execution statistics.
    pub fn reset_statistics(&mut self) {
        self.stats = PropertyFunctionStats::default();
    }

    /// Number of registered property functions.
    pub fn len(&self) -> usize {
        let mut count = self.functions.len();
        for iri in self.factories.keys() {
            if !self.functions.contains_key(iri) {
                count += 1;
            }
        }
        count
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.factories.is_empty()
    }

    /// Register all built-in property functions.
    fn register_builtins(&mut self) {
        // List operations
        self.register(
            "http://jena.apache.org/ARQ/list#member",
            Arc::new(ListMemberPF::new()),
        );
        self.register(
            "http://jena.apache.org/ARQ/list#index",
            Arc::new(ListIndexPF::new()),
        );
        self.register(
            "http://jena.apache.org/ARQ/list#length",
            Arc::new(ListLengthPF::new()),
        );

        // String decomposition
        self.register(
            "http://jena.apache.org/ARQ/property#splitIRI",
            Arc::new(SplitIriPF::new()),
        );
        self.register(
            "http://jena.apache.org/ARQ/property#localname",
            Arc::new(LocalNamePF::new()),
        );
        self.register(
            "http://jena.apache.org/ARQ/property#namespace",
            Arc::new(NamespacePF::new()),
        );

        // Text search (lightweight built-in)
        self.register(
            "http://jena.apache.org/text#search",
            Arc::new(TextSearchPF::new()),
        );

        // String concat property function
        self.register(
            "http://jena.apache.org/ARQ/property#concat",
            Arc::new(ConcatPF::new()),
        );

        // Str split property function
        self.register(
            "http://jena.apache.org/ARQ/property#strSplit",
            Arc::new(StrSplitPF::new()),
        );
    }
}

impl fmt::Debug for PropertyFunctionRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PropertyFunctionRegistry")
            .field("registered_count", &self.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// =============================================================================
// Built-in property function implementations
// =============================================================================

/// list:member - Enumerate members of an RDF list.
///
/// Usage: `?list list:member ?item`
///   Subject: a variable or term representing the list head
///   Object: a variable to bind each member to
#[derive(Debug)]
pub struct ListMemberPF {
    _private: (),
}

impl ListMemberPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ListMemberPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for ListMemberPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/list#member".to_string(),
            name: "list:member".to_string(),
            description: "Enumerates all members of an RDF list".to_string(),
            subject_must_be_bound: false,
            object_must_be_bound: false,
            min_subject_args: 0,
            max_subject_args: None,
            min_object_args: 0,
            max_object_args: None,
            category: "list".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        // If subject is a list, enumerate its members
        match subject {
            PropertyFunctionArg::List(members) => {
                let mut rows = Vec::new();
                for (i, member) in members.iter().enumerate() {
                    if let PropertyFunctionArg::Term(term) = member {
                        let binding = PropertyFunctionBinding::new()
                            .bind("index", make_integer_term(i as i64))
                            .bind("member", term.clone());
                        rows.push(binding);
                    }
                }
                Ok(PropertyFunctionResult::from_rows(rows))
            }
            PropertyFunctionArg::Term(term) => {
                // Single term: return it as the only member
                let binding = PropertyFunctionBinding::new()
                    .bind("index", make_integer_term(0))
                    .bind("member", term.clone());
                Ok(PropertyFunctionResult::single(binding))
            }
            PropertyFunctionArg::Variable(_) => {
                // Cannot enumerate without a bound list
                Ok(PropertyFunctionResult::empty())
            }
        }
    }

    fn estimated_cardinality(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        match subject {
            PropertyFunctionArg::List(members) => Some(members.len() as u64),
            PropertyFunctionArg::Term(_) => Some(1),
            PropertyFunctionArg::Variable(_) => None,
        }
    }
}

/// list:index - Return member at a specific index in an RDF list.
///
/// Usage: `?list list:index (?index ?item)`
#[derive(Debug)]
pub struct ListIndexPF {
    _private: (),
}

impl ListIndexPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ListIndexPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for ListIndexPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/list#index".to_string(),
            name: "list:index".to_string(),
            description: "Returns (index, member) pairs for an RDF list".to_string(),
            subject_must_be_bound: false,
            object_must_be_bound: false,
            min_subject_args: 0,
            max_subject_args: None,
            min_object_args: 0,
            max_object_args: Some(2),
            category: "list".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        match subject {
            PropertyFunctionArg::List(members) => {
                let mut rows = Vec::new();
                for (i, member) in members.iter().enumerate() {
                    if let PropertyFunctionArg::Term(term) = member {
                        let binding = PropertyFunctionBinding::new()
                            .bind("index", make_integer_term(i as i64))
                            .bind("item", term.clone());
                        rows.push(binding);
                    }
                }
                Ok(PropertyFunctionResult::from_rows(rows))
            }
            _ => Ok(PropertyFunctionResult::empty()),
        }
    }
}

/// list:length - Return the length of an RDF list.
///
/// Usage: `?list list:length ?len`
#[derive(Debug)]
pub struct ListLengthPF {
    _private: (),
}

impl ListLengthPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ListLengthPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for ListLengthPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/list#length".to_string(),
            name: "list:length".to_string(),
            description: "Returns the length of an RDF list".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 0,
            max_subject_args: None,
            min_object_args: 0,
            max_object_args: Some(1),
            category: "list".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let len = match subject {
            PropertyFunctionArg::List(members) => members.len(),
            PropertyFunctionArg::Term(_) => 1,
            PropertyFunctionArg::Variable(_) => {
                return Err(OxirsError::Query(
                    "list:length requires a bound subject".to_string(),
                ));
            }
        };

        let binding = PropertyFunctionBinding::new().bind("length", make_integer_term(len as i64));
        Ok(PropertyFunctionResult::single(binding))
    }

    fn estimated_cardinality(
        &self,
        _subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        Some(1) // Always returns exactly one row
    }
}

/// apf:splitIRI - Decompose an IRI into namespace and local name.
///
/// Usage: `?iri apf:splitIRI (?namespace ?localname)`
#[derive(Debug)]
pub struct SplitIriPF {
    _private: (),
}

impl SplitIriPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for SplitIriPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for SplitIriPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/property#splitIRI".to_string(),
            name: "apf:splitIRI".to_string(),
            description: "Splits an IRI into namespace and local name".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 1,
            max_subject_args: Some(1),
            min_object_args: 0,
            max_object_args: Some(2),
            category: "string".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let iri_str = match subject {
            PropertyFunctionArg::Term(term) => extract_iri_string(term)?,
            PropertyFunctionArg::Variable(_) => {
                return Err(OxirsError::Query(
                    "apf:splitIRI requires a bound IRI subject".to_string(),
                ));
            }
            PropertyFunctionArg::List(args) => {
                if let Some(PropertyFunctionArg::Term(term)) = args.first() {
                    extract_iri_string(term)?
                } else {
                    return Err(OxirsError::Query(
                        "apf:splitIRI requires an IRI argument".to_string(),
                    ));
                }
            }
        };

        // Split at the last # or /
        let (namespace, local_name) = split_iri(&iri_str);

        let binding = PropertyFunctionBinding::new()
            .bind("namespace", make_string_term(&namespace))
            .bind("localname", make_string_term(&local_name));

        Ok(PropertyFunctionResult::single(binding))
    }

    fn estimated_cardinality(
        &self,
        _subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        Some(1)
    }
}

/// apf:localname - Extract the local name from an IRI.
#[derive(Debug)]
pub struct LocalNamePF {
    _private: (),
}

impl LocalNamePF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for LocalNamePF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for LocalNamePF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/property#localname".to_string(),
            name: "apf:localname".to_string(),
            description: "Extracts the local name from an IRI".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 1,
            max_subject_args: Some(1),
            min_object_args: 0,
            max_object_args: Some(1),
            category: "string".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let iri_str = extract_arg_iri(subject)?;
        let (_, local_name) = split_iri(&iri_str);

        let binding =
            PropertyFunctionBinding::new().bind("localname", make_string_term(&local_name));
        Ok(PropertyFunctionResult::single(binding))
    }

    fn estimated_cardinality(
        &self,
        _subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        Some(1)
    }
}

/// apf:namespace - Extract the namespace from an IRI.
#[derive(Debug)]
pub struct NamespacePF {
    _private: (),
}

impl NamespacePF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NamespacePF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for NamespacePF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/property#namespace".to_string(),
            name: "apf:namespace".to_string(),
            description: "Extracts the namespace from an IRI".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 1,
            max_subject_args: Some(1),
            min_object_args: 0,
            max_object_args: Some(1),
            category: "string".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let iri_str = extract_arg_iri(subject)?;
        let (namespace, _) = split_iri(&iri_str);

        let binding =
            PropertyFunctionBinding::new().bind("namespace", make_string_term(&namespace));
        Ok(PropertyFunctionResult::single(binding))
    }
}

/// text:search - Simple text search property function.
///
/// Usage: `?s text:search ("search terms" ?score)`
#[derive(Debug)]
pub struct TextSearchPF {
    _private: (),
}

impl TextSearchPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for TextSearchPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for TextSearchPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/text#search".to_string(),
            name: "text:search".to_string(),
            description: "Full-text search across literal values".to_string(),
            subject_must_be_bound: false,
            object_must_be_bound: true,
            min_subject_args: 0,
            max_subject_args: None,
            min_object_args: 1,
            max_object_args: Some(3),
            category: "text".to_string(),
        }
    }

    fn evaluate(
        &self,
        _subject: &PropertyFunctionArg,
        object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        // Extract search query from object
        let query = match object {
            PropertyFunctionArg::Term(term) => extract_string_value(term),
            PropertyFunctionArg::List(args) => {
                if let Some(PropertyFunctionArg::Term(term)) = args.first() {
                    extract_string_value(term)
                } else {
                    return Err(OxirsError::Query(
                        "text:search requires a search query string".to_string(),
                    ));
                }
            }
            PropertyFunctionArg::Variable(_) => {
                return Err(OxirsError::Query(
                    "text:search requires a bound search query".to_string(),
                ));
            }
        };

        // In a full implementation, this would query a text index.
        // For now, return a placeholder binding showing the query was parsed.
        let binding = PropertyFunctionBinding::new()
            .bind("query", make_string_term(&query))
            .bind("score", make_double_term(1.0));

        Ok(PropertyFunctionResult::single(binding))
    }
}

/// apf:concat - Concatenate multiple string arguments.
///
/// Usage: `(?a ?b ?c) apf:concat ?result`
#[derive(Debug)]
pub struct ConcatPF {
    _private: (),
}

impl ConcatPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ConcatPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for ConcatPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/property#concat".to_string(),
            name: "apf:concat".to_string(),
            description: "Concatenates string arguments into a single string".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 1,
            max_subject_args: None,
            min_object_args: 0,
            max_object_args: Some(1),
            category: "string".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let parts: Vec<String> = match subject {
            PropertyFunctionArg::List(args) => args
                .iter()
                .filter_map(|a| {
                    if let PropertyFunctionArg::Term(t) = a {
                        Some(extract_string_value(t))
                    } else {
                        None
                    }
                })
                .collect(),
            PropertyFunctionArg::Term(term) => vec![extract_string_value(term)],
            PropertyFunctionArg::Variable(_) => {
                return Err(OxirsError::Query(
                    "apf:concat requires bound string arguments".to_string(),
                ));
            }
        };

        let concatenated = parts.join("");
        let binding =
            PropertyFunctionBinding::new().bind("result", make_string_term(&concatenated));
        Ok(PropertyFunctionResult::single(binding))
    }

    fn estimated_cardinality(
        &self,
        _subject: &PropertyFunctionArg,
        _object: &PropertyFunctionArg,
    ) -> Option<u64> {
        Some(1)
    }
}

/// apf:strSplit - Split a string by a delimiter.
///
/// Usage: `?str apf:strSplit (?delimiter ?part)`
#[derive(Debug)]
pub struct StrSplitPF {
    _private: (),
}

impl StrSplitPF {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for StrSplitPF {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFunction for StrSplitPF {
    fn metadata(&self) -> PropertyFunctionMetadata {
        PropertyFunctionMetadata {
            iri: "http://jena.apache.org/ARQ/property#strSplit".to_string(),
            name: "apf:strSplit".to_string(),
            description: "Splits a string by a delimiter, producing multiple bindings".to_string(),
            subject_must_be_bound: true,
            object_must_be_bound: false,
            min_subject_args: 1,
            max_subject_args: Some(1),
            min_object_args: 1,
            max_object_args: Some(2),
            category: "string".to_string(),
        }
    }

    fn evaluate(
        &self,
        subject: &PropertyFunctionArg,
        object: &PropertyFunctionArg,
    ) -> Result<PropertyFunctionResult, OxirsError> {
        let input = extract_arg_string(subject)?;

        // Get delimiter from object
        let delimiter = match object {
            PropertyFunctionArg::Term(term) => extract_string_value(term),
            PropertyFunctionArg::List(args) => {
                if let Some(PropertyFunctionArg::Term(term)) = args.first() {
                    extract_string_value(term)
                } else {
                    ",".to_string() // Default delimiter
                }
            }
            PropertyFunctionArg::Variable(_) => ",".to_string(),
        };

        let parts: Vec<&str> = input.split(&delimiter).collect();
        let mut rows = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            let binding = PropertyFunctionBinding::new()
                .bind("index", make_integer_term(i as i64))
                .bind("part", make_string_term(part));
            rows.push(binding);
        }

        Ok(PropertyFunctionResult::from_rows(rows))
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Create an xsd:integer Term.
fn make_integer_term(value: i64) -> Term {
    Term::Literal(crate::model::Literal::new_typed(
        value.to_string(),
        crate::model::NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
    ))
}

/// Create an xsd:double Term.
fn make_double_term(value: f64) -> Term {
    Term::Literal(crate::model::Literal::new_typed(
        value.to_string(),
        crate::model::NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#double"),
    ))
}

/// Create an xsd:string Term.
fn make_string_term(value: &str) -> Term {
    Term::Literal(crate::model::Literal::new(value))
}

/// Extract the string representation from a Term.
fn extract_string_value(term: &Term) -> String {
    match term {
        Term::Literal(lit) => lit.value().to_string(),
        Term::NamedNode(nn) => nn.as_str().to_string(),
        Term::BlankNode(bn) => bn.as_str().to_string(),
        _ => format!("{term:?}"),
    }
}

/// Extract IRI string from a Term.
fn extract_iri_string(term: &Term) -> Result<String, OxirsError> {
    match term {
        Term::NamedNode(nn) => Ok(nn.as_str().to_string()),
        _ => Err(OxirsError::Query(format!("Expected IRI, got: {term:?}"))),
    }
}

/// Extract IRI from a PropertyFunctionArg.
fn extract_arg_iri(arg: &PropertyFunctionArg) -> Result<String, OxirsError> {
    match arg {
        PropertyFunctionArg::Term(term) => extract_iri_string(term),
        PropertyFunctionArg::List(args) => {
            if let Some(PropertyFunctionArg::Term(term)) = args.first() {
                extract_iri_string(term)
            } else {
                Err(OxirsError::Query(
                    "Expected IRI argument in list".to_string(),
                ))
            }
        }
        PropertyFunctionArg::Variable(v) => Err(OxirsError::Query(format!(
            "Expected bound IRI, got unbound variable ?{v}"
        ))),
    }
}

/// Extract string from a PropertyFunctionArg.
fn extract_arg_string(arg: &PropertyFunctionArg) -> Result<String, OxirsError> {
    match arg {
        PropertyFunctionArg::Term(term) => Ok(extract_string_value(term)),
        PropertyFunctionArg::List(args) => {
            if let Some(PropertyFunctionArg::Term(term)) = args.first() {
                Ok(extract_string_value(term))
            } else {
                Err(OxirsError::Query(
                    "Expected string argument in list".to_string(),
                ))
            }
        }
        PropertyFunctionArg::Variable(v) => Err(OxirsError::Query(format!(
            "Expected bound string, got unbound variable ?{v}"
        ))),
    }
}

/// Split an IRI into namespace and local name at the last `#`, `/`, or `:`.
fn split_iri(iri: &str) -> (String, String) {
    // Try # first, then /, then : (for URN-style IRIs like urn:isbn:123456)
    if let Some(pos) = iri.rfind('#') {
        (iri[..=pos].to_string(), iri[pos + 1..].to_string())
    } else if let Some(pos) = iri.rfind('/') {
        (iri[..=pos].to_string(), iri[pos + 1..].to_string())
    } else if let Some(pos) = iri.rfind(':') {
        (iri[..=pos].to_string(), iri[pos + 1..].to_string())
    } else {
        (String::new(), iri.to_string())
    }
}

#[cfg(test)]
#[path = "property_function_registry_tests.rs"]
mod tests;
