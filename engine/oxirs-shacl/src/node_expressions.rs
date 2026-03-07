//! # SHACL Node Expressions (SHACL-AF sh:values)
//!
//! Implements SHACL Advanced Features node expression evaluation, enabling
//! declarative computation of RDF node values from graph patterns.
//!
//! Node expressions are the foundation of SHACL rules (sh:values, sh:condition)
//! and allow complex value derivation without SPARQL.
//!
//! ## Supported Expression Types
//!
//! - **Focus Node** (`sh:this`): The current focus node
//! - **Constant** (`sh:value`): A fixed IRI, literal, or blank node
//! - **Path** (`sh:path`): Follow a property path from the focus node
//! - **Filter Shape** (`sh:filterShape`): Nodes matching a given shape
//! - **Function** (`sh:function`): Apply a SHACL function to arguments
//! - **Intersection** (`sh:intersection`): Nodes common to all operands
//! - **Union** (`sh:union`): Nodes in any operand
//! - **Minus** (`sh:minus`): Nodes in the first but not the second operand
//! - **If-Then-Else** (`sh:if`): Conditional value selection
//! - **Count** (`sh:count`): Count matching nodes
//! - **Distinct** (`sh:distinct`): Deduplicate results
//! - **OrderBy** (`sh:orderBy`): Sort results
//! - **Limit/Offset** (`sh:limit`, `sh:offset`): Pagination
//!
//! ## References
//!
//! - <https://www.w3.org/TR/shacl-af/#node-expressions>

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Node expression AST
// ---------------------------------------------------------------------------

/// A SHACL node expression that can be evaluated against a focus node and graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeExpression {
    /// The current focus node itself (`sh:this`).
    FocusNode,

    /// A constant term.
    Constant(NodeTerm),

    /// Follow a property path from the focus node.
    Path(PropertyPath),

    /// Filter nodes through a shape.
    FilterShape {
        /// The set of candidate nodes.
        nodes: Box<NodeExpression>,
        /// The shape IRI to filter against.
        shape: String,
    },

    /// Intersection of multiple node sets.
    Intersection(Vec<NodeExpression>),

    /// Union of multiple node sets.
    Union(Vec<NodeExpression>),

    /// Set difference: nodes in `base` but not in `excluded`.
    Minus {
        /// Base node set.
        base: Box<NodeExpression>,
        /// Nodes to exclude.
        excluded: Box<NodeExpression>,
    },

    /// Conditional: if condition evaluates to non-empty, use `then_expr`,
    /// otherwise use `else_expr`.
    IfThenElse {
        /// Condition expression.
        condition: Box<NodeExpression>,
        /// Expression when condition is true.
        then_expr: Box<NodeExpression>,
        /// Expression when condition is false.
        else_expr: Box<NodeExpression>,
    },

    /// Apply a function to arguments.
    FunctionCall {
        /// Function IRI.
        function: String,
        /// Argument expressions.
        args: Vec<NodeExpression>,
    },

    /// Count the number of nodes from the inner expression.
    Count(Box<NodeExpression>),

    /// Deduplicate results from the inner expression.
    Distinct(Box<NodeExpression>),

    /// Sort results by a property path.
    OrderBy {
        /// Expression to sort.
        expr: Box<NodeExpression>,
        /// Property path to sort by.
        sort_path: PropertyPath,
        /// Ascending order.
        ascending: bool,
    },

    /// Limit the number of results.
    Limit {
        /// Expression to limit.
        expr: Box<NodeExpression>,
        /// Maximum number of results.
        limit: usize,
    },

    /// Skip the first N results.
    Offset {
        /// Expression to offset.
        expr: Box<NodeExpression>,
        /// Number of results to skip.
        offset: usize,
    },

    /// Group-concat: join string values with a separator.
    GroupConcat {
        /// Expression producing the values.
        expr: Box<NodeExpression>,
        /// Separator string.
        separator: String,
    },
}

/// A simplified property path for node expression evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyPath {
    /// A single predicate IRI.
    Predicate(String),
    /// Inverse path (^predicate).
    Inverse(Box<PropertyPath>),
    /// Sequence path (p1 / p2).
    Sequence(Vec<PropertyPath>),
    /// Alternative path (p1 | p2).
    Alternative(Vec<PropertyPath>),
    /// Zero-or-more repetition (p*).
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more repetition (p+).
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one (p?).
    ZeroOrOne(Box<PropertyPath>),
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Predicate(iri) => write!(f, "<{iri}>"),
            Self::Inverse(p) => write!(f, "^{p}"),
            Self::Sequence(steps) => {
                let parts: Vec<_> = steps.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" / "))
            }
            Self::Alternative(alts) => {
                let parts: Vec<_> = alts.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" | "))
            }
            Self::ZeroOrMore(p) => write!(f, "{p}*"),
            Self::OneOrMore(p) => write!(f, "{p}+"),
            Self::ZeroOrOne(p) => write!(f, "{p}?"),
        }
    }
}

/// An RDF term in node expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum NodeTerm {
    /// An IRI reference.
    Iri(String),
    /// A typed or untyped literal.
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A blank node.
    BlankNode(String),
}

impl fmt::Display for NodeTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{iri}>"),
            Self::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                }
                Ok(())
            }
            Self::BlankNode(id) => write!(f, "_:{id}"),
        }
    }
}

impl NodeTerm {
    /// Create an IRI term.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create a string literal.
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: None,
            lang: None,
        }
    }

    /// Create a typed literal.
    pub fn typed_literal(s: impl Into<String>, dt: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: Some(dt.into()),
            lang: None,
        }
    }

    /// Create a language-tagged literal.
    pub fn lang_literal(s: impl Into<String>, lang: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: None,
            lang: Some(lang.into()),
        }
    }

    /// Check if this is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }

    /// Check if this is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal { .. })
    }

    /// Get the string value of this term.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Iri(s) | Self::BlankNode(s) => s,
            Self::Literal { value, .. } => value,
        }
    }

    /// Try to parse a literal as an integer.
    pub fn as_integer(&self) -> Option<i64> {
        if let Self::Literal { value, .. } = self {
            value.parse().ok()
        } else {
            None
        }
    }

    /// Try to parse a literal as a float.
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Literal { value, .. } = self {
            value.parse().ok()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Simple triple store for node expression evaluation
// ---------------------------------------------------------------------------

/// A triple in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    /// Subject.
    pub subject: NodeTerm,
    /// Predicate.
    pub predicate: NodeTerm,
    /// Object.
    pub object: NodeTerm,
}

/// A minimal graph interface for evaluating node expressions.
pub trait NodeExprGraph {
    /// Get all objects for a given subject and predicate.
    fn objects(&self, subject: &NodeTerm, predicate: &str) -> Vec<NodeTerm>;

    /// Get all subjects for a given predicate and object.
    fn subjects(&self, predicate: &str, object: &NodeTerm) -> Vec<NodeTerm>;

    /// Check if a node conforms to a shape (by IRI).
    fn conforms_to_shape(&self, node: &NodeTerm, shape_iri: &str) -> bool;

    /// Get all triples in the graph.
    fn all_triples(&self) -> Vec<Triple>;
}

/// In-memory graph for testing and simple evaluation.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraph {
    triples: Vec<Triple>,
    /// Shape conformance answers (for testing).
    shape_conformance: HashMap<(String, String), bool>,
}

impl InMemoryGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, subject: NodeTerm, predicate: NodeTerm, object: NodeTerm) {
        self.triples.push(Triple {
            subject,
            predicate,
            object,
        });
    }

    /// Add a shape conformance fact (for testing shape filters).
    pub fn set_conforms(&mut self, node: &str, shape: &str, conforms: bool) {
        self.shape_conformance
            .insert((node.to_string(), shape.to_string()), conforms);
    }

    /// Number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

impl NodeExprGraph for InMemoryGraph {
    fn objects(&self, subject: &NodeTerm, predicate: &str) -> Vec<NodeTerm> {
        self.triples
            .iter()
            .filter(|t| {
                &t.subject == subject && matches!(&t.predicate, NodeTerm::Iri(p) if p == predicate)
            })
            .map(|t| t.object.clone())
            .collect()
    }

    fn subjects(&self, predicate: &str, object: &NodeTerm) -> Vec<NodeTerm> {
        self.triples
            .iter()
            .filter(|t| {
                &t.object == object && matches!(&t.predicate, NodeTerm::Iri(p) if p == predicate)
            })
            .map(|t| t.subject.clone())
            .collect()
    }

    fn conforms_to_shape(&self, node: &NodeTerm, shape_iri: &str) -> bool {
        let node_str = node.as_str().to_string();
        self.shape_conformance
            .get(&(node_str, shape_iri.to_string()))
            .copied()
            .unwrap_or(false)
    }

    fn all_triples(&self) -> Vec<Triple> {
        self.triples.clone()
    }
}

// ---------------------------------------------------------------------------
// Node expression evaluator
// ---------------------------------------------------------------------------

/// Configuration for node expression evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Maximum path traversal depth to prevent infinite loops.
    pub max_path_depth: usize,
    /// Maximum number of results before truncation.
    pub max_results: usize,
    /// Enable deduplication by default.
    pub deduplicate: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_path_depth: 100,
            max_results: 10000,
            deduplicate: false,
        }
    }
}

/// Statistics from evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalStats {
    /// Number of graph lookups performed.
    pub graph_lookups: u64,
    /// Number of nodes evaluated.
    pub nodes_evaluated: u64,
    /// Number of path traversals.
    pub path_traversals: u64,
    /// Number of shape checks.
    pub shape_checks: u64,
    /// Number of function calls.
    pub function_calls: u64,
}

/// Errors from node expression evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeExprError {
    /// Path depth exceeded.
    PathDepthExceeded { depth: usize, max: usize },
    /// Result limit exceeded.
    ResultLimitExceeded { count: usize, max: usize },
    /// Unknown function.
    UnknownFunction(String),
    /// Invalid argument count.
    InvalidArgCount {
        function: String,
        expected: usize,
        got: usize,
    },
    /// Type error in computation.
    TypeError(String),
    /// Empty sequence where at least one value was expected.
    EmptySequence(String),
}

impl fmt::Display for NodeExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathDepthExceeded { depth, max } => {
                write!(f, "Path depth {depth} exceeded maximum {max}")
            }
            Self::ResultLimitExceeded { count, max } => {
                write!(f, "Result count {count} exceeded limit {max}")
            }
            Self::UnknownFunction(name) => write!(f, "Unknown function: {name}"),
            Self::InvalidArgCount {
                function,
                expected,
                got,
            } => {
                write!(f, "Function {function} expects {expected} args, got {got}")
            }
            Self::TypeError(msg) => write!(f, "Type error: {msg}"),
            Self::EmptySequence(ctx) => write!(f, "Empty sequence in {ctx}"),
        }
    }
}

impl std::error::Error for NodeExprError {}

/// Evaluates SHACL node expressions against a graph.
pub struct NodeExprEvaluator {
    config: EvalConfig,
    stats: EvalStats,
    /// Built-in functions registry.
    functions: HashMap<String, BuiltinFunction>,
}

/// Function signature for node expression built-in functions.
type NodeExprFn = fn(&[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError>;

/// A built-in function for node expressions.
#[derive(Clone)]
struct BuiltinFunction {
    /// Number of expected arguments.
    arg_count: usize,
    /// The function implementation.
    eval: NodeExprFn,
}

impl fmt::Debug for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinFunction")
            .field("arg_count", &self.arg_count)
            .finish()
    }
}

impl NodeExprEvaluator {
    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self::with_config(EvalConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: EvalConfig) -> Self {
        let mut functions = HashMap::new();

        // Register built-in functions
        functions.insert(
            "sh:strlen".to_string(),
            BuiltinFunction {
                arg_count: 1,
                eval: builtin_strlen,
            },
        );
        functions.insert(
            "sh:concat".to_string(),
            BuiltinFunction {
                arg_count: 2,
                eval: builtin_concat,
            },
        );
        functions.insert(
            "sh:sum".to_string(),
            BuiltinFunction {
                arg_count: 1,
                eval: builtin_sum,
            },
        );

        Self {
            config,
            stats: EvalStats::default(),
            functions,
        }
    }

    /// Get evaluation statistics.
    pub fn stats(&self) -> &EvalStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = EvalStats::default();
    }

    /// Register a custom built-in function.
    pub fn register_function(
        &mut self,
        name: impl Into<String>,
        arg_count: usize,
        eval_fn: NodeExprFn,
    ) {
        self.functions.insert(
            name.into(),
            BuiltinFunction {
                arg_count,
                eval: eval_fn,
            },
        );
    }

    /// Evaluate a node expression against a focus node in a graph.
    pub fn evaluate(
        &mut self,
        expr: &NodeExpression,
        focus: &NodeTerm,
        graph: &dyn NodeExprGraph,
    ) -> Result<Vec<NodeTerm>, NodeExprError> {
        self.stats.nodes_evaluated += 1;
        self.eval_inner(expr, focus, graph, 0)
    }

    fn eval_inner(
        &mut self,
        expr: &NodeExpression,
        focus: &NodeTerm,
        graph: &dyn NodeExprGraph,
        depth: usize,
    ) -> Result<Vec<NodeTerm>, NodeExprError> {
        if depth > self.config.max_path_depth {
            return Err(NodeExprError::PathDepthExceeded {
                depth,
                max: self.config.max_path_depth,
            });
        }

        match expr {
            NodeExpression::FocusNode => Ok(vec![focus.clone()]),

            NodeExpression::Constant(term) => Ok(vec![term.clone()]),

            NodeExpression::Path(path) => {
                self.stats.path_traversals += 1;
                self.eval_path(path, focus, graph, depth)
            }

            NodeExpression::FilterShape { nodes, shape } => {
                let candidates = self.eval_inner(nodes, focus, graph, depth + 1)?;
                self.stats.shape_checks += candidates.len() as u64;
                let filtered = candidates
                    .into_iter()
                    .filter(|n| graph.conforms_to_shape(n, shape))
                    .collect();
                Ok(filtered)
            }

            NodeExpression::Intersection(exprs) => {
                if exprs.is_empty() {
                    return Ok(vec![]);
                }
                let mut sets: Vec<BTreeSet<NodeTerm>> = Vec::new();
                for e in exprs {
                    let nodes = self.eval_inner(e, focus, graph, depth + 1)?;
                    sets.push(nodes.into_iter().collect());
                }
                let mut result = sets[0].clone();
                for s in &sets[1..] {
                    result = result.intersection(s).cloned().collect();
                }
                Ok(result.into_iter().collect())
            }

            NodeExpression::Union(exprs) => {
                let mut combined = BTreeSet::new();
                for e in exprs {
                    let nodes = self.eval_inner(e, focus, graph, depth + 1)?;
                    combined.extend(nodes);
                }
                Ok(combined.into_iter().collect())
            }

            NodeExpression::Minus { base, excluded } => {
                let base_nodes = self.eval_inner(base, focus, graph, depth + 1)?;
                let excl_nodes: HashSet<NodeTerm> = self
                    .eval_inner(excluded, focus, graph, depth + 1)?
                    .into_iter()
                    .collect();
                let result = base_nodes
                    .into_iter()
                    .filter(|n| !excl_nodes.contains(n))
                    .collect();
                Ok(result)
            }

            NodeExpression::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond_result = self.eval_inner(condition, focus, graph, depth + 1)?;
                if cond_result.is_empty() {
                    self.eval_inner(else_expr, focus, graph, depth + 1)
                } else {
                    self.eval_inner(then_expr, focus, graph, depth + 1)
                }
            }

            NodeExpression::FunctionCall { function, args } => {
                self.stats.function_calls += 1;
                let func = self
                    .functions
                    .get(function)
                    .cloned()
                    .ok_or_else(|| NodeExprError::UnknownFunction(function.clone()))?;

                if args.len() != func.arg_count {
                    return Err(NodeExprError::InvalidArgCount {
                        function: function.clone(),
                        expected: func.arg_count,
                        got: args.len(),
                    });
                }

                let mut evaluated_args = Vec::new();
                for arg in args {
                    let val = self.eval_inner(arg, focus, graph, depth + 1)?;
                    evaluated_args.push(val);
                }
                (func.eval)(&evaluated_args)
            }

            NodeExpression::Count(inner) => {
                let nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                let count = nodes.len();
                Ok(vec![NodeTerm::typed_literal(
                    count.to_string(),
                    "http://www.w3.org/2001/XMLSchema#integer",
                )])
            }

            NodeExpression::Distinct(inner) => {
                let nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                let mut seen = HashSet::new();
                let result = nodes
                    .into_iter()
                    .filter(|n| seen.insert(n.clone()))
                    .collect();
                Ok(result)
            }

            NodeExpression::OrderBy {
                expr: inner,
                sort_path,
                ascending,
            } => {
                let mut nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                let asc = *ascending;
                // Sort by the value obtained from sort_path
                nodes.sort_by(|a, b| {
                    let a_val = self.eval_path(sort_path, a, graph, depth + 1).ok();
                    let b_val = self.eval_path(sort_path, b, graph, depth + 1).ok();
                    let a_str = a_val
                        .as_ref()
                        .and_then(|v| v.first())
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    let b_str = b_val
                        .as_ref()
                        .and_then(|v| v.first())
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    if asc {
                        a_str.cmp(&b_str)
                    } else {
                        b_str.cmp(&a_str)
                    }
                });
                Ok(nodes)
            }

            NodeExpression::Limit { expr: inner, limit } => {
                let nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                Ok(nodes.into_iter().take(*limit).collect())
            }

            NodeExpression::Offset {
                expr: inner,
                offset,
            } => {
                let nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                Ok(nodes.into_iter().skip(*offset).collect())
            }

            NodeExpression::GroupConcat {
                expr: inner,
                separator,
            } => {
                let nodes = self.eval_inner(inner, focus, graph, depth + 1)?;
                let parts: Vec<_> = nodes.iter().map(|n| n.as_str().to_string()).collect();
                let concatenated = parts.join(separator);
                Ok(vec![NodeTerm::literal(concatenated)])
            }
        }
    }

    /// Evaluate a property path from a starting node.
    fn eval_path(
        &mut self,
        path: &PropertyPath,
        start: &NodeTerm,
        graph: &dyn NodeExprGraph,
        depth: usize,
    ) -> Result<Vec<NodeTerm>, NodeExprError> {
        if depth > self.config.max_path_depth {
            return Err(NodeExprError::PathDepthExceeded {
                depth,
                max: self.config.max_path_depth,
            });
        }

        self.stats.graph_lookups += 1;

        match path {
            PropertyPath::Predicate(iri) => Ok(graph.objects(start, iri)),

            PropertyPath::Inverse(inner) => match inner.as_ref() {
                PropertyPath::Predicate(iri) => Ok(graph.subjects(iri, start)),
                other => {
                    // For complex inverse paths, evaluate forward then reverse
                    let forward = self.eval_path(other, start, graph, depth + 1)?;
                    let mut result = Vec::new();
                    for node in &forward {
                        let back = self.eval_path(other, node, graph, depth + 1)?;
                        if back.contains(start) {
                            result.push(node.clone());
                        }
                    }
                    Ok(result)
                }
            },

            PropertyPath::Sequence(steps) => {
                let mut current = vec![start.clone()];
                for step in steps {
                    let mut next = Vec::new();
                    for node in &current {
                        let step_result = self.eval_path(step, node, graph, depth + 1)?;
                        next.extend(step_result);
                    }
                    current = next;
                }
                Ok(current)
            }

            PropertyPath::Alternative(alts) => {
                let mut result = BTreeSet::new();
                for alt in alts {
                    let alt_result = self.eval_path(alt, start, graph, depth + 1)?;
                    result.extend(alt_result);
                }
                Ok(result.into_iter().collect())
            }

            PropertyPath::ZeroOrMore(inner) => self.eval_closure(inner, start, graph, depth, true),

            PropertyPath::OneOrMore(inner) => self.eval_closure(inner, start, graph, depth, false),

            PropertyPath::ZeroOrOne(inner) => {
                let mut result = vec![start.clone()]; // zero
                let one_step = self.eval_path(inner, start, graph, depth + 1)?;
                for node in one_step {
                    if !result.contains(&node) {
                        result.push(node);
                    }
                }
                Ok(result)
            }
        }
    }

    /// Evaluate transitive closure (zero-or-more or one-or-more).
    fn eval_closure(
        &mut self,
        path: &PropertyPath,
        start: &NodeTerm,
        graph: &dyn NodeExprGraph,
        depth: usize,
        include_start: bool,
    ) -> Result<Vec<NodeTerm>, NodeExprError> {
        let mut visited = HashSet::new();
        let mut queue = vec![start.clone()];
        let mut result = Vec::new();

        if include_start {
            result.push(start.clone());
            visited.insert(start.clone());
        }

        while let Some(current) = queue.pop() {
            let next = self.eval_path(path, &current, graph, depth + 1)?;
            for node in next {
                if visited.insert(node.clone()) {
                    result.push(node.clone());
                    queue.push(node);
                }
            }

            if result.len() > self.config.max_results {
                return Err(NodeExprError::ResultLimitExceeded {
                    count: result.len(),
                    max: self.config.max_results,
                });
            }
        }

        Ok(result)
    }
}

impl Default for NodeExprEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in functions
// ---------------------------------------------------------------------------

fn builtin_strlen(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut results = Vec::new();
    for term in &args[0] {
        let len = term.as_str().len();
        results.push(NodeTerm::typed_literal(
            len.to_string(),
            "http://www.w3.org/2001/XMLSchema#integer",
        ));
    }
    Ok(results)
}

fn builtin_concat(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut results = Vec::new();
    for a in &args[0] {
        for b in &args[1] {
            let concatenated = format!("{}{}", a.as_str(), b.as_str());
            results.push(NodeTerm::literal(concatenated));
        }
    }
    Ok(results)
}

fn builtin_sum(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut total = 0.0_f64;
    for term in &args[0] {
        let val = term.as_float().ok_or_else(|| {
            NodeExprError::TypeError(format!(
                "Cannot convert '{}' to number for sum",
                term.as_str()
            ))
        })?;
        total += val;
    }
    Ok(vec![NodeTerm::typed_literal(
        total.to_string(),
        "http://www.w3.org/2001/XMLSchema#double",
    )])
}

// ---------------------------------------------------------------------------
// Expression builder (fluent API)
// ---------------------------------------------------------------------------

/// Fluent builder for constructing node expressions.
pub struct ExprBuilder;

impl ExprBuilder {
    /// The focus node expression.
    pub fn focus() -> NodeExpression {
        NodeExpression::FocusNode
    }

    /// A constant term expression.
    pub fn constant(term: NodeTerm) -> NodeExpression {
        NodeExpression::Constant(term)
    }

    /// A path expression (single predicate).
    pub fn path(predicate: impl Into<String>) -> NodeExpression {
        NodeExpression::Path(PropertyPath::Predicate(predicate.into()))
    }

    /// A sequence path (p1 / p2 / ...).
    pub fn sequence_path(predicates: &[&str]) -> NodeExpression {
        NodeExpression::Path(PropertyPath::Sequence(
            predicates
                .iter()
                .map(|p| PropertyPath::Predicate(p.to_string()))
                .collect(),
        ))
    }

    /// An intersection of expressions.
    pub fn intersection(exprs: Vec<NodeExpression>) -> NodeExpression {
        NodeExpression::Intersection(exprs)
    }

    /// A union of expressions.
    pub fn union(exprs: Vec<NodeExpression>) -> NodeExpression {
        NodeExpression::Union(exprs)
    }

    /// A set-minus expression.
    pub fn minus(base: NodeExpression, excluded: NodeExpression) -> NodeExpression {
        NodeExpression::Minus {
            base: Box::new(base),
            excluded: Box::new(excluded),
        }
    }

    /// An if-then-else expression.
    pub fn if_then_else(
        condition: NodeExpression,
        then_expr: NodeExpression,
        else_expr: NodeExpression,
    ) -> NodeExpression {
        NodeExpression::IfThenElse {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// A count expression.
    pub fn count(inner: NodeExpression) -> NodeExpression {
        NodeExpression::Count(Box::new(inner))
    }

    /// A distinct expression.
    pub fn distinct(inner: NodeExpression) -> NodeExpression {
        NodeExpression::Distinct(Box::new(inner))
    }

    /// A limit expression.
    pub fn limit(inner: NodeExpression, limit: usize) -> NodeExpression {
        NodeExpression::Limit {
            expr: Box::new(inner),
            limit,
        }
    }

    /// An offset expression.
    pub fn offset(inner: NodeExpression, offset: usize) -> NodeExpression {
        NodeExpression::Offset {
            expr: Box::new(inner),
            offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> InMemoryGraph {
        let mut g = InMemoryGraph::new();
        // Alice knows Bob, Charlie; Bob knows Diana
        g.add_triple(
            NodeTerm::iri("http://ex.org/alice"),
            NodeTerm::iri("http://ex.org/knows"),
            NodeTerm::iri("http://ex.org/bob"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/alice"),
            NodeTerm::iri("http://ex.org/knows"),
            NodeTerm::iri("http://ex.org/charlie"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/bob"),
            NodeTerm::iri("http://ex.org/knows"),
            NodeTerm::iri("http://ex.org/diana"),
        );
        // Names
        g.add_triple(
            NodeTerm::iri("http://ex.org/alice"),
            NodeTerm::iri("http://ex.org/name"),
            NodeTerm::literal("Alice"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/bob"),
            NodeTerm::iri("http://ex.org/name"),
            NodeTerm::literal("Bob"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/charlie"),
            NodeTerm::iri("http://ex.org/name"),
            NodeTerm::literal("Charlie"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/diana"),
            NodeTerm::iri("http://ex.org/name"),
            NodeTerm::literal("Diana"),
        );
        // Ages
        g.add_triple(
            NodeTerm::iri("http://ex.org/alice"),
            NodeTerm::iri("http://ex.org/age"),
            NodeTerm::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        g.add_triple(
            NodeTerm::iri("http://ex.org/bob"),
            NodeTerm::iri("http://ex.org/age"),
            NodeTerm::typed_literal("25", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        g
    }

    fn alice() -> NodeTerm {
        NodeTerm::iri("http://ex.org/alice")
    }

    fn bob() -> NodeTerm {
        NodeTerm::iri("http://ex.org/bob")
    }

    // ── FocusNode ─────────────────────────────────────────────────────────

    #[test]
    fn test_focus_node() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let result = eval
            .evaluate(&NodeExpression::FocusNode, &alice(), &g)
            .expect("eval");
        assert_eq!(result, vec![alice()]);
    }

    // ── Constant ──────────────────────────────────────────────────────────

    #[test]
    fn test_constant() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let c = NodeTerm::literal("hello");
        let result = eval
            .evaluate(&NodeExpression::Constant(c.clone()), &alice(), &g)
            .expect("eval");
        assert_eq!(result, vec![c]);
    }

    // ── Path ──────────────────────────────────────────────────────────────

    #[test]
    fn test_path_simple_predicate() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::path("http://ex.org/knows");
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 2);
        assert!(result.contains(&bob()));
        assert!(result.contains(&NodeTerm::iri("http://ex.org/charlie")));
    }

    #[test]
    fn test_path_no_results() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::path("http://ex.org/nonexistent");
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert!(result.is_empty());
    }

    #[test]
    fn test_path_inverse() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Path(PropertyPath::Inverse(Box::new(PropertyPath::Predicate(
            "http://ex.org/knows".to_string(),
        ))));
        let result = eval.evaluate(&expr, &bob(), &g).expect("eval");
        assert_eq!(result, vec![alice()]);
    }

    #[test]
    fn test_path_sequence() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Alice -> knows -> knows (should reach Diana via Bob)
        let expr = ExprBuilder::sequence_path(&["http://ex.org/knows", "http://ex.org/knows"]);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
    }

    #[test]
    fn test_path_alternative() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Path(PropertyPath::Alternative(vec![
            PropertyPath::Predicate("http://ex.org/knows".to_string()),
            PropertyPath::Predicate("http://ex.org/name".to_string()),
        ]));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // Should find bob, charlie (from knows) and "Alice" (from name)
        assert!(result.len() >= 3);
    }

    #[test]
    fn test_path_zero_or_more() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Path(PropertyPath::ZeroOrMore(Box::new(
            PropertyPath::Predicate("http://ex.org/knows".to_string()),
        )));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // alice (start) + bob + charlie + diana (via bob)
        assert!(result.contains(&alice())); // zero steps
        assert!(result.contains(&bob()));
        assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
    }

    #[test]
    fn test_path_one_or_more() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Path(PropertyPath::OneOrMore(Box::new(
            PropertyPath::Predicate("http://ex.org/knows".to_string()),
        )));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // Should NOT include alice (requires at least one step)
        assert!(!result.contains(&alice()));
        assert!(result.contains(&bob()));
        assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
    }

    #[test]
    fn test_path_zero_or_one() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Path(PropertyPath::ZeroOrOne(Box::new(
            PropertyPath::Predicate("http://ex.org/knows".to_string()),
        )));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // alice (zero) + bob + charlie (one)
        assert!(result.contains(&alice()));
        assert!(result.contains(&bob()));
    }

    // ── Intersection ──────────────────────────────────────────────────────

    #[test]
    fn test_intersection() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::intersection(vec![
            ExprBuilder::path("http://ex.org/knows"),
            NodeExpression::Union(vec![
                NodeExpression::Constant(bob()),
                NodeExpression::Constant(NodeTerm::iri("http://ex.org/unknown")),
            ]),
        ]);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result, vec![bob()]);
    }

    #[test]
    fn test_intersection_empty() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::Intersection(vec![]);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert!(result.is_empty());
    }

    // ── Union ─────────────────────────────────────────────────────────────

    #[test]
    fn test_union() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::union(vec![
            ExprBuilder::path("http://ex.org/knows"),
            ExprBuilder::path("http://ex.org/name"),
        ]);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert!(result.len() >= 3); // bob, charlie, "Alice"
    }

    // ── Minus ─────────────────────────────────────────────────────────────

    #[test]
    fn test_minus() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Alice knows bob and charlie, minus bob
        let expr = ExprBuilder::minus(
            ExprBuilder::path("http://ex.org/knows"),
            NodeExpression::Constant(bob()),
        );
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], NodeTerm::iri("http://ex.org/charlie"));
    }

    // ── IfThenElse ────────────────────────────────────────────────────────

    #[test]
    fn test_if_then_else_true() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Alice has knows relations -> condition is non-empty -> then
        let expr = ExprBuilder::if_then_else(
            ExprBuilder::path("http://ex.org/knows"),
            NodeExpression::Constant(NodeTerm::literal("has friends")),
            NodeExpression::Constant(NodeTerm::literal("no friends")),
        );
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result[0], NodeTerm::literal("has friends"));
    }

    #[test]
    fn test_if_then_else_false() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Diana has no knows relations -> condition is empty -> else
        let expr = ExprBuilder::if_then_else(
            ExprBuilder::path("http://ex.org/knows"),
            NodeExpression::Constant(NodeTerm::literal("has friends")),
            NodeExpression::Constant(NodeTerm::literal("no friends")),
        );
        let diana = NodeTerm::iri("http://ex.org/diana");
        let result = eval.evaluate(&expr, &diana, &g).expect("eval");
        assert_eq!(result[0], NodeTerm::literal("no friends"));
    }

    // ── FilterShape ───────────────────────────────────────────────────────

    #[test]
    fn test_filter_shape() {
        let mut g = sample_graph();
        g.set_conforms("http://ex.org/bob", "http://ex.org/AdultShape", true);
        g.set_conforms("http://ex.org/charlie", "http://ex.org/AdultShape", false);

        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FilterShape {
            nodes: Box::new(ExprBuilder::path("http://ex.org/knows")),
            shape: "http://ex.org/AdultShape".to_string(),
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result, vec![bob()]);
    }

    // ── Count ─────────────────────────────────────────────────────────────

    #[test]
    fn test_count() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::count(ExprBuilder::path("http://ex.org/knows"));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_integer(), Some(2));
    }

    #[test]
    fn test_count_zero() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let diana = NodeTerm::iri("http://ex.org/diana");
        let expr = ExprBuilder::count(ExprBuilder::path("http://ex.org/knows"));
        let result = eval.evaluate(&expr, &diana, &g).expect("eval");
        assert_eq!(result[0].as_integer(), Some(0));
    }

    // ── Distinct ──────────────────────────────────────────────────────────

    #[test]
    fn test_distinct() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Union of same expression twice -> distinct should deduplicate
        let expr = ExprBuilder::distinct(ExprBuilder::union(vec![
            ExprBuilder::path("http://ex.org/knows"),
            ExprBuilder::path("http://ex.org/knows"),
        ]));
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 2); // bob and charlie, deduplicated
    }

    // ── Limit/Offset ─────────────────────────────────────────────────────

    #[test]
    fn test_limit() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::limit(ExprBuilder::path("http://ex.org/knows"), 1);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_offset() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = ExprBuilder::offset(ExprBuilder::path("http://ex.org/knows"), 1);
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1); // 2 results minus 1 offset
    }

    #[test]
    fn test_limit_and_offset() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        // Get knows results, skip 0, take 1
        let expr = ExprBuilder::limit(
            ExprBuilder::offset(ExprBuilder::path("http://ex.org/knows"), 0),
            1,
        );
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1);
    }

    // ── OrderBy ───────────────────────────────────────────────────────────

    #[test]
    fn test_order_by_ascending() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::OrderBy {
            expr: Box::new(ExprBuilder::path("http://ex.org/knows")),
            sort_path: PropertyPath::Predicate("http://ex.org/name".to_string()),
            ascending: true,
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // Bob and Charlie, sorted by name ascending
        assert_eq!(result.len(), 2);
        // "Bob" < "Charlie" alphabetically
        assert_eq!(result[0], bob());
    }

    #[test]
    fn test_order_by_descending() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::OrderBy {
            expr: Box::new(ExprBuilder::path("http://ex.org/knows")),
            sort_path: PropertyPath::Predicate("http://ex.org/name".to_string()),
            ascending: false,
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result[0], NodeTerm::iri("http://ex.org/charlie"));
    }

    // ── GroupConcat ───────────────────────────────────────────────────────

    #[test]
    fn test_group_concat() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::GroupConcat {
            expr: Box::new(ExprBuilder::path("http://ex.org/name")),
            separator: ", ".to_string(),
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_str(), "Alice");
    }

    // ── Function calls ────────────────────────────────────────────────────

    #[test]
    fn test_function_strlen() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FunctionCall {
            function: "sh:strlen".to_string(),
            args: vec![ExprBuilder::path("http://ex.org/name")],
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        // "Alice" -> strlen = 5
        assert_eq!(result[0].as_integer(), Some(5));
    }

    #[test]
    fn test_function_concat() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FunctionCall {
            function: "sh:concat".to_string(),
            args: vec![
                ExprBuilder::path("http://ex.org/name"),
                NodeExpression::Constant(NodeTerm::literal("!")),
            ],
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result[0].as_str(), "Alice!");
    }

    #[test]
    fn test_function_unknown() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FunctionCall {
            function: "sh:unknown_fn".to_string(),
            args: vec![],
        };
        let result = eval.evaluate(&expr, &alice(), &g);
        assert!(matches!(result, Err(NodeExprError::UnknownFunction(_))));
    }

    #[test]
    fn test_function_wrong_arg_count() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FunctionCall {
            function: "sh:strlen".to_string(),
            args: vec![
                NodeExpression::Constant(NodeTerm::literal("a")),
                NodeExpression::Constant(NodeTerm::literal("b")),
            ],
        };
        let result = eval.evaluate(&expr, &alice(), &g);
        assert!(matches!(result, Err(NodeExprError::InvalidArgCount { .. })));
    }

    // ── Custom function registration ──────────────────────────────────────

    #[test]
    fn test_register_custom_function() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();

        fn my_upper(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
            Ok(args[0]
                .iter()
                .map(|t| NodeTerm::literal(t.as_str().to_uppercase()))
                .collect())
        }

        eval.register_function("ex:upper", 1, my_upper);
        let expr = NodeExpression::FunctionCall {
            function: "ex:upper".to_string(),
            args: vec![ExprBuilder::path("http://ex.org/name")],
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result[0].as_str(), "ALICE");
    }

    // ── NodeTerm tests ────────────────────────────────────────────────────

    #[test]
    fn test_node_term_is_iri() {
        assert!(NodeTerm::iri("http://ex.org/x").is_iri());
        assert!(!NodeTerm::literal("hello").is_iri());
    }

    #[test]
    fn test_node_term_is_literal() {
        assert!(NodeTerm::literal("hello").is_literal());
        assert!(!NodeTerm::iri("http://ex.org/x").is_literal());
    }

    #[test]
    fn test_node_term_as_integer() {
        assert_eq!(
            NodeTerm::typed_literal("42", "xsd:int").as_integer(),
            Some(42)
        );
        assert_eq!(NodeTerm::literal("not_a_number").as_integer(), None);
    }

    #[test]
    fn test_node_term_as_float() {
        assert!(
            (NodeTerm::typed_literal("3.125", "xsd:double")
                .as_float()
                .expect("ok")
                - 3.125)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_node_term_display() {
        assert_eq!(
            format!("{}", NodeTerm::iri("http://ex.org/x")),
            "<http://ex.org/x>"
        );
        assert_eq!(format!("{}", NodeTerm::literal("hello")), "\"hello\"");
        assert_eq!(
            format!("{}", NodeTerm::typed_literal("42", "xsd:int")),
            "\"42\"^^<xsd:int>"
        );
        assert_eq!(
            format!("{}", NodeTerm::lang_literal("hello", "en")),
            "\"hello\"@en"
        );
        assert_eq!(format!("{}", NodeTerm::BlankNode("b0".to_string())), "_:b0");
    }

    // ── PropertyPath display ──────────────────────────────────────────────

    #[test]
    fn test_property_path_display() {
        let p = PropertyPath::Predicate("http://ex.org/p".to_string());
        assert_eq!(format!("{p}"), "<http://ex.org/p>");

        let inv = PropertyPath::Inverse(Box::new(p.clone()));
        assert_eq!(format!("{inv}"), "^<http://ex.org/p>");

        let seq = PropertyPath::Sequence(vec![p.clone(), p.clone()]);
        assert!(format!("{seq}").contains(" / "));

        let alt = PropertyPath::Alternative(vec![p.clone(), p.clone()]);
        assert!(format!("{alt}").contains(" | "));

        let star = PropertyPath::ZeroOrMore(Box::new(p.clone()));
        assert!(format!("{star}").ends_with('*'));

        let plus = PropertyPath::OneOrMore(Box::new(p.clone()));
        assert!(format!("{plus}").ends_with('+'));

        let opt = PropertyPath::ZeroOrOne(Box::new(p));
        assert!(format!("{opt}").ends_with('?'));
    }

    // ── Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = NodeExprError::PathDepthExceeded {
            depth: 101,
            max: 100,
        };
        assert!(format!("{e}").contains("101"));

        let e = NodeExprError::ResultLimitExceeded {
            count: 10001,
            max: 10000,
        };
        assert!(format!("{e}").contains("10001"));

        let e = NodeExprError::UnknownFunction("foo".to_string());
        assert!(format!("{e}").contains("foo"));

        let e = NodeExprError::InvalidArgCount {
            function: "bar".to_string(),
            expected: 2,
            got: 1,
        };
        assert!(format!("{e}").contains("bar"));

        let e = NodeExprError::TypeError("bad type".to_string());
        assert!(format!("{e}").contains("bad type"));

        let e = NodeExprError::EmptySequence("test".to_string());
        assert!(format!("{e}").contains("test"));
    }

    // ── InMemoryGraph ─────────────────────────────────────────────────────

    #[test]
    fn test_in_memory_graph_len() {
        let g = sample_graph();
        assert_eq!(g.len(), 9);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_in_memory_graph_empty() {
        let g = InMemoryGraph::new();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_in_memory_graph_all_triples() {
        let g = sample_graph();
        assert_eq!(g.all_triples().len(), 9);
    }

    // ── EvalConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_eval_config_default() {
        let config = EvalConfig::default();
        assert_eq!(config.max_path_depth, 100);
        assert_eq!(config.max_results, 10000);
        assert!(!config.deduplicate);
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let _ = eval.evaluate(&ExprBuilder::path("http://ex.org/knows"), &alice(), &g);
        assert!(eval.stats().path_traversals > 0);
        assert!(eval.stats().nodes_evaluated > 0);
    }

    #[test]
    fn test_stats_reset() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let _ = eval.evaluate(&ExprBuilder::path("http://ex.org/knows"), &alice(), &g);
        eval.reset_stats();
        assert_eq!(eval.stats().path_traversals, 0);
    }

    // ── ExprBuilder ───────────────────────────────────────────────────────

    #[test]
    fn test_expr_builder_focus() {
        assert_eq!(ExprBuilder::focus(), NodeExpression::FocusNode);
    }

    #[test]
    fn test_expr_builder_constant() {
        let c = NodeTerm::literal("x");
        assert_eq!(
            ExprBuilder::constant(c.clone()),
            NodeExpression::Constant(c)
        );
    }

    #[test]
    fn test_expr_builder_path() {
        let expr = ExprBuilder::path("http://ex.org/p");
        assert!(matches!(
            expr,
            NodeExpression::Path(PropertyPath::Predicate(_))
        ));
    }

    #[test]
    fn test_expr_builder_sequence_path() {
        let expr = ExprBuilder::sequence_path(&["http://ex.org/p1", "http://ex.org/p2"]);
        assert!(matches!(
            expr,
            NodeExpression::Path(PropertyPath::Sequence(_))
        ));
    }

    // ── Depth limit ───────────────────────────────────────────────────────

    #[test]
    fn test_max_path_depth_exceeded() {
        let g = sample_graph();
        let config = EvalConfig {
            max_path_depth: 1,
            ..Default::default()
        };
        let mut eval = NodeExprEvaluator::with_config(config);
        // Deep nesting: count(count(count(...)))
        let expr = ExprBuilder::count(ExprBuilder::count(ExprBuilder::path("http://ex.org/knows")));
        let result = eval.evaluate(&expr, &alice(), &g);
        assert!(matches!(
            result,
            Err(NodeExprError::PathDepthExceeded { .. })
        ));
    }

    // ── Function: sum ─────────────────────────────────────────────────────

    #[test]
    fn test_function_sum() {
        let g = sample_graph();
        let mut eval = NodeExprEvaluator::new();
        let expr = NodeExpression::FunctionCall {
            function: "sh:sum".to_string(),
            args: vec![ExprBuilder::path("http://ex.org/age")],
        };
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        let val = result[0].as_float().expect("should be float");
        assert!((val - 30.0).abs() < 0.01);
    }

    // ── Complex expression ────────────────────────────────────────────────

    #[test]
    fn test_complex_expression() {
        let mut g = sample_graph();
        g.set_conforms("http://ex.org/bob", "http://ex.org/FriendShape", true);
        g.set_conforms("http://ex.org/charlie", "http://ex.org/FriendShape", true);

        let mut eval = NodeExprEvaluator::new();
        // Count friends that conform to FriendShape
        let expr = ExprBuilder::count(NodeExpression::FilterShape {
            nodes: Box::new(ExprBuilder::path("http://ex.org/knows")),
            shape: "http://ex.org/FriendShape".to_string(),
        });
        let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
        assert_eq!(result[0].as_integer(), Some(2));
    }
}
