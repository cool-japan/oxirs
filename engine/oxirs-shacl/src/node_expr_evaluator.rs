//! Evaluator for SHACL node expressions.
//!
//! This module contains [`NodeExprEvaluator`], which walks a [`NodeExpression`]
//! AST against an arbitrary [`NodeExprGraph`] implementation. It is responsible
//! for property-path resolution, set operations, conditional dispatch, function
//! call execution, and pagination / ordering operators.
//!
//! Built-in function bodies live in
//! [`super::node_expr_builtins`] so this module can
//! stay focused on evaluation control flow without growing past the workspace
//! 2000-line refactor threshold.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use super::node_expr_builtins::register_builtins;
use super::node_expr_types::{
    EvalConfig, EvalStats, NodeExprError, NodeExprGraph, NodeExpression, NodeTerm, PropertyPath,
};

// ---------------------------------------------------------------------------
// Built-in function infrastructure
// ---------------------------------------------------------------------------

/// Function signature for node expression built-in functions.
pub type NodeExprFn = fn(&[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError>;

/// A built-in function for node expressions.
#[derive(Clone)]
pub(crate) struct BuiltinFunction {
    /// Number of expected arguments.
    pub(crate) arg_count: usize,
    /// The function implementation.
    pub(crate) eval: NodeExprFn,
}

impl BuiltinFunction {
    /// Construct a new built-in function descriptor.
    pub(crate) fn new(arg_count: usize, eval: NodeExprFn) -> Self {
        Self { arg_count, eval }
    }
}

impl fmt::Debug for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinFunction")
            .field("arg_count", &self.arg_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Evaluates SHACL node expressions against a graph.
pub struct NodeExprEvaluator {
    config: EvalConfig,
    stats: EvalStats,
    /// Built-in functions registry.
    functions: HashMap<String, BuiltinFunction>,
}

impl NodeExprEvaluator {
    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self::with_config(EvalConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: EvalConfig) -> Self {
        let mut functions = HashMap::new();
        register_builtins(&mut functions);

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
        self.functions
            .insert(name.into(), BuiltinFunction::new(arg_count, eval_fn));
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
