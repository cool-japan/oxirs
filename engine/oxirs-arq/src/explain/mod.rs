//! # SPARQL Query Explain / Plan
//!
//! Provides SQL-style `EXPLAIN` functionality for SPARQL queries: the query
//! optimiser produces a [`QueryPlan`] describing exactly which physical
//! operators will be executed, and a [`QueryExplainer`] renders that plan in
//! three formats (human-readable text, machine-readable JSON, Graphviz DOT).
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_arq::explain::{
//!     QueryPlan, PlanNode, IndexType, QueryExplainer, ExplainFormat,
//! };
//!
//! let plan = QueryPlan {
//!     root: PlanNode::TripleScan {
//!         pattern: "?s rdf:type ?t".to_string(),
//!         index_used: IndexType::Spo,
//!         estimated_rows: 500,
//!     },
//!     estimated_cost: 1.5,
//!     estimated_cardinality: 500,
//! };
//!
//! let explainer = QueryExplainer::new();
//! let text = explainer.explain(&plan);
//! assert!(text.contains("TripleScan"));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Public types ──────────────────────────────────────────────────────────────

/// A complete, annotated query execution plan.
///
/// The plan is a tree of [`PlanNode`]s that mirrors the physical execution
/// order decided by the optimizer.  Each node carries its own cost / row
/// estimate so callers can identify expensive sub-trees without executing the
/// query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Root operator of the physical plan tree.
    pub root: PlanNode,
    /// Aggregate estimated cost for the entire plan (abstract cost units).
    pub estimated_cost: f64,
    /// Estimated number of result rows produced by the root operator.
    pub estimated_cardinality: u64,
}

/// A single physical operator node in the query plan tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlanNode {
    /// Sequential / index scan over a triple pattern.
    TripleScan {
        /// Human-readable representation of the triple pattern.
        pattern: String,
        /// Which triple-store index was selected for this scan.
        index_used: IndexType,
        /// Estimated number of matching triples.
        estimated_rows: u64,
    },
    /// Classic hash-join: build a hash-table from the smaller side, probe with
    /// the larger side.
    HashJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        /// Shared variables used as the join key.
        join_vars: Vec<String>,
    },
    /// Nested-loop join; preferred when the inner side is very small or
    /// already indexed.
    NestedLoopJoin {
        outer: Box<PlanNode>,
        inner: Box<PlanNode>,
    },
    /// SPARQL FILTER expression applied on top of a child operator.
    Filter {
        /// String representation of the filter expression.
        expr: String,
        child: Box<PlanNode>,
    },
    /// ORDER BY clause.
    Sort {
        /// Ordered list of sort keys (prefixed with `+` / `-`).
        vars: Vec<String>,
        child: Box<PlanNode>,
    },
    /// LIMIT / OFFSET clause.
    Limit {
        limit: usize,
        offset: usize,
        child: Box<PlanNode>,
    },
    /// DISTINCT post-processing.
    Distinct { child: Box<PlanNode> },
    /// SPARQL UNION: evaluate both branches and concatenate results.
    Union {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
    },
    /// OPTIONAL (left outer join).
    Optional {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
    },
    /// GROUP BY / aggregate evaluation.
    Aggregate {
        /// Grouping key variables.
        group_by: Vec<String>,
        /// Aggregate expressions (e.g. `"COUNT(?x) AS ?count"`).
        aggs: Vec<String>,
        child: Box<PlanNode>,
    },
    /// A full sub-query (SELECT inside SELECT).
    Subquery { plan: Box<QueryPlan> },
    /// A SPARQL 1.1 property path evaluation.
    PropertyPath {
        subject: String,
        path: String,
        object: String,
    },
    /// Federated SERVICE clause: the sub-plan is evaluated on a remote endpoint.
    Service {
        endpoint: String,
        subplan: Box<QueryPlan>,
    },
    /// Merge-join on a pre-sorted index.
    MergeJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_vars: Vec<String>,
    },
    /// Pre-computed VALUES clause materialised as a table scan.
    ValuesScan { vars: Vec<String>, row_count: usize },
    /// GRAPH clause – scoped to a named graph.
    NamedGraph { graph: String, child: Box<PlanNode> },
}

/// Which physical triple-store index was chosen for a [`PlanNode::TripleScan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum IndexType {
    /// Subject → Predicate → Object (fastest for subject-bound patterns).
    Spo,
    /// Predicate → Object → Subject (fast for predicate / object lookups).
    Pos,
    /// Object → Subject → Predicate (fast when only object is bound).
    Osp,
    /// No useful index; a full store scan will be performed.
    FullScan,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spo => write!(f, "SPO"),
            Self::Pos => write!(f, "POS"),
            Self::Osp => write!(f, "OSP"),
            Self::FullScan => write!(f, "FULL_SCAN"),
        }
    }
}

/// Output format requested from [`QueryExplainer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplainFormat {
    /// Indented text tree, easy to read in a terminal.
    Text,
    /// Serialised JSON object – suitable for tooling and APIs.
    Json,
    /// Graphviz DOT language – pipe into `dot -Tpng` to produce a diagram.
    Dot,
}

/// Renders a [`QueryPlan`] in the requested [`ExplainFormat`].
///
/// ```rust
/// use oxirs_arq::explain::{QueryExplainer, ExplainFormat, QueryPlan, PlanNode, IndexType};
///
/// let plan = QueryPlan {
///     root: PlanNode::TripleScan {
///         pattern: "?s ?p ?o".into(),
///         index_used: IndexType::FullScan,
///         estimated_rows: 100_000,
///     },
///     estimated_cost: 12.5,
///     estimated_cardinality: 100_000,
/// };
/// let exp = QueryExplainer::builder().show_estimates(true).build();
/// let out = exp.explain_with_format(&plan, ExplainFormat::Json);
/// assert!(out.contains("\"type\""));
/// ```
#[derive(Debug, Clone)]
pub struct QueryExplainer {
    show_estimates: bool,
    show_costs: bool,
    format: ExplainFormat,
}

impl Default for QueryExplainer {
    fn default() -> Self {
        Self {
            show_estimates: true,
            show_costs: true,
            format: ExplainFormat::Text,
        }
    }
}

impl QueryExplainer {
    /// Create a new explainer with default settings (text format, all annotations shown).
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a builder for fine-grained configuration.
    pub fn builder() -> QueryExplainerBuilder {
        QueryExplainerBuilder::default()
    }

    /// Render the plan using the format configured at construction time.
    pub fn explain(&self, plan: &QueryPlan) -> String {
        match self.format {
            ExplainFormat::Text => self.explain_text(plan),
            ExplainFormat::Json => self.explain_json(plan),
            ExplainFormat::Dot => self.explain_dot(plan),
        }
    }

    /// Render the plan in a specific format, overriding the configured default.
    pub fn explain_with_format(&self, plan: &QueryPlan, format: ExplainFormat) -> String {
        match format {
            ExplainFormat::Text => self.explain_text(plan),
            ExplainFormat::Json => self.explain_json(plan),
            ExplainFormat::Dot => self.explain_dot(plan),
        }
    }

    // ── Text format ───────────────────────────────────────────────────────────

    /// Render as an indented text tree.
    pub fn explain_text(&self, plan: &QueryPlan) -> String {
        let mut out = String::new();
        out.push_str("Query Plan\n");
        out.push_str("==========\n");
        if self.show_costs {
            out.push_str(&format!(
                "Estimated cost        : {:.4}\n",
                plan.estimated_cost
            ));
        }
        if self.show_estimates {
            out.push_str(&format!(
                "Estimated cardinality : {}\n",
                plan.estimated_cardinality
            ));
        }
        out.push('\n');
        self.format_node_text(&plan.root, &mut out, 0);
        out
    }

    fn format_node_text(&self, node: &PlanNode, out: &mut String, depth: usize) {
        let indent = "  ".repeat(depth);
        let prefix = if depth == 0 {
            String::new()
        } else {
            format!("{indent}└─ ")
        };

        match node {
            PlanNode::TripleScan {
                pattern,
                index_used,
                estimated_rows,
            } => {
                out.push_str(&format!("{prefix}TripleScan\n"));
                let child_indent = "  ".repeat(depth + 1);
                out.push_str(&format!("{child_indent}   pattern   : {pattern}\n"));
                out.push_str(&format!("{child_indent}   index     : {index_used}\n"));
                if self.show_estimates {
                    out.push_str(&format!("{child_indent}   est. rows : {estimated_rows}\n"));
                }
            }
            PlanNode::HashJoin {
                left,
                right,
                join_vars,
            } => {
                out.push_str(&format!(
                    "{prefix}HashJoin  [key: {}]\n",
                    join_vars.join(", ")
                ));
                self.format_node_text(left, out, depth + 1);
                self.format_node_text(right, out, depth + 1);
            }
            PlanNode::NestedLoopJoin { outer, inner } => {
                out.push_str(&format!("{prefix}NestedLoopJoin\n"));
                self.format_node_text(outer, out, depth + 1);
                self.format_node_text(inner, out, depth + 1);
            }
            PlanNode::Filter { expr, child } => {
                out.push_str(&format!("{prefix}Filter  [{expr}]\n"));
                self.format_node_text(child, out, depth + 1);
            }
            PlanNode::Sort { vars, child } => {
                out.push_str(&format!("{prefix}Sort  [{}]\n", vars.join(", ")));
                self.format_node_text(child, out, depth + 1);
            }
            PlanNode::Limit {
                limit,
                offset,
                child,
            } => {
                out.push_str(&format!("{prefix}Limit  [{limit} offset {offset}]\n"));
                self.format_node_text(child, out, depth + 1);
            }
            PlanNode::Distinct { child } => {
                out.push_str(&format!("{prefix}Distinct\n"));
                self.format_node_text(child, out, depth + 1);
            }
            PlanNode::Union { left, right } => {
                out.push_str(&format!("{prefix}Union\n"));
                self.format_node_text(left, out, depth + 1);
                self.format_node_text(right, out, depth + 1);
            }
            PlanNode::Optional { left, right } => {
                out.push_str(&format!("{prefix}Optional\n"));
                self.format_node_text(left, out, depth + 1);
                self.format_node_text(right, out, depth + 1);
            }
            PlanNode::Aggregate {
                group_by,
                aggs,
                child,
            } => {
                out.push_str(&format!(
                    "{prefix}Aggregate  [group: {}]  [aggs: {}]\n",
                    group_by.join(", "),
                    aggs.join(", ")
                ));
                self.format_node_text(child, out, depth + 1);
            }
            PlanNode::Subquery { plan } => {
                out.push_str(&format!("{prefix}Subquery\n"));
                let child_indent = "  ".repeat(depth + 1);
                if self.show_costs {
                    out.push_str(&format!(
                        "{child_indent}   est. cost : {:.4}\n",
                        plan.estimated_cost
                    ));
                }
                if self.show_estimates {
                    out.push_str(&format!(
                        "{child_indent}   est. card : {}\n",
                        plan.estimated_cardinality
                    ));
                }
                self.format_node_text(&plan.root, out, depth + 1);
            }
            PlanNode::PropertyPath {
                subject,
                path,
                object,
            } => {
                out.push_str(&format!(
                    "{prefix}PropertyPath  [{subject} {path} {object}]\n"
                ));
            }
            PlanNode::Service { endpoint, subplan } => {
                out.push_str(&format!("{prefix}Service  [{endpoint}]\n"));
                let child_indent = "  ".repeat(depth + 1);
                if self.show_costs {
                    out.push_str(&format!(
                        "{child_indent}   est. cost : {:.4}\n",
                        subplan.estimated_cost
                    ));
                }
                self.format_node_text(&subplan.root, out, depth + 1);
            }
            PlanNode::MergeJoin {
                left,
                right,
                join_vars,
            } => {
                out.push_str(&format!(
                    "{prefix}MergeJoin  [key: {}]\n",
                    join_vars.join(", ")
                ));
                self.format_node_text(left, out, depth + 1);
                self.format_node_text(right, out, depth + 1);
            }
            PlanNode::ValuesScan { vars, row_count } => {
                out.push_str(&format!(
                    "{prefix}ValuesScan  [vars: {}]  [{row_count} rows]\n",
                    vars.join(", ")
                ));
            }
            PlanNode::NamedGraph { graph, child } => {
                out.push_str(&format!("{prefix}NamedGraph  [{graph}]\n"));
                self.format_node_text(child, out, depth + 1);
            }
        }
    }

    // ── JSON format ───────────────────────────────────────────────────────────

    /// Render as a JSON object.
    pub fn explain_json(&self, plan: &QueryPlan) -> String {
        match serde_json::to_string_pretty(plan) {
            Ok(s) => s,
            Err(e) => format!("{{\"error\": \"{e}\"}}"),
        }
    }

    // ── DOT format ────────────────────────────────────────────────────────────

    /// Render as Graphviz DOT language.
    pub fn explain_dot(&self, plan: &QueryPlan) -> String {
        let mut state = DotState::default();
        let mut out = String::new();
        out.push_str("digraph QueryPlan {\n");
        out.push_str("  node [shape=box fontname=\"Helvetica\" fontsize=10];\n");
        out.push_str("  rankdir=TB;\n");

        // Root plan label node
        let root_id = state.next_id();
        if self.show_costs {
            out.push_str(&format!(
                "  {root_id} [label=\"QueryPlan\\ncost={:.4}\\ncard={}\"];\n",
                plan.estimated_cost, plan.estimated_cardinality
            ));
        } else {
            out.push_str(&format!("  {root_id} [label=\"QueryPlan\"];\n"));
        }

        let child_id = self.emit_dot_node(&plan.root, &mut state, &mut out);
        out.push_str(&format!("  {root_id} -> {child_id};\n"));

        out.push_str("}\n");
        out
    }

    fn emit_dot_node(&self, node: &PlanNode, state: &mut DotState, out: &mut String) -> usize {
        let id = state.next_id();
        match node {
            PlanNode::TripleScan {
                pattern,
                index_used,
                estimated_rows,
            } => {
                let label = if self.show_estimates {
                    format!("TripleScan\\n{pattern}\\nidx={index_used}\\nrows={estimated_rows}")
                } else {
                    format!("TripleScan\\n{pattern}\\nidx={index_used}")
                };
                out.push_str(&format!("  {id} [label=\"{label}\"];\n"));
            }
            PlanNode::HashJoin {
                left,
                right,
                join_vars,
            } => {
                let vars = join_vars.join(",");
                out.push_str(&format!("  {id} [label=\"HashJoin\\nkey={vars}\"];\n"));
                let l = self.emit_dot_node(left, state, out);
                let r = self.emit_dot_node(right, state, out);
                out.push_str(&format!("  {id} -> {l} [label=\"left\"];\n"));
                out.push_str(&format!("  {id} -> {r} [label=\"right\"];\n"));
            }
            PlanNode::NestedLoopJoin { outer, inner } => {
                out.push_str(&format!("  {id} [label=\"NestedLoopJoin\"];\n"));
                let o = self.emit_dot_node(outer, state, out);
                let i = self.emit_dot_node(inner, state, out);
                out.push_str(&format!("  {id} -> {o} [label=\"outer\"];\n"));
                out.push_str(&format!("  {id} -> {i} [label=\"inner\"];\n"));
            }
            PlanNode::Filter { expr, child } => {
                out.push_str(&format!("  {id} [label=\"Filter\\n{expr}\"];\n"));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::Sort { vars, child } => {
                let keys = vars.join(",");
                out.push_str(&format!("  {id} [label=\"Sort\\n{keys}\"];\n"));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::Limit {
                limit,
                offset,
                child,
            } => {
                out.push_str(&format!(
                    "  {id} [label=\"Limit {limit}\\noffset {offset}\"];\n"
                ));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::Distinct { child } => {
                out.push_str(&format!("  {id} [label=\"Distinct\"];\n"));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::Union { left, right } => {
                out.push_str(&format!("  {id} [label=\"Union\"];\n"));
                let l = self.emit_dot_node(left, state, out);
                let r = self.emit_dot_node(right, state, out);
                out.push_str(&format!("  {id} -> {l} [label=\"left\"];\n"));
                out.push_str(&format!("  {id} -> {r} [label=\"right\"];\n"));
            }
            PlanNode::Optional { left, right } => {
                out.push_str(&format!("  {id} [label=\"Optional\"];\n"));
                let l = self.emit_dot_node(left, state, out);
                let r = self.emit_dot_node(right, state, out);
                out.push_str(&format!("  {id} -> {l} [label=\"left\"];\n"));
                out.push_str(&format!("  {id} -> {r} [label=\"right\"];\n"));
            }
            PlanNode::Aggregate {
                group_by,
                aggs,
                child,
            } => {
                let gb = group_by.join(",");
                let ag = aggs.join(",");
                out.push_str(&format!(
                    "  {id} [label=\"Aggregate\\ngroup={gb}\\naggs={ag}\"];\n"
                ));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::Subquery { plan } => {
                out.push_str(&format!("  {id} [label=\"Subquery\"];\n"));
                let c = self.emit_dot_node(&plan.root, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::PropertyPath {
                subject,
                path,
                object,
            } => {
                out.push_str(&format!(
                    "  {id} [label=\"PropertyPath\\n{subject} {path} {object}\"];\n"
                ));
            }
            PlanNode::Service { endpoint, subplan } => {
                out.push_str(&format!("  {id} [label=\"Service\\n{endpoint}\"];\n"));
                let c = self.emit_dot_node(&subplan.root, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
            PlanNode::MergeJoin {
                left,
                right,
                join_vars,
            } => {
                let vars = join_vars.join(",");
                out.push_str(&format!("  {id} [label=\"MergeJoin\\nkey={vars}\"];\n"));
                let l = self.emit_dot_node(left, state, out);
                let r = self.emit_dot_node(right, state, out);
                out.push_str(&format!("  {id} -> {l} [label=\"left\"];\n"));
                out.push_str(&format!("  {id} -> {r} [label=\"right\"];\n"));
            }
            PlanNode::ValuesScan { vars, row_count } => {
                let v = vars.join(",");
                out.push_str(&format!(
                    "  {id} [label=\"ValuesScan\\nvars={v}\\nrows={row_count}\"];\n"
                ));
            }
            PlanNode::NamedGraph { graph, child } => {
                out.push_str(&format!("  {id} [label=\"NamedGraph\\n{graph}\"];\n"));
                let c = self.emit_dot_node(child, state, out);
                out.push_str(&format!("  {id} -> {c};\n"));
            }
        }
        id
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`QueryExplainer`].
#[derive(Debug, Clone, Default)]
pub struct QueryExplainerBuilder {
    show_estimates: Option<bool>,
    show_costs: Option<bool>,
    format: Option<ExplainFormat>,
}

impl QueryExplainerBuilder {
    /// Whether to include estimated row counts in the output.
    pub fn show_estimates(mut self, val: bool) -> Self {
        self.show_estimates = Some(val);
        self
    }

    /// Whether to include estimated cost values in the output.
    pub fn show_costs(mut self, val: bool) -> Self {
        self.show_costs = Some(val);
        self
    }

    /// Set the default output format.
    pub fn format(mut self, fmt: ExplainFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Build the [`QueryExplainer`].
    pub fn build(self) -> QueryExplainer {
        QueryExplainer {
            show_estimates: self.show_estimates.unwrap_or(true),
            show_costs: self.show_costs.unwrap_or(true),
            format: self.format.unwrap_or(ExplainFormat::Text),
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct DotState {
    counter: usize,
}

impl DotState {
    fn next_id(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }
}

// ── Convenience constructors on PlanNode ──────────────────────────────────────

impl PlanNode {
    /// Build a `TripleScan` node.
    pub fn triple_scan(pattern: impl Into<String>, index: IndexType, rows: u64) -> Self {
        Self::TripleScan {
            pattern: pattern.into(),
            index_used: index,
            estimated_rows: rows,
        }
    }

    /// Build a `HashJoin` node.
    pub fn hash_join(left: PlanNode, right: PlanNode, vars: Vec<String>) -> Self {
        Self::HashJoin {
            left: Box::new(left),
            right: Box::new(right),
            join_vars: vars,
        }
    }

    /// Build a `Filter` node.
    pub fn filter(expr: impl Into<String>, child: PlanNode) -> Self {
        Self::Filter {
            expr: expr.into(),
            child: Box::new(child),
        }
    }

    /// Build a `Sort` node.
    pub fn sort(vars: Vec<String>, child: PlanNode) -> Self {
        Self::Sort {
            vars,
            child: Box::new(child),
        }
    }

    /// Build a `Limit` node.
    pub fn limit(limit: usize, offset: usize, child: PlanNode) -> Self {
        Self::Limit {
            limit,
            offset,
            child: Box::new(child),
        }
    }

    /// Build a `Distinct` node.
    pub fn distinct(child: PlanNode) -> Self {
        Self::Distinct {
            child: Box::new(child),
        }
    }

    /// Build a `Union` node.
    pub fn union(left: PlanNode, right: PlanNode) -> Self {
        Self::Union {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Build an `Optional` node.
    pub fn optional(left: PlanNode, right: PlanNode) -> Self {
        Self::Optional {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Count the total number of nodes in the plan tree rooted at this node.
    pub fn node_count(&self) -> usize {
        match self {
            Self::TripleScan { .. } | Self::PropertyPath { .. } | Self::ValuesScan { .. } => 1,
            Self::Distinct { child }
            | Self::Filter { child, .. }
            | Self::Sort { child, .. }
            | Self::Limit { child, .. }
            | Self::NamedGraph { child, .. } => 1 + child.node_count(),
            Self::HashJoin { left, right, .. }
            | Self::NestedLoopJoin {
                outer: left,
                inner: right,
            }
            | Self::Union { left, right }
            | Self::Optional { left, right }
            | Self::MergeJoin { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Self::Aggregate { child, .. } => 1 + child.node_count(),
            Self::Subquery { plan } => 1 + plan.root.node_count(),
            Self::Service { subplan, .. } => 1 + subplan.root.node_count(),
        }
    }

    /// Maximum depth of the plan tree rooted at this node (0 = leaf).
    pub fn depth(&self) -> usize {
        match self {
            Self::TripleScan { .. } | Self::PropertyPath { .. } | Self::ValuesScan { .. } => 0,
            Self::Distinct { child }
            | Self::Filter { child, .. }
            | Self::Sort { child, .. }
            | Self::Limit { child, .. }
            | Self::NamedGraph { child, .. }
            | Self::Aggregate { child, .. } => 1 + child.depth(),
            Self::HashJoin { left, right, .. }
            | Self::NestedLoopJoin {
                outer: left,
                inner: right,
            }
            | Self::Union { left, right }
            | Self::Optional { left, right }
            | Self::MergeJoin { left, right, .. } => 1 + left.depth().max(right.depth()),
            Self::Subquery { plan } => 1 + plan.root.depth(),
            Self::Service { subplan, .. } => 1 + subplan.root.depth(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scan(pattern: &str, index: IndexType, rows: u64) -> PlanNode {
        PlanNode::triple_scan(pattern, index, rows)
    }

    fn make_plan(root: PlanNode) -> QueryPlan {
        let cardinality = 100;
        QueryPlan {
            estimated_cost: 5.0,
            estimated_cardinality: cardinality,
            root,
        }
    }

    // ── Basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_triple_scan_construction() {
        let node = make_scan("?s rdf:type ?t", IndexType::Spo, 500);
        if let PlanNode::TripleScan {
            pattern,
            index_used,
            estimated_rows,
        } = &node
        {
            assert_eq!(pattern, "?s rdf:type ?t");
            assert_eq!(*index_used, IndexType::Spo);
            assert_eq!(*estimated_rows, 500);
        } else {
            panic!("expected TripleScan");
        }
    }

    #[test]
    fn test_hash_join_construction() {
        let left = make_scan("?s ?p ?o", IndexType::FullScan, 1000);
        let right = make_scan("?s foaf:name ?n", IndexType::Spo, 50);
        let join = PlanNode::hash_join(left, right, vec!["?s".to_string()]);
        assert!(matches!(join, PlanNode::HashJoin { .. }));
    }

    #[test]
    fn test_nested_loop_join_construction() {
        let outer = make_scan("?s a ?t", IndexType::Pos, 10);
        let inner = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let node = PlanNode::NestedLoopJoin {
            outer: Box::new(outer),
            inner: Box::new(inner),
        };
        assert!(matches!(node, PlanNode::NestedLoopJoin { .. }));
    }

    #[test]
    fn test_filter_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::filter("?o > 5", scan);
        if let PlanNode::Filter { expr, .. } = &node {
            assert_eq!(expr, "?o > 5");
        } else {
            panic!("expected Filter");
        }
    }

    #[test]
    fn test_sort_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::sort(vec!["+?s".into(), "-?o".into()], scan);
        if let PlanNode::Sort { vars, .. } = &node {
            assert_eq!(vars, &["+?s", "-?o"]);
        } else {
            panic!("expected Sort");
        }
    }

    #[test]
    fn test_limit_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::limit(10, 20, scan);
        if let PlanNode::Limit { limit, offset, .. } = &node {
            assert_eq!(*limit, 10);
            assert_eq!(*offset, 20);
        } else {
            panic!("expected Limit");
        }
    }

    #[test]
    fn test_distinct_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::distinct(scan);
        assert!(matches!(node, PlanNode::Distinct { .. }));
    }

    #[test]
    fn test_union_construction() {
        let left = make_scan("?s a owl:Class", IndexType::Pos, 30);
        let right = make_scan("?s a rdfs:Class", IndexType::Pos, 10);
        let node = PlanNode::union(left, right);
        assert!(matches!(node, PlanNode::Union { .. }));
    }

    #[test]
    fn test_optional_construction() {
        let main = make_scan("?s foaf:name ?n", IndexType::Spo, 200);
        let opt = make_scan("?s foaf:mbox ?m", IndexType::Spo, 80);
        let node = PlanNode::optional(main, opt);
        assert!(matches!(node, PlanNode::Optional { .. }));
    }

    #[test]
    fn test_aggregate_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 1000);
        let node = PlanNode::Aggregate {
            group_by: vec!["?s".into()],
            aggs: vec!["COUNT(?o) AS ?cnt".into()],
            child: Box::new(scan),
        };
        if let PlanNode::Aggregate { aggs, group_by, .. } = &node {
            assert_eq!(group_by[0], "?s");
            assert!(aggs[0].contains("COUNT"));
        } else {
            panic!("expected Aggregate");
        }
    }

    #[test]
    fn test_subquery_construction() {
        let inner_plan = make_plan(make_scan("?x ?y ?z", IndexType::FullScan, 5));
        let node = PlanNode::Subquery {
            plan: Box::new(inner_plan),
        };
        assert!(matches!(node, PlanNode::Subquery { .. }));
    }

    #[test]
    fn test_property_path_construction() {
        let node = PlanNode::PropertyPath {
            subject: "?s".into(),
            path: "foaf:knows+".into(),
            object: "?o".into(),
        };
        if let PlanNode::PropertyPath { path, .. } = &node {
            assert!(path.contains("foaf"));
        } else {
            panic!("expected PropertyPath");
        }
    }

    #[test]
    fn test_service_construction() {
        let sub = make_plan(make_scan("?s ?p ?o", IndexType::FullScan, 50));
        let node = PlanNode::Service {
            endpoint: "http://remote.example.org/sparql".into(),
            subplan: Box::new(sub),
        };
        if let PlanNode::Service { endpoint, .. } = &node {
            assert!(endpoint.contains("remote"));
        } else {
            panic!("expected Service");
        }
    }

    #[test]
    fn test_merge_join_construction() {
        let left = make_scan("?s ?p ?o", IndexType::Spo, 500);
        let right = make_scan("?s a ?t", IndexType::Pos, 100);
        let node = PlanNode::MergeJoin {
            left: Box::new(left),
            right: Box::new(right),
            join_vars: vec!["?s".into()],
        };
        assert!(matches!(node, PlanNode::MergeJoin { .. }));
    }

    #[test]
    fn test_values_scan_construction() {
        let node = PlanNode::ValuesScan {
            vars: vec!["?s".into(), "?p".into()],
            row_count: 3,
        };
        if let PlanNode::ValuesScan { row_count, .. } = &node {
            assert_eq!(*row_count, 3);
        } else {
            panic!("expected ValuesScan");
        }
    }

    #[test]
    fn test_named_graph_construction() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::NamedGraph {
            graph: "http://example.org/g1".into(),
            child: Box::new(scan),
        };
        if let PlanNode::NamedGraph { graph, .. } = &node {
            assert!(graph.contains("g1"));
        } else {
            panic!("expected NamedGraph");
        }
    }

    // ── Node count / depth ────────────────────────────────────────────────────

    #[test]
    fn test_node_count_leaf() {
        let node = make_scan("?s ?p ?o", IndexType::Spo, 0);
        assert_eq!(node.node_count(), 1);
    }

    #[test]
    fn test_node_count_nested() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec![]);
        assert_eq!(join.node_count(), 3);
    }

    #[test]
    fn test_node_count_deep() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let filter = PlanNode::filter("?o > 0", scan);
        let sort = PlanNode::sort(vec!["?s".into()], filter);
        let limit = PlanNode::limit(10, 0, sort);
        assert_eq!(limit.node_count(), 4);
    }

    #[test]
    fn test_depth_leaf() {
        let node = make_scan("?s ?p ?o", IndexType::Spo, 0);
        assert_eq!(node.depth(), 0);
    }

    #[test]
    fn test_depth_chain() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 10);
        let filter = PlanNode::filter("true", scan);
        let sort = PlanNode::sort(vec![], filter);
        assert_eq!(sort.depth(), 2);
    }

    #[test]
    fn test_depth_join() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec![]);
        assert_eq!(join.depth(), 1);
    }

    // ── Index type display ────────────────────────────────────────────────────

    #[test]
    fn test_index_type_display_spo() {
        assert_eq!(IndexType::Spo.to_string(), "SPO");
    }

    #[test]
    fn test_index_type_display_pos() {
        assert_eq!(IndexType::Pos.to_string(), "POS");
    }

    #[test]
    fn test_index_type_display_osp() {
        assert_eq!(IndexType::Osp.to_string(), "OSP");
    }

    #[test]
    fn test_index_type_display_fullscan() {
        assert_eq!(IndexType::FullScan.to_string(), "FULL_SCAN");
    }

    // ── Text format ───────────────────────────────────────────────────────────

    #[test]
    fn test_explain_text_contains_header() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Query Plan"));
        assert!(out.contains("=========="));
    }

    #[test]
    fn test_explain_text_contains_cost() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("5.0000"));
    }

    #[test]
    fn test_explain_text_contains_cardinality() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("100"));
    }

    #[test]
    fn test_explain_text_triple_scan() {
        let plan = make_plan(make_scan("?s rdf:type owl:Class", IndexType::Pos, 42));
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("TripleScan"));
        assert!(out.contains("owl:Class"));
        assert!(out.contains("POS"));
        assert!(out.contains("42"));
    }

    #[test]
    fn test_explain_text_hash_join() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec!["?s".into()]);
        let plan = make_plan(join);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("HashJoin"));
        assert!(out.contains("?s"));
    }

    #[test]
    fn test_explain_text_filter() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let filtered = PlanNode::filter("?o > 10", scan);
        let plan = make_plan(filtered);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Filter"));
        assert!(out.contains("?o > 10"));
    }

    #[test]
    fn test_explain_text_sort() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let sorted = PlanNode::sort(vec!["+?s".into()], scan);
        let plan = make_plan(sorted);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Sort"));
        assert!(out.contains("+?s"));
    }

    #[test]
    fn test_explain_text_limit() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let limited = PlanNode::limit(25, 0, scan);
        let plan = make_plan(limited);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Limit"));
        assert!(out.contains("25"));
    }

    #[test]
    fn test_explain_text_distinct() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::distinct(scan);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Distinct"));
    }

    #[test]
    fn test_explain_text_union() {
        let left = make_scan("?s a owl:Class", IndexType::Pos, 30);
        let right = make_scan("?s a rdfs:Class", IndexType::Pos, 10);
        let node = PlanNode::union(left, right);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Union"));
    }

    #[test]
    fn test_explain_text_optional() {
        let main = make_scan("?s foaf:name ?n", IndexType::Spo, 200);
        let opt = make_scan("?s foaf:mbox ?m", IndexType::Spo, 80);
        let node = PlanNode::optional(main, opt);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Optional"));
    }

    #[test]
    fn test_explain_text_aggregate() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 1000);
        let node = PlanNode::Aggregate {
            group_by: vec!["?s".into()],
            aggs: vec!["COUNT(?o) AS ?cnt".into()],
            child: Box::new(scan),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Aggregate"));
        assert!(out.contains("COUNT"));
    }

    #[test]
    fn test_explain_text_subquery() {
        let inner = make_plan(make_scan("?x ?y ?z", IndexType::FullScan, 5));
        let node = PlanNode::Subquery {
            plan: Box::new(inner),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Subquery"));
    }

    #[test]
    fn test_explain_text_property_path() {
        let node = PlanNode::PropertyPath {
            subject: "?s".into(),
            path: "foaf:knows+".into(),
            object: "?o".into(),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("PropertyPath"));
        assert!(out.contains("foaf:knows+"));
    }

    #[test]
    fn test_explain_text_service() {
        let sub = make_plan(make_scan("?s ?p ?o", IndexType::FullScan, 50));
        let node = PlanNode::Service {
            endpoint: "http://remote.example.org/sparql".into(),
            subplan: Box::new(sub),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Service"));
        assert!(out.contains("remote.example.org"));
    }

    #[test]
    fn test_explain_text_merge_join() {
        let left = make_scan("?s ?p ?o", IndexType::Spo, 500);
        let right = make_scan("?s a ?t", IndexType::Pos, 100);
        let node = PlanNode::MergeJoin {
            left: Box::new(left),
            right: Box::new(right),
            join_vars: vec!["?s".into()],
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("MergeJoin"));
    }

    #[test]
    fn test_explain_text_values_scan() {
        let node = PlanNode::ValuesScan {
            vars: vec!["?s".into()],
            row_count: 7,
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("ValuesScan"));
        assert!(out.contains("7"));
    }

    #[test]
    fn test_explain_text_named_graph() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 50);
        let node = PlanNode::NamedGraph {
            graph: "http://example.org/g1".into(),
            child: Box::new(scan),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("NamedGraph"));
        assert!(out.contains("g1"));
    }

    // ── JSON format ───────────────────────────────────────────────────────────

    #[test]
    fn test_explain_json_is_valid() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_json(&plan);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("invalid JSON");
        assert!(parsed.is_object());
    }

    #[test]
    fn test_explain_json_contains_type_field() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("\"type\""));
    }

    #[test]
    fn test_explain_json_triple_scan_fields() {
        let plan = make_plan(make_scan("?s rdf:type owl:Class", IndexType::Pos, 42));
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("triple_scan"));
        assert!(out.contains("owl:Class"));
        assert!(out.contains("POS"));
        assert!(out.contains("42"));
    }

    #[test]
    fn test_explain_json_hash_join() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec!["?s".into()]);
        let plan = make_plan(join);
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("hash_join"));
        assert!(out.contains("join_vars"));
    }

    #[test]
    fn test_explain_json_filter() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::filter("?o > 10", scan);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("filter"));
        assert!(out.contains("expr"));
    }

    #[test]
    fn test_explain_json_estimated_cost() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("estimated_cost"));
        assert!(out.contains("5.0"));
    }

    #[test]
    fn test_explain_json_aggregate() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 1000);
        let node = PlanNode::Aggregate {
            group_by: vec!["?s".into()],
            aggs: vec!["SUM(?v) AS ?total".into()],
            child: Box::new(scan),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("aggregate"));
        assert!(out.contains("group_by"));
        assert!(out.contains("aggs"));
    }

    #[test]
    fn test_explain_json_union() {
        let left = make_scan("?s a owl:Class", IndexType::Pos, 30);
        let right = make_scan("?s a rdfs:Class", IndexType::Pos, 10);
        let node = PlanNode::union(left, right);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_json(&plan);
        assert!(out.contains("union"));
    }

    // ── DOT format ────────────────────────────────────────────────────────────

    #[test]
    fn test_explain_dot_starts_with_digraph() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.starts_with("digraph QueryPlan {"));
    }

    #[test]
    fn test_explain_dot_ends_with_brace() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.trim().ends_with('}'));
    }

    #[test]
    fn test_explain_dot_contains_triple_scan() {
        let plan = make_plan(make_scan("?s a owl:Class", IndexType::Pos, 42));
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("TripleScan"));
    }

    #[test]
    fn test_explain_dot_contains_edge_arrows() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec!["?s".into()]);
        let plan = make_plan(join);
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("->"));
    }

    #[test]
    fn test_explain_dot_hash_join_labels() {
        let left = make_scan("?s a ?t", IndexType::Pos, 10);
        let right = make_scan("?s ?p ?o", IndexType::Spo, 5);
        let join = PlanNode::hash_join(left, right, vec!["?s".into()]);
        let plan = make_plan(join);
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("HashJoin"));
        assert!(out.contains("left"));
        assert!(out.contains("right"));
    }

    #[test]
    fn test_explain_dot_filter() {
        let scan = make_scan("?s ?p ?o", IndexType::FullScan, 100);
        let node = PlanNode::filter("?o > 10", scan);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("Filter"));
    }

    #[test]
    fn test_explain_dot_union() {
        let left = make_scan("?s a owl:Class", IndexType::Pos, 30);
        let right = make_scan("?s a rdfs:Class", IndexType::Pos, 10);
        let node = PlanNode::union(left, right);
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("Union"));
    }

    #[test]
    fn test_explain_dot_service() {
        let sub = make_plan(make_scan("?s ?p ?o", IndexType::FullScan, 50));
        let node = PlanNode::Service {
            endpoint: "http://remote.example.org/sparql".into(),
            subplan: Box::new(sub),
        };
        let plan = make_plan(node);
        let out = QueryExplainer::new().explain_dot(&plan);
        assert!(out.contains("Service"));
    }

    // ── explain_with_format ───────────────────────────────────────────────────

    #[test]
    fn test_explain_with_format_text() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let exp = QueryExplainer::new();
        let out = exp.explain_with_format(&plan, ExplainFormat::Text);
        assert!(out.contains("Query Plan"));
    }

    #[test]
    fn test_explain_with_format_json() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let exp = QueryExplainer::new();
        let out = exp.explain_with_format(&plan, ExplainFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(v.is_object());
    }

    #[test]
    fn test_explain_with_format_dot() {
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let exp = QueryExplainer::new();
        let out = exp.explain_with_format(&plan, ExplainFormat::Dot);
        assert!(out.contains("digraph"));
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    #[test]
    fn test_builder_default() {
        let exp = QueryExplainer::builder().build();
        assert!(exp.show_estimates);
        assert!(exp.show_costs);
        assert_eq!(exp.format, ExplainFormat::Text);
    }

    #[test]
    fn test_builder_no_estimates() {
        let exp = QueryExplainer::builder().show_estimates(false).build();
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 999));
        let out = exp.explain_text(&plan);
        // Row estimate for the scan should be absent
        assert!(!out.contains("999"));
    }

    #[test]
    fn test_builder_no_costs() {
        let exp = QueryExplainer::builder().show_costs(false).build();
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = exp.explain_text(&plan);
        assert!(!out.contains("5.0000"));
    }

    #[test]
    fn test_builder_json_format() {
        let exp = QueryExplainer::builder()
            .format(ExplainFormat::Json)
            .build();
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = exp.explain(&plan); // uses configured format
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(v.is_object());
    }

    #[test]
    fn test_builder_dot_format() {
        let exp = QueryExplainer::builder().format(ExplainFormat::Dot).build();
        let plan = make_plan(make_scan("?s ?p ?o", IndexType::Spo, 10));
        let out = exp.explain(&plan);
        assert!(out.contains("digraph"));
    }

    // ── Nested / complex plans ────────────────────────────────────────────────

    #[test]
    fn test_nested_plan_text() {
        // DISTINCT(SORT(FILTER(HASH_JOIN(scan1, scan2))))
        let s1 = make_scan("?s a ?t", IndexType::Pos, 500);
        let s2 = make_scan("?s foaf:name ?n", IndexType::Spo, 200);
        let join = PlanNode::hash_join(s1, s2, vec!["?s".into()]);
        let filter = PlanNode::filter("LANG(?n) = 'en'", join);
        let sort = PlanNode::sort(vec!["+?n".into()], filter);
        let distinct = PlanNode::distinct(sort);
        let plan = QueryPlan {
            root: distinct,
            estimated_cost: 42.7,
            estimated_cardinality: 150,
        };
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Distinct"));
        assert!(out.contains("Sort"));
        assert!(out.contains("Filter"));
        assert!(out.contains("HashJoin"));
        assert!(out.contains("TripleScan"));
    }

    #[test]
    fn test_nested_plan_json_roundtrip() {
        let scan = make_scan("?s ?p ?o", IndexType::Spo, 100);
        let filter = PlanNode::filter("?o > 0", scan);
        let plan = QueryPlan {
            root: filter,
            estimated_cost: std::f64::consts::PI,
            estimated_cardinality: 80,
        };
        let exp = QueryExplainer::new();
        let json = exp.explain_json(&plan);
        let decoded: QueryPlan = serde_json::from_str(&json).expect("roundtrip failed");
        assert_eq!(decoded.estimated_cardinality, 80);
        assert!((decoded.estimated_cost - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn test_deeply_nested_node_count() {
        // 5-level deep chain
        let s = make_scan("?s ?p ?o", IndexType::FullScan, 10);
        let f = PlanNode::filter("true", s);
        let so = PlanNode::sort(vec![], f);
        let li = PlanNode::limit(5, 0, so);
        let di = PlanNode::distinct(li);
        assert_eq!(di.node_count(), 5);
        assert_eq!(di.depth(), 4);
    }

    #[test]
    fn test_subquery_in_text() {
        let inner_scan = make_scan("?x ?y ?z", IndexType::FullScan, 5);
        let inner_plan = QueryPlan {
            root: inner_scan,
            estimated_cost: 1.0,
            estimated_cardinality: 5,
        };
        let outer_scan = make_scan("?a ?b ?c", IndexType::Spo, 100);
        let sub = PlanNode::Subquery {
            plan: Box::new(inner_plan),
        };
        let join = PlanNode::hash_join(outer_scan, sub, vec!["?x".into()]);
        let plan = make_plan(join);
        let out = QueryExplainer::new().explain_text(&plan);
        assert!(out.contains("Subquery"));
        assert!(out.contains("HashJoin"));
    }
}
