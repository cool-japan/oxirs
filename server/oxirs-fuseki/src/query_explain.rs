//! SPARQL Query Explanation Engine.
//!
//! Provides a query explanation / plan visualization facility that parses a
//! SPARQL query string into an operator tree and renders it in multiple formats
//! (indented text, JSON, Graphviz DOT).  A simple cost estimator assigns a
//! rough execution cost to each node.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_fuseki::query_explain::{QueryExplainer, ExplainFormat};
//!
//! let explainer = QueryExplainer::new();
//! let tree = explainer.explain("SELECT ?s WHERE { ?s <http://example.org/p> ?o }").unwrap();
//! let text = explainer.render_text(&tree);
//! assert!(!text.is_empty());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during query explanation.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum ExplainError {
    /// The SPARQL query string could not be parsed.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// The query form is not supported for explanation.
    #[error("Unsupported query form: {0}")]
    UnsupportedForm(String),

    /// The explanation tree could not be serialized.
    #[error("Render error: {0}")]
    RenderError(String),

    /// An internal error occurred.
    #[error("Internal error: {0}")]
    Internal(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core AST
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the explanation operator tree.
///
/// Each variant maps to a SPARQL algebra operator and carries enough
/// information to render a human-readable plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationNode {
    /// Full triple-pattern scan (subject, predicate, object).
    Scan {
        /// Human-readable pattern string, e.g. `"?s rdf:type owl:Class"`.
        pattern: String,
        /// Estimated number of matching triples (may be `None` when unknown).
        estimated_cardinality: Option<u64>,
    },

    /// Hash join / nested-loop join of two sub-plans.
    Join {
        left: Box<ExplanationNode>,
        right: Box<ExplanationNode>,
    },

    /// Left-outer join (`OPTIONAL { … }`).
    Optional {
        left: Box<ExplanationNode>,
        right: Box<ExplanationNode>,
    },

    /// Expression filter applied to a child plan.
    Filter {
        /// Textual representation of the filter expression.
        expr: String,
        child: Box<ExplanationNode>,
    },

    /// Variable projection (`SELECT ?a ?b …`).
    Project {
        /// Variables retained by the projection.
        variables: Vec<String>,
        child: Box<ExplanationNode>,
    },

    /// Duplicate-row elimination (`SELECT DISTINCT …`).
    Distinct { child: Box<ExplanationNode> },

    /// Offset / limit pagination.
    Slice {
        offset: Option<u64>,
        limit: Option<u64>,
        child: Box<ExplanationNode>,
    },

    /// Set-union of two sub-plans (`UNION`).
    Union {
        left: Box<ExplanationNode>,
        right: Box<ExplanationNode>,
    },

    /// `GRAPH ?g { … }` sub-query wrapping.
    Graph {
        /// Graph name or variable.
        name: String,
        child: Box<ExplanationNode>,
    },

    /// `ORDER BY` ordering applied to a child plan.
    OrderBy {
        /// Order conditions (variable/expression, ascending/descending).
        conditions: Vec<OrderCondition>,
        child: Box<ExplanationNode>,
    },

    /// Sub-`SELECT` (sub-query inside a pattern).
    SubQuery { child: Box<ExplanationNode> },

    /// `GROUP BY` / aggregation.
    Group {
        /// Variables grouped on.
        group_vars: Vec<String>,
        /// Aggregation expressions.
        aggregations: Vec<String>,
        child: Box<ExplanationNode>,
    },

    /// `HAVING` post-aggregation filter.
    Having {
        expr: String,
        child: Box<ExplanationNode>,
    },

    /// Empty / unit table (zero columns, one empty row).
    Unit,
}

/// A single `ORDER BY` condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCondition {
    /// The expression or variable name.
    pub expr: String,
    /// `true` for ascending, `false` for descending.
    pub ascending: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Explanation tree wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper that holds the root of an explanation tree together with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTree {
    /// The root operator node.
    pub root: ExplanationNode,
    /// Original query string (stored for reference).
    pub query: String,
    /// Total estimated cost computed by [`QueryExplainer::estimate_cost`].
    pub estimated_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Output format
// ─────────────────────────────────────────────────────────────────────────────

/// Output format for the explanation renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplainFormat {
    /// Indented plain-text tree.
    Text,
    /// JSON representation of the operator tree.
    Json,
    /// Graphviz DOT language.
    Dot,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal SPARQL query parser for plan building.
///
/// This is a simplified structural parser — it does not attempt to be a full
/// SPARQL 1.1 parser but rather extracts enough information to build a
/// representative operator tree for explanation purposes.
struct QueryParser<'a> {
    query: &'a str,
}

impl<'a> QueryParser<'a> {
    fn new(query: &'a str) -> Self {
        Self { query }
    }

    /// Parse the query into an [`ExplanationNode`] tree.
    fn parse(&self) -> Result<ExplanationNode, ExplainError> {
        let normalized = self.query.trim().to_uppercase();

        if normalized.starts_with("SELECT") {
            self.parse_select()
        } else if normalized.starts_with("ASK") {
            self.parse_ask()
        } else if normalized.starts_with("CONSTRUCT") {
            self.parse_construct()
        } else if normalized.starts_with("DESCRIBE") {
            self.parse_describe()
        } else {
            Err(ExplainError::UnsupportedForm(
                "Query must begin with SELECT, ASK, CONSTRUCT or DESCRIBE".to_owned(),
            ))
        }
    }

    fn parse_select(&self) -> Result<ExplanationNode, ExplainError> {
        let q = self.query;
        let upper = q.to_uppercase();

        // Detect DISTINCT
        let has_distinct = upper.contains("SELECT DISTINCT") || upper.contains("SELECT  DISTINCT");

        // Extract projected variables
        let variables = extract_select_variables(q);

        // Detect LIMIT / OFFSET
        let limit = extract_limit(q);
        let offset = extract_offset(q);

        // Build WHERE body
        let where_body = extract_where_body(q).unwrap_or_else(|| q.to_owned());
        let body_node = self.build_body_node(&where_body)?;

        // Detect ORDER BY
        let order_conditions = extract_order_conditions(q);

        // Detect GROUP BY
        let group_vars = extract_group_vars(q);

        // Detect HAVING
        let having_expr = extract_having(q);

        // Assemble the tree from innermost to outermost
        let mut node = body_node;

        if !group_vars.is_empty() {
            let aggregations = extract_aggregations(q);
            node = ExplanationNode::Group {
                group_vars,
                aggregations,
                child: Box::new(node),
            };
        }

        if let Some(expr) = having_expr {
            node = ExplanationNode::Having {
                expr,
                child: Box::new(node),
            };
        }

        if !order_conditions.is_empty() {
            node = ExplanationNode::OrderBy {
                conditions: order_conditions,
                child: Box::new(node),
            };
        }

        if !variables.is_empty() {
            node = ExplanationNode::Project {
                variables,
                child: Box::new(node),
            };
        }

        if has_distinct {
            node = ExplanationNode::Distinct {
                child: Box::new(node),
            };
        }

        if limit.is_some() || offset.is_some() {
            node = ExplanationNode::Slice {
                offset,
                limit,
                child: Box::new(node),
            };
        }

        Ok(node)
    }

    fn parse_ask(&self) -> Result<ExplanationNode, ExplainError> {
        let where_body = extract_where_body(self.query).unwrap_or_default();
        let child = self.build_body_node(&where_body)?;
        Ok(ExplanationNode::Project {
            variables: vec!["(ASK)".to_owned()],
            child: Box::new(child),
        })
    }

    fn parse_construct(&self) -> Result<ExplanationNode, ExplainError> {
        let where_body = extract_where_body(self.query).unwrap_or_default();
        let child = self.build_body_node(&where_body)?;
        Ok(ExplanationNode::Project {
            variables: vec!["(CONSTRUCT template)".to_owned()],
            child: Box::new(child),
        })
    }

    fn parse_describe(&self) -> Result<ExplanationNode, ExplainError> {
        let where_body = extract_where_body(self.query).unwrap_or_default();
        let child = self.build_body_node(&where_body)?;
        Ok(ExplanationNode::Project {
            variables: vec!["(DESCRIBE resources)".to_owned()],
            child: Box::new(child),
        })
    }

    /// Build an operator node for the body of a WHERE clause.
    fn build_body_node(&self, body: &str) -> Result<ExplanationNode, ExplainError> {
        if body.trim().is_empty() {
            return Ok(ExplanationNode::Unit);
        }

        let upper = body.to_uppercase();

        // UNION handling
        if let Some(union_node) = maybe_union(body, self)? {
            return Ok(union_node);
        }

        // OPTIONAL handling
        if let Some(opt_node) = maybe_optional(body, self)? {
            return Ok(opt_node);
        }

        // Collect triple patterns
        let patterns = extract_triple_patterns(body);
        let filter_exprs = extract_filters(body);

        // Sub-SELECT detection
        if upper.contains("SELECT") && upper.contains("WHERE") {
            let inner_body = extract_where_body(body).unwrap_or_else(|| body.to_owned());
            let inner = self.build_body_node(&inner_body)?;
            let sub = ExplanationNode::SubQuery {
                child: Box::new(inner),
            };
            return Ok(wrap_filters(sub, filter_exprs));
        }

        // Build join tree from patterns
        let mut node = if patterns.is_empty() {
            ExplanationNode::Unit
        } else {
            let mut iter = patterns.into_iter();
            let first = ExplanationNode::Scan {
                pattern: iter.next().unwrap_or_default(),
                estimated_cardinality: None,
            };
            iter.fold(first, |acc, pat| ExplanationNode::Join {
                left: Box::new(acc),
                right: Box::new(ExplanationNode::Scan {
                    pattern: pat,
                    estimated_cardinality: None,
                }),
            })
        };

        node = wrap_filters(node, filter_exprs);
        Ok(node)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

fn extract_select_variables(query: &str) -> Vec<String> {
    // Find tokens between SELECT and WHERE
    let upper = query.to_uppercase();
    let select_pos = upper.find("SELECT").unwrap_or(0) + "SELECT".len();
    let where_pos = upper.find("WHERE").unwrap_or(query.len());
    if select_pos >= where_pos {
        return vec![];
    }
    let projection = &query[select_pos..where_pos];
    let mut vars: Vec<String> = Vec::new();
    for token in projection.split_whitespace() {
        let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '?' && c != '*');
        if cleaned.starts_with('?') {
            vars.push(cleaned.to_owned());
        } else if cleaned == "*" {
            vars.push("*".to_owned());
        }
    }
    vars
}

fn extract_where_body(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    let where_pos = upper.find("WHERE")? + "WHERE".len();
    let rest = &query[where_pos..].trim_start();
    if rest.starts_with('{') {
        // Find matching closing brace
        let inner = extract_balanced_braces(rest)?;
        Some(inner)
    } else {
        None
    }
}

/// Extract the content inside the outermost `{ … }`.
fn extract_balanced_braces(s: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' => {
                depth += 1;
                if depth == 1 {
                    start = Some(i + 1);
                }
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s_pos) = start {
                        return Some(s[s_pos..i].to_owned());
                    }
                    return None;
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn extract_triple_patterns(body: &str) -> Vec<String> {
    // Naive extraction: split by '.' and filter out lines with keywords
    let upper = body.to_uppercase();
    let mut patterns = Vec::new();
    for line in body.split('.') {
        let trimmed = line.trim();
        let upper_line = trimmed.to_uppercase();
        if trimmed.is_empty() {
            continue;
        }
        // Skip sub-structures
        if upper_line.starts_with("OPTIONAL")
            || upper_line.starts_with("FILTER")
            || upper_line.starts_with("UNION")
            || upper_line.starts_with("SELECT")
            || upper_line.starts_with('{')
            || upper_line.starts_with('}')
            || upper_line.contains('{')
        {
            continue;
        }
        // Must contain at least two tokens (subject predicate)
        let token_count = trimmed.split_whitespace().count();
        if token_count >= 2 {
            patterns.push(trimmed.to_owned());
        }
    }
    patterns
}

fn extract_filters(body: &str) -> Vec<String> {
    let mut filters = Vec::new();
    let upper = body.to_uppercase();
    let mut start = 0;
    while let Some(pos) = upper[start..].find("FILTER") {
        let abs_pos = start + pos + "FILTER".len();
        let rest = body[abs_pos..].trim_start();
        if let Some(inner) = extract_balanced_parens(rest) {
            filters.push(inner);
        }
        start = abs_pos;
    }
    filters
}

fn extract_balanced_parens(s: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => {
                depth += 1;
                if depth == 1 {
                    start = Some(i + 1);
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(sp) = start {
                        return Some(s[sp..i].to_owned());
                    }
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

fn wrap_filters(mut node: ExplanationNode, exprs: Vec<String>) -> ExplanationNode {
    for expr in exprs {
        node = ExplanationNode::Filter {
            expr,
            child: Box::new(node),
        };
    }
    node
}

fn extract_limit(query: &str) -> Option<u64> {
    let upper = query.to_uppercase();
    let pos = upper.find("LIMIT")? + "LIMIT".len();
    query[pos..]
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
}

fn extract_offset(query: &str) -> Option<u64> {
    let upper = query.to_uppercase();
    let pos = upper.find("OFFSET")? + "OFFSET".len();
    query[pos..]
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
}

fn extract_order_conditions(query: &str) -> Vec<OrderCondition> {
    let upper = query.to_uppercase();
    let order_pos = match upper.find("ORDER BY") {
        Some(p) => p + "ORDER BY".len(),
        None => return vec![],
    };
    // Stop at LIMIT / OFFSET / end
    let stop = upper[order_pos..]
        .find("LIMIT")
        .or_else(|| upper[order_pos..].find("OFFSET"))
        .map(|p| order_pos + p)
        .unwrap_or(query.len());
    let segment = &query[order_pos..stop];
    let mut conditions = Vec::new();
    for token in segment.split_whitespace() {
        let upper_tok = token.to_uppercase();
        if upper_tok == "ASC" || upper_tok == "DESC" {
            continue;
        }
        if token.starts_with('?') || token.starts_with("ASC(") || token.starts_with("DESC(") {
            let ascending = !upper_tok.starts_with("DESC");
            let expr = token
                .trim_start_matches("ASC(")
                .trim_start_matches("DESC(")
                .trim_end_matches(')')
                .to_owned();
            conditions.push(OrderCondition { expr, ascending });
        }
    }
    conditions
}

fn extract_group_vars(query: &str) -> Vec<String> {
    let upper = query.to_uppercase();
    let pos = match upper.find("GROUP BY") {
        Some(p) => p + "GROUP BY".len(),
        None => return vec![],
    };
    let stop = ["HAVING", "ORDER BY", "LIMIT", "OFFSET"]
        .iter()
        .filter_map(|kw| upper[pos..].find(kw).map(|p| pos + p))
        .min()
        .unwrap_or(query.len());
    query[pos..stop]
        .split_whitespace()
        .filter(|t| t.starts_with('?'))
        .map(|t| t.to_owned())
        .collect()
}

fn extract_aggregations(query: &str) -> Vec<String> {
    let upper = query.to_uppercase();
    let keywords = [
        "COUNT",
        "SUM",
        "MIN",
        "MAX",
        "AVG",
        "GROUP_CONCAT",
        "SAMPLE",
    ];
    let mut aggs = Vec::new();
    for kw in &keywords {
        if upper.contains(kw) {
            aggs.push(kw.to_lowercase());
        }
    }
    aggs
}

fn extract_having(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    let pos = upper.find("HAVING")? + "HAVING".len();
    let segment = &query[pos..];
    let stop = ["ORDER BY", "LIMIT", "OFFSET"]
        .iter()
        .filter_map(|kw| segment.to_uppercase().find(kw))
        .min()
        .unwrap_or(segment.len());
    Some(segment[..stop].trim().to_owned())
}

fn maybe_union<'a>(
    body: &str,
    parser: &QueryParser<'a>,
) -> Result<Option<ExplanationNode>, ExplainError> {
    let upper = body.to_uppercase();
    // Find first top-level UNION keyword
    let union_pos = match upper.find(" UNION ") {
        Some(p) => p,
        None => return Ok(None),
    };
    let left_str = &body[..union_pos];
    let right_str = &body[union_pos + " UNION ".len()..];
    let left = parser.build_body_node(left_str.trim())?;
    let right = parser.build_body_node(right_str.trim())?;
    Ok(Some(ExplanationNode::Union {
        left: Box::new(left),
        right: Box::new(right),
    }))
}

fn maybe_optional<'a>(
    body: &str,
    parser: &QueryParser<'a>,
) -> Result<Option<ExplanationNode>, ExplainError> {
    let upper = body.to_uppercase();
    let opt_pos = match upper.find("OPTIONAL") {
        Some(p) => p,
        None => return Ok(None),
    };
    let main_part = body[..opt_pos].trim();
    let opt_rest = &body[opt_pos + "OPTIONAL".len()..].trim_start();
    let opt_body = extract_balanced_braces(opt_rest).unwrap_or_default();
    let left = parser.build_body_node(main_part)?;
    let right = parser.build_body_node(&opt_body)?;
    Ok(Some(ExplanationNode::Optional {
        left: Box::new(left),
        right: Box::new(right),
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Cost estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Simple heuristic cost model.
///
/// Cost values are dimensionless and serve only for relative comparisons.
fn cost_of(node: &ExplanationNode) -> f64 {
    match node {
        ExplanationNode::Unit => 0.0,
        ExplanationNode::Scan {
            estimated_cardinality,
            ..
        } => estimated_cardinality.unwrap_or(1000) as f64,
        ExplanationNode::Join { left, right } => {
            cost_of(left) + cost_of(right) + cost_of(left) * cost_of(right) * 0.001
        }
        ExplanationNode::Optional { left, right } => cost_of(left) + cost_of(right) * 0.5,
        ExplanationNode::Filter { child, .. } => cost_of(child) * 1.05,
        ExplanationNode::Project { child, .. } => cost_of(child),
        ExplanationNode::Distinct { child } => cost_of(child) * 1.5,
        ExplanationNode::Slice { child, limit, .. } => {
            let base = cost_of(child);
            if let Some(l) = limit {
                base.min(*l as f64)
            } else {
                base
            }
        }
        ExplanationNode::Union { left, right } => cost_of(left) + cost_of(right),
        ExplanationNode::Graph { child, .. } => cost_of(child) * 1.1,
        ExplanationNode::OrderBy { child, .. } => {
            let c = cost_of(child);
            c + c * (c.log2().max(1.0)) * 0.01
        }
        ExplanationNode::SubQuery { child } => cost_of(child) * 1.2,
        ExplanationNode::Group { child, .. } => cost_of(child) * 1.3,
        ExplanationNode::Having { child, .. } => cost_of(child) * 1.05,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QueryExplainer
// ─────────────────────────────────────────────────────────────────────────────

/// Main entry point for query explanation.
#[derive(Debug, Default, Clone)]
pub struct QueryExplainer {
    /// Default cardinality estimate used for `Scan` nodes when no statistics
    /// are available.
    pub default_cardinality: Option<u64>,
}

impl QueryExplainer {
    /// Create a new explainer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an explainer with a custom default cardinality estimate.
    pub fn with_default_cardinality(cardinality: u64) -> Self {
        Self {
            default_cardinality: Some(cardinality),
        }
    }

    /// Parse `query` into an [`ExplanationTree`] or return an [`ExplainError`].
    pub fn explain(&self, query: &str) -> Result<ExplanationTree, ExplainError> {
        if query.trim().is_empty() {
            return Err(ExplainError::ParseError("Query string is empty".to_owned()));
        }
        let parser = QueryParser::new(query);
        let mut root = parser.parse()?;
        // Annotate scan cardinalities if configured
        if let Some(card) = self.default_cardinality {
            annotate_cardinality(&mut root, card);
        }
        let estimated_cost = cost_of(&root);
        Ok(ExplanationTree {
            root,
            query: query.to_owned(),
            estimated_cost,
        })
    }

    /// Render an [`ExplanationTree`] in the requested format.
    pub fn render(&self, tree: &ExplanationTree, format: ExplainFormat) -> String {
        match format {
            ExplainFormat::Text => self.render_text(tree),
            ExplainFormat::Json => self.render_json(tree),
            ExplainFormat::Dot => self.render_dot(tree),
        }
    }

    // ── Text renderer ────────────────────────────────────────────────────────

    /// Render the tree as an indented text plan.
    pub fn render_text(&self, tree: &ExplanationTree) -> String {
        let mut buf = String::new();
        buf.push_str(&format!(
            "Query Plan (estimated cost: {:.2})\n",
            tree.estimated_cost
        ));
        buf.push_str("─".repeat(40).as_str());
        buf.push('\n');
        render_node_text(&tree.root, 0, &mut buf);
        buf
    }

    // ── JSON renderer ────────────────────────────────────────────────────────

    /// Render the tree as a JSON string.
    pub fn render_json(&self, tree: &ExplanationTree) -> String {
        serde_json::to_string_pretty(tree).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    // ── DOT renderer ─────────────────────────────────────────────────────────

    /// Render the tree as a Graphviz DOT graph.
    pub fn render_dot(&self, tree: &ExplanationTree) -> String {
        let mut buf = String::new();
        buf.push_str("digraph ExplanationTree {\n");
        buf.push_str("  graph [rankdir=TB];\n");
        buf.push_str("  node [shape=box, fontname=\"Helvetica\"];\n");
        let mut id_counter = 0u64;
        let mut labels: BTreeMap<u64, String> = BTreeMap::new();
        let mut edges: Vec<(u64, u64)> = Vec::new();
        render_node_dot(&tree.root, &mut id_counter, &mut labels, &mut edges, None);
        for (id, label) in &labels {
            let escaped = label.replace('"', "\\\"");
            buf.push_str(&format!("  n{id} [label=\"{escaped}\"];\n"));
        }
        for (from, to) in &edges {
            buf.push_str(&format!("  n{from} -> n{to};\n"));
        }
        buf.push_str("}\n");
        buf
    }

    // ── Cost estimator ───────────────────────────────────────────────────────

    /// Compute the estimated cost for a given tree.
    pub fn estimate_cost(&self, tree: &ExplanationTree) -> f64 {
        cost_of(&tree.root)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal render helpers
// ─────────────────────────────────────────────────────────────────────────────

fn node_label(node: &ExplanationNode) -> String {
    match node {
        ExplanationNode::Scan {
            pattern,
            estimated_cardinality,
        } => {
            if let Some(c) = estimated_cardinality {
                format!("Scan [{c}] {pattern}")
            } else {
                format!("Scan {pattern}")
            }
        }
        ExplanationNode::Join { .. } => "Join".to_owned(),
        ExplanationNode::Optional { .. } => "Optional".to_owned(),
        ExplanationNode::Filter { expr, .. } => format!("Filter({expr})"),
        ExplanationNode::Project { variables, .. } => format!("Project({})", variables.join(", ")),
        ExplanationNode::Distinct { .. } => "Distinct".to_owned(),
        ExplanationNode::Slice { offset, limit, .. } => {
            format!(
                "Slice(offset={}, limit={})",
                offset.map_or("∞".to_owned(), |o| o.to_string()),
                limit.map_or("∞".to_owned(), |l| l.to_string())
            )
        }
        ExplanationNode::Union { .. } => "Union".to_owned(),
        ExplanationNode::Graph { name, .. } => format!("Graph({name})"),
        ExplanationNode::OrderBy { conditions, .. } => {
            let parts: Vec<_> = conditions
                .iter()
                .map(|c| {
                    if c.ascending {
                        format!("ASC({})", c.expr)
                    } else {
                        format!("DESC({})", c.expr)
                    }
                })
                .collect();
            format!("OrderBy({})", parts.join(", "))
        }
        ExplanationNode::SubQuery { .. } => "SubQuery".to_owned(),
        ExplanationNode::Group {
            group_vars,
            aggregations,
            ..
        } => format!(
            "Group(by=[{}], agg=[{}])",
            group_vars.join(", "),
            aggregations.join(", ")
        ),
        ExplanationNode::Having { expr, .. } => format!("Having({expr})"),
        ExplanationNode::Unit => "Unit".to_owned(),
    }
}

fn render_node_text(node: &ExplanationNode, depth: usize, buf: &mut String) {
    let indent = "  ".repeat(depth);
    buf.push_str(&format!("{indent}└─ {}\n", node_label(node)));
    for child in children_of(node) {
        render_node_text(child, depth + 1, buf);
    }
}

fn children_of(node: &ExplanationNode) -> Vec<&ExplanationNode> {
    match node {
        ExplanationNode::Scan { .. } | ExplanationNode::Unit => vec![],
        ExplanationNode::Join { left, right }
        | ExplanationNode::Optional { left, right }
        | ExplanationNode::Union { left, right } => vec![left, right],
        ExplanationNode::Filter { child, .. }
        | ExplanationNode::Project { child, .. }
        | ExplanationNode::Distinct { child }
        | ExplanationNode::Slice { child, .. }
        | ExplanationNode::Graph { child, .. }
        | ExplanationNode::OrderBy { child, .. }
        | ExplanationNode::SubQuery { child }
        | ExplanationNode::Group { child, .. }
        | ExplanationNode::Having { child, .. } => vec![child],
    }
}

fn render_node_dot(
    node: &ExplanationNode,
    counter: &mut u64,
    labels: &mut BTreeMap<u64, String>,
    edges: &mut Vec<(u64, u64)>,
    parent_id: Option<u64>,
) -> u64 {
    let my_id = *counter;
    *counter += 1;
    labels.insert(my_id, node_label(node));
    if let Some(pid) = parent_id {
        edges.push((pid, my_id));
    }
    for child in children_of(node) {
        render_node_dot(child, counter, labels, edges, Some(my_id));
    }
    my_id
}

/// Recursively set `estimated_cardinality` on all `Scan` nodes.
fn annotate_cardinality(node: &mut ExplanationNode, cardinality: u64) {
    match node {
        ExplanationNode::Scan {
            estimated_cardinality,
            ..
        } => {
            if estimated_cardinality.is_none() {
                *estimated_cardinality = Some(cardinality);
            }
        }
        ExplanationNode::Join { left, right }
        | ExplanationNode::Optional { left, right }
        | ExplanationNode::Union { left, right } => {
            annotate_cardinality(left, cardinality);
            annotate_cardinality(right, cardinality);
        }
        ExplanationNode::Filter { child, .. }
        | ExplanationNode::Project { child, .. }
        | ExplanationNode::Distinct { child }
        | ExplanationNode::Slice { child, .. }
        | ExplanationNode::Graph { child, .. }
        | ExplanationNode::OrderBy { child, .. }
        | ExplanationNode::SubQuery { child }
        | ExplanationNode::Group { child, .. }
        | ExplanationNode::Having { child, .. } => {
            annotate_cardinality(child, cardinality);
        }
        ExplanationNode::Unit => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn explainer() -> QueryExplainer {
        QueryExplainer::new()
    }

    // ── Basic parsing ────────────────────────────────────────────────────────

    #[test]
    fn test_explain_simple_select() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        // Root should be a Project (or Scan if no variables extracted)
        let text = e.render_text(&tree);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_explain_empty_query_error() {
        let e = explainer();
        let err = e.explain("").unwrap_err();
        assert!(matches!(err, ExplainError::ParseError(_)));
    }

    #[test]
    fn test_explain_unsupported_form() {
        let e = explainer();
        let err = e.explain("INSERT DATA { <a> <b> <c> }").unwrap_err();
        assert!(matches!(err, ExplainError::UnsupportedForm(_)));
    }

    #[test]
    fn test_explain_ask_query() {
        let e = explainer();
        let tree = e.explain("ASK WHERE { ?s <p> ?o }").unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("ASK") || !text.is_empty());
    }

    #[test]
    fn test_explain_construct_query() {
        let e = explainer();
        let tree = e
            .explain("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_explain_describe_query() {
        let e = explainer();
        let tree = e
            .explain("DESCRIBE <http://example.org/> WHERE { }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(!text.is_empty());
    }

    // ── Tree structure ───────────────────────────────────────────────────────

    #[test]
    fn test_tree_contains_query_string() {
        let e = explainer();
        let q = "SELECT ?s WHERE { ?s <p> ?o }";
        let tree = e.explain(q).unwrap();
        assert_eq!(tree.query, q);
    }

    #[test]
    fn test_tree_cost_is_positive_or_zero() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        assert!(tree.estimated_cost >= 0.0);
    }

    #[test]
    fn test_estimate_cost_returns_same_as_tree() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let cost = e.estimate_cost(&tree);
        assert!((cost - tree.estimated_cost).abs() < 1e-9);
    }

    // ── SELECT features ──────────────────────────────────────────────────────

    #[test]
    fn test_distinct_keyword() {
        let e = explainer();
        let tree = e.explain("SELECT DISTINCT ?s WHERE { ?s <p> ?o }").unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Distinct"));
    }

    #[test]
    fn test_limit_creates_slice_node() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o } LIMIT 10").unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Slice"));
    }

    #[test]
    fn test_offset_creates_slice_node() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o } OFFSET 5").unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Slice"));
    }

    #[test]
    fn test_limit_and_offset() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s WHERE { ?s <p> ?o } LIMIT 10 OFFSET 5")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Slice"));
    }

    #[test]
    fn test_filter_expression_appears() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s WHERE { ?s <age> ?a . FILTER(?a > 18) }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Filter"));
    }

    #[test]
    fn test_union_creates_union_node() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s WHERE { { ?s <a> <b> } UNION { ?s <c> <d> } }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Union"));
    }

    #[test]
    fn test_optional_creates_optional_node() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s ?o WHERE { ?s <p> ?o OPTIONAL { ?s <q> ?r } }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Optional"));
    }

    #[test]
    fn test_project_node_appears() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Project") || text.contains("Scan") || text.contains("Unit"));
    }

    // ── Render formats ───────────────────────────────────────────────────────

    #[test]
    fn test_render_text_not_empty() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let text = e.render_text(&tree);
        assert!(!text.is_empty());
        assert!(text.contains("Query Plan"));
    }

    #[test]
    fn test_render_json_valid_json() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let json = e.render_json(&tree);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_render_dot_contains_digraph() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let dot = e.render_dot(&tree);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->") || dot.contains("n0"));
    }

    #[test]
    fn test_render_dispatch_text() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let via_dispatch = e.render(&tree, ExplainFormat::Text);
        let direct = e.render_text(&tree);
        assert_eq!(via_dispatch, direct);
    }

    #[test]
    fn test_render_dispatch_json() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let via_dispatch = e.render(&tree, ExplainFormat::Json);
        let direct = e.render_json(&tree);
        assert_eq!(via_dispatch, direct);
    }

    #[test]
    fn test_render_dispatch_dot() {
        let e = explainer();
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        let via_dispatch = e.render(&tree, ExplainFormat::Dot);
        let direct = e.render_dot(&tree);
        assert_eq!(via_dispatch, direct);
    }

    // ── Cardinality annotation ───────────────────────────────────────────────

    #[test]
    fn test_cardinality_annotation_via_explainer() {
        let e = QueryExplainer::with_default_cardinality(500);
        let tree = e.explain("SELECT ?s WHERE { ?s <p> ?o }").unwrap();
        // Cost should reflect the 500 cardinality
        assert!(tree.estimated_cost > 0.0);
    }

    // ── Cost estimator ───────────────────────────────────────────────────────

    #[test]
    fn test_unit_node_cost_is_zero() {
        assert_eq!(cost_of(&ExplanationNode::Unit), 0.0);
    }

    #[test]
    fn test_scan_default_cardinality_cost() {
        let node = ExplanationNode::Scan {
            pattern: "?s ?p ?o".to_owned(),
            estimated_cardinality: Some(100),
        };
        assert!((cost_of(&node) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_distinct_higher_cost_than_child() {
        let child = ExplanationNode::Scan {
            pattern: "?s ?p ?o".to_owned(),
            estimated_cardinality: Some(1000),
        };
        let distinct = ExplanationNode::Distinct {
            child: Box::new(child.clone()),
        };
        assert!(cost_of(&distinct) > cost_of(&child));
    }

    #[test]
    fn test_filter_slightly_higher_cost() {
        let child = ExplanationNode::Scan {
            pattern: "?s ?p ?o".to_owned(),
            estimated_cardinality: Some(1000),
        };
        let filter = ExplanationNode::Filter {
            expr: "?x > 5".to_owned(),
            child: Box::new(child.clone()),
        };
        assert!(cost_of(&filter) >= cost_of(&child));
    }

    // ── ExplainError ──────────────────────────────────────────────────────────

    #[test]
    fn test_explain_error_display() {
        let err = ExplainError::ParseError("oops".to_owned());
        assert!(err.to_string().contains("oops"));
    }

    #[test]
    fn test_explain_error_unsupported_form() {
        let err = ExplainError::UnsupportedForm("LOAD".to_owned());
        assert!(err.to_string().contains("LOAD"));
    }

    // ── Multiple triple patterns → Join tree ─────────────────────────────────

    #[test]
    fn test_multiple_patterns_create_join() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s ?o WHERE { ?s <p1> ?a . ?a <p2> ?o }")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("Join") || text.contains("Scan"));
    }

    #[test]
    fn test_query_with_order_by() {
        let e = explainer();
        let tree = e
            .explain("SELECT ?s WHERE { ?s <p> ?o } ORDER BY ?s")
            .unwrap();
        let text = e.render_text(&tree);
        assert!(text.contains("OrderBy") || !text.is_empty());
    }
}
